import ast
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI


PRINCIPLE_LABELS = {
    "coverage": "Comprehensive coverage",
    "specificity": "Specific and actionable",
    "non_redundancy": "Non-redundant",
    "discriminative": "Discriminative",
}


GLOBAL_SYSTEM_PROMPT = (
    "You are a strict rubric designer. Score rubric quality and return only valid JSON. "
    "Do not wrap the answer in markdown fences."
)

GLOBAL_USER_PROMPT = """Evaluate the rubric for the given instruction.

Instruction:
{instruction}

Rubric:
{rubric}

Score the rubric on four dimensions using integers from 1 to {scale_max}:
- coverage: whether the rubric covers the important aspects of the instruction.
- specificity: whether the criteria are specific and actionable.
- non_redundancy: whether the criteria avoid repetition and overlap.
- discriminative: whether the rubric can separate strong and weak responses.

Return JSON only with this schema:
{{
  "coverage": {{"score": <int>, "reason": "<short reason>"}},
  "specificity": {{"score": <int>, "reason": "<short reason>"}},
  "non_redundancy": {{"score": <int>, "reason": "<short reason>"}},
  "discriminative": {{"score": <int>, "reason": "<short reason>"}}
}}

Important:
- Return exactly one JSON object and nothing else.
- Do not use markdown code fences.
- Use the exact keys: coverage, specificity, non_redundancy, discriminative.
"""


def split_rubric_items(rubric: str) -> List[str]:
    lines = [line.strip() for line in rubric.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    pieces = re.split(r"(?:\b\d+\.\s+|(?:^|\s)-\s+)", rubric)
    items = [piece.strip() for piece in pieces if piece and piece.strip()]
    return items or [rubric.strip()]


def clamp_score(score: float, scale_max: int) -> float:
    return max(1.0, min(float(scale_max), float(score)))


def _coerce_score_value(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else 1.0


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_quotes(text: str) -> str:
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _jsonish_cleanup(text: str) -> str:
    text = _normalize_quotes(_strip_code_fences(text))
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    return text.strip()


def _extract_balanced_json_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    stack = []
    start = None
    for index, char in enumerate(text):
        if char == "{":
            if not stack:
                start = index
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text[start : index + 1])
                    start = None
    return candidates


def _coerce_payload(payload: Dict) -> Optional[Dict]:
    if not isinstance(payload, dict):
        return None
    if all(key in payload for key in PRINCIPLE_LABELS):
        return payload

    nested_candidates = [
        payload.get("scores"),
        payload.get("dimensions"),
        payload.get("result"),
        payload.get("rubric_scores"),
    ]
    for nested in nested_candidates:
        if isinstance(nested, dict) and all(key in nested for key in PRINCIPLE_LABELS):
            return nested

    alias_map = {
        "coverage": "coverage",
        "comprehensive coverage": "coverage",
        "specificity": "specificity",
        "specific and actionable": "specificity",
        "non redundancy": "non_redundancy",
        "non-redundancy": "non_redundancy",
        "non_redundancy": "non_redundancy",
        "discriminative": "discriminative",
    }
    normalized = {}
    for raw_key, value in payload.items():
        canon = alias_map.get(str(raw_key).strip().lower())
        if canon:
            normalized[canon] = value
    return normalized if all(key in normalized for key in PRINCIPLE_LABELS) else None


def _try_parse_json_payload(candidate: str) -> Optional[Dict]:
    variants = [candidate, _jsonish_cleanup(candidate)]
    for variant in variants:
        try:
            payload = json.loads(variant)
            coerced = _coerce_payload(payload)
            if coerced is not None:
                return coerced
        except json.JSONDecodeError:
            pass
        try:
            payload = ast.literal_eval(variant)
            coerced = _coerce_payload(payload)
            if coerced is not None:
                return coerced
        except Exception:
            pass
    return None


def _extract_json_blob(text: str) -> Optional[Dict]:
    text = _jsonish_cleanup(text)
    payload = _try_parse_json_payload(text)
    if payload is not None:
        return payload

    for candidate in _extract_balanced_json_candidates(text):
        payload = _try_parse_json_payload(candidate)
        if payload is not None:
            return payload
    return None


def _extract_dimension_from_text(text: str, key: str, label: str) -> Optional[Dict[str, object]]:
    aliases = [
        key,
        key.replace("_", " "),
        key.replace("_", "-"),
        label.lower(),
        label.lower().replace("-", " "),
    ]
    escaped = "|".join(re.escape(alias) for alias in aliases)
    pattern = (
        rf"(?is)(?:^|\n|\r)\s*(?:[-*]?\s*)?(?:{escaped})\s*[:：-]?\s*"
        rf"(?:score\s*[:：]?\s*)?(?P<score>\d+(?:\.\d+)?)"
        rf"(?:\s*/\s*\d+(?:\.\d+)?)?"
        rf"(?P<rest>.*?)(?=(?:\n\s*(?:[-*]?\s*)?(?:coverage|specificity|non[\s_-]?redundancy|discriminative)\b)|\Z)"
    )
    match = re.search(pattern, text)
    if not match:
        return None
    rest = re.sub(r"(?is)^(?:\s*[-–—]\s*|\s*reason\s*[:：]\s*)", "", match.group("rest") or "").strip()
    rest = re.sub(r"\s+", " ", rest).strip(" -:\n\r\t")
    return {
        "score": float(match.group("score")),
        "reason": rest,
    }


def _extract_structured_scores_from_text(text: str) -> Optional[Dict]:
    cleaned = _normalize_quotes(_strip_code_fences(text))
    cleaned = re.sub(r"[*_`]", "", cleaned)
    payload = {}
    for key, label in PRINCIPLE_LABELS.items():
        item = _extract_dimension_from_text(cleaned, key, label)
        if item is None:
            return None
        payload[key] = item
    return payload


@dataclass
class HeuristicGlobalScorer:
    scale_max: int = 5

    def _score_coverage(self, items: List[str]) -> float:
        return clamp_score(1 + min(len(items), 5) * 0.8, self.scale_max)

    def _score_specificity(self, items: List[str]) -> float:
        if not items:
            return 1.0
        action_markers = [
            "must",
            "should",
            "include",
            "explain",
            "cite",
            "step",
            "example",
            "accurate",
            "specific",
            "clear",
        ]
        hits = 0
        total = 0
        for item in items:
            lowered = item.lower()
            total += 1
            if any(marker in lowered for marker in action_markers):
                hits += 1
        return clamp_score(1 + 4 * (hits / max(total, 1)), self.scale_max)

    def _score_non_redundancy(self, items: List[str]) -> float:
        if len(items) <= 1:
            return float(self.scale_max)

        token_sets = [set(re.findall(r"[a-zA-Z]+", item.lower())) for item in items]
        overlaps = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                intersection = token_sets[i] & token_sets[j]
                union = token_sets[i] | token_sets[j]
                overlaps.append(len(intersection) / max(1, len(union)))
        redundancy = sum(overlaps) / max(1, len(overlaps))
        return clamp_score(1 + 4 * (1 - redundancy), self.scale_max)

    def _score_discriminative(self, items: List[str]) -> float:
        cues = [
            "accurate",
            "correct",
            "relevant",
            "specific",
            "evidence",
            "example",
            "reasoning",
            "compare",
            "justify",
        ]
        if not items:
            return 1.0
        hits = sum(any(cue in item.lower() for cue in cues) for item in items)
        return clamp_score(1 + 4 * (hits / max(1, len(items))), self.scale_max)

    def score(self, instruction: str, rubric: str) -> Dict[str, object]:
        items = split_rubric_items(rubric)
        coverage = self._score_coverage(items)
        specificity = self._score_specificity(items)
        non_redundancy = self._score_non_redundancy(items)
        discriminative = self._score_discriminative(items)
        average = (coverage + specificity + non_redundancy + discriminative) / 4.0
        return {
            "backend": "heuristic",
            "scale_max": self.scale_max,
            "dimensions": {
                "coverage": {
                    "label": PRINCIPLE_LABELS["coverage"],
                    "score": coverage,
                    "reason": "More distinct rubric items usually improve coverage.",
                },
                "specificity": {
                    "label": PRINCIPLE_LABELS["specificity"],
                    "score": specificity,
                    "reason": "Actionable verbs and concrete checks increase specificity.",
                },
                "non_redundancy": {
                    "label": PRINCIPLE_LABELS["non_redundancy"],
                    "score": non_redundancy,
                    "reason": "Lower pairwise overlap suggests less redundancy.",
                },
                "discriminative": {
                    "label": PRINCIPLE_LABELS["discriminative"],
                    "score": discriminative,
                    "reason": "Concrete quality cues usually make the rubric more discriminative.",
                },
            },
            "average_raw_score": average,
            "normalized_score": average / float(self.scale_max),
        }


class VLLMGlobalScorer:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        scale_max: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 512,
        fallback_to_heuristic: bool = True,
    ):
        self.model = model
        self.scale_max = scale_max
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback = HeuristicGlobalScorer(scale_max=scale_max) if fallback_to_heuristic else None
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _parse_response(self, content: str) -> Dict[str, object]:
        payload = _extract_json_blob(content)
        parse_mode = "json"
        if payload is None:
            payload = _extract_structured_scores_from_text(content)
            parse_mode = "text_recovery"
        if payload is None:
            raise ValueError("Could not parse JSON from global scorer output.")

        dimensions = {}
        scores = []
        for key, label in PRINCIPLE_LABELS.items():
            item = payload.get(key, {})
            if isinstance(item, dict):
                raw_score = item.get("score", 1)
                reason = str(item.get("reason", "")).strip()
            else:
                raw_score = item
                reason = ""
            score = clamp_score(_coerce_score_value(raw_score), self.scale_max)
            scores.append(score)
            dimensions[key] = {
                "label": label,
                "score": score,
                "reason": reason,
            }

        average = sum(scores) / len(scores)
        return {
            "backend": "vllm",
            "parse_mode": parse_mode,
            "raw_model_output": content,
            "scale_max": self.scale_max,
            "dimensions": dimensions,
            "average_raw_score": average,
            "normalized_score": average / float(self.scale_max),
        }

    def score(self, instruction: str, rubric: str) -> Dict[str, object]:
        if not self.model:
            if self.fallback is None:
                raise ValueError("No vLLM model configured and no fallback scorer available.")
            return self.fallback.score(instruction, rubric)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": GLOBAL_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": GLOBAL_USER_PROMPT.format(
                            instruction=instruction,
                            rubric=rubric,
                            scale_max=self.scale_max,
                        ),
                    },
                ],
            )
            content = completion.choices[0].message.content or ""
            return self._parse_response(content)
        except Exception as exc:
            if self.fallback is None:
                raise
            result = self.fallback.score(instruction, rubric)
            result["fallback_error"] = str(exc)
            return result


def build_global_scorer(
    model: str = "/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-4B/Qwen3-4B",
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    scale_max: int = 5,
    fallback_to_heuristic: bool = True,
):
    if model:
        return VLLMGlobalScorer(
            model=model,
            base_url=base_url,
            api_key=api_key,
            scale_max=scale_max,
            fallback_to_heuristic=fallback_to_heuristic,
        )
    return HeuristicGlobalScorer(scale_max=scale_max)
