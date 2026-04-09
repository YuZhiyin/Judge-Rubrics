import json
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, MutableMapping, Optional


DOMAIN_ORDER = [
    "coding",
    "math_reasoning",
    "writing_editing",
    "safety_policy",
    "instruction_following",
    "medicine_health",
    "science",
    "general",
]

SOURCE_TO_DOMAIN = {
    "biology": "science",
    "chemistry": "science",
    "physics": "science",
    "medicine": "medicine_health",
    "medical_o1": "medicine_health",
    "health_medicine": "medicine_health",
    "helpsteer3": "coding",
    "ifeval": "instruction_following",
    "skywork": "general",
    "tulu": "general",
    "ultrafeedback": "general",
}

DOMAIN_KEYWORDS = {
    "coding": [
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "cpp",
        "code",
        "program",
        "algorithm",
        "function",
        "class",
        "debug",
        "sql",
        "api",
    ],
    "math_reasoning": [
        "equation",
        "solve",
        "calculate",
        "proof",
        "theorem",
        "integral",
        "derivative",
        "algebra",
        "geometry",
        "sequence",
        "probability",
        "math",
    ],
    "writing_editing": [
        "essay",
        "write",
        "rewrite",
        "edit",
        "grammar",
        "translate",
        "story",
        "blog",
        "paragraph",
        "email",
        "summarize",
        "tone",
    ],
    "safety_policy": [
        "harmful",
        "dangerous",
        "safe",
        "safety",
        "policy",
        "illegal",
        "weapon",
        "self-harm",
        "suicide",
        "explosive",
        "malware",
        "cyberattack",
    ],
    "instruction_following": [
        "at least",
        "bullet",
        "format",
        "placeholder",
        "must include",
        "follow the instruction",
        "respond with",
        "markdown",
        "json",
    ],
    "medicine_health": [
        "patient",
        "diagnosis",
        "treatment",
        "disease",
        "symptom",
        "clinical",
        "medical",
        "medicine",
        "health",
        "doctor",
        "therapy",
    ],
    "science": [
        "physics",
        "chemistry",
        "biology",
        "experiment",
        "molecule",
        "cell",
        "protein",
        "force",
        "energy",
        "reaction",
    ],
}


def normalize_source(source: Optional[str]) -> str:
    return (source or "").strip().lower()


def normalize_text(text: Optional[str]) -> str:
    return " ".join((text or "").lower().split())


def _keyword_score(text: str, keyword: str) -> int:
    if not text or not keyword:
        return 0
    return max(1, len(keyword.split())) if keyword in text else 0


def infer_domain(instruction: str, source: Optional[str] = None) -> str:
    normalized_source = normalize_source(source)
    hard_domain = SOURCE_TO_DOMAIN.get(normalized_source)
    if hard_domain and hard_domain != "general":
        return hard_domain

    text = normalize_text(instruction)
    domain_scores = Counter()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            domain_scores[domain] += _keyword_score(text, keyword)

    if domain_scores:
        best_domain, best_score = domain_scores.most_common(1)[0]
        if best_score > 0:
            return best_domain

    return hard_domain or "general"


def attach_domain(record: MutableMapping) -> MutableMapping:
    record = dict(record)
    record["domain"] = infer_domain(record.get("instruction", ""), record.get("source", ""))
    return record


def attach_domain_to_records(records: Iterable[MutableMapping]) -> List[MutableMapping]:
    return [attach_domain(record) for record in records]


def build_domain_summary(records: Iterable[MutableMapping]) -> Dict[str, object]:
    source_counts = Counter()
    domain_counts = Counter()
    domain_by_source = defaultdict(Counter)

    for record in records:
        source = normalize_source(record.get("source", ""))
        domain = record.get("domain") or infer_domain(record.get("instruction", ""), source)
        source_counts[source] += 1
        domain_counts[domain] += 1
        domain_by_source[source][domain] += 1

    return {
        "total_examples": sum(source_counts.values()),
        "source_counts": dict(source_counts),
        "domain_counts": dict(domain_counts),
        "source_to_domain_breakdown": {
            source: dict(counter) for source, counter in sorted(domain_by_source.items())
        },
    }


def pretty_print_summary(summary: Dict[str, object]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
