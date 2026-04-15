#!/usr/bin/env python3
"""
Evaluation script for pairwise judge models.

Supported benchmarks:
  - rewardbench: RewardBench dataset (parquet format)
  - rmbench:     RM-Bench dataset (JSONL/JSON format)
  - rmb:         RMB dataset (JSON format)

Supported prompt types:
  - direct_judge:  Judge directly without a rubric
  - rubric_judge:  Judge using a pre-generated rubric (no reference answer)

Outputs:
  - per-sample generations JSONL
  - summary metrics JSON (overall + per category)
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from openai import OpenAI  # type: ignore
try:
    from vllm import LLM, SamplingParams  # type: ignore
except ModuleNotFoundError:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

# Shared modules live under the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Judge_Rubrics.prompts import (  # noqa: E402
    RUBRIC_JUDGE_SYSTEM,
    RUBRIC_JUDGE_USER_TEMPLATE,
    DIRECT_JUDGE_SYSTEM,
    DIRECT_JUDGE_USER_TEMPLATE,
    RUBRIC_GEN_SYSTEM,
    RUBRIC_GEN_USER_TEMPLATE,
)
from Judge_Rubrics.rubric_selection import build_rubric_selector, select_best_rubric, normalize_rubric_text


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class PairwiseSample:
    """A single pairwise evaluation sample."""
    sid: str
    instruction: str
    response_a: str
    response_b: str
    ground_truth: str  # "A" or "B"
    data_source: str
    extra_info: Optional[Dict[str, Any]] = None


# ============================================================================
# Data loading
# ============================================================================

def _prompt_to_instruction(raw_prompt: Any) -> str:
    """Convert a raw prompt field to a plain instruction string."""
    if isinstance(raw_prompt, list):
        # Chat messages format: extract the last user message
        last_user = [m for m in raw_prompt if isinstance(m, dict) and m.get("role") == "user"]
        if last_user:
            return str(last_user[-1].get("content", ""))
        return json.dumps(raw_prompt, ensure_ascii=False)

    if isinstance(raw_prompt, str):
        s = raw_prompt.strip()
        # Try to parse JSON strings (e.g. serialized message list)
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return _prompt_to_instruction(json.loads(s))
            except Exception:
                return s
        return s

    return str(raw_prompt) if raw_prompt is not None else ""


def load_rewardbench(path: str, shuffle_pairs: bool, seed: int) -> List[PairwiseSample]:
    """
    Load RewardBench parquet data.

    Expected columns: prompt, chosen, rejected, subset (optional), id (optional).
    """
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(path)

    missing = {"prompt", "chosen", "rejected"} - set(df.columns)
    if missing:
        raise ValueError(f"RewardBench parquet missing required columns: {sorted(missing)}. Got: {list(df.columns)}")

    samples: List[PairwiseSample] = []
    for idx, row in df.iterrows():
        instruction = _prompt_to_instruction(row.get("prompt"))
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))
        if not instruction or not chosen or not rejected:
            continue

        resp_a, resp_b, gt = chosen, rejected, "A"
        if shuffle_pairs and float(rng.random()) > 0.5:
            resp_a, resp_b, gt = rejected, chosen, "B"

        samples.append(PairwiseSample(
            sid=str(row.get("id", idx)),
            instruction=instruction,
            response_a=resp_a,
            response_b=resp_b,
            ground_truth=gt,
            data_source=str(row.get("data_source", row.get("subset", "rewardbench"))),
            extra_info={
                "dataset": "rewardbench",
                "subset": row.get("subset", ""),
                "data_source": row.get("data_source", ""),
            },
        ))
    return samples


def _read_rmbench_rows(path: str) -> List[Dict[str, Any]]:
    """Read RM-Bench data supporting both .jsonl and .json formats."""
    p = os.path.abspath(path)

    if p.lower().endswith(".json"):
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError(f"RM-Bench JSON top-level must be a list[dict]: {p} got {type(obj)}")
        return obj

    rows: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError(f"RM-Bench JSONL each line must be a dict: {p} L{ln} got {type(obj)}")
            rows.append(obj)
    return rows


def load_rmbench(path: str, shuffle_pairs: bool, seed: int) -> List[PairwiseSample]:
    """
    Load RM-Bench data and expand into pairwise comparisons.

    Each record has chosen/rejected as lists of 3 responses,
    expanded into 3×3 = 9 pairwise comparisons.
    """
    rng = np.random.default_rng(seed)
    rows = _read_rmbench_rows(path)
    samples: List[PairwiseSample] = []

    for idx, row in enumerate(rows):
        raw_prompt = row.get("prompt") or row.get("instruction") or ""
        instruction = _prompt_to_instruction(raw_prompt)
        if not instruction:
            continue

        domain = row.get("domain") or row.get("subset") or row.get("category") or "rmbench"
        chosen_list = row.get("chosen") or row.get("response_chosen")
        rejected_list = row.get("rejected") or row.get("response_rejected")
        if not isinstance(chosen_list, list) or not isinstance(rejected_list, list):
            continue
        if len(chosen_list) != 3 or len(rejected_list) != 3:
            continue

        rmbench_id = row.get("id", str(idx))
        for i in range(3):
            for j in range(3):
                chosen = str(chosen_list[i])
                rejected = str(rejected_list[j])
                if not chosen or not rejected:
                    continue

                resp_a, resp_b, gt = chosen, rejected, "A"
                if shuffle_pairs and float(rng.random()) > 0.5:
                    resp_a, resp_b, gt = rejected, chosen, "B"

                samples.append(PairwiseSample(
                    sid=f"{rmbench_id}:{i}:{j}",
                    instruction=instruction,
                    response_a=resp_a,
                    response_b=resp_b,
                    ground_truth=gt,
                    data_source=f"rmbench_{domain}",
                    extra_info={
                        "dataset": "rmbench",
                        "domain": domain,
                        "rmbench_id": rmbench_id,
                        "chosen_style_idx": i,
                        "rejected_style_idx": j,
                    },
                ))
    return samples


def _find_json_files(directory: str) -> List[str]:
    """Recursively find all .json files in a directory."""
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return sorted(json_files)


def load_rmb(path: str, shuffle_pairs: bool, seed: int, is_dir: bool = False) -> List[PairwiseSample]:
    """
    Load RMB dataset.

    Expected fields: conversation_input (or prompt), chosen.answer, reject.answer,
                     category_path (optional), pair_uid (optional).
    """
    rng = np.random.default_rng(seed)
    json_files = _find_json_files(path) if is_dir else [path]
    if is_dir and not json_files:
        raise ValueError(f"No JSON files found in RMB directory: {path}")

    samples: List[PairwiseSample] = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"RMB JSON top-level must be list[dict]: {json_file} got {type(data)}")

        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            raw_prompt = item.get("conversation_input") or item.get("prompt") or item.get("instruction", "")
            instruction = _prompt_to_instruction(raw_prompt)
            chosen = item.get("chosen", {}).get("answer", "")
            rejected = item.get("reject", {}).get("answer", "")
            if not instruction or not chosen or not rejected:
                continue

            # Extract top-level category from path like "Pairwise_set/Helpfulness/Brainstorming"
            category_path = item.get("category_path", item.get("category", ""))
            top_category = "unknown"
            if category_path:
                parts = [p.strip() for p in category_path.split("/") if p.strip()]
                if len(parts) >= 2 and parts[0].lower() in {"pairwise_set", "pairwise", "rmb"}:
                    top_category = parts[1]
                elif parts and parts[0].lower() not in {"pairwise_set", "pairwise", "rmb"}:
                    top_category = parts[0]

            resp_a, resp_b, gt = str(chosen), str(rejected), "A"
            if shuffle_pairs and float(rng.random()) > 0.5:
                resp_a, resp_b, gt = str(rejected), str(chosen), "B"

            samples.append(PairwiseSample(
                sid=str(item.get("pair_uid", item.get("id", f"{os.path.basename(json_file)}_{idx}"))),
                instruction=instruction,
                response_a=resp_a,
                response_b=resp_b,
                ground_truth=gt,
                data_source=category_path or top_category or "rmb",
                extra_info={
                    "dataset": "rmb",
                    "category_path": category_path,
                    "top_category": top_category,
                    "source_file": os.path.basename(json_file),
                },
            ))
    return samples


# ============================================================================
# Utilities
# ============================================================================

def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    """Build a chat prompt string using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_chat_messages(system: str, user: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def init_generation_backend(
    model_name_or_path: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    base_url: str = "",
    api_key: str = "EMPTY",
):
    if base_url:
        return {
            "mode": "openai",
            "model": model_name_or_path,
            "client": OpenAI(base_url=base_url, api_key=api_key),
            "tokenizer": None,
        }

    if LLM is None or SamplingParams is None:
        raise ModuleNotFoundError(
            "vllm is required for local-path inference mode. "
            "Install/use an environment with vllm, or pass a --*_base_url to use URL mode."
        )

    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    return {
        "mode": "vllm",
        "model": model_name_or_path,
        "llm": llm,
        "tokenizer": llm.get_tokenizer(),
    }


def generate_candidates_with_backend(
    backend,
    prompt_payloads,
    temperature: float,
    max_tokens: int,
    n: int = 1,
):
    if backend["mode"] == "vllm":
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n)
        outputs = backend["llm"].generate(prompt_payloads, sampling)
        return [[candidate.text for candidate in output.outputs] for output in outputs]

    results = []
    client = backend["client"]
    for messages in prompt_payloads:
        completion = client.chat.completions.create(
            model=backend["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            messages=messages,
        )
        results.append([(choice.message.content or "") for choice in completion.choices])
    return results


def _generate_best_of_n_rubrics(
    samples: List[PairwiseSample],
    backend,
    batch_size: int,
    num_candidates: int,
    temperature: float,
    max_tokens: int,
    selector,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    prompt_payloads = []
    for sample in samples:
        user = RUBRIC_GEN_USER_TEMPLATE.format(
            instruction=sample.instruction,
            response_a=sample.response_a,
            response_b=sample.response_b,
        )
        if backend["mode"] == "vllm":
            prompt_payloads.append(build_chat_prompt(backend["tokenizer"], RUBRIC_GEN_SYSTEM, user))
        else:
            prompt_payloads.append(build_chat_messages(RUBRIC_GEN_SYSTEM, user))

    raw_rubrics: List[str] = []
    formatted_rubrics: List[str] = []
    rubric_metadata: List[Dict[str, Any]] = []

    for start in tqdm(range(0, len(prompt_payloads), batch_size), desc="generate[rubric_best_of_n]"):
        batch_prompts = prompt_payloads[start : start + batch_size]
        batch_samples = samples[start : start + batch_size]
        outputs = generate_candidates_with_backend(
            backend=backend,
            prompt_payloads=batch_prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            n=num_candidates,
        )
        for sample, candidates in zip(batch_samples, outputs):
            result = select_best_rubric(
                selector=selector,
                instruction=sample.instruction,
                candidates=candidates,
                source=sample.data_source,
            )
            raw_rubrics.append(result["best_raw_rubric"])
            formatted_rubrics.append(normalize_rubric_text(result["best_formatted_rubric"]))
            rubric_metadata.append(result)

    return raw_rubrics, formatted_rubrics, rubric_metadata


def _parse_winner(text: str) -> Optional[str]:
    """Extract the predicted winner ('A' or 'B') from model output."""
    patterns = [
        r"Final Winner:\s*Response\s*([AB])\b",
        r"Final Winner:\s*([AB])\b",
        r"Winner:\s*Response\s*([AB])\b",
        r"Winner:\s*([AB])\b",
        r":\s*Response\s*([AB])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def _rewardbench_category(data_source: str) -> str:
    """Map a RewardBench data_source/subset name to one of 4 official categories."""
    s = str(data_source or "").strip().lower().replace("_", "-")
    if not s:
        return "other"

    direct_map = {
        "rewardbench-chat": "chat", "chat": "chat",
        "rewardbench-chat-hard": "chat_hard", "chat-hard": "chat_hard",
        "rewardbench-safety": "safety", "safety": "safety",
        "rewardbench-reasoning": "reasoning", "reasoning": "reasoning",
    }
    if s in direct_map:
        return direct_map[s]

    chat_subsets = {"alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard", "mt-bench-easy", "mt-bench-med"}
    chat_hard_subsets = {
        "mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor",
        "llmbar-adver-gptinst", "llmbar-adver-gptout", "llmbar-adver-manual",
    }
    safety_subsets = {
        "refusals-dangerous", "refusals-offensive", "xstest-should-refuse",
        "xstest-should-respond", "donotanswer",
    }
    reasoning_subsets = {"math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust"}

    if s in chat_subsets:
        return "chat"
    if s in chat_hard_subsets:
        return "chat_hard"
    if s in safety_subsets:
        return "safety"
    if s in reasoning_subsets:
        return "reasoning"
    return "other"


def _category_tag(sample: PairwiseSample, benchmark: str) -> str:
    """Return the category tag used for per-category metrics grouping."""
    if benchmark == "rmbench":
        return "rmbench"
    if benchmark == "rmb":
        top = (sample.extra_info or {}).get("top_category", "")
        if top:
            return top.lower()
        return sample.data_source.split("/")[0].lower() if "/" in sample.data_source else sample.data_source.lower()
    return _rewardbench_category(sample.data_source)


# ============================================================================
# Metrics
# ============================================================================

def _compute_metrics(winners: List[Optional[str]], ground_truths: List[str]) -> Dict[str, float]:
    """Compute accuracy and parse-rate metrics for a list of predictions."""
    if not winners:
        return {"num_samples": 0.0, "acc_all": 0.0, "acc_parseable": 0.0, "parse_rate": 0.0}

    parseable_pairs = [(p, g) for p, g in zip(winners, ground_truths) if p in {"A", "B"}]
    parse_rate = len(parseable_pairs) / len(winners)
    acc_all = sum(1 for p, g in zip(winners, ground_truths) if p in {"A", "B"} and p == g) / len(winners)
    acc_parseable = (
        sum(1 for p, g in parseable_pairs if p == g) / len(parseable_pairs) if parseable_pairs else 0.0
    )

    return {
        "num_samples": float(len(winners)),
        "acc_all": float(acc_all),
        "acc_parseable": float(acc_parseable),
        "parse_rate": float(parse_rate),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pairwise judge model on reward benchmarks")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base_url", type=str, default="", help="OpenAI-compatible URL for judge inference.")
    parser.add_argument("--model_api_key", type=str, default="EMPTY", help="API key for judge URL mode.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=32)

    # Data
    parser.add_argument("--benchmark", type=str, required=True, choices=["rewardbench", "rmbench", "rmb"])
    parser.add_argument("--test_parquet", type=str, default="", help="RewardBench parquet path (benchmark=rewardbench)")
    parser.add_argument("--rmbench_jsonl", type=str, default="", help="RM-Bench data path (benchmark=rmbench)")
    parser.add_argument("--rmb_json", type=str, default="", help="RMB data path or directory (benchmark=rmb)")
    parser.add_argument("--rmb_json_dir", action="store_true", help="Treat --rmb_json as a directory")
    parser.add_argument("--shuffle_pairs", action="store_true", help="Randomly swap A/B positions to reduce position bias")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0, help="Limit number of samples (0 = all)")

    # Prompt
    parser.add_argument(
        "--prompt_type", type=str, required=True,
        choices=["direct_judge", "rubric_judge"],
        help=(
            "direct_judge: judge without any rubric; "
            "rubric_judge: judge using a pre-generated rubric (no reference answer)"
        ),
    )
    parser.add_argument("--rubrics_file", type=str, default="", help="Rubrics JSONL file (required for rubric_judge)")
    parser.add_argument(
        "--generate_rubrics_on_the_fly",
        action="store_true",
        help="Generate rubric candidates per sample and select the best one during evaluation.",
    )
    parser.add_argument(
        "--rubric_generator_model_path",
        type=str,
        default="",
        help="Optional rubric generation model path or model name. If omitted, reuse --model_path.",
    )
    parser.add_argument("--rubric_generator_base_url", type=str, default="", help="OpenAI-compatible URL for rubric generator inference.")
    parser.add_argument("--rubric_generator_api_key", type=str, default="EMPTY", help="API key for rubric generator URL mode.")
    parser.add_argument("--rubric_num_candidates", type=int, default=4)
    parser.add_argument("--rubric_generation_temperature", type=float, default=0.7)
    parser.add_argument("--rubric_generation_max_tokens", type=int, default=4096)
    parser.add_argument("--rubric_selection_strategy", type=str, default="hierarchical", choices=["first", "local_ebm", "hierarchical"])
    parser.add_argument("--openrubrics_input_file", type=str, default=str(Path(__file__).resolve().parent.parent / "OpenRubrics.jsonl"))
    parser.add_argument("--training_artifact_dir", type=str, default=str(Path(__file__).resolve().parent.parent / "artifacts"))
    parser.add_argument("--local_ckpt", type=str, default=str(Path(__file__).resolve().parent.parent / "ebm_qwen3-4b_gsm_model.pt"))
    parser.add_argument("--local_tokenizer", type=str, default=str(Path(__file__).resolve().parent.parent / "ebm_qwen3-4b_gsm_tokenizer"))
    parser.add_argument("--global_vllm_model", type=str, default="")
    parser.add_argument("--global_vllm_base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--hierarchical_local_weight", type=float, default=1.0)
    parser.add_argument("--hierarchical_group_weight", type=float, default=1.0)
    parser.add_argument("--hierarchical_global_weight", type=float, default=1.0)

    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="", help="Custom path for metrics JSON (optional)")

    args = parser.parse_args()

    if args.prompt_type == "rubric_judge" and not args.rubrics_file and not args.generate_rubrics_on_the_fly:
        raise ValueError(
            "For prompt_type=rubric_judge you must provide --rubrics_file "
            "or enable --generate_rubrics_on_the_fly."
        )

    # --- Load dataset ---
    print(f"Loading {args.benchmark} dataset...")
    if args.benchmark == "rewardbench":
        if not args.test_parquet:
            raise ValueError("--test_parquet is required for benchmark=rewardbench")
        samples = load_rewardbench(args.test_parquet, args.shuffle_pairs, args.seed)
    elif args.benchmark == "rmbench":
        if not args.rmbench_jsonl:
            raise ValueError("--rmbench_jsonl is required for benchmark=rmbench")
        samples = load_rmbench(args.rmbench_jsonl, args.shuffle_pairs, args.seed)
    else:  # rmb
        if not args.rmb_json:
            raise ValueError("--rmb_json is required for benchmark=rmb")
        is_dir = args.rmb_json_dir or os.path.isdir(args.rmb_json)
        samples = load_rmb(args.rmb_json, args.shuffle_pairs, args.seed, is_dir=is_dir)

    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"Loaded {len(samples)} samples")

    # --- Load model ---
    judge_mode = "url" if args.model_base_url else "local_vllm"
    print(f"Loading judge model: {args.model_path} [{judge_mode}]")
    judge_backend = init_generation_backend(
        model_name_or_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        base_url=args.model_base_url,
        api_key=args.model_api_key,
    )

    # --- Load rubrics ---
    formatted_rubrics: List[str] = [""] * len(samples)
    rubric_raw_outputs: List[str] = [""] * len(samples)
    rubric_selection_metadata: List[Dict[str, Any]] = [{} for _ in samples]

    if args.prompt_type == "rubric_judge":
        if args.generate_rubrics_on_the_fly:
            selector = build_rubric_selector(
                strategy=args.rubric_selection_strategy,
                openrubrics_input_file=args.openrubrics_input_file,
                artifact_dir=args.training_artifact_dir,
                local_ckpt=args.local_ckpt,
                local_tokenizer=args.local_tokenizer,
                global_vllm_model=args.global_vllm_model,
                global_vllm_base_url=args.global_vllm_base_url,
                local_weight=args.hierarchical_local_weight,
                group_weight=args.hierarchical_group_weight,
                global_weight=args.hierarchical_global_weight,
            )

            generator_model_path = args.rubric_generator_model_path or args.model_path
            generator_base_url = args.rubric_generator_base_url or args.model_base_url
            generator_api_key = args.rubric_generator_api_key if args.rubric_generator_base_url else args.model_api_key
            if generator_model_path == args.model_path and generator_base_url == args.model_base_url:
                rubric_backend = judge_backend
            else:
                generator_mode = "url" if generator_base_url else "local_vllm"
                print(f"Loading rubric generator model: {generator_model_path} [{generator_mode}]")
                rubric_backend = init_generation_backend(
                    model_name_or_path=generator_model_path,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    base_url=generator_base_url,
                    api_key=generator_api_key,
                )

            print(
                f"Generating rubrics on the fly with strategy={args.rubric_selection_strategy}, "
                f"num_candidates={args.rubric_num_candidates}"
            )
            rubric_raw_outputs, formatted_rubrics, rubric_selection_metadata = _generate_best_of_n_rubrics(
                samples=samples,
                backend=rubric_backend,
                batch_size=args.batch_size,
                num_candidates=args.rubric_num_candidates,
                temperature=args.rubric_generation_temperature,
                max_tokens=args.rubric_generation_max_tokens,
                selector=selector,
            )
        else:
            print(f"Loading rubrics: {args.rubrics_file}")
            rubrics_by_sid: Dict[str, Dict[str, Any]] = {}
            with open(args.rubrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    sid = rec.get("sid", "")
                    if sid:
                        rubrics_by_sid[sid] = rec

            # formatted_rubrics = [rubrics_by_sid.get(s.sid, {}).get("formatted_rubric", "") for s in samples]
            formatted_rubrics = [
                normalize_rubric_text(
                    rubrics_by_sid.get(s.sid, {}).get("formatted_rubric", "")
                    or rubrics_by_sid.get(s.sid, {}).get("raw_rubric_output", "")
                )
                for s in samples
            ]
            rubric_raw_outputs = [rubrics_by_sid.get(s.sid, {}).get("raw_rubric_output", "") for s in samples]
            rubric_selection_metadata = [rubrics_by_sid.get(s.sid, {}).get("extra_info", {}) for s in samples]
            matched = sum(1 for r in formatted_rubrics if r)
            print(f"Matched {matched}/{len(samples)} rubrics")

    # --- Select prompt templates ---
    if args.prompt_type == "rubric_judge":
        SYSTEM_PROMPT = RUBRIC_JUDGE_SYSTEM
        USER_TEMPLATE = RUBRIC_JUDGE_USER_TEMPLATE
    else:  # direct_judge
        SYSTEM_PROMPT = DIRECT_JUDGE_SYSTEM
        USER_TEMPLATE = DIRECT_JUDGE_USER_TEMPLATE

    # --- Build prompts ---
    prompts: List[str] = []
    message_payloads: List[List[Dict[str, str]]] = []
    for i, s in enumerate(samples):
        if args.prompt_type == "rubric_judge":
            user = USER_TEMPLATE.format(
                instruction=s.instruction,
                response_a=s.response_a,
                response_b=s.response_b,
                rubric=formatted_rubrics[i],
            )
        else:
            user = USER_TEMPLATE.format(
                instruction=s.instruction,
                response_a=s.response_a,
                response_b=s.response_b,
            )
        message_payloads.append(build_chat_messages(SYSTEM_PROMPT, user))
        if judge_backend["mode"] == "vllm":
            prompts.append(build_chat_prompt(judge_backend["tokenizer"], SYSTEM_PROMPT, user))

    # --- Generate ---
    print("Running inference...")
    outs_text: List[str] = []
    winners: List[Optional[str]] = []
    out_lens: List[int] = []

    payloads = prompts if judge_backend["mode"] == "vllm" else message_payloads
    for i in tqdm(range(0, len(payloads), args.batch_size), desc=f"generate[{args.prompt_type}]"):
        batch = payloads[i : i + args.batch_size]
        batch_outputs = generate_candidates_with_backend(
            backend=judge_backend,
            prompt_payloads=batch,
            temperature=0.0,
            max_tokens=args.max_tokens,
            n=1,
        )
        for candidates in batch_outputs:
            txt = candidates[0] if candidates else ""
            outs_text.append(txt)
            winners.append(_parse_winner(txt))
            out_lens.append(len(txt))

    # --- Save per-sample results ---
    os.makedirs(args.output_dir, exist_ok=True)
    gen_path = os.path.join(args.output_dir, f"generations_{args.prompt_type}.jsonl")
    with open(gen_path, "w", encoding="utf-8") as f:
        for idx, (s, out_txt, pred, olen) in enumerate(zip(samples, outs_text, winners, out_lens)):
            row = {
                "sid": s.sid,
                "benchmark": args.benchmark,
                "prompt_type": args.prompt_type,
                "data_source": s.data_source,
                "category": _category_tag(s, args.benchmark),
                "ground_truth": s.ground_truth,
                "winner_pred": pred,
                "parse_ok": bool(pred in {"A", "B"}),
                "is_correct": bool(pred in {"A", "B"} and pred == s.ground_truth),
                "gen_len_chars": olen,
                "instruction": s.instruction,
                "response_a": s.response_a,
                "response_b": s.response_b,
                "model_output": out_txt,
                "extra_info": s.extra_info or {},
            }
            if args.prompt_type == "rubric_judge":
                row["rubric_used"] = formatted_rubrics[idx]
                row["raw_rubric_output"] = rubric_raw_outputs[idx]
                row["rubric_selection_metadata"] = rubric_selection_metadata[idx]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- Compute overall metrics ---
    ground_truths = [s.ground_truth for s in samples]
    overall = _compute_metrics(winners, ground_truths)

    # --- Per-category metrics ---
    idxs_by_cat: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        idxs_by_cat[_category_tag(s, args.benchmark)].append(i)

    by_category: Dict[str, Dict[str, float]] = {}
    for cat in sorted(idxs_by_cat):
        idxs = idxs_by_cat[cat]
        by_category[cat] = _compute_metrics([winners[i] for i in idxs], [ground_truths[i] for i in idxs])

    result: Dict[str, Any] = {
        "benchmark": args.benchmark,
        "prompt_type": args.prompt_type,
        "num_samples": len(samples),
        "accuracy_all": overall["acc_all"],
        "accuracy_parseable": overall["acc_parseable"],
        "parse_rate": overall["parse_rate"],
        "avg_gen_len_chars": float(np.mean(out_lens)) if out_lens else 0.0,
        "by_category": by_category,
        "generations_jsonl": os.path.abspath(gen_path),
        "rubric_generation": {
            "enabled": bool(args.generate_rubrics_on_the_fly),
            "selection_strategy": args.rubric_selection_strategy if args.prompt_type == "rubric_judge" else "",
            "rubric_generator_model_path": (
                (args.rubric_generator_model_path or args.model_path)
                if args.generate_rubrics_on_the_fly and args.prompt_type == "rubric_judge"
                else ""
            ),
            "judge_inference_mode": judge_backend["mode"],
            "judge_base_url": args.model_base_url,
            "num_candidates": args.rubric_num_candidates if args.prompt_type == "rubric_judge" else 0,
        },
    }

    # RMB: weighted accuracy across categories
    if args.benchmark == "rmb" and by_category:
        total = sum(v["num_samples"] for v in by_category.values())
        if total > 0:
            result["accuracy_weighted"] = float(
                sum(v["acc_all"] * v["num_samples"] for v in by_category.values()) / total
            )

    # RM-Bench: per-row-style accuracy matrix
    if args.benchmark == "rmbench":
        grouped: Dict[str, List[Optional[str]]] = defaultdict(list)
        grouped_gt: Dict[str, List[str]] = defaultdict(list)
        grouped_pos: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        for s, pred_w in zip(samples, winners):
            info = s.extra_info or {}
            rid = str(info.get("rmbench_id", s.sid.split(":")[0]))
            grouped[rid].append(pred_w)
            grouped_gt[rid].append(s.ground_truth)
            try:
                i, j = int(info.get("chosen_style_idx", -1)), int(info.get("rejected_style_idx", -1))
            except Exception:
                i, j = -1, -1
            grouped_pos[rid].append((i, j))

        MATRIX_SIZE = 3
        win_rates = []
        matrices: List[np.ndarray] = []

        for rid, preds in grouped.items():
            gts = grouped_gt[rid]
            m = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
            correct = total_pos = 0
            for (p, g), (ii, jj) in zip(zip(preds, gts), grouped_pos[rid]):
                if ii < 0 or jj < 0 or ii >= MATRIX_SIZE or jj >= MATRIX_SIZE:
                    continue
                if p in {"A", "B"}:
                    total_pos += 1
                    score = 1.0 if p == g else 0.0
                    if p == g:
                        correct += 1
                else:
                    score = 0.0
                m[ii, jj] = score
            win_rates.append(correct / total_pos if total_pos else 0.0)
            matrices.append(m)

        result["rmbench_num_original"] = len(grouped)
        result["rmbench_grouped_win_rate_mean"] = float(np.mean(win_rates)) if win_rates else 0.0
        result["rmbench_grouped_win_rate_std"] = float(np.std(win_rates)) if win_rates else 0.0

        if matrices:
            acc_matrix = np.mean(np.stack(matrices, axis=0), axis=0)
            n_off = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
            result["rmbench_hard_acc"] = float(np.sum(np.triu(acc_matrix, 1)) / n_off)
            result["rmbench_normal_acc"] = float(np.mean(np.diag(acc_matrix)))
            result["rmbench_easy_acc"] = float(np.sum(np.tril(acc_matrix, -1)) / n_off)
            result["rmbench_avg_acc"] = float(
                (result["rmbench_hard_acc"] + result["rmbench_normal_acc"] + result["rmbench_easy_acc"]) / 3.0
            )
            result["rmbench_acc_matrix"] = acc_matrix.tolist()

    # --- Save metrics ---
    out_path = (
        os.path.abspath(args.output_json)
        if args.output_json
        else os.path.join(args.output_dir, f"metrics_{args.prompt_type}.json")
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()
