#!/usr/bin/env python3
"""
Generate rubrics for pairwise benchmark data using a fine-tuned SFT model.

Supported benchmarks:
  - rewardbench:  RewardBench parquet file
  - rmbench:      RM-Bench JSONL/JSON file
  - rmb:          RMB JSON file or directory
  - openrubrics:  Custom JSONL file with instruction/response_a/response_b/winner fields

Output: a JSONL file with fields:
  sid, benchmark, instruction, response_a, response_b,
  raw_rubric_output, formatted_rubric, extra_info
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np  
from tqdm import tqdm  

# Shared modules live under the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from prompts import RUBRIC_GEN_SYSTEM, RUBRIC_GEN_USER_TEMPLATE
from evaluate import (  
    PairwiseSample,
    load_rewardbench,
    load_rmbench,
    load_rmb,
    build_chat_prompt,
    build_chat_messages,
    init_generation_backend,
    generate_candidates_with_backend,
)
from Judge_Rubrics.rubric_selection import build_rubric_selector, select_best_rubric, normalize_rubric_text


# ============================================================================
# OpenRubrics data loader
# ============================================================================

def load_openrubrics(path: str, shuffle_pairs: bool, seed: int) -> List[PairwiseSample]:
    """
    Load OpenRubrics format JSONL file.

    Expected fields per line:
      instruction (str), response_a (str), response_b (str),
      winner (str: "A" or "B"), pair_id (str, optional)
    """
    rng = np.random.default_rng(seed)
    samples: List[PairwiseSample] = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            instruction = row.get("instruction", "")
            response_a = row.get("response_a", "")
            response_b = row.get("response_b", "")
            winner = row.get("winner", "").upper()
            if not instruction or not response_a or not response_b:
                continue

            # Determine chosen/rejected based on the winner field
            if winner in {"A", "RESPONSE_A", "A_WIN"}:
                chosen, rejected, gt = response_a, response_b, "A"
            elif winner in {"B", "RESPONSE_B", "B_WIN"}:
                chosen, rejected, gt = response_b, response_a, "B"
            else:
                chosen, rejected, gt = response_a, response_b, "A"  # default

            resp_a, resp_b = chosen, rejected
            if shuffle_pairs and float(rng.random()) > 0.5:
                resp_a, resp_b, gt = rejected, chosen, "B" if gt == "A" else "A"

            sid = row.get("pair_id") or row.get("instruction_id") or str(row.get("id", idx))
            samples.append(PairwiseSample(
                sid=str(sid),
                instruction=instruction,
                response_a=resp_a,
                response_b=resp_b,
                ground_truth=gt,
                data_source="openrubrics",
                extra_info={
                    "dataset": "openrubrics",
                    "instruction_id": row.get("instruction_id", ""),
                    "pair_id": row.get("pair_id", ""),
                    "original_winner": winner,
                },
            ))
    return samples


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate pairwise rubrics using a fine-tuned SFT model")

    # Model
    parser.add_argument("--sft_model_path", type=str, required=True, help="Local rubric-generator path or remote model name.")
    parser.add_argument("--sft_model_base_url", type=str, default="", help="OpenAI-compatible URL for rubric generation.")
    parser.add_argument("--sft_model_api_key", type=str, default="EMPTY", help="API key for rubric generation URL mode.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--num_candidates", type=int, default=4, help="Number of rubric candidates per sample")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for rubric generation")
    parser.add_argument("--ebm_max_length", type=int, default=2048, help="Max length for EBM scoring input")
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="hierarchical",
        choices=["first", "local_ebm", "hierarchical"],
        help="How to select the final rubric from best-of-n candidates.",
    )
    parser.add_argument("--openrubrics_input_file", type=str, default=str(Path(__file__).resolve().parent.parent / "OpenRubrics.jsonl"))
    parser.add_argument("--training_artifact_dir", type=str, default=str(Path(__file__).resolve().parent.parent / "artifacts"))
    parser.add_argument("--local_ckpt", type=str, default=str(Path(__file__).resolve().parent.parent / "ebm_qwen3-4b_gsm_model.pt"))
    parser.add_argument("--local_tokenizer", type=str, default=str(Path(__file__).resolve().parent.parent / "ebm_qwen3-4b_gsm_tokenizer"))
    parser.add_argument("--global_vllm_model", type=str, default="")
    parser.add_argument("--global_vllm_base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--local_weight", type=float, default=1.0)
    parser.add_argument("--group_weight", type=float, default=1.0)
    parser.add_argument("--global_weight", type=float, default=1.0)

    # Data
    parser.add_argument(
        "--benchmark", type=str, default="rewardbench",
        choices=["rewardbench", "rmbench", "rmb", "openrubrics"],
    )
    parser.add_argument("--test_parquet", type=str, default="", help="RewardBench parquet path")
    parser.add_argument("--rmbench_jsonl", type=str, default="", help="RM-Bench data path")
    parser.add_argument("--rmb_json", type=str, default="", help="RMB data path or directory")
    parser.add_argument("--rmb_json_dir", action="store_true", help="Treat --rmb_json as a directory")
    parser.add_argument("--openrubrics_jsonl", type=str, default="", help="OpenRubrics JSONL path")
    parser.add_argument("--shuffle_pairs", action="store_true", help="Randomly swap A/B positions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0, help="Limit samples (0 = all)")

    # Output
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file path")

    args = parser.parse_args()

    # --- Load dataset ---
    if args.benchmark == "rewardbench":
        if not args.test_parquet:
            raise ValueError("--test_parquet is required for benchmark=rewardbench")
        samples = load_rewardbench(args.test_parquet, args.shuffle_pairs, args.seed)
    elif args.benchmark == "rmb":
        if not args.rmb_json:
            raise ValueError("--rmb_json is required for benchmark=rmb")
        is_dir = args.rmb_json_dir or os.path.isdir(args.rmb_json)
        samples = load_rmb(args.rmb_json, args.shuffle_pairs, args.seed, is_dir=is_dir)
    elif args.benchmark == "openrubrics":
        if not args.openrubrics_jsonl:
            raise ValueError("--openrubrics_jsonl is required for benchmark=openrubrics")
        samples = load_openrubrics(args.openrubrics_jsonl, args.shuffle_pairs, args.seed)
    else:  # rmbench
        if not args.rmbench_jsonl:
            raise ValueError("--rmbench_jsonl is required for benchmark=rmbench")
        samples = load_rmbench(args.rmbench_jsonl, args.shuffle_pairs, args.seed)

    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"Loaded {len(samples)} samples")

    # --- Load SFT model ---
    generation_mode = "url" if args.sft_model_base_url else "local_vllm"
    print(f"Loading SFT model: {args.sft_model_path} [{generation_mode}]")
    backend = init_generation_backend(
        model_name_or_path=args.sft_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        base_url=args.sft_model_base_url,
        api_key=args.sft_model_api_key,
    )

    # --- Build prompts ---
    prompts: List[str] = []
    message_payloads: List[List[Dict[str, str]]] = []
    for s in samples:
        user = RUBRIC_GEN_USER_TEMPLATE.format(
            instruction=s.instruction,
            response_a=s.response_a,
            response_b=s.response_b,
        )
        message_payloads.append(build_chat_messages(RUBRIC_GEN_SYSTEM, user))
        if backend["mode"] == "vllm":
            prompts.append(build_chat_prompt(backend["tokenizer"], RUBRIC_GEN_SYSTEM, user))

    # --- Build rubric selector ---
    selector = build_rubric_selector(
        strategy=args.selection_strategy,
        openrubrics_input_file=args.openrubrics_input_file,
        artifact_dir=args.training_artifact_dir,
        local_ckpt=args.local_ckpt,
        local_tokenizer=args.local_tokenizer,
        global_vllm_model=args.global_vllm_model,
        global_vllm_base_url=args.global_vllm_base_url,
        local_weight=args.local_weight,
        group_weight=args.group_weight,
        global_weight=args.global_weight,
        ebm_max_length=args.ebm_max_length,
    )

    # --- Generate rubrics ---
    raw_outputs: List[str] = []
    formatted: List[str] = []
    candidates_all: List[List[str]] = []
    selection_scores_all: List[List[float]] = []
    selection_details_all: List[List[Dict[str, Any]]] = []

    print("Generating rubrics...")
    payloads = prompts if backend["mode"] == "vllm" else message_payloads
    for i in tqdm(range(0, len(payloads), args.batch_size), desc="generate rubrics"):
        batch = payloads[i : i + args.batch_size]
        batch_samples = samples[i : i + args.batch_size]
        batch_outputs = generate_candidates_with_backend(
            backend=backend,
            prompt_payloads=batch,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=args.num_candidates,
        )
        for s, cands in zip(batch_samples, batch_outputs):
            if not cands:
                cands = [""]
            selection_result = select_best_rubric(
                selector=selector,
                instruction=s.instruction,
                candidates=cands,
                source=s.data_source,
            )
            best_idx = int(selection_result["best_index"])
            best_raw = selection_result["best_raw_rubric"]
            candidate_records = selection_result["candidate_records"]

            candidates_all.append(cands)
            selection_scores_all.append([float(item["selection_score"]) for item in candidate_records])
            selection_details_all.append([item["selection_details"] for item in candidate_records])
            raw_outputs.append(best_raw)
            formatted.append(normalize_rubric_text(best_raw))

    # --- Save output ---
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for s, raw, fmt, cands, scores, details in zip(
            samples,
            raw_outputs,
            formatted,
            candidates_all,
            selection_scores_all,
            selection_details_all,
        ):
            extra = dict(s.extra_info or {})
            extra.update({
                "rubric_selection_strategy": args.selection_strategy,
                "rubric_candidates": cands,
                "rubric_selection_scores": scores,
                "rubric_candidate_details": details,
                "best_rubric_index": int(max(range(len(scores)), key=lambda idx: scores[idx])) if scores else 0,
            })
            f.write(json.dumps({
                "sid": s.sid,
                "benchmark": args.benchmark,
                "instruction": s.instruction,
                "response_a": s.response_a,
                "response_b": s.response_b,
                "raw_rubric_output": raw,
                "formatted_rubric": fmt,
                "extra_info": extra,
            }, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} rubrics to: {args.output_file}")


if __name__ == "__main__":
    main()
