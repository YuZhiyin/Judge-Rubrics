"""Generate training data for rubric EBM scorer.

For each example in OpenRubrics:
  - Generate rubric rules N times (default 4) via vLLM
  - Use the generated rubric to judge response_a vs response_b
  - label=1 if the rubric correctly predicts the winner, else label=0

Output JSONL format:
  {"question_id": int, "question": str, "response_a": str, "response_b": str,
   "answer": int, "gen_text": str, "source": str, "label": int}

Usage (Qwen3-8B example):
  # 1. Start vLLM server:
  #    python -m vllm.entrypoints.openai.api_server \
  #        --model Qwen/Qwen3-8B \
  #        --dtype bfloat16 \
  #        --max-model-len 32768
  # 2. Run this script:
  #    python gen_data.py --model Qwen/Qwen3-8B --output-file output.jsonl
  #
  # Thinking mode (Qwen3):
  #   Rubric generation may return <think>...</think> before the actual rubric.
  #   We strip thinking text before judging and before writing gen_text, while
  #   preserving raw_gen_text for debugging if it differs from the cleaned rubric.
  #   Judge calls always disable thinking (/no_think) to guarantee clean A/B output.
"""

import argparse
from collections import Counter, defaultdict
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm



from prompts import (
    RUBRIC_GEN_SYSTEM,
    RUBRIC_GEN_USER_TEMPLATE,
    RUBRIC_JUDGE_SYSTEM,
    RUBRIC_JUDGE_USER_TEMPLATE,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = "/mnt/shared-storage-user/yuzhiyin/Judge-Rubrics/OpenRubrics.jsonl"
# if not DEFAULT_INPUT_FILE.exists():
#     DEFAULT_INPUT_FILE = SCRIPT_DIR / "OpenRubrics.jsonl"


def call_with_retry(fn, max_retries: int = 3, base_delay: float = 2.0):
    """Call fn(), retrying up to max_retries times with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = base_delay * (2 ** attempt)
            print(f"[retry {attempt+1}/{max_retries}] {e} — retrying in {wait:.0f}s")
            time.sleep(wait)


def chat_completion_with_token_backoff(
    client: OpenAI,
    *,
    max_tokens: int,
    min_max_tokens: int = 1,
    safety_margin: int = 256,
    max_retries: int = 3,
    base_delay: float = 2.0,
    **kwargs,
):
    """Retry chat completion, shrinking max_tokens if vLLM reports context overflow."""
    current_max_tokens = max_tokens
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                **kwargs,
                max_tokens=current_max_tokens,
            )
        except Exception as e:
            adjusted_max_tokens = _adjust_max_tokens_from_error(
                e,
                current_max_tokens=current_max_tokens,
                min_max_tokens=min_max_tokens,
                safety_margin=safety_margin,
            )
            if adjusted_max_tokens is not None and adjusted_max_tokens < current_max_tokens:
                print(
                    f"[retry {attempt+1}/{max_retries}] reducing max_tokens "
                    f"{current_max_tokens}->{adjusted_max_tokens}: {e}"
                )
                current_max_tokens = adjusted_max_tokens
                continue

            if attempt == max_retries:
                raise
            wait = base_delay * (2 ** attempt)
            print(f"[retry {attempt+1}/{max_retries}] {e} — retrying in {wait:.0f}s")
            time.sleep(wait)


def _adjust_max_tokens_from_error(
    error: Exception,
    *,
    current_max_tokens: int,
    min_max_tokens: int,
    safety_margin: int,
) -> Optional[int]:
    message = str(error)
    if "max_tokens" not in message and "max_completion_tokens" not in message:
        return None
    if "maximum context length" not in message:
        return None

    match = re.search(
        r"maximum context length is\s+(\d+)\s+tokens and your request has\s+(\d+)\s+input tokens",
        message,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    context_limit = int(match.group(1))
    input_tokens = int(match.group(2))
    adjusted = context_limit - input_tokens - safety_margin
    if adjusted < min_max_tokens:
        return None
    return min(current_max_tokens - 1, adjusted)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_winner(text: str) -> int:
    """Return 0 for A-wins, 1 for B-wins, -1 if unparseable.

    Prefer the explicit Winner line because rubric judge outputs include both
    "Response A" and "Response B" in their analysis sections.
    """
    t = text.strip()
    if not t:
        return -1

    for line in reversed(t.splitlines()):
        if not re.search(r"\bwinner\b", line, flags=re.IGNORECASE):
            continue
        winner = _parse_winner_fragment(line)
        if winner != -1:
            return winner

    final_sections = re.split(r"(?i)---\s*Final Judgment\s*---", t)
    if len(final_sections) > 1:
        winner = _parse_winner_fragment(final_sections[-1])
        if winner != -1:
            return winner

    stripped = re.sub(r"(?is)<think\b[^>]*>.*?(?:</think>|<\\think>)", " ", t).strip()
    if re.fullmatch(r"(?i)(?:Response\s*)?[AB]\.?", stripped):
        return 0 if "A" in stripped.upper() else 1

    # Last-resort fallback for terse outputs like "Final answer: B".
    matches = re.findall(r"(?im)^\s*(?:final answer|answer|choice)\s*[:：-]?\s*(?:response\s*)?([AB])\b", t)
    if matches:
        return 0 if matches[-1].upper() == "A" else 1
    return -1


def _parse_winner_fragment(text: str) -> int:
    normalized = re.sub(r"[*_`]", "", text).strip()
    normalized = re.sub(r"\[[^\]]*Response\s*A\s*/\s*Response\s*B[^\]]*\]", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\[[^\]]*Response\s*B\s*/\s*Response\s*A[^\]]*\]", "", normalized, flags=re.IGNORECASE)

    if re.search(r"(?i)\bResponse\s*A\s*/\s*Response\s*B\b|\bResponse\s*B\s*/\s*Response\s*A\b", normalized):
        return -1

    match = re.search(
        r"(?i)\b(?:winner|final answer|answer|choice)\b\s*[:：-]?\s*(?:the winner is\s*)?(?:response\s*)?([AB])\b",
        normalized,
    )
    if match:
        return 0 if match.group(1).upper() == "A" else 1

    match = re.fullmatch(r"(?i)(?:response\s*)?([AB])\.?", normalized)
    if match:
        return 0 if match.group(1).upper() == "A" else 1

    return -1


def winner_to_int(winner_str: str) -> int:
    """Convert dataset winner field ('A'/'B') to int (0/1). Returns -1 if invalid."""
    v = str(winner_str).strip().upper()
    if v == "A":
        return 0
    if v == "B":
        return 1
    return -1


def strip_thinking_text(text: str) -> str:
    """Remove Qwen-style thinking blocks before using model output as a rubric.

    Handles both the standard closing tag (</think>) and the common typo
    (<\think>). If a generation is truncated inside a thinking block, keep the
    original text rather than returning an empty rubric so the caller can still
    inspect/debug the sample.
    """
    if not text:
        return ""

    cleaned = re.sub(
        r"<think\b[^>]*>.*?(?:</think>|<\\think>)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    if cleaned:
        return cleaned

    # Fallback for outputs that close the thought with a variant tag/newline
    # but were not matched by the regex above.
    for tag in ("</think>", "<\\think>"):
        idx = text.lower().rfind(tag)
        if idx != -1:
            tail = text[idx + len(tag):].strip()
            if tail:
                return tail

    return text.strip()


# ---------------------------------------------------------------------------
# Core generation for a single example
# ---------------------------------------------------------------------------

def process_example(client: OpenAI, example: dict, idx: int, args: argparse.Namespace) -> list:
    """Generate N rubrics for one example and return labelled records."""
    instruction = example["instruction"]
    response_a  = example["response_a"]
    response_b  = example["response_b"]
    answer      = winner_to_int(example["winner"])
    source      = example.get("source", "")

    if not instruction or not response_a or not response_b or answer == -1:
        print(f"[warn] invalid example qid={idx}: answer={example.get('winner')!r}")
        return []

    # Step 1: generate N rubrics in one API call via n=N
    try:
        rubric_resp = chat_completion_with_token_backoff(
            client,
            model=args.model,
            messages=[
                {"role": "system", "content": RUBRIC_GEN_SYSTEM},
                {
                    "role": "user",
                    "content": RUBRIC_GEN_USER_TEMPLATE.format(
                        instruction=instruction,
                        response_a=response_a,
                        response_b=response_b,
                    ),
                },
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens_rubric,
            min_max_tokens=args.min_max_tokens_rubric,
            safety_margin=args.context_safety_margin,
            n=args.n_gen,
        )
        raw_rubrics = [c.message.content.strip() for c in rubric_resp.choices]
        rubrics = [strip_thinking_text(r) for r in raw_rubrics]
    except Exception as e:
        print(f"[warn] rubric gen failed qid={idx}: {e}")
        return []

    # Step 2: judge each rubric concurrently
    def judge_one(rubric):
        return chat_completion_with_token_backoff(
            client,
            model=args.model,
            messages=[
                {"role": "system", "content": RUBRIC_JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": RUBRIC_JUDGE_USER_TEMPLATE.format(
                        instruction=instruction,
                        response_a=response_a,
                        response_b=response_b,
                        rubric=rubric,
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=min(args.max_tokens_judge, args.max_tokens_judge_cap),
            min_max_tokens=1,
            safety_margin=args.context_safety_margin,
        ).choices[0].message.content.strip()

    judge_outputs = [""] * len(rubrics)
    with ThreadPoolExecutor(max_workers=len(rubrics)) as pool:
        future_to_i = {pool.submit(judge_one, r): i for i, r in enumerate(rubrics)}
        for fut in as_completed(future_to_i):
            i = future_to_i[fut]
            try:
                judge_outputs[i] = fut.result()
            except Exception as e:
                print(f"[warn] judge failed qid={idx} rubric_idx={i}: {e}")

    # Step 3: label
    records = []
    for rubric_idx, (raw_rubric, rubric, judge_out) in enumerate(zip(raw_rubrics, rubrics, judge_outputs)):
        if not judge_out:
            continue
        predicted = parse_winner(judge_out)
        if predicted == -1:
            preview = re.sub(r"\s+", " ", judge_out).strip()[:240]
            print(f"[warn] unparseable judge qid={idx} rubric_idx={rubric_idx}: {preview!r}")
            continue
        record = {
            "question_id": idx,
            "question":    instruction,
            "response_a":  response_a,
            "response_b":  response_b,
            "answer":      answer,
            "gen_text":    rubric,
            "source":      source,
            "domain":      example.get("domain", ""),
            "label":       1 if predicted == answer else 0,
        }
        if raw_rubric != rubric:
            record["raw_gen_text"] = raw_rubric
        records.append(record)

    return records

def sort_output_by_qid(out_path: Path):
    print("\n[sort] sorting output by question_id...")

    records = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                continue

    records.sort(key=lambda x: x["question_id"])

    sorted_path = out_path.with_name(out_path.stem + "_sorted.jsonl")

    with open(sorted_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[sort] saved sorted file to: {sorted_path}")

def compute_and_save_stats(out_path: Path):
    """
    统计按 question_id 聚合后的 label 分布，并计算 consistency。
    consistency = mean(label) = 该 question 下 label=1 的比例
    """
    print("\n[stats] computing label distribution per question_id...")

    qid_to_labels = defaultdict(list)

    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                qid = rec["question_id"]
                label = rec["label"]
                qid_to_labels[qid].append(label)
            except Exception:
                continue

    total_q = len(qid_to_labels)
    all_zero = 0
    all_one = 0
    mixed = 0
    record_count_hist = Counter()

    per_question_stats = []

    for qid, labels in qid_to_labels.items():
        if not labels:
            continue

        consistency = sum(labels) / len(labels)
        record_count_hist[len(labels)] += 1

        if all(l == 0 for l in labels):
            all_zero += 1
        elif all(l == 1 for l in labels):
            all_one += 1
        else:
            mixed += 1

        per_question_stats.append({
            "question_id": qid,
            "num_records": len(labels),
            "num_label_1": sum(labels),
            "num_label_0": len(labels) - sum(labels),
            "consistency": consistency,
            "all_zero": int(all(l == 0 for l in labels)),
            "all_one": int(all(l == 1 for l in labels)),
        })

    avg_consistency = (
        sum(x["consistency"] for x in per_question_stats) / len(per_question_stats)
        if per_question_stats else 0.0
    )

    print(f"[stats] total questions: {total_q}")
    print(f"[stats] all label=0 questions: {all_zero}")
    print(f"[stats] all label=1 questions: {all_one}")
    print(f"[stats] mixed-label questions: {mixed}")
    print(f"[stats] records per question: {dict(sorted(record_count_hist.items()))}")
    print(f"[stats] avg consistency: {avg_consistency:.4f}")

    # 保存每个 question 的统计结果
    stats_path = out_path.with_suffix(".question_stats.jsonl")
    with open(stats_path, "w", encoding="utf-8") as f:
        for item in per_question_stats:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[stats] per-question stats saved to: {stats_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate rubric EBM training data from OpenRubrics")
    p.add_argument("--input-file",         default=str(DEFAULT_INPUT_FILE), help="Local JSONL input file.")
    p.add_argument("--hf-dataset",         default=None, help="Optional HF dataset name if no local input file is used.")
    p.add_argument("--hf-split",           default="train")
    p.add_argument("--model",             required=True,  help="Model name/path served by vLLM")
    p.add_argument("--base-url",          default="http://localhost:8000/v1")
    p.add_argument("--output-file",       required=True,  help="Output JSONL path")
    p.add_argument("--n-gen",             type=int,   default=4,    help="Rubric generations per example")
    p.add_argument("--max-samples",       type=int,   default=None, help="Cap dataset size (for debugging)")
    p.add_argument("--source",            default=None,             help="Filter by source tag")
    p.add_argument("--workers",           type=int,   default=8,    help="Concurrent examples")
    p.add_argument("--temperature",       type=float, default=1,  help="Temperature for rubric generation")
    p.add_argument("--max-tokens-rubric", type=int,   default=24576,
                   help="Max tokens for rubric generation (includes thinking chain)")
    p.add_argument("--min-max-tokens-rubric", type=int, default=1024,
                   help="Smallest rubric max_tokens to retry with after context-overflow errors")
    p.add_argument("--max-tokens-judge",  type=int,   default=1024)
    p.add_argument("--max-tokens-judge-cap", type=int, default=2048,
                   help="Hard cap for judge output budget; judge should never need a long completion")
    p.add_argument("--context-safety-margin", type=int, default=256,
                   help="Token margin kept free when retrying context-overflow requests")
    p.add_argument("--resume",            action="store_true",      help="Skip already-written question_ids")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading OpenRubrics dataset...")
    if args.input_file:
        ds = load_dataset("json", data_files=args.input_file, split="train")
        print(f"Loaded local file: {args.input_file}")
    elif args.hf_dataset:
        ds = load_dataset(args.hf_dataset, split=args.hf_split)
        print(f"Loaded Hugging Face dataset: {args.hf_dataset} [{args.hf_split}]")
    else:
        raise ValueError("Either --input-file or --hf-dataset must be provided.")

    if args.source:
        ds = ds.filter(lambda x: x["source"] == args.source)
        print(f"Filtered to source='{args.source}': {len(ds)} examples")

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"Total examples: {len(ds)}")

    # Resume: collect already-done question_ids
    done_ids: set = set()
    if args.resume and Path(args.output_file).exists():
        with open(args.output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["question_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(done_ids)} question_ids already done")

    client = OpenAI(base_url=args.base_url, api_key="dummy")

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    mode = "a" if args.resume else "w"

    with open(out_path, mode, encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_example, client, example, idx, args): idx
                for idx, example in enumerate(ds)
                if idx not in done_ids
            }

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                idx = futures[fut]
                try:
                    records = fut.result()
                except Exception as e:
                    print(f"[warn] example idx={idx} raised: {e}")
                    records = []

                for rec in records:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_written += 1

                if total_written % 500 == 0 and total_written > 0:
                    out_f.flush()
                    print(f"written={total_written}")

    print(f"[done] total_written={total_written} saved to {out_path}")
    
    compute_and_save_stats(out_path)
    sort_output_by_qid(out_path)


if __name__ == "__main__":
    main()
