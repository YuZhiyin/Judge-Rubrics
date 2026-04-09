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
  #   Rubric generation uses thinking; full output (including <think>...</think>)
  #   is stored in gen_text so EBM trains on the same distribution as inference.
  #   Judge calls always disable thinking (/no_think) to guarantee clean A/B output.
"""

import argparse
from collections import defaultdict
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = SCRIPT_DIR / "artifacts" / "openrubrics_train.jsonl"
if not DEFAULT_INPUT_FILE.exists():
    DEFAULT_INPUT_FILE = SCRIPT_DIR / "OpenRubrics.jsonl"


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


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RUBRIC_GEN_SYSTEM = (
    "You are an expert evaluator. Your task is to write evaluation rubric rules "
    "for comparing two responses to a given question."
)

RUBRIC_GEN_USER = (
    "Given the following question, write a concise rubric (a numbered list of evaluation "
    "criteria) that can objectively determine which of two responses is better.\n\n"
    "Question:\n%s\n\n"
    "Output only the numbered rubric rules. Be specific and measurable."
)

# Judge always disables thinking to guarantee clean A/B output within max_tokens
JUDGE_SYSTEM = (
    "/no_think You are a strict evaluator. Follow the rubric exactly and output only 'A' or 'B'."
)

JUDGE_USER = (
    "Use the rubric below to evaluate which response is better.\n\n"
    "Question:\n%s\n\n"
    "Rubric:\n%s\n\n"
    "Response A:\n%s\n\n"
    "Response B:\n%s\n\n"
    "Which response better satisfies the rubric? Reply with exactly one letter: A or B."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_winner(text: str) -> int:
    """Return 0 for A-wins, 1 for B-wins, -1 if unparseable.

    Takes the last standalone A/B in case the model reasons before answering.
    """
    t = text.strip().upper()
    matches = re.findall(r'\b([AB])\b', t)
    if matches:
        return 0 if matches[-1] == "A" else 1
    return -1


def winner_to_int(winner_str: str) -> int:
    """Convert dataset winner field ('A'/'B') to int (0/1). Returns -1 if invalid."""
    v = str(winner_str).strip().upper()
    if v == "A":
        return 0
    if v == "B":
        return 1
    return -1


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
        return []

    # Step 1: generate N rubrics in one API call via n=N
    try:
        rubric_resp = call_with_retry(lambda: client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": RUBRIC_GEN_SYSTEM},
                {"role": "user",   "content": RUBRIC_GEN_USER % instruction},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens_rubric,
            n=args.n_gen,
        ))
        rubrics = [c.message.content.strip() for c in rubric_resp.choices]
    except Exception as e:
        print(f"[warn] rubric gen failed qid={idx}: {e}")
        return []

    # Step 2: judge each rubric concurrently
    def judge_one(rubric):
        return call_with_retry(lambda: client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": JUDGE_USER % (
                    instruction, rubric, response_a, response_b,
                )},
            ],
            temperature=0.0,
            max_tokens=args.max_tokens_judge,
        ).choices[0].message.content.strip())

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
    for rubric, judge_out in zip(rubrics, judge_outputs):
        if not judge_out:
            continue
        predicted = parse_winner(judge_out)
        if predicted == -1:
            continue
        records.append({
            "question_id": idx,
            "question":    instruction,
            "response_a":  response_a,
            "response_b":  response_b,
            "answer":      answer,
            "gen_text":    rubric,
            "source":      source,
            "domain":      example.get("domain", ""),
            "label":       1 if predicted == answer else 0,
        })

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

    per_question_stats = []

    for qid, labels in qid_to_labels.items():
        if not labels:
            continue

        consistency = sum(labels) / len(labels)

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
    p.add_argument("--max-tokens-rubric", type=int,   default=4*8192,
                   help="Max tokens for rubric generation (includes thinking chain)")
    p.add_argument("--max-tokens-judge",  type=int,   default=64)
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
