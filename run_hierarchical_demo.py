import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
    
from global_scorer import build_global_scorer
from group_scorer import GroupRubricScorer
from hierarchical_scorer import HierarchicalRubricScorer, HierarchicalWeights
from prepare_hierarchical_data import DEFAULT_ARTIFACT_DIR, DEFAULT_INPUT_FILE, ensure_training_artifacts
from score import LocalQualityScorer


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_artifacts(input_file: Path, artifact_dir: Path) -> Dict[str, Path]:
    artifacts = ensure_training_artifacts(
        input_file=str(input_file),
        artifact_dir=str(artifact_dir),
        hf_dataset=None,
        hf_split="train",
        overwrite=False,
    )
    return {
        "train_file": artifacts["train_file"],
        "group_artifact": artifacts["group_artifact"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a hierarchical rubric scoring smoke test.")
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--rubric", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--local-ckpt", default=None)
    parser.add_argument("--local-tokenizer", default=None)
    parser.add_argument("--vllm-model", default="/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-4B/Qwen3-4B")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--group-weight", type=float, default=1.0)
    parser.add_argument("--global-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_paths = ensure_artifacts(
        input_file=Path(args.input_file),
        artifact_dir=Path(args.artifact_dir),
    )

    if args.instruction and args.rubric:
        example = {
            "instruction": args.instruction,
            "rubric": args.rubric,
            "source": args.source or "",
        }
    else:
        train_records = load_jsonl(artifact_paths["train_file"])
        if not train_records:
            raise ValueError("No OpenRubrics training records found for smoke test.")
        example = train_records[0]

    local_scorer = LocalQualityScorer(
        ckpt_path=args.local_ckpt,
        tok_path=args.local_tokenizer,
        auto_fallback=True,
    )
    group_scorer = GroupRubricScorer.from_artifact_path(artifact_paths["group_artifact"])
    global_scorer = build_global_scorer(model=args.vllm_model, base_url=args.vllm_base_url)
    hierarchical = HierarchicalRubricScorer(
        local_scorer=local_scorer,
        group_scorer=group_scorer,
        global_scorer=global_scorer,
        weights=HierarchicalWeights(
            local=args.local_weight,
            group=args.group_weight,
            global_=args.global_weight,
        ),
    )

    result = hierarchical.score(
        instruction=example["instruction"],
        rubric=example["rubric"],
        source=example.get("source"),
        domain=example.get("domain"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
