import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys

from datasets import load_dataset

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from EnergyORM.domain_mapping import (
    attach_domain_to_records,
    build_domain_summary,
    pretty_print_summary,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = SCRIPT_DIR / "OpenRubrics.jsonl"
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
DEFAULT_TRAIN_OUTPUT = DEFAULT_ARTIFACT_DIR / "openrubrics_train.jsonl"
DEFAULT_WITH_DOMAIN_OUTPUT = DEFAULT_ARTIFACT_DIR / "openrubrics_with_domain.jsonl"
DEFAULT_STATS_OUTPUT = DEFAULT_ARTIFACT_DIR / "openrubrics_domain_stats.json"
DEFAULT_GROUP_ARTIFACT = DEFAULT_ARTIFACT_DIR / "group_centroids.pkl"


def load_records(input_file: Optional[str] = None, hf_dataset: Optional[str] = None, hf_split: str = "train") -> List[Dict]:
    if input_file:
        path = Path(input_file)
        records = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if hf_dataset:
        dataset = load_dataset(hf_dataset, split=hf_split)
        return [dict(row) for row in dataset]

    raise ValueError("Either input_file or hf_dataset must be provided.")


def stratified_split_by_source(
    records: Sequence[Dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for record in records:
        buckets[(record.get("source") or "").strip().lower()].append(dict(record))

    train_records: List[Dict] = []
    test_records: List[Dict] = []

    for source, items in buckets.items():
        rng.shuffle(items)
        if len(items) == 1:
            n_test = 0
        else:
            n_test = max(1, int(round(len(items) * test_ratio)))
            n_test = min(n_test, len(items) - 1)
        test_records.extend(items[:n_test])
        train_records.extend(items[n_test:])

    rng.shuffle(train_records)
    rng.shuffle(test_records)
    return train_records, test_records


def save_jsonl(records: Sequence[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_dataset_splits(
    input_file: Optional[str],
    hf_dataset: Optional[str],
    hf_split: str,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    records = load_records(input_file=input_file, hf_dataset=hf_dataset, hf_split=hf_split)
    records = attach_domain_to_records(records)
    train_records, test_records = stratified_split_by_source(records, test_ratio=test_ratio, seed=seed)
    summary = build_domain_summary(records)
    summary["train_size"] = len(train_records)
    summary["test_size"] = len(test_records)
    summary["test_ratio"] = test_ratio
    summary["seed"] = seed
    return train_records, test_records, summary


def prepare_training_corpus(
    input_file: Optional[str],
    hf_dataset: Optional[str],
    hf_split: str,
) -> Tuple[List[Dict], Dict[str, object]]:
    records = load_records(input_file=input_file, hf_dataset=hf_dataset, hf_split=hf_split)
    records = attach_domain_to_records(records)
    summary = build_domain_summary(records)
    summary["train_size"] = len(records)
    summary["test_size"] = 0
    summary["split_mode"] = "full_openrubrics_as_train"
    return records, summary


def ensure_training_artifacts(
    input_file: Optional[str] = str(DEFAULT_INPUT_FILE),
    artifact_dir: Optional[str] = str(DEFAULT_ARTIFACT_DIR),
    hf_dataset: Optional[str] = None,
    hf_split: str = "train",
    overwrite: bool = False,
) -> Dict[str, Path]:
    artifact_root = Path(artifact_dir or DEFAULT_ARTIFACT_DIR)
    artifact_root.mkdir(parents=True, exist_ok=True)

    train_output = artifact_root / "openrubrics_train.jsonl"
    with_domain_output = artifact_root / "openrubrics_with_domain.jsonl"
    stats_output = artifact_root / "openrubrics_domain_stats.json"
    group_artifact = artifact_root / "group_centroids.pkl"

    rebuild_training_files = overwrite or not train_output.exists() or not with_domain_output.exists() or not stats_output.exists()
    if not rebuild_training_files and stats_output.exists():
        try:
            existing_summary = json.loads(stats_output.read_text(encoding="utf-8"))
            if existing_summary.get("split_mode") != "full_openrubrics_as_train":
                rebuild_training_files = True
        except Exception:
            rebuild_training_files = True

    if rebuild_training_files:
        train_records, summary = prepare_training_corpus(
            input_file=input_file,
            hf_dataset=hf_dataset,
            hf_split=hf_split,
        )
        save_jsonl(train_records, train_output)
        save_jsonl(train_records, with_domain_output)
        stats_output.write_text(pretty_print_summary(summary), encoding="utf-8")

    if rebuild_training_files or overwrite or not group_artifact.exists():
        from EnergyORM.group_scorer import build_group_artifact, save_group_artifact

        train_records = load_records(input_file=str(train_output), hf_dataset=None, hf_split="train")
        artifact = build_group_artifact(train_records, rubric_key="rubric")
        save_group_artifact(artifact, group_artifact)

    return {
        "artifact_dir": artifact_root,
        "train_file": train_output,
        "with_domain_file": with_domain_output,
        "stats_file": stats_output,
        "group_artifact": group_artifact,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OpenRubrics training artifacts with domain labels.")
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE), help="Local JSONL input file.")
    parser.add_argument("--hf-dataset", default=None, help="Optional HF dataset name if no local file is used.")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--train-output", default=None)
    parser.add_argument("--test-output", default=None)
    parser.add_argument("--with-domain-output", default=None)
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-test-set",
        action="store_true",
        help="If set, keep the old behavior of splitting OpenRubrics into train/test. "
        "By default the full dataset is treated as training data.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_output = Path(args.train_output) if args.train_output else artifact_dir / "openrubrics_train.jsonl"
    with_domain_output = (
        Path(args.with_domain_output) if args.with_domain_output else artifact_dir / "openrubrics_with_domain.jsonl"
    )
    stats_output = Path(args.stats_output) if args.stats_output else artifact_dir / "openrubrics_domain_stats.json"

    if args.split_test_set:
        test_output = Path(args.test_output) if args.test_output else artifact_dir / "openrubrics_test.jsonl"
        train_records, test_records, summary = prepare_dataset_splits(
            input_file=args.input_file,
            hf_dataset=args.hf_dataset,
            hf_split=args.hf_split,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        all_records = list(train_records) + list(test_records)
        save_jsonl(all_records, with_domain_output)
        save_jsonl(train_records, train_output)
        save_jsonl(test_records, test_output)
        stats_output.write_text(pretty_print_summary(summary), encoding="utf-8")

        print(f"Saved {len(all_records)} domain-annotated records to {with_domain_output}")
        print(f"Saved {len(train_records)} train records to {train_output}")
        print(f"Saved {len(test_records)} test records to {test_output}")
        print(f"Saved domain summary to {stats_output}")
        print(pretty_print_summary(summary))
        return

    artifacts = ensure_training_artifacts(
        input_file=args.input_file,
        artifact_dir=str(artifact_dir),
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        overwrite=args.overwrite,
    )
    summary = json.loads(artifacts["stats_file"].read_text(encoding="utf-8"))
    print(f"Saved {summary['train_size']} full-training records to {artifacts['train_file']}")
    print(f"Saved domain-annotated file to {artifacts['with_domain_file']}")
    print(f"Saved domain summary to {artifacts['stats_file']}")
    print(f"Saved group centroid artifact to {artifacts['group_artifact']}")
    print(pretty_print_summary(summary))


if __name__ == "__main__":
    main()
