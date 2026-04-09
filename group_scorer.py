import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from EnergyORM.domain_mapping import infer_domain


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_PATH = SCRIPT_DIR / "artifacts" / "group_centroids.pkl"


def _load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _safe_norm(vector: np.ndarray) -> float:
    norm = float(np.linalg.norm(vector))
    return norm if norm > 0 else 1e-12


@dataclass
class GroupCentroidArtifact:
    vectorizer: TfidfVectorizer
    centroids: Dict[str, np.ndarray]
    domain_counts: Dict[str, int]


def build_group_artifact(
    records: Sequence[Dict],
    rubric_key: str = "rubric",
    min_df: int = 2,
    max_features: int = 20000,
) -> GroupCentroidArtifact:
    cleaned_records = []
    texts = []
    for record in records:
        rubric = (record.get(rubric_key) or record.get("gen_text") or "").strip()
        if not rubric:
            continue
        domain = record.get("domain") or infer_domain(record.get("instruction", ""), record.get("source", ""))
        cleaned_record = dict(record)
        cleaned_record["domain"] = domain
        cleaned_records.append(cleaned_record)
        texts.append(rubric)

    if not texts:
        raise ValueError("No rubric texts found for building group centroids.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
    )
    matrix = vectorizer.fit_transform(texts)

    indices_by_domain: Dict[str, List[int]] = {}
    for index, record in enumerate(cleaned_records):
        indices_by_domain.setdefault(record["domain"], []).append(index)

    centroids = {}
    domain_counts = {}
    for domain, indices in indices_by_domain.items():
        centroid = np.asarray(matrix[indices].mean(axis=0)).ravel()
        centroids[domain] = centroid
        domain_counts[domain] = len(indices)

    return GroupCentroidArtifact(
        vectorizer=vectorizer,
        centroids=centroids,
        domain_counts=domain_counts,
    )


def save_group_artifact(artifact: GroupCentroidArtifact, artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as handle:
        pickle.dump(artifact, handle)


def load_group_artifact(artifact_path: Path) -> GroupCentroidArtifact:
    with artifact_path.open("rb") as handle:
        return pickle.load(handle)


class GroupRubricScorer:
    def __init__(self, artifact: GroupCentroidArtifact):
        self.artifact = artifact

    @classmethod
    def from_artifact_path(cls, artifact_path: Path) -> "GroupRubricScorer":
        return cls(load_group_artifact(artifact_path))

    def _cosine_similarity(self, rubric: str, domain: str) -> float:
        vector = self.artifact.vectorizer.transform([rubric])
        dense = np.asarray(vector.toarray()[0]).ravel()
        centroid = self.artifact.centroids[domain]
        similarity = float(np.dot(dense, centroid) / (_safe_norm(dense) * _safe_norm(centroid)))
        return max(0.0, min(1.0, similarity))

    def score(
        self,
        instruction: str,
        rubric: str,
        source: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, object]:
        inferred_domain = domain or infer_domain(instruction, source)
        matched_domain = inferred_domain if inferred_domain in self.artifact.centroids else None

        if matched_domain is None:
            best_domain = None
            best_similarity = -1.0
            for candidate_domain in self.artifact.centroids:
                similarity = self._cosine_similarity(rubric, candidate_domain)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_domain = candidate_domain
            matched_domain = best_domain or "general"
            similarity = max(0.0, best_similarity)
        else:
            similarity = self._cosine_similarity(rubric, matched_domain)

        distance = 1.0 - similarity
        return {
            "backend": "tfidf_centroid",
            "requested_domain": inferred_domain,
            "matched_domain": matched_domain,
            "domain_count": self.artifact.domain_counts.get(matched_domain, 0),
            "cosine_similarity": similarity,
            "distance": distance,
            "normalized_score": similarity,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build group-level rubric centroids.")
    parser.add_argument("--train-file", required=True, help="JSONL file with rubric-bearing training records.")
    parser.add_argument("--artifact-path", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--rubric-key", default="rubric")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_records = _load_jsonl(Path(args.train_file))
    artifact = build_group_artifact(
        train_records,
        rubric_key=args.rubric_key,
        min_df=args.min_df,
        max_features=args.max_features,
    )
    save_group_artifact(artifact, Path(args.artifact_path))
    print(f"Saved group centroid artifact to {args.artifact_path}")
    print(json.dumps(artifact.domain_counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
