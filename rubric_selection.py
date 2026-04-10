import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from EnergyORM.global_scorer import build_global_scorer
from EnergyORM.group_scorer import GroupRubricScorer
from EnergyORM.hierarchical_scorer import HierarchicalRubricScorer, HierarchicalWeights
from EnergyORM.prepare_hierarchical_data import DEFAULT_ARTIFACT_DIR, DEFAULT_INPUT_FILE, ensure_training_artifacts
from EnergyORM.score import LocalQualityScorer, load_ebm_scorer, score_one


def normalize_rubric_text(rubric_text: str) -> str:
    text = (rubric_text or "").strip()
    if not text:
        return "1. (empty rubric)"
    return text


def _sample_source(sample: Any) -> str:
    if isinstance(sample, dict):
        return str(sample.get("source") or sample.get("data_source") or "")
    return str(getattr(sample, "data_source", "") or getattr(sample, "source", "") or "")


@dataclass
class RubricSelectorBundle:
    strategy: str
    local_ebm_scorer: Optional[Dict] = None
    hierarchical_scorer: Optional[HierarchicalRubricScorer] = None
    ebm_max_length: int = 2048


def build_rubric_selector(
    strategy: str = "hierarchical",
    openrubrics_input_file: str = str(DEFAULT_INPUT_FILE),
    artifact_dir: str = str(DEFAULT_ARTIFACT_DIR),
    local_ckpt: Optional[str] = None,
    local_tokenizer: Optional[str] = None,
    global_vllm_model: str = "",
    global_vllm_base_url: str = "http://localhost:8000/v1",
    local_weight: float = 1.0,
    group_weight: float = 1.0,
    global_weight: float = 1.0,
    ebm_max_length: int = 2048,
) -> RubricSelectorBundle:
    strategy = strategy.lower()
    if strategy not in {"first", "local_ebm", "hierarchical"}:
        raise ValueError(f"Unsupported rubric selection strategy: {strategy}")

    if strategy == "first":
        return RubricSelectorBundle(strategy=strategy, ebm_max_length=ebm_max_length)

    if strategy == "local_ebm":
        load_kwargs = {}
        if local_ckpt:
            load_kwargs["ckpt_path"] = local_ckpt
        if local_tokenizer:
            load_kwargs["tok_path"] = local_tokenizer
        local_ebm = load_ebm_scorer(**load_kwargs)
        return RubricSelectorBundle(
            strategy=strategy,
            local_ebm_scorer=local_ebm,
            ebm_max_length=ebm_max_length,
        )

    artifacts = ensure_training_artifacts(
        input_file=openrubrics_input_file,
        artifact_dir=artifact_dir,
        overwrite=False,
    )
    local_scorer = LocalQualityScorer(
        ckpt_path=local_ckpt,
        tok_path=local_tokenizer,
        auto_fallback=True,
    )
    group_scorer = GroupRubricScorer.from_artifact_path(artifacts["group_artifact"])
    global_scorer = build_global_scorer(model=global_vllm_model, base_url=global_vllm_base_url)
    hierarchical = HierarchicalRubricScorer(
        local_scorer=local_scorer,
        group_scorer=group_scorer,
        global_scorer=global_scorer,
        weights=HierarchicalWeights(
            local=local_weight,
            group=group_weight,
            global_=global_weight,
        ),
    )
    return RubricSelectorBundle(
        strategy=strategy,
        hierarchical_scorer=hierarchical,
        ebm_max_length=ebm_max_length,
    )


def select_best_rubric(
    selector: RubricSelectorBundle,
    instruction: str,
    candidates: Iterable[str],
    source: str = "",
) -> Dict[str, Any]:
    candidates = [candidate for candidate in candidates]
    if not candidates:
        candidates = [""]

    candidate_records: List[Dict[str, Any]] = []

    for index, candidate in enumerate(candidates):
        formatted = normalize_rubric_text(candidate)

        if selector.strategy == "first":
            candidate_records.append(
                {
                    "index": index,
                    "raw_rubric": candidate,
                    "formatted_rubric": formatted,
                    "selection_score": 0.0 if index > 0 else 1.0,
                    "selection_details": {"backend": "first_candidate"},
                }
            )
            continue

        if selector.strategy == "local_ebm":
            energy = score_one(
                instruction,
                formatted,
                max_length=selector.ebm_max_length,
                scorer=selector.local_ebm_scorer,
            )
            candidate_records.append(
                {
                    "index": index,
                    "raw_rubric": candidate,
                    "formatted_rubric": formatted,
                    "selection_score": -float(energy),
                    "selection_details": {
                        "backend": "local_ebm",
                        "energy": float(energy),
                    },
                }
            )
            continue

        result = selector.hierarchical_scorer.score(
            instruction=instruction,
            rubric=formatted,
            source=source,
        )
        candidate_records.append(
            {
                "index": index,
                "raw_rubric": candidate,
                "formatted_rubric": formatted,
                "selection_score": float(result["final_score"]),
                "selection_details": result,
            }
        )

    best_record = max(candidate_records, key=lambda item: item["selection_score"])
    return {
        "best_index": int(best_record["index"]),
        "best_raw_rubric": best_record["raw_rubric"],
        "best_formatted_rubric": best_record["formatted_rubric"],
        "candidate_records": candidate_records,
    }
