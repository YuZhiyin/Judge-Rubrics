from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class HierarchicalWeights:
    local: float = 1.0
    group: float = 1.0
    global_: float = 1.0

    def normalized(self) -> Dict[str, float]:
        total = self.local + self.group + self.global_
        if total <= 0:
            raise ValueError("At least one weight must be positive.")
        return {
            "local": self.local / total,
            "group": self.group / total,
            "global": self.global_ / total,
        }


class HierarchicalRubricScorer:
    def __init__(self, local_scorer, group_scorer, global_scorer, weights: Optional[HierarchicalWeights] = None):
        self.local_scorer = local_scorer
        self.group_scorer = group_scorer
        self.global_scorer = global_scorer
        self.weights = weights or HierarchicalWeights()

    def score(
        self,
        instruction: str,
        rubric: str,
        source: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, object]:
        local_result = self.local_scorer.score(instruction, rubric)
        group_result = self.group_scorer.score(instruction, rubric, source=source, domain=domain)
        global_result = self.global_scorer.score(instruction, rubric)
        weights = self.weights.normalized()

        final_score = (
            weights["local"] * float(local_result["normalized_score"])
            + weights["group"] * float(group_result["normalized_score"])
            + weights["global"] * float(global_result["normalized_score"])
        )

        return {
            "instruction": instruction,
            "rubric": rubric,
            "source": source or "",
            "domain": domain or group_result.get("matched_domain") or group_result.get("requested_domain"),
            "weights": weights,
            "local": local_result,
            "group": group_result,
            "global": global_result,
            "final_score": final_score,
        }
