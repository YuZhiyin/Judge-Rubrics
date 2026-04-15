import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional
import sys

import torch
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from Judge_Rubrics.ebm_model import TransEBM
from Judge_Rubrics.step1_train_ebm_v3_pretrained import PretrainedEnergyScorer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_PATH = SCRIPT_DIR / "ebm_qwen3-4b_gsm_model.pt"
DEFAULT_TOK_PATH = SCRIPT_DIR / "ebm_qwen3-4b_gsm_tokenizer"

_SCORER_CACHE = None


def _default_cfg_path_from_ckpt(ckpt_path: str) -> Path:
    ckpt = Path(ckpt_path)
    name = ckpt.name
    if name.endswith("_model.pt"):
        return ckpt.with_name(name[:-len("_model.pt")] + "_config.json")
    if ckpt.suffix == ".pt":
        return ckpt.with_suffix(".json")
    return ckpt.with_name(ckpt.name + "_config.json")


def _load_optional_config(ckpt_path: str) -> Dict[str, Any]:
    cfg_path = _default_cfg_path_from_ckpt(ckpt_path)
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_ebm_scorer(
    ckpt_path: str = str(DEFAULT_CKPT_PATH),
    tok_path: str = str(DEFAULT_TOK_PATH),
    device: Optional[str] = None,
    d_model: int = 768,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.2,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = _load_optional_config(ckpt_path)
    tok = AutoTokenizer.from_pretrained(tok_path)

    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})

    if tok.cls_token_id is not None:
        cls_id = tok.cls_token_id
    elif tok.bos_token_id is not None:
        cls_id = tok.bos_token_id
    else:
        cls_id = tok.eos_token_id

    pad_id = tok.pad_token_id

    state = torch.load(ckpt_path, map_location=device)
    is_pretrained = (
        cfg.get("arch") == "PretrainedEnergyScorer"
        or any(str(k).startswith("backbone.") for k in state.keys())
    )

    if is_pretrained:
        backbone = cfg.get("backbone")
        if not backbone:
            raise ValueError(
                f"Checkpoint {ckpt_path} looks like a pretrained EBM, but no backbone "
                f"was found in {_default_cfg_path_from_ckpt(ckpt_path)}."
            )
        model = PretrainedEnergyScorer(
            backbone_path=backbone,
            tokenizer_size=len(tok),
            pad_token_id=pad_id,
            dropout=float(cfg.get("dropout", dropout)),
            freeze_backbone=bool(cfg.get("freeze_backbone", False)),
            pooling=str(cfg.get("pooling", "cls_mean")),
            gradient_checkpointing=False,
        ).to(device)
        cls_id = int(cfg.get("cls_id", cls_id))
    else:
        model = TransEBM(
            vocab_size=len(tok),
            d_model=int(cfg.get("d_model", d_model)),
            n_heads=int(cfg.get("n_heads", n_heads)),
            n_layers=int(cfg.get("n_layers", n_layers)),
            dropout=float(cfg.get("dropout", dropout)),
        ).to(device)

        if len(tok) != model.emb.num_embeddings:
            model.resize_token_embeddings(len(tok))

    model.load_state_dict(state)
    model.eval()

    return {
        "model": model,
        "tok": tok,
        "cls_id": cls_id,
        "pad_id": pad_id,
        "device": device,
        "arch": cfg.get("arch", "TransEBM"),
        "max_length": int(cfg.get("max_length", 2048)),
    }


def _get_cached_scorer():
    global _SCORER_CACHE
    if _SCORER_CACHE is None:
        _SCORER_CACHE = load_ebm_scorer()
    return _SCORER_CACHE


def _split_rubric_items(rubric: str):
    lines = [line.strip() for line in rubric.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    pieces = re.split(r"(?:\b\d+\.\s+|(?:^|\s)-\s+)", rubric)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


@torch.no_grad()
def score_one(question: str, answer: str, max_length: int = 2048, scorer: Optional[Dict] = None) -> float:
    if scorer is None:
        scorer = _get_cached_scorer()

    tok = scorer["tok"]
    cls_id = scorer["cls_id"]
    pad_id = scorer["pad_id"]
    model = scorer["model"]
    device = scorer["device"]

    sep = tok.sep_token or tok.eos_token or "\n"
    combined = f"{question}{sep}{answer}"
    ids = tok.encode(
        combined,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - 1,
    )
    ids = [cls_id] + ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = (input_ids != pad_id).long()
    energy = model(input_ids, attention_mask)
    return float(energy.item())


class HeuristicLocalScorer:
    def __init__(self):
        self.backend = "heuristic"

    def score(self, instruction: str, rubric: str) -> Dict[str, object]:
        instruction_tokens = set(re.findall(r"[a-zA-Z]{3,}", instruction.lower()))
        rubric_tokens = set(re.findall(r"[a-zA-Z]{3,}", rubric.lower()))
        overlap = len(instruction_tokens & rubric_tokens) / max(1, len(instruction_tokens))

        rubric_lines = _split_rubric_items(rubric)
        line_count = len(rubric_lines) if rubric_lines else 1
        structure_score = min(line_count / 5.0, 1.0)

        measurable_cues = [
            "include",
            "explain",
            "accurate",
            "specific",
            "must",
            "should",
            "example",
            "evidence",
            "step",
        ]
        measurable_hits = sum(cue in rubric.lower() for cue in measurable_cues) / len(measurable_cues)

        normalized = max(0.0, min(1.0, 0.45 * overlap + 0.25 * structure_score + 0.30 * measurable_hits))
        return {
            "backend": self.backend,
            "energy": None,
            "normalized_score": normalized,
            "details": {
                "instruction_overlap": overlap,
                "structure_score": structure_score,
                "measurable_hits": measurable_hits,
            },
        }


class LocalQualityScorer:
    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        tok_path: Optional[str] = None,
        device: Optional[str] = None,
        auto_fallback: bool = True,
    ):
        self.ckpt_path = ckpt_path or str(DEFAULT_CKPT_PATH)
        self.tok_path = tok_path or str(DEFAULT_TOK_PATH)
        self.device = device
        self.auto_fallback = auto_fallback
        self._scorer = None
        self._fallback = HeuristicLocalScorer() if auto_fallback else None

    def _load(self):
        if self._scorer is not None:
            return self._scorer
        if not Path(self.ckpt_path).exists():
            raise FileNotFoundError(f"Missing local EBM checkpoint: {self.ckpt_path}")
        if not Path(self.tok_path).exists():
            raise FileNotFoundError(f"Missing local tokenizer directory: {self.tok_path}")
        self._scorer = load_ebm_scorer(
            ckpt_path=self.ckpt_path,
            tok_path=self.tok_path,
            device=self.device,
        )
        return self._scorer

    def score(self, instruction: str, rubric: str, max_length: int = 2048) -> Dict[str, object]:
        try:
            scorer = self._load()
            energy = score_one(instruction, rubric, max_length=max_length, scorer=scorer)
            normalized = 1.0 / (1.0 + math.exp(energy))
            return {
                "backend": "ebm",
                "energy": energy,
                "normalized_score": normalized,
            }
        except Exception as exc:
            if self._fallback is None:
                raise
            result = self._fallback.score(instruction, rubric)
            result["fallback_error"] = str(exc)
            return result


if __name__ == "__main__":
    q = "What is 2+2?"
    rubric = "1. State the answer. 2. Explain why the answer is correct."
    scorer = LocalQualityScorer(auto_fallback=True)
    result = scorer.score(q, rubric)
    print(result)
