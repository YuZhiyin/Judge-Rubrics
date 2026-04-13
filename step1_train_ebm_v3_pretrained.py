import argparse
import copy
import json
import os
import random
import sys
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, get_cosine_schedule_with_warmup

if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EnergyORM.dataset import TrainValChunkDS, collate_fn as original_collate_fn
from EnergyORM.utils import (
    get_device_and_amp_helpers,
    load_q2cands_from_jsonl,
    setup_tokenizer,
)


script_directory = os.path.dirname(os.path.abspath(__file__))
default_data_path = "/mnt/shared-storage-user/yuzhiyin/Judge-Rubrics/output-test_sorted.jsonl"


class PretrainedEnergyScorer(nn.Module):
    """Energy scorer backed by a pretrained HF model plus a small ranking head."""

    def __init__(
        self,
        backbone_path,
        tokenizer_size,
        pad_token_id=None,
        dropout=0.2,
        freeze_backbone=False,
        pooling="cls_mean",
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_path)
        self.backbone.resize_token_embeddings(tokenizer_size)
        self.pooling = pooling
        self.freeze_backbone = freeze_backbone

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.config, "n_embd")
        if hidden_size is None:
            raise ValueError(f"Could not infer hidden size for backbone: {backbone_path}")

        if pad_token_id is not None:
            self.backbone.config.pad_token_id = pad_token_id

        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        pooled_size = hidden_size * 2 if pooling == "cls_mean" else hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_size),
            nn.Linear(pooled_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        print(
            "Initialized PretrainedEnergyScorer: "
            f"backbone={backbone_path}, hidden={hidden_size}, dropout={dropout}, "
            f"freeze_backbone={freeze_backbone}, pooling={pooling}, "
            f"gradient_checkpointing={gradient_checkpointing}"
        )

    def forward(self, ids, mask):
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                out = self.backbone(input_ids=ids, attention_mask=mask)
        else:
            out = self.backbone(input_ids=ids, attention_mask=mask)
        hidden = out.last_hidden_state
        cls_rep = hidden[:, 0]
        denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(hidden.dtype)
        mean_rep = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / denom
        if self.pooling == "cls":
            pooled = cls_rep
        elif self.pooling == "mean":
            pooled = mean_rep
        elif self.pooling == "cls_mean":
            pooled = torch.cat([cls_rep, mean_rep], dim=-1)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")
        return self.head(pooled).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_backbone_max_length(backbone_path, requested_max_length):
    cfg = AutoConfig.from_pretrained(backbone_path)
    limits = [
        getattr(cfg, "max_position_embeddings", None),
        getattr(cfg, "n_positions", None),
    ]
    limits = [int(x) for x in limits if x is not None and int(x) > 0]
    if not limits:
        return requested_max_length
    model_limit = min(limits)
    if requested_max_length > model_limit:
        print(
            f"Warning: requested max_length={requested_max_length}, but backbone position limit is {model_limit}. "
            f"Using max_length={model_limit} for this pretrained run."
        )
        return model_limit
    return requested_max_length


def bt_loss(e, lab, margin=0.0):
    pos = e[lab == 1]
    neg = e[lab == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    return F.softplus(pos.unsqueeze(1) - neg.unsqueeze(0) + margin).mean()


@torch.no_grad()
def evaluate(model, loader, device, autocaster, eval_type="Validation"):
    model.eval()
    groups = 0
    top1_correct = 0
    first_correct = 0
    pair_correct = 0
    total_pairs = 0
    random_expected = 0.0
    gap_sum = 0.0
    gap_groups = 0

    pbar = tqdm(loader, desc=f"Evaluating ({eval_type})", leave=False, unit="batch")
    for idsL, maskL, labL in pbar:
        for ids, mask, lab in zip(idsL, maskL, labL):
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            if ids.numel() == 0 or lab.numel() == 0:
                continue
            with autocaster():
                e = model(ids, mask)
            if e.numel() == 0:
                continue

            top1_correct += int(lab[torch.argmin(e)].item() == 1)
            first_correct += int(lab[0].item() == 1)
            random_expected += float(lab.float().mean().item())
            groups += 1

            pos_e = e[lab == 1]
            neg_e = e[lab == 0]
            if pos_e.numel() and neg_e.numel():
                pair_matrix = pos_e.unsqueeze(1) < neg_e.unsqueeze(0)
                pair_correct += int(pair_matrix.sum().item())
                total_pairs += int(pair_matrix.numel())
                gap_sum += float((neg_e.mean() - pos_e.mean()).item())
                gap_groups += 1

            pbar.set_postfix(
                top1=f"{100.0 * top1_correct / max(1, groups):.2f}%",
                pair=f"{100.0 * pair_correct / max(1, total_pairs):.2f}%",
            )

    return {
        "top1_acc": 100.0 * top1_correct / max(1, groups),
        "pairwise_acc": 100.0 * pair_correct / max(1, total_pairs),
        "first_acc": 100.0 * first_correct / max(1, groups),
        "random_expected_acc": 100.0 * random_expected / max(1, groups),
        "mean_energy_gap": gap_sum / max(1, gap_groups),
        "groups": groups,
        "pairs": total_pairs,
    }


def print_metrics(prefix, m):
    print(
        f"{prefix} → Top1 Acc: {m['top1_acc']:.2f}% | "
        f"Pairwise Acc: {m['pairwise_acc']:.2f}% | "
        f"Random Expected: {m['random_expected_acc']:.2f}% | "
        f"First Acc: {m['first_acc']:.2f}% | "
        f"Mean Gap(E_neg-E_pos): {m['mean_energy_gap']:.4f}"
    )


def maybe_subset_dataset(ds, max_groups, seed):
    if max_groups <= 0 or max_groups >= len(ds):
        return ds
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    return Subset(ds, indices[:max_groups])


def main(args):
    set_seed(args.seed)
    DEV, use_amp, autocaster, scaler = get_device_and_amp_helpers(args.device, args.fp16)
    effective_max_length = infer_backbone_max_length(args.backbone, args.max_length)
    tok, PAD_ID, CLS_ID = setup_tokenizer(args.tok or args.backbone, effective_max_length)
    if PAD_ID == CLS_ID:
        raise ValueError(f"PAD_ID and CLS_ID are both {PAD_ID}; CLS would be masked as padding.")

    print("\n--- Loading Training Data ---")
    q2cands = load_q2cands_from_jsonl(args.train_llama_gsm_data, "Llama GSM training/validation")
    if not q2cands:
        raise RuntimeError(f"Could not load training data from {args.train_llama_gsm_data}")

    print("\n--- Initializing Datasets ---")
    train_ds = TrainValChunkDS(
        tok,
        effective_max_length,
        CLS_ID,
        PAD_ID,
        q2cands_data=copy.deepcopy(q2cands),
        split="train",
        holdout=args.val_holdout,
        dataset_name_log_prefix="llama_gsm_",
    )
    val_ds = TrainValChunkDS(
        tok,
        effective_max_length,
        CLS_ID,
        PAD_ID,
        q2cands_data=copy.deepcopy(q2cands),
        split="val",
        holdout=args.val_holdout,
        dataset_name_log_prefix="llama_gsm_",
    )

    collate = partial(original_collate_fn, pad_id=PAD_ID)
    pin_mem = DEV.type == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=args.bsz,
        shuffle=True,
        collate_fn=collate,
        pin_memory=pin_mem,
        num_workers=args.num_workers,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.bsz,
        shuffle=False,
        collate_fn=collate,
        pin_memory=pin_mem,
        num_workers=args.num_workers,
    )
    train_eval_ds = maybe_subset_dataset(train_ds, args.train_eval_groups, args.seed + 7)
    train_eval_dl = DataLoader(
        train_eval_ds,
        batch_size=args.bsz,
        shuffle=False,
        collate_fn=collate,
        pin_memory=pin_mem,
        num_workers=args.num_workers,
    )

    print("\n--- Model Setup ---")
    model = PretrainedEnergyScorer(
        args.backbone,
        tokenizer_size=len(tok),
        pad_token_id=PAD_ID,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        pooling=args.pooling,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(DEV)

    opt = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    print("\n--- Starting Training (pretrained EBM) ---")
    print(
        f"Backbone: {args.backbone}, tokenizer: {args.tok or args.backbone}, "
        f"max_len={effective_max_length}, freeze_backbone={args.freeze_backbone}, "
        f"pooling={args.pooling}"
    )
    print(f"CLS_ID={CLS_ID}, PAD_ID={PAD_ID}, train={len(train_ds)}, val={len(val_ds)}")
    print(f"LR={args.lr}, dropout={args.dropout}, margin={args.margin}, select_metric={args.select_metric}")

    best_metric = -1.0
    best_state = None
    best_metrics = None
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        pbar = tqdm(train_dl, desc=f"Epoch {ep}/{args.epochs} Training", unit="batch")
        for idsL, maskL, labL in pbar:
            opt.zero_grad(set_to_none=True)
            losses = []
            for ids, mask, lab in zip(idsL, maskL, labL):
                ids, mask, lab = ids.to(DEV), mask.to(DEV), lab.to(DEV)
                with autocaster():
                    e = model(ids, mask)
                    group_loss = bt_loss(e, lab, margin=args.margin)
                if group_loss is not None and torch.isfinite(group_loss):
                    losses.append(group_loss)
            if not losses:
                continue

            loss = torch.stack(losses).mean()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()

            sched.step()
            total_loss += loss.item()
            batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        print(f"Epoch {ep} finished. Average Training Loss: {total_loss / max(1, batches):.4f}")

        if ep % args.validate_every == 0:
            if args.train_eval_groups > 0:
                train_metrics = evaluate(model, train_eval_dl, DEV, autocaster, eval_type="TrainSample")
                print_metrics(f"Epoch {ep} TrainSample", train_metrics)
            val_metrics = evaluate(model, val_dl, DEV, autocaster, eval_type="Validation")
            print_metrics(f"Epoch {ep} Validation", val_metrics)

            metric = val_metrics[args.select_metric]
            if metric > best_metric + args.min_delta:
                best_metric = metric
                best_metrics = val_metrics
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
                print(f"New best {args.select_metric}: {best_metric:.2f}. State saved.")
            else:
                bad_epochs += 1
                print(f"(Current best {args.select_metric}: {best_metric:.2f}; bad_epochs={bad_epochs})")
                if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
                    print(f"Early stopping after {bad_epochs} validation checks without improvement.")
                    break

    if best_state is None:
        print("No best state was saved.")
        return

    model_path = f"{args.save_prefix}_model.pt"
    tok_path = f"{args.save_prefix}_tokenizer"
    cfg_path = f"{args.save_prefix}_config.json"
    print(f"Saving best model to {model_path}")
    torch.save(best_state, model_path)
    os.makedirs(tok_path, exist_ok=True)
    tok.save_pretrained(tok_path)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "arch": "PretrainedEnergyScorer",
                "backbone": args.backbone,
                "max_length": effective_max_length,
                "dropout": args.dropout,
                "freeze_backbone": args.freeze_backbone,
                "pooling": args.pooling,
                "gradient_checkpointing": args.gradient_checkpointing,
                "cls_id": CLS_ID,
                "pad_id": PAD_ID,
                "select_metric": args.select_metric,
                "best_metrics": best_metrics,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )
    print(f"Saved tokenizer to {tok_path}/")
    print(f"Saved config to {cfg_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train EBM with a pretrained HF backbone.")
    p.add_argument("--train_llama_gsm_data", default=default_data_path)
    p.add_argument("--backbone", default="gpt2", help="HF model path/name used by AutoModel.")
    p.add_argument("--tok", default=None, help="Tokenizer path/name. Defaults to --backbone.")
    p.add_argument("--val_holdout", type=float, default=0.2)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--pooling", choices=["cls_mean", "mean", "cls"], default="cls_mean")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bsz", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--validate_every", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--select_metric", choices=["top1_acc", "pairwise_acc"], default="pairwise_acc")
    p.add_argument("--train_eval_groups", type=int, default=512)
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=0.05)
    p.add_argument("--save_prefix", default="ebm_pretrained_gsm")
    main(p.parse_args())
