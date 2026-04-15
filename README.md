# Judge-Rubric: EnergyORM

EnergyORM is the rubric selection and scoring component of Judge-Rubric. It generates candidate rubrics for pairwise response evaluation, selects the best rubric with local or hierarchical scorers, and evaluates rubric-guided judges on reward-model benchmarks.

The core idea is:

1. Generate multiple rubrics for the same instruction and response pair.
2. Score/select a rubric with an Energy-Based Model (EBM), heuristic/group scorers, or a hierarchical mixture.
3. Use the selected rubric to judge which response is better.

Lower EBM energy means a better rubric candidate.

## Repository Layout

```text
Judge-Rubrics/
  gen_data.py                         # Generate EBM training JSONL from pairwise examples
  dataset.py                          # Grouped question/rubric dataset and collate function
  ebm_model.py                        # Scratch Transformer EBM backbone
  step1_train_ebm.py                  # Original scratch EBM training script
  step1_train_ebm_v3_pretrained.py    # Pretrained HF backbone EBM training script
  score.py                            # Local EBM scorer and heuristic fallback
  group_scorer.py                     # Group/domain-level rubric scorer artifacts
  global_scorer.py                    # Global heuristic or vLLM-based rubric scorer
  hierarchical_scorer.py              # Weighted local + group + global scorer
  rubric_selection.py                 # first/local_ebm/hierarchical rubric selection API
  prepare_hierarchical_data.py        # Build artifacts for hierarchical scoring
  run_hierarchical_demo.py            # Smoke test for hierarchical scoring
  eval/
    generate_rubrics.py               # Generate and select rubrics for benchmarks
    evaluate.py                       # Evaluate direct_judge or rubric_judge
    run_generate_rubrics.sh           # Shell wrapper for rubric generation
    run_eval.sh                       # Shell wrapper for evaluation
    eval_dataset/                     # Local benchmark data folders
```

## Setup

Use a Python environment with PyTorch and Hugging Face Transformers. The project also uses `datasets`, `openai`, `tqdm`, `numpy`, `pandas`/`pyarrow` for benchmark loading, and optionally `vllm` for local model serving.

```bash
pip install torch transformers datasets openai tqdm numpy pandas pyarrow scikit-learn
pip install vllm
```

If you use the repo-level `requirements.txt`, note that it is an exported environment file and may contain machine-specific packages.

## Training Data Format

The EBM training scripts expect a JSONL file where each line is one rubric candidate:

```json
{
  "question_id": 0,
  "question": "User instruction or pairwise evaluation prompt",
  "response_a": "Candidate response A",
  "response_b": "Candidate response B",
  "answer": 0,
  "gen_text": "1. Check correctness...\n2. Check completeness...",
  "source": "optional-domain",
  "label": 1
}
```

Records are grouped by `question`. A useful training group must contain at least one `label=1` rubric and at least one `label=0` rubric.

Important: `label=1` means that using this generated rubric caused the judge to predict the dataset's winner. It is not a direct human annotation that the rubric is intrinsically better in every way.

## Generate EBM Training Data

Start an OpenAI-compatible vLLM server first:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve /mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-4B/Qwen3-4B  --tensor-parallel-size 4 > 0402test.log 2>&1 &
```

Then generate rubric candidates and labels:

```bash
python Judge-Rubrics/gen_data.py \
  --model /mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-4B/Qwen3-4B \
  --base-url http://localhost:8000/v1 \
  --output-file Judge-Rubrics/artifacts/ebm_train.jsonl \
  --n-gen 4 \
  --workers 8 \
  --resume
```

## Train an EBM Scorer

The recommended path is `step1_train_ebm_v3_pretrained.py`, using the same tokenizer and backbone.

Full finetuning with a small learning rate:

```bash
nohup python step1_train_ebm_v3_pretrained_1.py \
  --backbone /mnt/shared-storage-user/ma4tool-shared/hug_ckpts/deberta-v3-base \
  --tok /mnt/shared-storage-user/ma4tool-shared/hug_ckpts/deberta-v3-base \
  --max_length 512 \
  --bsz 1 \
  --epochs 10 \
  --lr 5e-6 \
  --weight_decay 0.01 \
  --dropout 0.0 \
  --pooling mean \
  --fp16 \
  --select_metric pairwise_acc \
  --early_stop_patience 5 \
  --min_delta 0.0 \
  --save_prefix ebm_v3_deberta_base_unfrozen_mean_lr5e6 > 0413train_v3_deberta_base_unfrozen_mean_lr5e6.log 2>&1 &
```

The script reports:

- `Top1 Acc`: whether the lowest-energy rubric in a group has `label=1`.
- `Pairwise Acc`: across all positive-negative rubric pairs, whether `E(pos) < E(neg)`.
- `Random Expected`: expected top-1 accuracy from uniformly random candidate selection.
- `First Acc`: accuracy from always selecting the first candidate.
- `Mean Gap(E_neg-E_pos)`: positive values mean negatives receive higher energy than positives on average.

For EBM training, prefer `Pairwise Acc` over `Top1 Acc`, especially when groups contain many positive candidates.

## Build Hierarchical Scoring Artifacts

Hierarchical selection combines:

- local scorer: EBM or heuristic rubric quality score
- group scorer: TF-IDF/domain centroid style scoring
- global scorer: heuristic or vLLM-based global judgement

Prepare artifacts:

```bash
python Judge-Rubrics/prepare_hierarchical_data.py \
  --input-file Judge-Rubrics/OpenRubrics.jsonl \
  --artifact-dir Judge-Rubrics/artifacts \
  --overwrite
```

Smoke test:

```bash
python Judge-Rubrics/run_hierarchical_demo.py \
  --input-file Judge-Rubrics/OpenRubrics.jsonl \
  --artifact-dir Judge-Rubrics/artifacts \
  --local-ckpt Judge-Rubrics/ebm_v3_deberta_base_unfrozen_mean_model.pt \
  --local-tokenizer Judge-Rubrics/ebm_v3_deberta_base_unfrozen_mean_tokenizer
```

## Generate Rubrics for Evaluation & Evaluate a Judge

Local_EBM:
```
nohup python generate_rubrics.py \
  --sft_model_path /mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-4B/Qwen3-4B \
  --benchmark rmb \
  --rmb_json /mnt/shared-storage-user/yuzhiyin/Judge_Rubrics/eval/eval_dataset/RMB_dataset/Pairwise_set \
  --rmb_json_dir \
  --output_file ./rubrics/rubrics_bestofn_local_rmb.jsonl \
  --num_candidates 4 \
  --selection_strategy local_ebm \
  --local_ckpt /mnt/shared-storage-user/yuzhiyin/EnergyORM/ebm_v3_deberta_base_unfrozen_mean_lr2e6_dropout01_model.pt \
  --local_tokenizer /mnt/shared-storage-user/yuzhiyin/EnergyORM/ebm_v3_deberta_base_unfrozen_mean_lr2e6_dropout01_tokenizer \
  --tensor_parallel_size 2 \
  --batch_size 256 > generate_rubric_bestofn_local_rmb.log 2>&1 &
```

```
bash run_eval.sh \
  --model_path /mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-4B/Qwen3-4B \
  --benchmark rewardbench \
  --prompt_type rubric_judge \
  --rubrics_file /mnt/shared-storage-user/yuzhiyin/Judge_Rubrics/eval/rubrics/rubrics_bestofn_local_global_rewardbench.jsonl \
  --tensor_parallel_size 2 \
  --batch_size 256 \
  --gpu_memory_utilization 0.95 \
  --max_tokens 8192 \
  --seed 42 \
  --output_root /mnt/shared-storage-user/yuzhiyin/Judge_Rubrics/output/output_bestofn_local_global_judge_rewardbench \
  --shuffle 
```

DIRECT:
```
bash eval/run_eval.sh \
  --model_path /mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-4B/Qwen3-4B \
  --benchmark rewardbench \
  --prompt_type direct_judge \
  --tensor_parallel_size 2 \
  --batch_size 256 \
  --gpu_memory_utilization 0.95 \
  --max_tokens 8192 \
  --seed 42 \
  --output_root ./output/output_direct_judge \
  --shuffle 
```

Rubric Judge:
```
nohup python eval/generate_rubrics.py \
  --sft_model_path /mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-4B/Qwen3-4B \
  --benchmark rmb \
  --rmb_json /mnt/shared-storage-user/yuzhiyin/Judge_Rubrics/eval/eval_dataset/RMB_dataset/Pairwise_set \
  --rmb_json_dir \
  --output_file ./rubrics/rubrics_rmb_naive_.jsonl \
  --num_candidates 1 \
  --selection_strategy first \
  --tensor_parallel_size 2 \
  --batch_size 256 > generate_rubric_naive.log 2>&1 &
```

```
bash eval/run_eval.sh \
  --model_path /mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-4B/Qwen3-4B \
  --benchmark rmb \
  --rmb_json /mnt/shared-storage-user/yuzhiyin/Judge_Rubrics/eval/eval_dataset/RMB_dataset/Pairwise_set \
  --rmb_json_dir \
  --prompt_type rubric_judge \
  --rubrics_file ./rubrics/rubrics_rmb_naive_.jsonl \
  --tensor_parallel_size 2 \
  --batch_size 256 \
  --gpu_memory_utilization 0.95 \
  --max_tokens 8192 \
  --seed 42 \
  --output_root /mnt/shared-storage-user/yuzhiyin/Judge_Rubrics/output/output_rubric_judge_rmb \
  --shuffle 
```


Supported evaluation benchmarks:

- `rewardbench`
- `rmbench`
- `rmb`

## Practical Notes

- Keep tokenizer and pretrained backbone aligned, for example DeBERTa-v3 tokenizer with DeBERTa-v3 backbone.
- For EBM training, monitor `Pairwise Acc` first. `Top1 Acc` can be misleading when a group has more positive than negative rubrics.
- If full finetuning quickly drives train pairwise accuracy high but validation stays near 55-57%, the bottleneck is likely label noise or weak rubric-quality signal, not just model capacity.
- Large checkpoints, generated rubrics, and benchmark outputs should usually live under `Judge-Rubrics/artifacts/` or `Judge-Rubrics/eval/eval_results/` and should not be committed unless intentionally released.

## Citation

If you use this repository, please cite the Judge-Rubric project or the associated paper when available.
