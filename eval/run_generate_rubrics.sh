#!/usr/bin/env bash
# Generate pairwise rubrics using a fine-tuned SFT (rubric-generator) model.
#
# Usage:
#   bash run_generate_rubrics.sh \
#     --sft_model_path /path/to/rubric-generator \
#     --benchmark rewardbench \
#     --test_parquet /path/to/data.parquet \
#     --output_file ./rubrics/rubrics.jsonl
#
# Supported benchmarks: rewardbench | rmbench | rmb | openrubrics

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ---- Default values (override via environment or CLI flags) ----
SFT_MODEL_PATH_DEFAULT="${SFT_MODEL_PATH_DEFAULT:-}"
REWARDBENCH_PARQUET_DEFAULT="${REWARDBENCH_PARQUET_DEFAULT:-${SCRIPT_DIR}/eval_dataset/reward-bench/data/filtered-00000-of-00001.parquet}"
RMBENCH_JSON_DEFAULT="${RMBENCH_JSON_DEFAULT:-${SCRIPT_DIR}/eval_dataset/RM-Bench/total_dataset.json}"
RMB_JSON_DEFAULT="${RMB_JSON_DEFAULT:-${SCRIPT_DIR}/eval_dataset/RMB_dataset/Pairwise_set}"
OPENRUBRICS_JSONL_DEFAULT="${OPENRUBRICS_JSONL_DEFAULT:-${SCRIPT_DIR}/../OpenRubrics.jsonl}"
TRAINING_ARTIFACT_DIR_DEFAULT="${TRAINING_ARTIFACT_DIR_DEFAULT:-${SCRIPT_DIR}/../artifacts}"
LOCAL_CKPT_DEFAULT="${LOCAL_CKPT_DEFAULT:-${SCRIPT_DIR}/../ebm_qwen3-4b_gsm_model.pt}"
LOCAL_TOKENIZER_DEFAULT="${LOCAL_TOKENIZER_DEFAULT:-${SCRIPT_DIR}/../ebm_qwen3-4b_gsm_tokenizer}"

TP_DEFAULT="${TP_DEFAULT:-8}"
BATCH_SIZE_DEFAULT="${BATCH_SIZE_DEFAULT:-128}"
MAX_TOKENS_DEFAULT="${MAX_TOKENS_DEFAULT:-4096}"
GPU_MEM_UTIL_DEFAULT="${GPU_MEM_UTIL_DEFAULT:-0.9}"
SEED_DEFAULT="${SEED_DEFAULT:-42}"
OUTPUT_FILE_DEFAULT="${OUTPUT_FILE_DEFAULT:-./rubrics/rubrics.jsonl}"
SELECTION_STRATEGY_DEFAULT="${SELECTION_STRATEGY_DEFAULT:-hierarchical}"
GLOBAL_VLLM_MODEL_DEFAULT="${GLOBAL_VLLM_MODEL_DEFAULT:-}"
GLOBAL_VLLM_BASE_URL_DEFAULT="${GLOBAL_VLLM_BASE_URL_DEFAULT:-http://localhost:8000/v1}"

# ---- Usage function ----
usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Required:
  --sft_model_path PATH       Path to fine-tuned rubric generation model
  --benchmark NAME            Benchmark name: rewardbench | rmbench | rmb | openrubrics
  --output_file PATH          Output file path for generated rubrics

Benchmark-specific data files (required based on --benchmark):
  --test_parquet PATH         For rewardbench: path to test parquet file
  --rmbench_jsonl PATH        For rmbench: path to RM-Bench JSON file
  --rmb_json PATH             For rmb: path to RMB JSON file or directory
  --openrubrics_jsonl PATH    For openrubrics: path to OpenRubrics JSONL file

Optional:
  --batch_size N              Batch size (default: ${BATCH_SIZE_DEFAULT})
  --tensor_parallel_size N    Tensor parallel size (default: ${TP_DEFAULT})
  --gpu_memory_utilization F  GPU memory utilization (default: ${GPU_MEM_UTIL_DEFAULT})
  --max_tokens N              Max tokens to generate (default: ${MAX_TOKENS_DEFAULT})
  --seed N                    Random seed (default: ${SEED_DEFAULT})
  --shuffle                   Shuffle pairs before processing
  --max_samples N            Limit number of samples (0 = no limit, default: 0)
  --rmb_json_dir              Treat --rmb_json as directory (for rmb benchmark)
  --selection_strategy TYPE   first | local_ebm | hierarchical
  --training_artifact_dir PATH
  --local_ckpt PATH
  --local_tokenizer PATH
  --global_vllm_model NAME
  --global_vllm_base_url URL
  --local_weight F
  --group_weight F
  --global_weight F

Examples:
  # RewardBench
  $0 --sft_model_path /path/to/model \\
     --benchmark rewardbench \\
     --test_parquet eval_dataset/reward-bench/data/filtered-00000-of-00001.parquet \\
     --output_file ./rubrics/rubrics_rewardbench.jsonl

  # OpenRubrics
  $0 --sft_model_path /path/to/model \\
     --benchmark openrubrics \\
     --openrubrics_jsonl ../rubric_synthesis/data/rubric_generation_sampled_data.jsonl \\
     --output_file ./rubrics/rubrics_openrubrics.jsonl

  # RM-Bench
  $0 --sft_model_path /path/to/model \\
     --benchmark rmbench \\
     --rmbench_jsonl eval_dataset/RM-Bench/total_dataset.json \\
     --output_file ./rubrics/rubrics_rmbench.jsonl

  # RMB
  $0 --sft_model_path /path/to/model \\
     --benchmark rmb \\
     --rmb_json eval_dataset/RMB_dataset/Pairwise_set \\
     --output_file ./rubrics/rubrics_rmb.jsonl
EOF
}

# ---- Mutable state ----
SFT_MODEL_PATH="${SFT_MODEL_PATH_DEFAULT}"
BENCHMARK=""
TEST_PARQUET="${REWARDBENCH_PARQUET_DEFAULT}"
RMBENCH_JSON="${RMBENCH_JSON_DEFAULT}"
RMB_JSON="${RMB_JSON_DEFAULT}"
RMB_JSON_DIR="0"
OPENRUBRICS_JSONL="${OPENRUBRICS_JSONL_DEFAULT}"
OUTPUT_FILE="${OUTPUT_FILE_DEFAULT}"
TRAINING_ARTIFACT_DIR="${TRAINING_ARTIFACT_DIR_DEFAULT}"
LOCAL_CKPT="${LOCAL_CKPT_DEFAULT}"
LOCAL_TOKENIZER="${LOCAL_TOKENIZER_DEFAULT}"
SELECTION_STRATEGY="${SELECTION_STRATEGY_DEFAULT}"
GLOBAL_VLLM_MODEL="${GLOBAL_VLLM_MODEL_DEFAULT}"
GLOBAL_VLLM_BASE_URL="${GLOBAL_VLLM_BASE_URL_DEFAULT}"
LOCAL_WEIGHT="1.0"
GROUP_WEIGHT="1.0"
GLOBAL_WEIGHT="1.0"
BATCH_SIZE="${BATCH_SIZE_DEFAULT}"
TP="${TP_DEFAULT}"
MAX_TOKENS="${MAX_TOKENS_DEFAULT}"
GPU_MEM_UTIL="${GPU_MEM_UTIL_DEFAULT}"
SEED="${SEED_DEFAULT}"
SHUFFLE="0"
MAX_SAMPLES="0"

# ---- Parse command line arguments ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --sft_model_path)
      SFT_MODEL_PATH="$2"
      shift 2
      ;;
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
    --test_parquet)
      TEST_PARQUET="$2"
      shift 2
      ;;
    --rmbench_jsonl)
      RMBENCH_JSON="$2"
      shift 2
      ;;
    --rmb_json)
      RMB_JSON="$2"
      shift 2
      ;;
    --openrubrics_jsonl)
      OPENRUBRICS_JSONL="$2"
      shift 2
      ;;
    --output_file)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --tensor_parallel_size)
      TP="$2"
      shift 2
      ;;
    --gpu_memory_utilization)
      GPU_MEM_UTIL="$2"
      shift 2
      ;;
    --max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --shuffle)
      SHUFFLE="1"
      shift
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --rmb_json_dir)
      RMB_JSON_DIR="1"
      shift
      ;;
    --selection_strategy)
      SELECTION_STRATEGY="$2"
      shift 2
      ;;
    --training_artifact_dir)
      TRAINING_ARTIFACT_DIR="$2"
      shift 2
      ;;
    --local_ckpt)
      LOCAL_CKPT="$2"
      shift 2
      ;;
    --local_tokenizer)
      LOCAL_TOKENIZER="$2"
      shift 2
      ;;
    --global_vllm_model)
      GLOBAL_VLLM_MODEL="$2"
      shift 2
      ;;
    --global_vllm_base_url)
      GLOBAL_VLLM_BASE_URL="$2"
      shift 2
      ;;
    --local_weight)
      LOCAL_WEIGHT="$2"
      shift 2
      ;;
    --group_weight)
      GROUP_WEIGHT="$2"
      shift 2
      ;;
    --global_weight)
      GLOBAL_WEIGHT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# ---- Validation ----
if [[ -z "${SFT_MODEL_PATH}" ]]; then
  echo "Error: --sft_model_path is required" >&2
  usage >&2
  exit 1
fi

if [[ -z "${BENCHMARK}" ]]; then
  echo "Error: --benchmark is required" >&2
  usage >&2
  exit 1
fi

if [[ -z "${OUTPUT_FILE}" ]]; then
  echo "Error: --output_file is required" >&2
  usage >&2
  exit 1
fi

valid_benchmarks="rewardbench rmbench rmb openrubrics"
if [[ ! " ${valid_benchmarks} " =~ " ${BENCHMARK} " ]]; then
  echo "Error: unsupported benchmark: ${BENCHMARK}" >&2
  echo "Supported: ${valid_benchmarks}" >&2
  usage >&2
  exit 1
fi

# Benchmark-specific required arguments
case "${BENCHMARK}" in
  rewardbench)
    [[ -z "${TEST_PARQUET}" ]] && { echo "Error: --test_parquet is required for benchmark=rewardbench" >&2; exit 1; };;
  rmb)
    [[ -z "${RMB_JSON}" ]] && { echo "Error: --rmb_json is required for benchmark=rmb" >&2; exit 1; };;
  rmbench)
    [[ -z "${RMBENCH_JSON}" ]] && { echo "Error: --rmbench_jsonl is required for benchmark=rmbench" >&2; exit 1; };;
  openrubrics)
    [[ -z "${OPENRUBRICS_JSONL}" ]] && { echo "Error: --openrubrics_jsonl is required for benchmark=openrubrics" >&2; exit 1; };;
esac

# ---- Run ----
mkdir -p "$(dirname "${OUTPUT_FILE}")"

echo "==> Generating rubrics"
echo "    sft_model_path=${SFT_MODEL_PATH}"
echo "    benchmark=${BENCHMARK}"
echo "    output_file=${OUTPUT_FILE}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_rubrics.py"
  --sft_model_path         "${SFT_MODEL_PATH}"
  --benchmark              "${BENCHMARK}"
  --output_file            "${OUTPUT_FILE}"
  --batch_size             "${BATCH_SIZE}"
  --tensor_parallel_size   "${TP}"
  --gpu_memory_utilization "${GPU_MEM_UTIL}"
  --max_tokens             "${MAX_TOKENS}"
  --seed                   "${SEED}"
  --selection_strategy     "${SELECTION_STRATEGY}"
  --openrubrics_input_file "${OPENRUBRICS_JSONL}"
  --training_artifact_dir  "${TRAINING_ARTIFACT_DIR}"
  --local_ckpt             "${LOCAL_CKPT}"
  --local_tokenizer        "${LOCAL_TOKENIZER}"
  --global_vllm_model      "${GLOBAL_VLLM_MODEL}"
  --global_vllm_base_url   "${GLOBAL_VLLM_BASE_URL}"
  --local_weight           "${LOCAL_WEIGHT}"
  --group_weight           "${GROUP_WEIGHT}"
  --global_weight          "${GLOBAL_WEIGHT}"
)

case "${BENCHMARK}" in
  rewardbench)
    echo "    test_parquet=${TEST_PARQUET}"
    cmd+=(--test_parquet "${TEST_PARQUET}");;
  rmb)
    echo "    rmb_json=${RMB_JSON}"
    cmd+=(--rmb_json "${RMB_JSON}")
    [[ "${RMB_JSON_DIR}" == "1" ]] && cmd+=(--rmb_json_dir);;
  rmbench)
    echo "    rmbench_jsonl=${RMBENCH_JSON}"
    cmd+=(--rmbench_jsonl "${RMBENCH_JSON}");;
  openrubrics)
    echo "    openrubrics_jsonl=${OPENRUBRICS_JSONL}"
    cmd+=(--openrubrics_jsonl "${OPENRUBRICS_JSONL}");;
esac

[[ "${SHUFFLE}" == "1" ]] && cmd+=(--shuffle_pairs)
[[ "${MAX_SAMPLES}" != "0" ]] && cmd+=(--max_samples "${MAX_SAMPLES}")

"${cmd[@]}"

echo ""
echo "Done. Rubrics saved to: ${OUTPUT_FILE}"
