#!/usr/bin/env bash
# Run pairwise evaluation on reward benchmarks.
#
# Usage:
#   bash run_eval.sh --benchmark rewardbench --prompt_type direct_judge \
#                    --model_path /path/to/model --test_parquet /path/to/data.parquet
#
# Supported benchmarks:  rewardbench | rmbench | rmb
# Supported prompt types: direct_judge | rubric_judge

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ---- Default values (override via environment or CLI flags) ----
MODEL_PATH_DEFAULT="${MODEL_PATH_DEFAULT:-}"
MODEL_BASE_URL_DEFAULT="${MODEL_BASE_URL_DEFAULT:-}"
MODEL_API_KEY_DEFAULT="${MODEL_API_KEY_DEFAULT:-EMPTY}"
REWARDBENCH_PARQUET_DEFAULT="${REWARDBENCH_PARQUET_DEFAULT:-${SCRIPT_DIR}/eval_dataset/reward-bench/data/filtered-00000-of-00001.parquet}"
RMBENCH_JSON_DEFAULT="${RMBENCH_JSON_DEFAULT:-${SCRIPT_DIR}/eval_dataset/RM-Bench/total_dataset.json}"
RMB_JSON_DEFAULT="${RMB_JSON_DEFAULT:-${SCRIPT_DIR}/eval_dataset/RMB_dataset/Pairwise_set}"
OPENRUBRICS_JSONL_DEFAULT="${OPENRUBRICS_JSONL_DEFAULT:-${SCRIPT_DIR}/../OpenRubrics.jsonl}"
TRAINING_ARTIFACT_DIR_DEFAULT="${TRAINING_ARTIFACT_DIR_DEFAULT:-${SCRIPT_DIR}/../artifacts}"
LOCAL_CKPT_DEFAULT="${LOCAL_CKPT_DEFAULT:-${SCRIPT_DIR}/../ebm_qwen3-4b_gsm_model.pt}"
LOCAL_TOKENIZER_DEFAULT="${LOCAL_TOKENIZER_DEFAULT:-${SCRIPT_DIR}/../ebm_qwen3-4b_gsm_tokenizer}"

TP_DEFAULT="${TP_DEFAULT:-8}"
BATCH_SIZE_DEFAULT="${BATCH_SIZE_DEFAULT:-128}"
MAX_TOKENS_DEFAULT="${MAX_TOKENS_DEFAULT:-8192}"
GPU_MEM_UTIL_DEFAULT="${GPU_MEM_UTIL_DEFAULT:-0.9}"
SEED_DEFAULT="${SEED_DEFAULT:-42}"
OUTPUT_ROOT_DEFAULT="${OUTPUT_ROOT_DEFAULT:-./eval_results}"

# ---- Usage function ----
usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Required:
  --model_path PATH           Path to judge model
  --benchmark NAME            Benchmark name: rewardbench | rmbench | rmb
  --prompt_type TYPE          Prompt type: direct_judge | rubric_judge

Benchmark-specific data files (required based on --benchmark):
  --test_parquet PATH         For rewardbench: path to test parquet file
  --rmbench_jsonl PATH        For rmbench: path to RM-Bench JSON file
  --rmb_json PATH             For rmb: path to RMB JSON file or directory

Conditional:
  --rubrics_file PATH         Required when --prompt_type=rubric_judge
  --generate_rubrics_on_the_fly
                              Generate rubric candidates during evaluation and
                              pick the best rubric with the configured strategy.

Optional:
  --output_root PATH          Output root directory (default: ./eval_results)
  --model_base_url URL        OpenAI-compatible URL for judge inference
  --model_api_key KEY         API key for judge URL mode
  --batch_size N              Batch size (default: ${BATCH_SIZE_DEFAULT})
  --tensor_parallel_size N     Tensor parallel size (default: ${TP_DEFAULT})
  --gpu_memory_utilization F  GPU memory utilization (default: ${GPU_MEM_UTIL_DEFAULT})
  --max_tokens N              Max tokens to generate (default: ${MAX_TOKENS_DEFAULT})
  --seed N                    Random seed (default: ${SEED_DEFAULT})
  --shuffle                   Shuffle pairs before processing
  --max_samples N             Limit number of samples (0 = no limit, default: 0)
  --rmb_json_dir              Treat --rmb_json as directory (for rmb benchmark)
  --rubric_generator_model_path PATH
  --rubric_generator_base_url URL
  --rubric_generator_api_key KEY
  --rubric_num_candidates N
  --rubric_generation_temperature F
  --rubric_generation_max_tokens N
  --rubric_selection_strategy TYPE
                              first | local_ebm | hierarchical
  --openrubrics_input_file PATH
  --training_artifact_dir PATH
  --local_ckpt PATH
  --local_tokenizer PATH
  --global_vllm_model NAME
  --global_vllm_base_url URL
  --hierarchical_local_weight F
  --hierarchical_group_weight F
  --hierarchical_global_weight F

Examples:
  # Direct judge on RewardBench
  $0 --model_path /path/to/model \\
     --benchmark rewardbench \\
     --prompt_type direct_judge \\
     --test_parquet eval_dataset/reward-bench/data/filtered-00000-of-00001.parquet

  # Rubric judge on RM-Bench
  $0 --model_path /path/to/model \\
     --benchmark rmbench \\
     --prompt_type rubric_judge \\
     --rmbench_jsonl eval_dataset/RM-Bench/total_dataset.json \\
     --rubrics_file ./rubrics/rubrics_rmbench.jsonl

  # Rubric judge on RMB
  $0 --model_path /path/to/model \\
     --benchmark rmb \\
     --prompt_type rubric_judge \\
     --rmb_json eval_dataset/RMB_dataset/Pairwise_set \\
     --rubrics_file ./rubrics/rubrics_rmb.jsonl \\
     --rmb_json_dir
EOF
}

# ---- Mutable state ----
BENCHMARK=""
PROMPT_TYPE=""
MODEL_PATH="${MODEL_PATH_DEFAULT}"
MODEL_BASE_URL="${MODEL_BASE_URL_DEFAULT}"
MODEL_API_KEY="${MODEL_API_KEY_DEFAULT}"
RUBRICS_FILE=""
TEST_PARQUET="${REWARDBENCH_PARQUET_DEFAULT}"
RMBENCH_JSON="${RMBENCH_JSON_DEFAULT}"
RMB_JSON="${RMB_JSON_DEFAULT}"
RMB_JSON_DIR="0"
OPENRUBRICS_INPUT_FILE="${OPENRUBRICS_JSONL_DEFAULT}"
TRAINING_ARTIFACT_DIR="${TRAINING_ARTIFACT_DIR_DEFAULT}"
LOCAL_CKPT="${LOCAL_CKPT_DEFAULT}"
LOCAL_TOKENIZER="${LOCAL_TOKENIZER_DEFAULT}"
GLOBAL_VLLM_MODEL=""
GLOBAL_VLLM_BASE_URL="http://localhost:8000/v1"
GENERATE_RUBRICS_ON_THE_FLY="0"
RUBRIC_GENERATOR_MODEL_PATH=""
RUBRIC_GENERATOR_BASE_URL=""
RUBRIC_GENERATOR_API_KEY="EMPTY"
RUBRIC_NUM_CANDIDATES="4"
RUBRIC_GENERATION_TEMPERATURE="0.7"
RUBRIC_GENERATION_MAX_TOKENS="4096"
RUBRIC_SELECTION_STRATEGY="hierarchical"
HIERARCHICAL_LOCAL_WEIGHT="1.0"
HIERARCHICAL_GROUP_WEIGHT="1.0"
HIERARCHICAL_GLOBAL_WEIGHT="1.0"
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT}"
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
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
    --model_base_url)
      MODEL_BASE_URL="$2"
      shift 2
      ;;
    --model_api_key)
      MODEL_API_KEY="$2"
      shift 2
      ;;
    --prompt_type)
      PROMPT_TYPE="$2"
      shift 2
      ;;
    --rubrics_file)
      RUBRICS_FILE="$2"
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
    --output_root)
      OUTPUT_ROOT="$2"
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
    --generate_rubrics_on_the_fly)
      GENERATE_RUBRICS_ON_THE_FLY="1"
      shift
      ;;
    --rubric_generator_model_path)
      RUBRIC_GENERATOR_MODEL_PATH="$2"
      shift 2
      ;;
    --rubric_generator_base_url)
      RUBRIC_GENERATOR_BASE_URL="$2"
      shift 2
      ;;
    --rubric_generator_api_key)
      RUBRIC_GENERATOR_API_KEY="$2"
      shift 2
      ;;
    --rubric_num_candidates)
      RUBRIC_NUM_CANDIDATES="$2"
      shift 2
      ;;
    --rubric_generation_temperature)
      RUBRIC_GENERATION_TEMPERATURE="$2"
      shift 2
      ;;
    --rubric_generation_max_tokens)
      RUBRIC_GENERATION_MAX_TOKENS="$2"
      shift 2
      ;;
    --rubric_selection_strategy)
      RUBRIC_SELECTION_STRATEGY="$2"
      shift 2
      ;;
    --openrubrics_input_file)
      OPENRUBRICS_INPUT_FILE="$2"
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
    --hierarchical_local_weight)
      HIERARCHICAL_LOCAL_WEIGHT="$2"
      shift 2
      ;;
    --hierarchical_group_weight)
      HIERARCHICAL_GROUP_WEIGHT="$2"
      shift 2
      ;;
    --hierarchical_global_weight)
      HIERARCHICAL_GLOBAL_WEIGHT="$2"
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
if [[ -z "${MODEL_PATH}" ]]; then
  echo "Error: --model_path is required" >&2
  usage >&2
  exit 1
fi

if [[ -z "${BENCHMARK}" ]]; then
  echo "Error: --benchmark is required" >&2
  usage >&2
  exit 1
fi

if [[ -z "${PROMPT_TYPE}" ]]; then
  echo "Error: --prompt_type is required" >&2
  usage >&2
  exit 1
fi

valid_benchmarks="rewardbench rmbench rmb"
if [[ ! " ${valid_benchmarks} " =~ " ${BENCHMARK} " ]]; then
  echo "Error: unsupported benchmark: ${BENCHMARK}" >&2
  echo "Supported: ${valid_benchmarks}" >&2
  usage >&2
  exit 1
fi

valid_prompt_types="direct_judge rubric_judge"
if [[ ! " ${valid_prompt_types} " =~ " ${PROMPT_TYPE} " ]]; then
  echo "Error: unsupported prompt_type: ${PROMPT_TYPE}" >&2
  echo "Supported: ${valid_prompt_types}" >&2
  usage >&2
  exit 1
fi

if [[ "${PROMPT_TYPE}" == "rubric_judge" && -z "${RUBRICS_FILE}" && "${GENERATE_RUBRICS_ON_THE_FLY}" != "1" ]]; then
  echo "Error: --rubrics_file is required for prompt_type=rubric_judge" >&2
  echo "Hint: run generate_rubrics.py first or enable --generate_rubrics_on_the_fly" >&2
  usage >&2
  exit 1
fi

# ---- Run evaluation ----
OUT_DIR="${OUTPUT_ROOT}/${BENCHMARK}/${PROMPT_TYPE}"
mkdir -p "${OUT_DIR}"

echo "==> Running evaluation"
echo "    benchmark=${BENCHMARK}  prompt_type=${PROMPT_TYPE}"
echo "    model_path=${MODEL_PATH}"
[[ -n "${MODEL_BASE_URL}" ]] && echo "    model_base_url=${MODEL_BASE_URL}"
[[ -n "${RUBRICS_FILE}" ]] && echo "    rubrics_file=${RUBRICS_FILE}"
echo "    output_dir=${OUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate.py"
  --benchmark    "${BENCHMARK}"
  --prompt_type  "${PROMPT_TYPE}"
  --model_path   "${MODEL_PATH}"
  --model_base_url "${MODEL_BASE_URL}"
  --model_api_key "${MODEL_API_KEY}"
  --output_dir   "${OUT_DIR}"
  --batch_size   "${BATCH_SIZE}"
  --tensor_parallel_size "${TP}"
  --gpu_memory_utilization "${GPU_MEM_UTIL}"
  --max_tokens   "${MAX_TOKENS}"
  --seed         "${SEED}"
)

# Benchmark-specific data path
if [[ "${BENCHMARK}" == "rewardbench" ]]; then
  [[ -z "${TEST_PARQUET}" ]] && { echo "Error: --test_parquet is required for benchmark=rewardbench" >&2; exit 1; }
  echo "    test_parquet=${TEST_PARQUET}"
  cmd+=(--test_parquet "${TEST_PARQUET}")
elif [[ "${BENCHMARK}" == "rmb" ]]; then
  [[ -z "${RMB_JSON}" ]] && { echo "Error: --rmb_json is required for benchmark=rmb" >&2; exit 1; }
  echo "    rmb_json=${RMB_JSON}"
  cmd+=(--rmb_json "${RMB_JSON}")
  [[ "${RMB_JSON_DIR}" == "1" ]] && cmd+=(--rmb_json_dir)
else  # rmbench
  [[ -z "${RMBENCH_JSON}" ]] && { echo "Error: --rmbench_jsonl is required for benchmark=rmbench" >&2; exit 1; }
  echo "    rmbench_jsonl=${RMBENCH_JSON}"
  cmd+=(--rmbench_jsonl "${RMBENCH_JSON}")
fi

# Optional flags
[[ "${SHUFFLE}" == "1" ]] && cmd+=(--shuffle_pairs)
[[ "${MAX_SAMPLES}" != "0" ]] && cmd+=(--max_samples "${MAX_SAMPLES}")
[[ "${PROMPT_TYPE}" == "rubric_judge" && -n "${RUBRICS_FILE}" ]] && cmd+=(--rubrics_file "${RUBRICS_FILE}")
if [[ "${GENERATE_RUBRICS_ON_THE_FLY}" == "1" ]]; then
  cmd+=(
    --generate_rubrics_on_the_fly
    --rubric_generator_model_path "${RUBRIC_GENERATOR_MODEL_PATH}"
    --rubric_generator_base_url "${RUBRIC_GENERATOR_BASE_URL}"
    --rubric_generator_api_key "${RUBRIC_GENERATOR_API_KEY}"
    --rubric_num_candidates "${RUBRIC_NUM_CANDIDATES}"
    --rubric_generation_temperature "${RUBRIC_GENERATION_TEMPERATURE}"
    --rubric_generation_max_tokens "${RUBRIC_GENERATION_MAX_TOKENS}"
    --rubric_selection_strategy "${RUBRIC_SELECTION_STRATEGY}"
    --openrubrics_input_file "${OPENRUBRICS_INPUT_FILE}"
    --training_artifact_dir "${TRAINING_ARTIFACT_DIR}"
    --local_ckpt "${LOCAL_CKPT}"
    --local_tokenizer "${LOCAL_TOKENIZER}"
    --global_vllm_model "${GLOBAL_VLLM_MODEL}"
    --global_vllm_base_url "${GLOBAL_VLLM_BASE_URL}"
    --hierarchical_local_weight "${HIERARCHICAL_LOCAL_WEIGHT}"
    --hierarchical_group_weight "${HIERARCHICAL_GROUP_WEIGHT}"
    --hierarchical_global_weight "${HIERARCHICAL_GLOBAL_WEIGHT}"
  )
fi

"${cmd[@]}" | tee "${OUT_DIR}/stdout.log"

echo ""
echo "Done. Results saved to: ${OUTPUT_ROOT}/${BENCHMARK}"
