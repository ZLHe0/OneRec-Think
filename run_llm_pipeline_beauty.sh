#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"

cd "${REPO_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH; please install/miniconda and rerun." >&2
  exit 1
fi

ENV_NAME="${ENV_NAME:-onerec-think-llm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
FORCE_RECREATE_ENV="${FORCE_RECREATE_ENV:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
  if [[ "${FORCE_RECREATE_ENV}" == "1" ]]; then
    echo "Removing existing env: ${ENV_NAME}"
    conda env remove -n "${ENV_NAME}" -y
  fi
fi

if ! conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
  echo "Creating conda env ${ENV_NAME} with Python ${PYTHON_VERSION}"
  conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

# Optional: disable W&B logging
if [[ "${DISABLE_WANDB:-0}" == "1" ]]; then
  export WANDB_MODE=disabled
fi

# Optional: skip package installation
if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install --upgrade pip
  if [[ "${INSTALL_CUDA_TORCH:-1}" == "1" ]]; then
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  else
    python -m pip install torch torchvision torchaudio
  fi
  python -m pip install \
    transformers \
    datasets \
    accelerate \
    huggingface-hub \
    trl \
    unsloth \
    bitsandbytes \
    deepspeed \
    pandas \
    pyarrow \
    numpy \
    tqdm \
    sentencepiece \
    protobuf \
    wandb
fi

DATA_DIR="${DATA_DIR:-data}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs}"
NUM_GPUS="${NUM_GPUS:-}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-train/scripts/ds_config_zero2.json}"
HOSTFILE="${HOSTFILE:-}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

if [[ ! -f "${DATA_DIR}/Beauty.pretrain.json" ]]; then
  echo "Missing ${DATA_DIR}/Beauty.pretrain.json" >&2
  exit 1
fi

if [[ ! -f "${DATA_DIR}/sequential_data_processed.txt" ]]; then
  echo "Missing ${DATA_DIR}/sequential_data_processed.txt" >&2
  exit 1
fi

if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "Missing DeepSpeed config: ${DEEPSPEED_CONFIG}" >&2
  exit 1
fi

if [[ -z "${NUM_GPUS}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  else
    NUM_GPUS="1"
  fi
fi

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  NUM_GPUS="1"
fi

export TOKENIZERS_PARALLELISM=false

DEEPSPEED_CMD=(deepspeed --num_gpus "${NUM_GPUS}")
if [[ -n "${HOSTFILE}" ]]; then
  if [[ ! -f "${HOSTFILE}" ]]; then
    echo "HOSTFILE set but not found: ${HOSTFILE}" >&2
    exit 1
  fi
  DEEPSPEED_CMD+=(--hostfile "${HOSTFILE}")
fi

echo "[1/5] Preprocessing OneRec data"
python src/prep_onerec_data.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${DATA_DIR}"

echo "[2/5] Generating training conversations"
python src/generate_training_data_onerec.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${DATA_DIR}" \
  --seed 42

echo "[3/5] Stage 1: vocab extension"
"${DEEPSPEED_CMD[@]}" src/finetune_qwen3_8b_vocab.py \
  --deepspeed "${DEEPSPEED_CONFIG}"

echo "[4/5] Stage 2: full fine-tuning"
"${DEEPSPEED_CMD[@]}" src/finetune_qwen3_8b_full.py \
  --deepspeed "${DEEPSPEED_CONFIG}"

echo "[5/5] Evaluation on Beauty validation split"
python src/evaluate_llm.py \
  --model_path models/qwen3_8b_full_finetuned/final \
  --data_path "${DATA_DIR}/Beauty_sequences_val.parquet" \
  --split_type val \
  --batch_size 4 \
  --num_beams 10 \
  --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \
  --save_results "${RESULTS_DIR}/eval_val.json" \
  --log_file "${LOG_DIR}/eval_val.log"

echo "Pipeline complete. Results: ${RESULTS_DIR}/eval_val.json"
