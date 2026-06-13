#!/usr/bin/env bash
set -euo pipefail

# Default configuration parameters (overrideable via environment variables)
REPO_ROOT="${REPO_ROOT:-${HOME}/dace/ace-g}"
DATASET_ROOT="${DATASET_ROOT:-${HOME}/dace/datasets/cambridge}"
CONFIG_NAME="${CONFIG_NAME:-MoGU_N2_dinov2reg_dpt_aleatoric_epistermic.yaml}"
#SCENES="${SCENES:-shopfacade kingscollege oldhospital stmaryschurch greatcourt}"
SCENES="${SCENES:-shopfacade}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/dace/outputs}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/dace/cache}"
TORCH_HOME="${TORCH_HOME:-${HOME}/dace/cache/torch}"
WANDB_ENTITY="${WANDB_ENTITY:-yahav6893}"
WANDB_PROJECT="${WANDB_PROJECT:-DACE}"
SESSION_PREFIX="${SESSION_PREFIX:-}"
MODEL_NAME="${MODEL_NAME:-}"

# Automatically determine if another session is running
if [[ -z "${SESSION_PREFIX}" ]]; then
    OTHER_PIDS=$(pgrep -u "$USER" -f "run_pipeline.sh|train_scenes.py|register_eval.py" | grep -v "$$" || true)
    if [[ -n "${OTHER_PIDS}" ]]; then
        SESSION_PREFIX="run_$(date +%Y%m%d_%H%M%S)_$$"
        echo "Detected other running pipeline sessions (PIDs: $(echo ${OTHER_PIDS} | xargs))."
        echo "Automatically assigned unique SESSION_PREFIX: ${SESSION_PREFIX}"
    else
        SESSION_PREFIX=""
    fi
fi

# Define status json filename and path
if [[ -z "${TRAIN_STATUS_NAME:-}" ]]; then
    if [[ -n "${SESSION_PREFIX}" ]]; then
        TRAIN_STATUS_NAME="auto_pipeline_status_${SESSION_PREFIX}.json"
    else
        TRAIN_STATUS_NAME="auto_pipeline_status.json"
    fi
fi
STATUS_JSON_PATH="${OUTPUT_ROOT}/train_status/${TRAIN_STATUS_NAME}"

# Define pipeline log file path
if [[ -n "${SESSION_PREFIX}" ]]; then
    PIPELINE_LOG="${PIPELINE_LOG:-${OUTPUT_ROOT}/logs/pipeline_${SESSION_PREFIX}.log}"
else
    PIPELINE_LOG="${PIPELINE_LOG:-}"
fi

run_stages() {
    eval "$(conda shell.bash hook)"
    conda activate ~/dace_env310

    unset VIRTUAL_ENV
    hash -r

    export PATH="$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$(python -c 'import site,glob; print(glob.glob(site.getsitepackages()[0] + "/torch/lib")[0])')"
    export TORCH_HOME="${TORCH_HOME}"

    # Find GPU with most free memory if not pre-set
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        BEST_GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n1 || true)
        if [[ -z "$BEST_GPU_INFO" ]]; then
            echo "ERROR: Failed to run nvidia-smi or no GPUs found."
            exit 1
        fi

        BEST_GPU_ID=$(echo "$BEST_GPU_INFO" | awk -F, '{print $1}' | xargs)
        BEST_GPU_FREE=$(echo "$BEST_GPU_INFO" | awk -F, '{print $2}' | xargs)

        # 55 GB = 55 * 1024 MB = 56320 MB
        if [ "$BEST_GPU_FREE" -lt 56320 ]; then
            echo "ERROR: No GPU available with more than 55GB of free memory."
            echo "Best GPU is $BEST_GPU_ID with ${BEST_GPU_FREE} MB free."
            exit 1
        fi

        export CUDA_VISIBLE_DEVICES=$BEST_GPU_ID
        echo "Selected GPU $BEST_GPU_ID with ${BEST_GPU_FREE} MB free memory."
    else
        echo "Using pre-set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    fi

    export WANDB_RUN_ID="$(python -c 'import wandb; print(wandb.util.generate_id())')"
    export WANDB_RESUME="allow"

    echo "=========================================================="
    echo " Starting ACE-G Pipeline: Train -> Register -> Eval"
    echo "=========================================================="
    echo "Configuration:"
    echo "  CONFIG_NAME:       ${CONFIG_NAME}"
    echo "  SESSION_PREFIX:    ${SESSION_PREFIX:-<none>}"
    echo "  TRAIN_STATUS_NAME: ${TRAIN_STATUS_NAME}"
    echo "  STATUS_JSON_PATH:  ${STATUS_JSON_PATH}"
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    if [[ -n "${MODEL_NAME}" ]]; then
    echo "  MODEL_NAME:        ${MODEL_NAME}"
    fi
    echo "=========================================================="

    # Prepare extra arguments for train/eval scripts
    EXTRA_ARGS=()
    if [[ -n "${SESSION_PREFIX}" ]]; then
        EXTRA_ARGS+=(--session-prefix "${SESSION_PREFIX}")
    fi
    if [[ -n "${MODEL_NAME}" ]]; then
        EXTRA_ARGS+=(--model-name "${MODEL_NAME}")
    fi

    echo ">>> STAGE 1: TRAINING"
    python "${REPO_ROOT}/scripts/train_scenes.py" \
      --repo-root "${REPO_ROOT}" \
      --dataset-root "${DATASET_ROOT}" \
      --config-name "${CONFIG_NAME}" \
      --scenes ${SCENES} \
      --output-root "${OUTPUT_ROOT}" \
      --cache-root "${CACHE_ROOT}" \
      --torch-home "${TORCH_HOME}" \
      --wandb-entity "${WANDB_ENTITY}" \
      --wandb-project "${WANDB_PROJECT}" \
      --train-status-name "${TRAIN_STATUS_NAME}" \
      --pythonpath-prepend "${HOME}/dace/Depth-Anything-V2" \
      "${EXTRA_ARGS[@]}" \
      --promote-heads \
      "$@"

    test -f "${STATUS_JSON_PATH}"

    echo ">>> STAGE 2: REGISTRATION & EVALUATION"
    python "${REPO_ROOT}/scripts/register_eval.py" \
      --repo-root "${REPO_ROOT}" \
      --dataset-root "${DATASET_ROOT}" \
      --config-name "${CONFIG_NAME}" \
      --output-root "${OUTPUT_ROOT}" \
      --cache-root "${CACHE_ROOT}" \
      --torch-home "${TORCH_HOME}" \
      --train-status-json "${STATUS_JSON_PATH}" \
      --scenes ${SCENES} \
      --wandb-entity "${WANDB_ENTITY}" \
      --wandb-project "${WANDB_PROJECT}" \
      --pythonpath-prepend "${HOME}/dace/Depth-Anything-V2" \
      "${EXTRA_ARGS[@]}" \
      "$@"

    echo ""
    echo "=========================================================="
    echo " Pipeline Completed Successfully!"
    echo "=========================================================="
}

if [[ -n "${PIPELINE_LOG}" ]]; then
    mkdir -p "$(dirname "${PIPELINE_LOG}")"
    echo "Logging pipeline output to ${PIPELINE_LOG}"
    run_stages 2>&1 | tee "${PIPELINE_LOG}"
else
    run_stages
fi