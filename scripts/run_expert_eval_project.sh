#!/usr/bin/env bash
set -euo pipefail

# Eval-only MoGU expert comparison runner.
# Runs register+eval for fused / expert0 / expert1 using train_status JSON.
# Install this as: ace-g/scripts/run_expert_eval_project_v2.sh

# Default layout. If this script lives in ace-g/scripts, BASE_DIR resolves to the
# directory that contains ace-g/ and outputs/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

REPO_ROOT="${REPO_ROOT:-${BASE_DIR}/ace-g}"
DATASET_ROOT="${DATASET_ROOT:-${BASE_DIR}/datasets/cambridge}"
CONFIG_NAME="${CONFIG_NAME:-MoGU_N2_frozen_total_tau0_scalar_unc.yaml}"
SCENES="${SCENES:-shopfacade}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DIR}/outputs}"
CACHE_ROOT="${CACHE_ROOT:-${BASE_DIR}/cache}"
TORCH_HOME="${TORCH_HOME:-${BASE_DIR}/cache/torch}"
WANDB_ENTITY="${WANDB_ENTITY:-yahav6893}"
WANDB_PROJECT="${WANDB_PROJECT:-DACE}"
SESSION_PREFIX="${SESSION_PREFIX:-}"
MODEL_NAME="${MODEL_NAME:-}"
EVAL_MODES="${EVAL_MODES:-fused expert0 expert1}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-12000}"
PYTHONPATH_PREPEND="${PYTHONPATH_PREPEND:-${BASE_DIR}/Depth-Anything-V2}"

# Positional arg: train_status JSON. Do NOT pass a *_map.yaml here.
TRAIN_STATUS_JSON="${TRAIN_STATUS_JSON:-}"
if [[ $# -gt 0 ]]; then
    TRAIN_STATUS_JSON="$1"
    shift
fi

if [[ -n "${TRAIN_STATUS_JSON}" && "${TRAIN_STATUS_JSON}" == *_map.yaml ]]; then
    echo "ERROR: This eval runner expects a train_status JSON, not a *_map.yaml."
    echo "Example:"
    echo "  TRAIN_STATUS_JSON=${OUTPUT_ROOT}/train_status/auto_pipeline_status.json bash ${REPO_ROOT}/scripts/run_expert_eval_project_v2.sh"
    echo "Available train_status candidates:"
    find "${OUTPUT_ROOT}/train_status" -maxdepth 1 -type f -name '*.json' -printf '  %p\n' 2>/dev/null || true
    exit 2
fi

if [[ -z "${TRAIN_STATUS_JSON}" ]]; then
    if [[ -n "${TRAIN_STATUS_NAME:-}" ]]; then
        TRAIN_STATUS_JSON="${OUTPUT_ROOT}/train_status/${TRAIN_STATUS_NAME}"
    else
        mapfile -t STATUS_CANDIDATES < <(find "${OUTPUT_ROOT}/train_status" -maxdepth 1 -type f -name '*.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk '{print $2}')
        if [[ "${#STATUS_CANDIDATES[@]}" -eq 1 ]]; then
            TRAIN_STATUS_JSON="${STATUS_CANDIDATES[0]}"
        elif [[ "${#STATUS_CANDIDATES[@]}" -gt 1 ]]; then
            echo "ERROR: Multiple train_status JSON files found. Please pass one explicitly:"
            printf '  %s\n' "${STATUS_CANDIDATES[@]}"
            echo ""
            echo "Usage:"
            echo "  bash ${REPO_ROOT}/scripts/run_expert_eval_project_v2.sh /path/to/train_status.json"
            exit 2
        else
            echo "ERROR: No train_status JSON found under ${OUTPUT_ROOT}/train_status."
            echo "Pass it explicitly as the first argument or set TRAIN_STATUS_JSON."
            exit 2
        fi
    fi
fi

if [[ ! -f "${TRAIN_STATUS_JSON}" ]]; then
    echo "ERROR: TRAIN_STATUS_JSON does not exist: ${TRAIN_STATUS_JSON}"
    exit 2
fi

REQUIRED_SCRIPT="${REPO_ROOT}/scripts/register_eval_expert_modes.py"
if [[ ! -f "${REQUIRED_SCRIPT}" ]]; then
    echo "ERROR: Missing ${REQUIRED_SCRIPT}"
    echo "Install it with:"
    echo "  cp /mnt/data/register_eval_expert_modes.py ${REQUIRED_SCRIPT}"
    exit 2
fi

if ! grep -q "sanity_check_force_expert" "${REPO_ROOT}/src/ace_g/register_images.py"; then
    echo "ERROR: register_images.py does not look patched for sanity_check_force_expert."
    echo "Install patched register_images.py before running expert eval."
    exit 2
fi

if ! grep -q "wandb_metric_prefix" "${REPO_ROOT}/src/ace_g/eval_poses.py"; then
    echo "ERROR: eval_poses.py does not look patched for wandb_metric_prefix."
    echo "Install patched eval_poses.py before running expert eval."
    exit 2
fi

run_stages() {
    eval "$(conda shell.bash hook)"
    conda activate ~/dace_env310

    unset VIRTUAL_ENV
    hash -r

    export PATH="$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$(python -c 'import site,glob; paths=glob.glob(site.getsitepackages()[0] + "/torch/lib"); print(paths[0] if paths else "")')"
    export TORCH_HOME="${TORCH_HOME}"

    # Find GPU with most free memory if not pre-set.
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        BEST_GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n1 || true)
        if [[ -z "$BEST_GPU_INFO" ]]; then
            echo "ERROR: Failed to run nvidia-smi or no GPUs found."
            exit 1
        fi

        BEST_GPU_ID=$(echo "$BEST_GPU_INFO" | awk -F, '{print $1}' | xargs)
        BEST_GPU_FREE=$(echo "$BEST_GPU_INFO" | awk -F, '{print $2}' | xargs)

        if [[ "$BEST_GPU_FREE" -lt "${GPU_MIN_FREE_MB}" ]]; then
            echo "ERROR: No GPU available with more than ${GPU_MIN_FREE_MB} MB free memory."
            echo "Best GPU is $BEST_GPU_ID with ${BEST_GPU_FREE} MB free."
            exit 1
        fi

        export CUDA_VISIBLE_DEVICES=$BEST_GPU_ID
        echo "Selected GPU $BEST_GPU_ID with ${BEST_GPU_FREE} MB free memory."
    else
        echo "Using pre-set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    fi

    export WANDB_RUN_ID="${WANDB_RUN_ID:-$(python -c 'import wandb; print(wandb.util.generate_id())')}"
    export WANDB_RESUME="allow"

    echo "=========================================================="
    echo " Starting ACE-G Expert Eval: fused / expert0 / expert1"
    echo "=========================================================="
    echo "Configuration:"
    echo "  REPO_ROOT:        ${REPO_ROOT}"
    echo "  CONFIG_NAME:      ${CONFIG_NAME}"
    echo "  DATASET_ROOT:     ${DATASET_ROOT}"
    echo "  OUTPUT_ROOT:      ${OUTPUT_ROOT}"
    echo "  TRAIN_STATUS_JSON:${TRAIN_STATUS_JSON}"
    echo "  EVAL_MODES:       ${EVAL_MODES}"
    echo "  WANDB_ENTITY:     ${WANDB_ENTITY}"
    echo "  WANDB_PROJECT:    ${WANDB_PROJECT}"
    echo "  WANDB_RUN_ID:     ${WANDB_RUN_ID}"
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    if [[ -n "${MODEL_NAME}" ]]; then
        echo "  MODEL_NAME:       ${MODEL_NAME}"
    fi
    if [[ -n "${SESSION_PREFIX}" ]]; then
        echo "  SESSION_PREFIX:   ${SESSION_PREFIX}"
    fi
    echo "=========================================================="

    EXTRA_ARGS=()
    if [[ -n "${SESSION_PREFIX}" ]]; then
        EXTRA_ARGS+=(--session-prefix "${SESSION_PREFIX}")
    fi
    if [[ -n "${MODEL_NAME}" ]]; then
        EXTRA_ARGS+=(--model-name "${MODEL_NAME}")
    fi
    if [[ -n "${PYTHONPATH_PREPEND}" ]]; then
        EXTRA_ARGS+=(--pythonpath-prepend "${PYTHONPATH_PREPEND}")
    fi

    python "${REQUIRED_SCRIPT}" \
      --repo-root "${REPO_ROOT}" \
      --dataset-root "${DATASET_ROOT}" \
      --config-name "${CONFIG_NAME}" \
      --output-root "${OUTPUT_ROOT}" \
      --cache-root "${CACHE_ROOT}" \
      --torch-home "${TORCH_HOME}" \
      --train-status-json "${TRAIN_STATUS_JSON}" \
      --scenes ${SCENES} \
      --expert-eval-modes ${EVAL_MODES} \
      --wandb-entity "${WANDB_ENTITY}" \
      --wandb-project "${WANDB_PROJECT}" \
      "${EXTRA_ARGS[@]}" \
      "$@"

    echo ""
    echo "=========================================================="
    echo " Expert Eval Completed Successfully!"
    echo "=========================================================="
}

PIPELINE_LOG="${PIPELINE_LOG:-${OUTPUT_ROOT}/logs/expert_eval_${WANDB_RUN_ID:-manual}.log}"
mkdir -p "$(dirname "${PIPELINE_LOG}")"
echo "Logging expert eval output to ${PIPELINE_LOG}"
run_stages "$@" 2>&1 | tee "${PIPELINE_LOG}"
