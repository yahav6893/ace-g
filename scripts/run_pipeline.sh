#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${HOME}/dace/ace-g"
DATASET_ROOT="${HOME}/dace/datasets/cambridge"
CONFIG_NAME="latefusion_l2_unfreeze_N2_dinov2reg_dpt.yaml"
SCENES="shopfacade"
OUTPUT_ROOT="${HOME}/dace/outputs"
CACHE_ROOT="${HOME}/dace/cache"
TORCH_HOME="${HOME}/dace/cache/torch"
WANDB_ENTITY="yahav6893"
WANDB_PROJECT="DACE"
TRAIN_STATUS_NAME="auto_pipeline_status.json"
STATUS_JSON_PATH="${OUTPUT_ROOT}/train_status/${TRAIN_STATUS_NAME}"

eval "$(conda shell.bash hook)"
conda activate ~/dace_env310

unset VIRTUAL_ENV
hash -r

export PATH="$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$(python -c 'import site,glob; print(glob.glob(site.getsitepackages()[0] + "/torch/lib")[0])')"
export TORCH_HOME="${TORCH_HOME}"
# Find GPU with most free memory
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

export WANDB_RUN_ID="$(python -c 'import wandb; print(wandb.util.generate_id())')"
export WANDB_RESUME="allow"

echo "=========================================================="
echo " Starting ACE-G Pipeline: Train -> Register -> Eval"
echo "=========================================================="

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
  --promote-heads

test -f "${STATUS_JSON_PATH}"

echo ">>> STAGE 2: REGISTRATION & EVALUATION"
python "${REPO_ROOT}/scripts/register_eval.py" \
  --repo-root "${REPO_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --torch-home "${TORCH_HOME}" \
  --train-status-json "${STATUS_JSON_PATH}" \
  --scenes ${SCENES} \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT}" \
  --pythonpath-prepend "${HOME}/dace/Depth-Anything-V2"

echo ""
echo "=========================================================="
echo " Pipeline Completed Successfully!"
echo "=========================================================="