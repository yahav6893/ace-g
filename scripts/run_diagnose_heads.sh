#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${HOME}/dace/ace-g"
DATASET_ROOT="${HOME}/dace/datasets/cambridge"
CACHE_ROOT="${HOME}/dace/cache"
TORCH_HOME="${HOME}/dace/cache/torch"
OUTPUT_ROOT="${HOME}/dace/outputs"

# --- DIAGNOSTICS CONFIGURATION PLACEHOLDERS ---
CMD="gate-weights"
CONFIG_NAME="MoGU_N2_dinov2reg_dpt.yaml"
SCENES="shopfacade"
MODEL_PREFIX="MoGU_N2_dinov2reg_dpt-cambridge-shopfacade"

# Note: Adjust HEAD_PATH based on whether you used --promote-heads or not
# HEAD_PATH="${OUTPUT_ROOT}/promoted/${MODEL_PREFIX}_head.pt"
HEAD_PATH="wandb://yahav6893/DACE/model__MoGU_N2_dinov2reg_dpt-cambridge-shopfacade:v0"
CONFIG_PATH="${REPO_ROOT}/configs_custom/${CONFIG_NAME}"
RGB_GLOB="${DATASET_ROOT}/${SCENES}/test/rgb/*.png"

eval "$(conda shell.bash hook)"
conda activate ~/dace_env310

unset VIRTUAL_ENV
hash -r

export PATH="$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$(python -c 'import site,glob; print(glob.glob(site.getsitepackages()[0] + "/torch/lib")[0])')"
export TORCH_HOME="${TORCH_HOME}"
export PYTHONPATH="${HOME}/dace/Depth-Anything-V2${PYTHONPATH:+:$PYTHONPATH}"

# Find GPU with most free memory
BEST_GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n1 || true)
if [[ -z "$BEST_GPU_INFO" ]]; then
    echo "ERROR: Failed to run nvidia-smi or no GPUs found."
    exit 1
fi

BEST_GPU_ID=$(echo "$BEST_GPU_INFO" | awk -F, '{print $1}' | xargs)
BEST_GPU_FREE=$(echo "$BEST_GPU_INFO" | awk -F, '{print $2}' | xargs)

# Memory requirement check (55 GB = 56320 MB)
if [ "$BEST_GPU_FREE" -lt 56320 ]; then
    echo "ERROR: No GPU available with more than 55GB of free memory."
    echo "Best GPU is $BEST_GPU_ID with ${BEST_GPU_FREE} MB free."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$BEST_GPU_ID
echo "Selected GPU $BEST_GPU_ID with ${BEST_GPU_FREE} MB free memory."

echo "=========================================================="
echo " Running diagnose_heads.py"
echo "=========================================================="

case "$CMD" in
  gate-weights)
    python "${REPO_ROOT}/scripts/diagnose_heads.py" --repo-root "${REPO_ROOT}" --wandb-log gate-weights \
      --config-path "${CONFIG_PATH}" \
      --head-path "${HEAD_PATH}" \
      --rgb-glob "${RGB_GLOB}" \
      --use-half --random-sample --eval-registration "$@"
    ;;
  gate-last-layer)
    python "${REPO_ROOT}/scripts/diagnose_heads.py" --repo-root "${REPO_ROOT}" --wandb-log gate-last-layer \
      --head-path "${HEAD_PATH}" \
      --probe-config-path "${CONFIG_PATH}" \
      --probe-rgb-glob "${RGB_GLOB}" \
      --use-half "$@"
    ;;
  print-config)
    python "${REPO_ROOT}/scripts/diagnose_heads.py" --repo-root "${REPO_ROOT}" print-config \
      --head-path "${HEAD_PATH}" "$@"
    ;;
  *)
    # For other commands, we just pass the bare CMD and "$@"
    python "${REPO_ROOT}/scripts/diagnose_heads.py" --repo-root "${REPO_ROOT}" "$CMD" "$@"
    ;;
esac
