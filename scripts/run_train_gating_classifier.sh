#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${HOME}/dace/ace-g"
OUTPUT_DIR="${HOME}/dace/outputs/alpha_test"
SCENE="shopfacade"
SEED=42
EPOCHS=10
LR="1e-4"
WEIGHT_DECAY="1e-4"
BATCH_SIZE=128

# Conda and environment initialization
eval "$(conda shell.bash hook)"
conda activate ~/dace_env310

unset VIRTUAL_ENV
hash -r

export PATH="$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$(python -c 'import site,glob; print(glob.glob(site.getsitepackages()[0] + "/torch/lib")[0])')"
export PYTHONPATH="${REPO_ROOT}/src:${HOME}/dace/Depth-Anything-V2${PYTHONPATH:+:${PYTHONPATH}}"

# Find GPU with most free memory
BEST_GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n1 || true)
if [[ -z "$BEST_GPU_INFO" ]]; then
    echo "ERROR: Failed to run nvidia-smi or no GPUs found."
    exit 1
fi

BEST_GPU_ID=$(echo "$BEST_GPU_INFO" | awk -F, '{print $1}' | xargs)
BEST_GPU_FREE=$(echo "$BEST_GPU_INFO" | awk -F, '{print $2}' | xargs)

# Enforce a minimum free memory constraint (e.g., 5GB)
if [ "$BEST_GPU_FREE" -lt 5000 ]; then
    echo "ERROR: No GPU available with more than 5GB of free memory."
    echo "Best GPU is $BEST_GPU_ID with ${BEST_GPU_FREE} MB free."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$BEST_GPU_ID
echo "Selected GPU $BEST_GPU_ID with ${BEST_GPU_FREE} MB free memory."

DATASET_ROOT="${OUTPUT_DIR}/${SCENE}"
echo "=========================================================="
# Set random seed for PyTorch / NumPy
export PYTHONHASHSEED=$SEED

echo " Training MoE Gating Classifier Head for Scene: ${SCENE}"
echo " Dataset Root: ${DATASET_ROOT}"
echo "=========================================================="

python "${REPO_ROOT}/scripts/train_gating_classifier.py" \
  --dataset_root "${DATASET_ROOT}" \
  --scene "${SCENE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --batch_size "${BATCH_SIZE}" \
  "$@"

echo "=========================================================="
echo " MoE Gating Classifier Training Completed Successfully!"
echo "=========================================================="
