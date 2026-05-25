#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${HOME}/dace/ace-g"
DATASET_ROOT="${HOME}/dace/datasets/cambridge"
#SCENES="shopfacade kingscollege oldhospital stmaryschurch greatcourt"
SCENES="kingscollege oldhospital stmaryschurch greatcourt"

OUTPUT_DIR="${HOME}/dace/outputs/alpha_test"
MAX_BUFFER_SIZE="4000000"
SAMPLES_PER_IMAGE="1024"  # Increase this (e.g. 4000 or 10000) to sample more patches per image
CACHE_ROOT="${HOME}/dace/cache"
TORCH_HOME="${HOME}/dace/cache/torch"

eval "$(conda shell.bash hook)"
conda activate ~/dace_env310

unset VIRTUAL_ENV
hash -r

export PATH="$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$(python -c 'import site,glob; print(glob.glob(site.getsitepackages()[0] + "/torch/lib")[0])')"
export TORCH_HOME="${TORCH_HOME}"
export PYTHONPATH="${REPO_ROOT}/src:${HOME}/dace/Depth-Anything-V2${PYTHONPATH:+:${PYTHONPATH}}"

# Find GPU with most free memory
BEST_GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n1 || true)
if [[ -z "$BEST_GPU_INFO" ]]; then
    echo "ERROR: Failed to run nvidia-smi or no GPUs found."
    exit 1
fi

BEST_GPU_ID=$(echo "$BEST_GPU_INFO" | awk -F, '{print $1}' | xargs)
BEST_GPU_FREE=$(echo "$BEST_GPU_INFO" | awk -F, '{print $2}' | xargs)

# 10 GB = 10 * 1024 MB = 10240 MB
if [ "$BEST_GPU_FREE" -lt 40000 ]; then
    echo "ERROR: No GPU available with more than 10GB of free memory."
    echo "Best GPU is $BEST_GPU_ID with ${BEST_GPU_FREE} MB free."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$BEST_GPU_ID
echo "Selected GPU $BEST_GPU_ID with ${BEST_GPU_FREE} MB free memory."

echo "=========================================================="
echo " Starting Offline Alpha Fusion Test"
echo "=========================================================="

for SCENE in $SCENES; do
  echo ">>> Processing scene: ${SCENE}"
  SCENE_OUTPUT_DIR="${OUTPUT_DIR}/${SCENE}"
  mkdir -p "${SCENE_OUTPUT_DIR}"
  
  MODEL0="${HOME}/dace/outputs/DINOv2_vitl_reg-cambridge-${SCENE}_map.yaml"
  MODEL1="${HOME}/dace/outputs/dptv2_vitl-cambridge-${SCENE}_map.yaml"

  python "${REPO_ROOT}/scripts/offline_alpha_test.py" \
    --model0 "${MODEL0}" \
    --model1 "${MODEL1}" \
    --dataset_root "${DATASET_ROOT}" \
    --scene "${SCENE}" \
    --output_dir "${SCENE_OUTPUT_DIR}" \
    --max_buffer_size "${MAX_BUFFER_SIZE}" \
    --samples_per_image "${SAMPLES_PER_IMAGE}" \
    "$@"
done

echo "=========================================================="
echo " Completed Offline Alpha Fusion Test Successfully!"
echo "=========================================================="
