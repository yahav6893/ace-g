#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${HOME}/dace/ace-g"
DATASET_ROOT="${HOME}/dace/datasets/cambridge"
SCENES="shopfacade" # default scene, can be overridden by passing scenes as arguments
OUTPUT_ROOT="${HOME}/dace/outputs/alpha_test"
SEED=42
EPOCHS=10
LR="1e-4"
WEIGHT_DECAY="1e-4"
BATCH_SIZE=128
SPLIT="test"

# Conda and environment initialization
eval "$(conda shell.bash hook)"
conda activate ~/dace_env310

unset VIRTUAL_ENV
hash -r

export PATH="$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$(python -c "import site,glob; print(glob.glob(site.getsitepackages()[0] + '/torch/lib')[0])")
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

export PYTHONHASHSEED=$SEED

echo "=========================================================="
echo " Starting ACE-G Gating Classifier Pipeline"
echo "=========================================================="

for SCENE in $SCENES; do
  echo ">>> Processing Scene: ${SCENE}"
  SCENE_OUTPUT_DIR="${OUTPUT_ROOT}/${SCENE}"
  mkdir -p "${SCENE_OUTPUT_DIR}"
  
  MODEL0="${HOME}/dace/outputs/DINOv2_vitl_reg-cambridge-${SCENE}_map.yaml"
  MODEL1="${HOME}/dace/outputs/dptv2_vitl-cambridge-${SCENE}_map.yaml"
  GATING_HEAD="${SCENE_OUTPUT_DIR}/${SCENE}_gating_head_best.pt"

  echo "----------------------------------------------------------"
  echo " STAGE 1: TRAINING GATING CLASSIFIER HEAD"
  echo "----------------------------------------------------------"
  python "${REPO_ROOT}/scripts/train_gating_classifier.py" \
    --dataset_root "${SCENE_OUTPUT_DIR}" \
    --scene "${SCENE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --batch_size "${BATCH_SIZE}"

  echo "----------------------------------------------------------"
  echo " STAGE 2: RUNNING PNP CAMERA LOCALIZATION EVALUATION"
  echo "----------------------------------------------------------"
  python "${REPO_ROOT}/scripts/eval_gating_classifier.py" \
    --model0 "${MODEL0}" \
    --model1 "${MODEL1}" \
    --gating_head "${GATING_HEAD}" \
    --dataset_root "${DATASET_ROOT}" \
    --scene "${SCENE}" \
    --output_dir "${SCENE_OUTPUT_DIR}" \
    --split "${SPLIT}"
    
done

echo "=========================================================="
echo " MoE Gating Classifier Pipeline Completed Successfully!"
echo "=========================================================="
