#!/usr/bin/env bash
set -e

# Configuration variables
REPO_ROOT="${HOME}/dace/ace-g"
DATASET_ROOT="${HOME}/dace/datasets/cambridge"
CONFIG_NAME="listmultiencoder_freezed_latefusion_N2_dinov2reg_dpt.yaml"
SCENES="shopfacade"
OUTPUT_ROOT="${HOME}/dace/outputs"
CACHE_ROOT="${HOME}/dace/cache"
TORCH_HOME="${HOME}/dace/cache/torch"
WANDB_ENTITY="yahav6893"
WANDB_PROJECT="DACE"
TRAIN_STATUS_NAME="auto_pipeline_status.json"
STATUS_JSON_PATH="${OUTPUT_ROOT}/train_status/${TRAIN_STATUS_NAME}"

# Add this to the top of run_pipeline.sh:
export WANDB_RUN_ID=$(python -c 'import wandb; print(wandb.util.generate_id())')
export WANDB_RESUME="allow"


# Setup our PyTorch environment bypass
export CUDA_VISIBLE_DEVICES=3
export CUDA_FORCE_PTX_JIT=1

echo "=========================================================="
echo " Starting ACE-G Pipeline: Train -> Register -> Eval"
echo "=========================================================="

echo ""
echo ">>> STAGE 1: TRAINING"
echo ""
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
  --promote-heads

echo ""
echo ">>> STAGE 2: REGISTRATION & EVALUATION"
echo ""
python "${REPO_ROOT}/scripts/register_eval.py" \
  --repo-root "${REPO_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --config-name "${CONFIG_NAME}" \
  --output-root "${OUTPUT_ROOT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT}" \
  --scenes ${SCENES} \
  --train-status-json "${STATUS_JSON_PATH}"

echo ""
echo "=========================================================="
echo " Pipeline Completed Successfully!"
echo "=========================================================="
