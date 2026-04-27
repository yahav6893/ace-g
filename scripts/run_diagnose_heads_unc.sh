#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${HOME}/dace/ace-g"
DATASET_ROOT="${HOME}/dace/datasets/cambridge"
CACHE_ROOT="${HOME}/dace/cache"
TORCH_HOME="${HOME}/dace/cache/torch"
OUTPUT_ROOT="${HOME}/dace/outputs"

# ==========================================================
# Uncertainty / MoGU diagnostics configuration
# ==========================================================
CONFIG_NAME="MoGU_N2_dinov2reg_dpt.yaml"
SCENES="shopfacade"
MODEL_PREFIX="MoGU_N2_dinov2reg_dpt-cambridge-shopfacade"

# If you promoted heads locally, use this instead:
# HEAD_PATH="${OUTPUT_ROOT}/promoted/${MODEL_PREFIX}_head.pt"
HEAD_PATH="wandb://yahav6893/DACE/model__MoGU_N2_dinov2reg_dpt-cambridge-shopfacade:v5"

CONFIG_PATH="${REPO_ROOT}/configs_custom/${CONFIG_NAME}"
RGB_GLOB="${DATASET_ROOT}/${SCENES}/test/rgb/*.png"

# Use diagnose_heads_unc.py if you keep it separate, otherwise fall back to diagnose_heads.py.
DIAGNOSE_SCRIPT="${REPO_ROOT}/scripts/diagnose_heads_unc.py"
if [[ ! -f "${DIAGNOSE_SCRIPT}" ]]; then
    DIAGNOSE_SCRIPT="${REPO_ROOT}/scripts/diagnose_heads.py"
fi

OUT_DIR="${OUTPUT_ROOT}/diagnostics/uncertainty/${MODEL_PREFIX}"
OUT_CSV="${OUT_DIR}/uncertainty_decisions.csv"
OUT_JSON="${OUT_DIR}/uncertainty_decisions.json"
MAP_DIR="${OUT_DIR}/maps"

NUM_IMAGES=25
MAX_SIDE=640
SEED=42
MIN_FREE_MB=56320

# ==========================================================
# Environment
# ==========================================================
eval "$(conda shell.bash hook)"
conda activate ~/dace_env310

unset VIRTUAL_ENV
hash -r

export PATH="${CONDA_PREFIX}/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$(python -c 'import site,glob; print(glob.glob(site.getsitepackages()[0] + "/torch/lib")[0])')"
export TORCH_HOME="${TORCH_HOME}"
export PYTHONPATH="${REPO_ROOT}/src:${HOME}/dace/Depth-Anything-V2${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUT_DIR}" "${MAP_DIR}"

# ==========================================================
# Select GPU with most free memory
# ==========================================================
BEST_GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n1 || true)
if [[ -z "${BEST_GPU_INFO}" ]]; then
    echo "ERROR: Failed to run nvidia-smi or no GPUs found."
    exit 1
fi

BEST_GPU_ID=$(echo "${BEST_GPU_INFO}" | awk -F, '{print $1}' | xargs)
BEST_GPU_FREE=$(echo "${BEST_GPU_INFO}" | awk -F, '{print $2}' | xargs)

if [[ "${BEST_GPU_FREE}" -lt "${MIN_FREE_MB}" ]]; then
    echo "ERROR: No GPU available with more than $((MIN_FREE_MB / 1024))GB of free memory."
    echo "Best GPU is ${BEST_GPU_ID} with ${BEST_GPU_FREE} MB free."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${BEST_GPU_ID}"
echo "Selected GPU ${BEST_GPU_ID} with ${BEST_GPU_FREE} MB free memory."

# ==========================================================
# Run uncertainty diagnostics
# ==========================================================
echo "=========================================================="
echo " Running uncertainty diagnostics"
echo "=========================================================="
echo "script      : ${DIAGNOSE_SCRIPT}"
echo "config      : ${CONFIG_PATH}"
echo "head        : ${HEAD_PATH}"
echo "rgb_glob    : ${RGB_GLOB}"
echo "out_csv     : ${OUT_CSV}"
echo "out_json    : ${OUT_JSON}"
echo "map_dir     : ${MAP_DIR}"
echo "=========================================================="

python "${DIAGNOSE_SCRIPT}" \
  --repo-root "${REPO_ROOT}" \
  --wandb-log \
  uncertainty-decisions \
  --config-path "${CONFIG_PATH}" \
  --head-path "${HEAD_PATH}" \
  --rgb-glob "${RGB_GLOB}" \
  --num-images "${NUM_IMAGES}" \
  --random-sample \
  --seed "${SEED}" \
  --max-side "${MAX_SIDE}" \
  --device cuda \
  --use-half \
  --out-csv "${OUT_CSV}" \
  --out-json "${OUT_JSON}" \
  --save-maps \
  --out-dir "${MAP_DIR}" \
  "$@"

echo "=========================================================="
echo " Done."
echo " Outputs:"
echo "   CSV : ${OUT_CSV}"
echo "   JSON: ${OUT_JSON}"
echo "   MAPS: ${MAP_DIR}"
echo "=========================================================="
