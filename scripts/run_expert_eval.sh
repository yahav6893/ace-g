#!/usr/bin/env bash
set -euo pipefail

# Expert-forcing eval runner for ACE-G / MoGU checkpoints.
#
# Purpose:
#   Run registration + immediate pose evaluation for the SAME trained checkpoint in three modes:
#     1) fused   : normal MoGU/fusion output
#     2) expert0 : force expert 0 only
#     3) expert1 : force expert 1 only
#
# Required patched repo files:
#   - ace_g/register_images.py must include: eval_after_register, sanity_check_force_expert
#   - ace_g/eval_poses.py      must include: wandb_metric_prefix
#
# Usage examples:
#   BASE_REG_CONFIG=/path/to/shopfacade_map.yaml bash run_expert_eval.sh
#   bash run_expert_eval.sh /path/to/shopfacade_map.yaml
#
# Optional overrides:
#   REPO_ROOT=$HOME/dace/ace-g
#   CONDA_ENV=$HOME/dace_env310
#   WANDB_ENTITY=yahav6893
#   WANDB_PROJECT=DACE
#   EVAL_MODES="fused expert0 expert1"
#   OVERRIDE_DIR=/path/to/yaml_overrides
#   CUDA_VISIBLE_DEVICES=0
#
# Extra args after the base config are forwarded to `python -m ace_g.register_images`.

REPO_ROOT="${REPO_ROOT:-${HOME}/dace/ace-g}"
CONDA_ENV="${CONDA_ENV:-${HOME}/dace_env310}"
TORCH_HOME="${TORCH_HOME:-${HOME}/dace/cache/torch}"
WANDB_ENTITY="${WANDB_ENTITY:-yahav6893}"
WANDB_PROJECT="${WANDB_PROJECT:-DACE}"
AUTO_SELECT_GPU="${AUTO_SELECT_GPU:-1}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-0}"
EVAL_MODES="${EVAL_MODES:-fused expert0 expert1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERRIDE_DIR="${OVERRIDE_DIR:-${SCRIPT_DIR}}"
BASE_REG_CONFIG="${BASE_REG_CONFIG:-}"

if [[ $# -gt 0 && "${1}" != --* ]]; then
    BASE_REG_CONFIG="$1"
    shift
fi

if [[ -z "${BASE_REG_CONFIG}" ]]; then
    echo "ERROR: BASE_REG_CONFIG is required."
    echo ""
    echo "Use one of:"
    echo "  BASE_REG_CONFIG=/path/to/<session>_map.yaml bash $0"
    echo "  bash $0 /path/to/<session>_map.yaml"
    echo ""
    echo "The base config must include reg.head_path, reg.dataset.rgb_files, and reg.dataset.pose_files."
    exit 1
fi

if [[ ! -f "${BASE_REG_CONFIG}" ]]; then
    echo "ERROR: BASE_REG_CONFIG does not exist: ${BASE_REG_CONFIG}"
    exit 1
fi

# Activate the same environment style as the full pipeline, unless skipped explicitly.
if [[ "${SKIP_CONDA:-0}" != "1" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

unset VIRTUAL_ENV || true
hash -r

export PATH="${CONDA_PREFIX:-}/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME}"

# Match the full pipeline's torch runtime library handling, but keep it best-effort for portability.
TORCH_LIB_DIR="$(python - <<'PY' || true
import glob, site
roots = site.getsitepackages()
for root in roots:
    matches = glob.glob(root + '/torch/lib')
    if matches:
        print(matches[0])
        break
PY
)"
if [[ -n "${TORCH_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
fi

REG_PY="${REPO_ROOT}/src/ace_g/register_images.py"
EVAL_PY="${REPO_ROOT}/src/ace_g/eval_poses.py"

if [[ ! -f "${REG_PY}" ]]; then
    echo "ERROR: Cannot find ${REG_PY}"
    exit 1
fi
if [[ ! -f "${EVAL_PY}" ]]; then
    echo "ERROR: Cannot find ${EVAL_PY}"
    exit 1
fi
if ! grep -q "eval_after_register" "${REG_PY}"; then
    echo "ERROR: ${REG_PY} does not look patched. Missing eval_after_register."
    echo "Copy the patched register_images.py before running this script."
    exit 1
fi
if ! grep -q "sanity_check_force_expert" "${REG_PY}"; then
    echo "ERROR: ${REG_PY} does not expose sanity_check_force_expert."
    exit 1
fi
if ! grep -q "wandb_metric_prefix" "${EVAL_PY}"; then
    echo "ERROR: ${EVAL_PY} does not look patched. Missing wandb_metric_prefix."
    echo "Copy the patched eval_poses.py before running this script."
    exit 1
fi

# Find the GPU with most free memory unless the user already pinned one.
if [[ "${AUTO_SELECT_GPU}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    BEST_GPU_INFO="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n1 || true)"
    if [[ -z "${BEST_GPU_INFO}" ]]; then
        echo "WARNING: nvidia-smi failed or no GPUs found. Continuing without setting CUDA_VISIBLE_DEVICES."
    else
        BEST_GPU_ID="$(echo "${BEST_GPU_INFO}" | awk -F, '{print $1}' | xargs)"
        BEST_GPU_FREE="$(echo "${BEST_GPU_INFO}" | awk -F, '{print $2}' | xargs)"
        if [[ "${BEST_GPU_FREE}" -lt "${GPU_MIN_FREE_MB}" ]]; then
            echo "ERROR: Best GPU ${BEST_GPU_ID} has ${BEST_GPU_FREE} MB free, below GPU_MIN_FREE_MB=${GPU_MIN_FREE_MB}."
            exit 1
        fi
        export CUDA_VISIBLE_DEVICES="${BEST_GPU_ID}"
        echo "Selected GPU ${BEST_GPU_ID} with ${BEST_GPU_FREE} MB free memory."
    fi
else
    echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
fi

# One W&B run for all three eval modes, so eval/fused, eval/expert0, eval/expert1 appear together.
export WANDB_ENTITY="${WANDB_ENTITY}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
    export WANDB_RUN_ID="$(python - <<'PY'
try:
    import wandb
    print(wandb.util.generate_id())
except Exception:
    import time, os
    print(f"expert_eval_{int(time.time())}_{os.getpid()}")
PY
)"
fi

mode_to_yaml() {
    case "$1" in
        fused)   echo "${OVERRIDE_DIR}/reg_fused.yaml" ;;
        expert0) echo "${OVERRIDE_DIR}/reg_expert0.yaml" ;;
        expert1) echo "${OVERRIDE_DIR}/reg_expert1.yaml" ;;
        *)
            echo "ERROR: Unknown eval mode '$1'. Expected fused, expert0, expert1." >&2
            exit 1
            ;;
    esac
}

echo "=========================================================="
echo " Starting ACE-G Expert Eval: fused / expert0 / expert1"
echo "=========================================================="
echo "Configuration:"
echo "  REPO_ROOT:        ${REPO_ROOT}"
echo "  BASE_REG_CONFIG:  ${BASE_REG_CONFIG}"
echo "  OVERRIDE_DIR:     ${OVERRIDE_DIR}"
echo "  EVAL_MODES:       ${EVAL_MODES}"
echo "  WANDB_ENTITY:     ${WANDB_ENTITY}"
echo "  WANDB_PROJECT:    ${WANDB_PROJECT}"
echo "  WANDB_RUN_ID:     ${WANDB_RUN_ID}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "=========================================================="

for MODE in ${EVAL_MODES}; do
    MODE_YAML="$(mode_to_yaml "${MODE}")"
    if [[ ! -f "${MODE_YAML}" ]]; then
        echo "ERROR: Missing override yaml for mode '${MODE}': ${MODE_YAML}"
        exit 1
    fi

    echo ""
    echo ">>> REGISTER + EVAL MODE: ${MODE}"
    echo "    Override: ${MODE_YAML}"

    python -m ace_g.register_images \
      --config "${BASE_REG_CONFIG}" "${MODE_YAML}" \
      "$@"
done

echo ""
echo "=========================================================="
echo " Expert Eval Completed Successfully"
echo "=========================================================="
echo "Use these W&B keys:"
echo "  eval/fused/median_error_cm,   eval/fused/median_error_deg"
echo "  eval/expert0/median_error_cm, eval/expert0/median_error_deg"
echo "  eval/expert1/median_error_cm, eval/expert1/median_error_deg"
echo ""
echo "Decision metric:"
echo "  fused_minus_best_cm  = eval/fused/median_error_cm  - min(eval/expert0/median_error_cm, eval/expert1/median_error_cm)"
echo "  fused_minus_best_deg = eval/fused/median_error_deg - min(eval/expert0/median_error_deg, eval/expert1/median_error_deg)"
echo ""
echo "Interpretation:"
echo "  < 0  : fusion beats both experts"
echo "  ~ 0  : fusion matches the better expert"
echo "  > 0  : fusion is hurting pose"
