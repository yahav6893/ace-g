#!/usr/bin/env bash
set -euo pipefail

# Run the end-to-end fusion smoking gun (scripts/smoking_gun_fusion.py) over one or more
# scenes. Plants complementary experts and asserts the REAL fusion stack turns a correct
# per-patch gate signal into "fused < best single". See smoking_gun_fusion.py for the rationale.
#
# All parameters are overrideable via environment variables, e.g.:
#   MODEL_NAME=MoGU_N2_frozen_total_tau0_scalar_unc SCENES="shopfacade kingscollege" \
#     bash scripts/run_smoking_gun.sh
# Or point at an explicit checkpoint:
#   HEAD_PATH=~/dace/outputs/heads/my_fused_head.pt SCENES=shopfacade \
#     bash scripts/run_smoking_gun.sh

REPO_ROOT="${REPO_ROOT:-${HOME}/dace/ace-g}"
DATASET_ROOT="${DATASET_ROOT:-${HOME}/dace/datasets/cambridge}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/dace/outputs}"
TORCH_HOME="${TORCH_HOME:-${HOME}/dace/cache/torch}"

# Fusion config (must contain sst.encoder; lives in configs_custom/). Override CONFIG_NAME or
# pass an absolute CONFIG_PATH directly.
CONFIG_NAME="${CONFIG_NAME:-MoGU_N2_frozen_total_tau0_scalar_unc.yaml}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs_custom/${CONFIG_NAME}}"

SCENES="${SCENES:-shopfacade}"

# Trained fused head. If HEAD_PATH is empty it is derived per scene from MODEL_NAME as
#   ${OUTPUT_ROOT}/${MODEL_NAME}-${DATASET_LABEL}-${scene}_head.pt
# (matching the pipeline's artifact naming). Set HEAD_PATH to use one explicit checkpoint
# for every scene instead.
MODEL_NAME="${MODEL_NAME:-}"
HEAD_PATH="${HEAD_PATH:-}"
DATASET_LABEL="${DATASET_LABEL:-$(basename "${DATASET_ROOT}")}"

# Test-image glob. RGB_GLOB overrides the whole pattern; otherwise it is built per scene as
#   ${DATASET_ROOT}/${scene}/test/rgb/*.${RGB_EXT}
# Cambridge uses .png; set RGB_EXT=jpg for datasets that use JPEGs.
RGB_GLOB="${RGB_GLOB:-}"
RGB_EXT="${RGB_EXT:-png}"

# Smoking-gun knobs (forwarded to the python script).
NUM_IMAGES="${NUM_IMAGES:-10}"
MAX_SIDE="${MAX_SIDE:-640}"
CORRUPT_OFFSET_M="${CORRUPT_OFFSET_M:-5.0}"
MARGIN="${MARGIN:-2.0}"

# Inference-only check; far below the pipeline's 55 GB training requirement.
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-8000}"

if [[ -z "${HEAD_PATH}" && -z "${MODEL_NAME}" ]]; then
    echo "ERROR: set HEAD_PATH=<checkpoint.pt> (explicit) or MODEL_NAME=<name> (to derive per scene)." >&2
    exit 2
fi

run_stages() {
    eval "$(conda shell.bash hook)"
    conda activate ~/dace_env310

    unset VIRTUAL_ENV
    hash -r

    export PATH="$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$(python -c 'import site,glob; print(glob.glob(site.getsitepackages()[0] + "/torch/lib")[0])')"
    export TORCH_HOME="${TORCH_HOME}"
    # Depth encoder lives outside the repo; the fusion config needs it on the path.
    export PYTHONPATH="${HOME}/dace/Depth-Anything-V2:${PYTHONPATH:-}"

    # Pick the GPU with the most free memory unless one is pre-set.
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        BEST_GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n1 || true)
        if [[ -z "$BEST_GPU_INFO" ]]; then
            echo "ERROR: Failed to run nvidia-smi or no GPUs found."
            exit 1
        fi
        BEST_GPU_ID=$(echo "$BEST_GPU_INFO" | awk -F, '{print $1}' | xargs)
        BEST_GPU_FREE=$(echo "$BEST_GPU_INFO" | awk -F, '{print $2}' | xargs)
        if [ "$BEST_GPU_FREE" -lt "$MIN_GPU_FREE_MB" ]; then
            echo "ERROR: No GPU with >= ${MIN_GPU_FREE_MB} MB free. Best is GPU $BEST_GPU_ID with ${BEST_GPU_FREE} MB."
            exit 1
        fi
        export CUDA_VISIBLE_DEVICES=$BEST_GPU_ID
        echo "Selected GPU $BEST_GPU_ID with ${BEST_GPU_FREE} MB free memory."
    else
        echo "Using pre-set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    fi

    echo "=========================================================="
    echo " Fusion Smoking Gun (planted complementarity, end-to-end)"
    echo "=========================================================="
    echo "  CONFIG_PATH:          ${CONFIG_PATH}"
    echo "  SCENES:               ${SCENES}"
    echo "  HEAD_PATH:            ${HEAD_PATH:-<derived from MODEL_NAME=${MODEL_NAME}>}"
    echo "  NUM_IMAGES/MAX_SIDE:  ${NUM_IMAGES} / ${MAX_SIDE}"
    echo "  CORRUPT_OFFSET_M:     ${CORRUPT_OFFSET_M}"
    echo "  MARGIN:               ${MARGIN}"
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "=========================================================="

    test -f "${CONFIG_PATH}" || { echo "ERROR: config not found: ${CONFIG_PATH}" >&2; exit 2; }

    local failed=()
    for scene in ${SCENES}; do
        if [[ -n "${HEAD_PATH}" ]]; then
            scene_head="${HEAD_PATH}"
        else
            scene_head="${OUTPUT_ROOT}/${MODEL_NAME}-${DATASET_LABEL}-${scene}_head.pt"
        fi
        if [[ -n "${RGB_GLOB}" ]]; then
            rgb_glob="${RGB_GLOB}"
        else
            rgb_glob="${DATASET_ROOT}/${scene}/test/rgb/*.${RGB_EXT}"
        fi

        echo ""
        echo ">>> SCENE: ${scene}"
        echo "    head: ${scene_head}"
        echo "    rgb : ${rgb_glob}"
        if [[ ! -f "${scene_head}" ]]; then
            echo "    SKIP/FAIL: head checkpoint not found." >&2
            failed+=("${scene} (missing head)")
            continue
        fi

        # The python script exits 0 on PASS, 1 on FAIL; don't let it abort the loop.
        if python "${REPO_ROOT}/scripts/smoking_gun_fusion.py" \
            --repo-root "${REPO_ROOT}" \
            --config-path "${CONFIG_PATH}" \
            --head-path "${scene_head}" \
            --rgb-glob "${rgb_glob}" \
            --num-images "${NUM_IMAGES}" \
            --max-side "${MAX_SIDE}" \
            --corrupt-offset-m "${CORRUPT_OFFSET_M}" \
            --margin "${MARGIN}" \
            "$@"; then
            echo "    ${scene}: PASS"
        else
            echo "    ${scene}: FAIL" >&2
            failed+=("${scene}")
        fi
    done

    echo ""
    echo "=========================================================="
    if [[ ${#failed[@]} -eq 0 ]]; then
        echo " Smoking gun PASSED for all scenes."
        echo "=========================================================="
    else
        echo " Smoking gun FAILED for: ${failed[*]}"
        echo "=========================================================="
        exit 1
    fi
}

run_stages "$@"
