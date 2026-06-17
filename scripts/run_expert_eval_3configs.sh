#!/usr/bin/env bash
set -euo pipefail

# Run expert-eval (fused/expert0/expert1) for the three post-freeze MoGU configs.
# Install into ace-g/scripts/ and run from /home/eng/vinokuy2/dace or any directory.
#
# Usage, automatic train_status discovery by config stem:
#   bash ace-g/scripts/run_expert_eval_3configs.sh
#
# Safer explicit usage:
#   STATUS_TOTAL_TAU0=/path/to/status_total_tau0.json \
#   STATUS_NONE_MOGU001=/path/to/status_none_mogu001.json \
#   STATUS_TOTAL_MOGU001=/path/to/status_total_mogu001.json \
#   bash ace-g/scripts/run_expert_eval_3configs.sh
#
# Optional:
#   SCENES=shopfacade EVAL_MODES="fused expert0 expert1" bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

REPO_ROOT="${REPO_ROOT:-${BASE_DIR}/ace-g}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DIR}/outputs}"
DATASET_ROOT="${DATASET_ROOT:-${BASE_DIR}/datasets/cambridge}"
SCENES="${SCENES:-shopfacade}"
EVAL_MODES="${EVAL_MODES:-fused expert0 expert1}"
WANDB_ENTITY="${WANDB_ENTITY:-yahav6893}"
WANDB_PROJECT="${WANDB_PROJECT:-DACE}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

RUNNER="${RUNNER:-${REPO_ROOT}/scripts/run_expert_eval_project.sh}"
STATUS_DIR="${STATUS_DIR:-${OUTPUT_ROOT}/train_status}"

CONFIG_TOTAL_TAU0="MoGU_N2_frozen_total_tau0_fixed.yaml"
CONFIG_NONE_MOGU001="MoGU_N2_frozen_none_mogu001_fixed.yaml"
CONFIG_TOTAL_MOGU001="MoGU_N2_frozen_total_mogu001_fixed.yaml"

find_status_for_stem() {
    local stem="$1"
    local explicit_path="${2:-}"

    if [[ -n "${explicit_path}" ]]; then
        if [[ ! -f "${explicit_path}" ]]; then
            echo "ERROR: explicit train_status JSON does not exist: ${explicit_path}" >&2
            return 2
        fi
        echo "${explicit_path}"
        return 0
    fi

    mapfile -t candidates < <(find "${STATUS_DIR}" -maxdepth 1 -type f -name "*${stem}*.json" -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk '{print $2}')

    if [[ "${#candidates[@]}" -eq 1 ]]; then
        echo "${candidates[0]}"
        return 0
    fi

    if [[ "${#candidates[@]}" -gt 1 ]]; then
        echo "ERROR: multiple train_status candidates found for ${stem}; set the explicit STATUS_* variable." >&2
        printf '  %s\n' "${candidates[@]}" >&2
        return 2
    fi

    echo "ERROR: no train_status JSON found for ${stem} under ${STATUS_DIR}." >&2
    echo "Recent train_status files:" >&2
    find "${STATUS_DIR}" -maxdepth 1 -type f -name '*.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 20 | awk '{print "  "$2}' >&2 || true
    return 2
}

run_one() {
    local label="$1"
    local config_name="$2"
    local status_json="$3"
    local stem="${config_name%.yaml}"

    echo ""
    echo "=========================================================="
    echo " Expert eval for ${label}"
    echo "=========================================================="
    echo "CONFIG_NAME:       ${config_name}"
    echo "TRAIN_STATUS_JSON: ${status_json}"
    echo "SCENES:            ${SCENES}"
    echo "EVAL_MODES:        ${EVAL_MODES}"
    echo "=========================================================="

    CONFIG_NAME="${config_name}" \
    TRAIN_STATUS_JSON="${status_json}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    SCENES="${SCENES}" \
    EVAL_MODES="${EVAL_MODES}" \
    WANDB_ENTITY="${WANDB_ENTITY}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    SESSION_PREFIX="expert_eval_${stem}" \
    bash "${RUNNER}"
}

main() {
    if [[ ! -x "${RUNNER}" ]]; then
        echo "ERROR: missing or non-executable runner: ${RUNNER}" >&2
        echo "Expected: ace-g/scripts/run_expert_eval_project_v2.sh" >&2
        return 2
    fi

    local rc=0

    local status_total_tau0 status_none_mogu001 status_total_mogu001
    status_total_tau0="$(find_status_for_stem "${CONFIG_TOTAL_TAU0%.yaml}" "${STATUS_TOTAL_TAU0:-}")" || rc=$?
    status_none_mogu001="$(find_status_for_stem "${CONFIG_NONE_MOGU001%.yaml}" "${STATUS_NONE_MOGU001:-}")" || rc=$?
    status_total_mogu001="$(find_status_for_stem "${CONFIG_TOTAL_MOGU001%.yaml}" "${STATUS_TOTAL_MOGU001:-}")" || rc=$?

    if [[ "${rc}" -ne 0 ]]; then
        echo ""
        echo "Could not resolve all train_status JSON files. Set explicit paths, for example:" >&2
        echo "  STATUS_TOTAL_TAU0=/path/to/total_tau0.json \\" >&2
        echo "  STATUS_NONE_MOGU001=/path/to/none_mogu001.json \\" >&2
        echo "  STATUS_TOTAL_MOGU001=/path/to/total_mogu001.json \\" >&2
        echo "  bash ${REPO_ROOT}/scripts/run_expert_eval_3configs.sh" >&2
        return "${rc}"
    fi

    local failures=0
    if ! run_one "total_tau0" "${CONFIG_TOTAL_TAU0}" "${status_total_tau0}"; then
        failures=$((failures + 1))
        [[ "${CONTINUE_ON_ERROR}" == "1" ]] || return 1
    fi
    if ! run_one "none_mogu001" "${CONFIG_NONE_MOGU001}" "${status_none_mogu001}"; then
        failures=$((failures + 1))
        [[ "${CONTINUE_ON_ERROR}" == "1" ]] || return 1
    fi
    if ! run_one "total_mogu001" "${CONFIG_TOTAL_MOGU001}" "${status_total_mogu001}"; then
        failures=$((failures + 1))
        [[ "${CONTINUE_ON_ERROR}" == "1" ]] || return 1
    fi

    echo ""
    echo "=========================================================="
    echo " Expert eval batch complete. failures=${failures}"
    echo " Summaries are under: ${OUTPUT_ROOT}/summaries"
    echo "=========================================================="
    return "${failures}"
}

main "$@"
