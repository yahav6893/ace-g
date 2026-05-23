#!/usr/bin/env bash
set -euo pipefail

# Sweep runner for ACE-G Cambridge MoGU experiments.
# - Generates one YAML config per experiment from BASE_CONFIG_NAME/BASE_CONFIG_PATH.
# - Runs train_scenes.py then register_eval.py.
# - Stores local outputs under ${SWEEP_OUTPUT_ROOT}/${RUN_ID}.
# - Sends each scene to its own W&B project: DACE_<scene>.
#
# Usage:
#   chmod +x run_cambridge_mogu_sweep.sh
#   ./run_cambridge_mogu_sweep.sh
#
# Useful dry run:
#   DRY_RUN=1 ./run_cambridge_mogu_sweep.sh
#
# Resume behavior:
#   Existing run.done markers are skipped.
#   Existing status.json skips training and continues to registration/eval.

# -----------------------------
# User-tunable defaults
# -----------------------------
REPO_ROOT="${REPO_ROOT:-${HOME}/dace/ace-g}"
DATASET_ROOT="${DATASET_ROOT:-${HOME}/dace/datasets/cambridge}"
BASE_CONFIG_NAME="${BASE_CONFIG_NAME:-MoGU_N2_dinov2reg_dpt.yaml}"
BASE_CONFIG_PATH="${BASE_CONFIG_PATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/dace/outputs}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-${OUTPUT_ROOT}/sweeps/mogu_cambridge}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/dace/cache}"
TORCH_HOME="${TORCH_HOME:-${HOME}/dace/cache/torch}"
BASE_HEAD_ROOT="${BASE_HEAD_ROOT:-${OUTPUT_ROOT}/heads}"

CONDA_ENV="${CONDA_ENV:-${HOME}/dace_env310}"
PYTHONPATH_PREPEND="${PYTHONPATH_PREPEND:-${HOME}/dace/Depth-Anything-V2}"
WANDB_ENTITY="${WANDB_ENTITY:-yahav6893}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-DACE}"

# Default Cambridge scenes. The script resolves these case-insensitively against DATASET_ROOT.
# Override with, for example:
#   SCENES="kingscollege oldhospital shopfacade stmaryschurch greatcourt" ./run_cambridge_mogu_sweep.sh
DEFAULT_SCENES=(kingscollege oldhospital shopfacade stmaryschurch greatcourt)
SCENES_STR="${SCENES:-}"

ITERATIONS=(50000 25000 10000)
MOGU_LOSS_WEIGHTS=(1 0.1)
PRETRAINED_MODES=(yes no)
FREEZE_MODES=(yes no)

# Buffer policy requested by the user:
# - 8M is not used for 10000 iterations.
# - 2M is used only for 10000 iterations.
# - 4M is used for all iteration settings.
BUFFERS_LONG=(8000000 4000000)
BUFFERS_SHORT=(4000000 2000000)

# Operational controls.
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
RESUME="${RESUME:-1}"
REQUIRE_PRETRAINED_HEADS="${REQUIRE_PRETRAINED_HEADS:-1}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-56320}"
MAX_RUNS="${MAX_RUNS:-0}"          # 0 means no limit.
START_AT_RUN="${START_AT_RUN:-1}"  # 1-based run index after filtering.

# How to map dataset folder name to pretrained head filename.
# Current uploaded config uses shopfacade in head filenames, so lower is a safe default.
# Use HEAD_SCENE_NAME_MODE=preserve if your head filenames preserve dataset folder case.
HEAD_SCENE_NAME_MODE="${HEAD_SCENE_NAME_MODE:-lower}"

# -----------------------------
# Helpers
# -----------------------------
log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

sanitize() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//; s/_$//'
}

mogu_tag() {
  echo "$1" | sed 's/\./p/g'
}

buffer_tag() {
  case "$1" in
    8000000) echo "8M" ;;
    4000000) echo "4M" ;;
    2000000) echo "2M" ;;
    *) echo "$1" ;;
  esac
}

head_scene_name() {
  local scene="$1"
  case "${HEAD_SCENE_NAME_MODE}" in
    lower) echo "${scene,,}" ;;
    preserve) echo "${scene}" ;;
    *) fail "Unknown HEAD_SCENE_NAME_MODE=${HEAD_SCENE_NAME_MODE}; use lower or preserve." ;;
  esac
}

resolve_scene_dir_name() {
  local requested="$1"
  local exact="${DATASET_ROOT}/${requested}"
  if [[ -d "${exact}" ]]; then
    basename "${exact}"
    return 0
  fi

  local found
  found=$(find "${DATASET_ROOT}" -mindepth 1 -maxdepth 1 -type d -iname "${requested}" -printf '%f\n' | head -n 1 || true)
  if [[ -n "${found}" ]]; then
    echo "${found}"
    return 0
  fi

  return 1
}

pick_scenes() {
  local scenes=()

  if [[ -n "${SCENES_STR}" ]]; then
    read -r -a scenes <<< "${SCENES_STR}"
  else
    # Prefer exactly discovered Cambridge scene folders if present.
    local discovered=()
    while IFS= read -r d; do
      [[ -d "${DATASET_ROOT}/${d}/train" || -d "${DATASET_ROOT}/${d}/test" || -d "${DATASET_ROOT}/${d}/val" ]] || continue
      discovered+=("${d}")
    done < <(find "${DATASET_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort || true)

    if [[ "${#discovered[@]}" -eq 5 ]]; then
      scenes=("${discovered[@]}")
    else
      scenes=("${DEFAULT_SCENES[@]}")
    fi
  fi

  local resolved=()
  for s in "${scenes[@]}"; do
    local r
    if ! r=$(resolve_scene_dir_name "${s}"); then
      fail "Could not resolve scene '${s}' under DATASET_ROOT=${DATASET_ROOT}. Set SCENES explicitly or check folder names."
    fi
    resolved+=("${r}")
  done

  printf '%s\n' "${resolved[@]}"
}

find_base_config() {
  if [[ -n "${BASE_CONFIG_PATH}" ]]; then
    [[ -f "${BASE_CONFIG_PATH}" ]] || fail "BASE_CONFIG_PATH does not exist: ${BASE_CONFIG_PATH}"
    echo "${BASE_CONFIG_PATH}"
    return 0
  fi

  local candidates=(
    "${REPO_ROOT}/configs/${BASE_CONFIG_NAME}"
    "${REPO_ROOT}/config/${BASE_CONFIG_NAME}"
    "${REPO_ROOT}/${BASE_CONFIG_NAME}"
  )
  for p in "${candidates[@]}"; do
    if [[ -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done

  local found
  found=$(find "${REPO_ROOT}" -type f -name "${BASE_CONFIG_NAME}" -print 2>/dev/null | head -n 1 || true)
  [[ -n "${found}" ]] || fail "Could not find ${BASE_CONFIG_NAME} under ${REPO_ROOT}. Set BASE_CONFIG_PATH=/path/to/${BASE_CONFIG_NAME}."
  echo "${found}"
}

select_gpu() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    log "Using pre-set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."
    return 0
  fi

  local best_gpu_info best_gpu_id best_gpu_free
  best_gpu_info=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n 1 || true)
  [[ -n "${best_gpu_info}" ]] || fail "Failed to run nvidia-smi or no GPUs found."

  best_gpu_id=$(echo "${best_gpu_info}" | awk -F, '{print $1}' | xargs)
  best_gpu_free=$(echo "${best_gpu_info}" | awk -F, '{print $2}' | xargs)

  if [[ "${best_gpu_free}" -lt "${MIN_GPU_FREE_MB}" ]]; then
    fail "No GPU has at least ${MIN_GPU_FREE_MB} MB free. Best GPU ${best_gpu_id} has ${best_gpu_free} MB free."
  fi

  export CUDA_VISIBLE_DEVICES="${best_gpu_id}"
  log "Selected GPU ${best_gpu_id} with ${best_gpu_free} MB free memory."
}

prepare_env() {
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"

  unset VIRTUAL_ENV
  hash -r

  export PATH="${CONDA_PREFIX}/bin:/usr/local/bin:/usr/bin:/bin"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$(python -c 'import site, glob; print(glob.glob(site.getsitepackages()[0] + "/torch/lib")[0])')"
  export TORCH_HOME="${TORCH_HOME}"

  python - <<'PY'
try:
    import yaml  # noqa: F401
except Exception as exc:
    raise SystemExit("PyYAML is required to generate configs. Install with: pip install pyyaml") from exc
PY
}

make_wandb_id() {
  python - "$1" <<'PY'
import hashlib, sys
print(hashlib.sha1(sys.argv[1].encode("utf-8")).hexdigest()[:12])
PY
}

write_config() {
  local base_config="$1"
  local out_config="$2"
  local run_output_root="$3"
  local iterations="$4"
  local buffer_size="$5"
  local pretrained="$6"
  local freeze="$7"
  local scene="$8"
  local mogu_weight="$9"
  local head_scene="${10}"

  python - \
    "${base_config}" \
    "${out_config}" \
    "${run_output_root}" \
    "${iterations}" \
    "${buffer_size}" \
    "${pretrained}" \
    "${freeze}" \
    "${scene}" \
    "${mogu_weight}" \
    "${head_scene}" \
    "${BASE_HEAD_ROOT}" <<'PY'
import os
import sys
import yaml

(
    base_config,
    out_config,
    run_output_root,
    iterations,
    buffer_size,
    pretrained,
    freeze,
    scene,
    mogu_weight,
    head_scene,
    base_head_root,
) = sys.argv[1:]

with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if not isinstance(cfg, dict):
    raise SystemExit(f"Base config did not parse to a dict: {base_config}")

sst = cfg.setdefault("sst", {})
head = sst.setdefault("head", {})

sst["output_dir"] = run_output_root
sst["num_iterations"] = int(iterations)
sst["max_buffer_size"] = int(buffer_size)
head["mogu_loss_weight"] = float(mogu_weight)

if pretrained == "yes":
    expert_paths = [
        os.path.join(base_head_root, f"DINOv2_vitl_reg-aceg_cambridge-{head_scene}_head.pt"),
        os.path.join(base_head_root, f"dptv2_vitl-aceg_cambridge-{head_scene}_head.pt"),
    ]
    head["expert_head_paths"] = expert_paths
    head["freeze_loaded_experts"] = freeze == "yes"
else:
    # This assumes UncExpertFusionHead can initialize experts from `expert_head`
    # when no pretrained paths are supplied. If your implementation requires the
    # key to be absent instead of an empty list, change this to: head.pop("expert_head_paths", None)
    head["expert_head_paths"] = []
    head["freeze_loaded_experts"] = False

# Add lightweight metadata that should be ignored by normal config consumers but
# makes generated YAMLs self-describing.
cfg["sweep"] = {
    "scene": scene,
    "iterations": int(iterations),
    "max_buffer_size": int(buffer_size),
    "pretrained_heads": pretrained == "yes",
    "freeze_loaded_experts": (freeze == "yes") if pretrained == "yes" else None,
    "mogu_loss_weight": float(mogu_weight),
}

os.makedirs(os.path.dirname(out_config), exist_ok=True)
with open(out_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

validate_heads() {
  local pretrained="$1"
  local head_scene="$2"
  [[ "${pretrained}" == "yes" ]] || return 0
  [[ "${REQUIRE_PRETRAINED_HEADS}" == "1" ]] || return 0

  local dino_head="${BASE_HEAD_ROOT}/DINOv2_vitl_reg-aceg_cambridge-${head_scene}_head.pt"
  local dpt_head="${BASE_HEAD_ROOT}/dptv2_vitl-aceg_cambridge-${head_scene}_head.pt"

  [[ -f "${dino_head}" ]] || fail "Missing pretrained DINO head: ${dino_head}"
  [[ -f "${dpt_head}" ]] || fail "Missing pretrained DPT head: ${dpt_head}"
}

run_pipeline_once() {
  local run_id="$1"
  local scene="$2"
  local config_name="$3"
  local run_output_root="$4"
  local status_json_path="$5"
  local wandb_project="$6"
  local log_dir="$7"

  mkdir -p "${run_output_root}" "${log_dir}"

  local done_marker="${run_output_root}/run.done"
  local fail_marker="${run_output_root}/run.failed"

  if [[ "${RESUME}" == "1" && -f "${done_marker}" ]]; then
    log "SKIP done: ${run_id}"
    return 0
  fi

  export WANDB_PROJECT="${wandb_project}"
  export WANDB_NAME="${run_id}"
  export WANDB_GROUP="${scene}"
  export WANDB_RUN_ID
  WANDB_RUN_ID=$(make_wandb_id "${run_id}")
  export WANDB_RESUME="allow"

  log "=========================================================="
  log "RUN ${run_id}"
  log "Scene=${scene} W&B=${WANDB_PROJECT}/${WANDB_NAME} Output=${run_output_root}"
  log "=========================================================="

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1, not executing train/register."
    return 0
  fi

  select_gpu

  rm -f "${fail_marker}"

  if [[ "${RESUME}" == "1" && -f "${status_json_path}" ]]; then
    log "Found existing train status; skipping training: ${status_json_path}"
  else
    log ">>> STAGE 1: TRAINING"
    if ! python "${REPO_ROOT}/scripts/train_scenes.py" \
      --repo-root "${REPO_ROOT}" \
      --dataset-root "${DATASET_ROOT}" \
      --config-name "${config_name}" \
      --scenes "${scene}" \
      --output-root "${run_output_root}" \
      --cache-root "${CACHE_ROOT}" \
      --torch-home "${TORCH_HOME}" \
      --wandb-entity "${WANDB_ENTITY}" \
      --wandb-project "${WANDB_PROJECT}" \
      --train-status-name "status.json" \
      --pythonpath-prepend "${PYTHONPATH_PREPEND}" \
      --promote-heads 2>&1 | tee "${log_dir}/${run_id}.train.log"; then
        echo "training failed" > "${fail_marker}"
        return 1
    fi
  fi

  [[ -f "${status_json_path}" ]] || {
    echo "missing status json: ${status_json_path}" > "${fail_marker}"
    return 1
  }

  log ">>> STAGE 2: REGISTRATION & EVALUATION"
  if ! python "${REPO_ROOT}/scripts/register_eval.py" \
    --repo-root "${REPO_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${run_output_root}" \
    --cache-root "${CACHE_ROOT}" \
    --torch-home "${TORCH_HOME}" \
    --train-status-json "${status_json_path}" \
    --scenes "${scene}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --pythonpath-prepend "${PYTHONPATH_PREPEND}" 2>&1 | tee "${log_dir}/${run_id}.register_eval.log"; then
      echo "registration/evaluation failed" > "${fail_marker}"
      return 1
  fi

  date '+%Y-%m-%d %H:%M:%S' > "${done_marker}"
  log "DONE: ${run_id}"
}

# -----------------------------
# Main
# -----------------------------
prepare_env

BASE_CONFIG_PATH_RESOLVED=$(find_base_config)
CONFIG_DIR=$(dirname "${BASE_CONFIG_PATH_RESOLVED}")
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-${CONFIG_DIR}}"
mkdir -p "${GENERATED_CONFIG_DIR}" "${SWEEP_OUTPUT_ROOT}"

mapfile -t SCENE_LIST < <(pick_scenes)
log "Using base config: ${BASE_CONFIG_PATH_RESOLVED}"
log "Writing generated configs to: ${GENERATED_CONFIG_DIR}"
log "Scenes: ${SCENE_LIST[*]}"

MANIFEST="${SWEEP_OUTPUT_ROOT}/manifest.csv"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "run_index,run_id,scene,wandb_project,config_path,output_root,status_json,iterations,max_buffer_size,pretrained_heads,freeze_loaded_experts,mogu_loss_weight" > "${MANIFEST}"
fi

run_index=0
started_count=0
failed_count=0

for scene in "${SCENE_LIST[@]}"; do
  scene_tag=$(sanitize "${scene}")
  scene_for_head=$(head_scene_name "${scene}")
  wandb_project="${WANDB_PROJECT_PREFIX}_${scene}"

  for iterations in "${ITERATIONS[@]}"; do
    if [[ "${iterations}" == "10000" ]]; then
      buffers=("${BUFFERS_SHORT[@]}")
    else
      buffers=("${BUFFERS_LONG[@]}")
    fi

    for buffer_size in "${buffers[@]}"; do
      for pretrained in "${PRETRAINED_MODES[@]}"; do
        if [[ "${pretrained}" == "yes" ]]; then
          freeze_values=("${FREEZE_MODES[@]}")
        else
          freeze_values=(na)
        fi

        for freeze in "${freeze_values[@]}"; do
          for mogu_weight in "${MOGU_LOSS_WEIGHTS[@]}"; do
            run_index=$((run_index + 1))
            if [[ "${run_index}" -lt "${START_AT_RUN}" ]]; then
              continue
            fi
            if [[ "${MAX_RUNS}" -gt 0 && "${started_count}" -ge "${MAX_RUNS}" ]]; then
              log "Reached MAX_RUNS=${MAX_RUNS}."
              log "Total enumerated run count so far: ${run_index}."
              log "Failures in executed subset: ${failed_count}."
              exit 0
            fi

            pre_tag=$([[ "${pretrained}" == "yes" ]] && echo "pre" || echo "scratch")
            if [[ "${pretrained}" == "yes" ]]; then
              freeze_tag=$([[ "${freeze}" == "yes" ]] && echo "frozen" || echo "unfrozen")
            else
              freeze_tag="nofreeze"
            fi
            buf_tag=$(buffer_tag "${buffer_size}")
            mlw_tag=$(mogu_tag "${mogu_weight}")

            run_id="mogu_${scene_tag}_it${iterations}_buf${buf_tag}_${pre_tag}_${freeze_tag}_mlw${mlw_tag}"
            run_output_root="${SWEEP_OUTPUT_ROOT}/${run_id}"
            config_path="${GENERATED_CONFIG_DIR}/${run_id}.yaml"
            config_name="$(basename "${config_path}")"
            status_json_path="${run_output_root}/train_status/status.json"
            log_dir="${run_output_root}/logs"

            validate_heads "${pretrained}" "${scene_for_head}"
            write_config \
              "${BASE_CONFIG_PATH_RESOLVED}" \
              "${config_path}" \
              "${run_output_root}" \
              "${iterations}" \
              "${buffer_size}" \
              "${pretrained}" \
              "${freeze}" \
              "${scene}" \
              "${mogu_weight}" \
              "${scene_for_head}"

            echo "${run_index},${run_id},${scene},${wandb_project},${config_path},${run_output_root},${status_json_path},${iterations},${buffer_size},${pretrained},${freeze},${mogu_weight}" >> "${MANIFEST}"
            started_count=$((started_count + 1))

            if ! run_pipeline_once \
              "${run_id}" \
              "${scene}" \
              "${config_name}" \
              "${run_output_root}" \
              "${status_json_path}" \
              "${wandb_project}" \
              "${log_dir}"; then
                failed_count=$((failed_count + 1))
                log "FAILED: ${run_id}"
                if [[ "${STOP_ON_FAIL}" == "1" ]]; then
                  exit 1
                fi
            fi
          done
        done
      done
    done
  done
done

log "Sweep complete. Started/examined=${started_count}, failed=${failed_count}, manifest=${MANIFEST}"
