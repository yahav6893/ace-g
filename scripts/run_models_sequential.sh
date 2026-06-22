#!/usr/bin/env bash
# Launch run_pipeline.sh for each config in sequence, waiting between launches until each run has
# actually OCCUPIED a GPU (its selected GPU's free memory drops below the threshold). Because
# run_pipeline.sh auto-selects the freest GPU, this guarantees consecutive runs land on different
# GPUs instead of racing for the same one.
#
# Usage:
#   bash scripts/run_models_sequential.sh [config1.yaml config2.yaml ...]
#   (no args -> defaults to the offline-runnable new single-expert configs)
#
# Env overrides: REPO_ROOT, OUTPUT_ROOT, MIN_FREE_MB (GPU "free" threshold), MAX_WAIT_S.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-${HOME}/dace/ace-g}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/dace/outputs}"
MIN_FREE_MB="${MIN_FREE_MB:-56320}"     # same threshold run_pipeline.sh uses to pick a GPU
MAX_WAIT_S="${MAX_WAIT_S:-1200}"        # max seconds to wait for one run to occupy a GPU
LOGDIR="${OUTPUT_ROOT}/logs/sequential"
mkdir -p "${LOGDIR}"

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(FiT3D_dinov2.yaml SemanticDINOv2_vitl.yaml)
fi

gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

echo "=========================================================="
echo " Sequential model launcher"
echo " configs : ${CONFIGS[*]}"
echo " logdir  : ${LOGDIR}"
echo " min_free: ${MIN_FREE_MB} MB   max_wait: ${MAX_WAIT_S}s"
echo "=========================================================="

for cfg in "${CONFIGS[@]}"; do
  ts="$(date +%Y%m%d_%H%M%S)"
  log="${LOGDIR}/${cfg%.yaml}_${ts}.log"
  echo "[$(date +%T)] launching ${cfg} -> ${log}"
  CONFIG_NAME="${cfg}" nohup bash "${REPO_ROOT}/scripts/run_pipeline.sh" >"${log}" 2>&1 &
  pid=$!

  deadline=$(( $(date +%s) + MAX_WAIT_S ))
  occupied=0
  while [ "$(date +%s)" -lt "${deadline}" ]; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[$(date +%T)]   WARNING: ${cfg} (pid ${pid}) exited early; check ${log}. Moving on."
      occupied=1; break
    fi
    gpu="$(grep -oE 'Selected GPU [0-9]+' "${log}" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)"
    if [ -n "${gpu:-}" ]; then
      free="$(gpu_free_mb "${gpu}")"
      if [ -n "${free}" ] && [ "${free}" -lt "${MIN_FREE_MB}" ]; then
        echo "[$(date +%T)]   ${cfg} occupied GPU ${gpu} (free=${free} MB). Proceeding to next."
        occupied=1; break
      fi
    fi
    sleep 5
  done
  [ "${occupied}" -eq 0 ] && echo "[$(date +%T)]   timeout (${MAX_WAIT_S}s) waiting for ${cfg} to occupy a GPU; proceeding anyway."
done

echo "[$(date +%T)] all launches issued; runs continue in background. Logs: ${LOGDIR}"
