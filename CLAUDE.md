# CLAUDE.md — ace-g

Guidance for working in this repo. Keep it accurate as the code evolves.

## What this is
A fork of Niantic's **ACE-G** (scene coordinate regression for visual relocalization, ICCV 2025),
extended into a **Mixture-of-Experts + uncertainty** research project. Upstream flow is
map → relocalize → eval. This fork adds a layer that fuses multiple expert encoders
(e.g. DINOv2-reg and a DPT/Depth-Anything-V2 depth encoder).

`~/dace/ace-g` is the only git repo. The parent `~/dace` is a workspace holding non-repo artifacts:
`datasets/`, `outputs/`, `wandb/`, `cache/`, `Depth-Anything-V2/`.

## Two active research tracks
Both are live; know which one a file belongs to before changing it.

1. **MoGU — uncertainty-driven fusion.** `UncExpertFusionHead` in `src/ace_g/scr_heads.py`:
   each expert predicts coords **and** a variance; experts combined by inverse-variance weighting;
   aleatoric/epistemic/total decomposition; MoGU NLL + 2D-reprojection loss; a per-pixel `tau`
   override fed into `ReproLoss` (`src/ace_g/losses.py`). Drivers: `scripts/register_eval_expert_modes.py`,
   `scripts/diagnose_heads_unc.py`, `scripts/run_cambridge_mogu_sweep*.sh`.
2. **Gating-classifier MoE.** `LateFusionCoordHead` (`src/ace_g/scr_heads.py`) +
   `scripts/offline_alpha_test.py` → `scripts/train_gating_classifier.py` →
   `scripts/eval_gating_classifier.py`, wired by `scripts/run_gating_pipeline.sh`.

`DINOv3Encoder` (`src/ace_g/encoders.py`) exists but is **not** a current focus.

## Repo layout
- `src/ace_g/` — the library (installed editable as `ace_g`). Stock ACE-G: `regressors.py`, `buffer.py`,
  `datasets.py`, `data_io.py`, `train_single_scene.py`, `register_images.py`, `eval_poses*.py`,
  `schedule.py`, `vis_*.py`. Fork additions concentrate in `scr_heads.py`, `losses.py`, `encoders.py`,
  and the per-head-LR / dynamic-unfreeze / MoGU-loss / W&B logic in `train_single_scene.py`.
- `src/ace_g/configs/` — upstream configs loaded by **name** via `yoco` (`ace_g_5min.yaml`,
  `ace_g_25min.yaml`, `ace.yaml`, `ace_dino.yaml`).
- `configs_custom/` — hand-written experiment configs (e.g. `MoGU_N2_*.yaml`, `reg_expert*.yaml`).
- `configs_auto/` — machine-generated configs (e.g. from the sweep runner).
- `scripts/` — the server-side experiment harness (not shipped upstream). `_common.py` holds shared
  path/subprocess/W&B/status helpers; `train_scenes.py` and `register_eval.py` are the pipeline stages.
- `dsacstar/` — RANSAC pose solver (`import dsacstar`), installed from `./dsacstar`.

## Environment
Two supported environments — match what the workflow expects:
- **Server scripts** (`scripts/*.sh`, `train_scenes.py`, `register_eval.py`) use a **conda** env:
  `conda activate ~/dace_env310`. They also prepend `~/dace/Depth-Anything-V2` to `PYTHONPATH`
  (`--pythonpath-prepend`) for the depth encoder.
- **Quick/manual** use the pixi env from `pyproject.toml` (`pixi run example`, `pixi shell`), or
  `pip install -e .`.

Other expectations: a GPU with **~55 GB free** (the sweep/pipeline auto-select the freest GPU and
abort below ~56320 MB); `TORCH_HOME=~/dace/cache/torch`; W&B entity `yahav6893`, project `DACE`
(sweeps use `DACE_<scene>`).

## How to run
**Upstream single-scene (manual), from repo root:**
```bash
python -m ace_g.train_single_scene --config ace_g_5min.yaml \
  --dataset.rgb_files ".../rgb/*.jpg" --dataset.pose_files ".../poses/*.txt"
python -m ace_g.register_images --config ".../<session>_map.yaml" --dataset.rgb_files ".../query/*.jpg"
python -m ace_g.eval_poses ...   # see run_eval.sh
```

**Full experiment pipeline (server harness):** `scripts/run_pipeline.sh` runs
`train_scenes.py` → `register_eval.py`. Override via env vars, e.g.:
```bash
CONFIG_NAME=MoGU_N2_frozen_total_tau0_scalar_unc.yaml SCENES="shopfacade" \
  bash scripts/run_pipeline.sh
```
Sweeps: `bash scripts/run_cambridge_mogu_sweep.sh` (use `DRY_RUN=1` first; resumable via
`run.done` markers; writes `outputs/sweeps/mogu_cambridge/manifest.csv`).

## Data & output conventions
- **Dataset:** `~/dace/datasets/cambridge/<scene>/{train,test}/...`; the 5 Cambridge scenes are
  `greatcourt, kingscollege, oldhospital, shopfacade, stmaryschurch`.
- **Dataset label** in output names comes from `dataset_root.name` (so `cambridge`). Scripts that take
  `--dataset-root` should point at the scene-collection dir, not a placeholder.
- **Trained artifacts:** `~/dace/outputs/<model_name>-cambridge-<scene>_{map.yaml,head.pt,reg.yaml,
  registered_poses.txt,eval.txt,eval.yaml,log.txt}`.
- **Pretrained expert heads** (for frozen-expert configs): `~/dace/outputs/heads/` named
  `DINOv2_vitl_reg-aceg_cambridge-<scene>_head.pt` and `dptv2_vitl-aceg_cambridge-<scene>_head.pt`.
- **Pipeline status JSON:** `~/dace/outputs/train_status/*.json` (written by `train_scenes.py`,
  consumed by `register_eval.py` via `--train-status-json`).

## Gotchas / drift watch-list
This repo iterates fast; the recurring failure mode is **stale references after renames**, not bad
math. Before changing or running something, check:
- **Renamed scripts:** `.sh` drivers sometimes point at an old filename. (Fixed:
  `run_eval_gating_classifier.sh` once called the renamed-away `eval_gating_classifier_no_alpha_bins.py`.)
  Grep for the target before assuming a "dead" file is unused — fix the dangling reference rather than
  deleting the file.
- **Expert-head naming:** two conventions exist — `DINOv2_vitl_reg-aceg_cambridge-<scene>` (sweep/heads)
  vs `DINOv2_vitl_reg-cambridge-<scene>` (some gating drivers). Confirm which the file on disk uses.
- **`1cyclepoly` schedule** (`schedule.py`): passes config validation but raises `NotImplementedError`
  at runtime with the new loss — don't select it expecting it to work.
- **`--dataset-root` defaults** were historically a `TODO_DATASET_NAME` placeholder; now default to
  `~/dace/datasets/cambridge`. Still pass it explicitly for non-Cambridge data.

## Conventions
- Lint/format with `ruff` (config in `pyproject.toml`, line length 120). Run on changed files.
- Don't commit large artifacts into `src/` (e.g. `ace_g_source.zip` export blobs).
- Don't commit or push unless asked; if on `main`, branch first.
