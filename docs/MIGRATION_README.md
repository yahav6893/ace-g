# DPTv2_ACE.ipynb -> server script conversion

This bundle converts the notebook workflows that matter into explicit server-side scripts.

## Converted notebook processes

### 1. Train pipeline
Notebook cells: **25-26**

Converted to:
- `scripts/train_scenes.py`

What it does:
- validates repo/config/dataset paths
- discovers or selects scenes
- runs `python -m ace_g.train_single_scene` per scene
- writes per-scene logs
- writes `train_status.json`
- optionally promotes found `<session>_head.pt`
- logs one W&B run for the script invocation
- logs one model artifact per successful scene head

### 2. Register + eval pipeline
Notebook cells: **29 + 38**

Converted to:
- `scripts/register_eval.py`

What it does:
- reads `train_status.json`
- optionally runs `python -m ace_g.register_images`
- runs `python -m ace_g.eval_poses`
- parses `<session>_eval.yaml`
- writes CSV + JSON summaries
- logs one W&B run for the script invocation
- logs eval summary artifacts

### 3. Head / fusion diagnostics
Notebook cells: **31-35**

Converted to:
- `scripts/diagnose_heads.py`

Subcommands:
- `compare-means`
- `compare-logs`
- `gate-weights`
- `gate-last-layer`
- `print-config`

## Not converted on purpose

- Colab / Drive mount / `/content` bootstrap cells
- config generator cells (assuming the YAMLs already exist in `~/dace/ace-g/configs_custom`)
- indoor6 rebuild / repair cells
- Cambridge setup cell
- ad hoc import-debug cells (their useful env/path logic was folded into the scripts)

## Config flag from the notebook

Config cell **15** is **not** a writer cell. It points at an existing config path instead of writing a YAML.

So the migration assumes the real configs already exist under:
- `~/dace/ace-g/configs_custom`

If a required YAML is missing, the scripts stop immediately.

## Expected server layout

```text
~/dace/
  ace-g/
  data/
    ace_datasets/
  cache/
    ace_cache/
  outputs/
    logs/
    runs/
    summaries/
    promoted/
    metrics/
    train_status/
```

## W&B assumptions

- entity: `yahav6893`
- project default: `dace`
- one W&B run per script invocation
- best/final per-scene `.pt` logged as W&B Artifacts from `train_scenes.py`

## Placeholders intentionally left for you to fill in later

### Config placeholders
Use either:
- `--config-name TODO_CONFIG.yaml`
- or `--config-path /absolute/path/to/config.yaml`

### Dataset placeholders
Default dataset-root placeholder:
- `~/dace/data/ace_datasets/TODO_DATASET_NAME`

### Cache placeholders
Default cache-root placeholder:
- `~/dace/cache/ace_cache`

Default torch-home placeholder:
- `~/dace/cache/ace_cache/torch`

If your encoders/checkpoints depend on additional cache paths, pass them via:
- config YAML paths
- `--extra-env KEY=VALUE`
- `--pythonpath-prepend PATH`

## First-run checklist

1. Copy datasets to `~/dace/data/ace_datasets`
2. Copy cache to `~/dace/cache/ace_cache`
3. Confirm configs exist under `~/dace/ace-g/configs_custom`
4. Activate your environment
5. Run a one-scene dry run
6. Run real training
7. Run register + eval
8. Run diagnostics only when needed

## Example commands

### Dry-run train
```bash
python scripts/train_scenes.py \
  --repo-root ~/dace/ace-g \
  --dataset-root ~/dace/data/ace_datasets/TODO_DATASET_NAME \
  --config-name TODO_CONFIG.yaml \
  --scenes TODO_SCENE \
  --dry-run
```

### Real train
```bash
python scripts/train_scenes.py \
  --repo-root ~/dace/ace-g \
  --dataset-root ~/dace/data/ace_datasets/TODO_DATASET_NAME \
  --config-name TODO_CONFIG.yaml \
  --scenes TODO_SCENE \
  --promote-heads
```

### Register + eval
```bash
python scripts/register_eval.py \
  --repo-root ~/dace/ace-g \
  --dataset-root ~/dace/data/ace_datasets/TODO_DATASET_NAME \
  --train-status-json ~/dace/outputs/train_status/TODO_MODEL__TODO_DATASET__train_status.json
```

### Eval only
```bash
python scripts/register_eval.py \
  --repo-root ~/dace/ace-g \
  --dataset-root ~/dace/data/ace_datasets/TODO_DATASET_NAME \
  --train-status-json ~/dace/outputs/train_status/TODO_MODEL__TODO_DATASET__train_status.json \
  --skip-register
```

### Compare head means
```bash
python scripts/diagnose_heads.py compare-means \
  --repo-root ~/dace/ace-g \
  --head-a ~/dace/outputs/TODO_A_head.pt \
  --head-b ~/dace/outputs/TODO_B_head.pt
```

### Gate weights probe
```bash
python scripts/diagnose_heads.py gate-weights \
  --repo-root ~/dace/ace-g \
  --config-path ~/dace/ace-g/configs_custom/TODO_FUSION.yaml \
  --head-path ~/dace/outputs/TODO_head.pt \
  --rgb-glob '~/dace/data/ace_datasets/TODO_DATASET/TODO_SCENE/test/rgb/*.png' \
  --num-images 25 \
  --max-side 640 \
  --use-half
```

## Notes on process/results logging

The converted scripts are intentionally explicit about processes and results:
- each subprocess command is written at the top of its log file
- logs record start time, finish time, return code, and duration
- train status is serialized as JSON
- eval results are serialized as CSV + JSON
- console output reports scene, session_id, rc, duration, and log path

This is meant to make server execution easier to debug than the notebook.
