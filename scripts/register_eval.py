#!/usr/bin/env python3
"""Register and evaluate ACE-G scenes, replacing notebook cells 29 and 38.

This script supports both:
- full register + eval
- eval-only using existing <session>_reg.yaml files

It preserves the notebook's per-scene status logic while adding:
- explicit server paths
- clearer process/result logging
- summary CSV/JSON
- optional W&B metric logging and summary artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _common import (
    ScriptError,
    build_subprocess_env,
    default_output_paths,
    ensure_dir,
    expand,
    flatten_eval_res,
    load_eval_yaml,
    maybe_init_wandb,
    pick_img_glob,
    print_scene_result,
    print_stage_header,
    read_json,
    repo_src_dir,
    resolve_scenes,
    run_cmd,
    scene_split_paths,
    slugify,
    summarize_returncodes,
    timestamp_now,
    wandb_finish,
    wandb_log_artifact_file,
    wandb_log_table,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="~/dace/ace-g", help="ACE-G repo root")
    p.add_argument("--dataset-root", default="~/dace/data/ace_datasets/TODO_DATASET_NAME", help="Dataset root containing scene directories")
    p.add_argument("--output-root", default="~/dace/outputs", help="Output root used during training")
    p.add_argument("--cache-root", default="~/dace/cache/ace_cache", help="Cache root placeholder")
    p.add_argument("--torch-home", default="~/dace/cache/ace_cache/torch", help="TORCH_HOME for subprocesses")
    p.add_argument("--train-status-json", required=True, help="Path to train_status JSON written by train_scenes.py")
    p.add_argument("--model-name", default=None, help="Override model name. Default: parsed from status filename")
    p.add_argument("--scenes", nargs="*", default=None, help="Optional subset of scenes to register/evaluate")
    p.add_argument("--skip-register", action="store_true", help="Eval only. Assumes <session>_reg.yaml already exists under output root")

    p.add_argument("--wandb-entity", default="yahav6893")
    p.add_argument("--wandb-project", default="dace")
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--disable-wandb", action="store_true")
    p.add_argument("--wandb-tag", action="append", default=[])

    p.add_argument("--extra-env", action="append", default=[])
    p.add_argument("--pythonpath-prepend", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def infer_model_name(train_status_json: Path) -> str:
    stem = train_status_json.stem
    if stem.endswith("__train_status"):
        stem = stem[: -len("__train_status")]
    # usually model__dataset, but model itself may already contain underscores.
    return stem.split("__")[0] if "__" in stem else stem


def main() -> int:
    args = parse_args()

    repo_root = expand(args.repo_root)
    dataset_root = expand(args.dataset_root)
    output_dirs = default_output_paths(args.output_root)
    train_status_json = expand(args.train_status_json)
    train_status = read_json(train_status_json)
    if not isinstance(train_status, dict):
        raise ScriptError(f"Expected scene-keyed dict in train status JSON: {train_status_json}")

    model_name = args.model_name or infer_model_name(train_status_json)
    dataset_name = dataset_root.name
    scenes = resolve_scenes(dataset_root, args.scenes) if args.scenes else list(train_status.keys())
    wandb_group = args.wandb_group or f"{slugify(model_name)}__{slugify(dataset_name)}"

    print_stage_header("REGISTER + EVAL")
    print(f"repo_root         : {repo_root}")
    print(f"dataset_root      : {dataset_root}")
    print(f"output_root       : {output_dirs['root']}")
    print(f"train_status_json : {train_status_json}")
    print(f"model_name        : {model_name}")
    print(f"dataset_name      : {dataset_name}")
    print(f"scenes            : {scenes}")
    print(f"skip_register     : {args.skip_register}")
    print(f"cache_root        : {expand(args.cache_root)}")
    print(f"torch_home        : {expand(args.torch_home)}")

    _ = repo_src_dir(repo_root)
    env = build_subprocess_env(
        repo_root=repo_root,
        torch_home=args.torch_home,
        extra_env=args.extra_env,
        prepend_pythonpath=args.pythonpath_prepend,
    )

    run_name = f"register-eval__{slugify(model_name)}__{slugify(dataset_name)}__{timestamp_now()}"
    wb_run = maybe_init_wandb(
        enabled=not args.disable_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="register_eval",
        name=run_name,
        group=wandb_group,
        tags=[*args.wandb_tag, dataset_name, model_name, "register_eval"],
        config={
            "repo_root": str(repo_root),
            "dataset_root": str(dataset_root),
            "output_root": str(output_dirs['root']),
            "cache_root": str(expand(args.cache_root)),
            "torch_home": str(expand(args.torch_home)),
            "train_status_json": str(train_status_json),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "scenes": scenes,
            "skip_register": args.skip_register,
        },
    )

    rows: list[dict] = []
    status_rows: dict[str, dict] = {}

    metrics_dir = ensure_dir(output_dirs["metrics"])

    for idx, scene in enumerate(scenes, start=1):
        if scene not in train_status:
            raise ScriptError(f"Scene '{scene}' not present in train status JSON: {train_status_json}")

        st = train_status[scene]
        if int(st.get("returncode", 1)) != 0:
            row = {
                "scene": scene,
                "session_id": st.get("session_id"),
                "stage": "train_precheck",
                "returncode": int(st.get("returncode", 1)),
                "note": "skipped because training failed",
            }
            rows.append(row)
            status_rows[scene] = row
            print(f"[SKIP] scene={scene} because training rc={row['returncode']}")
            continue

        session_id = st["session_id"]
        scene_test = scene_split_paths(dataset_root, scene, "test")
        map_yaml = expand(st["map_yaml"])
        reg_yaml = output_dirs["root"] / f"{session_id}_reg.yaml"

        reg_result = None
        reg_log = output_dirs["logs"] / f"{session_id}__register.log"
        eval_log = output_dirs["logs"] / f"{session_id}__eval.log"

        if not args.skip_register:
            cmd_reg = [
                sys.executable,
                "-m",
                "ace_g.register_images",
                "--config",
                str(map_yaml),
                "--dataset.rgb_files",
                scene_test["rgb"],
                "--dataset.calibration_files",
                scene_test["calibration"],
                "--output_dir",
                str(output_dirs["root"]),
            ]
            print_stage_header(f"REGISTER [{idx}/{len(scenes)}] scene={scene} session={session_id}")
            print("cmd:", " ".join(cmd_reg))
            print("log:", reg_log)
            if args.dry_run:
                reg_rc = -999
            else:
                reg_result = run_cmd(cmd_reg, log_path=reg_log, cwd=repo_root, env=env)
                reg_rc = reg_result.returncode
                print_scene_result("REGISTER", scene, session_id, reg_result)

            if reg_rc != 0 or (not args.dry_run and not reg_yaml.is_file()):
                row = {
                    "scene": scene,
                    "session_id": session_id,
                    "stage": "register",
                    "returncode": reg_rc,
                    "register_log": str(reg_log),
                    "eval_log": None,
                    "reg_yaml": str(reg_yaml),
                    "eval_yaml": None,
                }
                rows.append(row)
                status_rows[scene] = row
                continue
        else:
            if not args.dry_run and not reg_yaml.is_file():
                raise ScriptError(f"--skip-register was given, but reg yaml is missing: {reg_yaml}")

        eval_yaml = output_dirs["root"] / f"{session_id}_eval.yaml"
        cmd_eval = [
            sys.executable,
            "-m",
            "ace_g.eval_poses",
            "--config",
            str(reg_yaml),
            "--gt_pose_files",
            scene_test["poses"],
            "--output_dir",
            str(metrics_dir),
        ]
        print_stage_header(f"EVAL [{idx}/{len(scenes)}] scene={scene} session={session_id}")
        print("cmd:", " ".join(cmd_eval))
        print("log:", eval_log)
        if args.dry_run:
            eval_rc = -999
            eval_metrics = {}
            eval_result = None
        else:
            eval_result = run_cmd(cmd_eval, log_path=eval_log, cwd=repo_root, env=env)
            eval_rc = eval_result.returncode
            print_scene_result("EVAL", scene, session_id, eval_result)
            eval_metrics = flatten_eval_res(load_eval_yaml(eval_yaml)) if eval_rc == 0 and eval_yaml.is_file() else {}

        row = {
            "scene": scene,
            "session_id": session_id,
            "stage": "eval",
            "returncode": eval_rc,
            "register_log": str(reg_log),
            "eval_log": str(eval_log),
            "reg_yaml": str(reg_yaml),
            "eval_yaml": str(eval_yaml),
        }
        row.update(eval_metrics)
        rows.append(row)
        status_rows[scene] = row

    summary_csv = write_csv(output_dirs["summaries"] / f"{slugify(model_name)}__{slugify(dataset_name)}__eval_summary.csv", rows)
    summary_json = write_json(output_dirs["summaries"] / f"{slugify(model_name)}__{slugify(dataset_name)}__eval_summary.json", rows)

    print_stage_header("REGISTER + EVAL SUMMARY")
    print(json.dumps(rows, indent=2))
    print(f"summary_csv : {summary_csv}")
    print(f"summary_json: {summary_json}")

    if wb_run is not None:
        wandb_log_table(wb_run, "eval/summary_table", rows)
        wandb_log_artifact_file(
            wb_run,
            path=summary_csv,
            artifact_name=f"eval-summary-csv__{model_name}__{dataset_name}",
            artifact_type="eval-summary",
            metadata={"model_name": model_name, "dataset_name": dataset_name, "format": "csv"},
        )
        wandb_log_artifact_file(
            wb_run,
            path=summary_json,
            artifact_name=f"eval-summary-json__{model_name}__{dataset_name}",
            artifact_type="eval-summary",
            metadata={"model_name": model_name, "dataset_name": dataset_name, "format": "json"},
        )
        # Also log each eval yaml, when present, for reproducibility.
        for row in rows:
            eval_yaml = row.get("eval_yaml")
            if eval_yaml and Path(eval_yaml).is_file() and int(row.get("returncode", 1)) == 0:
                wandb_log_artifact_file(
                    wb_run,
                    path=eval_yaml,
                    artifact_name=f"eval-yaml__{row['session_id']}",
                    artifact_type="eval-yaml",
                    metadata={
                        "scene": row.get("scene"),
                        "session_id": row.get("session_id"),
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                    },
                )
        wandb_finish(
            wb_run,
            summary_updates={
                "eval/model_name": model_name,
                "eval/dataset_name": dataset_name,
                **{f"eval/{k}": v for k, v in summarize_returncodes(rows).items()},
                "eval/summary_csv": str(summary_csv),
                "eval/summary_json": str(summary_json),
            },
        )

    failures = [r for r in rows if int(r.get("returncode", 1)) != 0 and r.get("returncode") != -999]
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScriptError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
