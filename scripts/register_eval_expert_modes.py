#!/usr/bin/env python3
"""Register and evaluate ACE-G scenes for fused / forced-expert modes.

This is an eval-only companion for MoGU expert-fusion debugging. It uses the
existing train_status JSON to find each trained head/map, then runs:

  fused   -> normal head output
  expert0 -> UncExpertFusionHead with sanity_check_force_expert=0
  expert1 -> UncExpertFusionHead with sanity_check_force_expert=1

For each mode it writes a separate *_reg.yaml and *_eval.yaml and logs W&B
metrics under non-colliding prefixes such as eval/fused/*, eval/expert0/*,
and eval/expert1/*.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _common import (
    ScriptError,
    build_subprocess_env,
    config_path_from_args,
    default_output_paths,
    ensure_dir,
    expand,
    flatten_eval_res,
    load_eval_yaml,
    maybe_init_wandb,
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


EXPERT_MODES = ("fused", "expert0", "expert1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="~/dace/ace-g", help="ACE-G repo root")
    p.add_argument("--dataset-root", default="~/dace/datasets/cambridge", help="Dataset root containing scene directories")
    p.add_argument("--output-root", default="~/dace/outputs", help="Output root used during training")
    p.add_argument("--cache-root", default="~/dace/cache/ace_cache", help="Cache root placeholder")
    p.add_argument("--torch-home", default="~/dace/cache/ace_cache/torch", help="TORCH_HOME for subprocesses")
    p.add_argument("--train-status-json", required=True, help="Path to train_status JSON written by train_scenes.py")
    p.add_argument("--model-name", default=None, help="Override model name. Default: parsed from config filename")
    p.add_argument("--scenes", nargs="*", default=None, help="Optional subset of scenes to register/evaluate")
    p.add_argument("--skip-register", action="store_true", help="Eval only. Assumes mode-specific <session>__<mode>_reg.yaml files already exist")
    p.add_argument("--session-prefix", default=None, help="Optional prefix before '<model>-<dataset>-<scene>'")

    p.add_argument("--expert-eval-modes", nargs="+", default=list(EXPERT_MODES), choices=EXPERT_MODES,
                   help="Modes to run. Default: fused expert0 expert1")
    p.add_argument("--wandb-eval-prefix-root", default="eval",
                   help="Root prefix for eval metrics. Mode is appended, e.g. eval/fused")

    p.add_argument("--wandb-entity", default="yahav6893")
    p.add_argument("--wandb-project", default="dace")
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--disable-wandb", action="store_true")
    p.add_argument("--wandb-tag", action="append", default=[])

    p.add_argument("--extra-env", action="append", default=[])
    p.add_argument("--pythonpath-prepend", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")

    group_cfg = p.add_mutually_exclusive_group(required=True)
    group_cfg.add_argument("--config-name", help="Config filename under <repo>/configs_custom/")
    group_cfg.add_argument("--config-path", help="Absolute config path")

    return p.parse_args()


def force_expert_for_mode(mode: str) -> int | None:
    if mode == "fused":
        return None
    if mode.startswith("expert"):
        return int(mode.removeprefix("expert"))
    raise ValueError(f"Unsupported expert eval mode: {mode}")


def infer_model_name_from_status(train_status_json: Path) -> str:
    stem = train_status_json.stem
    if stem.endswith("__train_status"):
        stem = stem[: -len("__train_status")]
    return stem.split("__")[0] if "__" in stem else stem


def metric_prefix(root: str, mode: str) -> str:
    return f"{root.strip('/')}/{mode}".strip("/")


def main() -> int:
    args = parse_args()

    repo_root = expand(args.repo_root)
    dataset_root = expand(args.dataset_root)
    output_dirs = default_output_paths(args.output_root)
    train_status_json = expand(args.train_status_json)
    train_status = read_json(train_status_json)
    if not isinstance(train_status, dict):
        raise ScriptError(f"Expected scene-keyed dict in train status JSON: {train_status_json}")

    config_path = config_path_from_args(repo_root, args.config_name, args.config_path)
    model_name = args.model_name or config_path.stem or infer_model_name_from_status(train_status_json)
    dataset_name = dataset_root.name
    scenes = resolve_scenes(dataset_root, args.scenes) if args.scenes else list(train_status.keys())
    modes = list(args.expert_eval_modes)
    wandb_group = args.wandb_group or f"{slugify(model_name)}__{slugify(dataset_name)}__expert_eval"

    print_stage_header("EXPERT EVAL: REGISTER + EVAL")
    print(f"repo_root         : {repo_root}")
    print(f"dataset_root      : {dataset_root}")
    print(f"output_root       : {output_dirs['root']}")
    print(f"train_status_json : {train_status_json}")
    print(f"config_path       : {config_path}")
    print(f"model_name        : {model_name}")
    print(f"dataset_name      : {dataset_name}")
    print(f"scenes            : {scenes}")
    print(f"modes             : {modes}")
    print(f"skip_register     : {args.skip_register}")
    print(f"cache_root        : {expand(args.cache_root)}")
    print(f"torch_home        : {expand(args.torch_home)}")

    _ = repo_src_dir(repo_root)
    env = build_subprocess_env(
        repo_root=repo_root,
        torch_home=args.torch_home,
        extra_env=args.extra_env + [
            f"WANDB_PROJECT={args.wandb_project}",
            f"WANDB_ENTITY={args.wandb_entity}",
        ],
        prepend_pythonpath=args.pythonpath_prepend,
    )

    run_prefix = f"{slugify(args.session_prefix)}__" if args.session_prefix else ""
    run_name = f"expert-eval__{run_prefix}{slugify(model_name)}__{slugify(dataset_name)}__{timestamp_now()}"
    wb_run = maybe_init_wandb(
        enabled=not args.disable_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="expert_eval",
        name=run_name,
        group=wandb_group,
        tags=[*args.wandb_tag, dataset_name, model_name, "expert_eval"],
        config={
            "repo_root": str(repo_root),
            "dataset_root": str(dataset_root),
            "output_root": str(output_dirs['root']),
            "cache_root": str(expand(args.cache_root)),
            "torch_home": str(expand(args.torch_home)),
            "train_status_json": str(train_status_json),
            "config_path": str(config_path),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "scenes": scenes,
            "modes": modes,
            "skip_register": args.skip_register,
            "session_prefix": args.session_prefix,
        },
    )

    rows: list[dict] = []
    metrics_dir = ensure_dir(output_dirs["metrics"])

    for scene_idx, scene in enumerate(scenes, start=1):
        if scene not in train_status:
            raise ScriptError(f"Scene '{scene}' not present in train status JSON: {train_status_json}")

        st = train_status[scene]
        if int(st.get("returncode", 1)) != 0:
            row = {
                "scene": scene,
                "mode": None,
                "session_id": st.get("session_id"),
                "stage": "train_precheck",
                "returncode": int(st.get("returncode", 1)),
                "note": "skipped because training failed",
            }
            rows.append(row)
            print(f"[SKIP] scene={scene} because training rc={row['returncode']}")
            continue

        base_session_id = st["session_id"]
        scene_test = scene_split_paths(dataset_root, scene, "test")
        map_yaml = expand(st["map_yaml"])

        for mode_idx, mode in enumerate(modes, start=1):
            force_expert = force_expert_for_mode(mode)
            mode_session_id = f"{base_session_id}__{mode}"
            eval_prefix = metric_prefix(args.wandb_eval_prefix_root, mode)

            reg_yaml = output_dirs["root"] / f"{mode_session_id}_reg.yaml"
            eval_yaml = metrics_dir / f"{mode_session_id}_eval.yaml"
            reg_log = output_dirs["logs"] / f"{mode_session_id}__register.log"
            eval_log = output_dirs["logs"] / f"{mode_session_id}__eval.log"

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
                    "--session_id",
                    mode_session_id,
                ]
                if force_expert is not None:
                    cmd_reg.extend(["--sanity_check_force_expert", str(force_expert)])

                print_stage_header(
                    f"REGISTER [{scene_idx}/{len(scenes)} scene] [{mode_idx}/{len(modes)} mode] "
                    f"scene={scene} mode={mode} session={mode_session_id}"
                )
                print("cmd:", " ".join(cmd_reg))
                print("log:", reg_log)
                if args.dry_run:
                    reg_rc = -999
                    reg_result = None
                else:
                    reg_result = run_cmd(cmd_reg, log_path=reg_log, cwd=repo_root, env=env)
                    reg_rc = reg_result.returncode
                    print_scene_result("REGISTER", scene, mode_session_id, reg_result)

                if reg_rc != 0 or (not args.dry_run and not reg_yaml.is_file()):
                    row = {
                        "scene": scene,
                        "mode": mode,
                        "session_id": mode_session_id,
                        "base_session_id": base_session_id,
                        "stage": "register",
                        "returncode": reg_rc,
                        "force_expert": force_expert,
                        "metric_prefix": eval_prefix,
                        "register_log": str(reg_log),
                        "eval_log": None,
                        "reg_yaml": str(reg_yaml),
                        "eval_yaml": None,
                    }
                    rows.append(row)
                    continue
            else:
                if not args.dry_run and not reg_yaml.is_file():
                    raise ScriptError(f"--skip-register was given, but reg yaml is missing: {reg_yaml}")

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
                "--session_id",
                mode_session_id,
                "--wandb_metric_prefix",
                eval_prefix,
            ]
            print_stage_header(
                f"EVAL [{scene_idx}/{len(scenes)} scene] [{mode_idx}/{len(modes)} mode] "
                f"scene={scene} mode={mode} session={mode_session_id}"
            )
            print("cmd:", " ".join(cmd_eval))
            print("log:", eval_log)
            if args.dry_run:
                eval_rc = -999
                eval_metrics = {}
                eval_result = None
            else:
                eval_result = run_cmd(cmd_eval, log_path=eval_log, cwd=repo_root, env=env)
                eval_rc = eval_result.returncode
                print_scene_result("EVAL", scene, mode_session_id, eval_result)
                eval_metrics = flatten_eval_res(load_eval_yaml(eval_yaml)) if eval_rc == 0 and eval_yaml.is_file() else {}

            row = {
                "scene": scene,
                "mode": mode,
                "session_id": mode_session_id,
                "base_session_id": base_session_id,
                "stage": "eval",
                "returncode": eval_rc,
                "force_expert": force_expert,
                "metric_prefix": eval_prefix,
                "register_log": str(reg_log),
                "eval_log": str(eval_log),
                "reg_yaml": str(reg_yaml),
                "eval_yaml": str(eval_yaml),
            }
            row.update(eval_metrics)
            rows.append(row)

    # Add per-scene fused-vs-best summary rows where possible.
    by_scene: dict[str, dict[str, dict]] = {}
    for row in rows:
        scene = row.get("scene")
        mode = row.get("mode")
        if scene is not None and mode is not None:
            by_scene.setdefault(str(scene), {})[str(mode)] = row

    comparison_rows = []
    for scene, mode_rows in by_scene.items():
        fused = mode_rows.get("fused")
        expert_rows = [r for m, r in mode_rows.items() if m.startswith("expert") and int(r.get("returncode", 1)) == 0]
        if not fused or int(fused.get("returncode", 1)) != 0 or not expert_rows:
            continue

        def first_present(row: dict, keys: list[str]):
            for key in keys:
                if key in row:
                    return row[key]
            return None

        fused_cm = first_present(fused, ["res.median_error_cm", "median_error_cm"])
        fused_deg = first_present(fused, ["res.median_error_deg", "median_error_deg"])
        expert_cms = [first_present(r, ["res.median_error_cm", "median_error_cm"]) for r in expert_rows]
        expert_degs = [first_present(r, ["res.median_error_deg", "median_error_deg"]) for r in expert_rows]
        expert_cms = [x for x in expert_cms if x is not None]
        expert_degs = [x for x in expert_degs if x is not None]

        if fused_cm is not None and expert_cms:
            fused_minus_best_cm = float(fused_cm) - min(float(x) for x in expert_cms)
        else:
            fused_minus_best_cm = None
        if fused_deg is not None and expert_degs:
            fused_minus_best_deg = float(fused_deg) - min(float(x) for x in expert_degs)
        else:
            fused_minus_best_deg = None

        comparison_rows.append({
            "scene": scene,
            "base_session_id": fused.get("base_session_id"),
            "fused_minus_best_expert_cm": fused_minus_best_cm,
            "fused_minus_best_expert_deg": fused_minus_best_deg,
        })

    summary_prefix = f"{slugify(args.session_prefix)}__" if args.session_prefix else ""
    summary_csv = write_csv(
        output_dirs["summaries"] / f"{summary_prefix}{slugify(model_name)}__{slugify(dataset_name)}__expert_eval_summary.csv",
        rows,
    )
    summary_json = write_json(
        output_dirs["summaries"] / f"{summary_prefix}{slugify(model_name)}__{slugify(dataset_name)}__expert_eval_summary.json",
        rows,
    )
    comparison_csv = write_csv(
        output_dirs["summaries"] / f"{summary_prefix}{slugify(model_name)}__{slugify(dataset_name)}__expert_eval_comparison.csv",
        comparison_rows,
    )
    comparison_json = write_json(
        output_dirs["summaries"] / f"{summary_prefix}{slugify(model_name)}__{slugify(dataset_name)}__expert_eval_comparison.json",
        comparison_rows,
    )

    print_stage_header("EXPERT EVAL SUMMARY")
    print(json.dumps(rows, indent=2))
    print(f"summary_csv     : {summary_csv}")
    print(f"summary_json    : {summary_json}")
    print(f"comparison_csv  : {comparison_csv}")
    print(f"comparison_json : {comparison_json}")

    if wb_run is not None:
        wandb_log_table(wb_run, "expert_eval/summary_table", rows)
        if comparison_rows:
            wandb_log_table(wb_run, "expert_eval/comparison_table", comparison_rows)
            # Log comparisons to run summary for quick W&B filtering. For multi-scene, suffix by scene.
            for comp in comparison_rows:
                scene = comp["scene"]
                for key in ["fused_minus_best_expert_cm", "fused_minus_best_expert_deg"]:
                    val = comp.get(key)
                    if val is not None:
                        wb_run.summary[f"expert_eval/{scene}/{key}"] = val
        for path, artifact_type in [
            (summary_csv, "expert-eval-summary"),
            (summary_json, "expert-eval-summary"),
            (comparison_csv, "expert-eval-comparison"),
            (comparison_json, "expert-eval-comparison"),
        ]:
            wandb_log_artifact_file(
                wb_run,
                path=path,
                artifact_name=f"{artifact_type}__{summary_prefix}{model_name}__{dataset_name}__{Path(path).suffix.lstrip('.')}",
                artifact_type=artifact_type,
                metadata={"model_name": model_name, "dataset_name": dataset_name},
            )
        for row in rows:
            eval_yaml = row.get("eval_yaml")
            if eval_yaml and Path(eval_yaml).is_file() and int(row.get("returncode", 1)) == 0:
                wandb_log_artifact_file(
                    wb_run,
                    path=eval_yaml,
                    artifact_name=f"expert-eval-yaml__{row['session_id']}",
                    artifact_type="eval-yaml",
                    metadata={
                        "scene": row.get("scene"),
                        "mode": row.get("mode"),
                        "session_id": row.get("session_id"),
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                    },
                )
        wandb_finish(
            wb_run,
            summary_updates={
                "expert_eval/model_name": model_name,
                "expert_eval/dataset_name": dataset_name,
                "expert_eval/modes": ",".join(modes),
                **{f"expert_eval/{k}": v for k, v in summarize_returncodes(rows).items()},
                "expert_eval/summary_csv": str(summary_csv),
                "expert_eval/summary_json": str(summary_json),
                "expert_eval/comparison_csv": str(comparison_csv),
                "expert_eval/comparison_json": str(comparison_json),
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
