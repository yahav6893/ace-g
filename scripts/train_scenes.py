#!/usr/bin/env python3
"""Train ACE-G scenes from a config, replacing notebook cells 25-26.

Key differences vs the Colab notebook:
- server-local paths instead of /content or Drive
- explicit output directories
- explicit subprocess environment
- structured train_status JSON
- optional W&B logging and model artifact upload
- placeholders left configurable for cache/config naming
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import yoco

from _common import (
    ScriptError,
    build_subprocess_env,
    config_path_from_args,
    dataset_name_from_root,
    default_output_paths,
    find_head_checkpoint,
    maybe_init_wandb,
    pick_img_glob,
    print_scene_result,
    print_stage_header,
    promote_file,
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
    write_json,
    expand,
    ensure_dir,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--repo-root", default="~/dace/ace-g", help="ACE-G repo root")
    p.add_argument("--dataset-root", default="~/dace/data/ace_datasets/TODO_DATASET_NAME", help="Dataset root containing scene directories")
    p.add_argument("--output-root", default="~/dace/outputs", help="Output root for logs, status, promoted artifacts")
    p.add_argument("--cache-root", default="~/dace/cache/ace_cache", help="Cache root placeholder for your server cache tree")
    p.add_argument("--torch-home", default="~/dace/cache/ace_cache/torch", help="TORCH_HOME for subprocesses")

    group_cfg = p.add_mutually_exclusive_group(required=True)
    group_cfg.add_argument("--config-name", help="Config filename under <repo>/configs_custom/ (placeholder until names are finalized)")
    group_cfg.add_argument("--config-path", help="Absolute config path (use this if config is not under configs_custom)")

    p.add_argument("--model-name", default=None, help="Explicit model/session prefix. Default: config stem")
    p.add_argument("--scenes", nargs="*", default=None, help="Scene names. Omit to auto-discover scenes under dataset root")
    p.add_argument("--session-prefix", default=None, help="Optional prefix before '<model>-<dataset>-<scene>'")

    p.add_argument("--wandb-entity", default="yahav6893")
    p.add_argument("--wandb-project", default="dace")
    p.add_argument("--wandb-group", default=None, help="Default: <model>__<dataset>")
    p.add_argument("--disable-wandb", action="store_true")
    p.add_argument("--wandb-tag", action="append", default=[], help="Repeatable W&B tag")

    p.add_argument("--extra-env", action="append", default=[], help="Repeatable KEY=VALUE passed to subprocesses")
    p.add_argument("--pythonpath-prepend", action="append", default=[], help="Additional entries prepended to PYTHONPATH")

    p.add_argument("--train-status-name", default=None, help="Override train status JSON filename")
    p.add_argument("--promote-heads", action="store_true", help="Copy found head checkpoints into outputs/promoted before logging to W&B")
    p.add_argument("--dry-run", action="store_true", help="Print commands and status targets without executing")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = expand(args.repo_root)
    dataset_root = expand(args.dataset_root)
    output_dirs = default_output_paths(args.output_root)
    config_path = config_path_from_args(repo_root, args.config_name, args.config_path)
    model_name = args.model_name or config_path.stem
    dataset_name = dataset_name_from_root(dataset_root)
    wandb_group = args.wandb_group or f"{slugify(model_name)}__{slugify(dataset_name)}"
    scenes = resolve_scenes(dataset_root, args.scenes)

    print_stage_header("TRAIN SCENES")
    print(f"repo_root    : {repo_root}")
    print(f"dataset_root : {dataset_root}")
    print(f"config_path  : {config_path}")
    print(f"model_name   : {model_name}")
    print(f"dataset_name : {dataset_name}")
    print(f"scenes       : {scenes}")
    print(f"output_root  : {output_dirs['root']}")
    print(f"cache_root   : {expand(args.cache_root)}")
    print(f"torch_home   : {expand(args.torch_home)}")
    print(f"wandb        : {'disabled' if args.disable_wandb else 'enabled'}")

    _ = repo_src_dir(repo_root)  # validate early
    env = build_subprocess_env(
        repo_root=repo_root,
        torch_home=args.torch_home,
        extra_env=args.extra_env + [
            f"WANDB_PROJECT={args.wandb_project}",
            f"WANDB_ENTITY={args.wandb_entity}",
        ],
        prepend_pythonpath=args.pythonpath_prepend,
    )

    config_dict = yoco.load_config_from_file(config_path)

    run_name = f"train__{slugify(model_name)}__{slugify(dataset_name)}__{timestamp_now()}"
    wb_run = maybe_init_wandb(
        enabled=not args.disable_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="train",
        name=run_name,
        group=wandb_group,
        tags=[*args.wandb_tag, dataset_name, model_name, "train"],
        config={
            "repo_root": str(repo_root),
            "dataset_root": str(dataset_root),
            "output_root": str(output_dirs['root']),
            "cache_root": str(expand(args.cache_root)),
            "torch_home": str(expand(args.torch_home)),
            "config_path": str(config_path),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "scenes": scenes,
            **config_dict,
        },
    )

    def _find_expert_heads(cfg):
        if isinstance(cfg, dict):
            if "expert_head_paths" in cfg:
                return cfg["expert_head_paths"]
            for v in cfg.values():
                res = _find_expert_heads(v)
                if res: return res
        elif isinstance(cfg, list):
            for item in cfg:
                res = _find_expert_heads(item)
                if res: return res
        return []

    expert_paths = _find_expert_heads(config_dict)

    rows: list[dict] = []
    status_payload: dict[str, dict] = {}

    for idx, scene in enumerate(scenes, start=1):
        if expert_paths:
            for ep in expert_paths:
                if not ep:
                    continue
                ep_stem = Path(ep).name
                if ep_stem.endswith(".pt"):
                    ep_stem = ep_stem[:-3]
                elif ep_stem.endswith(".p"):
                    ep_stem = ep_stem[:-2]
                if ep_stem.endswith("_head"):
                    ep_stem = ep_stem[:-5]
                if not (ep_stem.endswith(f"_{scene}") or ep_stem.endswith(f"-{scene}")):
                    raise ScriptError(f"Sanity check failed: --scene '{scene}' does not match expert_head_path '{ep}'")

        split = scene_split_paths(dataset_root, scene, "train")
        base_session = f"{model_name}-{dataset_name}-{scene}"
        session_id = f"{args.session_prefix}-{base_session}" if args.session_prefix else base_session
        log_path = output_dirs["logs"] / f"{session_id}__train.log"
        map_yaml = output_dirs["root"] / f"{session_id}_map.yaml"

        cmd = [
            sys.executable,
            "-m",
            "ace_g.train_single_scene",
            "--config",
            str(config_path),
            "--dataset.rgb_files",
            split["rgb"],
            "--dataset.pose_files",
            split["poses"],
            "--dataset.calibration_files",
            split["calibration"],
            "--session_id",
            session_id,
            "--output_dir",
            str(output_dirs["root"]),
        ]

        print_stage_header(f"TRAIN [{idx}/{len(scenes)}] scene={scene} session={session_id}")
        print("cmd:", " ".join(cmd))
        print("log:", log_path)

        if args.dry_run:
            row = {
                "scene": scene,
                "session_id": session_id,
                "returncode": -999,
                "dry_run": True,
                "log": str(log_path),
                "map_yaml": str(map_yaml),
                "head_path": None,
                "promoted_head_path": None,
            }
            rows.append(row)
            status_payload[scene] = row.copy()
            continue

        result = run_cmd(cmd, log_path=log_path, cwd=repo_root, env=env)
        print_scene_result("TRAIN", scene, session_id, result)

        head_path = find_head_checkpoint(output_dirs["root"], session_id)
        promoted_head_path = None
        if head_path and args.promote_heads:
            promoted_head_path = promote_file(head_path, output_dirs["promoted"])

        row = {
            "scene": scene,
            "session_id": session_id,
            "returncode": result.returncode,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "duration_sec": round(result.duration_sec, 3),
            "log": str(result.log_path),
            "map_yaml": str(map_yaml),
            "head_path": str(head_path) if head_path else None,
            "promoted_head_path": str(promoted_head_path) if promoted_head_path else None,
        }
        rows.append(row)
        status_payload[scene] = row.copy()

        if wb_run is not None and head_path and head_path.is_file():
            to_log = promoted_head_path or head_path
            wandb_log_artifact_file(
                wb_run,
                path=to_log,
                artifact_name=f"model__{session_id}",
                artifact_type="model",
                metadata={
                    "scene": scene,
                    "session_id": session_id,
                    "config_path": str(config_path),
                    "dataset_root": str(dataset_root),
                    "dataset_name": dataset_name,
                    "returncode": result.returncode,
                },
            )

    status_name = args.train_status_name or f"{slugify(model_name)}__{slugify(dataset_name)}__train_status.json"
    status_path = write_json(output_dirs["train_status"] / status_name, status_payload)
    print_stage_header("TRAIN SUMMARY")
    print(json.dumps(status_payload, indent=2))
    print(f"train_status_json: {status_path}")

    if wb_run is not None:
        wandb_log_table(wb_run, "train/status_table", rows)
        wandb_log_artifact_file(
            wb_run,
            path=status_path,
            artifact_name=f"train-status__{model_name}__{dataset_name}",
            artifact_type="train-status",
            metadata={
                "model_name": model_name,
                "dataset_name": dataset_name,
                "num_scenes": len(scenes),
            },
        )
        wandb_finish(
            wb_run,
            summary_updates={
                "train/model_name": model_name,
                "train/dataset_name": dataset_name,
                **{f"train/{k}": v for k, v in summarize_returncodes(rows).items()},
                "train/status_json": str(status_path),
            },
        )

    failures = [r for r in rows if int(r.get("returncode", 1)) != 0]
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScriptError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
