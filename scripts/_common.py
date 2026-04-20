#!/usr/bin/env python3
"""Shared helpers for migrating the DPTv2_ACE.ipynb workflows to server scripts.

These helpers deliberately replace Colab-specific assumptions with explicit server paths,
explicit environment setup, and structured status/result files.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping


IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")


class ScriptError(RuntimeError):
    """Raised for expected, user-facing script errors."""


@dataclass
class CommandResult:
    cmd: list[str]
    returncode: int
    log_path: Path
    started_at: str
    finished_at: str
    duration_sec: float


# --------------------------
# Path / naming helpers
# --------------------------
def expand(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def ensure_dir(path: str | Path) -> Path:
    p = expand(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_file(path: str | Path, label: str = "file") -> Path:
    p = expand(path)
    if not p.is_file():
        raise ScriptError(f"Missing {label}: {p}")
    return p


def require_dir(path: str | Path, label: str = "directory") -> Path:
    p = expand(path)
    if not p.is_dir():
        raise ScriptError(f"Missing {label}: {p}")
    return p


def repo_src_dir(repo_root: str | Path) -> Path:
    repo_root = require_dir(repo_root, "repo root")
    src = repo_root / "src"
    if not (src / "ace_g").is_dir():
        raise ScriptError(f"Could not find ace_g package under: {src}")
    return src


def config_path_from_args(repo_root: str | Path, config_name: str | None, config_path: str | Path | None) -> Path:
    if config_path:
        return require_file(config_path, "config file")
    if not config_name:
        raise ScriptError("Provide either --config-name or --config-path")
    cfg = expand(repo_root) / "configs_custom" / config_name
    return require_file(cfg, "config file")


def dataset_name_from_root(dataset_root: str | Path) -> str:
    return expand(dataset_root).name


def default_output_paths(output_root: str | Path) -> dict[str, Path]:
    root = ensure_dir(output_root)
    return {
        "root": root,
        "logs": ensure_dir(root / "logs"),
        "runs": ensure_dir(root / "runs"),
        "summaries": ensure_dir(root / "summaries"),
        "promoted": ensure_dir(root / "promoted"),
        "metrics": ensure_dir(root / "metrics"),
        "train_status": ensure_dir(root / "train_status"),
    }


# --------------------------
# Dataset helpers
# --------------------------
def pick_img_glob(rgb_dir: str | Path) -> str:
    rgb_dir = require_dir(rgb_dir, "rgb directory")
    for pat in IMG_EXTS:
        if any(rgb_dir.glob(pat)):
            return str(rgb_dir / pat)
    return str(rgb_dir / "*.*")


def discover_scenes(dataset_root: str | Path) -> list[str]:
    root = require_dir(dataset_root, "dataset root")
    scenes: list[str] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "train" / "rgb").is_dir() and (d / "train" / "poses").is_dir():
            scenes.append(d.name)
    return scenes


def resolve_scenes(dataset_root: str | Path, scenes: list[str] | None) -> list[str]:
    if scenes:
        return scenes
    found = discover_scenes(dataset_root)
    if not found:
        raise ScriptError(f"No scenes found under {expand(dataset_root)}")
    return found


def scene_split_paths(dataset_root: str | Path, scene: str, split: str) -> dict[str, str]:
    base = require_dir(expand(dataset_root) / scene / split, f"scene split directory ({scene}/{split})")
    rgb = pick_img_glob(base / "rgb")
    poses = str(require_dir(base / "poses", f"poses directory ({scene}/{split})") / "*.txt")
    calibration = str(require_dir(base / "calibration", f"calibration directory ({scene}/{split})") / "*.txt")
    return {"rgb": rgb, "poses": poses, "calibration": calibration}


# --------------------------
# Environment helpers
# --------------------------
def build_subprocess_env(
    repo_root: str | Path,
    torch_home: str | Path | None = None,
    extra_env: Iterable[str] | None = None,
    prepend_pythonpath: Iterable[str | Path] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()

    # Make ace_g importable in subprocesses.
    repo_src = repo_src_dir(repo_root)
    py_parts = [str(repo_src)]
    if prepend_pythonpath:
        py_parts.extend(str(expand(p)) for p in prepend_pythonpath)
    old_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join([*py_parts, old_py]).strip(":")

    # Reproduce the notebook's torch lib / LD_LIBRARY_PATH handling when torch is present.
    try:
        import torch  # type: ignore

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        old_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join([str(torch_lib), old_ld]).strip(":")
    except Exception:
        pass

    if torch_home:
        env["TORCH_HOME"] = str(expand(torch_home))

    if extra_env:
        for item in extra_env:
            if "=" not in item:
                raise ScriptError(f"Invalid --extra-env entry (expected KEY=VALUE): {item}")
            key, value = item.split("=", 1)
            env[key] = value

    return env


# --------------------------
# Process / logging helpers
# --------------------------
def format_cmd(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_cmd(cmd: list[str], log_path: str | Path, cwd: str | Path, env: Mapping[str, str]) -> CommandResult:
    log_path = expand(log_path)
    ensure_dir(log_path.parent)
    started = datetime.now()
    started_iso = started.isoformat(timespec="seconds")

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"STARTED_AT: {started_iso}\n")
        f.write(f"CWD: {expand(cwd)}\n")
        f.write(f"CMD: {format_cmd(cmd)}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(expand(cwd)), env=dict(env), stdout=f, stderr=subprocess.STDOUT)

    finished = datetime.now()
    finished_iso = finished.isoformat(timespec="seconds")
    duration = (finished - started).total_seconds()

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"FINISHED_AT: {finished_iso}\n")
        f.write(f"RETURN_CODE: {proc.returncode}\n")
        f.write(f"DURATION_SEC: {duration:.3f}\n")

    return CommandResult(
        cmd=cmd,
        returncode=proc.returncode,
        log_path=log_path,
        started_at=started_iso,
        finished_at=finished_iso,
        duration_sec=duration,
    )


def print_stage_header(title: str) -> None:
    bar = "=" * 88
    print(f"\n{bar}\n{title}\n{bar}")


def print_scene_result(stage: str, scene: str, session_id: str, result: CommandResult) -> None:
    print(
        f"[{stage}] scene={scene} session={session_id} rc={result.returncode} "
        f"duration={result.duration_sec:.1f}s log={result.log_path}"
    )


# --------------------------
# Status / summary helpers
# --------------------------
def write_json(path: str | Path, obj: object) -> Path:
    path = expand(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")
    return path


def read_json(path: str | Path) -> object:
    return json.loads(require_file(path, "JSON file").read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: list[dict]) -> Path:
    path = expand(path)
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def summarize_returncodes(rows: list[dict], key: str = "returncode") -> dict[str, int]:
    total = len(rows)
    ok = sum(1 for r in rows if int(r.get(key, 1)) == 0)
    return {"total": total, "ok": ok, "failed": total - ok}


# --------------------------
# Eval parsing
# --------------------------
def load_eval_yaml(path: str | Path) -> dict:
    import yaml  # type: ignore

    p = require_file(path, "eval yaml")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ScriptError(f"Unexpected YAML structure in {p}")
    return data


def flatten_eval_res(eval_yaml: dict) -> dict:
    res = eval_yaml.get("res", {}) if isinstance(eval_yaml, dict) else {}
    if not isinstance(res, dict):
        return {}
    flat: dict[str, object] = {}
    for key, value in res.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat[key] = value
        else:
            flat[key] = json.dumps(value)
    return flat


def parse_text_metrics(text: str) -> dict[str, object]:
    def grab(patterns: list[str], cast=float):
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                try:
                    return cast(m.group(1))
                except Exception:
                    return m.group(1)
        return None

    out: dict[str, object] = {
        "present": bool(text),
        "crashed": ("Traceback (most recent call last)" in text) or ("RuntimeError" in text and "Traceback" in text),
        "median_m": grab([
            r"median[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*m",
            r"median translation[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "median_deg": grab([
            r"median[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*deg",
            r"median rotation[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "mean_m": grab([
            r"mean[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*m",
            r"mean translation[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "mean_deg": grab([
            r"mean[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*deg",
            r"mean rotation[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "success_pct": grab([
            r"success[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"accuracy[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"recall[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*%",
        ]),
        "registered_xy": grab([
            r"registered[^0-9]*([0-9]+\s*/\s*[0-9]+)",
            r"success[^0-9]*([0-9]+\s*/\s*[0-9]+)",
        ], cast=str),
    }
    return out


def find_first_glob(patterns: Iterable[str | Path]) -> Path | None:
    hits: list[str] = []
    for pat in patterns:
        hits.extend(glob.glob(str(pat), recursive=True))
    hits = sorted(set(hits))
    return Path(hits[0]) if hits else None


def tail_text(text: str, n: int = 40) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if lines else ""


def read_text_if_exists(path: str | Path | None) -> str:
    if path is None:
        return ""
    p = expand(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


# --------------------------
# Artifact / result discovery
# --------------------------
def find_head_checkpoint(output_root: str | Path, session_id: str) -> Path | None:
    root = expand(output_root)
    candidates = [
        root / f"{session_id}_head.pt",
        root / f"{session_id}.pt",
        *root.glob(f"{session_id}*head*.pt"),
        *root.glob(f"{session_id}*.pt"),
    ]
    seen: set[Path] = set()
    for cand in candidates:
        c = expand(cand) if not isinstance(cand, Path) else cand.expanduser().resolve()
        if c in seen:
            continue
        seen.add(c)
        if c.is_file():
            return c
    return None


def promote_file(src: str | Path, promoted_dir: str | Path) -> Path:
    import shutil

    src = require_file(src, "artifact file")
    dst_dir = ensure_dir(promoted_dir)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


# --------------------------
# W&B helpers
# --------------------------
def wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
        return True
    except Exception:
        return False


def maybe_init_wandb(
    *,
    enabled: bool,
    project: str,
    entity: str,
    job_type: str,
    name: str,
    group: str,
    config: dict,
    tags: list[str] | None = None,
):
    if not enabled:
        return None
    if not wandb_available():
        raise ScriptError("W&B requested but wandb is not importable in this environment")

    import wandb  # type: ignore

    return wandb.init(
        project=project,
        entity=entity,
        job_type=job_type,
        name=name,
        group=group,
        config=config,
        tags=tags or [],
    )


def wandb_log_table(run, key: str, rows: list[dict]) -> None:
    if run is None:
        return
    import wandb  # type: ignore

    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for col in row.keys():
            if col not in columns:
                columns.append(col)
    data = [[row.get(col) for col in columns] for row in rows]
    table = wandb.Table(columns=columns, data=data)
    run.log({key: table})


def wandb_log_artifact_file(run, *, path: str | Path, artifact_name: str, artifact_type: str, metadata: dict | None = None) -> None:
    if run is None:
        return
    import wandb  # type: ignore

    p = require_file(path, "artifact file")
    art = wandb.Artifact(name=slugify(artifact_name), type=artifact_type, metadata=metadata or {})
    art.add_file(str(p))
    run.log_artifact(art)


def wandb_finish(run, summary_updates: dict | None = None) -> None:
    if run is None:
        return
    if summary_updates:
        for k, v in summary_updates.items():
            run.summary[k] = v
    run.finish()
