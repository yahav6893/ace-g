#!/usr/bin/env python3
"""Diagnostics migrated from notebook cells 31-35.

Subcommands:
- compare-means: compare learned head means across two checkpoints
- compare-logs: compare register/eval logs across session prefixes
- gate-weights: inspect per-expert softmax weights for late-fusion heads
- gate-last-layer: inspect gating last-layer params, and optionally probe one image
- print-config: print selected head config fields
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import yaml

from _common import (
    ScriptError,
    expand,
    find_first_glob,
    parse_text_metrics,
    print_stage_header,
    read_text_if_exists,
    repo_src_dir,
    tail_text,
    write_csv,
)


def ensure_ace_g_importable(repo_root: str | Path) -> None:
    src = repo_src_dir(repo_root)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def setup_wandb_logging(args: argparse.Namespace, job_type: str):
    if not getattr(args, "wandb_log", False):
        return None
    import wandb

    project = getattr(args, "wandb_project", None)
    entity = getattr(args, "wandb_entity", None)

    # Try to infer entity/project from head_path if not provided
    head_path = getattr(args, "head_path", "")
    if str(head_path).startswith("wandb://") and not (project and entity):
        art_path = str(head_path)[len("wandb://"):]
        parts = art_path.split("/")
        if len(parts) >= 3:
            entity = entity or parts[0]
            project = project or parts[1]

    run = wandb.init(project=project, entity=entity, job_type=job_type)
    
    if str(head_path).startswith("wandb://"):
        art_path = str(head_path)[len("wandb://"):]
        run.use_artifact(art_path)
    return run


def load_head(repo_root: str | Path, head_path: str | Path):
    ensure_ace_g_importable(repo_root)
    from ace_g import scr_heads  # type: ignore

    head_path_str = str(head_path)
    if head_path_str.startswith("wandb://"):
        import wandb
        artifact_path = head_path_str[len("wandb://"):]
        print(f"Downloading wandb artifact: {artifact_path} ...")
        api = wandb.Api()
        try:
            artifact = api.artifact(artifact_path)
            download_dir = artifact.download()
            pt_files = list(Path(download_dir).glob("*.pt"))
            if not pt_files:
                raise ScriptError(f"No .pt file found in artifact {artifact_path}")
            p = pt_files[0]
            print(f"Using downloaded artifact head: {p}")
        except Exception as e:
            raise ScriptError(f"Error fetching wandb artifact: {e}")
    else:
        p = expand(head_path)

    if not p.is_file():
        raise ScriptError(f"Missing head checkpoint: {p}")
    return scr_heads.create_head(p)


def cmd_compare_means(args: argparse.Namespace) -> int:
    print_stage_header("COMPARE HEAD MEANS")
    h0 = load_head(args.repo_root, args.head_a).cpu().eval()
    h1 = load_head(args.repo_root, args.head_b).cpu().eval()

    mean0 = h0.mean.view(-1).tolist()
    mean1 = h1.mean.view(-1).tolist()
    max_abs_diff = float((h0.mean - h1.mean).abs().max().item())
    out = {
        "head_a": str(expand(args.head_a)),
        "head_b": str(expand(args.head_b)),
        "mean_a": mean0,
        "mean_b": mean1,
        "max_abs_diff": max_abs_diff,
    }
    print(json.dumps(out, indent=2))
    return 0


def find_log(log_dir: Path, session_prefix: str, kind: str) -> Path | None:
    return find_first_glob(
        [
            log_dir / f"{session_prefix}__{kind}.log",
            log_dir / f"{session_prefix}*__{kind}.log",
            log_dir / f"{session_prefix}*{kind}*.log",
        ]
    )


def cmd_compare_logs(args: argparse.Namespace) -> int:
    print_stage_header("COMPARE REGISTER/EVAL LOGS")
    log_dir = expand(args.log_dir)
    session_pairs: list[tuple[str, str]] = []
    for item in args.session:
        if "=" not in item:
            raise ScriptError(f"Expected --session LABEL=PREFIX, got: {item}")
        label, prefix = item.split("=", 1)
        session_pairs.append((label, prefix))

    rows: list[dict[str, Any]] = []
    tail_bundle: dict[str, str] = {}
    for label, prefix in session_pairs:
        reg_p = find_log(log_dir, prefix, "register")
        eval_p = find_log(log_dir, prefix, "eval")
        reg_txt = read_text_if_exists(reg_p)
        eval_txt = read_text_if_exists(eval_p)
        reg_m = parse_text_metrics(reg_txt)
        eval_m = parse_text_metrics(eval_txt)
        row = {
            "label": label,
            "session_prefix": prefix,
            "register_log": str(reg_p) if reg_p else None,
            "eval_log": str(eval_p) if eval_p else None,
            "reg_present": reg_m.get("present", False),
            "reg_crashed": reg_m.get("crashed", False),
            "reg_registered": reg_m.get("registered_xy", None),
            "eval_present": eval_m.get("present", False),
            "eval_crashed": eval_m.get("crashed", False),
            "median_m": eval_m.get("median_m", None),
            "median_deg": eval_m.get("median_deg", None),
            "mean_m": eval_m.get("mean_m", None),
            "mean_deg": eval_m.get("mean_deg", None),
            "success_pct": eval_m.get("success_pct", None),
        }
        rows.append(row)
        tail_bundle[f"{label}/register"] = tail_text(reg_txt, args.tail_lines)
        tail_bundle[f"{label}/eval"] = tail_text(eval_txt, args.tail_lines)

    print(json.dumps(rows, indent=2))
    print("\n=== LOG TAILS ===")
    for key, tail in tail_bundle.items():
        print(f"\n--- {key} ---")
        print(tail or "(no log / empty)")

    if args.out_csv:
        out_csv = write_csv(args.out_csv, rows)
        print(f"wrote_csv: {out_csv}")
    return 0


def _import_obj(obj_type: str):
    mod, name = obj_type.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def load_img_tensor(path: str | Path, max_side: int | None) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    if max_side is not None:
        w, h = im.size
        s = max(w, h)
        if s > max_side:
            scale = max_side / float(s)
            im = im.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
    arr = np.asarray(im).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x


def build_encoders_from_cfg(cfg: dict) -> list[torch.nn.Module]:
    enc_cfg = cfg["sst"]["encoder"]
    obj_type = enc_cfg["obj_type"]

    if obj_type.endswith("ListMultiEncoder"):
        encs = []
        for e in enc_cfg["kwargs"]["encoders"]:
            cls = _import_obj(e["obj_type"])
            encs.append(cls(**(e.get("kwargs", {}) or {})))
        return encs

    if obj_type.endswith("MultiDinoEncoder"):
        encs = []
        main = enc_cfg["kwargs"]["main"]
        fusion = enc_cfg["kwargs"].get("fusion", []) or []
        cls = _import_obj(main["obj_type"])
        encs.append(cls(**(main.get("kwargs", {}) or {})))
        for e in fusion:
            cls = _import_obj(e["obj_type"])
            encs.append(cls(**(e.get("kwargs", {}) or {})))
        return encs

    raise ScriptError(f"Unsupported encoder obj_type for diagnostics: {obj_type}")


def run_concat_features(encs: list[torch.nn.Module], x: torch.Tensor) -> torch.Tensor:
    feats = []
    h0 = w0 = None
    for enc in encs:
        f = enc(x)
        if h0 is None:
            h0, w0 = f.shape[-2], f.shape[-1]
        elif f.shape[-2] != h0 or f.shape[-1] != w0:
            raise ScriptError(f"Spatial mismatch across encoders: {(h0, w0)} vs {(f.shape[-2], f.shape[-1])}")
        feats.append(f)
    return torch.cat(feats, dim=1)


def extract_softmax_weights(head, patch_embeddings: torch.Tensor) -> torch.Tensor:
    cfg = head.config
    k = getattr(cfg, "num_experts", getattr(cfg, "num_encoders", None))
    if k is None:
        raise ScriptError("Head config has neither num_experts nor num_encoders")
    main_index = getattr(cfg, "main_index", 0)
    gate_input = getattr(cfg, "gate_input", "main")
    weights_per_patch = getattr(cfg, "weights_per_patch", False)
    temperature = float(getattr(cfg, "temperature", 1.0))

    b, kc, h, w = patch_embeddings.shape
    c = kc // k
    feats = patch_embeddings.view(b, k, c, h, w)
    if gate_input == "main":
        g_in = feats[:, main_index]
    elif gate_input == "concat":
        g_in = patch_embeddings
    else:
        raise ScriptError(f"Unsupported gate_input: {gate_input}")

    logits = head.gate(g_in)
    if weights_per_patch:
        wts = torch.softmax((logits / temperature).float(), dim=1).to(logits.dtype)
    else:
        lg = logits.mean(dim=(2, 3))
        wg = torch.softmax((lg / temperature).float(), dim=1).to(logits.dtype)
        wts = wg[:, :, None, None].expand(-1, -1, h, w)
    return wts


def cmd_gate_weights(args: argparse.Namespace) -> int:
    print_stage_header("GATE WEIGHTS")
    ensure_ace_g_importable(args.repo_root)
    run = setup_wandb_logging(args, "gate-weights")
    if run:
        import wandb
        table = wandb.Table(columns=["image", "expert_idx", "mean", "min", "max", "win_pct"])

    cfg_path = expand(args.config_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    encs = build_encoders_from_cfg(cfg)
    head = load_head(args.repo_root, args.head_path)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_half = args.use_half and device == "cuda"

    for i in range(len(encs)):
        encs[i] = encs[i].to(device).eval()
    head = head.to(device).eval()

    paths = sorted(glob.glob(args.rgb_glob))[: args.num_images]
    if not paths:
        if run: run.finish(exit_code=1)
        raise ScriptError(f"No images matched: {args.rgb_glob}")

    print(f"config_path: {cfg_path}")
    print(f"head_path  : {expand(args.head_path)}")
    print(f"device     : {device}")
    print(f"use_half   : {use_half}")
    print(f"images     : {len(paths)}")
    print(f"first_image: {paths[0]}")

    with torch.no_grad():
        for p in paths:
            img_name = Path(p).name
            x = load_img_tensor(p, args.max_side).to(device)
            with torch.autocast("cuda", enabled=use_half):
                feats = run_concat_features(encs, x)
                wts = extract_softmax_weights(head, feats)
            w = wts[0]
            k = w.shape[0]
            mean = w.mean(dim=(1, 2)).detach().float().cpu().numpy()
            mn = w.amin(dim=(1, 2)).detach().float().cpu().numpy()
            mx = w.amax(dim=(1, 2)).detach().float().cpu().numpy()
            winner = w.argmax(dim=0).detach().cpu().numpy()
            win_frac = np.bincount(winner.reshape(-1), minlength=k) / winner.size
            print(f"\n--- {img_name} ---")
            for idx in range(k):
                print(
                    f"expert[{idx}] mean={mean[idx]:.4f} min={mn[idx]:.4f} "
                    f"max={mx[idx]:.4f} win%={100 * win_frac[idx]:.1f}"
                )
                if run:
                    table.add_data(img_name, idx, float(mean[idx]), float(mn[idx]), float(mx[idx]), float(100 * win_frac[idx]))
    
    if run:
        run.log({"gate_weights_distribution": table})
        run.finish()
        
    print("\nDone.")
    return 0


def cmd_gate_last_layer(args: argparse.Namespace) -> int:
    print_stage_header("GATE LAST LAYER")
    run = setup_wandb_logging(args, "gate-last-layer")

    head = load_head(args.repo_root, args.head_path).cpu().eval()
    if not hasattr(head, "gate"):
        if run: run.finish(exit_code=1)
        raise ScriptError("Loaded head has no .gate attribute")
    last = head.gate[-1]
    if not isinstance(last, torch.nn.Conv2d):
        if run: run.finish(exit_code=1)
        raise ScriptError(f"Expected head.gate[-1] to be Conv2d, got {type(last)}")

    bias = last.bias.detach().cpu().numpy() if last.bias is not None else None
    weight_sum = last.weight.detach().cpu().sum(dim=(1, 2, 3)).numpy()
    weight_l2 = torch.linalg.vector_norm(last.weight.detach().cpu().reshape(last.weight.shape[0], -1), dim=1).numpy()

    out = {
        "last_conv": str(last),
        "bias": bias.tolist() if bias is not None else None,
        "weight_sum_per_expert": weight_sum.tolist(),
        "weight_l2_per_expert": weight_l2.tolist(),
        "bias_only_softmax": torch.softmax(last.bias.detach().cpu(), dim=0).numpy().tolist() if last.bias is not None else None,
    }
    print(json.dumps(out, indent=2))
    
    if run:
        run.config.update({"gate_last_layer_stats": out})

    if args.probe_config_path and args.probe_rgb_glob:
        ensure_ace_g_importable(args.repo_root)
        cfg = yaml.safe_load(expand(args.probe_config_path).read_text(encoding="utf-8"))
        encs = build_encoders_from_cfg(cfg)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_half = args.use_half and device == "cuda"
        for i in range(len(encs)):
            encs[i] = encs[i].to(device).eval()
        head = head.to(device).eval()

        images = sorted(glob.glob(args.probe_rgb_glob))[:1]
        if not images:
            if run: run.finish(exit_code=1)
            raise ScriptError(f"No images matched probe glob: {args.probe_rgb_glob}")
        x = load_img_tensor(images[0], args.max_side).to(device)

        captured: dict[str, torch.Tensor] = {}

        def hook_fn(module, inputs, output):
            captured["pre_last"] = inputs[0].detach().cpu()
            captured["logits"] = output.detach().cpu()

        hook = last.register_forward_hook(hook_fn)
        with torch.no_grad():
            with torch.autocast("cuda", enabled=use_half):
                feats = run_concat_features(encs, x)
                _ = extract_softmax_weights(head, feats)
        hook.remove()

        h = captured.get("pre_last")
        logits = captured.get("logits")
        if h is None or logits is None:
            if run: run.finish(exit_code=1)
            raise ScriptError("Failed to capture last-layer input/logits during probe")

        w = last.weight.detach().cpu()
        b = last.bias.detach().cpu() if last.bias is not None else torch.zeros(w.shape[0])
        logits_from_w = torch.nn.functional.conv2d(h, w, bias=None)
        logits_from_b = b.view(1, -1, 1, 1).expand_as(logits_from_w)
        total = logits_from_w + logits_from_b
        probs = torch.softmax(logits, dim=1)
        probe_stats = {
            "probe_image": images[0],
            "mean_abs_w_dot_h_per_expert": logits_from_w.abs().mean(dim=(0, 2, 3)).numpy().tolist(),
            "mean_abs_bias_per_expert": logits_from_b.abs().mean(dim=(0, 2, 3)).numpy().tolist(),
            "mean_total_logits_per_expert": total.mean(dim=(0, 2, 3)).numpy().tolist(),
            "mean_logits_over_batch": logits.mean(dim=(2, 3)).mean(dim=0).numpy().tolist(),
            "std_logits_over_images": logits.mean(dim=(2, 3)).std(dim=0).numpy().tolist(),
            "mean_probs_over_batch": probs.mean(dim=(2, 3)).mean(dim=0).numpy().tolist(),
            "std_probs_over_images": probs.mean(dim=(2, 3)).std(dim=0).numpy().tolist(),
        }
        print("\nPROBE_STATS")
        print(json.dumps(probe_stats, indent=2))
        
        if run:
            run.log({"probe_stats": probe_stats})
            
    if run:
        run.finish()
        
    return 0


def cmd_print_config(args: argparse.Namespace) -> int:
    print_stage_header("PRINT HEAD CONFIG")
    head = load_head(args.repo_root, args.head_path).cpu().eval()
    cfg = head.config
    out = {
        "weights_per_patch": getattr(cfg, "weights_per_patch", None),
        "gate_input": getattr(cfg, "gate_input", None),
        "temperature": getattr(cfg, "temperature", None),
        "num_experts": getattr(cfg, "num_experts", getattr(cfg, "num_encoders", None)),
        "main_index": getattr(cfg, "main_index", None),
    }
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="~/dace/ace-g", help="ACE-G repo root")
    p.add_argument("--wandb-log", action="store_true", help="Log diagnostic results to W&B")
    p.add_argument("--wandb-project", default=None, help="W&B project")
    p.add_argument("--wandb-entity", default=None, help="W&B entity")

    sp = p.add_subparsers(dest="cmd", required=True)

    p_means = sp.add_parser("compare-means", help="Compare two head mean tensors")
    p_means.add_argument("--head-a", required=True)
    p_means.add_argument("--head-b", required=True)
    p_means.set_defaults(func=cmd_compare_means)

    p_logs = sp.add_parser("compare-logs", help="Compare register/eval logs for session prefixes")
    p_logs.add_argument("--log-dir", default="~/dace/outputs/logs")
    p_logs.add_argument("--session", action="append", required=True, help="LABEL=SESSION_PREFIX")
    p_logs.add_argument("--tail-lines", type=int, default=40)
    p_logs.add_argument("--out-csv", default=None)
    p_logs.set_defaults(func=cmd_compare_logs)

    p_gw = sp.add_parser("gate-weights", help="Inspect late-fusion softmax weights over sample images")
    p_gw.add_argument("--config-path", required=True, help="Path to the late-fusion config YAML")
    p_gw.add_argument("--head-path", required=True, help="Path to the trained head checkpoint")
    p_gw.add_argument("--rgb-glob", required=True, help="Glob for sample RGB test images")
    p_gw.add_argument("--num-images", type=int, default=25)
    p_gw.add_argument("--max-side", type=int, default=640)
    p_gw.add_argument("--device", default=None)
    p_gw.add_argument("--use-half", action="store_true")
    p_gw.set_defaults(func=cmd_gate_weights)

    p_gl = sp.add_parser("gate-last-layer", help="Inspect gating last-layer params, optionally with a probe image")
    p_gl.add_argument("--head-path", required=True)
    p_gl.add_argument("--probe-config-path", default=None, help="Optional config path to probe a single image through the gate")
    p_gl.add_argument("--probe-rgb-glob", default=None, help="Optional image glob for gate probing")
    p_gl.add_argument("--max-side", type=int, default=640)
    p_gl.add_argument("--device", default=None)
    p_gl.add_argument("--use-half", action="store_true")
    p_gl.set_defaults(func=cmd_gate_last_layer)

    p_cfg = sp.add_parser("print-config", help="Print selected head config fields")
    p_cfg.add_argument("--head-path", required=True)
    p_cfg.set_defaults(func=cmd_print_config)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScriptError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
