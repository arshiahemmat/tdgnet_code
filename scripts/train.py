#!/usr/bin/env python3
"""Train TDGNet on directories of prepared traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tdgnet.checkpoint import save_checkpoint
from tdgnet.config import load_configs
from tdgnet.model import TDGNet
from tdgnet.storage import load_trace_dir
from tdgnet.train import set_seed, train_tdgnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--train_dir", type=Path, required=True)
    parser.add_argument("--val_dir", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--task", choices=["response", "token", "joint"], default=None)
    args = parser.parse_args()
    cfg = load_configs(args.config)
    train_cfg = dict(cfg.get("train", {}))
    if args.task is not None:
        train_cfg["task"] = args.task
    if args.device is not None:
        train_cfg["device"] = args.device
    device = train_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg["device"] = device
    train_cfg.setdefault("seed", 42)
    set_seed(int(train_cfg["seed"]))
    train_samples, val_samples = load_trace_dir(args.train_dir), load_trace_dir(args.val_dir)
    if not train_samples or not val_samples:
        raise SystemExit("training and validation directories must contain traces")
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["in_dim"] = train_samples[0].in_dim
    model = TDGNet(**model_cfg)
    result = train_tdgnet(model, train_samples, val_samples, **train_cfg, verbose=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(result.model, args.run_dir / "best.pt",
                    metadata={"task": train_cfg.get("task", "response"), "seed": train_cfg["seed"]})
    summary = {"best_val_auroc": result.best_val_auroc, "best_epoch": result.best_epoch,
               "selection_head": result.selection_head, "trained_heads": result.model.trained_heads,
               "model_config": result.model.get_config(), "history": result.history,
               "train_n": len(train_samples), "val_n": len(val_samples)}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
