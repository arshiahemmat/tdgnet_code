#!/usr/bin/env python3
"""Evaluate the trained heads of a checkpoint on prepared traces."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tdgnet.checkpoint import load_checkpoint
from tdgnet.storage import load_trace_dir
from tdgnet.train import evaluate_response_auroc, evaluate_token_auroc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint, device=args.device)
    samples = load_trace_dir(args.test_dir)
    if not samples:
        raise SystemExit("test directory contains no traces")
    if not model.trained_heads:
        raise SystemExit("checkpoint contains no trained prediction head")
    metrics = {"trained_heads": model.trained_heads}
    for head, evaluate in [("response", evaluate_response_auroc), ("token", evaluate_token_auroc)]:
        if head in model.trained_heads:
            result = evaluate(model, samples, batch_size=args.batch_size, device=args.device)
            metrics[head + "_auroc"] = result["auroc"] if math.isfinite(result["auroc"]) else None
            metrics[head + "_n"] = result["n"]
        else:
            metrics[head + "_auroc"], metrics[head + "_n"] = None, 0
    text = json.dumps(metrics, indent=2, allow_nan=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
