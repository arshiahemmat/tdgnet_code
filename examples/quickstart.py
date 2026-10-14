#!/usr/bin/env python3
"""Synthetic CPU example: traces, joint training and checkpoint round-trip."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tdgnet import TDGNet, collate_traces, load_checkpoint, save_checkpoint, select_trace_keyframes
from tdgnet.train import set_seed, train_tdgnet


def make_samples(count=8, seed=42):
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for i in range(count):
        n, d, steps = 5 + i % 2, 8, 16
        attentions, features = [], []
        for state in range(steps + 1):
            a = torch.rand(n, n, generator=generator)
            attentions.append(a / a.sum(-1, keepdim=True))
            features.append(torch.randn(n, d, generator=generator) + state / steps * 0.1)
        mask = torch.arange(n) >= 2
        token_labels = (torch.arange(int(mask.sum())) % 2).float()
        samples.append(select_trace_keyframes(attentions, features, mask, label=float(i % 2),
                                              token_labels=token_labels,
                                              metadata={"synthetic": True, "example_id": i}))
    return samples


def main():
    torch.set_num_threads(1)
    set_seed(42)
    train_samples, val_samples = make_samples(), make_samples(4, seed=43)
    model = TDGNet(in_dim=8, hidden_dim=16)
    result = train_tdgnet(model, train_samples, val_samples, task="joint", epochs=2,
                         batch_size=4, device="cpu")
    batch = collate_traces(val_samples[:2])
    model.eval()
    with torch.no_grad():
        out = model(features=batch["features"], attentions=batch["attentions"],
                    response_mask=batch["response_mask"], node_mask=batch["node_mask"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        save_checkpoint(model, path, metadata={"synthetic": True})
        restored = load_checkpoint(path)
        with torch.no_grad():
            check = restored(features=batch["features"], attentions=batch["attentions"],
                             response_mask=batch["response_mask"], node_mask=batch["node_mask"])
        torch.testing.assert_close(out.response_logit, check.response_logit)
        torch.testing.assert_close(out.token_logits, check.token_logits)
    print("Synthetic CPU example completed.")
    print("Keyframe indices:", train_samples[0].metadata["keyframe_indices"])
    print("Response logits:", tuple(out.response_logit.shape))
    print("Token logits:", tuple(out.token_logits.shape))
    print("Finite loss:", all(torch.isfinite(torch.tensor(row["train_loss"])) for row in result.history))
    print("Trained heads:", restored.trained_heads)
    print("Checkpoint round-trip: passed")


if __name__ == "__main__":
    main()
