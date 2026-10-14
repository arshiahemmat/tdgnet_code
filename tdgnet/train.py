"""BCE training and response/token AUROC on caller-supplied traces."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from tdgnet.data import TraceSample, collate_traces
from tdgnet.model import TDGNet


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(samples, batch_size, *, shuffle=False):
    if not samples:
        raise ValueError("trace collection is empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return DataLoader(list(samples), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_traces)


def _check_threshold(model, samples):
    for sample in samples:
        minimum = float(sample.metadata.get("attention_storage_tau_min", 0.0))
        if model.tau < minimum:
            raise ValueError(f"tau={model.tau} is below the stored attention threshold {minimum}")


def _forward(model, batch, device):
    return model(features=batch["features"].to(device),
                 attentions=batch["attentions"].to(device),
                 response_mask=batch["response_mask"].to(device),
                 node_mask=batch["node_mask"].to(device))


@torch.no_grad()
def _evaluate(model, samples, *, head, batch_size, device):
    if head not in model.trained_heads:
        raise ValueError(f"the {head} head has not been trained")
    _check_threshold(model, samples)
    model.to(device).eval()
    predictions, targets = [], []
    for batch in _loader(samples, batch_size):
        out = _forward(model, batch, device)
        if head == "response":
            y = batch["labels"].to(device)
            logits = out.response_logit
        else:
            y = batch["token_labels"].to(device)
            logits = out.token_logits
        valid = ~torch.isnan(y)
        predictions.extend(torch.sigmoid(logits[valid]).cpu().tolist())
        targets.extend(y[valid].cpu().tolist())
    auroc = float(roc_auc_score(targets, predictions)) if len(set(targets)) == 2 else float("nan")
    return {"n": len(targets), "auroc": auroc}


def evaluate_response_auroc(model: TDGNet, samples: Sequence[TraceSample],
                            batch_size: int = 16,
                            device: Union[str, torch.device] = "cpu") -> Dict[str, float]:
    return _evaluate(model, samples, head="response", batch_size=batch_size, device=device)


def evaluate_token_auroc(model: TDGNet, samples: Sequence[TraceSample],
                         batch_size: int = 16,
                         device: Union[str, torch.device] = "cpu") -> Dict[str, float]:
    return _evaluate(model, samples, head="token", batch_size=batch_size, device=device)


@dataclass
class TrainResult:
    best_val_auroc: float
    history: List[Dict[str, float]]
    model: TDGNet
    best_epoch: int
    selection_head: str


def _label_values(samples, head):
    values = []
    for sample in samples:
        if head == "response":
            if sample.label is not None:
                values.append(float(sample.label))
        elif sample.token_labels is not None:
            t = sample.token_labels.float()
            if t.numel() == sample.num_nodes:
                t = t[sample.response_mask.to(t.device)]
            values.extend(t[~torch.isnan(t)].tolist())
    return values


def train_tdgnet(
    model: TDGNet, train_samples: Sequence[TraceSample], val_samples: Sequence[TraceSample],
    *, epochs: int = 30, lr: float = 1e-3, batch_size: int = 16,
    device: Union[str, torch.device] = "cpu", seed: int = 42,
    task: str = "response", token_loss_weight: float = 1.0,
    patience: int = 5, verbose: bool = False,
) -> TrainResult:
    """Train response, token or joint heads and select by validation AUROC.

    Token supervision uses token_labels, with prompt/padding/unlabelled
    positions excluded. Token-only training selects on token AUROC;
    response and joint training select on response AUROC.
    """
    if task not in ("response", "token", "joint"):
        raise ValueError("task must be response, token or joint")
    if epochs < 1 or batch_size < 1 or not math.isfinite(lr) or lr < 0:
        raise ValueError("require positive epochs/batch_size and finite nonnegative lr")
    if task == "joint" and (not math.isfinite(token_loss_weight) or token_loss_weight <= 0):
        raise ValueError("joint training requires a positive token_loss_weight")
    if not train_samples or not val_samples:
        raise ValueError("training and validation traces must be non-empty")
    heads = ["response", "token"] if task == "joint" else [task]
    for head in heads:
        if not _label_values(train_samples, head):
            raise ValueError(f"no {head} supervision in training traces")
    selection = "token" if task == "token" else "response"
    if set(_label_values(val_samples, selection)) != {0.0, 1.0}:
        raise ValueError(f"validation {selection} labels must contain both classes for AUROC")
    _check_threshold(model, train_samples)
    _check_threshold(model, val_samples)
    set_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = _loader(train_samples, batch_size, shuffle=True)
    best_state, best_score, best_epoch, stale = None, -1.0, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            y = batch["labels"].to(device)
            token_y = batch["token_labels"].to(device)
            response_valid, token_valid = ~torch.isnan(y), ~torch.isnan(token_y)
            use_response = "response" in heads and bool(response_valid.any())
            use_token = "token" in heads and bool(token_valid.any())
            if not use_response and not use_token:
                continue
            out = _forward(model, batch, device)
            terms = []
            if use_response:
                terms.append(F.binary_cross_entropy_with_logits(out.response_logit[response_valid], y[response_valid]))
            if use_token:
                weight = token_loss_weight if task == "joint" else 1.0
                terms.append(weight * F.binary_cross_entropy_with_logits(out.token_logits[token_valid], token_y[token_valid]))
            loss = sum(terms)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.mark_trained(response=use_response, token=use_token)
            losses.append(float(loss.item()))
        if not losses:
            raise ValueError("no supervised training batches")
        metrics = _evaluate(model, val_samples, head=selection, batch_size=batch_size, device=device)
        score = metrics["auroc"]
        if not math.isfinite(score):
            raise ValueError("validation AUROC is undefined")
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auroc": score}
        history.append(row)
        if verbose:
            print(f"epoch {epoch}: loss={row['train_loss']:.4f} val_auroc={score:.4f}")
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                break
    model.load_state_dict(best_state)
    return TrainResult(best_score, history, model, best_epoch, selection)


__all__ = ["set_seed", "TrainResult", "train_tdgnet", "evaluate_response_auroc", "evaluate_token_auroc"]
