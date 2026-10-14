# Trace format

TDGNet consumes prepared attention and hidden-state keyframes. All snapshots in one trace share a fixed node set and hidden-state width.

## Fields

| Field | Shape | Meaning |
|---|---|---|
| `attentions` | K tensors of `[N,N]` | Head-averaged probabilities; rows are queries, columns are keys |
| `features` | K tensors of `[N,D]` | Hidden states at the same token positions and keyframes |
| `response_mask` | Boolean `[N]` | True at response positions |
| `label` | Optional scalar | 1 for a hallucinated response, 0 otherwise |
| `token_labels` | Optional `[N]` or `[R]` | Binary supervision; R is the number of response positions |
| `metadata` | JSON-compatible dictionary | Optional information about the supplied trace |

Attention and features must be finite floating-point tensors. Attention scores lie in `[0,1]`. NaN token labels mark positions without supervision. Missing response supervision is represented by `label=None`.

Choose the node set when preparing traces. `response_mask` controls token supervision and optional response-only graph pooling. Default graph pooling includes every valid node in the supplied graph.

`collate_traces` pads variable N and returns `node_mask` alongside the data. K and D must be shared within a batch. Prompt positions, padding positions and missing token labels are excluded from token loss and evaluation.

## Keyframes

Supply snapshots in denoising order, from fully masked to fully denoised. For T denoising transitions, a complete trajectory contains T+1 state snapshots including both endpoints.

State index u corresponds to diffusion time t=T-u. The diffusion times `{T, floor(T/2), floor(T/4), 0}` therefore map to:

```text
0, T-floor(T/2), T-floor(T/4), T
```

For T=16, `select_trace_keyframes` selects `[0,8,12,16]` from 17 snapshots. The endpoint contains the hidden states and attention of the fully denoised sequence.

```python
from tdgnet import select_trace_keyframes

# Each list contains T+1 aligned state snapshots.
sample = select_trace_keyframes(
    attentions=attention_states,
    features=hidden_states,
    response_mask=response_mask,
    label=response_label,
    token_labels=token_labels,
)
```

Prepared keyframes can also be passed directly to `TraceSample`; preserve their chronological order.

## Persistence

Use `tdgnet.data.save_trace` and `load_trace` for dense `.pt` traces. `tdgnet.storage.save_shard` stores attention sparsely and preserves tensor precision. Its `tau_min` is recorded in metadata: subsequent graph thresholds must be greater than or equal to that value.

Labels are supplied with the traces. Response training uses response labels; token training uses token labels. Joint training uses both. Validation AUROC requires both classes for the selected head.

[Back to the README](../README.md)
