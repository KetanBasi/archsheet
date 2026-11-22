"""archsheet spec: the three operating modes of Compressed Sparse Attention 2 (paper figure 4).

    python -m archsheet examples/deepseek_csa2_modes_spec.py -o out/csa2_modes --png

The attention modules are the real ones from `examples/deepseek_v41_flash/model.py` (released shapes, meta
device); nothing here re-implements them. Three decoder layers, one per mode, are run on the same input and
drawn as three regions side by side:

  Full     layer 20 -- computes its own Main KV and Indexer K, scores with its Indexer Q, selects Top-K.
  Reindex  layer 24 -- reuses Main KV / Indexer K of the Full layer, re-scores them with its own Indexer Q.
  Reuse    layer 25 -- reuses Main KV *and* the Top-K indices; no indexer at all.

The mode of a layer is decided by `kv_source_layers` / `index_source_layers`, so the only thing the spec adds
is `SharedAttention` (the object layers hand their results down in) and one `phase()` per layer. The reused
tensors (the yellow / red blocks of the paper figure) are the lines that leave one region and enter a later one.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepseek_v41_flash.model import ModelArgs, SharedAttention, Transformer  # noqa: E402

from archsheet import phase  # noqa: E402

FLAGS = dict(
    depth=2,
    trace_methods="deepseek_v41_flash.model:sparse_attention,deepseek_v41_flash.model:rotate_tail",
)

ARGS = ModelArgs.flash()
B, S = 5, 1040
MODES = (("Full", 20), ("Reindex", 24), ("Reuse", 25))


class CSA2Modes(nn.Module):
    def __init__(self):
        super().__init__()
        full = Transformer(ARGS)
        self.layers = nn.ModuleDict({str(i): full.layers[i].attn for _, i in MODES})   # only the attention modules

    def forward(self, x):
        shared, out = SharedAttention(), []
        for name, i in MODES:
            with phase(f"{name} · layer {i}"):
                out.append(self.layers[str(i)](x, shared))
        return out


def build():
    with torch.device("meta"):
        model = CSA2Modes()
    d = ARGS
    return dict(
        model=model,
        inputs=dict(x=torch.zeros(B, S, d.dim)),
        symbols={
            "b": (B, "batch"), "s": (S, "tokens"), "d": (d.dim, "hidden size"), "h": (d.n_heads, "query heads (= rope dims, both 64)"),
            "rq": (d.q_lora_rank, "query low-rank dim"), "r": (d.o_lora_rank, "output low-rank dim (per group)"),
            "g": (d.o_groups, "output groups"), "ih": (d.index_n_heads, "indexer heads (= rope pairs, both 32)"),
            "di": (d.index_head_dim, "indexer head dim"), "dh": (d.head_dim, "head dim"),
            "dn": (d.head_dim - d.rope_head_dim, "head dims without rope"),
            "hg": (d.n_heads * d.head_dim // d.o_groups, "h · dh / g"),
            "kt": (d.window_size + d.index_topk, "attended positions: window 128 + top-k 512"),
        },
        title="CSA2: Full / Reindex / Reuse",
    )
