"""archsheet spec: the three operating modes of Compressed Sparse Attention 2 (paper figure 4).

    python -m archsheet examples/deepseek_csa2_modes_spec.py -o out/csa2_modes --png

Three consecutive decoder layers, one per mode, drawn as three regions side by side:

  Full     computes its own Main KV and Indexer K, scores with its Indexer Q, selects Top-K.
  Reindex  reuses Main KV / Indexer K of the Full layer, re-scores them with its own Indexer Q.
  Reuse    reuses Main KV *and* the Top-K indices; no indexer at all.

Every mode computes its own Main Q and SWA KV. The reused tensors (the yellow / red blocks of the paper
figure) are the lines that leave one region and enter a later one. Same stand-in conventions as
`deepseek_v41_flash_spec.py` (released shapes, plain torch ops instead of the custom kernels).
"""
import torch
import torch.nn as nn

from archsheet import label, phase

D, H, DH, ROPE = 5120, 64, 512, 64
RQ, RO, G = 1280, 1024, 8
IH, IDH, TOPK = 32, 128, 512
B, S = 5, 1040

FLAGS = dict(depth=2)


def rope(x):                                              # rotate the last 64 dims of each head
    return torch.cat([x[..., :-ROPE], x[..., -ROPE:] * 0.5 + x[..., -ROPE:].flip(-1) * 0.5], dim=-1)


class MainQ(nn.Module):
    """Low-rank query: wq_a -> norm -> wq_b. Returns the heads and the low-rank latent (the Indexer reads it)."""
    def __init__(self):
        super().__init__()
        self.wq_a, self.wq_b = nn.Linear(D, RQ, bias=False), nn.Linear(RQ, H * DH, bias=False)
        self.q_norm = nn.LayerNorm(RQ)

    def forward(self, x):
        qr = self.q_norm(self.wq_a(x))
        return rope(self.wq_b(qr).unflatten(-1, (H, DH))), qr


class SWAKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.wkv, self.norm = nn.Linear(D, DH, bias=False), nn.LayerNorm(DH)

    def forward(self, x):
        return rope(self.norm(self.wkv(x)))


class MainKV(nn.Module):                                  # compression ratio 1 in the decoder: a projection
    def __init__(self):
        super().__init__()
        self.wkv, self.norm = nn.Linear(D, DH, bias=False), nn.LayerNorm(DH)

    def forward(self, x):
        return self.norm(self.wkv(x))


class IndexerK(nn.Module):
    def __init__(self):
        super().__init__()
        self.wk, self.norm = nn.Linear(DH, IDH, bias=False), nn.LayerNorm(IDH)

    def forward(self, main_kv):
        return self.norm(self.wk(main_kv))


class IndexerQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq_b, self.weights_proj = nn.Linear(RQ, IH * IDH, bias=False), nn.Linear(D, IH, bias=False)

    def forward(self, x, qr):
        return self.wq_b(qr).unflatten(-1, (IH, IDH)), self.weights_proj(x)


class Indexer(nn.Module):
    """Scores every visible Main KV entry and keeps the Top-512."""
    def forward(self, q, w, index_k):
        score = torch.einsum("bshd,btd->bsht", q, index_k)
        return (score.relu() * w.unsqueeze(-1)).sum(2).topk(TOPK, dim=-1).indices


class Selection(nn.Module):
    def forward(self, main_kv, idxs):
        b, s, k = idxs.shape
        return torch.gather(main_kv.unsqueeze(1).expand(b, s, -1, DH), 2, idxs.unsqueeze(-1).expand(b, s, k, DH))


class Concatenation(nn.Module):
    def forward(self, swa_kv, selected):
        b, s, _ = swa_kv.shape
        return torch.cat([swa_kv.unsqueeze(1).expand(b, s, -1, DH), selected], dim=2)


class CoreAttention(nn.Module):
    def forward(self, q, kv):
        p = torch.einsum("bshd,bskd->bshk", q, kv).softmax(-1)
        return torch.einsum("bshk,bskd->bshd", p, kv)


class AttentionOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.wo_a = nn.Parameter(torch.empty(G, RO, H * DH // G))
        self.wo_b = nn.Linear(G * RO, D, bias=False)

    def forward(self, o):
        o = rope(o).reshape(o.size(0), o.size(1), G, -1)
        return self.wo_b(torch.einsum("bsgd,grd->bsgr", o, self.wo_a).flatten(2))


class CacheRead(nn.Module):
    """A tensor read back from the layer that produced it (the yellow / red blocks of the paper figure)."""
    def forward(self, x):
        return x.clone()


class CSA2Layer(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.main_q, self.swa_kv = MainQ(), SWAKV()
        if mode == "Full":
            self.main_kv, self.indexer_k = MainKV(), IndexerK()
        if mode in ("Full", "Reindex"):
            self.indexer_q, self.indexer = IndexerQ(), Indexer()
        if mode != "Full":
            self.main_kv_cache = CacheRead()
        if mode == "Reindex":
            self.indexer_k_cache = CacheRead()
        if mode == "Reuse":
            self.topk_cache = CacheRead()
        self.selection, self.concatenation = Selection(), Concatenation()
        self.core_attention, self.output = CoreAttention(), AttentionOutput()

    def forward(self, x, main_kv=None, index_k=None, topk=None):
        q, qr = self.main_q(x)
        swa = self.swa_kv(x)
        if self.mode != "Full":
            main_kv = self.main_kv_cache(main_kv)
        if self.mode == "Reindex":
            index_k = self.indexer_k_cache(index_k)
        if self.mode == "Reuse":
            topk = self.topk_cache(topk)
        if self.mode == "Full":
            main_kv = label(self.main_kv(x), "Main KV")
            index_k = label(self.indexer_k(main_kv), "Indexer K")
        if self.mode in ("Full", "Reindex"):
            iq, w = self.indexer_q(x, qr)
            topk = label(self.indexer(iq, w, index_k), "Top-K indices")
        kv = self.concatenation(swa, self.selection(main_kv, topk))
        return self.output(self.core_attention(q, kv)), main_kv, index_k, topk


class CSA2Modes(nn.Module):
    def __init__(self):
        super().__init__()
        self.full, self.reindex, self.reuse = CSA2Layer("Full"), CSA2Layer("Reindex"), CSA2Layer("Reuse")

    def forward(self, x_full, x_reindex, x_reuse):
        """Three consecutive layers; each reads its own hidden states, so the only lines between the
        regions are the reused tensors."""
        with phase("Full Mode"):
            y_full, main_kv, index_k, topk = self.full(x_full)
        with phase("Reindex Mode"):
            y_reindex, main_kv, _, topk = self.reindex(x_reindex, main_kv, index_k)
        with phase("Reuse Mode"):
            y_reuse, *_ = self.reuse(x_reuse, main_kv, None, topk)
        return y_full, y_reindex, y_reuse


def build():
    return dict(
        model=CSA2Modes(),
        inputs=dict(x_full=torch.zeros(B, S, D), x_reindex=torch.zeros(B, S, D), x_reuse=torch.zeros(B, S, D)),
        symbols={
            "b": (B, "batch"), "s": (S, "tokens"), "d": (D, "hidden size"), "h": (H, "query heads"),
            "rq": (RQ, "query low-rank dim"), "ro": (RO, "output low-rank dim"), "g": (G, "output groups"),
            "ih": (IH, "indexer heads"), "di": (IDH, "indexer head dim"),
            "dh": (DH, "head dim (= Top-K, both 512)"), "hd": (H * DH, "h · dh"),
            "hg": (H * DH // G, "h · dh / g  (= ih · di)"),
        },
        title="CSA2: Full / Reindex / Reuse",
    )
