"""Example spec: two entry points sharing state, drawn as two dashed regions.

    shortie: source tokens -> Encoder layers -> Projector (stores K/V)
    tallie:  query tokens  -> Encoder layers -> Decoder layers  (+ stored K/V appended to every attention)

Uses the two helpers the tool provides to specs:
    phase("name")      group everything executed inside the `with` block into a dashed region
    label(tensor, "x") name the tensor flow that leaves its producer (shown on the edge)

    python -m archsheet examples/two_phase_spec.py -o out/two_phase --png
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from archsheet import label, phase

B, S, T = 3, 17, 29                      # batch, query length, source length
D, H, DH, FF, V, LE, LD = 64, 4, 16, 128, 320, 6, 6


class SelfAttn(nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.causal = causal
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))

    def forward(self, x, ext_k=None, ext_v=None):
        b, s, _ = x.shape
        sp = lambda t: t.view(b, -1, H, DH).transpose(1, 2)
        q, k, v = sp(self.q(x)), sp(self.k(x)), sp(self.v(x))
        if ext_k is not None:                       # stored K/V appended after the sequence
            k, v = torch.cat([k, ext_k], 2), torch.cat([v, ext_v], 2)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.o(o.transpose(1, 2).reshape(b, s, D))


class Block(nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = SelfAttn(causal)
        self.mlp = nn.Sequential(nn.Linear(D, FF), nn.GELU(), nn.Linear(FF, D))

    def forward(self, x, ext_k=None, ext_v=None):
        x = x + self.attn(self.norm1(x), ext_k, ext_v)
        return x + self.mlp(self.norm2(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.layers = nn.ModuleList([Block(causal=False) for _ in range(LE)])

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        return x


class Projector(nn.Module):
    """Projects encoder states to K/V and keeps them for later calls."""
    def __init__(self):
        super().__init__()
        self.k_proj, self.v_proj = nn.Linear(D, D), nn.Linear(D, D)
        self.k = self.v = None

    def forward(self, h):
        b, t, _ = h.shape
        sp = lambda x: x.view(b, t, H, DH).transpose(1, 2)
        self.k, self.v = sp(self.k_proj(h)), sp(self.v_proj(h))
        return self.k, self.v


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(causal=True) for _ in range(LD)])
        self.norm = nn.LayerNorm(D)

    def forward(self, x, ext_k, ext_v):
        for layer in self.layers:
            x = layer(x, ext_k, ext_v)
        return self.norm(x)


class TwoPhaseLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.projector = Encoder(), Projector()
        self.decoder, self.lm_head = Decoder(), nn.Linear(D, V, bias=False)

    def shortie(self, src_tokens):
        return self.projector(self.encoder(src_tokens))

    def forward(self, tokens):
        h = self.encoder(tokens)
        k, v = self.projector.k, self.projector.v
        label(k, "K · appended")
        label(v, "V · appended")
        return self.lm_head(self.decoder(h, k, v))


def build():
    model = TwoPhaseLM()
    src = torch.zeros(B, T, dtype=torch.long)
    q = torch.zeros(B, S, dtype=torch.long)

    def run(m):
        with phase("shortie"):
            m.shortie(src)
        with phase("tallie"):
            m(q)

    return dict(
        model=model, run=run, title="TwoPhaseLM: dual encoder tower",
        symbols={"b": (B, "batch"), "s": (S, "query length"), "t": (T, "source length"),
                 "d": (D, "d_model"), "h": (H, "heads"), "dh": (DH, "head dim"),
                 "f": (FF, "d_ff"), "v": (V, "vocab")},
    )
