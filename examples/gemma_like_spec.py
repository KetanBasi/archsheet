"""Example spec: a small Gemma-style decoder (sandwich RMSNorm, QK-norm, RoPE, GQA,
gated-GELU MLP, 5 local : 1 global attention pattern). Run:

    python -m archsheet examples/gemma_like_spec.py -o out/gemma_like --png
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- config -----------------------------------------------------------------
B, S = 3, 17                      # dynamic dims: choose values that appear nowhere else
D, H, HKV, DH, FF, V, L = 96, 8, 2, 20, 256, 512, 12
WINDOW = 8


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y * (1 + self.weight)


class Rotary(nn.Module):
    def __init__(self, dh, theta=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / theta ** (torch.arange(0, dh, 2).float() / dh), persistent=False)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).float()
        f = torch.outer(pos, self.inv_freq)
        emb = torch.cat([f, f], -1)
        return emb.cos()[None, None], emb.sin()[None, None]


def rotate_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat([-b, a], -1)


class Attention(nn.Module):
    def __init__(self, local):
        super().__init__()
        self.sliding_window = WINDOW if local else 0
        self.q_proj = nn.Linear(D, H * DH, bias=False)
        self.k_proj = nn.Linear(D, HKV * DH, bias=False)
        self.v_proj = nn.Linear(D, HKV * DH, bias=False)
        self.o_proj = nn.Linear(H * DH, D, bias=False)
        self.q_norm, self.k_norm = RMSNorm(DH), RMSNorm(DH)
        self.rope = Rotary(DH)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, H, DH).transpose(1, 2)
        k = self.k_proj(x).view(b, s, HKV, DH).transpose(1, 2)
        v = self.v_proj(x).view(b, s, HKV, DH).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = self.rope(x)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        k = k[:, :, None].expand(b, HKV, H // HKV, s, DH).reshape(b, H, s, DH)
        v = v[:, :, None].expand(b, HKV, H // HKV, s, DH).reshape(b, H, s, DH)
        mask = torch.ones(s, s, dtype=torch.bool, device=x.device).tril()
        if self.sliding_window:
            mask &= torch.ones(s, s, dtype=torch.bool, device=x.device).triu(-self.sliding_window)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(o.transpose(1, 2).reshape(b, s, H * DH))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(D, FF, bias=False)
        self.up_proj = nn.Linear(D, FF, bias=False)
        self.down_proj = nn.Linear(FF, D, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, local):
        super().__init__()
        self.input_layernorm = RMSNorm(D)
        self.self_attn = Attention(local)
        self.post_attention_layernorm = RMSNorm(D)
        self.pre_feedforward_layernorm = RMSNorm(D)
        self.mlp = MLP()
        self.post_feedforward_layernorm = RMSNorm(D)

    def forward(self, x):
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x)))
        return x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))


class TinyGemmaLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, D)
        self.layers = nn.ModuleList([Layer(local=(i % 6 != 5)) for i in range(L)])
        self.norm = RMSNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


def build():
    model = TinyGemmaLM()
    return dict(
        model=model,
        inputs={"input_ids": torch.zeros(B, S, dtype=torch.long)},
        symbols={
            "b": (B, "batch"), "s": (S, "sequence length"), "d": (D, "d_model"),
            "h": (H, "query heads"), "kv": (HKV, "kv heads"), "dh": (DH, "head dim"),
            "f": (FF, "d_ff"), "v": (V, "vocab"),
        },
        title="TinyGemmaLM (example)",
    )
