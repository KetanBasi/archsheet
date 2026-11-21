"""Example spec: LLaVA / Gemma-3 style VLM. The image branch (vision tower -> multimodal
projector) *branches in* to the text branch at the input-embedding level: image features
overwrite the embeddings at <image> token positions, then the LLM runs on the merged hidden state.

    python -m archsheet examples/vlm_spec.py -o out/vlm --png
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

B, S, HW = 5, 23, 42                 # batch, text length (incl. image tokens), image size
P = 14                               # patch size
N = (HW // P) ** 2                   # image patches = image tokens = 9
DV, HV, DHV, FV, LV = 48, 6, 8, 96, 4          # vision: width, heads, head dim, ffn, layers
D, H, DH, F_, V, LT = 64, 4, 16, 128, 320, 4   # text:   width, heads, head dim, ffn, vocab, layers
IMG_TOKEN = 7


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class MHA(nn.Module):
    """Generic multi-head self attention (fused qkv optional)."""
    def __init__(self, dim, heads, causal):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.proj(o.transpose(1, 2).reshape(b, s, d))


class ViTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(DV), nn.LayerNorm(DV)
        self.attn = MHA(DV, HV, causal=False)
        self.mlp = nn.Sequential(nn.Linear(DV, FV), nn.GELU(), nn.Linear(FV, DV))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, DV, P, P)
        self.pos_embed = nn.Parameter(torch.zeros(1, N, DV))
        self.layers = nn.ModuleList([ViTBlock() for _ in range(LV)])
        self.post_norm = nn.LayerNorm(DV)

    def forward(self, pixel_values):
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.post_norm(x)


class MultiModalProjector(nn.Module):          # "mmproj"
    def __init__(self):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(DV, D), nn.GELU(), nn.Linear(D, D)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm, self.post_norm = RMSNorm(D), RMSNorm(D)
        self.self_attn = MHA(D, H, causal=True)
        self.gate, self.up, self.down = nn.Linear(D, F_), nn.Linear(D, F_), nn.Linear(F_, D)

    def forward(self, x):
        x = x + self.self_attn(self.input_norm(x))
        h = self.post_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TextLayer() for _ in range(LT)])
        self.norm = RMSNorm(D)

    def forward(self, inputs_embeds):
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = VisionTower()
        self.multi_modal_projector = MultiModalProjector()
        self.embed_tokens = nn.Embedding(V, D)
        self.language_model = LanguageModel()
        self.lm_head = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, pixel_values):
        image_features = self.multi_modal_projector(self.vision_tower(pixel_values))   # [b, n, d]
        inputs_embeds = self.embed_tokens(input_ids)                                    # [b, s, d]
        mask = (input_ids == IMG_TOKEN).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)             # <- branch-in
        return self.lm_head(self.language_model(inputs_embeds))


def build():
    return dict(
        model=TinyVLM(),
        inputs={"input_ids": torch.zeros(B, S, dtype=torch.long),
                "pixel_values": torch.zeros(B, 3, HW, HW)},
        symbols={
            "b": (B, "batch"), "s": (S, "text tokens incl. <image>"), "hw": (HW, "image size"),
            "n": (N, "image patches"), "dv": (DV, "vision width"), "d": (D, "d_model"),
            "fv": (FV, "vision ffn"), "f": (F_, "text ffn"), "v": (V, "vocab"),
            "hv": (HV, "vision heads"), "dhv": (DHV, "vision head dim"),
            "h": (H, "text heads"), "dh": (DH, "head dim"),
        },
        title="TinyVLM (vision tower → mmproj → LLM)",
    )
