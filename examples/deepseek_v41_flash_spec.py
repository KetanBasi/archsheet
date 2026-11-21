"""archsheet spec for DeepSeek-V4.1-Flash  (paper figure 3, "Overall architecture").

    python -m archsheet examples/deepseek_v41_flash_spec.py -o out/deepseek_overview --depth 2 --png
    python -m archsheet examples/deepseek_v41_flash_spec.py -o out/deepseek_detail   --depth 4 --png

This is a *structural stand-in*, not the released code: the real `inference/model.py` calls custom
fp8/fp4 kernels, keeps KV caches in buffers and routes experts with `.tolist()`, none of which can run on
the `meta` device. The stand-in keeps the module hierarchy, the tensor shapes and the dataflow of the
released code (names such as `wq_a`, `wkv`, `compressor`, `indexer`, `wo_a`, `hc_*` are the same) and
uses the released config (hidden 5120, 40 layers = 20 encoder + 20 decoder, 64 heads of 512, 384 routed
experts, top-6 ...). Replaced by plain torch ops: the Sinkhorn normalisation of the mHC mixing matrix
(softmax), `sparse_attn` (gather + softmax), the routed experts (one SwiGLU shown, weights of all 384 kept
as a parameter so the parameter count is right), quantisation (dropped).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from archsheet import label, phase

# ── released config (config.json) ─────────────────────────────────────────────
D, H, DH, ROPE = 5120, 64, 512, 64
RQ, RO, G = 1280, 1024, 8                  # q low-rank, o low-rank, o groups
HC = 4                                     # mHC residual streams
E, K, FM, LIM = 384, 6, 2304, 10.0         # routed experts, active, expert width, swiglu clamp
V = 129280
IH, IDH = 32, 128                          # indexer heads / head dim
TOPK = 512
WIN = 128
N_ENC = N_DEC = 20
ENGRAM_LAYERS = (1, 14)                    # encoder layer ids carrying an Engram
HASHES = 24                                # 3 n-gram orders x 8 hash heads
ENGRAM_ROWS, ENGRAM_DIM = 384_006_168, 256

B, S = 5, 1040                             # example batch / chunk length (S divisible by 2 and 8)
M = 37                                     # image tokens after the 3x3 pixel-unshuffle
NP = 9 * M                                 # image patches before it

FLAGS = dict(leaf_types="DeepSeekViT,SparseAttn")      # drawn as one box each


# ── small pieces ──────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20)) * self.weight


class RoPE(nn.Module):
    """Rotates the last `ROPE` dims of every head (the other 448 are left alone)."""
    def forward(self, x):
        nope, rot = x[..., :-ROPE], x[..., -ROPE:]
        half = ROPE // 2
        rot = torch.cat([-rot[..., half:], rot[..., :half]], dim=-1) * 0.5 + rot * 0.5
        return torch.cat([nope, rot], dim=-1)


class GroupedLinear(nn.Module):
    """wo_a: block-diagonal over the o-groups (each group projects only its own heads)."""
    def __init__(self, groups, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, d_out, d_in))

    def forward(self, o):                                   # [b, s, g, d_in]
        return torch.einsum("bsgd,grd->bsgr", o, self.weight)


class SparseAttn(nn.Module):
    """Sliding-window attention over `win_kv` plus, if given, the `idxs`-selected entries of
    `comp_kv`. One shared KV head (MQA), learned attention sink."""
    def __init__(self):
        super().__init__()
        self.attn_sink = nn.Parameter(torch.zeros(H))

    def forward(self, q, win_kv, comp_kv=None, idxs=None):
        b, s, h, dh = q.shape
        sc = torch.einsum("bshd,btd->bsht", q, win_kv)                      # window part (causal mask)
        v = win_kv
        if comp_kv is not None:
            sel = torch.gather(comp_kv.unsqueeze(1).expand(b, s, -1, dh), 2,
                               idxs.unsqueeze(-1).expand(b, s, idxs.size(-1), dh))
            sc = torch.cat([sc, torch.einsum("bshd,bskd->bshk", q, sel)], dim=-1)
            v = torch.cat([win_kv.unsqueeze(1).expand(b, s, -1, dh), sel], dim=2)
            p = sc.softmax(-1)
            return torch.einsum("bshk,bskd->bshd", p, v)
        return torch.einsum("bsht,btd->bshd", sc.softmax(-1), win_kv)


class Shared:
    """What attention layers hand down the stack instead of recomputing (as in the released code)."""
    compress_kv = index_k = topk_idxs = candidates = None


shared = Shared()


class Compressor(nn.Module):
    """Pools `ratio` tokens into one KV latent with a learned softmax gate (ratio 1: plain projection)."""
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio
        self.wkv = nn.Linear(D, DH, bias=False)
        if ratio > 1:
            self.wgate = nn.Linear(D, DH, bias=False)
        self.norm = RMSNorm(DH)

    def forward(self, x):
        if self.ratio == 1:
            return self.norm(self.wkv(x))
        b, s, _ = x.shape
        kv = self.wkv(x.float()).unflatten(1, (-1, self.ratio))
        sc = self.wgate(x.float()).unflatten(1, (-1, self.ratio))
        return self.norm((kv * sc.softmax(dim=2)).sum(dim=2))


class Indexer(nn.Module):
    """Scores the compressed positions with a small side attention and keeps the top-512."""
    def __init__(self, owns_k, uses_candidates, is_candidate_source):
        super().__init__()
        self.uses_candidates, self.is_candidate_source = uses_candidates, is_candidate_source
        self.wq_b = nn.Linear(RQ, IH * IDH, bias=False)
        self.weights_proj = nn.Linear(D, IH, bias=False)
        self.owns_k = owns_k
        if owns_k:
            self.wk = nn.Linear(DH, IDH, bias=False)
            self.k_norm = RMSNorm(IDH)
        self.rope = RoPE()

    def forward(self, x, qr, latent):
        if self.owns_k:
            shared.index_k = self.k_norm(self.wk(latent))
        q = self.wq_b(qr).unflatten(-1, (IH, IDH))
        w = self.weights_proj(x)
        score = torch.einsum("bshd,btd->bsht", q, shared.index_k)
        score = (score.relu() * w.unsqueeze(-1)).sum(dim=2)                 # [b, s, t]
        if self.is_candidate_source:                                        # level 1: blocks of 8
            blk = score.unflatten(-1, (-1, 8)).amax(-1)
            shared.candidates = blk.topk(min(2048, blk.size(-1)), dim=-1).indices
        elif self.uses_candidates:                                          # level 2: inside the pool
            score = score.masked_fill(~F.one_hot(shared.candidates, score.size(-1) // 8)
                                      .sum(-2).bool().repeat_interleave(8, -1), -1e9)
        return score.topk(min(TOPK, score.size(-1)), dim=-1).indices


class Attention(nn.Module):
    """SWA (ratio 0) or CSA2. `mode` is Full / Reindex / Reuse: Full owns the compressed KV and the
    indexer keys, Reindex owns an indexer only, Reuse owns neither and reads the shared state."""
    def __init__(self, ratio, mode, enc_kv=False, uses_candidates=False, is_candidate_source=False):
        super().__init__()
        self.mode = mode
        self.wq_a = nn.Linear(D, RQ, bias=False)
        self.q_norm = RMSNorm(RQ)
        self.wq_b = nn.Linear(RQ, H * DH, bias=False)
        self.wkv = nn.Linear(D, DH, bias=False)
        self.kv_norm = RMSNorm(DH)
        self.rope_q, self.rope_kv, self.rope_o = RoPE(), RoPE(), RoPE()
        self.sparse_attn = SparseAttn()
        self.wo_a = GroupedLinear(G, H * DH // G, RO)
        self.wo_b = nn.Linear(G * RO, D, bias=False)
        self.enc_kv = enc_kv
        if mode == "Full":
            self.compressor = Compressor(ratio)
        if mode in ("Full", "Reindex"):
            self.indexer = Indexer(mode == "Full", uses_candidates, is_candidate_source)

    def forward(self, x, enc_h=None):
        b, s, _ = x.shape
        qr = self.q_norm(self.wq_a(x))
        q = self.rope_q(self.wq_b(qr).unflatten(-1, (H, DH)))
        win = self.rope_kv(self.kv_norm(self.wkv(x)))
        if self.mode == "SWA":
            o = self.sparse_attn(q, win)
        else:
            latent = None
            if self.mode == "Full":
                latent = self.compressor(x if enc_h is None else enc_h)
            if self.mode in ("Full", "Reindex"):
                shared.topk_idxs = self.indexer(x, qr, latent)
            if latent is not None:
                shared.compress_kv = latent
            o = self.sparse_attn(q, win, shared.compress_kv, shared.topk_idxs)
        o = self.rope_o(o).reshape(b, s, G, -1)
        return self.wo_b(self.wo_a(o).flatten(2))


# ── DeepSeekMoE ───────────────────────────────────────────────────────────────
class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(E, D))
        self.bias = nn.Parameter(torch.zeros(E))
        self.bias_vl = nn.Parameter(torch.zeros(E))        # separate correction bias for image tokens

    def forward(self, x, image_mask):
        scores = F.softplus(F.linear(x.float(), self.weight)).sqrt()
        bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, self.bias)
        idx = (scores + bias).topk(K, dim=-1).indices
        w = scores.gather(-1, idx)
        return w / w.sum(-1, keepdim=True) * 1.5, idx


class Expert(nn.Module):
    """SwiGLU; the up branch is clamped both ways, the gate branch from above."""
    def __init__(self):
        super().__init__()
        self.w1, self.w3 = nn.Linear(D, FM, bias=False), nn.Linear(D, FM, bias=False)
        self.w2 = nn.Linear(FM, D, bias=False)

    def forward(self, x):
        g = self.w1(x).float().clamp(max=LIM)
        u = self.w3(x).float().clamp(-LIM, LIM)
        return self.w2((F.silu(g) * u).to(x.dtype))


class RoutedExperts(nn.Module):
    """384 experts, top-6 per token. One SwiGLU is traced as the representative; `bank` holds the
    weights of all of them so the parameter count is the real one."""
    def __init__(self):
        super().__init__()
        self.expert = Expert()
        self.bank = nn.Parameter(torch.empty(E - 1, 3, FM, D))

    def forward(self, x, weights):
        y = self.expert(x)                                           # [b, s, d]
        return (y.unsqueeze(2) * weights.unsqueeze(-1)).sum(2)       # weighted sum over the 6 picks


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate, self.routed, self.shared_experts = Gate(), RoutedExperts(), Expert()

    def forward(self, x, image_mask):
        w, idx = self.gate(x, image_mask)
        return self.routed(x, w) + self.shared_experts(x)


# ── mHC: HC parallel residual streams ─────────────────────────────────────────
class HCMix(nn.Module):
    """One projection of the flattened stream gives pre / post / comb (Sinkhorn replaced by softmax)."""
    def __init__(self):
        super().__init__()
        self.fn = nn.Parameter(torch.empty((2 + HC) * HC, HC * D))
        self.base = nn.Parameter(torch.zeros((2 + HC) * HC))

    def forward(self, x):                                            # [b, s, hc, d]
        f = x.flatten(2).float()
        m = F.linear(f, self.fn) * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6) + self.base
        pre, post, comb = m[..., :HC], m[..., HC:2 * HC], m[..., 2 * HC:]
        return pre.sigmoid(), 2 * post.sigmoid(), comb.unflatten(-1, (HC, HC)).softmax(-1)


class HCPre(nn.Module):
    def forward(self, x, pre):                                       # collapse hc copies -> 1
        return (pre.unsqueeze(-1) * x.float()).sum(2).to(x.dtype)


class HCPost(nn.Module):
    def forward(self, y, residual, post, comb):                      # expand back, mix the residual in
        return post.unsqueeze(-1) * y.unsqueeze(2) + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(2)


class EngramGate(nn.Module):
    """Gates the retrieved n-gram memory into the stream."""
    def __init__(self):
        super().__init__()
        self.key_proj, self.value_proj = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
        self.norm_h, self.norm_k = RMSNorm(D), RMSNorm(D)

    def forward(self, h, mem):
        k, v = self.key_proj(mem), self.value_proj(mem)
        g = (self.norm_h(h) * self.norm_k(k).unsqueeze(2)).sum(-1) / D ** 0.5
        g = (g.abs().sqrt() * g.sign()).sigmoid()
        return h + g.unsqueeze(-1) * v.unsqueeze(2)


class Block(nn.Module):
    def __init__(self, ratio, mode, engram=False, **kw):
        super().__init__()
        self.attn = Attention(ratio, mode, **kw)
        self.ffn = MoE()
        self.attn_norm, self.ffn_norm = RMSNorm(D), RMSNorm(D)
        self.hc_attn, self.hc_ffn = HCMix(), HCMix()
        self.hc_pre, self.hc_post = HCPre(), HCPost()
        if engram:
            self.engram = EngramGate()

    def forward(self, x, pre_mix, image_mask, mem=None, enc_h=None):
        if hasattr(self, "engram"):
            x = self.engram(x, mem)
        res = x
        a_pre, a_post, a_comb = self.hc_attn(x)
        y = self.attn(self.attn_norm(self.hc_pre(x, pre_mix)), enc_h)
        x = self.hc_post(y, res, a_post, a_comb)
        res = x
        f_pre, f_post, f_comb = self.hc_ffn(x)
        y = self.ffn(self.ffn_norm(self.hc_pre(x, a_pre)), image_mask)
        return self.hc_post(y, res, f_post, f_comb), f_pre


# ── encoder / decoder ─────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """2 SWA layers, then 18 CSA2(ratio 2) layers: 3 groups of [Full, 5 x Reuse]."""
    def __init__(self):
        super().__init__()
        layers = []
        for i in range(N_ENC):
            eg = i in ENGRAM_LAYERS
            if i < 2:
                layers.append(Block(0, "SWA", eg))
            elif (i - 2) % 6 == 0:
                layers.append(Block(2, "Full", eg))
            else:
                layers.append(Block(2, "Reuse", eg))
        self.layers = nn.ModuleList(layers)
        self.hc_head = HCPre()

    def forward(self, h, pre_mix, image_mask, mem):
        for i, layer in enumerate(self.layers):
            h, pre_mix = layer(h, pre_mix, image_mask, mem[ENGRAM_LAYERS.index(i)] if i in ENGRAM_LAYERS else None)
        return h, pre_mix, self.hc_head(h, pre_mix)


class Decoder(nn.Module):
    """20 CSA2(ratio 1) layers: Full, 3 x Reuse, then 4 x [Reindex, 3 x Reuse]."""
    def __init__(self):
        super().__init__()
        layers = [Block(1, "Full", is_candidate_source=True)]
        for i in range(1, N_DEC):
            layers.append(Block(1, "Reindex", uses_candidates=True) if (i - 4) % 4 == 0 and i >= 4
                          else Block(1, "Reuse"))
        self.layers = nn.ModuleList(layers)
        self.hc_head = HCPre()

    def forward(self, h, pre_mix, image_mask, enc_h):
        targets = []                                           # inputs of the last 3 layers feed DSpark
        for i, layer in enumerate(self.layers):
            if i >= N_DEC - 3:
                targets.append(h.mean(dim=2))
            h, pre_mix = layer(h, pre_mix, image_mask, enc_h=enc_h if i == 0 else None)
        return h, pre_mix, torch.cat(targets, dim=-1)


class EngramMemory(nn.Module):
    """Hashed n-gram embedding tables of the two Engram layers (384M rows x 256 each)."""
    def __init__(self):
        super().__init__()
        self.tables = nn.ModuleList(nn.Embedding(ENGRAM_ROWS, ENGRAM_DIM) for _ in ENGRAM_LAYERS)
        self.proj = nn.ModuleList(nn.Linear(HASHES * ENGRAM_DIM, D, bias=False) for _ in ENGRAM_LAYERS)

    def forward(self, hashes):                                        # [b, s, 24] int
        return [p(t(hashes).flatten(2)) for t, p in zip(self.tables, self.proj)]


class DeepSeekViT(nn.Module):
    """Vision tower stand-in: patches -> 32-layer ViT -> 3x3 pixel-unshuffle -> 2-layer MLP projector."""
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(14 * 14 * 3, 1024, bias=False)
        self.blocks = nn.Sequential(*[nn.Linear(1024, 1024, bias=False) for _ in range(2)])
        self.proj1, self.proj2 = nn.Linear(1024 * 9, D, bias=False), nn.Linear(D, D, bias=False)

    def forward(self, patches):                                       # [b, n, 588]
        x = self.blocks(self.patch_embed(patches))
        x = x.unflatten(1, (-1, 9)).flatten(2)                        # pixel-unshuffle 3x3
        return self.proj2(F.gelu(self.proj1(x)))


class DSpark(nn.Module):
    """Speculative-decoding draft head: reads the hidden states of the last three layers."""
    def __init__(self):
        super().__init__()
        self.main_proj = nn.Linear(3 * D, D, bias=False)
        self.main_norm = RMSNorm(D)
        self.block = Block(0, "SWA")
        self.norm = RMSNorm(D)

    def forward(self, targets, h, pre_mix, image_mask):
        main = self.main_norm(self.main_proj(targets))
        y, _ = self.block(h, pre_mix, image_mask)
        return self.norm(y.mean(2) + main)


class DeepSeekV41(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.vision = DeepSeekViT()
        self.image_start = nn.Parameter(torch.zeros(D))
        self.engram = EngramMemory()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        self.dspark = DSpark()
        self.register_buffer("image_mask", torch.zeros(B, S, dtype=torch.bool), persistent=False)

    def forward(self, input_ids, patches, ngram_hashes):
        image_mask = self.image_mask[:, :input_ids.size(1)]      # token modality: model state, not a drawn input
        with phase("Causal Encoder"):
            h = self.embed(input_ids)
            h = h.masked_scatter(image_mask.unsqueeze(-1), self.vision(patches))     # image tokens
            h = h.unsqueeze(2).repeat(1, 1, HC, 1)                                   # expand to HC streams
            pre_mix = F.one_hot(torch.zeros_like(input_ids), HC).float()
            mem = self.engram(ngram_hashes)
            h, pre_mix, enc_h = self.encoder(h, pre_mix, image_mask, mem)
            label(enc_h, "Enc. hidden states")
        with phase("Decoder"):
            h, pre_mix, targets = self.decoder(h, pre_mix, image_mask, enc_h)
            x = self.decoder.hc_head(h, pre_mix)
            logits = self.head(self.norm(x)[:, -1])
            draft = self.dspark(targets, h, pre_mix, image_mask)
        return logits, draft


def build():
    model = DeepSeekV41()
    ids = torch.zeros(B, S, dtype=torch.long)
    return dict(
        model=model,
        inputs=dict(
            input_ids=ids,
            patches=torch.zeros(B, NP, 14 * 14 * 3),
            ngram_hashes=torch.zeros(B, S, HASHES, dtype=torch.long),
        ),
        symbols={
            "b": (B, "batch"), "s": (S, "tokens in the chunk"), "n": (NP, "image patches"), "m": (M, "image tokens"),
            "eg": (HASHES * ENGRAM_DIM, "Engram width (24 lookups x 256)"), "de": (ENGRAM_DIM, "Engram head dim"),
            "d": (D, "hidden size"), "h": (H, "query heads"), "hc": (HC, "mHC residual streams"),
            "rq": (RQ, "query low-rank dim"), "ro": (RO, "output low-rank dim (per group)"),
            "g": (G, "output groups"), "dh": (DH, "head dim (= Top-K, both 512)"), "hd": (H * DH, "h · dh"),
            "hg": (H * DH // G, "h · dh / g  (= ih · di)"),
            "E": (E, "routed experts"), "k": (K, "experts per token"), "fm": (FM, "expert width"),
            "ih": (IH, "indexer heads"), "di": (IDH, "indexer head dim"), "v": (V, "vocab"),
        },
        title="DeepSeek-V4.1-Flash",
    )
