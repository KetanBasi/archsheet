"""DeepSeek-V4.1-Flash, adapted for TRAINING.

Adapted from deepseek-ai/DeepSeek-V4.1-Flash `inference/model.py` (MIT License, see ./LICENSE). The released file
is inference code: it generates one chunk at a time against KV caches, and its hot paths are custom fp8/fp4
kernels. This copy keeps the architecture, the module hierarchy and the parameter names (so a checkpoint maps
one-to-one, except the routed experts, see below) and changes what stops it from being trained:

  released                                          here
  ------------------------------------------------  -------------------------------------------------------
  `@torch.inference_mode()`, in-place `copy_`       differentiable: rotary and quantisation are out of place
  `start_pos`, ring-buffer KV caches, `kv_state`    one full sequence per call; the Compressor drops a
                                                    trailing partial group (it is only a decode-time carry)
  fp8 / fp4 GEMMs, `act_quant`, `fp4_act_quant`      plain `nn.Linear` in the default dtype; the quantisation
                                                    steps of the released model are NOT reproduced (add
                                                    fake-quant with a straight-through estimator if you train
                                                    quantisation-aware)
  `sparse_attn` (TileLang, forward only)            `sparse_attention`: gather + softmax with the sink, same
                                                    maths, any autograd backend
  `hc_split_sinkhorn` (TileLang, forward only)      `hc_split_sinkhorn`: the same Sinkhorn loop in torch
  routed experts: `.tolist()` + a Python loop over  `RoutedExperts`: stacked weights + `F.grouped_mm`, no host
  384 `Expert` modules                              sync (checkpoint: stack `experts.{i}.w{1,2,3}.weight`)
  expert-selection bias updated by the optimiser    `Gate.bias` / `bias_vl` are buffers (aux-loss-free
                                                    balancing updates them outside autograd)
  module-global `shared_attn`, `world_size`, ...    an explicit `SharedAttention` object per forward; no
                                                    tensor parallelism (shard with FSDP / EP outside)
  `ParallelHead` returns the last position only     logits for every position
  token sampling inside `forward`                   `forward` returns logits; `sample` is gone
  n-gram hashing with a tokenizer and a cache       `engram.NgramHasher`, a stateless function; its output
                                                    is an input of `forward`
  per-image Python loop writing into `h` in place   `encode_images` + one `masked_scatter`
  DSpark: one block per decode step, sampled        `forward_spec`: all blocks of a sequence at once with
  tokens feed the Markov head                       teacher forcing (the true next tokens feed it)

The indexer's parameters receive no gradient from the language-model loss: it only selects positions (top-k
indices), which is not differentiable. Train it with a separate indexer loss; none is included here.

The routed experts are the one place where parameter names change on purpose: 384 separate `nn.Linear`s do not
train at a useful speed, so each layer holds three stacked tensors instead.
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from .engram import EngramLayout
from .vision import Aligner, ViT

TEXT, IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = -1, 0, 1, 2, 3        # `token_types` values


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch's default dtype, restoring it even if the body raises."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@dataclass
class ModelArgs:
    """Field names are the inference `ModelArgs` of the release; `flash()` holds the released shapes."""

    vocab_size: int = 129280
    dim: int = 1024
    moe_inter_dim: int = 1024
    n_layers: int = 5
    n_mtp_layers: int = 1                      # DSpark stages after the backbone
    n_heads: int = 16
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: str = "sqrtsoftplus"           # softmax | sigmoid | sqrtsoftplus
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    # attention: latent q/kv projections, plus a LoRA-factorised output projection over o_groups
    q_lora_rank: int = 256
    head_dim: int = 128
    rope_head_dim: int = 32
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 256
    # sparse attention: every layer attends over a sliding window, and may add compressed KV on top
    window_size: int = 128
    compress_ratios: tuple[int, ...] = (0, 2, 2, 1, 1, 0)       # per layer, DSpark stages included
    kv_source_layers: tuple[int, ...] = (1, 3)                  # layers that compress their own KV ...
    index_source_layers: tuple[int, ...] = (1, 3)               # ... or run an indexer; the rest reuse it
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # indexer and two-level candidate pre-filter (candidate_source_layer < 0 turns the latter off)
    index_n_heads: int = 16
    index_head_dim: int = 64
    index_topk: int = 64
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    # hyper-connections: the residual stream is carried as hc_mult parallel copies
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # engram
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    # vision (VL); vision_n_layers == 0 disables the vision path
    vision_n_layers: int = 0
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0
    vision_downsample_ratio: int = 3
    # dspark draft head
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 0
    dspark_n_activated_experts: int = 0

    @property
    def vision_enabled(self) -> bool:
        return self.vision_n_layers > 0

    def get_moe_config(self, layer_id: int) -> tuple[int, int]:
        if layer_id < self.n_layers:
            return self.n_routed_experts, self.n_activated_experts
        return (self.dspark_n_routed_experts or self.n_routed_experts,
                self.dspark_n_activated_experts or self.n_activated_experts)

    @classmethod
    def flash(cls, **over) -> "ModelArgs":
        """The released DeepSeek-V4.1-Flash (config.json)."""
        kw = dict(
            vocab_size=129280, dim=5120, moe_inter_dim=2304, n_layers=40, n_mtp_layers=3, n_heads=64,
            n_routed_experts=384, n_shared_experts=1, n_activated_experts=6, route_scale=1.5, swiglu_limit=10.0,
            q_lora_rank=1280, head_dim=512, rope_head_dim=64, o_groups=8, o_lora_rank=1024,
            window_size=128,
            compress_ratios=(0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0),
            kv_source_layers=(2, 8, 14, 20), index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36),
            compress_rope_theta=160000.0, original_seq_len=65536, rope_theta=10000.0, rope_factor=16,
            index_n_heads=32, index_head_dim=128, index_topk=512,
            candidate_source_layer=20, candidate_topk_blocks=2048, candidate_block_size=8,
            hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
            engram_layer_ids=(1, 14), engram_num_embeddings=(384006168, 384016682), engram_max_ngram_size=4,
            engram_vocab_size=16000000, engram_n_heads=8, engram_head_dim=256,
            vision_n_layers=32, vision_dim=1024, vision_n_heads=16, vision_inter_dim=2816,
            dspark_block_size=5, dspark_noise_token_id=128799, dspark_target_layer_ids=(37, 38, 39),
            dspark_markov_rank=256, dspark_n_routed_experts=128, dspark_n_activated_experts=3,
        )
        kw.update(over)
        return cls(**kw)


# ------------------------------------------------------------------------------------------------- small pieces
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class GroupedLinear(nn.Module):
    """`wo_a`: block-diagonal over the output groups (each group projects only its own heads). The weight is
    stored as [groups * rank, d_in] like the released `wo_a`."""

    def __init__(self, groups: int, d_in: int, rank: int, dtype=None):
        super().__init__()
        self.groups, self.rank = groups, rank
        self.weight = nn.Parameter(torch.empty(groups * rank, d_in, dtype=dtype))

    def forward(self, o: torch.Tensor) -> torch.Tensor:                 # [.., g, d_in] -> [.., g, rank]
        return torch.einsum("...gd,grd->...gr", o, self.weight.view(self.groups, self.rank, -1).to(o.dtype))


@lru_cache(4)
def _rotary_table(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Rotary frequencies as complex exponentials, one row per position (YaRN when original_seq_len > 0:
    dimensions whose wavelength fits the training context keep their frequency, those far beyond it are divided
    by `factor`, the `beta_fast`..`beta_slow` band is faded across with a linear ramp)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate the last dim of `x`, taking adjacent pairs as complex numbers. `freqs_cis` must broadcast against
    x's complex view [.., d/2]. `inverse` conjugates the rotation: it removes the query's rotation from the
    attention output so the cache can stay in one shared rotated form."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    return torch.view_as_real(xc * freqs_cis).flatten(-2).to(x.dtype)


def rotate_tail(x: torch.Tensor, freqs_cis: torch.Tensor, rd: int, inverse: bool = False) -> torch.Tensor:
    """Rotate only the last `rd` dims of x (the rope part of a head); the other dims are left alone."""
    return torch.cat([x[..., :-rd], apply_rotary_emb(x[..., -rd:], freqs_cis, inverse)], dim=-1)


def window_idxs(seqlen: int, window: int, device) -> torch.Tensor:
    """Which kv rows each query attends to inside its sliding window: [seqlen, min(seqlen, window)], -1 where
    the window reaches before the start of the sequence."""
    end = torch.arange(seqlen, device=device).unsqueeze(1)
    idxs = (end - window + 1).clamp(min=0) + torch.arange(min(seqlen, window), device=device)
    return torch.where(idxs > end, -1, idxs)


def sparse_attention(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, idxs: torch.Tensor,
                     softmax_scale: float) -> torch.Tensor:
    """q: [b, s, h, d]; kv: [b, n, d] (one shared KV head); idxs: [b, s, k] rows of kv each query reads, -1 =
    none. Softmax over the gathered rows plus a learned per-head sink logit that only absorbs probability mass.
    A query with no valid row returns zeros. Same maths as the released `sparse_attn` kernel."""
    b, s, h, _ = q.shape
    valid = idxs >= 0
    rows = kv[torch.arange(b, device=kv.device)[:, None, None], idxs.clamp(min=0)]            # [b, s, k, d]
    scores = torch.einsum("bshd,bskd->bshk", q.float(), rows.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
    probs = torch.cat([scores, sink], dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probs, rows.float()).to(q.dtype)


# ------------------------------------------------------------------------------------------------- engram
class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches that stream.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per hc copy plus a shared value. The
    gate is a normalized dot product of stream against key."""

    def __init__(self, args: ModelArgs, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.clamp_value = 1e-6
        self.embed = nn.Embedding(layout.num_embeddings[self.layer_hash_index], layout.head_dim)
        self.wkv = nn.Linear(layout.n_hash_cols * layout.head_dim, args.dim * (args.hc_mult + 1), bias=False)
        self.eps = args.norm_eps
        self.q_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))
        self.k_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, L, hc_mult, dim]; hash_ids: [B, L, n_hash_cols]; token_mask: [B, L], False shuts the gate so
        those positions pass through untouched."""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()          # only ever used as a product
        h, eps = x.float(), self.eps
        # normalized per (token, hc copy) over `dim`, NOT jointly over the copies
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        # signed sqrt before the sigmoid, matching the training kernel
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


# ------------------------------------------------------------------------------------------------- attention
class SharedAttention:
    """What attention layers hand down the stack instead of recomputing. Layers run in order and every source
    writes before its consumers read. One per forward call: compress_kv and index_k from kv_source_layers,
    topk_idxs from index_source_layers, candidates from candidate_source_layer."""

    def __init__(self):
        self.compress_kv = None
        self.index_k = None
        self.topk_idxs = None
        self.candidates = None


class Compressor(nn.Module):
    """Pools `compress_ratio` consecutive tokens into one KV latent with a learned softmax gate. Returns the
    latent before RoPE (the indexer needs the unrotated form, so Attention rotates afterwards)."""

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        ratio = args.compress_ratios[layer_id]
        self.compress_ratio = ratio
        self.head_dim = args.head_dim
        self.norm = RMSNorm(args.head_dim, args.norm_eps)
        # ratio 1 is a plain projection; the softmax pooling above ratio 1 runs in fp32
        self.wkv = nn.Linear(args.dim, args.head_dim, bias=False, dtype=torch.float32 if ratio > 1 else None)
        if ratio > 1:
            self.wgate = nn.Linear(args.dim, args.head_dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:                                      # one token per group: nothing to pool
            return self.norm(self.wkv(x))
        n = x.size(1) // ratio * ratio                      # a trailing partial group has no latent yet
        x = x[:, :n].float()
        kv = self.wkv(x).unflatten(1, (-1, ratio))
        score = self.wgate(x).unflatten(1, (-1, ratio))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        return self.norm(kv.to(dtype))


def select_candidate_blocks(logits: torch.Tensor, compress_lens: torch.Tensor, topk_blocks: int,
                            block_size: int) -> torch.Tensor:
    """Level one of the two-level top-k: keep the `topk_blocks` highest-scoring blocks per query. `logits` is
    [.., n_positions] with positions the query cannot reach already at -inf. Returns a bool mask shaped like
    `logits`, so the layers consuming it just mask and never think about blocks again."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)        # a block scores as its best position
    num_blocks = scores.size(-1)
    # the block holding this query's newest position is only partly filled: pin it in
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer(nn.Module):
    """Keeps the `index_topk` best compressed positions per query: a small side attention, fp4 query heads
    against one shared key per compressed position, scores rectified then combined by `weights_proj`. With a
    candidate source this is the second of two levels; `select_candidate_blocks` is the first."""

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.owns_k = layer_id in args.kv_source_layers      # only a layer that compresses its own KV makes keys
        self.compress_ratio = args.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.n_heads = args.index_n_heads
        self.index_head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(args.dim, self.n_heads, bias=False)
        if self.owns_k:
            self.wk = nn.Linear(args.head_dim, self.index_head_dim, bias=False)
            self.k_norm = RMSNorm(self.index_head_dim, args.norm_eps)

    def forward(self, x, qr, latent, freqs_cis, shared: SharedAttention, offset: int):
        """`latent`: this layer's RoPE-free compressed latent (None unless the layer compresses its own KV);
        `freqs_cis`: rotary rows of the sequence. Returns [b, s, topk] indices into [window kv ; compressed kv]
        (shifted by `offset`), -1 where the query cannot see a position."""
        b, s, _ = x.size()
        ratio, rd = self.compress_ratio, self.rope_head_dim
        if self.owns_k:
            k = self.k_norm(self.wk(latent))
            # a latent stands for the first token of its group, so group j takes position j * ratio
            k = rotate_tail(k, freqs_cis[::ratio][: k.size(1)], rd)
            shared.index_k = k
        index_k = shared.index_k
        n = index_k.size(1)

        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.index_head_dim))
        q = rotate_tail(q, freqs_cis[:s].unsqueeze(1), rd)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)               # [b, s, n]

        # how many compressed positions each query can see: a block becomes visible once the query has passed
        # its last token
        compress_lens = (torch.arange(1, s + 1, device=x.device) // ratio).unsqueeze(-1)    # [s, 1]
        index_score = index_score.masked_fill(torch.arange(n, device=x.device) >= compress_lens, -torch.inf)

        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size)
        elif self.uses_candidates:                           # level two: only inside the source's candidate blocks
            index_score = index_score.masked_fill(~shared.candidates, -torch.inf)

        # top-k by score, re-sorted into position order; unreachable -> -1, rest shifted by offset
        idxs = index_score.topk(min(self.index_topk, n), dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1)


class Attention(nn.Module):
    """Latent attention over two KV sources at once, concatenated into one `sparse_attention` call: a sliding
    window of raw KV, plus -- when compress_ratio > 0 -- `index_topk` compressed positions reaching further
    back. Q and the output projection are both low-rank, the latter grouped. compress_ratio > 0 does not mean the
    layer compresses its own KV: only kv_source_layers do, the rest read that same latent."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps
        self.softmax_scale = self.head_dim**-0.5
        if self.compress_ratio:
            self.rope_args = (args.rope_head_dim, args.original_seq_len, args.compress_rope_theta)
        else:   # pure sliding-window layers: no YaRN, base theta
            self.rope_args = (args.rope_head_dim, 0, args.rope_theta)
        self.rope_extra = (args.rope_factor, args.beta_fast, args.beta_slow)

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.wq_a = nn.Linear(self.dim, args.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(args.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = GroupedLinear(self.n_groups, self.n_heads * self.head_dim // self.n_groups, args.o_lora_rank)
        self.wo_b = nn.Linear(self.n_groups * args.o_lora_rank, self.dim, bias=False)

        is_backbone = layer_id < args.n_layers
        self.is_kv_source = is_backbone and layer_id in args.kv_source_layers
        self.is_index_source = is_backbone and layer_id in args.index_source_layers
        self.compressor = Compressor(args, layer_id) if self.is_kv_source else None
        self.indexer = Indexer(args, layer_id) if self.is_index_source else None

    def freqs_cis(self, seqlen: int, device) -> torch.Tensor:
        dim, orig, theta = self.rope_args
        return _rotary_table(dim, seqlen, orig, theta, *self.rope_extra).to(device)

    def _compress_topk_idxs(self, x, qr, latent, freqs_cis, shared, offset, compress_len):
        """Which compressed positions each query attends to. Index sources run their own indexer; the layers in
        between reuse the result their source published."""
        if not self.is_index_source:
            return shared.topk_idxs
        if compress_len == 0:
            idxs = torch.empty(x.size(0), x.size(1), 0, dtype=torch.long, device=x.device)
        else:
            idxs = self.indexer(x, qr, latent, freqs_cis, shared, offset)
        shared.topk_idxs = idxs
        return idxs

    def _compress_kv(self, x, qr, freqs_cis, shared, offset):
        """The shared compressed KV and the compressed positions every query may attend to."""
        ratio = self.compress_ratio
        compress_len = x.size(1) // ratio
        latent = self.compressor(x) if self.is_kv_source else None
        # the indexer needs the latent before RoPE, so it runs first
        idxs = self._compress_topk_idxs(x, qr, latent, freqs_cis, shared, offset, compress_len)
        if latent is not None:
            # a latent stands for the first token of its group, so group j takes position j * ratio
            shared.compress_kv = rotate_tail(latent, freqs_cis[::ratio][: latent.size(1)], self.rope_head_dim)
        return shared.compress_kv, idxs

    def forward(self, x: torch.Tensor, shared: SharedAttention):
        b, s, _ = x.size()
        rd = self.rope_head_dim
        freqs_cis = self.freqs_cis(s, x.device)

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = rotate_tail(q, freqs_cis.unsqueeze(1), rd)

        kv = rotate_tail(self.kv_norm(self.wkv(x)), freqs_cis, rd)
        topk_idxs = window_idxs(s, self.window_size, x.device).expand(b, -1, -1)
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(x, qr, freqs_cis, shared, offset=s)
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        o = sparse_attention(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        o = rotate_tail(o, freqs_cis.unsqueeze(1), rd, inverse=True)
        o = self.wo_a(o.view(b, s, self.n_groups, -1))
        return self.wo_b(o.flatten(2))


# ------------------------------------------------------------------------------------------------- moe
class Gate(nn.Module):
    """MoE gating. The correction bias steers expert selection only; the routing weights come from the unbiased
    scores. Image-span tokens use a separate bias (training `noaux_tc_for_vl`). The biases are buffers: the
    aux-loss-free balancing rule updates them from the measured expert load, outside autograd."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        n_routed_experts, n_activated_experts = args.get_moe_config(layer_id)
        self.topk = n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts, args.dim))
        self.register_buffer("bias", torch.zeros(n_routed_experts, dtype=torch.float32))
        if args.vision_enabled:
            self.register_buffer("bias_vl", torch.zeros(n_routed_experts, dtype=torch.float32))
        else:
            self.bias_vl = None

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [n, dim]; image_mask: [n] bool, True for tokens inside an image span."""
        scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        indices = (scores + bias).topk(self.topk, dim=-1)[1]     # the bias picks experts but does not scale them
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)      # not norm_eps, matches training
        return weights * self.route_scale, indices


def _swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """The clamps come straight from training: the up branch is clamped on both sides, the gate branch only from
    above."""
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    return F.silu(gate) * up


class Expert(nn.Module):
    """One SwiGLU FFN (the shared expert)."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = _swiglu(self.w1(x).float(), self.w3(x).float(), self.swiglu_limit)
        return self.w2(h.to(x.dtype))


def _grouped_mm(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """x: [m, k] rows sorted by group; w: [groups, k, n]; offs: [groups] cumulative row counts."""
    fn = getattr(F, "grouped_mm", None) or torch._grouped_mm
    return fn(x, w, offs=offs)


class RoutedExperts(nn.Module):
    """All routed experts of a layer as three stacked weights, applied with grouped GEMMs: tokens are sorted by
    expert, each expert multiplies its own contiguous slice, and the results are scattered back. No host sync,
    no per-expert loop. w1 / w3: [E, inter, dim]; w2: [E, dim, inter]."""

    def __init__(self, n_experts: int, dim: int, inter_dim: int, swiglu_limit=0.0):
        super().__init__()
        self.n_experts = n_experts
        self.swiglu_limit = swiglu_limit
        self.w1 = nn.Parameter(torch.empty(n_experts, inter_dim, dim))
        self.w3 = nn.Parameter(torch.empty(n_experts, inter_dim, dim))
        self.w2 = nn.Parameter(torch.empty(n_experts, dim, inter_dim))

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """x: [n, dim]; weights, indices: [n, k]. Returns the weighted sum of the chosen experts, fp32."""
        n, k = indices.shape
        flat = indices.flatten()
        order = flat.argsort(stable=True)
        token = order // k                                                   # source row of every sorted slot
        counts = flat.new_zeros(self.n_experts).scatter_add_(0, flat, torch.ones_like(flat))
        offs = counts.cumsum(0).to(torch.int32)
        xs = x[token].to(self.w1.dtype)
        gate = _grouped_mm(xs, self.w1.transpose(-2, -1), offs).float()
        up = _grouped_mm(xs, self.w3.transpose(-2, -1), offs).float()
        h = _swiglu(gate, up, self.swiglu_limit) * weights.flatten()[order].unsqueeze(-1)
        out = _grouped_mm(h.to(xs.dtype), self.w2.transpose(-2, -1), offs).float()
        return torch.zeros(n, x.size(-1), dtype=torch.float32, device=x.device).index_add_(0, token, out)


class MoE(nn.Module):
    """Top-k routed experts plus one shared expert every token goes through."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        n_routed_experts, _ = args.get_moe_config(layer_id)
        self.dim = args.dim
        self.gate = Gate(layer_id, args)
        self.experts = RoutedExperts(n_routed_experts, args.dim, args.moe_inter_dim, args.swiglu_limit)
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.size()
        x = x.reshape(-1, self.dim)
        weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
        y = self.experts(x, weights, indices)
        y = y + self.shared_experts(x).float()
        return y.type_as(x).view(shape)


# ------------------------------------------------------------------------------------------------- hyper-connections
def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4,
                      sinkhorn_iters: int = 20, eps: float = 1e-6):
    """Split the mixing projection into pre / post / comb and make `comb` (nearly) doubly stochastic with
    Sinkhorn iterations. mixes: [b, s, (2 + hc) * hc] -> pre [b, s, hc], post [b, s, hc], comb [b, s, hc, hc]."""
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class Block(nn.Module):
    """A block whose residual stream is `hc_mult` parallel copies (Hyper-Connections).

    Attention and FFN each sit between `hc_pre` (collapse the copies into one sublayer input) and `hc_post`
    (expand back out, mixing the residual in through `comb`). `hc_mixes` derives all three coefficient sets from
    the stream itself, `comb` made doubly stochastic by Sinkhorn. The coefficients a sublayer computes are used
    by the *next* one -- see `forward`."""

    attention_cls = Attention

    def __init__(self, layer_id: int, args: ModelArgs, engram_layout: EngramLayout | None = None):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = self.attention_cls(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.engram = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            self.engram = Engram(args, layer_id, engram_layout)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_mixes(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """x: [b,s,hc,d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc]. Returns the pre / post / comb
        coefficients, split out of one projection of the flattened stream."""
        x = x.flatten(2).float()                              # normalized over the whole flattened hc*d stream
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor):
        """Collapse the hc copies into one, weighted by pre_mix. [b,s,hc,d] x [b,s,hc] -> [b,s,d]"""
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        """Expand the sublayer output back to hc copies and mix the residual in through `comb`.
        x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc] -> [b,s,hc,d]"""
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: torch.Tensor, pre_mix: torch.Tensor, image_mask: torch.Tensor | None, *attn_args):
        """`pre_mix` collapses the hc_mult copies down to one input for this block's attention. Each sub-block's
        own `hc_mixes` produces the mix for the *next* one, so attention uses what the previous layer's FFN
        produced and the FFN uses what this attention produced.

        image_mask: [b, s] bool, True inside image spans (selects the VL routing bias)."""
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.hc_pre(x, pre_mix)
        x = self.attn_norm(x)
        x = self.attn(x, *attn_args)
        x = self.hc_post(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.hc_pre(x, attn_pre)
        x = self.ffn_norm(x)
        x = self.ffn(x, image_mask)
        x = self.hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre


class Head(nn.Module):
    """Output projection, kept in fp32 so the logits come out in fp32 directly."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight)


# ------------------------------------------------------------------------------------------------- dspark
class DSparkAttention(Attention):
    """Attention of a draft block. Each draft block (`block_size` tokens, one per start position t) attends to
    the window of main-model states that end at t plus the block itself, bidirectionally. The main states are
    projected to keys/values by this layer's own `wkv`. All start positions of a sequence run at once: the
    blocks are folded into the batch dim."""

    def forward(self, x: torch.Tensor, main_x: torch.Tensor):
        """x: [b * T, B, d] draft states; main_x: [b, s, d] projected main-model states, T = s - B blocks."""
        assert self.compress_ratio == 0
        b, s, _ = main_x.size()
        n, blk, _ = x.size()
        t, win, rd = n // b, self.window_size, self.rope_head_dim
        freqs_cis = self.freqs_cis(s + blk, x.device)

        main_kv = rotate_tail(self.kv_norm(self.wkv(main_x)), freqs_cis[:s], rd)             # [b, s, dh]
        wid = window_idxs(t, win, x.device)                                                 # [T, w]
        win_kv = main_kv[:, wid.clamp(min=0)].reshape(n, wid.size(-1), -1)                  # [b*T, w, dh]

        # block position j of the block starting at t sits at absolute position t + 1 + j
        pos = torch.arange(t, device=x.device).unsqueeze(1) + 1 + torch.arange(blk, device=x.device)
        f = freqs_cis[pos].unsqueeze(0).expand(b, -1, -1, -1).reshape(n, blk, -1)           # [b*T, B, rd/2]
        qr = self.q_norm(self.wq_a(x))
        q = rotate_tail(self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim)), f.unsqueeze(2), rd)
        blk_kv = rotate_tail(self.kv_norm(self.wkv(x)), f, rd)                              # [b*T, B, dh]

        kv = torch.cat([win_kv, blk_kv], dim=1)
        w = wid.size(-1)
        idxs = torch.cat([torch.where(wid >= 0, torch.arange(w, device=x.device), -1),
                          (w + torch.arange(blk, device=x.device)).expand(t, -1)], dim=-1)  # [T, w + B]
        idxs = idxs.repeat(b, 1).unsqueeze(1).expand(n, blk, -1)
        o = sparse_attention(q, kv, self.attn_sink, idxs, self.softmax_scale)
        o = rotate_tail(o, f.unsqueeze(2), rd, inverse=True)
        o = self.wo_a(o.view(n, blk, self.n_groups, -1))
        return self.wo_b(o.flatten(2))


class DSparkMarkovHead(nn.Module):
    """A low-rank bigram prior over the next token, added to the draft logits."""

    def __init__(self, vocab_size: int, dspark_markov_rank: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dspark_markov_rank)
        self.head = Head(vocab_size, dspark_markov_rank)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embed(token_ids)
        return self.head(embed), embed


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor):
        return self.proj(torch.cat([hidden, markov_embed], dim=-1).float()).squeeze(-1)


class DSparkBlock(Block):
    """DSpark stage stored under the mtp.* checkpoint namespace. The first stage owns the projection of the
    main model's hidden states; the last owns the norm and the heads. Embedding and output head are the main
    model's, passed in by the caller instead of tied through attributes."""

    attention_cls = DSparkAttention

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        stage_id = layer_id - args.n_layers
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        if stage_id == 0:
            assert len(args.dspark_target_layer_ids) > 0, "DSpark needs target layers"
            self.main_proj = nn.Linear(args.dim * len(args.dspark_target_layer_ids), args.dim, bias=False)
            self.main_norm = RMSNorm(args.dim, args.norm_eps)
        if stage_id == args.n_mtp_layers - 1:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(args.vocab_size, args.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(args.dim + args.dspark_markov_rank)

    def forward(self, x: torch.Tensor, pre_mix: torch.Tensor, main_x: torch.Tensor):
        return super().forward(x, pre_mix, None, main_x)         # drafts are text: no VL bias

    def forward_embed(self, main_hidden: torch.Tensor, input_ids: torch.Tensor, embed: nn.Module):
        """main_hidden: [b, s, n_targets * d]; input_ids: [b, s]. Returns the draft block inputs [b * T, B, hc, d]
        (first token = the real token at t, the rest noise), the projected main states, and the true tokens
        [b * T, B + 1] of every block (teacher forcing)."""
        b, s = input_ids.shape
        blk = self.block_size
        main_x = self.main_norm(self.main_proj(main_hidden))
        true = input_ids.unfold(1, blk + 1, 1)[:, : s - blk].reshape(-1, blk + 1)           # [b*T, B+1]
        draft = torch.cat([true[:, :1], true.new_full((true.size(0), blk - 1), self.noise_token_id)], dim=1)
        x = embed(draft).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x, true

    def forward_head(self, x: torch.Tensor, pre_mix: torch.Tensor, prev_ids: torch.Tensor, head: nn.Module):
        """Draft logits of every block position. The Markov prior of step i reads the token at step i: in
        training that is the true token, at inference the one just sampled."""
        x = self.hc_pre(x, pre_mix)
        logits = head(self.norm(x))
        logits_bias, markov_embed = self.markov_head(prev_ids)
        confidence = self.confidence_head(x, markov_embed)
        return logits + logits_bias, confidence


# ------------------------------------------------------------------------------------------------- the model
def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """initial one-hot mix"""
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


class Transformer(nn.Module):
    """DeepSeek-V4.1: embed -> expand to hc_mult copies -> blocks -> collapse -> logits."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hc_mult = args.hc_mult
        self.engram_layout = EngramLayout.from_args(args)
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(i, args, self.engram_layout) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = Head(args.vocab_size, args.dim)
        self.target_layer_ids = args.dspark_target_layer_ids
        self.mtp = nn.ModuleList()
        if args.dspark_block_size:
            self.mtp = nn.ModuleList([DSparkBlock(args.n_layers + i, args) for i in range(args.n_mtp_layers)])
        self.vision = None
        if args.vision_enabled:
            self.vision = ViT(args)
            self.aligner = Aligner(args)
            # learned embeddings for the image span delimiters
            self.image_start = nn.Parameter(torch.empty(args.dim))
            self.image_end = nn.Parameter(torch.empty(args.dim))
            self.image_newline = nn.Parameter(torch.empty(args.dim))

    # -- vision
    def encode_images(self, patches: torch.Tensor, grids) -> torch.Tensor:
        """patches: [total_patches, 3 * p * p], the images of the batch back to back; grids: [(n_h, n_w)] per
        image. Returns [total_image_tokens, dim]."""
        out = []
        for chunk, (n_h, n_w) in zip(patches.split([h * w for h, w in grids]), grids):
            out.append(self.aligner(self.vision(chunk, n_h, n_w), n_h, n_w))
        return torch.cat(out, dim=0)

    def merge_image_embeddings(self, h: torch.Tensor, token_types: torch.Tensor, image_embeds: torch.Tensor):
        """Overwrite each image's token span with its ViT/aligner features: the IMAGE slots take the aligner rows
        in row-major order, the span delimiters take learned embeddings."""
        for kind, vec in ((IMAGE_START, self.image_start), (IMAGE_END, self.image_end),
                          (IMAGE_NEW_LINE, self.image_newline)):
            h = torch.where((token_types == kind).unsqueeze(-1), vec.to(h.dtype), h)
        return h.masked_scatter((token_types == IMAGE).unsqueeze(-1), image_embeds.to(h.dtype))

    # -- the three steps of forward, split out so a caller can wrap them (e.g. to draw the model in phases)
    def embed_inputs(self, input_ids, token_types=None, image_patches=None, image_grids=None):
        """Embedding (+ vision merge), expansion to hc_mult copies. Returns h, pre_mix, image_mask, engram_mask."""
        image_mask = None if token_types is None else token_types >= 0                      # TEXT is -1
        # image tokens take no part in an n-gram and get no engram contribution
        engram_mask = None if image_mask is None else ~image_mask
        h = self.embed(input_ids)
        if image_patches is not None:
            h = self.merge_image_embeddings(h, token_types, self.encode_images(image_patches, image_grids))
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return h, make_identity_pre_mix(h, self.hc_mult), image_mask, engram_mask

    def run_layer(self, i, h, pre_mix, image_mask, engram_mask, engram_hashes, shared, main_hiddens):
        layer = self.layers[i]
        if layer.engram is not None:
            h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], engram_mask)
        # the DSpark head reads the attention input of its target layers, not their output
        if i in self.target_layer_ids:
            main_hiddens.append(h.mean(dim=2))
        return layer(h, pre_mix, image_mask, shared)

    def logits(self, h, pre_mix):
        return self.head(self.norm(self.layers[-1].hc_pre(h, pre_mix)))

    def forward(self, input_ids, token_types=None, image_patches=None, image_grids=None, engram_hashes=None):
        """input_ids: [b, s]. token_types / image_patches / image_grids carry the VL inputs (token_types: -1 text,
        0 image start, 1 image, 2 image newline, 3 image end); engram_hashes: [b, s, n_engram_layers,
        n_hash_cols] from `engram.NgramHasher`. Returns logits [b, s, vocab] and the hidden states the DSpark
        head reads ([b, s, n_targets * dim], None without targets)."""
        h, pre_mix, image_mask, engram_mask = self.embed_inputs(input_ids, token_types, image_patches, image_grids)
        shared, main_hiddens = SharedAttention(), []
        for i in range(len(self.layers)):
            h, pre_mix = self.run_layer(i, h, pre_mix, image_mask, engram_mask, engram_hashes, shared, main_hiddens)
        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        return self.logits(h, pre_mix), main_hidden

    def forward_spec(self, input_ids: torch.Tensor, main_hidden: torch.Tensor):
        """DSpark training pass. For every start position t < s - block_size a draft block predicts the next
        `block_size` tokens. Returns draft logits [b * T, B, vocab], confidence [b * T, B] and the target tokens
        [b * T, B] (teacher forcing: the Markov prior reads the true tokens)."""
        x, main_x, true = self.mtp[0].forward_embed(main_hidden, input_ids, self.embed)
        pre_mix = make_identity_pre_mix(x, self.hc_mult)
        for layer in self.mtp:
            x, pre_mix = layer(x, pre_mix, main_x)
        blk = self.mtp[-1].block_size
        logits, confidence = self.mtp[-1].forward_head(x, pre_mix, true[:, :blk], self.head)
        return logits, confidence, true[:, 1:]
