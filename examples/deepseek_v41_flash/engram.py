"""Engram n-gram hashing of DeepSeek-V4.1-Flash, as a stateless function of the token ids.

Adapted from deepseek-ai/DeepSeek-V4.1-Flash `inference/engram.py` (MIT License, see ./LICENSE).
Changes for training: no tokenizer inside the model (pass the `token_map` you built once with the released
`build_compressed_token_map`), no running `cache` / `start_pos` (the whole sequence is hashed at once, so every
position sees its own history), and `NgramHasher` is meant to run in the data pipeline: its output is an input
of `Transformer.forward`, which keeps the int64 hash arithmetic out of the autograd graph.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):       # deterministic for n < 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def compute_hash_multipliers(layer_ids, max_ngram_size: int, tokenizer_vocab_size: int) -> torch.Tensor:
    """One multiplier per (layer, lookback), from a per-layer RNG so layers hash differently. Kept odd, and
    bounded so that `token_id * multiplier` cannot overflow int64."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        values = np.random.default_rng(10007 * layer_id).integers(0, bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables. A position is hashed as `max_ngram_size - 1` n-grams
    (2-gram .. max_ngram_size-gram), each split over `n_heads` heads; every (n-gram size, head) pair owns its own
    prime-sized bucket range in the layer's table."""

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]       # [layer][n-gram size][head] bucket modulus
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args) -> "EngramLayout | None":
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(args.engram_max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(args.engram_n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(args.engram_max_ngram_size, layer_ids, tuple(args.engram_num_embeddings), tuple(primes),
                   args.engram_n_heads, args.engram_head_dim)

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads


class NgramHasher(nn.Module):
    """Hash ids of the n-grams ending at every position: [B, L, n_engram_layers, n_hash_cols].

    Ids go through the compressed table `token_map`; a position is hashed with the `max_ngram_size - 1` tokens
    before it. Look-back stops at the start of the sequence and at any token the mask excludes (an image span),
    so an n-gram never spans one."""

    DEAD = -1

    def __init__(self, layout: EngramLayout, token_map: torch.Tensor, compressed_vocab_size: int, pad_token_id: int):
        super().__init__()
        self.layout = layout
        self.pad_id = int(token_map[pad_token_id])
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
        self.register_buffer("multipliers",
                             compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, compressed_vocab_size),
                             persistent=False)
        self.register_buffer("token_map", torch.as_tensor(token_map), persistent=False)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """token_mask: [B, L], False for tokens that take no part in an n-gram (image spans)."""
        batch, seqlen = input_ids.shape
        ids = self.token_map[input_ids]
        if token_mask is not None:
            ids = torch.where(token_mask, ids, self.DEAD)
        positions = torch.arange(seqlen, device=input_ids.device).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = ids.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)                                    # [B, L, max_ngram_size]
        # XOR the multiplied ids together one lookback at a time: the running value after step i is the hash of
        # the (i+1)-gram, and each lands in its own prime-sized bucket range
        products = tokens.unsqueeze(2) * self.multipliers                       # [B, L, n_engram_layers, ngram]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets
