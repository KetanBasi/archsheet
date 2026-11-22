"""archsheet spec for DeepSeek-V4.1-Flash  (paper figure 3, "Overall architecture").

    python -m archsheet examples/deepseek_v41_flash_spec.py -o out/deepseek_overview --depth 1 --png      # the sheet
    python -m archsheet examples/deepseek_v41_flash_spec.py -o out/deepseek_blocks   --depth 2             # + every Block opened (very large)

The model is `examples/deepseek_v41_flash/model.py`: the released inference code adapted for training (see its
docstring), at the released shapes and traced on the `meta` device, so nothing is allocated. This spec only
chooses what to draw:

  * `PhasedTransformer` runs the same three steps as `Transformer.forward` but opens a `phase()` around the
    causal encoder (layers 0-19) and the decoder (layers 20-39, the head and DSpark);
  * FLAGS asks archsheet to draw the hyper-connection helpers and the sparse-attention function as boxes -- the
    model itself is not touched: they are a plain method and a plain function.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepseek_v41_flash.model import ModelArgs, Transformer  # noqa: E402

from archsheet import label, phase  # noqa: E402

FLAGS = dict(
    leaf_types="ViT",
    trace_methods="Block.hc_mixes,Block.hc_pre,Block.hc_post,deepseek_v41_flash.model:sparse_attention",
)

ARGS = ModelArgs.flash()
N_ENC = ARGS.n_layers // 2
B, S = 7, 1040                                  # example batch / sequence length
GRID = (12, 15)                                 # ViT patch grid of one image
NP = GRID[0] * GRID[1]                          # patches per image
M = NP // ARGS.vision_downsample_ratio**2       # image tokens per image after the 3x3 pixel-unshuffle


class PhasedTransformer(Transformer):
    """Transformer.forward, with the encoder and the decoder drawn as phases."""

    def forward(self, input_ids, token_types, image_patches, image_grids, engram_hashes):
        from deepseek_v41_flash.model import SharedAttention
        shared, main_hiddens = SharedAttention(), []
        with phase("Causal Encoder"):
            h, pre_mix, image_mask, engram_mask = self.embed_inputs(input_ids, token_types, image_patches, image_grids)
            for i in range(N_ENC):
                h, pre_mix = self.run_layer(i, h, pre_mix, image_mask, engram_mask, engram_hashes, shared, main_hiddens)
            label(h, "Enc. hidden states")
        with phase("Decoder"):
            for i in range(N_ENC, len(self.layers)):
                h, pre_mix = self.run_layer(i, h, pre_mix, image_mask, engram_mask, engram_hashes, shared, main_hiddens)
            logits = self.logits(h, pre_mix)
            draft = self.forward_spec(input_ids, torch.cat(main_hiddens, dim=-1))
        return logits, draft


def build():
    with torch.device("meta"):
        model = PhasedTransformer(ARGS)
    d = ARGS
    return dict(
        model=model,
        inputs=dict(
            input_ids=torch.zeros(B, S, dtype=torch.long),
            token_types=torch.zeros(B, S, dtype=torch.long),
            image_patches=torch.zeros(B * NP, 3 * d.vision_patch_size**2),
            image_grids=[GRID] * B,
            engram_hashes=torch.zeros(B, S, len(d.engram_layer_ids), 3 * d.engram_n_heads, dtype=torch.long),
        ),
        symbols={
            "b": (B, "batch"), "s": (S, "tokens"), "n": (B * NP, "image patches, all images"),
            "np": (NP, "patches of one image"), "mi": (M, "tokens of one image"),
            "m": (B * M, "image tokens, all images"), "pp": (3 * d.vision_patch_size**2, "pixels per patch (3·14·14)"),
            "nh": (3 * d.engram_n_heads, "Engram hash columns (3 n-gram orders x 8 heads)"),
            "gh": (GRID[0], "ViT grid rows"), "gw": (GRID[1], "ViT grid columns"),
            "dm": (d.dspark_markov_rank, "DSpark Markov rank = Engram head dim"),
            "td": (3 * d.dim, "3 · d: hidden states of the 3 DSpark target layers"),
            "eg": (3 * d.engram_n_heads * d.engram_head_dim, "Engram row width (24 x 256)"),
            "ek": (d.dim * (d.hc_mult + 1), "d · (hc + 1): Engram keys + value"),
            "ad": (9 * d.vision_dim, "aligner input: 3x3 patches x ViT width"), "z": (B * (S - d.dspark_block_size), "DSpark draft blocks b·(s−bs)"),
            "bs": (d.dspark_block_size, "DSpark block size"),
            "d": (d.dim, "hidden size"), "h": (d.n_heads, "query heads"), "hc": (d.hc_mult, "mHC residual streams"),
            "rq": (d.q_lora_rank, "query low-rank dim"), "r": (d.o_lora_rank, "output low-rank dim (per group) = ViT width"),
            "g": (d.o_groups, "output groups"), "dh": (d.head_dim, "head dim (= index top-k)"),
            "E": (d.n_routed_experts, "routed experts"), "k": (d.n_activated_experts, "experts per token"),
            "fm": (d.moe_inter_dim, "expert width"),
            "ih": (d.index_n_heads, "indexer heads"), "di": (d.index_head_dim, "indexer head dim"),
            "v": (d.vocab_size, "vocab"),
        },
        title="DeepSeek-V4.1-Flash",
    )
