---
name: archsheet
description: Generate an informative, blueprint-style architecture sheet (SVG / HTML / PNG diagram) of a PyTorch model (LLM, transformer, CNN, anything nn.Module) with symbolic input/output tensor shapes on every node (b, s, d, h·dh ...), nested submodules, residual connections, repeated-layer badges (×N) and detected layer patterns (e.g. 5 local : 1 global). Use when the user asks to visualize, diagram, draw, document or "blueprint" a model architecture, or asks for an architecture sheet, or wants shapes shown per layer.
---

# archsheet

Renders a **dataflow sheet** of a PyTorch model to SVG / HTML (pan-zoom, hover for exact shapes) / PNG.
The model is traced on the `meta` device (no weights, no memory), so it works for huge models. It shows:

- every submodule, nested, with class, config (`Linear 3840 → 15360`) and param count
- **input → output shapes** on every node as symbols (`[b, s, h·dh]`); hidden reshapes under `out`:
  `↳ view,transpose [b, h, s, dh]`
- real dataflow: residual adds (⊕), gating (⊗), fan-out (q/k/v), skips, concatenations `(s; t)`
- repeated layers collapsed to `×N`, periodic patterns detected even after an irregular start or before
  an irregular end (`12 layers = 2 × pattern of 6`, `Full + 3 × Reuse`, then `4 × (Reindex + 3 × Reuse)`);
  attributes that differ between variants are named on the owning node (`self_attn · sliding window 8` vs
  `· global attention`)
- branching-in graphs (vision tower → projector → text embeddings), multiple inputs (named after
  `forward()`'s parameters) and multi-entry-point models as dashed phase regions

It runs with the **project's own Python** (the spec imports the user's model); nothing is installed for the
skill itself, `PYTHONPATH=${CLAUDE_SKILL_DIR}` is enough. `torch` is assumed present; `cairosvg` is only
needed for `--png` (without it SVG and HTML are still written; offer `pip install cairosvg`). If the project
uses `uv` / `poetry` / a venv, run the commands through it.

## Workflow

1. **Find the model** (the `nn.Module` and its config in the user's repo; ask if unclear). Never modify it.
2. **Write a spec** in the user's repo (e.g. `archsheet_spec.py`) from `templates/spec_template.py`:
   - build the model and tiny example inputs in `build()` (runs under `torch.device("meta")`; HF:
     `AutoModelForCausalLM.from_config(cfg)` works, no download)
   - `symbols`: name every dimension worth naming (batch, seq, d_model, heads, head dim, ffn, vocab ...).
     **Values must be unique and dynamic dims must not collide with config dims** (`b=3, s=17`); a spurious
     match (`10 = 2·b`) is the main failure mode.
3. **Run**:
   ```bash
   PYTHONPATH=${CLAUDE_SKILL_DIR} python -m archsheet archsheet_spec.py -o docs/archsheet/model --png
   ```
   A spec may set flag defaults for itself (`FLAGS = dict(leaf_types="SparseAttn", depth=2)`);
   command-line flags win. Read the printed text **outline** first: shapes that stay numeric
   (`[b, s, 160]`) mean a missing symbol (products of symbols are found automatically).
4. **Look** at the PNG (crop/resize if tall), tell the user the output paths (`.html` is nicest to browse),
   and iterate with the flags until readable. Usually two views: overview (`--depth 2`) and detail
   (`--depth 4 --ops key`).

## Reading the diagram (and how the layout behaves)

- Data flows **bottom → top**; graph inputs sit on the bottom row, outputs on top. Nodes fed only by
  graph inputs are pulled up beside their first consumer (a VLM's text embedding sits at the merge point),
  unless a sibling reads the same tensor in the lower row (q/k/v share one row).
- Boxes: leaf modules (coloured by kind), tinted containers, small dashed cards for functional ops
  (SDPA, cat, masked_scatter), circles for `+` / `×`. Dashed border with `×N` = repeated pattern; large
  dashed region = phase (entry point). `in` / `out` rows list shapes; extra inputs line up under the first.
- **Lines are continuous across container borders** and a flow's name (from `label()`) is printed once.
  A filled **dot** = one tensor really splits; two different tensors that only cross have no dot and the
  crossed line has a small gap.
- **Centre rule.** The central neighbour is the main flow: among Q K V the stream enters K and splits left
  (Q) / right (V); SDPA's arrows land Q left, K centre, V right. Two consumers (k_proj / v_proj, gate / up)
  get a symmetric T. A chain that is central at both ends (a *spine*: input → embed → merge → layers → head)
  is one exact vertical line; off-centre links jog only in the last gap.
- A tensor with several consumers is one straight trunk that splits only in the last gap before each
  consumer; if it continues past a real consumer of the same tensor, that consumer stays on the line and the
  rest runs to its right.
- **Residual skips** (`+` fed from the stream several rows below) leave the stream just above the source, run
  in a lane 16 px right of the rows they pass, and enter `+` from the right. Lanes are routed after
  placement, so a skip never moves anything (`--hide-skips` draws the same layout minus the lines). Every
  other input, including values that merely skip a row (RoPE `x·cos + rot(x)·sin`), enters from below.
- **Containers** are as wide as their content; entry and exit share one vertical line so nested and sibling
  blocks (`layers[0–4]`, `layers[5]`) line up. Corners are ≥ 22 px from the arrowhead they feed.
- **Phases** (`shortie` / `tallie`, encoder / decoder) sit side by side, bottom-aligned, never padded to a
  common height: the first tower's output on top runs down a channel between the towers to wherever it is
  used in the second, so nothing is left empty under a shorter tower. Flows leave the earlier phase's right
  border and enter the later one's left border; several of them are ordered to avoid crossing each other.
  A tensor that only travels to another phase also leaves / enters the blocks it sits in through their side
  border at the height of its producer / consumer, instead of looping over the block's top / bottom.
- Rows, track levels and shared trunks are ordered to minimise crossings (barycentre + sifting passes); a
  trunk stays on one vertical line where it can, and two ports a hair apart on one flow become one straight
  line, not a double bend.
- Flows are identified by their *producer*: K and V from one projector are two lines; give each its own
  `label(...)`.

## Multimodal / branching / multi-entry models

Plain `model(**inputs)` already handles branch-in (every tensor is followed from producer to consumer, so
`inputs_embeds.masked_scatter(mask, image_features)` becomes a merge node with three inputs). Give each
input its own symbol (`n` patches, `hw` image size ...).

For several entry points or persistent state (stored K/V, KV cache, encoder/decoder split) give the spec a
`run(model)` and wrap each call in `phase(...)`, or open `phase(...)` inside a `forward()` to split one pass
into regions (encoder | decoder). Name tensors with `label(...)`. A phase counts as one `--depth` level;
when the root module is dissolved (as it is with phases) the phases are level 0, so `--depth 1` is the
overview with the blocks closed. A graph input read by several phases is drawn once inside each of them
(a copy per phase, so no long line has to cross the sheet). Tensors kept on the model between calls keep their origin, so the arrow between
phases is drawn automatically:

```python
from archsheet import phase, label

def run(m):
    with phase("shortie"):
        m.shortie(src)                    # writes m.projector.k / .v
    with phase("tallie"):
        m(query)                          # in forward(): label(k, "K · appended"); label(v, "V · appended")
```
(`model.shortie` is not a module call, so the encoder appears directly in the region. The root wrapper class
is dissolved automatically when phases are used; `--unwrap Cls` dissolves others.) A phase may also wrap
a few calls inside `forward()` (the VLM spec draws `Vision` and `Language` side by side that way). See
`examples/two_phase_spec.py`, `examples/vlm_spec.py` and `examples/deepseek_v41_flash_spec.py`.

## Work that is not a module

Never wrap code in a dummy `nn.Module` to get it drawn: the sheet adapts to the model.

- Tensor ops outside any module's `forward` (in the spec's `run()`, in a method called from there) are
  recorded like any other op; their results are not "unknown inputs".
- A plain **method or function** can be drawn as a box with `--trace-methods Class.method,pkg.mod:function`
  (in `FLAGS` as `trace_methods=`). `Class` matches the class or any base class, so `Block.hc_pre` also
  covers subclasses. Without module calls inside, the call collapses into one box with its input and output
  shapes (a sparse-attention kernel, a Sinkhorn normalisation); with module calls inside, it becomes a
  container around them. Each call site is its own node.
- A module **called several times** (weight sharing, a ViT run once per image, one `rotary_emb` per layer)
  is one box: `×N` badge and `N calls` under the title.

## Flags

| flag | meaning |
|---|---|
| `-o PATH` | output path without extension |
| `--depth N` | module levels to expand (default 4); `2` = block overview |
| `--ops none\|merge\|key\|all` | functional ops drawn as nodes: `key` (default) adds/muls/matmul/sdpa/softmax/cat…; `merge` only ops with ≥2 tensor inputs; `none` modules only; `all` everything non-reshape |
| `--theme light\|blueprint` | white, or dark-blue engineering style with grid |
| `--png` | also rasterise (cairosvg or rsvg-convert); `--png-width PX` pins the width (default 2× SVG, ≤ 16000 px per side) |
| `--no-collapse` | draw every repeated layer instead of `×N` |
| `--hide-types A,B` | classes treated as pass-through (default `Identity,Dropout`) |
| `--unwrap A,B` | dissolve wrapper modules (children drawn in their place) |
| `--leaf-types A,B` | draw these classes as one box even if they have children |
| `--trace-methods A.m,pkg.mod:f` | draw these methods / functions as boxes (`Class.method` matches subclasses too) |
| `--no-card-io` / `--no-edge-labels` | drop in/out rows in cards / flow names on edges |
| `--hide-skips` | don't draw residual skip lines (their lane space is kept) |
| `--edge-shapes` | legacy style: shapes and `↳` notes on edges |
| `--dtypes auto\|all\|none` | when to print dtypes (auto: non-float or differing from params) |
| `--device cpu` | if tracing fails on `meta` (data-dependent control flow); use **tiny** dims |
| `--title TEXT` | diagram title |

## Self-check and troubleshooting

The run prints e.g. `layout check: 120 edge pieces, 12 branch points, 5 crossings, 0 short jogs`. Many
crossings (> ~12) or short jogs (> ~3) mean the view is too dense: lower `--depth`, use `--ops merge` /
`--no-card-io`, or collapse a module with `--leaf-types`. Never edit the generated SVG/HTML by hand;
change the spec or flags and rerun.

- **`forward pass failed`**: the model does something meta tensors can't (`.item()`, `if x.any()`,
  `nonzero`, value-routed MoE; `torch.autocast` on meta is already a no-op while tracing). Retry `--device cpu` with tiny dims, or add `run=` in the
  spec.
- **`symbols X and Y have the same value`**: change the example sizes.
- **Module missing**: never called in `forward`, or listed in `--hide-types`.
- **Layers not collapsed**: they differ in a config attribute or shape (the outline shows what) or are not
  in a `ModuleList`/`Sequential`.
- **A function or method is only a few small ops**: name it with `--trace-methods` (see above).
- Only PyTorch is supported; for JAX/TF port the structure to a stub `nn.Module` with the same shapes.

## Files

`archsheet/` (`cli` · `trace` · `graph` · `layout` · `render`), `templates/spec_template.py`,
`examples/{gemma_like,vlm,two_phase,deepseek_v41_flash,deepseek_csa2_modes}_spec.py` (+ `examples/output/`
pre-rendered results; the two DeepSeek specs follow figures 3 and 4 of the V4.1-Flash paper).
