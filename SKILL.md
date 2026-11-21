---
name: archsheet
description: Generate an informative, blueprint-style architecture sheet (SVG / HTML / PNG diagram) of a PyTorch model (LLM, transformer, CNN, anything nn.Module) with symbolic input/output tensor shapes on every node (b, s, d, h·dh ...), nested submodules, residual connections, repeated-layer badges (×N) and detected layer patterns (e.g. 5 local : 1 global). Use when the user asks to visualize, diagram, draw, document or "blueprint" a model architecture, or asks for an architecture sheet, or wants shapes shown per layer.
---

# archsheet

Renders a **dataflow sheet** of a PyTorch model to SVG / HTML (pan-zoom, hover for exact numeric
shapes) / PNG. The model is traced on the `meta` device (no weights, no memory, no real compute), so
it works for huge models. Unlike a block diagram it shows:

- every submodule, nested, with class, config (`Linear 3840 → 15360`) and param count
- **input → output tensor shapes** on every node, as symbols (`[b, s, h·dh]`); edges stay clean lines
- real dataflow: residual adds (⊕), gating (⊗), fan-out (q/k/v), skip connections; hidden reshapes are
  written under the producing node's `out` row: `↳ view,transpose [b, h, s, dh]`
- repeated layers collapsed into one drawing with `×N`; periodic patterns detected
  (`12 layers = 2 × pattern of 6`). Attributes that differ between the variants are written on the
  node that owns them, in plain words: `self_attn · sliding window 8` vs `self_attn · global attention`
  (window attributes equal to `0`, `None` or `-1` are read as global attention)
- branching-in graphs (vision tower → projector → merged into text embeddings), multi-input models,
  and multi-entry-point models drawn as dashed phase regions (`shortie` / `tallie`)
- lines are **continuous across container borders** (no intermediate nodes): a tensor leaves a
  container exactly where the inner line ends and enters the next one where its inner line starts
- multiple inputs are named after `forward()`'s parameters and aligned under `in`:
  `in x=[b, s, d]` / `ext_k=[b, h, t, dh]` / `ext_v=[...]`; concatenations show the joined dim as
  `(s; t)`: `cat · dim 2  [b, h, (s; t), dh]`

The tool is the Python package `archsheet/` in this skill directory (`cli`, `trace`, `graph`, `layout`,
`render`). It has **no environment of its own**: it must run with the *project's* Python (the one that
can import the user's model), because the spec imports the model code. Nothing is installed for the
skill itself — `PYTHONPATH=${CLAUDE_SKILL_DIR}` is enough.

**Requirements (check before the first run, in the project's environment):**

- `torch >= 2.1` — required. Any PyTorch project already has it. Check with
  `python -c "import torch"`; if the project uses `uv` / `poetry` / a venv, run every command below
  through it (`uv run python …`, `poetry run python …`, `.venv/bin/python …`).
- `cairosvg` (or `rsvg-convert`) — only for `--png`. If it is missing, do not stop: SVG and HTML are
  still written. Offer `pip install cairosvg` (in that same environment) and ask before installing.

## Workflow

0. **Preflight.** In the project's environment run `python -c "import torch"` (see Requirements above). If it
   fails, stop and tell the user which interpreter lacks torch; do not install torch yourself.
1. **Find the model.** Locate the `nn.Module` class and its config in the user's repo (or ask which
   one). Do not modify their model code.
2. **Write a spec** in the user's repo (e.g. `archsheet_spec.py`) by copying
   `templates/spec_template.py` from this skill directory, then fill in:
   - build the model + tiny example inputs inside `build()` (it runs under `torch.device("meta")`)
   - `symbols`: name every dimension worth naming (batch, seq, d_model, heads, kv heads, head dim,
     ffn dim, vocab, channels, image size, num classes ...).
     **Values must be unique and dynamic dims must not collide with config dims** — use e.g.
     `b=3, s=17`. A spurious match (e.g. `10 = 2·b`) is the main failure mode.
   - HF models: `AutoModelForCausalLM.from_config(cfg)` inside `build()` works (no weights downloaded).
3. **Run** (from anywhere; with the project's Python):
   ```bash
   PYTHONPATH=${CLAUDE_SKILL_DIR} python -m archsheet archsheet_spec.py -o docs/archsheet/model --png
   ```
   It prints a text **outline** with all shapes. Read it to verify the architecture and the symbols
   *before* looking at the picture: shapes that stay numeric (e.g. `[b, s, 160]`) mean a symbol is
   missing — add it to the spec (`"hd": (H*DH, "h·dh")` is rarely needed; products of symbols are
   found automatically).
4. **Look at the result.** If `--png` worked, open the PNG (it can be very tall; crop/resize to
   inspect). Tell the user the output paths; the `.html` is the nicest to browse.
5. **Iterate** with the flags below until it is readable at the right level of detail. Typically
   produce two views: an overview (`--depth 2`) and a detailed one (`--depth 4 --ops key`).

## How the layout behaves (so you can predict / explain it)

- Data flows bottom → top. Graph inputs are pinned to the bottom row, outputs to the top.
- Nodes whose only inputs are graph inputs are pulled up next to their first consumer, so e.g. the
  text embedding of a VLM sits beside the merge point instead of at the bottom — unless a sibling
  reads the same tensor in the lower row (q / k / v always share one row).
- **Centre rule.** The *central* neighbour is the main flow, not the first one: among Q K V, K is
  central, so the stream enters K, splits left (Q) and right (V) in a T, and SDPA's three arrows land
  symmetrically (Q left, K centre, V right). With two consumers the source sits exactly between them
  (symmetric T, e.g. k_proj / v_proj; gate / up). A trunk that merely passes by never pulls a node.
- **Spines.** Every chain of links that is central at both ends (input → embed → merge → layers →
  head, norm1 → k → cat → SDPA → o …) is forced onto one exact vertical line. Links that are off-centre
  at either end are weak and jog in the last gap only.
- A tensor with several consumers is one trunk that stays a single straight line and splits only in
  the last gap before each consumer. If the trunk keeps going past a real consumer of the *same* tensor
  (input_ids → embed_tokens **and** → masked_scatter) the consumer stays on the straight line and the
  continuing branch runs to its right.
- **Residual skips** (`+` fed from the stream several rows below) leave the stream just above their
  source, run in a *lane* hugging the right-most content of the rows they pass (16 px), and enter `+`
  from the right. Lanes are routed after everything else is placed, so a skip never moves, widens or
  re-centres anything (`--hide-skips` draws the same layout minus the skip lines). Only a residual skip
  into `+` runs beside the content; every other input, including values that merely skip a row
  (RoPE: `x·cos + rot(x)·sin`), and every input of `×`, `−`, `÷` enters from below.
- **Containers.** A container is as wide as its content (no empty margin) and its single entry and exit
  share one vertical line, so the stream runs straight through nested blocks and sibling blocks
  (`layers[0–4]`, `layers[5]`) line up.
- **Clearance.** Every corner/branch is ≥ 22 px away from the arrowhead it feeds (head ≈ 10 px + a
  visible shaft) and ≥ 10 px from the node it leaves.
- Phases (`shortie` / `tallie`) are placed **side by side** with their inputs at the bottom;
  flows between them leave through the right border of the earlier phase and enter through the
  left border of the later one (Transformer-figure style). Both ends are raised to the same height,
  so a flow crossing the gap is one straight horizontal line, never a little up/down jog.
- Flows are identified by their *producer*: two tensors from the same container (K and V from a
  projector) stay two separate lines. Give them separate `label(...)`s.

## Multimodal / branching / multi-entry models

Plain `model(**inputs)` already handles branch-in: every tensor is followed from producer to
consumer, so e.g. `inputs_embeds.masked_scatter(mask, image_features)` is drawn as a merge node with
three incoming edges (text embeddings, projector output, the `<image>` mask path). Give each input
its own symbol (`n` image patches, `hw` image size ...).

For models with several entry points or persistent state (stored K/V, KV cache, encoder/decoder
split), give the spec a `run(model)` and wrap each call in `phase(...)`; name interesting
tensors with `label(...)`. Tensors that live on the model between calls keep their origin, so the
arrow from `shortie` to `tallie` is drawn automatically:

```python
from archsheet import phase, label

def run(m):
    with phase("shortie"):
        m.shortie(src)                    # writes m.projector.k / .v
    with phase("tallie"):
        m(query)                          # inside forward(): label(k, "K · appended"); label(v, "V · appended")
```
(`model.shortie` itself is not a module call, so the encoder appears directly inside the region. The
wrapper class of the root model is dissolved automatically when phases are used; use `--unwrap Cls`
to dissolve other wrapper modules.) See `examples/two_phase_spec.py` and `examples/vlm_spec.py`.

## Flags

| flag | meaning |
|---|---|
| `-o PATH` | output path without extension (`.svg`, `.html`, `.png` appended) |
| `--depth N` | how many module levels to expand (default 4). `2` = block overview |
| `--ops none\|merge\|key\|all` | functional ops drawn as nodes. `key` (default): adds/muls/matmul/sdpa/softmax/cat… ; `merge`: only ops with ≥2 tensor inputs; `none`: modules only; `all`: everything non-reshape |
| `--theme light\|blueprint` | white diagram, or dark-blue engineering-drawing style with grid |
| `--png` | also rasterise (needs cairosvg or rsvg-convert) |
| `--no-collapse` | draw every repeated layer instead of `×N` |
| `--hide-types A,B` | module classes treated as invisible pass-through (default `Identity,Dropout`) |
| `--unwrap A,B` | dissolve wrapper modules of these classes (their children are drawn in their place) |
| `--leaf-types A,B` | draw these classes as a single box even if they have children |
| `--no-card-io` / `--no-edge-labels` | declutter: drop in/out rows in cards / the flow names on edges |
| `--hide-skips` | don't draw residual skip lines; everything else stays exactly where it was (their lane space is kept) |
| `--edge-shapes` | legacy style: put shapes and `↳` notes on the edges instead of on the nodes |
| `--dtypes auto\|all\|none` | when to print dtypes (auto: only if non-float or differs from params) |
| `--device cpu` | if tracing fails on `meta` (data-dependent control flow); use **tiny** dims |
| `--title TEXT` | diagram title |

## Reading the diagram

Data flows **bottom → top**. Solid rounded boxes are leaf modules (colored by kind: embedding, norm,
attention, MLP, activation, …); large tinted boxes are containers; small dashed cards are functional
ops (SDPA, cat, masked_scatter …) with their output shape; circles are `+` / `×`. A dashed container
border with `×N` is a repeated *pattern*; a large dashed region is a phase (entry point).

- **Rows** inside a card/header: `in` / `out` in a left column; further shapes line up under the first
  (`in  x=[b, s, d]` / `    ext_k=[…]`). A `↳ ops [shape]` row under `out` says which hidden
  reshapes follow before the consumer sees the tensor.
- **Lines** are continuous across container borders. A flow's *name* (from `label()`) is printed once.
- **Branch vs. crossing**: a filled dot marks a place where one tensor really splits; two different
  tensors that merely cross have no dot, and the line being crossed is interrupted by a small gap.
- Residual skips branch off the stream to the right and enter `+` from the right side; the main path
  enters from below. The *central* input of a node (K among Q K V) always uses its central port.

## Layout self-check

`archsheet` prints one line, e.g. `layout check: 120 edge pieces, 12 branch points, 5 crossings,
0 short jogs`. Use it instead of eyeballing: many crossings (> ~12) or short jogs (> ~3) mean the view
is too dense — lower `--depth`, use `--ops merge`, or collapse a module with `--leaf-types`. Do **not**
edit the generated SVG/HTML by hand; change the spec or the flags and rerun.

## Troubleshooting

- **`forward pass failed`**: the model does something meta tensors can't (`.item()`, `if x.any()`,
  `nonzero`, MoE routing by value). Retry `--device cpu` with tiny dims, or add `run=` in the spec.
- **warnings `symbols X and Y have the same value`**: change example sizes in the spec.
- **A module is missing**: it was never called in `forward` (unused / only used in training), or it is
  in `--hide-types`.
- **Layers not collapsed**: they differ in a config attribute or shape; the outline shows what differs.
  Layers must live in a `ModuleList`/`Sequential` (names ending in `.0`, `.1`, …).
- **Huge diagram**: lower `--depth`, use `--ops merge`, or `--no-card-io`.
- A module called several times (weight sharing) is drawn once per call.
- Only PyTorch is supported. For JAX/TF models, port the structure to a stub `nn.Module` with the same
  shapes, or ask the user.

## Files

- `archsheet/` package: `cli.py` CLI · `trace.py` tracer · `graph.py` hierarchy/symbols · `layout.py` per-container layout+routing · `render.py` sizing+SVG/HTML/PNG (`python -m archsheet`)
- `templates/spec_template.py` spec to copy
- `examples/gemma_like_spec.py` GQA, QK-norm, RoPE, sandwich norms, 5:1 local/global
- `examples/vlm_spec.py` vision tower + mm-projector branching into the LLM's input embeddings
- `examples/two_phase_spec.py` `shortie` / `tallie` regions, projector, appended-K/V labels
- `examples/output/` pre-rendered results
