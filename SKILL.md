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
- repeated layers collapsed to `×N`, periodic patterns detected (`12 layers = 2 × pattern of 6`); attributes
  that differ between variants are named on the owning node (`self_attn · sliding window 8` vs
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
   Read the printed text **outline** first: shapes that stay numeric (`[b, s, 160]`) mean a missing symbol
   (products of symbols are found automatically).
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
- **Phases** (`shortie` / `tallie`) sit side by side; flows between them leave the earlier phase's right
  border and enter the later one's left border as one straight horizontal line.
- Flows are identified by their *producer*: K and V from one projector are two lines; give each its own
  `label(...)`.

## Multimodal / branching / multi-entry models

Plain `model(**inputs)` already handles branch-in (every tensor is followed from producer to consumer, so
`inputs_embeds.masked_scatter(mask, image_features)` becomes a merge node with three inputs). Give each
input its own symbol (`n` patches, `hw` image size ...).

For several entry points or persistent state (stored K/V, KV cache, encoder/decoder split) give the spec a
`run(model)`, wrap each call in `phase(...)` and name tensors with `label(...)`. Tensors kept on the model
between calls keep their origin, so the arrow between phases is drawn automatically:

```python
from archsheet import phase, label

def run(m):
    with phase("shortie"):
        m.shortie(src)                    # writes m.projector.k / .v
    with phase("tallie"):
        m(query)                          # in forward(): label(k, "K · appended"); label(v, "V · appended")
```
(`model.shortie` is not a module call, so the encoder appears directly in the region. The root wrapper class
is dissolved automatically when phases are used; `--unwrap Cls` dissolves others.) See
`examples/two_phase_spec.py` and `examples/vlm_spec.py`.

## Flags

| flag | meaning |
|---|---|
| `-o PATH` | output path without extension |
| `--depth N` | module levels to expand (default 4); `2` = block overview |
| `--ops none\|merge\|key\|all` | functional ops drawn as nodes: `key` (default) adds/muls/matmul/sdpa/softmax/cat…; `merge` only ops with ≥2 tensor inputs; `none` modules only; `all` everything non-reshape |
| `--theme light\|blueprint` | white, or dark-blue engineering style with grid |
| `--png` | also rasterise (cairosvg or rsvg-convert) |
| `--no-collapse` | draw every repeated layer instead of `×N` |
| `--hide-types A,B` | classes treated as pass-through (default `Identity,Dropout`) |
| `--unwrap A,B` | dissolve wrapper modules (children drawn in their place) |
| `--leaf-types A,B` | draw these classes as one box even if they have children |
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
  `nonzero`, value-routed MoE). Retry `--device cpu` with tiny dims, or add `run=` in the spec.
- **`symbols X and Y have the same value`**: change the example sizes.
- **Module missing**: never called in `forward`, or listed in `--hide-types`.
- **Layers not collapsed**: they differ in a config attribute or shape (the outline shows what) or are not
  in a `ModuleList`/`Sequential`. A module called several times (weight sharing) is drawn once per call.
- Only PyTorch is supported; for JAX/TF port the structure to a stub `nn.Module` with the same shapes.

## Files

`archsheet/` (`cli` · `trace` · `graph` · `layout` · `render`), `templates/spec_template.py`,
`examples/{gemma_like,vlm,two_phase}_spec.py` (+ `examples/output/` pre-rendered results).
