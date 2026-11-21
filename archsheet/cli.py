"""archsheet — render an informative, blueprint-style architecture sheet of a PyTorch model.

    python -m archsheet SPEC.py -o out/mymodel [--depth 4] [--theme light|blueprint] [--png]

SPEC.py must define `build()` (see templates/spec_template.py). The model is traced on the
`meta` device, so nothing is allocated and nothing is computed.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback

import torch

from . import layout as _layout
from .graph import Graph, Symbols, human
from .render import Layout, Renderer, THEMES, to_html, to_png
from .trace import Tracer


def load_spec(path):
    spec = importlib.util.spec_from_file_location("archsheet_user_spec", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    sys.path.insert(0, os.getcwd())
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build"):
        sys.exit(f"{path} must define build()")
    return mod


def normalize(cfg):
    if isinstance(cfg, dict):
        d = dict(cfg)
    else:
        cfg = tuple(cfg)
        d = dict(model=cfg[0], inputs=cfg[1], symbols=cfg[2] if len(cfg) > 2 else {})
    inputs = d.pop("inputs", None)
    d.setdefault("args", ())
    d.setdefault("kwargs", {})
    if inputs is not None:
        if isinstance(inputs, dict):
            d["kwargs"] = inputs
        elif isinstance(inputs, (list, tuple)):
            d["args"] = tuple(inputs)
        else:
            d["args"] = (inputs,)
    if "model" not in d:
        sys.exit("build() must return a model")
    d["args"] = tuple(d["args"])
    d.setdefault("symbols", {})
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m archsheet", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("-o", "--out", default="archsheet", help="output path without extension")
    ap.add_argument("--depth", type=int, default=4, help="levels of submodules to expand (default 4)")
    ap.add_argument("--ops", choices=["none", "merge", "key", "all"], default="key",
                    help="functional ops to draw: none | merge (multi-input only) | key (default) | all")
    ap.add_argument("--theme", choices=list(THEMES), default="light")
    ap.add_argument("--formats", default="svg,html", help="comma list of svg,html,png")
    ap.add_argument("--png", action="store_true", help="also write a PNG (needs cairosvg or rsvg-convert)")
    ap.add_argument("--no-collapse", action="store_true", help="draw every repeated layer")
    ap.add_argument("--hide-types", default="Identity,Dropout", help="module class names to treat as pass-through")
    ap.add_argument("--leaf-types", default="", help="module class names to draw as one box (no expansion)")
    ap.add_argument("--unwrap", default="", help="module class names to dissolve (children drawn in their place)")
    ap.add_argument("--device", default="meta", help="meta (default) or cpu (if the model needs real values)")
    ap.add_argument("--dtypes", choices=["auto", "all", "none"], default="auto")
    ap.add_argument("--no-card-io", action="store_true", help="omit in/out lines inside cards")
    ap.add_argument("--edge-shapes", action="store_true",
                    help="write shapes and reshape notes on the edges instead of on the nodes")
    ap.add_argument("--hide-skips", action="store_true",
                    help="do not draw residual skip lines (the layout of everything else is unchanged)")
    ap.add_argument("--no-edge-labels", action="store_true", help="omit shape labels on edges")
    ap.add_argument("--title", default=None)
    ap.add_argument("--outline-lines", type=int, default=250)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    _layout.HIDE_SKIPS = a.hide_skips
    mod = load_spec(a.spec)
    dev = torch.device(a.device)
    try:
        with torch.device(dev):
            cfg = normalize(mod.build())
    except Exception:
        traceback.print_exc()
        sys.exit("build() failed. If it needs real weights/values, use --device cpu with tiny dims.")
    model = cfg["model"].eval()
    main_dtype = next((str(p.dtype).replace("torch.", "") for p in model.parameters()), "float32")
    sym = Symbols(cfg["symbols"], main_dtype, a.dtypes)

    tracer = Tracer(model, ops=a.ops,
                    hide_types=[s for s in a.hide_types.split(",") if s],
                    leaf_types=[s for s in a.leaf_types.split(",") if s]).install()
    try:
        with torch.no_grad(), torch.device(dev), tracer:
            run = cfg.get("run")
            run(model) if run else model(*cfg["args"], **cfg["kwargs"])
    except Exception:
        traceback.print_exc()
        sys.exit("forward pass failed while tracing. Data-dependent code (.item(), nonzero, python "
                 "control flow on values) cannot run on 'meta': retry with --device cpu and tiny sizes, "
                 "or give a `run(model)` function in the spec.")
    finally:
        tracer.remove()
    tracer.finalize()

    unwrap = {t for t in a.unwrap.split(",") if t}
    if any(c.kind == "phase" for c in tracer.scene.children):      # phases already say "forward()" etc.
        unwrap.add(type(model).__name__)
    g = Graph(tracer, sym, max_depth=a.depth, collapse=not a.no_collapse, unwrap=unwrap,
              node_ann=not a.edge_shapes)
    lay = Layout(g, a.theme, card_io=not a.no_card_io, edge_labels=not a.no_edge_labels,
                 edge_shapes=a.edge_shapes)
    lay.size(g.scene)
    title = a.title or cfg.get("title") or tracer.root_node.type
    svg = Renderer(lay, title, a.theme).svg()

    out = a.out
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fmts = {f.strip() for f in a.formats.split(",") if f.strip()}
    if a.png:
        fmts.add("png")
    written = []
    if "svg" in fmts or "png" in fmts:
        open(out + ".svg", "w", encoding="utf-8").write(svg)
        written.append(out + ".svg")
    if "html" in fmts:
        open(out + ".html", "w", encoding="utf-8").write(to_html(svg, title, a.theme))
        written.append(out + ".html")
    if "png" in fmts:
        tool = to_png(out + ".svg", out + ".png")
        if tool:
            written.append(out + ".png")
        else:
            print("! PNG skipped: install cairosvg (pip install cairosvg) or rsvg-convert", file=sys.stderr)

    if not a.quiet:
        print(f"== {title}: {human(tracer.total_params)} params, symbols: "
              + ", ".join(f"{k}={v}" for k, v, _ in sym.legend()))
        print(g.outline(a.outline_lines))
    warns = list(dict.fromkeys(g.warnings + lay.warnings))
    for w in warns[:20]:
        print("! " + w, file=sys.stderr)
    r = lay.report
    print(f"layout check: {r.get('edges', 0)} edge pieces, {r.get('junctions', 0)} branch points, "
          f"{r.get('crossings', 0)} crossings, {r.get('short_jogs', 0)} short jogs"
          + ("  <- try a lower --depth or --ops merge" if r.get("crossings", 0) > 12 or r.get("short_jogs", 0) > 3 else ""))
    print("wrote: " + ", ".join(written))

