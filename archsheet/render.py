"""archsheet.render — layered (bottom→top) orthogonal layout and SVG/HTML output.

Every container lays out its own children with a small Sugiyama-style pass:
  layers  = longest path from the container inputs (data flows upward)
  order   = execution order, refined with barycenter sweeps
  routing = leave a node from its top, travel in a per-edge "track" inside the gap
            between layers, enter the consumer from its bottom; edges that skip
            layers (residuals!) travel in a vertical lane on the right.
Edge labels sit on the tracks, so labels of different edges never overlap.
"""
from __future__ import annotations

import bisect
import html
import re
from collections import defaultdict

from .graph import human
from .layout import layout_items as _layout_items
from .trace import SHAPE_OPS

FONT = 'ui-monospace, "SF Mono", Menlo, Consolas, "DejaVu Sans Mono", "Liberation Mono", monospace'
CWR = 0.602                      # monospace advance / font-size

# ---------------------------------------------------------------- themes
THEMES = {
    "light": dict(
        bg="#ffffff", fg="#111827", dim="#6b7280", edge="#374151", card="#ffffff",
        neutral_stroke="#6b7280", container=["#f5f6f8", "#eceff3", "#e3e7ee"],
        mix_to="#ffffff", c_amt=0.90, k_amt=0.80, grid=None,
        cat=dict(embed="#d6336c", pos="#ae3ec9", norm="#d9a400", attn="#f76707", moe="#7048e8",
                 mlp="#1c7ed6", act="#2f9e44", conv="#0c8599", linear="#5c6b7a", resid="#374151",
                 gate="#374151", op="#6b7280")),
    "blueprint": dict(
        bg="#0b2a4a", fg="#e8f3ff", dim="#8db6dd", edge="#a8d0f5", card="#0b2a4a",
        neutral_stroke="#6fa3d6", container=["#0e3257", "#0c2c4e", "#0a2644"],
        mix_to="#0b2a4a", c_amt=0.88, k_amt=0.74, grid="#12395f",
        cat=dict(embed="#ff8fb3", pos="#e599f7", norm="#ffe066", attn="#ffa94d", moe="#b197fc",
                 mlp="#74c0fc", act="#8ce99a", conv="#66d9e8", linear="#ced4da", resid="#a8d0f5",
                 gate="#a8d0f5", op="#8db6dd")),
}
CAT_RULES = [
    ("rotary|rope", "pos"), ("embed|wte|wpe|lookup", "embed"), ("position|pos_", "pos"),
    ("norm", "norm"), ("attn|attention", "attn"), ("moe|expert|router", "moe"),
    ("mlp|ffn|feedforward|feed_forward", "mlp"),
    ("gelu|silu|relu|swish|sigmoid|tanh|softmax|act_fn|activation", "act"), ("conv", "conv"),
    ("linear|proj|lm_head|dense", "linear"),
]
CAT_LABEL = dict(embed="embedding", pos="position / rope", norm="normalization", attn="attention",
                 moe="mixture of experts", mlp="feed-forward", act="activation", conv="convolution",
                 linear="linear / projection")
OP_GLYPH = {"add": "+", "mul": "×", "sub": "−", "div": "÷"}
OP_RENAME = {"scaled_dot_product_attention": "SDPA", "sdpa": "SDPA", "getitem": "idx",
             "multi_head_attention_forward": "mha"}


def category(n):
    if n.kind == "op":
        return {"add": "resid", "mul": "gate"}.get(n.name, "op")
    if n.kind != "module":
        return None
    s = f"{n.name} {n.type}".lower()
    for pat, cat in CAT_RULES:
        if re.search(pat, s):
            return cat
    return None


def mix(a, b, t):
    """t=0 -> a, t=1 -> b"""
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(pa, pb))


def esc(s):
    return html.escape(str(s), quote=True)


def tw(text, fs=11.0):
    return len(text) * CWR * fs


def clip(text, n):
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------------------------------------------------------------- layout
EQUALIZE_PHASE_LINKS = False      # True: shift phases until every cross-phase flow is one straight line (leaves gaps)


class Layout:
    def __init__(self, graph, theme="light", card_io=True, edge_labels=True, edge_shapes=False):
        self.g, self.sym = graph, graph.sym
        self.T = THEMES[theme]
        self.card_io, self.edge_labels, self.edge_shapes = card_io, edge_labels, edge_shapes
        self.report = {}
        self.glyph_ops = set(OP_GLYPH)
        self.used_cats = set()
        self.warnings = []

    def io_rows(self, n, limit=4):
        """(label, text, kind) rows. `in` / `out` sit in a label column and further shapes line up under
        the first one (like a table); `↳` rows say which hidden reshapes follow, next to the producer."""
        rows = []
        names = n.in_names if len(n.in_names) == len(n.in_metas) else []
        multi = len(n.in_metas) > 1
        for i, m in enumerate(n.in_metas[:limit]):
            txt = self.sym.shape(m)
            if multi and names:
                txt = f"{names[i]}={txt}"
            rows.append(("in" if i == 0 else "", txt, "io"))
        if len(n.in_metas) > limit:
            rows.append(("", f"+{len(n.in_metas) - limit} more", "io"))
        for m in n.in_ext[:3]:
            rows.append(("", "+ " + self.sym.shape(m), "io"))
        if len(n.in_ext) > 3:
            rows.append(("", f"+{len(n.in_ext) - 3} more read", "io"))
        rows += [("", a, "ann") for a in n.in_ann[:3]]
        for i, m in enumerate(n.out_metas[:limit]):
            rows.append(("out" if i == 0 else "", self.sym.shape(m), "io"))
        if len(n.out_metas) > limit:
            rows.append(("", f"+{len(n.out_metas) - limit} more", "io"))
        rows += [("", a, "ann") for a in n.out_ann[:3]]
        if len(n.out_ann) > 3:
            rows.append(("", f"+{len(n.out_ann) - 3} more", "ann"))
        return rows

    @staticmethod
    def row_w(r):
        label, txt, kind = r
        return tw(("    " if kind == "io" else " ") + txt, 10.5)

    # -------- sizing of leaves
    def shapes(self, metas, limit=3):
        s = [self.sym.shape(m) for m in metas[:limit]]
        if len(metas) > limit:
            s.append(f"+{len(metas) - limit} more")
        return s

    def card_lines(self, n):
        """list of (text, style) for leaf / collapsed cards."""
        L = []
        params = human(n.params) if n.params else ""
        L.append(("title", n.name, params))
        sub = n.type
        if n.extra:
            sub += "  " + n.extra
        if n.kind == "module" and not n.leaf and n.dchildren:
            sub = f"{n.type} · {len(n.dchildren)} submodules"
        L.append(("sub", clip(sub, 52), ""))
        if n.note:
            L.append(("note", clip(n.note, 52), ""))
        for r in self.io_rows(n):
            if self.card_io or r[2] == "ann":
                L.append(("row", r, ""))
        return L

    def size(self, n):
        k = n.kind
        if k == "tag":                         # invisible port on a container border
            n.lines = []
            n.w = n.h = 0.0
        elif k in ("input", "output"):
            m = (n.out_metas or n.in_metas)[0]
            n.lines = [n.name, self.sym.shape(m)]
            n.ann = (n.out_ann if k == "input" else n.in_ann)[:3]
            n.w = max([tw(n.lines[0], 12), tw(n.lines[1], 11)] + [tw(a, 10) for a in n.ann]) + 26
            n.h = 38 + 14 * len(n.ann)
        elif k == "op":
            if n.name in OP_GLYPH:
                n.w = n.h = 28
            else:                                  # functional op drawn as a small card: name, out shape, notes
                label = f"cat · dim {n.cat[0]}" if n.cat else OP_RENAME.get(n.name, n.name)
                shp = self.sym.shape(n.out_metas[0], dtype=False, cat=n.cat) if n.out_metas else ""
                if len(n.out_metas) > 1:
                    shp += f" +{len(n.out_metas) - 1}"
                notes = (n.out_ann + n.in_ann)[:3]
                n.oplines = [("title", label), ("shape", shp)] + [("ann", a) for a in notes]
                n.w = max([tw(label, 11), tw(shp, 10)] + [tw(a, 10) for a in notes]) + 22
                n.h = 10 + 14 * len(n.oplines)
        elif n.expanded:
            self.size_container(n)
        else:
            n.lines = self.card_lines(n)
            wmax = 0
            for st, a, b in n.lines:
                if st == "title":
                    wmax = max(wmax, tw(a, 12) + (tw(b, 10) + 14 if b else 0))
                elif st == "row":
                    wmax = max(wmax, self.row_w(a))
                else:
                    wmax = max(wmax, tw(a, 10.5))
            n.w = wmax + 22 + (30 if n.repeat > 1 else 0)
            n.h = 12 + 15 * len(n.lines)

    # -------- containers
    def header(self, n):
        if n.kind == "scene":
            return [], 0, 0
        if n.kind == "phase":
            return [("title", n.name, "")], 27, tw(n.name, 12.5) + 40
        lines = []
        params = human(n.params) if n.params else ""
        if n.kind == "group":
            title = n.name
            lines.append(("title", title, f"{n.type}"))
        else:
            lines.append(("title", n.name, f"{n.type}" + (f" · {params}" if params else "")))
        if len(n.in_metas) <= 1 and len(n.out_metas) <= 1 and not (n.in_ann or n.out_ann):
            ins = ", ".join(self.shapes(n.in_metas, 2))
            outs = ", ".join(self.shapes(n.out_metas, 2))
            if ins or outs:
                lines.append(("io", clip(f"in {ins}  →  out {outs}", 96), ""))
        else:
            for r in self.io_rows(n):
                lines.append(("row", r, ""))
        if n.note:
            lines.append(("note", clip(n.note, 90), ""))
        hw = 0
        for st, a, b in lines:
            hw = max(hw, self.row_w(a) if st == "row" else tw(a, 12 if st == "title" else 10.5) + (tw(b, 10.5) + 16 if b else 0))
        hw += 28 + (44 if n.repeat > 1 else 0)
        return lines, 12 + 15 * len(lines), hw

    def plate_rights(self, n):
        """Right edges of the header text plates of a container (lines must not run under many of them)."""
        out = []
        for st, a_, b_ in n.hlines:
            wtxt = self.row_w(a_) if st == "row" else tw(a_, 12.5 if st == "title" else 10.5) + (
                (tw(b_, 10.5) + 10) if (st == "title" and b_) else 0)
            out.append(10 + wtxt + 8)
        return out

    def size_container(self, n):
        items = list(n.dchildren) + list(n.tags_in.values()) + list(n.tags_out.values())
        for it in items:
            self.size(it)
        lines, header_h, header_w = self.header(n)
        n.hlines = lines
        pad = 0 if n.kind == "scene" else 12
        n._layout_args = (items, header_h, header_w, pad)
        self.layout_items(n, items, header_h, header_w, pad)
        if n.kind == "scene" and EQUALIZE_PHASE_LINKS:
            self.equalize_phase_links(n)

    def equalize_phase_links(self, scene):
        """A flow that leaves one phase and enters the next must be ONE straight horizontal line: no little
        up/down jogs at the dashed borders. Phases are bottom-aligned, so that means the exit track and the
        entry track must be the same distance above the bottom. Tracks can only be pushed up (by growing a gap),
        so both ends are raised to the higher one and the two phases are laid out again (a few rounds, because
        raising one flow moves the other flows above it)."""
        links = [sg for sg in self.g.segments.get(scene.id, [])
                 if sg.src.kind == "phase" and sg.dst.kind == "phase" and sg.src_tag is not None and sg.dst_tag is not None]
        for _ in range(12):
            redo = set()
            for sg in links:
                e, d = sg.src.h - sg.src_tag.y, sg.dst.h - sg.dst_tag.y
                top = max(e, d)
                for ph, tag, cur in ((sg.src, sg.src_tag, e), (sg.dst, sg.dst_tag, d)):
                    if cur < top - 0.5:
                        tag.min_from_bottom = max(getattr(tag, "min_from_bottom", 0.0), top)
                        redo.add(ph.id)
            if not redo:
                break
            for ph in [c for c in scene.dchildren if c.id in redo]:
                self.layout_items(ph, *ph._layout_args)
            self.layout_items(scene, *scene._layout_args)

    def layout_items(self, n, items, header_h, header_w, pad):
        _layout_items(self, n, items, header_h, header_w, pad)

    # -------- labels
    def seg_lines(self, s, seen):
        """(group lines, annotation lines) for an edge. By default an edge only carries the *name* of a flow
        (from `label()`), once; shapes and `↳` notes are written on the nodes. `edge_shapes=True` restores
        shapes + notes on the edges."""
        if not self.edge_labels:
            return [], []
        g, a = [], []
        if s.label and s.carry:
            g.append(s.label)
        if self.edge_shapes:
            plain = s.src.kind in ("tag", "input", "output") or s.dst.kind in ("tag", "input", "output")
            key = ("S", s.src.id, s.fid)
            if not plain and key not in seen:
                seen.add(key)
                g.append(self.sym.shape(s.mfrom, cat=s.src.cat if s.src.kind == "op" else None))
            if s.ann:
                txt = self.g._ann_text(s)
                if txt:
                    a.append(txt)
        return g, a


# ---------------------------------------------------------------- SVG
class Renderer:
    def __init__(self, layout: Layout, title="", theme="light"):
        self.L, self.g, self.sym = layout, layout.g, layout.sym
        self.T = THEMES[theme]
        self.theme = theme
        self.title = title

    # -- primitives
    def text(self, x, cy, s, fs=11, w="normal", fill=None, anchor="start", style=""):
        fill = fill or self.T["fg"]
        return (f'<text x="{x:.1f}" y="{cy + fs * 0.36:.1f}" font-size="{fs}" font-weight="{w}" '
                f'fill="{fill}" text-anchor="{anchor}"{style}>{esc(s)}</text>')

    def colors(self, n, container=False):
        T = self.T
        cat = category(n)
        if cat:
            self.L.used_cats.add(cat)
            acc = T["cat"][cat]
            if container:
                return mix(acc, T["mix_to"], T["c_amt"]), acc
            return mix(acc, T["mix_to"], T["k_amt"]), acc
        if container:
            return T["container"][n.depth % 3], T["neutral_stroke"]
        return T["card"], T["neutral_stroke"]

    def row(self, x, cy, r):
        label, txt, kind = r
        cw = 10.5 * CWR
        if kind == "ann":
            return self.text(x + cw, cy, txt, 10.5, "normal", self.T["dim"])
        out = self.text(x, cy, label, 10.5, "normal", self.T["fg"]) if label else ""
        return out + self.text(x + 4 * cw, cy, txt, 10.5, "normal", self.T["fg"])

    def tip(self, n):
        bits = [f"{n.type or n.kind}  {n.path or n.name}"]
        if n.extra:
            bits.append(n.extra)
        if n.params:
            bits.append(f"params: {n.params:,}")
        for m in n.in_metas[:6]:
            bits.append("in:  " + self.sym.concrete(m))
        for m in n.out_metas[:6]:
            bits.append("out: " + self.sym.concrete(m))
        return "<title>" + esc("\n".join(bits)) + "</title>"

    def badge(self, w, txt):
        bw = tw(txt, 10.5) + 12
        return (f'<rect x="{w - bw - 8:.1f}" y="-9" width="{bw:.1f}" height="18" rx="9" fill="{self.T["fg"]}"/>'
                + self.text(w - 8 - bw / 2, 0, txt, 10.5, "bold", self.T["bg"], "middle"))

    # -- nodes
    def node(self, n):
        out = [f'<g transform="translate({n.x:.1f},{n.y:.1f})">']
        k = n.kind
        T = self.T
        if k == "tag":
            return ""
        elif k in ("input", "output"):
            out.append(self.tip(n))
            out.append(f'<rect width="{n.w:.1f}" height="{n.h}" rx="{min(19, n.h / 2):.0f}" fill="{T["fg"]}" stroke="none"/>')
            out.append(self.text(n.w / 2, 12, n.lines[0], 12, "bold", T["bg"], "middle"))
            out.append(self.text(n.w / 2, 27, n.lines[1], 10.5, "normal", T["bg"], "middle"))
            for i, a_ in enumerate(n.ann):
                out.append(self.text(n.w / 2, 41 + 14 * i, a_, 10, "normal", T["bg"], "middle"))
        elif k == "op":
            out.append(self.tip(n))
            fill, stroke = self.colors(n)
            if n.name in OP_GLYPH:
                r = n.w / 2
                out.append(f'<circle cx="{r}" cy="{r}" r="{r - 1}" fill="{T["bg"]}" stroke="{T["edge"]}" stroke-width="1.6"/>')
                out.append(self.text(r, r - 0.5, OP_GLYPH[n.name], 17, "normal", T["fg"], "middle"))
            else:
                out.append(f'<rect width="{n.w:.1f}" height="{n.h}" rx="10" fill="{T["bg"]}" stroke="{T["edge"]}" stroke-dasharray="4 2"/>')
                yy = 5 + 7
                for st, txt in n.oplines:
                    out.append(self.text(n.w / 2, yy, txt, 11 if st == "title" else 10, "bold" if st == "title" else "normal",
                                         T["fg"] if st == "title" else T["dim"], "middle"))
                    yy += 14
        elif n.expanded:
            out += self.container(n)
        else:
            out += self.card(n)
        out.append("</g>")
        return "".join(out)

    def card(self, n):
        T = self.T
        fill, stroke = self.colors(n)
        o = [self.tip(n),
             f'<rect width="{n.w:.1f}" height="{n.h:.1f}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>']
        y = 6 + 7.5
        for st, a, b in n.lines:
            if st == "title":
                o.append(self.text(11, y, a, 12, "bold"))
                if b:
                    o.append(self.text(n.w - 11 - (30 if n.repeat > 1 else 0), y, b, 10, "normal", T["dim"], "end"))
            elif st == "sub":
                o.append(self.text(11, y, a, 10.5, "normal", T["dim"]))
            elif st == "note":
                o.append(self.text(11, y, a, 10.5, "normal", stroke, style=' font-style="italic"'))
            elif st == "row":
                o.append(self.row(11, y, a))
            else:
                o.append(self.text(11, y, a, 10.5, "normal", T["fg"]))
            y += 15
        if n.repeat > 1:
            o.append(self.badge(n.w, f"×{n.repeat}"))
        return o

    def container(self, n):
        T = self.T
        o = []
        head = []
        if n.kind != "scene":
            fill, stroke = self.colors(n, container=True)
            if n.kind == "phase":
                fill, stroke = "none", T["dim"]
            dash = ' stroke-dasharray="7 4"' if n.kind in ("group", "phase") else ""
            o.append(self.tip(n))
            o.append(f'<rect width="{n.w:.1f}" height="{n.h:.1f}" rx="{18 if n.kind == "phase" else 12}" fill="{fill}" stroke="{stroke}" '
                     f'stroke-width="1.8"{dash}/>')
            head = []                      # drawn last, on a plate, so lines never strike through it
            y = 6 + 7.5
            for st, a_, b_ in n.hlines:
                fs = 12.5 if st == "title" else 10.5
                wtxt = self.L.row_w(a_) if st == "row" else tw(a_, fs) + ((tw(b_, 10.5) + 10) if (st == "title" and b_) else 0)
                head.append(f'<rect x="10" y="{y - 8:.1f}" width="{wtxt + 8:.1f}" height="16" rx="3" fill="{T["bg"] if fill == "none" else fill}"/>')
                if st == "title":
                    head.append(self.text(14, y, a_, 12.5, "bold"))
                    if b_:
                        head.append(self.text(14 + tw(a_, 12.5) + 10, y, b_, 10.5, "normal", T["dim"]))
                elif st == "note":
                    head.append(self.text(14, y, a_, 10.5, "normal", stroke, style=' font-style="italic"'))
                elif st == "row":
                    head.append(self.row(14, y, a_))
                else:
                    head.append(self.text(14, y, a_, 10.5, "normal", T["dim"]))
                y += 15
            if n.repeat > 1:
                head.append(self.badge(n.w, f"×{n.repeat}"))

        # edges first (under children)
        for p in n.pieces:
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in p["pts"])
            mk = ' marker-end="url(#arr)"' if p["arrow"] else ""
            o.append(f'<polyline points="{pts}" fill="none" stroke="{T["edge"]}" stroke-width="1.5" '
                     f'stroke-linejoin="round"{mk}><title>{esc(self.sym.concrete(p["shape"]))}</title></polyline>')
        for c in list(n.dchildren) + list(n.tags_in.values()) + list(n.tags_out.values()):
            o.append(self.node(c))
        for lb in n.labels:
            lines, h = lb["lines"], 12 * len(lb["lines"])
            wmax, top = lb["w"], lb["y"] - h / 2
            bx = lb["x"] - wmax / 2 if lb["anchor"] == "middle" else lb["x"] - 3
            o.append(f'<rect x="{bx:.1f}" y="{top:.1f}" width="{wmax:.1f}" height="{h}" rx="3" fill="{T["bg"]}" opacity="0.92"/>')
            for i, l in enumerate(lines):
                ann = l.startswith("↳")
                name = not ann and "[" not in l
                x = lb["x"] if lb["anchor"] == "middle" else lb["x"] + 1
                o.append(self.text(x, top + 6 + 12 * i, l, 10.5 if name else 10, "bold" if name else "normal",
                                   T["dim"] if ann else T["fg"], lb["anchor"]))
        return o + head

    def overlay(self):
        """Junction dots where a flow really branches, small breaks where two different flows merely cross,
        plus the numbers for the layout self-check."""
        T = self.T
        pcs = []

        def collect(n, ox, oy):
            if not (n.kind == "scene" or n.expanded):
                return
            fill = T["bg"] if n.kind in ("scene", "phase") else self.colors(n, container=True)[0]
            for p in getattr(n, "pieces", []):
                pcs.append((p.get("fid"), [(ox + x, oy + y) for x, y in p["pts"]], fill))
            for c in n.dchildren:
                collect(c, ox + c.x, oy + c.y)

        collect(self.g.scene, 0.0, 0.0)
        byflow, fill_of = defaultdict(set), {}
        for fid, pts, fill in pcs:
            for a, b in zip(pts, pts[1:]):
                p1, p2 = sorted([(round(a[0], 1), round(a[1], 1)), (round(b[0], 1), round(b[1], 1))])
                if p1 != p2:
                    byflow[fid].add(p1 + p2)
                    fill_of[(fid,) + p1 + p2] = fill
        H, V = [], []
        for fid, segs in byflow.items():
            for x1, y1, x2, y2 in segs:
                (H if y1 == y2 else V).append((fid, x1, y1, x2, y2))
        Vs = sorted(V, key=lambda v: v[1])
        xs = [v[1] for v in Vs]
        crossings = {}
        for fid, x1, y, x2, _ in H:
            for v in Vs[bisect.bisect_right(xs, x1 + 2): bisect.bisect_left(xs, x2 - 2)]:
                if v[0] != fid and v[2] + 2 < y < v[4] - 2:
                    crossings[(v[1], y)] = fill_of[(v[0],) + v[1:]]
        sgn = lambda v: (v > 0) - (v < 0)
        dots = set()
        for fid, segs in byflow.items():
            pts = {(s_[0], s_[1]) for s_ in segs} | {(s_[2], s_[3]) for s_ in segs}
            for px, py in pts:
                dirs = set()
                for x1, y1, x2, y2 in segs:
                    if (x1, y1) == (px, py):
                        dirs.add((sgn(x2 - x1), sgn(y2 - y1)))
                    elif (x2, y2) == (px, py):
                        dirs.add((sgn(x1 - x2), sgn(y1 - y2)))
                    elif y1 == y2 == py and x1 < px < x2:
                        dirs |= {(1, 0), (-1, 0)}
                    elif x1 == x2 == px and y1 < py < y2:
                        dirs |= {(0, 1), (0, -1)}
                if len(dirs) >= 3:
                    dots.add((px, py))
        jogs = len({(h[1], h[2]) for h in H if 0.5 < h[3] - h[1] < 10})
        self.L.report = dict(edges=len(pcs), crossings=len(crossings), junctions=len(dots), short_jogs=jogs)
        o = []
        for (x, y), fill in crossings.items():          # the crossed (vertical) line is interrupted
            fill = T["bg"] if fill == "none" else fill
            o.append(f'<rect x="{x - 2.4:.1f}" y="{y - 5:.1f}" width="4.8" height="3.8" fill="{fill}"/>')
            o.append(f'<rect x="{x - 2.4:.1f}" y="{y + 1.2:.1f}" width="4.8" height="3.8" fill="{fill}"/>')
        for x, y in dots:
            o.append(f'<circle cx="{x}" cy="{y}" r="2.9" fill="{T["edge"]}"/>')
        return "".join(o)

    # -- whole document
    def svg(self):
        T, L = self.T, self.L
        scene = self.g.scene
        body = self.node(scene)       # lays out already done by caller
        over = self.overlay()
        banner, bh, bw = self.banner()
        margin = 28
        W = max(scene.w, bw) + 2 * margin + 24
        H = bh + scene.h + 2 * margin + 6
        sx = (W - scene.w) / 2
        grid = ""
        if T["grid"]:
            grid = (f'<defs><pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">'
                    f'<path d="M24 0H0V24" fill="none" stroke="{T["grid"]}" stroke-width="1"/></pattern></defs>'
                    f'<rect width="100%" height="100%" fill="url(#grid)"/>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}" '
                f'font-family=\'{FONT}\'>'
                f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{T["edge"]}"/></marker></defs>'
                f'<rect width="100%" height="100%" fill="{T["bg"]}"/>{grid}'
                f'<g transform="translate({margin},{margin})">{banner}</g>'
                f'<g transform="translate({sx:.1f},{margin + bh + 6})">{body}{over}</g></svg>')

    def banner(self):
        T = self.T
        root = self.g.root
        total = human(self.g.total_params) if self.g.total_params else ""
        o, y, wmax = [], 10, 0
        title = self.title or (root.type if root else "model")
        o.append(self.text(0, y, title, 20, "bold"))
        wmax = tw(title, 20)
        sub = f"{total} parameters · tensors shown as [dims] · symbols below" if total else "tensors shown as [dims]"
        o.append(self.text(0, y + 24, sub, 11, "normal", T["dim"]))
        wmax = max(wmax, tw(sub, 11))
        y += 50
        syms = self.sym.legend()
        x = 0
        for k, v, d in syms:
            chunk = f"{k} = {v}" + (f" ({d})" if d else "")
            wd = tw(chunk, 10.5) + 18
            if x + wd > 900 and x > 0:
                x, y = 0, y + 20
            o.append(f'<rect x="{x:.0f}" y="{y - 9}" width="{wd - 6:.0f}" height="18" rx="9" fill="none" stroke="{T["dim"]}"/>')
            o.append(self.text(x + 8, y, chunk, 10.5))
            x += wd
            wmax = max(wmax, x)
        if syms:
            y += 26
        # category legend
        x = 0
        for cat in sorted(self.L.used_cats):
            if cat not in CAT_LABEL:
                continue
            acc = T["cat"][cat]
            lab = CAT_LABEL[cat]
            o.append(f'<rect x="{x:.0f}" y="{y - 6}" width="12" height="12" rx="3" fill="{mix(acc, T["mix_to"], T["k_amt"])}" stroke="{acc}"/>')
            o.append(self.text(x + 18, y, lab, 10.5, "normal", T["dim"]))
            x += 18 + tw(lab, 10.5) + 16
        wmax = max(wmax, x)
        y += 20
        return "".join(o), y, wmax


# ---------------------------------------------------------------- html wrapper
HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  html,body{{margin:0;height:100%;background:{bg};color:{fg};font:13px ui-monospace,Menlo,Consolas,monospace}}
  #bar{{position:fixed;top:10px;right:10px;z-index:5;display:flex;gap:6px}}
  #bar button{{background:{bg};color:{fg};border:1px solid {dim};border-radius:6px;padding:4px 10px;cursor:pointer;font:inherit}}
  #bar button:hover{{border-color:{fg}}}
  #hint{{position:fixed;left:12px;bottom:10px;color:{dim};font-size:11px}}
  #v{{width:100%;height:100%;overflow:hidden;cursor:grab;touch-action:none}}
  #v.drag{{cursor:grabbing}} #stage{{transform-origin:0 0;display:inline-block}}
</style></head><body>
<div id="bar"><button id="zi">+</button><button id="zo">−</button><button id="fit">fit</button><button id="one">1:1</button></div>
<div id="v"><div id="stage">{svg}</div></div>
<div id="hint">drag to pan · wheel to zoom · hover a node or edge for exact numeric shapes</div>
<script>
(()=>{{const v=document.getElementById('v'),s=document.getElementById('stage'),svg=s.querySelector('svg');
let k=1,x=0,y=0;const W=svg.width.baseVal.value,H=svg.height.baseVal.value;
const ap=()=>s.style.transform=`translate(${{x}}px,${{y}}px) scale(${{k}})`;
const fit=()=>{{k=Math.min(v.clientWidth/W,v.clientHeight/H,1);x=(v.clientWidth-W*k)/2;y=Math.max(0,(v.clientHeight-H*k)/2);ap()}};
const zoom=(f,cx=v.clientWidth/2,cy=v.clientHeight/2)=>{{const nk=Math.min(8,Math.max(.05,k*f));x=cx-(cx-x)*nk/k;y=cy-(cy-y)*nk/k;k=nk;ap()}};
v.addEventListener('wheel',e=>{{e.preventDefault();zoom(e.deltaY<0?1.12:1/1.12,e.clientX,e.clientY)}},{{passive:false}});
let d=null;v.addEventListener('pointerdown',e=>{{d={{x:e.clientX-x,y:e.clientY-y}};v.classList.add('drag');v.setPointerCapture(e.pointerId)}});
v.addEventListener('pointermove',e=>{{if(d){{x=e.clientX-d.x;y=e.clientY-d.y;ap()}}}});
v.addEventListener('pointerup',()=>{{d=null;v.classList.remove('drag')}});
document.getElementById('zi').onclick=()=>zoom(1.25);document.getElementById('zo').onclick=()=>zoom(.8);
document.getElementById('fit').onclick=fit;document.getElementById('one').onclick=()=>{{k=1;x=10;y=10;ap()}};
addEventListener('resize',fit);fit();}})();
</script></body></html>"""


def to_html(svg, title, theme):
    T = THEMES[theme]
    return HTML.format(svg=svg, title=esc(title), bg=T["bg"], fg=T["fg"], dim=T["dim"])


def _svg_size(svg_path):
    head = open(svg_path, encoding="utf-8").read(2000)
    w = re.search(r'<svg[^>]*?\swidth="([\d.]+)', head)
    h = re.search(r'<svg[^>]*?\sheight="([\d.]+)', head)
    return (float(w.group(1)), float(h.group(1))) if w and h else (None, None)


def to_png(svg_path, png_path, scale=2.0, width=None, max_side=16000):
    """Best-effort rasterisation. `width` pins the PNG width in px; otherwise `scale` is used, reduced
    so that neither side exceeds `max_side` px. Returns (tool, (px_w, px_h)) or None."""
    import shutil
    import subprocess
    sw, sh = _svg_size(svg_path)
    if sw and sh:
        if width:
            scale = width / sw
        scale = min(scale, max_side / max(sw, sh))
    size = (round(sw * scale), round(sh * scale)) if sw and sh else None
    try:
        import cairosvg
        cairosvg.svg2png(url=svg_path, write_to=png_path, scale=scale)
        return "cairosvg", size
    except Exception:
        pass
    for exe, cmd in (("rsvg-convert", ["rsvg-convert", "-z", str(scale), "-o", png_path, svg_path]),
                     ("inkscape", ["inkscape", svg_path, "--export-type=png", f"--export-filename={png_path}",
                                   f"--export-dpi={96 * scale}"])):
        if shutil.which(exe):
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                return exe, size
    return None
