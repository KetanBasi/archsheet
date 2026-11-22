"""archsheet.graph — turn the raw trace into a displayable hierarchy.

  * Symbols      : concrete dims  ->  symbolic dims (b, s, d, 3d, b·s, h·dh, ...)
  * collapse     : identical repeated layers  ->  one drawn layer with a ×N badge
                   (also finds periodic patterns such as 5 local : 1 global)
  * lift         : tensor edges between leaf nodes -> edges between siblings of the
                   lowest common container, plus input/output "tags" on every
                   container boundary that is crossed
"""
from __future__ import annotations

import itertools
import re
from collections import OrderedDict, defaultdict

from .trace import Edge, Meta, Node

DTYPE_SHORT = {
    "float32": "f32", "float16": "f16", "bfloat16": "bf16", "float64": "f64",
    "int64": "i64", "int32": "i32", "int16": "i16", "int8": "i8", "uint8": "u8",
    "bool": "bool", "float8_e4m3fn": "f8", "float8_e5m2": "f8",
}


# ---------------------------------------------------------------------------
# symbolic shapes
# ---------------------------------------------------------------------------
class Symbols:
    def __init__(self, table=None, main_dtype="float32", dtype_mode="auto"):
        self.vals, self.desc = OrderedDict(), {}
        for k, v in (table or {}).items():
            if isinstance(v, (tuple, list)):
                v, d = v[0], (v[1] if len(v) > 1 else "")
            else:
                d = ""
            self.vals[k] = int(v)
            self.desc[k] = d
        self.main_dtype, self.dtype_mode = main_dtype, dtype_mode
        self.warnings = []
        self._check_collisions()
        self.exprs = self._build()

    def _check_collisions(self):
        seen = {}
        for k, v in self.vals.items():
            if v in seen:
                self.warnings.append(
                    f"symbols '{seen[v]}' and '{k}' have the same value {v}: shapes will be ambiguous. "
                    f"Pick different example sizes in the spec (use primes unlikely to appear elsewhere).")
            seen.setdefault(v, k)

    def _build(self):
        ex = {}

        def put(val, cost, text, minval=0):
            # derived names for small numbers (3 channels = n/3, 10 classes = 2b) are usually
            # coincidences: keep those literal. Halving is the one common exception.
            if val > 1 and val >= minval and (val not in ex or cost < ex[val][0]):
                ex[val] = (cost, text)

        names = list(self.vals)
        for n in names:
            put(self.vals[n], 1, n)
        for a, b in itertools.combinations_with_replacement(names, 2):
            put(self.vals[a] * self.vals[b], 2.5, f"{a}·{b}", 4)
        for n in names:
            for c in range(2, 9):
                put(c * self.vals[n], 3, f"{c}{n}", 16)
                if self.vals[n] % c == 0:
                    put(self.vals[n] // c, 2.8 + 0.02 * c, f"{n}/{c}", 8 if c == 2 else 16)
        for a, b in itertools.permutations(names, 2):
            if self.vals[b] > 1 and self.vals[a] % self.vals[b] == 0:
                put(self.vals[a] // self.vals[b], 3.5, f"{a}/{b}", 4)
        for n in names:
            put(self.vals[n] + 1, 3.6, f"{n}+1", 16)
            put(self.vals[n] - 1, 3.6, f"{n}-1", 16)
        for a, b in itertools.combinations(names, 2):
            put(self.vals[a] + self.vals[b], 3.2, f"{a}+{b}", 16)
        for a, b, c in itertools.combinations_with_replacement(names, 3):
            put(self.vals[a] * self.vals[b] * self.vals[c], 4, f"{a}·{b}·{c}", 16)
        return ex

    def dim(self, d):
        if d == 1:
            return "1"
        return self.exprs[d][1] if d in self.exprs else str(d)

    def shape(self, m: Meta, dtype=True, cat=None):
        """cat=(dim, parts): show the concatenated dim as (a; b), e.g. [b, h, (s; m), dh]."""
        dims = [self.dim(d) for d in m.shape]
        if cat and cat[0] < len(dims) and sum(cat[1]) == m.shape[cat[0]] and len(cat[1]) > 1:
            dims[cat[0]] = "(" + "; ".join(self.dim(p) for p in cat[1]) + ")"
        s = "[" + ", ".join(dims) + "]"
        if dtype and self._show_dtype(m.dtype):
            s += " " + DTYPE_SHORT.get(m.dtype, m.dtype)
        return s

    def concrete(self, m: Meta):
        return "(" + ", ".join(str(d) for d in m.shape) + (",)" if len(m.shape) == 1 else ")") + " " + m.dtype

    def _show_dtype(self, dt):
        if self.dtype_mode == "none":
            return False
        if self.dtype_mode == "all":
            return True
        return not dt.startswith(("float", "bfloat")) or dt != self.main_dtype

    def legend(self):
        return [(k, v, self.desc.get(k, "")) for k, v in self.vals.items()]


def human(n):
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}".rstrip("0").rstrip(".") + unit
    return str(n)


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------
class Seg:
    """An edge drawn inside one container (between two of its items)."""
    __slots__ = ("src", "dst", "mfrom", "mto", "ops", "ann", "lines", "label", "src_tag", "dst_tag", "glines", "alines", "fid", "carry")

    def __init__(self, src, dst, mfrom, mto, ops, ann, label=""):
        self.src, self.dst, self.mfrom, self.mto, self.ops, self.ann = src, dst, mfrom, mto, ops, ann
        self.lines, self.label = [], label
        self.glines, self.alines = [], []
        self.carry = False                        # this segment carries the flow's name label
        self.fid = None                           # identity of the tensor flow (producer, shape, label)
        self.src_tag = self.dst_tag = None      # inner boundary tags this edge lines up with

    def key(self):
        return (self.src.id, self.dst.id, self.mfrom, self.mto, self.ops, self.ann, self.label, self.fid)


def friendly_attr(k, v):
    """Turn an attribute that differs between layer variants into reader-friendly wording."""
    kl = k.lower()
    if "window" in kl:
        return "global attention" if v in (0, None, False, -1) else f"sliding window {v}"
    if "sliding" in kl and isinstance(v, bool):
        return "sliding-window attention" if v else "global attention"
    if isinstance(v, str):
        lv = v.lower()
        if "full" in lv or "global" in lv:
            return "global attention"
        if "sliding" in lv or "local" in lv:
            return "sliding-window attention"
    return f"{k}={v}"


_IDX = re.compile(r"^(.*)\.(\d+)$")


class Graph:
    def __init__(self, tracer, symbols: Symbols, max_depth=4, collapse=True, unwrap=(), node_ann=True):
        self.scene = tracer.scene
        self.root = tracer.root_node
        self.total_params = tracer.total_params
        self.sym = symbols
        self.max_depth = max_depth
        self.unwrap = set(unwrap)
        self._label_done = set()
        self.warnings = list(symbols.warnings)
        self.alias, self.member = {}, {}
        self._sig_cache = {}
        self._gid = itertools.count()
        seen, self.edges = set(), []
        for e in tracer.edges:
            if e.key() not in seen:
                seen.add(e.key())
                self.edges.append(e)
        self._prepass(self.scene)
        self._dedupe_outputs()
        self._build_display(self.scene, collapse)
        self._assign(self.scene, 0)
        self._mark(self.scene, None)
        self.segments = defaultdict(list)
        self._dropped = self._acyclic()
        self._lift()
        if node_ann:
            self._attach_annotations()

    # ---------------------------------------------------------- signatures
    def sig(self, n):
        s = self._sig_cache.get(n.id)
        if s is None:
            s = (n.kind, n.type, n.extra, tuple(sorted(n.attrs.items())), tuple(n.in_metas),
                 tuple(n.out_metas), tuple(self.sig(c) for c in n.children))
            self._sig_cache[n.id] = s
        return s

    def _flat_attrs(self, n, prefix=""):
        out = {(prefix, k): (n, v) for k, v in n.attrs.items()}
        for c in n.children:
            out.update(self._flat_attrs(c, f"{prefix}{c.name}."))
        return out

    # ---------------------------------------------------------- repeat collapse
    # ---------------------------------------------------------- one module called several times
    def _merge_into(self, rep, other):
        """`other` is a later call of the same module as `rep`: draw it once. Sub-calls are matched by path (and by
        order among equal paths); a sub-call that only `other` made becomes a child of `rep`."""
        self.alias[other.id] = rep
        pool = defaultdict(list)
        for c in rep.children:
            pool[(c.kind, c.path or c.name)].append(c)
        seen = defaultdict(int)
        for c in list(other.children):
            key = (c.kind, c.path or c.name)
            i = seen[key]
            seen[key] += 1
            if i < len(pool[key]):
                self._merge_into(pool[key][i], c)
            else:
                c.parent = rep
                rep.children.append(c)
                pool[key].append(c)

    def _prepass(self, node):
        """Walk the whole call tree and draw every module that was called several times once."""
        if node.kind not in ("scene", "module", "phase"):
            return
        node.children = self._merge_calls(node)
        for c in node.children:
            self._prepass(c)

    def _dedupe_outputs(self):
        """Result nodes of calls that were merged into one box: the same producer, name and shape is one output."""
        prod = defaultdict(set)
        for e in self.edges:
            if e.dst.kind == "output":
                prod[e.dst.id].add((self.resolve(e.src).id, e.mfrom))

        def walk(n):
            seen, keep = {}, []
            for c in n.children:
                if c.kind == "output":
                    k = (c.name, tuple(c.in_metas), tuple(sorted(prod.get(c.id, ()))))
                    if k in seen and prod.get(c.id):
                        self.alias[c.id] = seen[k]
                        continue
                    seen[k] = c
                keep.append(c)
            n.children = keep
            for c in keep:
                if c.kind in ("module", "phase"):
                    walk(c)
        walk(self.scene)

    def _merge_calls(self, node):
        """Children of `node` that are calls of one module (same path): keep the first, alias the others."""
        first, calls, merged = {}, defaultdict(list), {}
        out = []
        for c in node.children:
            if c.kind != "module" or not c.path:
                out.append(c)
                continue
            rep = first.setdefault((c.path, c.type), c)
            if rep is c:
                out.append(c)
            else:
                calls[rep.id].append(c)
        for c in list(out):
            # a block passed through again (chunks, weight sharing) is drawn once; a small module used 2x in a
            # pipeline (pre/post mixing around attention and ffn) keeps its two boxes, 3+ calls are noise
            if c.id in calls and (not c.leaf or len(calls[c.id]) >= 2):
                merged[c.id] = 1 + len(calls[c.id])
                for o in calls.pop(c.id):
                    self._merge_into(c, o)
        for later in calls.values():
            out.extend(later)
        out.sort(key=lambda c: c.seq)
        for c in out:
            n = merged.get(c.id)
            if n:
                c.repeat = n
                c.note = " · ".join([x for x in (c.note, f"{n} calls") if x])
        return out

    def _acyclic(self):
        """Edges that are not drawn because they would close a loop in the drawn graph: a module that is called
        several times (or a layer drawn once for N) feeds its own next call. The box says `×N`; only the first
        call's wiring is kept. Edges are admitted in execution order, so a later call's input that is computed
        from the earlier calls is the one that goes."""
        adj = defaultdict(set)
        pending = []
        for e in self.edges:
            if self._cross_iteration(e.src, e.dst):
                continue
            u, v = self.resolve(e.src), self.resolve(e.dst)
            if u is v:
                continue
            if u is e.src and v is e.dst:
                adj[u.id].add(v.id)
            else:
                pending.append((e, u, v))
        drop = set()
        for e, u, v in pending:
            seen, todo = {v.id}, [v.id]
            while todo and u.id not in seen:
                for j in adj.get(todo.pop(), ()):
                    if j not in seen:
                        seen.add(j)
                        todo.append(j)
            if u.id in seen:
                drop.add(id(e))
            else:
                adj[u.id].add(v.id)
        return drop

    def _display_children(self, node, collapse):
        groups = OrderedDict()
        kids = node.children
        if collapse:
            for c in kids:
                m = _IDX.match(c.path) if c.kind == "module" else None
                if m and c.name.isdigit():
                    groups.setdefault(m.group(1), []).append(c)
        in_group = {c.id: k for k, ms in groups.items() if len(ms) > 1 for c in ms}
        dch, done = [], set()
        for c in kids:
            k = in_group.get(c.id)
            if k is None:
                dch.append(c)
            elif k not in done:
                done.add(k)
                dch.extend(self._collapse_group(node, k, groups[k]))
        out = []
        for c in dch:                         # dissolve wrapper modules (--unwrap)
            if c.kind == "module" and c.type in self.unwrap and not c.leaf:
                out.extend(self._display_children(c, collapse))
            else:
                out.append(c)
        return out

    def _build_display(self, node, collapse):
        if node.kind not in ("scene", "module", "phase"):
            node.dchildren = []
            return
        node.dchildren = self._display_children(node, collapse)
        for c in node.dchildren:
            if c.kind == "group":
                for r in c.dchildren:
                    self._build_display(r, collapse)
            else:
                self._build_display(c, collapse)

    @staticmethod
    def _segments(sigs):
        """Cheapest way to draw a layer list: plain runs of identical layers, and repeated multi-layer
        patterns (period p, R >= 2 times). Cost = boxes drawn; ties prefer patterns that start later, so
        `F R R R | (I R R R) x4` wins over `F | (R R R I) x4 | R R R`."""
        n = len(sigs)
        runs_in = lambda i, p: 1 + sum(sigs[i + t] != sigs[i + t - 1] for t in range(1, p))
        best = [None] * (n + 1)
        best[0] = (0, 0, None)
        for i in range(n):
            if best[i] is None:
                continue
            cost, tie, _ = best[i]

            def relax(j, c, t, seg):
                if best[j] is None or (c, t) < best[j][:2]:
                    best[j] = (c, t, (i, seg))
            j = i + 1
            while j <= n:
                relax(j, cost + 1, tie, ("run", 1))
                if j == n or sigs[j] != sigs[i]:
                    break
                j += 1
            if n <= 300:
                for p in range(2, (n - i) // 2 + 1):
                    r = 1
                    while i + (r + 1) * p <= n and sigs[i + r * p:i + (r + 1) * p] == sigs[i:i + p]:
                        r += 1
                    if r >= 2 and runs_in(i, p) >= 2:
                        for rr in range(2, r + 1):
                            relax(i + rr * p, cost + runs_in(i, p) + 1, tie - i, ("pat", p))
        segs, j = [], n
        while j > 0:
            i, (kind, p) = best[j][2]
            segs.append((i, j, p if kind == "pat" else 1))
            j = i
        return segs[::-1]

    def _collapse_group(self, parent, prefix, members):
        sigs = [self.sig(m) for m in members]
        short = prefix.rsplit(".", 1)[-1]
        out = []
        for lo0, hi0, p in self._segments(sigs):
            out.extend(self._emit_segment(parent, prefix, short, members[lo0:hi0], sigs[lo0:hi0], p, lo0))
        self._variant_notes([m for m in out if m.kind != "group"] +
                            [r for g in out if g.kind == "group" for r in g.dchildren])
        return out

    def _emit_segment(self, parent, prefix, short, members, sigs, p, off):
        n = len(members)
        R = n // p
        # runs of equal consecutive layers inside the first period
        runs, run_of_pos = [], []
        for i in range(p):
            if runs and sigs[i] == sigs[runs[-1][0]]:
                runs[-1].append(i)
            else:
                runs.append([i])
            run_of_pos.append(len(runs) - 1)
        reps = [members[r[0]] for r in runs]
        gid = next(self._gid)
        for i, m in enumerate(members):
            ridx = run_of_pos[i % p]
            self.member[m.id] = (gid, i, ridx)
            if m is not reps[ridx]:
                self._alias_subtree(m, reps[ridx])
        for r, rep in zip(runs, reps):
            rep.repeat = len(r) * (R if p == 1 else 1)
            lo, hi = off + r[0], off + r[-1]
            if p == 1:
                lo, hi = off, off + n - 1
            rep.name = f"{short}[{lo}]" if lo == hi else f"{short}[{lo}–{hi}]"
        if p == 1 or R == 1:               # nothing repeats as a block: no wrapper needed
            return reps
        g = Node("group", short, "pattern", prefix)
        g.seq = members[0].seq
        g.parent = parent
        g.repeat = R
        g.params = sum(m.params for m in members)
        g.in_metas, g.out_metas = members[0].in_metas, members[-1].out_metas
        g.dchildren = reps
        g.note = f"{n} layers = {R} × pattern of {p}"
        return [g]

    def _alias_subtree(self, h, r):
        self.alias[h.id] = r
        for hc, rc in zip(h.children, r.children):
            self._alias_subtree(hc, rc)

    def _variant_notes(self, reps):
        """Layer variants (e.g. local vs global attention): write each differing attribute on the
        node that actually owns it (self_attn, rope, ...) instead of on the whole layer."""
        if len(reps) < 2 or len({r.type for r in reps}) != 1:
            return
        flats = [self._flat_attrs(r) for r in reps]
        for key in sorted(set().union(*[set(f) for f in flats])):
            if len({repr(f.get(key, (None, None))[1]) for f in flats}) <= 1:
                continue
            for f in flats:
                if key in f:
                    node, v = f[key]
                    parts = [x for x in node.note.split(" · ") if x]
                    txt = friendly_attr(key[1], v)
                    if txt not in parts and len(parts) < 3:
                        parts.append(txt)
                    node.note = " · ".join(parts)

    def resolve(self, n):
        while n.id in self.alias:
            n = self.alias[n.id]
        return n

    # ---------------------------------------------------------- display tree
    def _assign(self, n, depth):
        n.depth = depth
        n.expanded = (n.kind in ("scene", "group", "phase")) or (
            n.kind == "module" and not n.leaf and bool(n.dchildren) and depth < self.max_depth)
        cd = depth if n.kind in ("scene", "group") else depth + 1      # a phase counts as a level, like the root module
        for c in n.dchildren or []:
            c.dparent = n
            self._assign(c, cd)

    def _mark(self, n, owner):
        n.disp = owner or n
        if owner is None and not n.expanded:
            owner = n
        for c in n.dchildren or []:
            self._mark(c, owner)

    # ---------------------------------------------------------- edge lifting
    def _memberships(self, n):
        out = {}
        while n is not None:
            if n.id in self.member:
                gid, idx, rank = self.member[n.id]
                out[gid] = (idx, rank)
            n = n.parent
        return out

    def _cross_iteration(self, u, v):
        mu, mv = self._memberships(u), self._memberships(v)
        for gid in set(mu) & set(mv):
            (iu, ru), (iv, rv) = mu[gid], mv[gid]
            if iu != iv and not ru < rv:
                return True
        return False

    @staticmethod
    def _chain(n):
        out = []
        while n is not None:
            out.append(n)
            n = n.dparent
        return out

    def _tag(self, container, side, key, m, caption):
        d = container.tags_in if side == "in" else container.tags_out
        t = d.get(key)
        if t is None:
            t = Node("tag", side, side)
            t.out_metas = [m]
            t.note = caption
            t.seq = container.seq - 0.5 + 0.001 * len(d) if side == "in" else container.seq + 1e9 + len(d)
            t.dparent = container
            d[key] = t
        return t

    def _add(self, container, seg):
        for s in self.segments[container.id]:
            if s.key() == seg.key():
                return
        self.segments[container.id].append(seg)

    # shapes of a flow at container boundaries: every hidden reshape remembers where it ran
    def _inside(self, c, X):
        while c is not None:
            if self.resolve(c) is X:
                return True
            c = c.parent
        return False

    def _leaving(self, e, X):
        m = e.mfrom
        for c, mm, _ in e.trail:
            if self._inside(c, X):
                m = mm
        return m

    def _entering(self, e, X):
        m = e.mfrom
        for c, mm, _ in e.trail:
            if self._inside(c, X):
                break
            m = mm
        return m

    def _seg(self, e, a, b, mfrom, mto, ctr):
        ops = tuple(name for c, _, name in e.trail if self.resolve(c) is ctr)
        sg = Seg(a, b, mfrom, mto, ops, bool(ops) or mto != mfrom, e.label)
        sg.fid = (self.resolve(e.src).id, e.mfrom, e.label)
        return sg

    def _lift(self):
        lat_use = {}                # boundary tag -> True while every flow through it runs phase to phase

        def mark(tag, cross):
            u = lat_use.setdefault(id(tag), [tag, True])
            u[1] = u[1] and cross
        for e in self.edges:
            if id(e) in self._dropped or self._cross_iteration(e.src, e.dst):
                continue
            u, v = self.resolve(e.src), self.resolve(e.dst)
            if u.disp is None or v.disp is None:
                self.warnings.append(f"edge {u} -> {v} touches a node that is not displayed")
                continue
            du, dv = u.disp, v.disp
            if du is dv:
                continue
            pu, pv = self._chain(du), self._chain(dv)
            ids_v = {n.id for n in pv}
            lu = next((i for i, n in enumerate(pu) if n.id in ids_v), None)
            if lu is None:
                continue
            lca = pu[lu]
            lv = next(i for i, n in enumerate(pv) if n is lca)
            if lu == 0 or lv == 0:
                continue
            cross_phase = lca.kind == "scene" and pu[lu - 1].kind == "phase" and pv[lv - 1].kind == "phase"
            # tensor leaves containers pu[1:lu] ...
            for k in range(1, lu):
                X = pu[k]
                tm = self._leaving(e, X)
                tag = self._tag(X, "out", (u.id, e.mfrom), tm, "")
                mark(tag, cross_phase)
                sg = self._seg(e, pu[k - 1], tag, self._leaving(e, pu[k - 1]), tm, X)
                if k > 1:                                   # nested container: line up with its own port
                    sg.src_tag = pu[k - 1].tags_out.get((u.id, e.mfrom))
                self._add(X, sg)
            # ... crosses the lowest common container ...
            top = self._seg(e, pu[lu - 1], pv[lv - 1], self._leaving(e, pu[lu - 1]),
                            self._entering(e, pv[lv - 1]), lca)
            if lu > 1:
                top.src_tag = pu[lu - 1].tags_out.get((u.id, e.mfrom))
            # ... and enters containers pv[lv-1 .. 1]
            for k in range(1, lv):
                X = pv[k]
                tm = self._entering(e, X)
                tag = self._tag(X, "in", (u.id, e.mfrom), tm, pu[lu - 1].name)
                mark(tag, cross_phase)
                if cross_phase:                              # arrives from the left: sort left of the main input
                    tag.bias = -0.3
                sg = self._seg(e, tag, pv[k - 1], tm, self._entering(e, pv[k - 1]), X)
                if k > 1:
                    sg.dst_tag = pv[k - 1].tags_in.get((u.id, e.mfrom))
                self._add(X, sg)
            if lv > 1:
                top.dst_tag = pv[lv - 1].tags_in.get((u.id, e.mfrom))
            cu, cv = pu[lu - 1], pv[lv - 1]
            if e.label and top.fid not in self._label_done:      # a flow's name is shown exactly once
                top.carry = True
                self._label_done.add(top.fid)
            if lca.kind == "scene" and cu.kind == "phase" and cv.kind == "phase":
                if top.src_tag is not None:
                    top.src_tag.side = "R"      # leaves the earlier phase through its right border
                if top.dst_tag is not None:
                    top.dst_tag.side = "L"      # enters the later phase through its left border
            self._add(lca, top)
        # a tensor that only travels to another phase leaves / enters its containers through the side
        # border (at the height of its producer / consumer) instead of looping over the container's top / bottom
        for tag, only_cross in lat_use.values():
            if only_cross:
                tag.side = "R" if tag.name == "out" else "L"

    # ---------------------------------------------------------- hidden-op annotations live on nodes
    _GLYPH = {"add", "mul", "sub", "div"}
    _RENAME = {"scaled_dot_product_attention": "SDPA", "sdpa": "SDPA", "getitem": "idx",
               "multi_head_attention_forward": "mha"}

    def _ann_text(self, s):
        ops = []
        for o in s.ops:
            o = self._RENAME.get(o, o)
            if o in ("contiguous", "detach", "clone") or (ops and ops[-1] == o):
                continue
            ops.append(o)
        from .trace import SHAPE_OPS
        comp = [o for o in s.ops if o not in SHAPE_OPS]
        if s.mto.shape != s.mfrom.shape:
            return "↳ " + ",".join(ops[:3]) + ("…" if len(ops) > 3 else "") + " " + self.sym.shape(s.mto, dtype=False)
        if s.mto.dtype != s.mfrom.dtype:
            return f"↳ cast {s.mto.dtype}"
        if comp:
            return "↳ " + ",".join(dict.fromkeys(comp))
        return None

    def _attach_annotations(self):
        """Shapes/reshapes are written under the producing node's `out` row (`↳ view,transpose [..]`),
        not on the edges. If the producer is not a card (invisible port, + / × circle) the note goes
        under the consumer's `in` rows."""
        card = lambda n: (n.kind in ("module", "input") or (n.kind == "op" and n.name not in self._GLYPH))
        for segs in self.segments.values():
            for s in segs:
                d = s.dst
                if d.kind == "module" and d.in_metas and not any(m.shape == s.mto.shape for m in d.in_metas) \
                        and not any(m.shape == s.mto.shape for m in d.in_ext):
                    d.in_ext.append(s.mto)         # an arrow with no `in` row: say what it carries
                if not s.ann:
                    continue
                txt = self._ann_text(s)
                if not txt:
                    continue
                if card(s.src):
                    tgt = s.src.out_ann
                elif s.dst.kind in ("module", "op", "output") and not (s.dst.kind == "op" and s.dst.name in self._GLYPH):
                    tgt = s.dst.in_ann
                else:
                    continue
                if txt not in tgt:
                    tgt.append(txt)

    # ---------------------------------------------------------- outline (text)
    def outline(self, max_lines=300):
        s = self.sym
        lines = []

        def io(n):
            ins = ", ".join(s.shape(m) for m in n.in_metas[:3]) + (" …" if len(n.in_metas) > 3 else "")
            outs = ", ".join(s.shape(m) for m in n.out_metas[:3]) + (" …" if len(n.out_metas) > 3 else "")
            return f"{ins} → {outs}"

        def walk(n, prefix, last):
            if len(lines) >= max_lines:
                return
            if n.kind == "scene":
                kids = n.dchildren
                for i, c in enumerate(kids):
                    walk(c, "", i == len(kids) - 1)
                return
            branch = "" if prefix == "" and n.kind in ("input", "output") else ("└─ " if last else "├─ ")
            if n.kind == "phase":
                text = f"▢ phase {n.name}"
            elif n.kind == "input":
                text = f"[in]  {n.name}  {s.shape(n.out_metas[0])}"
            elif n.kind == "output":
                text = f"[out] {n.name}  {s.shape(n.in_metas[0])}"
            elif n.kind == "op":
                text = f"· {n.name}  {io(n)}"
            elif n.kind == "group":
                text = f"{n.name} ×{n.repeat} ({n.note})  {io(n)}"
            else:
                rep = f" ×{n.repeat}" if n.repeat > 1 else ""
                note = f" [{n.note}]" if n.note else ""
                p = f"  {human(n.params)} params" if n.params and n.depth <= 1 else ""
                text = f"{n.name}: {n.type}{rep}{note}  {io(n)}{p}"
            lines.append(prefix + branch + text)
            if n.expanded:
                kids = n.dchildren
                child_prefix = prefix + ("" if not branch else ("   " if last else "│  "))
                for i, c in enumerate(kids):
                    walk(c, child_prefix, i == len(kids) - 1)

        walk(self.scene, "", True)
        if len(lines) >= max_lines:
            lines.append("… (outline truncated)")
        return "\n".join(lines)
