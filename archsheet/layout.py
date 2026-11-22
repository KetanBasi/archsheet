"""archsheet.layout — layered layout of ONE container (children + invisible boundary ports).

Pipeline
  1. layering      longest path (data flows upward); inputs pinned to the bottom, outputs to the
                   top; nodes whose inputs are all graph inputs are pulled up next to their first
                   consumer (unless a sibling reads the same tensor in the lower row)
  2. dummy chains  every tensor that skips layers gets ONE chain of zero-width dummy nodes that all
                   its consumers share => a single straight trunk that only splits in the last gap
                   before each consumer ("branch late"). Residual `+` skips are NOT in the grid: they
                   become lanes (step 5) so they can never move anything.
  3. ordering      barycenter sweeps (a trunk that continues past a consumer sorts to its right)
  4. x placement   ports: the CENTRAL neighbour is the main flow (K among Q K V; mid-point for an even
                   count => symmetric T). Alignment passes use strong links only (central at both ends),
                   then "spines" put every strong chain on one exact vertical line.
  5. tracks        one horizontal track per tensor per gap (a bus), labels sit on the tracks; residual
                   lanes hug the right-most content of the rows they pass; every track keeps
                   TOP_CLEAR / BOT_CLEAR away from the nodes so corners never touch arrowheads
  6. routing       polylines ("pieces"); residuals into `+` enter from the right side
Phases (dashed regions of a multi-entry model) are placed side by side; flows between them leave and
enter through side borders; a flow whose two ends are at different heights runs down (or up) the channel
between the two phases, so towers of different height sit bottom-aligned with no gap under the shorter one.
"""
from __future__ import annotations

from collections import defaultdict


TOP_CLEAR = 22        # room between a consumer's input edge and the first track: arrowhead + visible shaft
BOT_CLEAR = 10        # room between the last track and the producer's output edge
LANE_GAP = 16         # residual lane: distance to the right-most content of the rows it passes
LANE_STEP = 12        # nested lanes
LANE_MIN = 30         # lane to the `+` it enters (arrowhead + shaft)
HIDE_SKIPS = False    # drop residual skip lines altogether (archsheet --hide-skips); never changes the grid


class Dummy:
    _n = 0

    def __init__(self, layer, seq):
        Dummy._n += 1
        self.id = -Dummy._n
        self.kind = "dummy"
        self.layer, self.seq = layer, seq
        self.w = self.h = 0.0
        self.lx = self.ly = 0.0
        self.after = None
        self.beside = False
        self.far = False


class Piece:
    def __init__(self, frm, to, gk):
        self.frm, self.to, self.gk = frm, to, gk
        self.segs, self.finals = [], []
        self.side = False
        self.xside = None        # 'out' / 'in': container leaves / enters through its side border: one straight line
        self.lane = False        # residual skip: routed after placement in a lane right of the content
        self.lane_x = 0.0
        self.hidden = False      # --hide-skips: keeps its lane / track reserved but is not drawn


def _pav(nodes, des, wt, gap):
    """Order-preserving placement: minimise sum wt*(lx-des)^2 s.t. lx[i+1] >= lx[i]+w[i]+gap."""
    S = [0.0]
    for a, b in zip(nodes, nodes[1:]):
        S.append(S[-1] + a.w + gap(a, b))
    blocks = []
    for d, s, w in zip(des, S, wt):
        blocks.append([(d - s) * w, w, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b2, b1 = blocks.pop(), blocks.pop()
            blocks.append([b1[0] + b2[0], b1[1] + b2[1], b1[2] + b2[2]])
    z = []
    for b in blocks:
        z += [b[0] / b[1]] * b[2]
    for v, zi, si in zip(nodes, z, S):
        v.lx = zi + si


def _channel_order(grp):
    """Flows from one phase to the next that end at a different height run down (or up) the channel between the
    two. Each gets its own vertical; the order of the verticals that has the fewest crossings wins (all flows
    going down: the one starting highest takes the outer vertical, going up: the one starting lowest)."""
    from itertools import permutations
    sx = grp[0].src.x + grp[0].src.w
    ex = grp[0].dst.x
    rows = [(g.src.y + g.src_tag.y, g.dst.y + g.dst_tag.y, g) for g in grp]
    bent = [r for r in rows if abs(r[0] - r[1]) > 0.5]
    out = {}
    if not bent:
        return out
    k = len(bent)
    xs = [sx + (ex - sx) * (i + 1) / (k + 1) for i in range(k)]

    def crossings(order):                       # order[i] = flow using the i-th vertical from the left
        hs, vs = [], []
        for x, (a, b, _g) in zip(xs, order):
            hs += [(a, sx, x), (b, x, ex)]       # horizontals: (y, x0, x1)
            vs.append((x, min(a, b), max(a, b)))
        c = 0
        for y, x0, x1 in hs:
            for x, y0, y1 in vs:
                if x0 < x < x1 and y0 < y < y1:
                    c += 1
        return c

    cand = permutations(bent) if k <= 6 else [sorted(bent, key=lambda r: r[0]), sorted(bent, key=lambda r: -r[0])]
    best = min(cand, key=lambda o: (crossings(o), sum(abs(a - b) * abs(i - (k - 1) / 2) for i, (a, b, _g) in enumerate(o))))
    for x, (_a, _b, g) in zip(xs, best):
        out[id(g)] = x
    return out


def layout_items(L, n, items, header_h, header_w, pad):
    G = L.g
    ids = {it.id for it in items}
    segs = [s for s in G.segments.get(n.id, []) if s.src.id in ids and s.dst.id in ids]
    phases_mode = n.kind == "scene" and any(it.kind == "phase" for it in items)
    lateral = lambda s: phases_mode and s.src.kind == "phase" and s.dst.kind == "phase"
    tin = {t.id for t in n.tags_in.values()}
    tout = {t.id for t in n.tags_out.values()}
    lat_in = {t.id for t in n.tags_in.values() if t.side == "L"}
    lat_out = {t.id for t in n.tags_out.values() if t.side == "R"}
    pinned = lat_in | lat_out
    glyph = L.glyph_ops
    order = sorted(items, key=lambda i: i.seq)
    for it in items:
        it.lx = it.ly = 0.0

    # ------------------------------------------------------------------ 1. layering
    vsegs, preds, succs = [], defaultdict(list), defaultdict(list)
    for s in segs:
        if lateral(s):
            continue
        if s.src.seq >= s.dst.seq:
            L.warnings.append(f"skipped back-edge {s.src} -> {s.dst} (cycle / reused module)")
            continue
        vsegs.append(s)
        preds[s.dst.id].append(s.src)
        succs[s.src.id].append(s.dst)
    layer = {}
    for it in order:
        if it.id in tout and it.id not in lat_out:
            continue
        if phases_mode and it.kind == "phase":
            layer[it.id] = 0
            continue
        layer[it.id] = max([layer[p.id] + 1 for p in preds[it.id] if p.id in layer] or [0])
    low = {it.id for it in order if it.kind == "input" or it.id in tin}
    free = set()
    for it in order:
        if it.kind in ("module", "op", "group") and all(p.id in low or p.id in free for p in preds[it.id]):
            free.add(it.id)
    for it in reversed(order):                       # pull slack nodes up next to their consumer
        if it.id in free and succs[it.id]:
            cand = [layer[d.id] for d in succs[it.id] if d.id in layer]
            if cand and min(cand) - 1 > layer[it.id]:
                # ... but never away from siblings that read the same tensor (q / k / v stay in one row)
                sibling_here = any(layer.get(o.id) == layer[it.id]
                                   for pr in preds[it.id] for o in succs[pr.id] if o is not it)
                if not sibling_here:
                    layer[it.id] = min(cand) - 1
    # a gate / scale (x * cos) whose inputs are all shared with other consumers and whose single consumer is
    # several rows up sits next to that consumer: the long edge becomes part of the shared input trunks
    # instead of a separate line that crosses everything in between (RoPE)
    for it in reversed(order):
        if it.kind == "op" and it.name in glyph and it.name != "add" and len(succs[it.id]) == 1 \
                and preds[it.id] and all(len(succs[q.id]) > 1 for q in preds[it.id]):
            c = layer.get(succs[it.id][0].id)
            if c is not None and c - 1 > layer[it.id]:
                layer[it.id] = c - 1
    body_max = max(layer.values(), default=0)
    tops = [it for it in order if it.kind == "output" or (it.id in tout and it.id not in lat_out)]
    top_layer = body_max + 1 if tops else body_max
    for it in tops:
        layer[it.id] = top_layer
    for tid in lat_in:                                 # side-entering ports sit just below their consumer
        t = next(i for i in items if i.id == tid)
        cons = [layer[d.id] for d in succs[tid] if d.id in layer]
        layer[tid] = max(0, min(cons) - 1) if cons else 0

    nl = lambda v: v.layer if v.kind == "dummy" else layer[v.id]

    n_in = defaultdict(int)
    for s in vsegs:
        n_in[s.dst.id] += 1
    for it in items:
        if it.kind == "op" and it.name not in glyph and n_in[it.id] > 1:
            it.w = max(it.w, 26 * n_in[it.id] + 14)

    # ------------------------------------------------------------------ 2. dummy chains
    layers = defaultdict(list)
    for it in order:
        if it.id not in pinned:
            layers[layer[it.id]].append(it)
    dummies, pieces, chain_of = {}, {}, {}

    def piece(frm, to, gk):
        k = (gk, frm.id, to.id)
        if k not in pieces:
            pieces[k] = Piece(frm, to, gk)
        return pieces[k]

    seg_ok = []

    def gk_of(s):
        return (s.src.id, s.fid, id(s.src_tag))

    def long_add(s):
        a, b = layer[s.src.id], layer[s.dst.id]
        return b - a > 1 and s.dst.kind == "op" and s.dst.name == "add" and s.dst.name in glyph

    # flows that also feed something else (a norm, a projection ...): for those a long edge into `+` is a
    # residual skip
    main_flows = {gk_of(s) for s in vsegs if layer[s.dst.id] > layer[s.src.id] and not long_add(s)}

    def is_side(s):     # only a residual skip into `+` runs beside the content; a computed value that merely
        return long_add(s) and gk_of(s) in main_flows      # skips a layer (rope: x*cos + rot(x)*sin) enters from below
    for s in vsegs:
        a, b = layer[s.src.id], layer[s.dst.id]
        if b <= a:
            L.warnings.append(f"skipped flat edge {s.src} -> {s.dst}")
            continue
        gk = gk_of(s)
        side = is_side(s)
        if side and s.dst.name == "add" and gk in main_flows:
            # residual skip: the stream also feeds a real consumer, so the skip is NOT part of the grid.
            # It is routed afterwards in a lane just right of the content => adding / removing a skip never
            # moves, widens or reorders anything else.
            p = piece(s.src, s.dst, gk)
            p.segs.append(s)
            p.finals.append(s)
            p.side = p.lane = True
            p.hidden = HIDE_SKIPS
            seg_ok.append(s)
            continue
        nodes = [s.src]
        for l in range(a + 1, b + (1 if side else 0)):
            if side and l == b:
                d = Dummy(l, s.src.seq + 0.5)
                d.after = s.dst
                layers[l].append(d)
            else:
                d = dummies.get((gk, l))
                if d is None:
                    d = dummies[(gk, l)] = Dummy(l, s.src.seq + 0.5)
                    d.chain = chain_of.setdefault(gk, object())
                    d.far = s.src.kind == "tag"             # started outside this container
                    layers[l].append(d)
            nodes.append(d)
        nodes.append(s.dst)
        for i, (u, v) in enumerate(zip(nodes, nodes[1:])):
            p = piece(u, v, gk)
            p.segs.append(s)
            if i == len(nodes) - 2:
                p.finals.append(s)
            if side and i == len(nodes) - 2:
                p.side = True
            if len(nodes) == 2:
                if v.id in lat_out and s.src_tag is not None and s.src_tag.side == "R":
                    p.side, p.xside = True, "out"
                elif u.id in lat_in and s.dst_tag is not None and s.dst_tag.side == "L":
                    p.side, p.xside = True, "in"
        seg_ok.append(s)
    # a trunk that keeps going past a real consumer of the SAME tensor (input_ids -> embed_tokens AND on to
    # masked_scatter) branches off to the right of that consumer; the consumer itself stays on the straight line
    cons_in_layer = defaultdict(dict)
    for s in seg_ok:
        cons_in_layer[gk_of(s)].setdefault(layer[s.dst.id], s.dst)
    for ls in list(layers.values()):
        for d in ls:
            if d.kind == "dummy" and getattr(d, "chain", None) is not None:
                for gk_, ch in chain_of.items():
                    if ch is d.chain and d.layer in cons_in_layer[gk_]:
                        d.beside = True
                        d.anchor = cons_in_layer[gk_][d.layer]
                        d.seq = cons_in_layer[gk_][d.layer].seq + 0.1
    K = max([layer.get(i.id, 0) for i in items] + list(layers.keys()) + [0])

    # group labels / annotations
    seen = set()
    grp_lines = {}
    for s in seg_ok:
        s.glines, s.alines = L.seg_lines(s, seen)
        gk = (s.src.id, s.fid, id(s.src_tag))
        if s.glines and gk not in grp_lines:
            grp_lines[gk] = s.glines

    ins_of, outs_of = defaultdict(list), defaultdict(list)
    ins_all, outs_all = defaultdict(list), defaultdict(list)     # incl. pieces ending/starting at side ports
    for p in pieces.values():
        if p.side:
            continue
        ins_all[p.to.id].append(p)
        outs_all[p.frm.id].append(p)
        if p.frm.id in pinned or p.to.id in pinned:
            continue
        ins_of[p.to.id].append(p)
        outs_of[p.frm.id].append(p)

    # ------------------------------------------------------------------ 3. ordering
    PH_GAP = 70
    if phases_mode:
        lat = [s for s in segs if lateral(s)]
        for s in lat:
            s.glines, s.alines = L.seg_lines(s, seen)
            s.glines = s.glines or []
        wl = max([len(t) * 6.3 for s in lat for t in s.glines + s.alines] + [0])
        PH_GAP = max(70, wl + 46)

    def gap(u, v):
        if u.kind == "phase" and v.kind == "phase":
            return PH_GAP
        if u.kind in ("dummy", "tag") and v.kind in ("dummy", "tag"):
            return 14
        if u.kind in ("dummy", "tag") or v.kind in ("dummy", "tag"):
            return 18
        return 28

    def fix_after(nodes):
        for d in [v for v in nodes if getattr(v, "after", None) is not None]:
            nodes.remove(d)
            nodes.insert(nodes.index(d.after) + 1, d)

    for l in layers:
        layers[l].sort(key=lambda v: v.seq + getattr(v, "bias", 0.0))
        fix_after(layers[l])

    def width_of(nodes):
        return sum(v.w for v in nodes) + sum(gap(a, b) for a, b in zip(nodes, nodes[1:]))

    def place_all():
        W = max([width_of(layers[l]) for l in layers] + [0])
        for l, nodes in layers.items():
            x = (W - width_of(nodes)) / 2
            for i, v in enumerate(nodes):
                v.lx = x
                x += v.w + (gap(v, nodes[i + 1]) if i + 1 < len(nodes) else 0)

    cx = lambda v: v.lx + v.w / 2

    def tgt_x(p):          # where a piece lands on its target (inside a container: at the inner port)
        tag = next((s.dst_tag for s in p.finals if s.dst is p.to and s.dst_tag), None)
        return p.to.lx + (min(max(tag.x, 0), p.to.w) if tag is not None else p.to.w / 2)

    def src_x(p):
        tag = next((s.src_tag for s in p.segs if s.src is p.frm and s.src_tag), None)
        return p.frm.lx + (min(max(tag.x, 0), p.frm.w) if tag is not None else p.frm.w / 2)

    place_all()
    for _ in range(4):
        for l in range(1, K + 1):
            layers[l].sort(key=lambda v: ((sum(src_x(p) for p in ins_of[v.id]) / len(ins_of[v.id]))
                                          if ins_of[v.id] else cx(v), v.beside if v.kind == "dummy" else 0))
            fix_after(layers[l])
            place_all()
        for l in range(K - 1, -1, -1):
            layers[l].sort(key=lambda v: ((sum(tgt_x(p) for p in outs_of[v.id]) / len(outs_of[v.id]))
                                          if outs_of[v.id] else cx(v), v.beside if v.kind == "dummy" else 0))
            fix_after(layers[l])
            place_all()

    # crossing minimisation: the barycentre sweeps above leave avoidable crossings behind. Sift every node to
    # the position in its row that crosses the fewest edges of the rows above and below, until nothing improves.
    def sift_rows():
        up, down = defaultdict(list), defaultdict(list)
        LEFT, RIGHT = object(), object()     # flows that enter / leave through a phase border start / end beyond the row
        for p in pieces.values():
            if p.side:
                continue
            if p.frm.id in lat_in and p.to.id not in pinned:
                down[id(p.to)].append(LEFT)
            elif p.to.id in lat_out and p.frm.id not in pinned:
                up[id(p.frm)].append(RIGHT)
            elif p.frm.id not in pinned and p.to.id not in pinned:
                up[id(p.frm)].append(p.to)
                down[id(p.to)].append(p.frm)
        pos = {id(LEFT): -1e6, id(RIGHT): 1e6}

        def reindex(nodes):
            for i, v in enumerate(nodes):
                pos[id(v)] = i + 0.0
        for ls in layers.values():
            reindex(ls)

        def units_of(nodes):
            """a consumer and the trunk dummies that must stay right beside it move together"""
            units = []
            for v in nodes:
                a = (getattr(v, "anchor", None) or getattr(v, "after", None)) if v.kind == "dummy" else None
                if a is not None and units and units[-1][0] is a or (a is not None and units and a in units[-1]):
                    units[-1].append(v)
                else:
                    units.append([v])
            return units

        def delta(left, right):      # crossings added by swapping two adjacent units (negative = fewer)
            t = 0
            for nb in (up, down):
                for x in left:
                    for y in right:
                        for a in nb[id(x)]:
                            for b in nb[id(y)]:
                                if a is not b:
                                    t += (pos[id(a)] < pos[id(b)]) - (pos[id(a)] > pos[id(b)])
            return t
        for _ in range(8):
            changed = False
            for l in sorted(layers):
                nodes = layers[l]
                if len(nodes) < 2 or any(v.kind == "phase" for v in nodes):
                    continue
                units = units_of(nodes)
                for u in list(units):
                    if u[0].kind == "dummy" and (getattr(u[0], "beside", False) or getattr(u[0], "after", None) is not None):
                        continue                                  # glued trunk that found no consumer next to it
                    i = units.index(u)
                    best, where, cum = 0, i, 0
                    for j in range(i - 1, -1, -1):
                        cum += delta(units[j], u)
                        if cum < best:
                            best, where = cum, j
                    cum = 0
                    for j in range(i + 1, len(units)):
                        cum += delta(u, units[j])
                        if cum < best:
                            best, where = cum, j
                    if where != i:
                        units.insert(where, units.pop(i))
                        nodes[:] = [v for un in units for v in un]
                        reindex(nodes)
                        changed = True
            if not changed:
                break
    sift_rows()
    place_all()

    # ------------------------------------------------------------------ 4. ports + x alignment
    out_off, in_off = {}, {}
    all_nodes = [v for l in layers for v in layers[l]]

    def is_real(o):
        """a real node (not a trunk dummy / side port): its position matters for alignment"""
        return not (o.kind == "dummy" or o.id in pinned)

    def real_in(p):
        """is this input a "peer"? A real producer, or a trunk that carries a real producer's tensor up
        (up_proj -> × passes beside GELU) counts like a real node. A trunk that merely arrives from outside the
        container (ext_k -> cat) does not: it joins beside the node that is on the stream."""
        return is_real(p.frm) or (p.frm.kind == "dummy" and not p.frm.far and p.frm.id not in pinned)

    def centre_of(real):
        """Index (maybe x.5) of the *central* element among n neighbours sorted left -> right.
        Odd n: the middle one (e.g. K among Q K V). Even n: the mid-point (a symmetric T junction), unless
        exactly one of the two middle ones is a real node and the other only a passing trunk."""
        n = len(real)
        if n % 2:
            return float(n // 2)
        lo, hi = n // 2 - 1, n // 2
        if real[lo] != real[hi]:
            return float(lo if real[lo] else hi)
        return (n - 1) / 2

    def assign_ports():
        for v in all_nodes:
            pp = 3 if v.w < 40 else min(10, v.w * 0.22)
            room = max(v.w / 2 - pp - (4 if v.w < 40 else 0), 0.0)
            spread = lambda n, c: 0.0 if n <= 1 else min(22.0, room / max(c, n - 1 - c, 0.5))
            groups = defaultdict(list)
            for p in outs_all[v.id]:
                groups[p.gk].append(p)
            gks = sorted(groups, key=lambda g: sum(tgt_x(p) for p in groups[g]) / len(groups[g]))
            c, step = 0.0, 0.0
            if gks:
                c = centre_of([any(is_real(p.to) for p in groups[g]) for g in gks])
                step = spread(len(gks), c)
            for j, gk in enumerate(gks):
                off = 0.0 if v.w == 0 else v.w / 2 + (j - c) * step
                tag = next((s.src_tag for p in groups[gk] for s in p.segs if s.src is v and s.src_tag), None)
                if tag is not None and v.w:
                    off = min(max(tag.x, pp), v.w - pp)
                out_off[(v.id, gk)] = off
            ins = sorted(ins_all[v.id], key=lambda p: src_x(p))
            c, step = 0.0, 0.0
            if ins:
                c = centre_of([real_in(p) for p in ins])
                step = spread(len(ins), c)
            for j, p in enumerate(ins):
                off = 0.0 if v.w == 0 else v.w / 2 + (j - c) * step
                tag = next((s.dst_tag for s in p.finals if s.dst is v and s.dst_tag), None)
                if tag is not None and v.w:
                    off = min(max(tag.x, pp), v.w - pp)
                in_off[(v.id, id(p))] = off

    assign_ports()
    def wgt(p):        # long straight trunks (dummy-dummy) matter most; a trunk must not drag its source around
        d = (p.frm.kind == "dummy") + (p.to.kind == "dummy")
        return 3.0 if d == 2 else (0.6 if d == 1 else 1.0)
    def relax():
        for it in range(40):
            for l in (range(0, K + 1) if it % 2 == 0 else range(K, -1, -1)):
                nodes = layers[l]
                if not nodes:
                    continue
                des, wt = [], []
                for v in nodes:
                    vals = []
                    for p in ins_of[v.id]:
                        vals.append((p.frm.lx + out_off[(p.frm.id, p.gk)] - in_off[(v.id, id(p))], wgt(p)))
                    for p in outs_of[v.id]:          # the outgoing edge is the main flow: weigh it more
                        vals.append((p.to.lx + in_off[(p.to.id, id(p))] - out_off[(v.id, p.gk)],
                                     wgt(p) * (2.0 if p.to.kind != "dummy" else 1.0)))
                    if vals:
                        sw = sum(w for _, w in vals)
                        des.append(sum(x * w for x, w in vals) / sw)
                        wt.append(sw)
                    else:
                        des.append(v.lx)
                        wt.append(0.05)
                _pav(nodes, des, wt, gap)

    relax()

    # straighten: a link is STRONG only when it can really become a straight line: it must be the central port
    # at both ends (e.g. K among Q K V - but not Q or V, which sit beside K and jog in the last gap).
    # Weak links must not drag nodes off the centre line (that is what made the stream wobble inside attn).
    def src_central(p):
        R = sorted([q for q in outs_of[p.frm.id] if q.gk == p.gk and is_real(q.to)], key=tgt_x)
        if not is_real(p.to):
            return not R                 # a pure trunk follows its source; a branch beside a consumer does not
        return len(R) % 2 == 1 and R[len(R) // 2] is p

    def dst_central(p):
        if not is_real(p.to):
            return True
        ins = sorted(ins_of[p.to.id], key=src_x)
        c = centre_of([real_in(q) for q in ins])
        return c == int(c) and ins[int(c)] is p

    def aim(v, use_ins):
        """(desired lx, strong?) of v from its inputs (use_ins) or its outputs, or None"""
        ps = ins_of[v.id] if use_ins else outs_of[v.id]
        if not ps:
            return None
        if use_ins:
            ps = sorted(ps, key=src_x)
            c = centre_of([real_in(q) for q in ps])
            pair = [ps[int(c)], ps[int(c + 0.5)]]
            tx = lambda q: q.frm.lx + out_off[(q.frm.id, q.gk)] - in_off[(v.id, id(q))]
            ok = all(src_central(q) for q in pair)
        else:                                       # a trunk that merely passes by must not pull
            ps = sorted([q for q in ps if is_real(q.to)] or ps, key=tgt_x)
            c = (len(ps) - 1) / 2
            pair = [ps[int(c)], ps[int(c + 0.5)]]
            tx = lambda q: q.to.lx + in_off[(q.to.id, id(q))] - out_off[(v.id, q.gk)]
            ok = all(dst_central(q) for q in pair)
        return sum(tx(q) for q in pair) / 2, ok

    def primary_pass(layer_range, use_ins):
        for l in layer_range:
            nodes = layers[l]
            des, wt = [], []
            for v in nodes:
                first = True if v.kind == "dummy" else use_ins   # a trunk dummy always follows its source:
                cands = [aim(v, first), aim(v, not first)]       # one straight line, jog only in the last gap
                cands = [c for c in cands if c is not None]
                strong = next((c for c in cands if c[1]), None)
                pick = strong or (cands[0] if cands else None)
                if pick is None:
                    des.append(v.lx)
                    wt.append(0.01)
                else:
                    des.append(pick[0])
                    wt.append((12.0 if v.kind != "dummy" else 40.0) if strong else 1.0)
            _pav(nodes, des, wt, gap)

    for _ in range(3):
        primary_pass(range(1, K + 1), True)
        primary_pass(range(K - 1, -1, -1), False)
    # level trunks: each dummy chain becomes one vertical line (at its right-most position)
    chains = defaultdict(list)
    for d in all_nodes:
        if d.kind == "dummy" and not d.beside:         # a branch beside a consumer is not part of the shared trunk
            chains[id(d.chain)] .append(d) if getattr(d, "chain", None) is not None else None
    for ch in chains.values():
        if len(ch) < 2:
            continue
        target = max(d.lx for d in ch)
        for l in sorted({d.layer for d in ch}):
            nodes = layers[l]
            ids_ = {id(d) for d in ch}
            des = [target if id(v) in ids_ else v.lx for v in nodes]
            wt = [200.0 if id(v) in ids_ else 0.05 for v in nodes]
            _pav(nodes, des, wt, gap)
    trunk_x = {id(d): d.lx for ch in chains.values() if len(ch) > 1 for d in ch}
    # spines: every chain of links that is central at both ends (input -> embed -> merge -> layers -> head,
    # norm1 -> k -> cat -> SDPA -> o ...) becomes exactly ONE vertical line
    def spines():
        nbr = defaultdict(list)
        for p in pieces.values():
            if p.side or p.frm.id in pinned or p.to.id in pinned:
                continue
            if src_central(p) and dst_central(p):
                d = out_off[(p.frm.id, p.gk)] - in_off[(p.to.id, id(p))]     # to.lx = frm.lx + d
                nbr[p.frm.id].append((p.to, d))
                nbr[p.to.id].append((p.frm, -d))
        byid = {v.id: v for v in all_nodes}
        rel, comps = {}, []
        for v in all_nodes:
            if v.id in rel or v.id not in nbr:
                continue
            rel[v.id], comp, stack = 0.0, [v], [v]
            while stack:
                u = stack.pop()
                for w, d in nbr[u.id]:
                    if w.id not in rel and w.id in byid:
                        rel[w.id] = rel[u.id] + d
                        comp.append(w)
                        stack.append(w)
            comps.append(comp)
        return rel, comps

    rel, comps = spines()
    for _ in range(3):
        target = {}
        for comp in comps:
            t = sum(v.lx - rel[v.id] for v in comp) / len(comp)
            for v in comp:
                target[v.id] = t + rel[v.id]
        for l in range(K + 1):
            nodes = layers[l]
            if nodes:
                _pav(nodes, [target.get(v.id, trunk_x.get(id(v), v.lx)) for v in nodes],
                     [100.0 if v.id in target else (60.0 if id(v) in trunk_x else 0.5) for v in nodes], gap)

    # snap: nearly-straight edges become exactly straight (move the node with the fewest links)
    for _ in range(3):
        for p in sorted(pieces.values(), key=lambda q: nl(q.frm)):
            if p.side or p.frm.id in pinned or p.to.id in pinned:
                continue
            dx = (p.frm.lx + out_off[(p.frm.id, p.gk)]) - (p.to.lx + in_off[(p.to.id, id(p))])
            if 0.5 < abs(dx) < 10:
                mover = p.to if len(ins_of[p.to.id]) + len(outs_of[p.to.id]) <= len(ins_of[p.frm.id]) + len(outs_of[p.frm.id]) else p.frm
                sign = 1 if mover is p.to else -1
                new = mover.lx + sign * dx
                nodes = layers[nl(mover)]
                i = nodes.index(mover)
                lo = nodes[i - 1].lx + nodes[i - 1].w + gap(nodes[i - 1], mover) if i > 0 else -1e9
                hi = nodes[i + 1].lx - mover.w - gap(mover, nodes[i + 1]) if i + 1 < len(nodes) else 1e9
                if lo - 1e-6 <= new <= hi + 1e-6:
                    mover.lx = new
    # slide ports along node edges: near-misses become straight lines instead of short double jogs
    pp_of = lambda v: 3 if v.w < 40 else min(10, v.w * 0.22)
    for _ in range(2):
        for p in sorted(pieces.values(), key=lambda q: nl(q.frm)):
            if p.side or p.frm.id in pinned or p.to.id in pinned:
                continue
            f, t = p.frm, p.to
            if f.kind == "dummy" or t.kind == "dummy" or f.w == 0 or t.w == 0:
                continue
            if any(sg.src_tag for sg in p.segs if sg.src is f) or any(sg.dst_tag for sg in p.finals if sg.dst is t):
                continue
            fx, tx = f.lx + out_off[(f.id, p.gk)], t.lx + in_off[(t.id, id(p))]
            if not 0.5 < abs(fx - tx) <= 140:
                continue
            tlo, thi = t.lx + pp_of(t), t.lx + t.w - pp_of(t)
            free_in = lambda x: all(abs(x - t.lx - in_off[(t.id, id(q))]) >= 12 for q in ins_all[t.id] if q is not p)
            if tlo <= fx <= thi and free_in(fx):
                in_off[(t.id, id(p))] = fx - t.lx
                continue
            single = len([q for q in outs_all[f.id] if q.gk == p.gk]) == 1
            if not single:
                continue
            flo, fhi = f.lx + pp_of(f), f.lx + f.w - pp_of(f)
            free_out = lambda x: all(abs(x - f.lx - out_off[(f.id, q.gk)]) >= 12 for q in outs_all[f.id] if q.gk != p.gk)
            if flo <= tx <= fhi and free_out(tx):
                out_off[(f.id, p.gk)] = tx - f.lx
                continue
            lo, hi = max(flo, tlo), min(fhi, thi)
            if lo <= hi:
                x = min(max((fx + tx) / 2, lo), hi)
                if free_in(x) and free_out(x):
                    out_off[(f.id, p.gk)] = x - f.lx
                    in_off[(t.id, id(p))] = x - t.lx
    # a trunk that is a hair off the port it grows from (rope -> cos / sin) joins that port exactly, so the
    # whole vertical line is one straight stroke instead of showing a sub-pixel nudge
    for l in range(1, K + 1):
        for d in layers[l]:
            if d.kind != "dummy" or getattr(d, "beside", False):
                continue
            for q in ins_of[d.id]:
                px = q.frm.lx + out_off[(q.frm.id, q.gk)] if (q.frm.id, q.gk) in out_off else q.frm.lx
                if 0.05 < abs(d.lx - px) < 3.0 and not any(
                        o is not d and o.lx - 6 <= px <= o.lx + o.w + 6 for o in layers[l]):
                    d.lx = px
                break
    # the container's single entry and single exit share one vertical line, so the outer diagram can run the
    # stream straight through it (a T-junction at the entry follows the exit, not the other way round)
    t_in = [v for v in all_nodes if v.id in tin and v.id not in pinned and v.kind == "tag"]
    t_out = [v for v in all_nodes if v.id in tout and v.id not in pinned and v.kind == "tag"]
    if len(t_in) == 1 and len(t_out) == 1 and abs(t_in[0].lx - t_out[0].lx) > 0.5:
        a, b = t_in[0], t_out[0]
        n_a = len([p for p in pieces.values() if p.frm is a])
        n_b = len([p for p in pieces.values() if p.to is b])
        mover, anchor = (a, b) if n_a > n_b else ((b, a) if n_b > n_a else (None, None))
        if mover is not None and not any(o is not mover and o.lx - 1 <= anchor.lx <= o.lx + o.w + 1
                                         for o in layers[mover.layer if mover.kind == "dummy" else layer[mover.id]]):
            mover.lx = anchor.lx
    # trunks: a long flow (a chain of dummies) runs as ONE vertical line wherever the rows around it leave room,
    # and bends only where a node is in the way (stabbing the free interval of each row with as few x as possible)
    def straighten_trunks():
        INF = 1e9
        for ch in sorted(chains.values(), key=lambda c: -len(c)):
            ch = sorted(ch, key=lambda d: d.layer)
            if len(ch) < 2:
                continue
            def free(d):
                nodes = layers[d.layer]
                i = nodes.index(d)
                lo = nodes[i - 1].lx + nodes[i - 1].w + gap(nodes[i - 1], d) if i > 0 else -INF
                hi = nodes[i + 1].lx - d.w - gap(d, nodes[i + 1]) if i + 1 < len(nodes) else INF
                return lo, hi
            seg, lo, hi = [], -INF, INF
            runs = []
            for d in ch:
                a_, b_ = free(d)
                if seg and (max(lo, a_) > min(hi, b_) + 1e-6 or d.layer != seg[-1].layer + 1):
                    runs.append((seg, lo, hi))
                    seg, lo, hi = [], -INF, INF
                seg.append(d)
                lo, hi = max(lo, a_), min(hi, b_)
            runs.append((seg, lo, hi))
            for seg, lo, hi in runs:
                m = sum(d.lx for d in seg) / len(seg)
                ends = []                       # a run that touches a real node joins that node's port exactly
                if seg[0] is ch[0] and ins_of[seg[0].id]:
                    q = ins_of[seg[0].id][0]
                    if q.frm.kind != "dummy" and (q.frm.id, q.gk) in out_off:
                        ends.append(q.frm.lx + out_off[(q.frm.id, q.gk)])
                if seg[-1] is ch[-1] and outs_of[seg[-1].id]:
                    q = outs_of[seg[-1].id][0]
                    if q.to.kind != "dummy" and (q.to.id, id(q)) in in_off:
                        ends.append(q.to.lx + in_off[(q.to.id, id(q))])
                ends = [e for e in ends if lo <= e <= hi]
                x = min(ends, key=lambda e: abs(e - m)) if ends else min(max(m, lo), hi)
                for d in seg:
                    d.lx = x
    straighten_trunks()
    minx = min((v.lx for v in all_nodes), default=0)      # tight on the left: no empty margin
    for v in all_nodes:
        v.lx -= minx
    W = max([v.lx + v.w for v in all_nodes] + [0])

    # ------------------------------------------------------------------ 5. rows, gaps, tracks
    row_h = {l: max([v.h for v in layers[l]] + [0]) for l in range(K + 1)}
    port_x = lambda p: (p.frm.lx + out_off.get((p.frm.id, p.gk), 0.0))
    land_x = lambda p: p.lane_x if p.lane else (p.to.lx + in_off.get((p.to.id, id(p)), 0.0))
    gap_pieces = defaultdict(list)
    for p in pieces.values():
        if p.side:
            continue
        gap_pieces[nl(p.frm)].append(p)
    tracks = defaultdict(list)                       # gap -> list of track dicts
    for g, plist in gap_pieces.items():
        by_key = {}
        for p in plist:
            first = p.frm.id == p.gk[0]
            own = next((s for s in p.finals if s.alines), None)
            key = ("s", id(own)) if own is not None else ("g", p.gk)
            lines = []
            if first:
                lines += grp_lines.get(p.gk, [])
            if own is not None:
                lines += own.alines
            tr = by_key.setdefault(key, dict(pieces=[], lines=lines))
            tr["pieces"].append(p)
        for tr in by_key.values():
            lat_piece = any(p.frm.id in lat_in or p.to.id in lat_out for p in tr["pieces"])
            horiz = lat_piece or any(
                p.to.id in lat_out or abs(
                    (0 if p.frm.id in lat_in else port_x(p)) - (0 if p.to.id in lat_out else land_x(p))) > 0.5
                for p in tr["pieces"])
            if horiz or tr["lines"]:
                tr["h"] = 12 * len(tr["lines"]) + 10 if tr["lines"] else (18 if lat_piece else 8)
                spans = []
                for p in tr["pieces"]:
                    a = 0.0 if p.frm.id in lat_in else port_x(p)
                    b = 1e9 if p.to.id in lat_out else land_x(p)
                    spans.append((min(a, b), max(a, b)))
                tr["lo"], tr["hi"] = min(a for a, _ in spans), max(b for _, b in spans)
                tr["excl"] = bool(tr["lines"]) or lat_piece       # labelled / side-border tracks keep their own row
                tracks[g].append(tr)
    # ---- residual lanes: placed AFTER the grid, hugging the content of the rows they pass
    row_right = lambda l: max([v.lx + v.w for v in layers[l]] + [0.0])
    lanes = []
    for p in pieces.values():
        if not p.lane:
            continue
        a, b = nl(p.frm), nl(p.to)
        base = max([row_right(l) for l in range(a, b + 1)] +
                   [tr["hi"] for g in range(a, b) for tr in tracks.get(g, []) if tr["hi"] < 1e8] + [0.0])
        base = max(base, p.to.lx + p.to.w + LANE_MIN - LANE_GAP)
        lanes.append([p, a, b, base + LANE_GAP])
    lanes.sort(key=lambda q: (q[2] - q[1], q[1]))
    for i, q in enumerate(lanes):             # lanes whose vertical extents overlap are nested, inner = shorter
        for o in lanes[:i]:
            if q[1] < o[2] and o[1] < q[2] and q[3] < o[3] + LANE_STEP:
                q[3] = o[3] + LANE_STEP
    for p, a, b, lx in lanes:
        p.lane_x = lx
        tracks[a].append(dict(pieces=[p], lines=[], h=8, lo=port_x(p), hi=lx, excl=False))
    W0 = W                                   # content width WITHOUT lanes: lanes never re-centre anything
    if lanes:
        W = max(W, max(q[3] for q in lanes))

    gap_levels, gap_h, track_y = {}, {}, {}
    for g in range(K):                       # gap g sits between layer g and g+1
        trs = sorted(tracks.get(g, []), key=lambda t: (t["lo"], t["hi"]))
        levels = []
        for tr in trs:                       # horizontal runs with disjoint x-spans share a level
            for lv in levels:
                if not tr["excl"] and not lv["excl"] and tr["lo"] >= lv["hi"] + 12:
                    lv["hi"] = max(lv["hi"], tr["hi"])
                    lv["trs"].append(tr)
                    break
            else:
                levels.append(dict(hi=tr["hi"], h=tr["h"], excl=tr["excl"], trs=[tr]))
        def lvl_key(lv):
            """Order of horizontal runs inside a gap that avoids avoidable crossings: runs going left are
            stacked with the right-most start on top, runs going right with the left-most start on top."""
            ps = [p for tr in lv["trs"] for p in tr["pieces"] if p.frm.id not in lat_in and p.to.id not in lat_out]
            if not ps:
                return (3, 0.0)
            d = sum((land_x(p) > port_x(p)) - (land_x(p) < port_x(p)) for p in ps)
            sx = sum(port_x(p) for p in ps) / len(ps)
            return (0, -sx) if d < 0 else ((1, sx) if d > 0 else (2, sx))
        levels.sort(key=lvl_key)

        def live(lv):
            return [(p.gk, port_x(p), land_x(p)) for tr in lv["trs"] for p in tr["pieces"]
                    if p.frm.id not in lat_in and p.to.id not in lat_out]
        cache = {id(lv): live(lv) for lv in levels}

        def above(X, Y):
            """crossings between two levels if X is drawn above Y (nearer the target): Y's target vertical runs up
            through X's horizontal, and X's source vertical runs down through Y's horizontal"""
            c = 0
            for gx, sx, tx in cache[id(X)]:
                for gy, sy, ty in cache[id(Y)]:
                    if gx == gy:
                        continue
                    c += (min(sx, tx) + 1 < ty < max(sx, tx) - 1) + (min(sy, ty) + 1 < sx < max(sy, ty) - 1)
            return c
        pair = {}
        for X in levels:
            for Y in levels:
                if X is not Y:
                    pair[(id(X), id(Y))] = above(X, Y)

        def total(order):
            return sum(pair[(id(order[i]), id(order[j]))] for i in range(len(order)) for j in range(i + 1, len(order)))
        if 1 < len(levels) <= 40:
            for _ in range(4):
                improved = False
                for lv in list(levels):
                    i = levels.index(lv)
                    base = total(levels)
                    best, where = base, i
                    rest = levels[:i] + levels[i + 1:]
                    for j in range(len(rest) + 1):
                        if j == i:
                            continue
                        c = total(rest[:j] + [lv] + rest[j:])
                        if c < best:
                            best, where = c, j
                    if where != i:
                        levels[:] = rest[:where] + [lv] + rest[where:]
                        improved = True
                if not improved:
                    break
        gap_levels[g] = levels
        gap_h[g] = max(26, TOP_CLEAR + BOT_CLEAR + sum(lv["h"] for lv in levels)) if levels else 26
    # a flow entering from the left must never sit LOWER than where it left the previous phase
    for tid in lat_in | lat_out:
        need = getattr(next(i for i in items if i.id == tid), "min_from_bottom", 0.0)
        if not need:
            continue
        for g, levels in gap_levels.items():
            off = float(TOP_CLEAR)
            for lv in levels:
                if any(tid in (p.frm.id, p.to.id) for tr in lv["trs"] for p in tr["pieces"]):
                    below = pad + sum(row_h[l] for l in range(g + 1)) + sum(gap_h[k] for k in range(g))
                    dist = below + gap_h[g] - (off + lv["h"] / 2)
                    if dist < need:
                        gap_h[g] += need - dist
                off += lv["h"]
    y = header_h + 8
    gap_top = {}
    for l in range(K, -1, -1):
        for v in layers[l]:
            # flow runs upward: phases are bottom-aligned so their inputs line up; everything else centred
            v.ly = y + (row_h[l] - v.h) if v.kind == "phase" else y + (row_h[l] - v.h) / 2
        y += row_h[l]
        if l > 0:
            g = l - 1
            gap_top[g] = y
            ty = y + TOP_CLEAR
            for lv in gap_levels[g]:
                for tr in lv["trs"]:
                    tr["y"] = ty + lv["h"] / 2
                    for p in tr["pieces"]:
                        track_y[id(p)] = tr["y"]
                ty += lv["h"]
            y += gap_h[g]
    n.h = y + pad

    # ------------------------------------------------------------------ 6. final coordinates + routes
    base0 = 2 * pad + W0
    ox = pad + max(0.0, header_w - base0) / 2          # centred on the content only
    n.w = max(header_w, base0, ox + W + pad)            # a lane may only widen the container on its right
    for v in all_nodes:
        v.x, v.y = v.lx + ox, v.ly
    for it in items:
        if it.id in pinned:
            it.x = 0.0 if it.id in lat_in else n.w
    for p in pieces.values():
        if p.xside == "in":
            p.frm.y = p.to.y + p.segs[0].dst_tag.y
        elif p.xside == "out":
            p.to.y = p.frm.y + p.segs[0].src_tag.y
        elif p.frm.id in lat_in:
            p.frm.y = track_y.get(id(p), p.frm.y if hasattr(p.frm, "y") else 0)
        elif p.to.id in lat_out:
            p.to.y = track_y.get(id(p), 0)
    n.pieces, n.labels = [], []
    side_count = defaultdict(int)
    JOG = 14.0                    # two ports closer than this on one flow get one straight line, not a double bend

    def snap_x(p, f, t, sx, ex):
        """Common x for a Z-shaped piece whose ends are nearly aligned, or None if neither end can move."""
        if t.id not in tout and t.kind != "dummy" and t.x + 5 < sx < t.x + t.w - 5 and all(
                abs(t.x + o - sx) > 6 for (tid, pid), o in in_off.items() if tid == t.id and pid != id(p)):
            return sx
        if f.id not in tin and f.kind != "dummy" and f.x + 5 < ex < f.x + f.w - 5 and all(
                abs(f.x + o - ex) > 6 for (fid_, gk_), o in out_off.items() if fid_ == f.id and gk_ != p.gk):
            return ex
        return None
    for p in pieces.values():
        f, t = p.frm, p.to
        arrow = any(s.dst is t and s.dst.kind != "tag" and s.dst_tag is None for s in p.finals)
        if p.lane and p.hidden:
            continue
        if p.lane:                                   # residual: stream -> own track -> lane -> into `+` from the right
            k = side_count[t.id]
            side_count[t.id] += 1
            cy = t.y + t.h / 2 + k * 5
            sx = f.x + out_off.get((f.id, p.gk), f.w / 2)
            sy = n.h if f.id in tin else f.y
            ty, lx = track_y[id(p)], p.lane_x + ox
            n.pieces.append(dict(pts=[(sx, sy), (sx, ty), (lx, ty), (lx, cy), (t.x + t.w, cy)], arrow=True,
                                 shape=p.segs[0].mfrom, fid=p.gk[1]))
            continue
        if p.xside:
            y0 = (f.y + p.segs[0].src_tag.y) if p.xside == "out" else (t.y + p.segs[0].dst_tag.y)
            a, b = (f.x + f.w, n.w) if p.xside == "out" else (0.0, t.x)
            for o in items:
                if o is not f and o is not t and o.kind != "tag" and o.x < b and o.x + o.w > a and o.y < y0 < o.y + o.h:
                    L.warnings.append(f"side line of {f if p.xside == 'out' else t} runs through {o}")
            n.pieces.append(dict(pts=[(a, y0), (b, y0)], arrow=False, shape=p.segs[0].mfrom, fid=p.gk[1]))
            continue
        if p.side:
            k = side_count[t.id]
            side_count[t.id] += 1
            cy = t.y + t.h / 2 + (k - 0.0) * 5
            pts = [(f.x, cy), (t.x + t.w, cy)]
            n.pieces.append(dict(pts=pts, arrow=True, shape=p.segs[0].mfrom, fid=p.gk[1]))
            continue
        ty = track_y.get(id(p))
        if f.id in lat_in:
            sx, sy = 0.0, ty
        else:
            sx = f.x + out_off.get((f.id, p.gk), 0.0)
            sy = (n.h if f.id in tin else (f.ly if f.kind == "dummy" else f.y))
        if t.id in lat_out:
            ex, ey = n.w, ty
        else:
            ex = t.x + in_off.get((t.id, id(p)), 0.0)
            ey = (0.0 if t.id in tout else (t.ly if t.kind == "dummy" else t.y + t.h))
        if 0.5 < abs(sx - ex) < JOG and not (f.id in lat_in or t.id in lat_out):
            # a trunk that is a hair off the port it leaves from / arrives at: slide the port onto the trunk
            if t.kind == "dummy" and f.kind != "dummy" and f.id not in tin and f.x + 5 < ex < f.x + f.w - 5 \
                    and len([q for q in outs_all[f.id] if q.gk == p.gk]) == 1 \
                    and all(abs(f.x + o - ex) > 6 for (fid_, gk_), o in out_off.items() if fid_ == f.id and gk_ != p.gk):
                out_off[(f.id, p.gk)] = ex - f.x
                sx = ex
            elif f.kind == "dummy" and t.kind != "dummy" and t.id not in tout and t.x + 5 < sx < t.x + t.w - 5 \
                    and all(abs(t.x + o - sx) > 6 for (tid, pid), o in in_off.items() if tid == t.id and pid != id(p)):
                in_off[(t.id, id(p))] = sx - t.x
                ex = sx
        if f.id in lat_in:
            pts = [(sx, ty), (ex, ty), (ex, ey)]
        elif t.id in lat_out:
            pts = [(sx, sy), (sx, ty), (ex, ty)]
        elif 0.5 < abs(sx - ex) < 3.0 and (f.kind == "dummy" or t.kind == "dummy"):
            x = ex if f.kind == "dummy" else sx           # a trunk that is a hair off its port: follow the real end
            pts = [(x, sy), (x, ey)]
        elif 0.5 < abs(sx - ex) < JOG and f.kind != "dummy" and t.kind != "dummy" and snap_x(p, f, t, sx, ex) is not None:
            x = snap_x(p, f, t, sx, ex)
            pts = [(x, sy), (x, ey)]                     # a hair off: one straight line instead of a double bend
        elif abs(sx - ex) > 0.5:
            pts = [(sx, sy), (sx, ty), (ex, ty), (ex, ey)]
        else:
            pts = [(sx, sy), (sx, ey)]
        n.pieces.append(dict(pts=pts, arrow=arrow, shape=p.segs[0].mfrom, fid=p.gk[1]))
    for g, trs in tracks.items():
        for tr in trs:
            if not tr["lines"]:
                continue
            xs = []
            for p in tr["pieces"]:
                a = 0.0 if p.frm.id in lat_in else p.frm.x + out_off.get((p.frm.id, p.gk), 0.0)
                b = n.w if p.to.id in lat_out else p.to.x + in_off.get((p.to.id, id(p)), 0.0)
                xs.append((abs(a - b), a, b))
            span, a, b = max(xs)
            wl = max(len(t) * 6.0 for t in tr["lines"]) + 8
            if span >= wl + 14:
                n.labels.append(dict(lines=tr["lines"], x=(a + b) / 2, y=tr["y"], anchor="middle", w=wl))
            else:
                n.labels.append(dict(lines=tr["lines"], x=max(a, b) + 6, y=tr["y"], anchor="start", w=wl))

    # ------------------------------------------------------------------ lateral (phase to phase) links
    if phases_mode:
        lat_segs = [s for s in segs if lateral(s) and s.src_tag is not None and s.dst_tag is not None]
        lat_segs.sort(key=lambda s: (s.src.x, s.src_tag.y))
        mids = {}
        by_pair = defaultdict(list)
        for s in lat_segs:
            by_pair[(s.src.id, s.dst.id)].append(s)
        for grp in by_pair.values():
            mids.update(_channel_order(grp))
        for i, s in enumerate(lat_segs):
            P1, P2 = s.src, s.dst
            sx, sy = P1.x + P1.w, P1.y + s.src_tag.y
            ex, ey = P2.x, P2.y + s.dst_tag.y
            straight = abs(sy - ey) <= 0.5          # both ends equally high: a single line
            mid = (sx + ex) / 2 if straight else mids[id(s)]
            pts = [(sx, sy), (ex, ey)] if straight else [(sx, sy), (mid, sy), (mid, ey), (ex, ey)]
            n.pieces.append(dict(pts=pts, arrow=False, shape=s.mfrom, fid=s.fid))
            lines = s.glines + s.alines
            if lines:
                wl = max(len(t) * 6.0 for t in lines) + 8
                ly = sy if straight else (sy + ey) / 2 + (i - (len(lat_segs) - 1) / 2) * 44    # label sits on its line
                n.labels.append(dict(lines=lines, x=mid, y=ly, anchor="middle", w=wl))
