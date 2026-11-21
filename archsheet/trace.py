"""archsheet.trace — trace a PyTorch model into a hierarchical *dataflow* graph.

What gets recorded
  * every module call (hierarchy, per-call input/output tensor metadata, param count)
  * functional ops executed inside `forward` (add, mul, matmul, sdpa, cat, ...)
  * which tensor flows from which producer to which consumer (residuals, skips, fan-out)

Tracing is done with forward hooks + a TorchFunctionMode, normally on the `meta`
device, so no memory is allocated and no real math happens.
"""
from __future__ import annotations

import contextlib
import dataclasses
import inspect
import itertools
from collections import namedtuple
from functools import partial

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

Meta = namedtuple("Meta", "shape dtype")

# ops that only move / reinterpret data: never drawn as nodes (shown as edge annotations)
SHAPE_OPS = {
    "view", "reshape", "transpose", "permute", "contiguous", "unsqueeze", "squeeze",
    "flatten", "unflatten", "to", "type_as", "float", "half", "bfloat16", "double",
    "long", "int", "bool", "detach", "getitem", "split", "chunk", "unbind", "narrow",
    "t", "expand", "expand_as", "view_as", "movedim", "swapaxes", "swapdims", "T", "mT",
    "clone", "cpu", "cuda", "type", "reshape_as", "tensor_split", "select", "repeat",
    "repeat_interleave", "as_strided", "unfold", "squeeze_", "unsqueeze_",
}
# single-source ops still worth drawing in the default `key` mode
KEY_OPS = {
    "matmul", "bmm", "mm", "einsum", "sdpa", "scaled_dot_product_attention", "softmax",
    "log_softmax", "cat", "concat", "stack", "embedding", "topk", "gather", "scatter",
    "index_select", "where", "multi_head_attention_forward", "linear", "layer_norm",
    "rms_norm", "group_norm", "batch_norm", "instance_norm", "gelu", "silu", "relu",
    "sigmoid", "tanh", "softplus", "conv1d", "conv2d", "conv3d", "conv_transpose1d",
    "conv_transpose2d", "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d", "interpolate",
    "cross_entropy", "mse_loss", "sort", "argmax", "cumsum", "pad",
}
ALIASES = {
    "radd": "add", "iadd": "add", "rmul": "mul", "imul": "mul", "rsub": "sub",
    "isub": "sub", "truediv": "div", "rtruediv": "div", "itruediv": "div",
    "floordiv": "floor_div", "rmatmul": "matmul", "rpow": "pow", "ipow": "pow",
    "mul_": "mul", "div_": "div", "add_": "add", "sub_": "sub",
}
_EXCLUDED_ATTR_HINTS = ("idx", "index", "layer_id", "layer_num", "training", "layer_number")

_ids = itertools.count()
ACTIVE = None          # the Tracer currently installed (used by phase() / label())


@contextlib.contextmanager
def phase(name):
    """Group everything executed inside the block into a named dashed region
    (e.g. `with phase("shortie"): model.shortie(src)`). No-op outside archsheet."""
    t = ACTIVE
    if t is None:
        yield
        return
    t.begin_phase(name)
    try:
        yield
    finally:
        t.end_phase()


def label(tensor, text):
    """Name the tensor flow leaving its producer (e.g. label(k, "K · appended")). Returns tensor."""
    if ACTIVE is not None:
        ACTIVE.label_tensor(tensor, text)
    return tensor


def meta(t: torch.Tensor) -> Meta:
    return Meta(tuple(int(d) for d in t.shape), str(t.dtype).replace("torch.", ""))


def iter_tensors(obj, _d=0):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif _d > 6:
        return
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_tensors(v, _d + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_tensors(v, _d + 1)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from iter_tensors(getattr(obj, f.name, None), _d + 1)


def iter_named(obj, name, _d=0):
    if isinstance(obj, torch.Tensor):
        yield name, obj
    elif _d > 6:
        return
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_named(v, f"{name}.{k}" if name else str(k), _d + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from iter_named(v, f"{name}[{i}]", _d + 1)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from iter_named(getattr(obj, f.name, None), f"{name}.{f.name}" if name else f.name, _d + 1)


class Node:
    """A thing in the diagram: scene | module | op | input | output | group | tag."""

    def __init__(self, kind, name, type_="", path=""):
        self.id = next(_ids)
        self.seq = float(self.id)
        self.kind = kind
        self.name = name
        self.type = type_
        self.path = path
        self.parent = None          # original call-tree parent
        self.children = []          # original call-tree children
        self.in_metas = []
        self.in_names = []          # forward() parameter names, parallel to in_metas
        self.out_metas = []
        self.cat = None             # (dim, [sizes along dim]) for concatenations
        self.out_ann, self.in_ann = [], []     # "↳ view,transpose [..]" rows shown on the node itself
        self.side = None            # tags: 'L' / 'R' when the port sits on a side border (phases)
        self.extra = ""
        self.params = 0
        self.attrs = {}
        self.leaf = False
        self.repeat = 1
        self.note = ""
        # filled by archsheet.graph
        self.dchildren = None
        self.dparent = None
        self.depth = 0
        self.expanded = False
        self.disp = None
        self.tags_in = {}
        self.tags_out = {}
        # filled by layout
        self.w = self.h = 0.0
        self.x = self.y = 0.0

    def __repr__(self):
        return f"<{self.kind} {self.path or self.name}>"


class Edge:
    __slots__ = ("src", "dst", "mfrom", "mto", "ops", "label", "trail")

    def __init__(self, src, dst, mfrom, mto, ops, label="", trail=()):
        self.src, self.dst, self.mfrom, self.mto, self.ops, self.label = src, dst, mfrom, mto, tuple(ops), label
        self.trail = tuple(trail)       # ((container node, meta after op, op name), ...) for hidden ops

    def key(self):
        return (self.src.id, self.dst.id, self.mfrom, self.mto, self.ops, self.label)


def norm_op_name(func) -> str:
    n = getattr(func, "__name__", None) or str(func)
    if n == "__get__":
        return "T"
    n = ALIASES.get(n, n)
    n = n.strip("_")
    return ALIASES.get(n, n)


def _fmt_num(n):
    return f"{n:,}".replace(",", "_") if isinstance(n, int) and n >= 100000 else str(n)


def pretty_extra(m: nn.Module) -> str:
    """One-line, human-friendly configuration string for a module."""
    try:
        if isinstance(m, nn.Linear):
            return f"{m.in_features} → {m.out_features}" + ("" if m.bias is not None else ", no bias")
        if isinstance(m, nn.Embedding):
            return f"{m.num_embeddings} × {m.embedding_dim}"
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return f"{m.in_channels} → {m.out_channels}, k={m.kernel_size}, s={m.stride}"
        if isinstance(m, nn.LayerNorm):
            return f"{tuple(m.normalized_shape)}, eps={m.eps}"
        r = m.extra_repr()
    except Exception:
        return ""
    r = " ".join(r.split())
    return r if len(r) <= 70 else r[:67] + "…"


def named_inputs(module, args, kwargs):
    """[(forward-parameter name, tensor)] for a module call."""
    try:
        sig = inspect.signature(module.forward)
        bound = sig.bind_partial(*args, **kwargs)
        items = []
        for pname, val in bound.arguments.items():
            kind = sig.parameters[pname].kind
            if kind is inspect.Parameter.VAR_POSITIONAL:
                for i, v in enumerate(val):
                    items += list(iter_named(v, f"{pname}[{i}]"))
            elif kind is inspect.Parameter.VAR_KEYWORD:
                for k, v in val.items():
                    items += list(iter_named(v, k))
            else:
                items += list(iter_named(val, pname))
        return items
    except (TypeError, ValueError):
        return [(f"arg{i}", t) for i, t in enumerate(iter_tensors((args, kwargs)))]


def scalar_attrs(m: nn.Module) -> dict:
    out = {}
    for k, v in vars(m).items():
        if k.startswith("_") or not isinstance(v, (int, float, bool, str, type(None))):
            continue
        if any(h in k for h in _EXCLUDED_ATTR_HINTS):
            continue
        out[k] = v
    return out


class Tracer(TorchFunctionMode):
    def __init__(self, root: nn.Module, ops="key", hide_types=(), leaf_types=()):
        super().__init__()
        self.root = root
        self.ops = ops
        self.hide = set(hide_types)
        self.leaf_types = set(leaf_types)
        self.scene = Node("scene", "scene")
        self.cur_top = self.scene
        self.phase_of, self._phase_stack = {}, []
        self.labels, self.pending_out = {}, []
        self.total_params = sum(p.numel() for p in root.parameters())
        self.root_node = None
        self.stack, self.calls = [], []
        self.leaf_depth = 0
        self.paused = 0
        self.prov, self.keep, self.edges = {}, [], []
        self._handles = []

    # ------------------------------------------------------------ install
    def install(self):
        global ACTIVE
        ACTIVE = self
        for name, m in self.root.named_modules():
            self._handles.append(m.register_forward_pre_hook(partial(self._pre, name), with_kwargs=True))
            self._handles.append(m.register_forward_hook(partial(self._post, name), with_kwargs=True))
        return self

    def remove(self):
        global ACTIVE
        ACTIVE = None
        for h in self._handles:
            h.remove()
        self._handles = []

    # ------------------------------------------------------------ helpers
    def _is_leaf(self, m):
        return (not any(True for _ in m.children())) or type(m).__name__ in self.leaf_types

    def _set_prov(self, t, node, m, ops=(), trail=()):
        self.prov[id(t)] = (node, m, tuple(ops), tuple(trail))
        self.keep.append(t)

    def _consume(self, dst, tensors):
        for t in tensors:
            p = self.prov.get(id(t))
            if p:
                self.edges.append(Edge(p[0], dst, p[1], meta(t), p[2], self.labels.get((p[0].id, p[1]), ""), p[3]))

    def begin_phase(self, name):
        """Phases opened between module calls (run(model)) hang off the scene; phases opened inside a
        module's forward() become children of that module and adopt the calls made while they are open."""
        n = Node("phase", name, "phase")
        host = self.stack[-1] if self.stack else self.scene
        n.parent = host
        host.children.append(n)
        if self.stack:
            self.phase_of[host.id] = n
        else:
            self.cur_top = n
        self._phase_stack.append(host)

    def end_phase(self):
        host = self._phase_stack.pop()
        if host is self.scene:
            self.cur_top = self.scene
        else:
            self.phase_of.pop(host.id, None)

    def _parent(self):
        host = self.stack[-1]
        return self.phase_of.get(host.id, host)

    def label_tensor(self, t, text):
        p = self.prov.get(id(t))
        if p:
            self.labels[(p[0].id, p[1])] = text

    # ------------------------------------------------------------ module hooks
    def _pre(self, path, module, args, kwargs):
        self.paused += 1
        try:
            tname = type(module).__name__
            if self.leaf_depth or tname in self.hide:
                self.calls.append(("skip", args, kwargs))
                self.leaf_depth += 1
                return
            first = not self.stack
            top = self.cur_top
            if first:
                self._make_inputs(top, module, args, kwargs)
            node = Node("module", path.rsplit(".", 1)[-1] if path else tname, tname, path)
            node.leaf = self._is_leaf(module)
            node.extra = pretty_extra(module)
            node.attrs = scalar_attrs(module)
            node.params = sum(p.numel() for p in module.parameters())
            named = named_inputs(module, args, kwargs)
            node.in_names = [nm for nm, _ in named]
            node.in_metas = [meta(t) for _, t in named]
            parent = top if first else self._parent()
            node.parent = parent
            parent.children.append(node)
            if first and self.root_node is None:
                self.root_node = node
            if node.leaf:
                self._consume(node, list(iter_tensors((args, kwargs))))
                self.leaf_depth += 1
            self.stack.append(node)
            self.calls.append(("node", node))
        finally:
            self.paused -= 1

    def _post(self, path, module, args, kwargs, out):
        self.paused += 1
        try:
            kind, *rest = self.calls.pop()
            if kind == "skip":
                self.leaf_depth -= 1
                a, kw = rest
                srcs = [self.prov[id(t)] for t in iter_tensors((a, kw)) if id(t) in self.prov]
                if srcs:
                    for t in iter_tensors(out):
                        if id(t) not in self.prov and meta(t).shape == srcs[0][1].shape:
                            self._set_prov(t, srcs[0][0], srcs[0][1], srcs[0][2], srcs[0][3])
                return
            node = rest[0]
            self.stack.pop()
            if node.leaf:
                self.leaf_depth -= 1
            outs = list(iter_tensors(out))
            node.out_metas = [meta(t) for t in outs]
            if node.leaf:
                for t in outs:
                    self._set_prov(t, node, meta(t))
            if not self.stack:
                self.pending_out.append((self.cur_top, list(iter_named(out, ""))))
        finally:
            self.paused -= 1

    def _make_inputs(self, top, module, args, kwargs):
        try:
            params = [p for p in inspect.signature(module.forward).parameters]
        except (TypeError, ValueError):
            params = []
        items = []
        for i, a in enumerate(args):
            items += list(iter_named(a, params[i] if i < len(params) else f"arg{i}"))
        for k, v in kwargs.items():
            items += list(iter_named(v, k))
        for name, t in items:
            if id(t) in self.prov:          # already produced earlier (another phase): not an input
                continue
            n = Node("input", name, "input")
            n.parent = top
            n.out_metas = [meta(t)]
            top.children.append(n)
            self._set_prov(t, n, meta(t))

    def _phase_of_node(self, n):
        while n is not None:
            if n.kind == "phase":
                return n
            n = n.parent
        return None

    def _home_inputs(self):
        """A graph input that only one phase reads is drawn inside that phase (at its bottom)."""
        for n in list(self.scene.children):
            if n.kind != "input":
                continue
            ph = {id(self._phase_of_node(e.dst)): self._phase_of_node(e.dst) for e in self.edges if e.src is n}
            if len(ph) == 1:
                p = next(iter(ph.values()))
                if p is not None and p is not n.parent:
                    self.scene.children.remove(n)
                    n.parent = p
                    p.children.insert(0, n)

    def finalize(self):
        """Create output nodes for top-level results that nothing consumed afterwards."""
        self._home_inputs()
        for top, items in self.pending_out:
            for name, t in items:
                p = self.prov.get(id(t))
                if p is None or any(e.src is p[0] and e.mfrom == p[1] for e in self.edges):
                    continue
                name = name or "out"
                if name.startswith("["):
                    name = "out" + name
                n = Node("output", name, "output")
                n.parent = top
                n.in_metas = [meta(t)]
                top.children.append(n)
                self._consume(n, [t])

    # ------------------------------------------------------------ functional ops
    def _should_show(self, name, nsrc):
        if name in SHAPE_OPS or self.ops == "none":
            return False
        if self.ops == "all" or nsrc >= 2:
            return True
        return self.ops == "key" and name in KEY_OPS

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if self.paused or self.leaf_depth or not self.stack:
            return out
        outs = list(iter_tensors(out))
        if not outs:
            return out
        ins = [t for t in iter_tensors((args, kwargs)) if id(t) in self.prov]
        if not ins:
            return out                      # constant / creation op: nothing flows in
        name = norm_op_name(func)
        srcs = {}
        for t in ins:
            p = self.prov[id(t)]
            srcs.setdefault(p[0].id, p)
        if self._should_show(name, len(srcs)):
            node = Node("op", name, name)
            node.parent = self._parent()
            node.parent.children.append(node)
            node.in_metas = [meta(t) for t in ins]
            node.out_metas = [meta(t) for t in outs]
            if name in ("cat", "concat", "concatenate"):
                try:
                    dim = kwargs.get("dim", args[1] if len(args) > 1 and isinstance(args[1], int) else 0)
                    seq = args[0] if args else kwargs.get("tensors")
                    dim %= outs[0].dim()
                    node.cat = (dim, [int(t.shape[dim]) for t in seq])
                except Exception:
                    node.cat = None
            self._consume(node, ins)
            for t in outs:
                self._set_prov(t, node, meta(t))
        else:
            n0, m0, ops0, tr0 = next(iter(srcs.values()))
            for t in outs:
                self._set_prov(t, n0, m0, ops0 + (name,), tr0 + ((self.stack[-1], meta(t), name),))
        return out


class _NoAutocast:
    """Stand-in for torch.autocast(device_type="meta"): context manager and decorator, does nothing."""
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __call__(self, fn):
        return fn


@contextlib.contextmanager
def meta_autocast_noop():
    """HF rotary embeddings (and many custom fp32 regions) open `torch.autocast(device_type="meta")`,
    which torch rejects. Make that one a no-op while tracing; every other device is untouched."""
    targets = [(torch, "autocast")]
    for mod in (getattr(torch, "amp", None), getattr(getattr(torch, "amp", None), "autocast_mode", None)):
        if mod is not None and hasattr(mod, "autocast"):
            targets.append((mod, "autocast"))
    orig = [(m, n, getattr(m, n)) for m, n in targets]

    def make(real):
        def autocast(*a, **k):
            dev = k.get("device_type", a[0] if a and isinstance(a[0], str) else None)
            return _NoAutocast() if dev == "meta" else real(*a, **k)
        autocast.__wrapped__ = real
        return autocast

    try:
        for m, n, real in orig:
            setattr(m, n, make(real))
        yield
    finally:
        for m, n, real in orig:
            setattr(m, n, real)
