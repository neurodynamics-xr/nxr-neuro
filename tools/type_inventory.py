#!/usr/bin/env python3
"""Inventory of what flows through nxr-neuro's pipelines, for the pipeline type system (spec docs/specs/2026-11-17-type-system.md).

Four sections, each rerunnable on the tracked sources:
  ports     every shipped definition (pipelines/definitions/*.yaml): its inputs (class, kind, relation), its uses (section, step, wiring) and its
            declared outputs; every registered step (pipelines/src/*.cpp): kind, input, optional and output ports
  checks    today's per-step checks: every `need(...)` / `refuse(...)` whose message names a step, counted per step and sorted by what it checks
            (keyword classification: approximate, read it as an order of magnitude)
  types     the spec's typing rules run on every definition: each value's inferred type (class, domain with its manifolds, cells, value type,
            quantity, band, spatial level, linearity), and the declared output kinds checked against them. The input types are those the runner
            would read from the found rows; where the definition does not say (a stored kernel's value type, a joint partition's surface) the
            prototype states its assumption in STORED below.
  mistakes  seeded mistakes, each a small edit of a shipped definition, and the refusal the rules give before any data is read

The prototype is the spec's executable sketch, not the implementation (that is C++, in pipelines/graph/). It is kept so that a change to a
definition or to a step's ports shows up here: rerun it after each PR of the plan; a definition that stops checking, or a step with no rule
(listed as UNRULED), is a gap in the spec.

Usage:  tools/type_inventory.py [--repo .] [--section ports|checks|types|mistakes|all]
"""
from __future__ import annotations

import argparse, collections, copy, glob, itertools, json, os, re, sys
from dataclasses import dataclass, field, replace
from typing import Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("type_inventory: needs PyYAML (pip install pyyaml)")


# ───────────────────────────────────────────── ports: definitions and registered steps ─────────────────────────────────────────────

def expand_for(items, env):
    """A definition's `for:` blocks expanded (cartesian product over the named parameters' values); `{name}` filled from the loop variables."""
    out = []
    for it in items:
        if isinstance(it, dict) and "for" in it and "do" in it:
            names = list(it["for"].keys())
            values = [env.get(str(v).lstrip("$"), [None]) if str(v).startswith("$") else v for v in it["for"].values()]
            for combo in itertools.product(*values):
                sub = dict(env)
                sub.update({n: c for n, c in zip(names, combo)})
                out += expand_for(it["do"], sub)
        else:
            out.append(fill(it, env))
    return out


def fmt(v):
    return "_".join(f"{x:g}" for x in v) if isinstance(v, list) else str(v)


def fill(obj, env):
    if isinstance(obj, str):
        if obj.startswith("$") and obj[1:] in env:
            return env[obj[1:]]
        return re.sub(r"\{(\w+)\}", lambda m: fmt(env[m.group(1)]) if m.group(1) in env and m.group(1) != "recording" else m.group(0), obj)
    if isinstance(obj, dict):
        return {fill(k, env): fill(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill(v, env) for v in obj]
    return obj


def load_definition(path):
    d = yaml.safe_load(open(path))
    params = {k: (v.get("default") if isinstance(v, dict) else v) for k, v in (d.get("parameters") or {}).items()}
    inputs = {}
    for k, v in d["inputs"].items():
        if isinstance(v, dict) and "for" in v:
            names = list(v["for"].keys())
            vals = [params.get(str(x).lstrip("$"), [None]) for x in v["for"].values()]
            for combo in itertools.product(*vals):
                env = dict(zip(names, combo))
                inputs[fill(k, env)] = fill({a: b for a, b in v.items() if a != "for"}, env)
        else:
            inputs[k] = v
    steps = d["steps"]
    uses = []
    for sec in ("before", "setup", "chunks", "after"):
        s = steps.get(sec)
        if not s:
            continue
        do = s.get("do", []) if isinstance(s, dict) else s
        for u in expand_for(do, params):
            uses.append((sec, u))
    outputs = {}
    for k, v in d["outputs"].items():
        if isinstance(v, dict) and "for" in v:
            names = list(v["for"].keys())
            vals = [params.get(str(x).lstrip("$"), [None]) for x in v["for"].values()]
            for combo in itertools.product(*vals):
                outputs[fill(k, dict(zip(names, combo)))] = {a: b for a, b in v.items() if a != "for"}
        else:
            outputs[k] = v or {}
    return {"name": d["pipeline"], "params": params, "inputs": inputs, "uses": uses, "outputs": outputs}


STEP_FORMS = [
    re.compile(r'push_back\(\{"(?P<name>[a-z_0-9]+)", "(?P<kind>[a-z]+)", \{(?P<ins>[^}]*)\}, \{(?P<opt>[^}]*)\}, \{(?P<outs>[^}]*)\}'),
    re.compile(r'simple\("(?P<name>[a-z_0-9]+)", \{(?P<ins>[^}]*)\}, \{(?P<opt>[^}]*)\}, \{(?P<outs>[^}]*)\}'),
]
NAMED = re.compile(r'\b(?P<v>[st])\.name = "(?P<name>[a-z_0-9]+)";(?: \1\.kind = "(?P<kind>[a-z]+)";)?')


def names_in(s):
    return re.findall(r'"([a-z_0-9]+)"', s)


def registered_steps(repo):
    steps = {}
    for path in sorted(glob.glob(os.path.join(repo, "pipelines/src/*.cpp"))):
        text = open(path, encoding="utf-8").read()
        for rx in STEP_FORMS:
            for m in rx.finditer(text):
                steps[m["name"]] = {"file": os.path.basename(path), "kind": m.groupdict().get("kind") or "after",
                                    "inputs": names_in(m["ins"]), "optional": names_in(m["opt"]), "outputs": names_in(m["outs"])}
        for m in NAMED.finditer(text):
            v, name = m["v"], m["name"]
            tail = text[m.end(): m.end() + 900]
            ins = re.search(v + r'\.inputs = \{([^}]*)\}', tail)
            opt = re.search(v + r'\.optional = \{([^}]*)\}', tail)
            outs = re.search(v + r'\.outputs = \{([^}]*)\}', tail)
            if not ins and not outs:
                continue   # a row's name (a selection, a dataset), not a step
            steps[name] = {"file": os.path.basename(path), "kind": m["kind"] or "after", "inputs": names_in(ins[1]) if ins else [],
                           "optional": names_in(opt[1]) if opt else [], "outputs": names_in(outs[1]) if outs else []}
    return steps


def section_ports(repo, defs, steps):
    print("## Definitions: inputs, uses, outputs\n")
    for d in defs:
        print(f"### {d['name']}")
        for k, v in d["inputs"].items():
            cls = next((c for c in ("field", "operator", "selection", "statistic") if c in v), "?")
            rel = {r: v[r] for r in ("manifold", "from", "to") if r in v}
            print(f"  input  {k:28s} {cls}: {v[cls]}" + (f"  {rel}" if rel else "") + ("  (optional)" if v.get("required") is False else ""))
        for sec, u in d["uses"]:
            outs = u.get("out")
            print(f"  {sec:6s} {u['step']:18s} {json.dumps(u.get('in') or {})} -> {outs}")
        kinds = collections.Counter((v.get("kind") or "?") for v in d["outputs"].values())
        print("  outputs " + ", ".join(f"{n} {k}" for k, n in kinds.items()) + "\n")
    print("## Registered steps\n")
    used = collections.Counter(u["step"] for d in defs for _, u in d["uses"])
    for name, s in sorted(steps.items()):
        print(f"  {name:20s} {s['kind']:10s} in {s['inputs']} opt {s['optional']} out {s['outputs']}  uses {used.get(name, 0)}  ({s['file']})")
    missing = sorted(set(used) - set(steps))
    if missing:
        print("\n  used but not found in the sources:", missing)


# ───────────────────────────────────────────── checks: today's per-step checks ─────────────────────────────────────────────

CHECK_KINDS = [  # (category, regex); the first that matches wins
    ("linearity (D106)", r"non-linear|nonlinear|\bD106\b|phase is gone|band limit is gone"),
    ("class or producer of an input", r"must be (a|an|the) (field|operator|selection|measurement|statistic|spans|labelling|sample series|band's|relative power|frame tiling|tile ladder)|\bmade by\b|made_by|`[a-z_]+`'s |must be `"),
    ("value type", r"\bcomplex\b|real field|\bvector|\bscalar\b|monogenic|ℝ|ℂ|ℍ|value type|field type|components"),
    ("manifold identity", r"\bmanifold\b|\bsurface\b|time line|first factor|same manifold|\bglued\b|of recording|another (time line|surface|recording)|vertex count|\bchannels\b"),
    ("partition, ladder, level", r"\bpartition|\bladder\b|\blevels?\b|\btiles?\b|\bjoint\b|\bcells?\b|\bspans\b|snap"),
    ("band, rate", r"\bband\b|Nyquist|resolves|samples a second|sfreq|\brate\b"),
    ("stored record", r"not among the subject's rows|not stored|records no|holds no|not recorded"),
    ("parameter or array shape", r""),
]
CALL_RX = re.compile(r'\b(?:need|refuse)\((?P<body>(?:[^;]|;(?!\s*\n))*?)\);', re.S)
LIT_RX = re.compile(r'"((?:[^"\\]|\\.)*)"')


def section_checks(repo, steps):
    per_step = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(repo, "pipelines/src/*.cpp"))):
        text = open(path, encoding="utf-8").read()
        for m in CALL_RX.finditer(text):
            lits = LIT_RX.findall(m["body"])
            if not lits:
                continue
            k = next((j for j, l in enumerate(lits) if re.match(r"([a-z_]+): ", l) and l.split(":")[0] in steps), None)
            if k is None:
                continue
            st = lits[k].split(":")[0]
            msg = " … ".join(lits[k:])[len(st) + 2:]
            cat = next(c for c, rx in CHECK_KINDS if re.search(rx, msg))
            per_step[st][cat] += 1
            if len(examples[cat]) < 3:
                examples[cat].append(f"{st}: {msg[:110]}")
    total = collections.Counter()
    for c in per_step.values():
        total.update(c)
    print("## Today's checks in the registered steps, by what they check\n")
    print(f"  {sum(total.values())} checks in {len(per_step)} steps\n")
    for cat, _ in CHECK_KINDS:
        print(f"  {total[cat]:4d}  {cat}")
        for e in examples[cat]:
            print(f"          e.g. {e}")
    print("\n## Per step\n")
    for st in sorted(per_step):
        print(f"  {st:20s} " + ", ".join(f"{c} {n}" for c, n in per_step[st].most_common()))


# ───────────────────────────────────────────── types: the spec's rules, executable ─────────────────────────────────────────────
# A manifold is named by a string term:  an id ("cortex(kernel)"), members(P) (the member set of a partition), frames(P) (the members of a
# frame tiling: the frame line), sub(M, S) (M restricted to the elements of selection S), glued(M).  KIND records what is known of each.

class Refusal(Exception):
    pass


def need(ok, msg):
    if not ok:
        raise Refusal(msg)


@dataclass(frozen=True)
class Field:
    dom: tuple                    # the factors, in axis order; the time-like factor last
    vt: str = "real"              # real · complex · vector3 · monogenic
    quantity: str = "signal"      # what the values ARE (the field's kind): recording · amplitude · power · phase_current · …
    linear: bool = True           # a linear function of the recording (D106)
    band: Optional[str] = None    # the time band it is limited to
    level: Optional[str] = None   # the spatial band: "<ladder>@<level>" or "<ladder>@[levels]" (a stack of levels as components)
    streamed: bool = True
    cls: str = "field"


@dataclass(frozen=True)
class Selection:
    on: tuple                     # the manifold(s) it indexes (a product's factors for a joint partition)
    type: str                     # spans · set · ladder · tiling · indices
    name: str                     # identity: the stored row, or the node and use that made it
    partition: bool = False       # its members are disjoint: a measurement may be taken over it
    band: Optional[str] = None    # a time ladder level's band (the level IS the band), or a joint partition's
    level: Optional[str] = None   # a spatial ladder level, or a joint partition's
    over: Optional[str] = None    # detected events: the partition whose members they were found on
    cls: str = "selection"


@dataclass(frozen=True)
class Measurement:
    dom: tuple                    # the field's factors with the partitioned ones replaced by members(P)
    quantity: str
    vt: str = "real"
    over: str = ""                # the partition's name
    band: Optional[str] = None
    level: Optional[str] = None
    cls: str = "measurement"


@dataclass(frozen=True)
class Statistic:
    axes: tuple
    measure: str
    of: Optional[tuple] = None    # what it is of: (quantity, band, level, domain) of the field or measurement
    components: tuple = ()
    cls: str = "statistic"


@dataclass(frozen=True)
class Operator:
    frm: str
    to: str
    vt: str = "vector3"
    kind: str = "inverse kernel"
    cls: str = "operator"


KIND = {}   # manifold term -> {"type": points|line|surface|members|…, "role": time|frames|…, "cs": coordinate system}


def mk(term, **facts):
    KIND[term] = facts
    return term


def is_time(term):
    return KIND.get(term, {}).get("role") in ("time", "frames")


def base(term):
    """the manifold a restriction is of: sub(M, S) -> M"""
    m = re.match(r"sub\((.*), (.*)\)$", term)
    return m.group(1) if m else term


def members(sel):
    return mk(f"members({sel.name})", type="members", role="frames" if sel.type == "tiling" else ("time-tiles" if sel.on and is_time(sel.on[-1]) else "members"))


STORED = {  # what the runner would read from the found rows, where the definition does not say it (the prototype's assumptions, each a fact of a row)
    "inverse kernel value type": "vector3 (free orientation); a fixed-orientation kernel is scalar",
    "joint partition": "on (the kernel's target surface × the recording's time line), at the space level and band its `where` names",
    "spans on recording.time": "a spans Selection on that time line; `tiling` in a definition that reads frames is a frame tiling",
    "statistic: mean": "the spatial baseline: axes (the cortex, levels), of the relative power chain it names",
    "statistic: histogram": "a distribution of a relative power or a curl at frames (axes: bins)",
    "statistic: welch": "axes (channels of the recording or of a kernel, frequency line)",
}


def input_type(name, spec, found, params):
    """the type of a found input, from the definition's relation and the STORED facts"""
    if "field" in spec:
        s = mk(f"sensors({name})", type="points", cs="head/scs")
        t = mk(f"time({name})", type="line", role="time", cs="time/clock")
        return Field(dom=(s, t), quantity="recording")
    if "operator" in spec:
        frm = rel(spec.get("from"), found)
        to = mk(f"cortex({name})", type="surface", cs="head/scs")
        return Operator(frm=mk(f"sub({frm}, channels({name}))", type="points"), to=to)
    if "selection" in spec:
        kind = spec["selection"]
        on = rel(spec.get("manifold"), found) if spec.get("manifold") else None
        if kind == "ladder":
            return Selection(on=(on,), type="ladder", name=name)
        if kind == "spans":
            if name == "tiling":
                return Selection(on=(on,), type="tiling", name=name, partition=True)
            return Selection(on=(on,), type="spans", name=name, partition=True)
        if kind == "set":   # a joint partition: STORED["joint partition"]
            w = spec.get("where", {})
            lvl = w.get("params.joint.space_level")
            lvl = params.get(lvl[1:]) if isinstance(lvl, str) and lvl.startswith("$") else lvl
            band = w.get("params.joint.band")
            band = params.get(band[1:]) if isinstance(band, str) and band.startswith("$") else band
            surf = next((t.to for t in found.values() if isinstance(t, Operator)), None)
            time = next((t.dom[-1] for t in found.values() if isinstance(t, Field)), None)
            tiles = next((k for k, t in found.items() if isinstance(t, Selection) and t.type == "ladder" and t.on == (surf,)), "tiles")
            return Selection(on=(surf, time), type="set", name=name, partition=True, band=fmt(band), level=f"{tiles}@{lvl}")
    if "statistic" in spec:
        m = spec["statistic"]
        if m == "mean":
            return Statistic(axes=("cortex(kernel)",), measure="mean", of=("relative-power-chain",))
        if m == "histogram":
            return Statistic(axes=("bins",), measure="histogram")
        if m == "welch":
            return Statistic(axes=("channels", "frequency"), measure="welch")
        return Statistic(axes=("tiles",), measure=m)
    raise Refusal(f"input {name}: class not understood")


def rel(ref, found):
    if ref is None:
        return None
    x, _, what = ref.partition(".")
    t = found[x]
    if what == "time":
        return t.dom[-1]
    if what.startswith("manifold."):
        return t.dom[int(what.split(".")[1])]
    if what == "to":
        return t.to
    if what == "from":
        return t.frm
    raise Refusal(f"relation {ref} not understood")


def in_loop(x, who, frames_ok=False):
    need(isinstance(x, Field) and x.streamed and is_time(x.dom[-1]), f"{who}: x must be a field on … × time inside the loop; it is {describe(x)}")
    need(frames_ok or KIND[x.dom[-1]]["role"] == "time", f"{who}: x is on the frame line {x.dom[-1]}; this node reads the samples of a time line")


def same(a, b, what, who):
    need(a == b, f"{who}: {what}: {a} is not {b}")


def describe(t):
    if isinstance(t, Field):
        return f"Field<{' × '.join(t.dom)}; {t.vt}; {t.quantity}" + (f"; band {t.band}" if t.band else "") + (f"; level {t.level}" if t.level else "") + ("" if t.linear else "; non-linear") + ("" if t.streamed else "; array") + ">"
    if isinstance(t, Measurement):
        return f"Measurement<{' × '.join(t.dom)}; {t.quantity}; over {t.over}>"
    if isinstance(t, Selection):
        return f"Selection<{t.type} on {' × '.join(str(o) for o in t.on)}" + (" partition" if t.partition else "") + (f"; band {t.band}" if t.band else "") + (f"; level {t.level}" if t.level else "") + ">"
    if isinstance(t, Statistic):
        return f"Statistic<{t.measure}; axes {' × '.join(t.axes)}>"
    if isinstance(t, Operator):
        return f"Operator<{t.frm} → {t.to}; {t.vt}>"
    return repr(t)


def pointwise(x, who, **change):
    in_loop(x, who, frames_ok=True)
    return replace(x, **change)


# the rules: one per step, (inputs by port, parameters, the use's name) -> outputs by port. MOVE says which move each step is.
MOVE = {}


def rule(name, move):
    def deco(f):
        MOVE[name] = (move, f)
        return f
    return deco


@rule("analytic", "pointwise (linear, windowed)")
def _analytic(i, p, u):
    x = i["x"]; in_loop(x, "analytic")
    need(x.vt == "real", f"analytic: x is {x.vt}; the analytic signal is taken of a real field")
    return {"y": replace(x, vt="complex", band=fmt(p.get("band")) if p.get("band") else x.band)}


@rule("band_filter", "pointwise (linear, windowed)")
def _band_filter(i, p, u):
    x = i["x"]; in_loop(x, "band_filter"); need(x.vt == "real", "band_filter: x is complex")
    return {"y": replace(x, band=fmt(p.get("band")))}


@rule("envelope", "pointwise (non-linear, windowed)")
def _envelope(i, p, u):
    x = i["x"]; in_loop(x, "envelope"); need(x.vt == "real", "envelope: x is complex; the envelope is taken of the real field")
    return {"y": replace(x, quantity="envelope", linear=False, band=fmt(p.get("band")) if p.get("band") else x.band)}


@rule("frames", "apply (an interpolation operator on the time factor: time line → the frame tiling's members)")
def _frames(i, p, u):
    x, t = i["x"], i["tiling"]; in_loop(x, "frames")
    need(x.linear, f"frames: x is non-linear ({x.quantity}): its band limit is gone, it cannot be interpolated")
    need(t.type == "tiling", f"frames: tiling is a {t.type} Selection, not a frame tiling")
    same(t.on[0], x.dom[-1], "the frame tiling's time line", "frames")
    if "within" in i:
        same(i["within"].on[0], x.dom[-1], "`within`'s time line", "frames")
    return {"y": replace(x, dom=x.dom[:-1] + (members(t),))}


@rule("apply_kernel", "apply (an operator A → B on the first factor), with a spatial band folded in when `tiles` is given")
def _apply_kernel(i, p, u):
    x, k = i["x"], i["kernel"]; in_loop(x, "apply_kernel", frames_ok=True)
    need(isinstance(k, Operator), "apply_kernel: kernel must be an operator")
    need(x.linear, f"apply_kernel: x is non-linear ({x.quantity}); a linear map reads the chain before any non-linear node (D106)")
    a = x.dom[0]
    need(k.frm == a or base(k.frm) == a, f"apply_kernel: the kernel maps from {k.frm}; x's first factor is {a}")
    level = None
    if "tiles" in i:
        t = i["tiles"]
        need(t.type == "ladder" and t.on == (k.to,), f"apply_kernel: tiles must be the tile ladder of the kernel's target {k.to}; it is on {t.on}")
        lv = p.get("level") if p.get("level") is not None else p.get("levels")
        need(lv is not None, "apply_kernel: `tiles` comes with `level` or `levels`")
        level = f"{t.name}@{lv}"
    to = mk(f"glued({k.to})", type="surface") if p.get("glued") else k.to
    vt = k.vt if p.get("form", "none") == "none" else "real"
    if p.get("envelope") == "spatiotemporal":
        vt = "monogenic"
    if x.vt == "complex" and vt == "vector3":
        vt = "complex vector3"
    elif x.vt == "complex" and vt == "real":
        vt = "complex"
    return {"y": replace(x, dom=(to,) + x.dom[1:], vt=vt, quantity="current" if p.get("form", "none") == "none" else p["form"], level=level)}


@rule("magnitude", "pointwise (non-linear)")
def _magnitude(i, p, u):
    x = i["x"]
    return {"y": pointwise(x, "magnitude", vt="real", quantity="amplitude", linear=False)}


@rule("power", "pointwise (non-linear)")
def _power(i, p, u):
    x = i["x"]
    need("complex" in x.vt, f"power: x is {x.vt}; the band power |z|² is taken of the analytic signal's map")
    return {"y": pointwise(x, "power", vt="real", quantity="power", linear=False)}


@rule("phase_current", "pointwise (non-linear; a tangent vector, stored embedded: no gauge)")
def _phase_current(i, p, u):
    x = i["x"]; in_loop(x, "phase_current", frames_ok=True)
    need("complex" in x.vt and x.linear, f"phase_current: x must be complex and linear (a phase is kept); it is {describe(x)}")
    need(KIND.get(base(x.dom[0]), {}).get("type") == "surface", f"phase_current: x's first factor {x.dom[0]} is not a surface")
    return {"y": replace(x, vt="vector3", quantity="phase_current", linear=False)}


@rule("measure", "measure (a field over a partition's members; the only reduction)")
def _measure(i, p, u):
    x = i["x"]; in_loop(x, "measure")
    need(x.vt in ("real", "vector3"), f"measure: x is {x.vt}; measure a real quantity of it (power, a form, the phase current)")
    m = p.get("measure", "mean")
    if "exclude" in i:
        same(i["exclude"].on[0], x.dom[-1], "`exclude`'s time line", "measure")
    if "over" not in i:   # the run's span: one member
        return {"y": Measurement(dom=x.dom[:-1], quantity=f"{m} of {x.quantity}", over="the run's span", band=x.band, level=x.level)}
    P = i["over"]
    need(P.partition or P.type == "ladder", f"measure: over must be a partition (its members disjoint); {P.name} is a {P.type} Selection")
    want = {"amplitude": "amplitude|envelope", "envelope_max": "amplitude|envelope", "median": "envelope", "power": "power", "phase_current": "phase_current"}.get(m)
    if want:
        need(re.fullmatch(want, x.quantity), f"measure: {m} is a measure of {want.replace('|', ' or ')}; x is {x.quantity}")
    if P.type == "ladder":   # a time ladder: the level whose band is x's band (the level IS the band)
        same(P.on[0], x.dom[-1], "the ladder's time line", "measure")
        need(x.band is not None, "measure: x carries no band; a constant-Q level is chosen by the band")
        lvl = replace(P, type="tiling", name=f"{P.name}@{x.band}", band=x.band)
        return {"y": Measurement(dom=x.dom[:-1] + (members(lvl),), quantity=m, over=lvl.name, band=x.band, level=x.level)}
    if P.type == "set":      # a joint partition of space × time
        same(P.on, (x.dom[0], x.dom[-1]), "the joint partition's manifolds (surface, time line)", "measure")
        same(P.level, x.level, "the joint partition's space level against x's spatial band", "measure")
        same(P.band, x.band, "the joint partition's band against x's time band", "measure")
        return {"y": Measurement(dom=(members(P),), quantity=m, vt="vector3" if m == "phase_current" else "real", over=P.name, band=x.band, level=x.level)}
    raise Refusal(f"measure: over {P.name} ({P.type}) is not a partition this node measures over")


@rule("tile_max", "measure (over the spatial tiles of a level, per sample) then the max across the members")
def _tile_max(i, p, u):
    x, t = i["x"], i["tiles"]; in_loop(x, "tile_max")
    need(x.quantity == "amplitude", f"tile_max: x is {x.quantity}; |z| per vertex (`magnitude`) is the trace's input")
    need(t.type == "ladder" and t.on == (x.dom[0],), f"tile_max: tiles must be the tile ladder of x's surface {x.dom[0]}")
    same(f"{t.name}@{p.get('level')}", x.level, "the tiles' level against x's spatial band", "tile_max")
    g = mk(f"groups({t.name}@{p.get('level')}, {p.get('groups', 'whole_brain')})", type="members")
    return {"y": Measurement(dom=(g, x.dom[-1]), quantity="trace", over=g, band=x.band, level=x.level)}


@rule("collect", "pointwise (materialize: the streamed value kept as an array)")
def _collect(i, p, u):
    x = i["x"]
    need(getattr(x, "vt", "real") == "real", "collect: x is complex")
    return {"y": Field(dom=x.dom, quantity=x.quantity, band=x.band, level=x.level, linear=False, streamed=False)}


@rule("reduce_cells", "summary across members (the max over the spatial tiles of each time tile)")
def _reduce_cells(i, p, u):
    x = i["x"]
    need(isinstance(x, Measurement) and x.over and x.dom[0].startswith("members(cells"), f"reduce_cells: x must be a measurement over a joint partition; it is {describe(x)}")
    need(x.vt == "real", "reduce_cells: x is a vector measurement; the max over tiles is of a scalar")
    tt = mk(f"time-tiles({x.over})", type="members", role="time-tiles")
    g = mk(f"groups({x.over}, {p.get('groups', 'whole_brain')})", type="members")
    return {"y": Measurement(dom=(g, tt), quantity=x.quantity, over=x.over, band=x.band, level=x.level)}


@rule("summarize", "statistic across a partition's members")
def _summarize(i, p, u):
    x = i["x"]
    need(isinstance(x, Measurement), f"summarize: x must be a measurement over a partition; it is {describe(x)} (a field is reduced by `measure` first)")
    need(x.vt == "real", "summarize: x is a vector measurement")
    keep = tuple(a for a in x.dom if not (KIND.get(a, {}).get("role") in ("time-tiles", "frames") or a.startswith("members(") and (len(x.dom) == 1 or a == x.dom[-1])))
    return {"y": Statistic(axes=keep, measure="summaries", of=(x.quantity, x.band, x.level, x.dom), components=tuple(p.get("summaries", ["median", "mad"])))}


@rule("detect_spans", "detect (a measurement over a time partition's members → spans on the time line)")
def _detect_spans(i, p, u):
    x = i["x"]
    need(isinstance(x, Measurement), f"detect_spans: x must be a measurement over a time partition; it is {describe(x)}")
    need(KIND.get(x.dom[-1], {}).get("role") in ("time-tiles", "frames"), f"detect_spans: x's last factor {x.dom[-1]} is not the members of a time partition; events are found on a time line")
    need(x.vt == "real", "detect_spans: x is a vector measurement; events are found on a scalar trace")
    if "baseline" in i:
        b = i["baseline"]
        need(isinstance(b, Statistic) and b.of and b.of[0] == x.quantity and b.of[3] == x.dom, f"detect_spans: the baseline is not a statistic of x ({describe(b)})")
    time = next((m for m in KIND if KIND[m].get("role") == "time"), "time")
    return {"spans": Selection(on=(time,), type="spans", name=u, partition=True, over=x.over, band=x.band)}


@rule("rollup_events", "measure (a tile measurement re-measured over the events, the exact roll-up of their tiles)")
def _rollup(i, p, u):
    x, ev = i["x"], i["events"]
    need(isinstance(x, Measurement) and x.over, f"rollup_events: x must be a measurement over a tile level; it is {describe(x)}")
    need(isinstance(ev, Selection) and ev.type == "spans", "rollup_events: events must be a spans Selection")
    if ev.over:
        same(ev.over, x.over, "the partition the events were detected on against x's", "rollup_events")
    m = p.get("measure")
    if m not in ("duration", "interval_next"):
        same(m, x.quantity, "the measure against x's", "rollup_events")
    return {"y": Measurement(dom=x.dom[:-1] + (members(ev),), quantity=m, vt=x.vt, over=ev.name, band=x.band, level=x.level)}


@rule("risefall", "measure (a fit per event of the sample trace)")
def _risefall(i, p, u):
    x, ev = i["x"], i["events"]
    need(isinstance(x, Measurement) or (isinstance(x, Field) and not x.streamed), "risefall: x must be the sample series of the run (`collect`)")
    need(isinstance(ev, Selection) and ev.type == "spans" and ev.over, "risefall: events must be the spans `detect_spans` made")
    outs = ["t_on", "t_off", "tau_rise", "tau_fall", "fit_baseline", "fit_height", "fit_r2", "plateau_duration"]
    return {o: Measurement(dom=(members(ev),), quantity=o, over=ev.name) for o in outs}


@rule("fit_gate", "detect (a subset of the events)")
def _fit_gate(i, p, u):
    ev, r2 = i["events"], i["fit_r2"]
    need(isinstance(r2, Measurement) and r2.quantity == "fit_r2" and r2.over == ev.name, "fit_gate: fit_r2 must be `risefall`'s fit_r2 of these events")
    return {"spans": replace(ev, name=u)}


@rule("burst_stages", "detect (three spans Selections from each event's fit)")
def _burst_stages(i, p, u):
    ev = i["events"]
    for m in ("t_on", "t_off", "tau_rise", "tau_fall"):
        need(isinstance(i[m], Measurement) and i[m].over == ev.name, f"burst_stages: {m} must be `risefall`'s {m} of these events")
    return {s: Selection(on=ev.on, type="spans", name=f"{u}.{s}") for s in ("onset", "within", "offset")}


@rule("cell_extent", "detect (regions of the joint cells per event → a labelling of the cells)")
def _cell_extent(i, p, u):
    x, b, ev = i["x"], i["baseline"], i["events"]
    need(isinstance(x, Measurement) and x.dom[0].startswith("members(cells"), "cell_extent: x must be a measurement over a joint partition")
    need(isinstance(b, Statistic) and b.of and b.of[3] == x.dom, f"cell_extent: the baseline is not a statistic of x ({describe(b)})")
    need(isinstance(ev, Selection) and ev.type == "spans", "cell_extent: events must be spans")
    return {"labels": Selection(on=x.dom, type="set", name=u, partition=True, over=x.over)}


@rule("extent_measure", "measure (over the labelling's members: the events)")
def _extent_measure(i, p, u):
    lab = i["labels"]
    need(isinstance(lab, Selection) and lab.type == "set" and lab.on[0].startswith("members(cells"), "extent_measure: labels must be the labelling `cell_extent` made")
    if "x" in i:
        same(i["x"].dom, lab.on, "x's cells against the labelling's", "extent_measure")
    return {"y": Measurement(dom=(members(lab),), quantity=p.get("measure"), over=lab.name)}


@rule("relative", "pointwise (divided by a statistic: a constant)")
def _relative(i, p, u):
    x, b = i["x"], i["baseline"]; in_loop(x, "relative", frames_ok=True)
    need(x.quantity == "power", f"relative: x is {x.quantity}; a band's power is divided by its baseline")
    need(isinstance(b, Statistic) and b.measure == "mean", "relative: baseline must be the spatial baseline (a statistic: the mean)")
    return {"y": replace(x, quantity="relative power")}


@rule("at_level", "restrict (one level of a stack of levels: a component)")
def _at_level(i, p, u):
    x = i["x"]
    need(x.level and "[" in x.level, f"at_level: x is not a multi-level field (`apply_kernel` with `levels`); its spatial band is {x.level}")
    return {"y": pointwise(x, "at_level", level=x.level.split("@")[0] + f"@{p.get('level')}")}


@rule("scale", "pointwise (across the stack of levels: the characteristic level)")
def _scale(i, p, u):
    x = i["x"]
    need(x.level and "[" in x.level, "scale: x carries no stack of spatial levels")
    need(x.dom[0].startswith("glued("), "scale: x must be on the glued surface (`apply_kernel` with `glued`)")
    return {"y": pointwise(x, "scale", quantity="characteristic level", level=None)}


@rule("baseline", "statistic (of a field over the good frames: a time partition's members)")
def _baseline(i, p, u):
    x = i["x"]; in_loop(x, "baseline", frames_ok=True)
    need(x.quantity == "power", f"baseline: x is {x.quantity}; the baseline is of a band's power")
    return {"y": Statistic(axes=x.dom[:-1], measure="mean", of=(x.quantity, x.band, x.level, x.dom[:-1]))}


@rule("relative_histogram", "statistic (a histogram over the frames)")
def _relative_histogram(i, p, u):
    x = i["x"]; in_loop(x, "relative_histogram", frames_ok=True)
    need(x.quantity == "relative power", f"relative_histogram: x is {x.quantity}")
    return {"y": Statistic(axes=("bins",), measure="histogram", of=(x.quantity, x.band, x.level, x.dom[:-1]))}


@rule("core_histogram", "statistic (a histogram over the frames)")
def _core_histogram(i, p, u):
    x = i["x"]; in_loop(x, "core_histogram", frames_ok=True)
    return {"y": Statistic(axes=("bins",), measure="histogram", of=(x.quantity, x.band, x.level, x.dom[:-1]))}


def _events(x, u, outs, who):
    in_loop(x, who, frames_ok=True)
    ev = Selection(on=x.dom, type="spans", name=u, partition=True)
    res = {outs[0]: ev}
    for o in outs[1:]:
        res[o] = Measurement(dom=(members(ev),), quantity=o, over=u)
    return res


@rule("cores", "detect (vortex cores at frames) with their measurements")
def _cores(i, p, u):
    x = i["x"]
    need(x.quantity == "curl_tan", f"cores: x is {x.quantity}; the cores are found on the tangential curl (`apply_kernel` with form curl_tan)")
    return _events(x, u, ["spans", "frame", "sense", "centre_vertex", "centre", "area", "peak"], "cores")


@rule("patches", "detect (patches at frames) with their measurements and a summary statistic")
def _patches(i, p, u):
    x = i["x"]
    need(x.quantity == "relative power", f"patches: x is {x.quantity}; patches are found on a relative power map")
    if "scale" in i:
        same(i["scale"].dom[-1], x.dom[-1], "the scale's frames against x's", "patches")
    r = _events(x, u, ["spans", "frame", "burst", "centre_vertex", "centre", "area", "diameter_cm", "characteristic_level", "size_cm", "relative_peak"], "patches")
    r["home"] = Statistic(axes=x.dom[:-1], measure="home", of=(x.quantity, x.band, x.level, x.dom[:-1]))   # per vertex: the good frames in a patch
    r["summary"] = Statistic(axes=(), measure="summary")
    return r


@rule("tracks", "detect (cores linked across frames: a partition of the events) with their measurements")
def _tracks(i, p, u):
    ev = i["events"]
    for m in ("frame", "sense", "centre"):
        need(isinstance(i[m], Measurement) and i[m].over == ev.name, f"tracks: {m} must be a measurement over the events {ev.name}")
    tr = Selection(on=(members(ev),), type="set", name=u, partition=True)
    res = {"spans": tr}
    for o in ("first_frame", "last_frame", "n_frames", "path_length", "displacement", "sense"):
        res[o] = Measurement(dom=(members(tr),), quantity=o, over=u)
    res["track"] = Measurement(dom=(members(ev),), quantity="track", over=ev.name)
    return res


@rule("tile_crossings", "measure (crossings over the tiles of a level)")
def _tile_crossings(i, p, u):
    ev, t = i["events"], i["tiles"]
    need(t.type == "ladder", "tile_crossings: tiles must be a tile ladder")
    same(t.on[0], base(ev.on[0]).replace("glued(", "").rstrip(")") if ev.on[0].startswith("glued(") else ev.on[0], "the ladder's surface against the events'", "tile_crossings")
    lvl = replace(t, type="set", name=f"{t.name}@{p.get('level')}")
    if p.get("as") == "statistic":   # a null draw's crossings: a statistic of its own, never the data's rows
        return {"crossings": Statistic(axes=(members(lvl),), measure="crossings")}
    return {"crossings": Measurement(dom=(members(lvl),), quantity="crossings", over=lvl.name)}


@rule("power_series", "statistic (band power per dyadic time tile over the cross-spectrum's channels)")
def _power_series(i, p, u):
    x, cs = i["x"], i["cross_spectrum"]; in_loop(x, "power_series")
    need("complex" in x.vt, "power_series: x must be the analytic signal")
    return {"series": Statistic(axes=("tiles",), measure="power_tiles", of=("power", x.band, None, x.dom[:-1]))}


@rule("surrogate.lagfree", "pointwise (a surrogate draw, unmixed: a field made from statistics on the cross-spectrum's channels; it awaits the zero-lag mixing)")
def _lagfree(i, p, u):
    x = i["x"]; in_loop(x, "surrogate.lagfree")
    s = mk(f"sub({x.dom[0]}, channels(kernel))", type="points")   # STORED: the cross-spectrum was made with --channels-of the kernel
    return {"y": replace(x, dom=(s, x.dom[-1]), quantity="surrogate")}


def check_definition(d, edits=None):
    """the spec's checker on one definition: every value's type, or the first refusal (with the use that refused)"""
    KIND.clear()
    params = dict(d["params"])
    found = {}
    for name, spec in d["inputs"].items():
        found[name] = input_type(name, spec, found, params)
    values = dict(found)
    for k, (sec, u) in enumerate(d["uses"]):
        st = u["step"]
        if st not in MOVE:
            raise Refusal(f"UNRULED {st}")
        p = {}
        for a, b in (u.get("with") or {}).items():
            p[a] = params.get(b[1:]) if isinstance(b, str) and b.startswith("$") else b
        ins = {}
        for port, v in (u.get("in") or {}).items():
            if v not in values:
                if d["inputs"].get(v, {}).get("required") is False:
                    continue
                raise Refusal(f"{sec}[{k}] {st}: in {port}: no value {v}")
            ins[port] = values[v]
        outs = u.get("out")
        outs = [outs] if isinstance(outs, str) else list(outs)
        name = outs[0]
        try:
            made = MOVE[st][1](ins, p, name)
        except Refusal as e:
            raise Refusal(f"{sec}[{k}] {st} -> {name}: {e}")
        ports = list(made.keys())
        for o, port in zip(outs, ports):
            values[o] = made[port]
    for o, decl in d["outputs"].items():
        need(o in values, f"outputs: {o} is not a value of the graph")
        kind = decl.get("kind", "?")
        cls = {"spans": "selection", "labelling": "selection"}.get(kind, kind)
        need(values[o].cls == cls or (kind == "statistic" and isinstance(values[o], Statistic)),
             f"outputs: {o} is declared {kind}; the graph makes {describe(values[o])}")
    return values


def section_types(defs):
    print("## Every definition checked by the spec's rules\n")
    print("  What the prototype assumes the found rows say (the runner reads these at plan time):")
    for k, v in STORED.items():
        print(f"    {k}: {v}")
    print()
    ok = 0
    for d in defs:
        try:
            values = check_definition(d)
            ok += 1
            print(f"### {d['name']}: checks")
            for n, t in values.items():
                if n in d["inputs"]:
                    continue
                print(f"  {n:30s} {describe(t)}")
        except Refusal as e:
            print(f"### {d['name']}: REFUSED  {e}")
        print()
    used = {u["step"] for d in defs for _, u in d["uses"]}
    print(f"  {ok} of {len(defs)} definitions check; steps used: {len(used)}, with a rule: {len(used & set(MOVE))}")
    moves = collections.Counter(MOVE[s][0].split(" ")[0] for s in used if s in MOVE)
    print("  moves of the steps used: " + ", ".join(f"{m} {n}" for m, n in moves.most_common()))


MISTAKES = [  # (definition, what the edit does, the edit)
    ("band_bursts", "measure the analytic signal (complex) instead of its envelope",
     lambda d: d["uses"][0][1].update(step="analytic")),
    ("global_bursts", "the spatial scale changed in the chain while the cells were made at another scale (the coupled product is the caller's)",
     lambda d: d["uses"][1][1]["with"].update(spatial_scale="wavelength 20 mm")),
    ("spatial_events", "frames interpolated after `power` (the band limit is gone)",
     lambda d: (d["uses"].insert(1, ("chunks", {"step": "power", "in": {"x": "z"}, "out": "zp"})), d["uses"][2][1]["in"].update(x="zp"))),
    ("band_bursts", "events detected on a field instead of a measurement over the time tiles",
     lambda d: next(u for _, u in d["uses"] if u["step"] == "detect_spans")["in"].update(x="amp")),
    ("global_bursts", "summarize a streamed field (a reduction that is not a measurement over a partition)",
     lambda d: d["uses"].append(("after", {"step": "summarize", "in": {"x": "mag"}, "out": "oops"}))),
    ("global_bursts", "tile_max on a ladder of another surface (the recording's sensors)",
     lambda d: d["inputs"]["tiles"].update(manifold="recording.manifold.0")),
    ("global_bursts", "an output declared a measurement that is a statistic",
     lambda d: d["outputs"]["baseline"].update(kind="measurement")),
    ("band_bursts", "roll-up over events detected on another partition (the peak's events on the amplitude's tiles of another band)",
     lambda d: (d["uses"].insert(2, ("chunks", {"step": "envelope", "in": {"x": "recording"}, "with": {"band": [14.0, 30.0]}, "out": "amp2"})),
                d["uses"][3][1]["in"].update(x="amp2"))),
]


def section_mistakes(defs):
    print("## Seeded mistakes: each refused before any data is read\n")
    by = {d["name"]: d for d in defs}
    caught = 0
    for name, what, edit in MISTAKES:
        d = copy.deepcopy(by[name])
        edit(d)
        try:
            check_definition(d)
            print(f"  NOT CAUGHT  {name}: {what}")
        except Refusal as e:
            caught += 1
            print(f"  {name}: {what}\n      refused: {e}")
    print(f"\n  {caught} of {len(MISTAKES)} caught")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--section", default="all", choices=["ports", "checks", "types", "mistakes", "all"])
    a = ap.parse_args()
    defs = [load_definition(p) for p in sorted(glob.glob(os.path.join(a.repo, "pipelines/definitions/*.yaml")))]
    steps = registered_steps(a.repo)
    if a.section in ("ports", "all"):
        section_ports(a.repo, defs, steps)
    if a.section in ("checks", "all"):
        section_checks(a.repo, steps)
    if a.section in ("types", "all"):
        section_types(defs)
    if a.section in ("mistakes", "all"):
        section_mistakes(defs)


if __name__ == "__main__":
    main()
