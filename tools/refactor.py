#!/usr/bin/env python3
"""Tooling for the toolbox refactor (headers and sources move between folders, names change).

  tools/refactor.py check-map [--map FILE]        every former name of the rename map is found in its former header; prints the missing ones
  tools/refactor.py move OLD NEW [options]        git mv a header (and its source), rewrite namespaces and includes, leave a forwarder at OLD
  tools/refactor.py forwarders [--remove]         list the temporary forwarding headers (or delete them)
  tools/refactor.py owned B [--base REF]          fail when the diff against REF (default feat/refactor) leaves batch B's files
  tools/refactor.py rename OLD NEW [--dry-run]    whole-word rename of an identifier across compute, meeg, pipelines, cli and bindings

Header paths are relative to compute/include/nxr/ (meeg/X.h is meeg/include/nxr/meeg/X.h).
The rename map is board/proposed.json of the planning folder (set NXR_REFACTOR_MAP or pass --map); the ownership of each batch is tools/refactor-ownership.json.
"""
import argparse, fnmatch, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORWARDER_MARK = "// TEMPORARY FORWARDER (toolbox refactor): deleted at the end of batch B5, see tools/refactor.py forwarders."
CODE_DIRS = ["compute", "meeg", "pipelines", "cli", "bindings"]
CODE_EXT = (".h", ".hpp", ".cpp", ".cc", ".cxx", ".inl")


def p(*a):
    return os.path.join(ROOT, *a)


def git(*a, check=True):
    return subprocess.run(["git", "-C", ROOT, *a], check=check, capture_output=True, text=True).stdout


def header_file(rel):
    """Where the header named relative to include/nxr/ lives (compute first, then meeg)."""
    for base in ("compute/include/nxr", "meeg/include/nxr"):
        f = p(base, rel)
        if os.path.exists(f):
            return f
    return None


def code_files(dirs=CODE_DIRS, ext=CODE_EXT):
    for d in dirs:
        for dp, dn, fn in os.walk(p(d)):
            dn[:] = [x for x in dn if x not in ("deps", "build", "_deps", "node_modules", ".git", "third_party")]
            for f in fn:
                if f.endswith(ext):
                    yield os.path.join(dp, f)


# ---------------------------------------------------------------- check-map
def cmd_check_map(a):
    mp = a.map or os.environ.get("NXR_REFACTOR_MAP") or "/Volumes/Pergammon/workspace/research/dev-data/nxr-neuro/2026-11-17-toolboxes/board/proposed.json"
    data = json.load(open(mp))
    cache, missing, total = {}, [], 0
    for tb in data["toolboxes"]:
        for h in tb["headers"]:
            for fn in h["functions"]:
                was = fn.get("was") or ""
                for seg in [s.strip() for s in was.split(";")]:
                    if not seg or seg == "(new)":
                        continue
                    m = re.match(r"^([\w/]+\.h):\s*(.+)$", seg)
                    if not m:
                        missing.append((seg, h["path"] + ": " + fn["name"], "unparsed"))
                        continue
                    hdr, name = m.group(1), m.group(2).strip()
                    total += 1
                    if hdr not in cache:
                        f = header_file(hdr)
                        cache[hdr] = open(f, errors="replace").read() if f else None
                    txt = cache[hdr]
                    if txt is None:
                        missing.append((seg, h["path"] + ": " + fn["name"], "header not found"))
                        continue
                    tail = name.split("::")[-1]
                    if not re.search(r"(?<![\w])" + re.escape(tail) + r"(?![\w])", txt):
                        missing.append((seg, h["path"] + ": " + fn["name"], "name not in header"))
    for seg, new, why in missing:
        print("MISSING  %-50s -> %-45s (%s)" % (seg, new, why))
    print("checked %d former names, %d missing" % (total, len(missing)))
    return 1 if missing else 0


# ---------------------------------------------------------------- move
def rewrite_namespaces(text, pairs, sub):
    for old, new in pairs:
        new_full = new + ("::" + sub if sub else "")
        text = re.sub(r"(namespace\s+)(?:nxr::)?%s(\s*\{)" % re.escape(old), lambda m: m.group(1) + new_full + m.group(2), text)
        # nested form `namespace nxr { namespace old {`: the same rule applies (the inner block matched above)
    return text


def cmd_move(a):
    old, new = a.old, a.new
    fo, fn_ = header_file(old), p("compute/include/nxr", new)
    if not fo:
        sys.exit("no such header: " + old)
    if os.path.exists(fn_):
        sys.exit("destination exists: " + new)
    pairs = [tuple(x.split("=", 1)) for x in (a.namespace or [])]
    os.makedirs(os.path.dirname(fn_), exist_ok=True)
    git("mv", os.path.relpath(fo, ROOT), os.path.relpath(fn_, ROOT))
    touched = [fn_]
    # the source: compute/src/<stem>.cpp -> compute/src/<toolbox>/<stem>.cpp
    stem = os.path.splitext(os.path.basename(old))[0]
    src_old = p("compute/src", a.source or (stem + ".cpp"))
    src_new = None
    if os.path.exists(src_old):
        tb = new.split("/")[0] if "/" in new else None
        if tb:
            src_new = p("compute/src", tb, os.path.basename(src_old))
            os.makedirs(os.path.dirname(src_new), exist_ok=True)
            git("mv", os.path.relpath(src_old, ROOT), os.path.relpath(src_new, ROOT))
            for cm in sorted(os.listdir(p("compute/cmake"))) if os.path.isdir(p("compute/cmake")) else []:
                f = p("compute/cmake", cm)
                t = open(f).read()
                o_rel = "src/" + os.path.basename(src_old)
                n_rel = "src/%s/%s" % (tb, os.path.basename(src_old))
                t2 = re.sub(r"(?<![\w/])" + re.escape(o_rel) + r"(?![\w])", n_rel, t)
                if t2 != t:
                    open(f, "w").write(t2)
        else:
            src_new = src_old
        touched.append(src_new)
    inc_old, inc_new = '#include "nxr/%s"' % old, '#include "nxr/%s"' % new
    for f in touched:
        t = open(f).read()
        t = rewrite_namespaces(t, pairs, a.sub_namespace)
        t = t.replace(inc_old, inc_new)
        open(f, "w").write(t)
    # the forwarder at the old path
    lines = ["#pragma once", FORWARDER_MARK, inc_new]
    for extra in a.extra_include or []:
        lines.append('#include "nxr/%s"' % extra)
    for o, n in pairs:
        if o != n or a.sub_namespace:
            lines.append("namespace nxr::%s { using namespace nxr::%s%s; }" % (o, n, ("::" + a.sub_namespace) if a.sub_namespace else ""))
    open(fo, "w").write("\n".join(lines) + "\n")
    git("add", "-A", os.path.relpath(fo, ROOT))
    print("moved %s -> %s%s; forwarder left at %s" % (old, new, (" and " + os.path.relpath(src_new, ROOT)) if src_new else "", os.path.relpath(fo, ROOT)))
    return 0


# ---------------------------------------------------------------- forwarders
def cmd_forwarders(a):
    found = []
    for d in ("compute/include", "meeg/include"):
        for dp, _, fn in os.walk(p(d)):
            for f in fn:
                path = os.path.join(dp, f)
                if f.endswith(".h") and FORWARDER_MARK in open(path, errors="replace").read():
                    found.append(path)
    for f in sorted(found):
        print(os.path.relpath(f, ROOT))
        if a.remove:
            git("rm", "-f", os.path.relpath(f, ROOT))
    print("%d forwarder(s)" % len(found))
    return 0


# ---------------------------------------------------------------- owned
def owned_patterns(own, b):
    d = own[b]
    pats = []
    for k in ("patterns", "files"):
        pats += d.get(k, [])
    for h in d.get("headers", []) + d.get("creates", []):
        pats += ["compute/include/nxr/" + h, "meeg/include/nxr/" + h]
    for s in d.get("sources", []):
        pats += ["compute/src/%s.cpp" % s, "compute/src/*/%s.cpp" % s]
    for t in d.get("tests", []):
        pats += ["compute/test/%s.cpp" % t]
    for tb in d.get("source_dirs", []):
        pats += ["compute/src/%s/*" % tb]
    return pats


def cmd_owned(a):
    own = json.load(open(p("tools/refactor-ownership.json")))
    own.pop("_comment", None)
    if a.batch not in own:
        sys.exit("unknown batch %s (have %s)" % (a.batch, ", ".join(sorted(own))))
    base = subprocess.run(["git", "-C", ROOT, "merge-base", a.base, "HEAD"], capture_output=True, text=True)
    ref = base.stdout.strip() or a.base
    changed = set(git("diff", "--name-only", ref).split()) | set(git("ls-files", "--others", "--exclude-standard").split())
    pats = owned_patterns(own, a.batch)
    out = sorted(f for f in changed if f and not any(fnmatch.fnmatch(f, pt) for pt in pats))
    for f in out:
        print("NOT OWNED by %s: %s" % (a.batch, f))
    print("%d changed file(s), %d outside %s" % (len(changed), len(out), a.batch))
    return 1 if out else 0


# ---------------------------------------------------------------- rename
def cmd_rename(a):
    rx = re.compile(r"(?<![\w])" + re.escape(a.old) + r"(?![\w])")
    n_files = n_hits = 0
    for f in code_files():
        t = open(f, errors="surrogateescape").read()
        k = len(rx.findall(t))
        if not k:
            continue
        n_files += 1
        n_hits += k
        print("%5d  %s" % (k, os.path.relpath(f, ROOT)))
        if not a.dry_run:
            open(f, "w", errors="surrogateescape").write(rx.sub(a.new, t))
    print("%s %s -> %s: %d hit(s) in %d file(s)" % ("would rename" if a.dry_run else "renamed", a.old, a.new, n_hits, n_files))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("check-map"); s.add_argument("--map"); s.set_defaults(f=cmd_check_map)
    s = sp.add_parser("move"); s.add_argument("old"); s.add_argument("new")
    s.add_argument("--namespace", action="append", metavar="OLD=NEW", help="rename the namespace block OLD to NEW (repeatable)")
    s.add_argument("--sub-namespace", help="put the moved declarations in NEW::NAME (types that collide)")
    s.add_argument("--source", help="the header's source if it is not <stem>.cpp")
    s.add_argument("--extra-include", action="append", help="another header the forwarder includes (a split header)")
    s.set_defaults(f=cmd_move)
    s = sp.add_parser("forwarders"); s.add_argument("--remove", action="store_true"); s.set_defaults(f=cmd_forwarders)
    s = sp.add_parser("owned"); s.add_argument("batch"); s.add_argument("--base", default="feat/refactor"); s.set_defaults(f=cmd_owned)
    s = sp.add_parser("rename"); s.add_argument("old"); s.add_argument("new"); s.add_argument("--dry-run", action="store_true"); s.set_defaults(f=cmd_rename)
    a = ap.parse_args()
    sys.exit(a.f(a))


if __name__ == "__main__":
    main()
