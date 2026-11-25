#!/usr/bin/env python3
"""What already exists: search the compute functions, the registered nodes and the pipeline definitions before building anything new.

    tools/what_exists.py mexican hat            every function, node and definition whose name or description holds all the words
    tools/what_exists.py --kind node bank       only nodes (kinds: function, type, node, definition)
    tools/what_exists.py --all filterbank       no limit on the number of results (default 25)

It reads the source tree, so it is never stale: the public declarations of `compute/include` and `meeg/include` with the first sentence of their
doc comment, the nodes registered in `pipelines/src` (name, description, parameters) and the definitions in `pipelines/definitions` (their header
comment and parameters). Words match a whole word or the start of one, in any case; a name is split at `_`, `.`, `::` and camelCase, so `mexican hat`
finds `mexicanHat`. Prints `kind  name  file:line  first sentence`, the best matches first (words in the name count more than words in the text).
Run it before writing a compute function, a node, a node option or a pipeline, and name in the commit what it found (CLAUDE.md)."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER_ROOTS = ["compute/include", "meeg/include"]
DECLARATION = re.compile(r"^\s*(?:template\s*<[^>]*>\s*)?(?:(?:inline|static|constexpr|extern|virtual|explicit|friend)\s+)*"
                         r"[\w:<>,\*&\s]+?[\s\*&](~?\w+)\s*\(")
TYPE = re.compile(r"^\s*(?:enum\s+class|enum|struct|class|using)\s+(\w+)")
NOT_A_DECLARATION = re.compile(r"^\s*(return|if|for|while|switch|else|case|#|//|\*|/\*|throw|delete|new|static_assert)\b")
NAMESPACE = re.compile(r"^\s*namespace\s+([\w:]+)\s*\{")


def words_of(name: str) -> list[str]:
    """A name's words: split at non-letters and at camelCase boundaries, lowercased."""
    parts = re.split(r"[^A-Za-z0-9]+", name)
    out = []
    for p in parts:
        out += re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+", p)
    return [w.lower() for w in out if w]


def first_sentence(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    m = re.match(r"(.+?[.:;])(\s|$)", text)
    s = m.group(1) if m else text
    return s if len(s) <= limit else s[:limit - 1] + "…"


def headers():
    for root in HEADER_ROOTS:
        base = ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.h")):
            yield path


def index_headers(entries: list[dict]) -> None:
    for path in headers():
        lines = path.read_text(errors="ignore").splitlines()
        namespaces: list[str] = []
        comment: list[str] = []
        in_block = False
        depth = 0
        for number, line in enumerate(lines, 1):
            stripped = line.strip()
            if in_block:
                comment.append(stripped.rstrip("*/").lstrip("*").strip())
                if "*/" in stripped:
                    in_block = False
                continue
            if stripped.startswith("/**") or stripped.startswith("/*"):
                comment = [stripped.lstrip("/*").rstrip("*/").strip()]
                in_block = "*/" not in stripped[2:]
                continue
            if stripped.startswith("//"):
                comment.append(stripped.lstrip("/").strip())
                continue
            m = NAMESPACE.match(line)
            if m:
                namespaces.append(m.group(1))
                comment = []
                continue
            trailing = line.split("//", 1)[1].strip() if "//" in line else ""
            name, kind = None, None
            t = TYPE.match(line)
            if t and depth == 0:
                name, kind = t.group(1), "type"
            elif not NOT_A_DECLARATION.match(line) and depth == 0:
                d = DECLARATION.match(line)
                if d and d.group(1) not in ("operator",):
                    name, kind = d.group(1), "function"
            if name:
                doc = " ".join(c for c in comment if c) or trailing
                qualified = "::".join(namespaces + [name]) if namespaces else name
                entries.append({"kind": kind, "name": qualified, "file": f"{path.relative_to(ROOT)}:{number}", "doc": first_sentence(doc),
                                "text": doc + " " + trailing})
            if stripped and not stripped.startswith("}"):
                comment = []
            # struct/class bodies: their members are not indexed as free functions
            if t and t.group(0).split()[0] in ("struct", "class") and "{" in line and "};" not in line:
                depth += line.count("{") - line.count("}")
            elif depth > 0:
                depth += line.count("{") - line.count("}")
                depth = max(depth, 0)
            if stripped.startswith("}") and namespaces and depth == 0 and re.match(r"^\}\s*(//.*)?$", stripped):
                namespaces.pop()


def index_nodes(entries: list[dict]) -> None:
    for path in sorted((ROOT / "pipelines/src").glob("*.cpp")):
        text = path.read_text(errors="ignore")
        lines = text.splitlines()
        for number, line in enumerate(lines, 1):
            m = re.search(r'\b(\w+)\.name\s*=\s*"([\w.]+)"', line)
            if not m or "." not in m.group(2):
                continue
            var = m.group(1)
            window = "\n".join(lines[number - 1:number + 40])
            d = re.search(rf'\b{var}\.description\s*=\s*"((?:[^"\\]|\\.)*)"', window)
            params = " ".join(re.findall(r'"((?:[^"\\]|\\.){12,})"', window))
            doc = d.group(1) if d else ""
            entries.append({"kind": "node", "name": m.group(2), "file": f"{path.relative_to(ROOT)}:{number}", "doc": first_sentence(doc or params),
                            "text": doc + " " + params})


def index_definitions(entries: list[dict]) -> None:
    for path in sorted((ROOT / "pipelines/definitions").glob("*.yaml")):
        lines = path.read_text(errors="ignore").splitlines()
        header = " ".join(l.lstrip("#").strip() for l in lines if l.startswith("#"))
        meanings = " ".join(re.findall(r'meaning:\s*"([^"]*)"', "\n".join(lines)))
        entries.append({"kind": "definition", "name": path.stem, "file": f"{path.relative_to(ROOT)}:1", "doc": first_sentence(header),
                        "text": header + " " + meanings})


def matches(entry: dict, query: list[str]) -> int:
    name_words = words_of(entry["name"])
    text_words = re.findall(r"[a-z0-9]+", (entry["text"] + " " + entry["file"]).lower())
    score = 0
    for q in query:
        in_name = any(w.startswith(q) for w in name_words) or q in entry["name"].lower()
        in_text = any(w.startswith(q) for w in text_words)
        if not (in_name or in_text):
            return 0
        score += 3 if in_name else 1
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("words", nargs="+", help="what you are about to build, in words (all must match)")
    parser.add_argument("--kind", choices=["function", "type", "node", "definition"], help="only this kind")
    parser.add_argument("--all", action="store_true", help="every match, not the best 25")
    args = parser.parse_args()
    query = [w for word in args.words for w in words_of(word)] or [w.lower() for w in args.words]
    entries: list[dict] = []
    index_headers(entries)
    index_nodes(entries)
    index_definitions(entries)
    found = sorted(((matches(e, query), e) for e in entries if not args.kind or e["kind"] == args.kind), key=lambda p: (-p[0], p[1]["name"]))
    found = [(s, e) for s, e in found if s > 0]
    if not found:
        print(f"nothing found for: {' '.join(args.words)} (searched {len(entries)} functions, types, nodes and definitions)")
        return 1
    shown = found if args.all else found[:25]
    for _, e in shown:
        print(f"{e['kind']:<10}  {e['name']}  {e['file']}\n            {e['doc']}")
    if len(found) > len(shown):
        print(f"… {len(found) - len(shown)} more (--all)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
