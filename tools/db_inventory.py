#!/usr/bin/env python3
"""Inventory of database access outside io/ (the io record API spec, docs/specs/2026-11-17-io-record-api.md).

Scans pipelines/, cli/ and bindings/ (C++ sources; vendored extern/ trees skipped) and reports:
  - every SQL statement: a string literal (adjacent literals joined, raw strings read) whose text starts with an SQL verb;
  - every database call: `db.` / `db().` / `db->` followed by one of the Database methods (io/include/nxr/io/database.h);
  - every place the raw handle leaves through an argument (`f(sub.db(), ...)`, `f(db, ...)`): an io helper called with the handle.
Each SQL statement is classified into a group (what it does) and given the proposed record-API call.

Usage:  tools/db_inventory.py [--repo .] [--section summary|groups|files|statements|schema|all]
  all (the default) prints the spec's appendix A tables; statements prints one TSV line per statement; schema prints appendix B's table use.
Rerun after each migration step: the counts must fall to zero outside io/ (tests: the test-only helper).
"""
import argparse, collections, os, re, sys

DB_METHODS = "one|all|value|execute|tx|read|insert|upsert|exec_script|columns|primary_key|with_writable_schema|handle"
# SQL is written in upper case throughout the code base; a statement needs its verb and a second keyword (FROM, INTO, SET, TABLE, ...)
SQL_START = re.compile(r"^\s*(SELECT\s.*\bFROM\b|SELECT\s+(1|COUNT|EXISTS|\(|[a-z_]+\s*\()|INSERT\s+(OR\s+\w+\s+)?INTO\b|REPLACE\s+INTO\b|UPDATE\s+\w+\s|UPDATE\s*$|DELETE\s+FROM\b|WITH\s+\w+\s+AS\b|(CREATE|DROP|ALTER)\s+(TABLE|INDEX|VIEW|TRIGGER|UNIQUE)\b|PRAGMA\s|BEGIN\b|COMMIT\b|ROLLBACK\b|ATTACH\s|DETACH\s|SAVEPOINT\s|RELEASE\s|VACUUM\b)", re.S)
SQL_FRAGMENT = re.compile(r"^\s*(AND|OR|WHERE|ORDER BY|GROUP BY|LIMIT|FROM|JOIN|LEFT JOIN|SET|VALUES)\s|\b[a-z_]+ (=|IN|<|>|<=|>=) \?|\b[a-z_]+ IS (NOT )?NULL\b")   # a piece of a statement built in code
CALL = re.compile(r"\b(db\s*\(\s*\)|db|ds)\s*(?:\.|->)\s*(" + DB_METHODS + r"|query)\s*\(")   # ds.query: io's Dataset (cli/read_command.cpp)
HANDLE_ARG = re.compile(r"[(,]\s*(?:[A-Za-z_][\w]*\s*(?:\.|->)\s*)?db\s*\(\s*\)\s*[,)]|[(,]\s*db\s*[,)]")
OPEN = re.compile(r"\bDatabase::open(?:_plain)?\s*\(")
INCLUDE = re.compile(r'#\s*include\s*[<"](nxr/io/database\.h|sqlite3\.h)[>"]')
TABLE = re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+([A-Za-z_][\w]*)", re.I)


def lex(src):
    """Blank comments and string literals (keeping line structure); return (code, literals[(line, text, end_offset)])."""
    out, lits, i, n = [], [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i); j = n if j < 0 else j
            out.append(" " * (j - i)); i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2); j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j])); i = j
        elif c == "R" and src.startswith('R"', i) and (i == 0 or not (src[i-1].isalnum() or src[i-1] == "_")):
            k = src.find("(", i + 2); delim = src[i+2:k]
            j = src.find(")" + delim + '"', k); j = n if j < 0 else j + len(delim) + 2
            lits.append((src.count("\n", 0, i) + 1, src[k+1:j-len(delim)-2], i, j))
            out.append(re.sub(r"[^\n]", " ", src[i:j])); i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j += 1
            lits.append((src.count("\n", 0, i) + 1, bytes(src[i+1:j-1], "utf-8").decode("unicode_escape", "ignore"), i, j))
            out.append(" " * (j - i)); i = j
        elif c == "'" and i + 1 < n:
            j = i + 1
            while j < n and src[j] != "'" and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            out.append(" " * (j + 1 - i)); i = j + 1
        else:
            out.append(c); i += 1
    code = "".join(out)
    merged = []   # join adjacent literals ("a" "b")
    for ln, t, s, e in lits:
        if merged and code[merged[-1][3]:s].strip() == "":
            pl, pt, ps, _ = merged[-1]; merged[-1] = (pl, pt + t, ps, e)
        else:
            merged.append((ln, t, s, e))
    return code, merged


# What a statement DOES -> (group, proposed record-API call). First match wins; (verb, table regex, sql regex).
RULES = [
    # --- io internals that leaked out
    ("schema and metadata probes", r".*", r"^\s*(PRAGMA|CREATE|DROP|ALTER|BEGIN|COMMIT|ATTACH|SAVEPOINT|RELEASE)|sqlite_master|pragma_", "none: io internal (Session::open runs migrations; the schema floor retires has_table/has_column)"),
    ("storage listing: every stored array of a dataset", r".*", r"AS sp\b.*IS NOT NULL", "records.stored_arrays(Where{subject}) -> vector<StoredArray>"),
    ("generic: one row of a table named at run time", r"^$", r".", "records.read(Ref) / records.remove(Ref) (Ref = {Kind, id})"),
    # --- subjects, datasets, catalogue
    ("subjects: count nodes per subject (status)", r"^(space|topology|manifold|dictionary|selection|field|operator|subject)$", r"^\s*SELECT\s+(count|coalesce)\(", "catalogue.status(subject) (the subject_status view)"),
    ("subjects: write", r"^(dataset|subject)$", r"^\s*(INSERT|UPDATE|DELETE)", "subjects.update(row, by) / subjects.remove(id) (io writer)"),
    ("subjects: find one by name/path/id", r"^subject$", r"WHERE.*\b(name|path|id)\s*=\s*\?", "catalogue.subject(name_or_id) -> optional<SubjectRow>"),
    ("subjects: list members", r"^(subject|dataset)$", r"^\s*SELECT", "catalogue.subjects(SubjectWhere{kind, status, where}) / session.dataset()"),
    # --- dictionaries
    ("dictionaries: recipe parameters by name", r"^dictionary(_entry)?$", r"recipes|kRecipes", "recipes.get(name) -> optional<Json>"),
    ("dictionaries: entries (code -> name)", r"^dictionary(_entry)?$", r"^\s*SELECT", "dictionaries.read(id).entries / dictionaries.find(DictionaryWhere{scope, name})"),
    ("dictionaries: write", r"^dictionary(_entry)?$", r"^\s*(INSERT|UPDATE|DELETE)", "dictionaries.create(row, entries, by) / update / remove"),
    # --- measurements
    ("measurements: write", r"^selection_measurement$", r"^\s*(INSERT|REPLACE)", "measurements.create(span<const NewMeasurement>, by)"),
    ("measurements: delete", r"^selection_measurement$", r"^\s*DELETE", "measurements.remove(MeasurementWhere{selection, ofField, measure, ofStatistic})"),
    ("measurements: reduce at read (group by / aggregate)", r"^selection_measurement$", r"^\s*SELECT.*\b(SUM|AVG|MIN|MAX|TOTAL|GROUP BY)\b", "measurements.query(MeasurementQuery{selection, measure, shift, reduction, within, ...})"),
    ("measurements: provenance (computed_by, unit)", r"^selection_measurement$", r"^\s*SELECT\s+(DISTINCT\s+)?computed_by", "measurements.kinds(selection) (computed_by, unit per kind)"),
    ("measurements: what is measured (kinds, existence)", r"^selection_measurement$", r"^\s*SELECT\s+(DISTINCT\s+)?(measure|1|selection_id|within_selection_id)\b", "measurements.kinds(selection) / measurements.exists(MeasurementWhere)"),
    ("measurements: array-backed rows (array_path)", r"^selection_measurement$", r"array_path", "measurements.list(MeasurementWhere{selection, ofField, measure}) -> rows with array_path"),
    ("measurements: read rows", r"^selection_measurement$", r"^\s*SELECT", "measurements.list(MeasurementWhere)"),
    # --- spans and extents
    ("spans: read a selection's time spans", r"^selection_span$", r"^\s*SELECT", "selections.read(id).spans / spans.list(selection) -> vector<SpanRow>"),
    ("spans and extents: write", r"^selection_(span|extent)$", r"^\s*(INSERT|UPDATE|DELETE)", "selections.create/update(row, SelectionChildren{spans, extents}, by)"),
    # --- statistics
    ("statistics: find and read", r"^statistic$", r"^\s*SELECT", "statistics.list(StatisticWhere{subject, name, ofField, status}) / read(id)"),
    ("statistics: write", r"^statistic$", r"^\s*(INSERT|UPDATE|DELETE)", "statistics.create / update / remove"),
    # --- manifolds and geometry
    ("manifolds: geometry of one (grid axis, coordinate system, unit, factor lines)", r"^(geometry|geometry_dimension|coordinate_system|space|manifold)$", r"geometry_dimension|coordinate_system|geometry g\b|JOIN geometry|FROM geometry\b|space_id", "manifolds.read(id) -> ManifoldRecord{row, topology, geometry, dimensions, coordinate_system}"),
    ("manifolds: topology cells", r"^(topology|topology_cell)$", r"^\s*SELECT", "manifolds.read(id).topology.cells"),
    ("manifolds: one by id (type, name, size)", r"^manifold$", r"^\s*SELECT.*WHERE\s+(m\.)?id\s*=\s*\?", "manifolds.read(id)"),
    ("manifolds: find by name/type on a subject", r"^manifold$", r"^\s*SELECT", "manifolds.list(ManifoldWhere{subject, type, name, componentOf, status})"),
    ("manifolds and acquisitions: write", r"^(manifold|geometry|geometry_dimension|topology|topology_cell|space|coordinate_system|acquisition)$", r"^\s*(INSERT|UPDATE|DELETE)", "manifolds.create(row, ManifoldChildren, by) / update / remove"),
    ("acquisitions: label", r"^acquisition$", r"^\s*SELECT", "acquisitions.read(id)"),
    # --- fields
    ("fields: one by id", r"^(field|field_contribution)$", r"^\s*SELECT.*\bWHERE\s+(f\.)?id\s*=\s*\?", "fields.read(id)"),
    ("fields: virtual or pending (path IS NULL)", r"^field$", r"path IS NULL", "fields.list(FieldWhere{subject, name, stored=false})"),
    ("fields: on a manifold", r"^field$", r"manifold_id\s*=\s*\?|d\.manifold_id", "fields.list(FieldWhere{subject, manifold, status})"),
    ("fields: find by name (and kind) on a subject", r"^(field|field_contribution)$", r"^\s*SELECT", "fields.find(FieldWhere{subject, name, kind}) -> optional<FieldRow>"),
    ("fields: write", r"^(field|field_contribution)$", r"^\s*(INSERT|UPDATE|DELETE)", "fields.create(row, by, FieldChildren) / update / remove"),
    # --- operators
    ("operators: find by kind (and name, target)", r"^operator(_factor)?$", r"^\s*SELECT", "operators.find / list(OperatorWhere{subject, kind, name, toManifold, stored})"),
    ("operators: write", r"^operator(_factor)?$", r"^\s*(INSERT|UPDATE|DELETE)", "operators.create(row, by, OperatorChildren) / update / remove"),
    # --- selections
    ("selections: members of a partition (member_of_id)", r"^selection$", r"member_of_id\s*=\s*\?|JOIN selection p ON p\.id = d\.member_of_id", "selections.members(partition_id[, ordinal]) -> vector<SelectionRow>"),
    ("selections: found by their parameters (json_extract, LIKE)", r"^selection$", r"json_extract|params_json LIKE", "selections.list(SelectionWhere{..., params}) (open question 3)"),
    ("selections: of a field", r"^selection$", r"of_field_id\s*=", "selections.list(SelectionWhere{subject, ofField, name})"),
    ("selections: labellings and partitions by name/type", r"^selection(_element|_extent)?$", r"^\s*SELECT", "selections.find / list(SelectionWhere{subject, manifold, type, name, stored, status})"),
    ("selections: write (params, rows)", r"^selection(_element|_extent)?$", r"^\s*(INSERT|UPDATE|DELETE)", "selections.create(row, children, by) / update / remove"),
    # --- jobs
    ("jobs", r"^job$", r".", "jobs.read / list(JobWhere) / create / update / remove"),
    ("other", r".*", r".", "(review by hand)"),
]


def classify(sql, tables):
    verb = sql.strip().split(None, 1)[0].upper() if sql.strip() else ""
    main = tables[0] if tables else ""
    if verb in ("UPDATE", "INSERT", "REPLACE", "DELETE"):
        m = re.search(r"\b(?:INTO|UPDATE|FROM)\s+([A-Za-z_]\w*)", sql, re.I); main = m.group(1) if m else main
    for group, treg, sreg, api in RULES:
        if re.search(treg, main or "", re.I) and re.search(sreg, sql, re.I | re.S):
            return group, api, main
    return "other", "(review by hand)", main


def area(path):
    p = path.replace(os.sep, "/")
    if "/test/" in p or p.startswith("tests/") or "smoke" in p:
        return "tests"
    return p.split("/")[0]


def scan(repo):
    roots = ["pipelines", "cli", "bindings"]
    stmts, calls, handles, opens, incs = [], collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    method_counts = collections.Counter(); frags = collections.Counter(); reads = collections.Counter()
    for r in roots:
        for d, dirs, files in os.walk(os.path.join(repo, r)):
            dirs[:] = [x for x in dirs if x not in ("extern", "node_modules", "build", ".git")]
            for f in sorted(files):
                if not f.endswith((".cpp", ".h", ".hpp", ".cc")):
                    continue
                full = os.path.join(d, f); rel = os.path.relpath(full, repo)
                src = open(full, encoding="utf-8", errors="replace").read()
                code, lits = lex(src)
                for ln, t, s, e in lits:
                    if SQL_START.match(t):
                        sql = " ".join(t.split())
                        tables = [x for x in TABLE.findall(sql)]
                        g, api, main = classify(sql, tables)
                        stmts.append(dict(file=rel, line=ln, sql=sql, group=g, api=api, table=main, area=area(rel)))
                    elif SQL_FRAGMENT.search(t):
                        frags[rel] += 1
                for m in CALL.finditer(code):
                    calls[rel] += 1; method_counts[(area(rel), m.group(2))] += 1
                    if m.group(2) == "read":   # db.read("table", id): a row by id (code keeps the source's offsets)
                        t = re.match(r'\s*"(\w+)"', src[m.end():])
                        reads[(area(rel), t.group(1) if t else "(named at run time)")] += 1
                handles[rel] += len(HANDLE_ARG.findall(code))
                opens[rel] += len(OPEN.findall(code))
                incs[rel] += len(INCLUDE.findall(src))
    return stmts, calls, handles, opens, incs, method_counts, frags, reads


def schema_use(repo):
    """Which model.sql tables and columns code names: a table in an SQL literal (FROM/JOIN/INTO/UPDATE/TABLE t) or as a whole literal "t";
    a column as a word inside any string literal (SQL text, Row column names, Cols keys). model.sql and its generated views are not code."""
    sql = open(os.path.join(repo, "io/schema/model.sql")).read()
    tables = collections.OrderedDict()
    for m in re.finditer(r"CREATE TABLE (\w+) \((.*?)\n\)( STRICT)?;", sql, re.S):
        tables[m.group(1)] = [mm.group(1) for line in m.group(2).split("\n")
                              for mm in [re.match(r"\s*([a-z_][a-z_0-9]*)\s+(TEXT|INTEGER|REAL|BLOB)", line.split("--")[0])] if mm]
    areas = {"io": ["io/src", "io/include", "io/convert"], "pipelines": ["pipelines/src", "pipelines/include"], "cli": ["cli"],
             "bindings": ["bindings/wasm-io/src", "bindings/node", "bindings/mex/src"], "tests": ["pipelines/test", "io/test", "cli/test"]}
    lits = collections.defaultdict(list)
    for a, ds in areas.items():
        for d in ds:
            for r, dd, ff in os.walk(os.path.join(repo, d)):
                dd[:] = [x for x in dd if x not in ("extern", "node_modules")]
                for f in ff:
                    t = open(os.path.join(r, f), encoding="utf-8", errors="replace").read()
                    if f.endswith((".cpp", ".h", ".hpp", ".cc")):
                        lits[a] += [x for _, x, _, _ in lex(t)[1]]
                    elif f.endswith((".mjs", ".js", ".ts")):
                        lits[a] += re.findall(r"[\"'`]([^\"'`\n]*)[\"'`]", t)
    def tuse(t, a):
        rx = re.compile(r"\b(FROM|JOIN|INTO|UPDATE|TABLE)\s+" + t + r"\b")
        return sum(1 for x in lits[a] if rx.search(x) or x == t)
    def cuse(c, a):
        rx = re.compile(r"\b" + c + r"\b")
        return sum(1 for x in lits[a] if rx.search(x))
    W = sys.stdout.write
    W("### Tables named in code (literals naming the table)\n\n| table | " + " | ".join(areas) + " |\n|---|" + "---:|" * len(areas) + "\n")
    for t in tables:
        W(f"| `{t}` | " + " | ".join(str(tuse(t, a)) for a in areas) + " |\n")
    W("\n### Columns named by at most two literals outside tests\n\n")
    for t, cols in tables.items():
        z = []
        for c in cols:
            n = {a: cuse(c, a) for a in areas}
            if sum(v for a, v in n.items() if a != "tests") <= 2:
                z.append(f"`{c}` (" + ", ".join(f"{a} {v}" for a, v in n.items() if v) + ")")
        if z:
            W(f"- `{t}`: " + "; ".join(z) + "\n")
    W("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."))
    ap.add_argument("--write-spec", metavar="SPEC", help="fill the spec's GENERATED blocks (appendix A: all sections; appendix B: schema) in place")
    ap.add_argument("--section", default="all", choices=["summary", "groups", "files", "statements", "schema", "all"])
    a = ap.parse_args()
    if a.write_spec:
        import contextlib, io as _io
        def run(section):
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = [sys.argv[0], "--repo", a.repo, "--section", section]; main()
            return buf.getvalue()
        text = open(a.write_spec, encoding="utf-8").read()
        for tag, section in (("tools/db_inventory.py", "all"), ("schema", "schema")):
            b, e = f"<!-- BEGIN GENERATED: {tag} -->\n", f"<!-- END GENERATED: {tag} -->"
            i, j = text.index(b) + len(b), text.index(e)
            text = text[:i] + run(section) + text[j:]
        open(a.write_spec, "w", encoding="utf-8").write(text)
        return
    if a.section == "schema":
        return schema_use(os.path.abspath(a.repo))
    stmts, calls, handles, opens, incs, mc, frags, reads = scan(os.path.abspath(a.repo))
    files = sorted(set(calls) | set(handles) | set(opens) | set(incs) | {s["file"] for s in stmts})
    files = sorted(set(files) | set(frags))
    files = [f for f in files if frags[f] or calls[f] or handles[f] or opens[f] or incs[f] or any(s["file"] == f for s in stmts)]
    by_area = collections.defaultdict(lambda: collections.Counter())
    for f in files:
        A = area(f); c = by_area[A]
        c["files"] += 1; c["calls"] += calls[f]; c["handles"] += handles[f]; c["opens"] += opens[f]; c["includes"] += incs[f]; c["fragments"] += frags[f]
    for s in stmts:
        by_area[s["area"]]["statements"] += 1
    W = sys.stdout.write
    if a.section in ("summary", "all"):
        W("### Totals by area\n\n| area | files | SQL statements | SQL fragments | database calls | handle passed to a helper | `Database::open` | includes `database.h`/`sqlite3.h` |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for A in ("pipelines", "cli", "bindings", "tests"):
            c = by_area[A]
            W(f"| {A} | {c['files']} | {c['statements']} | {c['fragments']} | {c['calls']} | {c['handles']} | {c['opens']} | {c['includes']} |\n")
        ms = [m for m in (DB_METHODS + "|query").split("|") if any(mc[(A, m)] for A in by_area)]
        W("\n### Database calls by method\n\n| area | " + " | ".join(ms) + " |\n|---|" + "---:|" * len(ms) + "\n")
        for A in ("pipelines", "cli", "bindings", "tests"):
            W(f"| {A} | " + " | ".join(str(mc[(A, m)]) for m in ms) + " |\n")
        W("\n### `db.read(table, id)` (a row by id, no SQL text) by table\n\n| table | pipelines | cli | bindings | tests | proposed API call |\n|---|---:|---:|---:|---:|---|\n")
        for t in sorted({t for _, t in reads}):
            W(f"| `{t}` | " + " | ".join(str(reads[(A, t)]) for A in ("pipelines", "cli", "bindings", "tests")) + f" | {t}s.read(id) |\n")
        W("\n")
    if a.section in ("groups", "all"):
        W("### SQL statements by what they do\n\n| group | pipelines | cli | bindings | tests | proposed API call |\n|---|---:|---:|---:|---:|---|\n")
        gc = collections.defaultdict(collections.Counter); api = {}
        for s in stmts:
            gc[s["group"]][s["area"]] += 1; api[s["group"]] = s["api"]
        for g, *_ in RULES:
            if g in gc and g not in api.get("_seen", ()):
                c = gc[g]; W(f"| {g} | {c['pipelines']} | {c['cli']} | {c['bindings']} | {c['tests']} | {api[g]} |\n")
                api.setdefault("_seen", set()).add(g)
        tc = collections.Counter(s["table"] for s in stmts if s["area"] != "tests")
        W("\n### Tables named by statements outside tests\n\n" + ", ".join(f"`{t or '(named at run time)'}` {n}" for t, n in tc.most_common()) + "\n\n")
    if a.section in ("files", "all"):
        W("### File -> statements -> groups (each group's API call: the groups table above)\n\n| file | SQL | fragments | calls | handle args | groups (count) |\n|---|---:|---:|---:|---:|---|\n")
        for f in files:
            ss = [s for s in stmts if s["file"] == f]
            gs = collections.Counter(s["group"] for s in ss)
            apis = "; ".join(f"{g} ({n})" for g, n in gs.most_common())
            W(f"| `{f}` | {len(ss)} | {frags[f]} | {calls[f]} | {handles[f]} | {apis or '(includes or handle only)'} |\n")
        W("\n")
    if a.section in ("statements",):
        W("file\tline\tgroup\ttable\tapi\tsql\n")
        for s in stmts:
            W(f"{s['file']}\t{s['line']}\t{s['group']}\t{s['table']}\t{s['api']}\t{s['sql']}\n")


if __name__ == "__main__":
    main()
