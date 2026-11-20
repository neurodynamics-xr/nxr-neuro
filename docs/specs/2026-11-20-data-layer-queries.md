# nxr-neuro as the data layer: the queries clients use (spec, 2026-11-20)

Extends [2026-11-19-io-service.md](2026-11-19-io-service.md), which made every command of the command line a function in Python and JavaScript.

## 1. The roles

The roles are those of a standard database application:

| role | who | what it does |
|---|---|---|
| storage | the SQLite catalogue and the subject packages (zarr) | **private to nxr-neuro** |
| data layer | nxr-neuro, an embedded library (native in the command line and Python, WebAssembly in JavaScript) | owns the schema, migrations, integrity (types, provenance), one writer at a time, transactions (a command is one unit), and the queries |
| host | e.g. a desktop application's main process | runs the data layer, owns windows and file dialogs, puts long calls on a worker, relays requests; no data logic of its own |
| clients | an application's interface, Python analysis code, scripts | ask the data layer and show or analyse what comes back |

**The rule.** No client opens the SQLite database or the zarr stores. The interface between a client and the stored data is nxr-neuro's queries, and their
results are in nxr-neuro's typed language (the same descriptions `compile` uses). The schema and the store layout can change without breaking a client.

## 2. What a client asks

| question | command (Python `nn.<verb>.<noun>`, JavaScript `nxr.<verb>.<noun>`) | exists |
|---|---|---|
| what a subject holds, as one index | `read subject` | **new** (3.1) |
| the nodes of one class, filtered and paged | `list fields \| manifolds \| selections \| operators \| statistics \| dictionaries \| spaces \| acquisitions \| jobs` | **new** (3.2); `list recordings \| kernels \| results` exist |
| a dataset's counts | `show dataset` | exists; gains the counts per class |
| one node and its arrays, a region or a time range at a time | `read field \| manifold \| selection \| operator \| statistic` | exists |
| a reduction, filtered, ordered, limited | `query measurements` | exists |
| store what a client computed | `add field \| selection \| statistic \| measurements \| operator \| points \| line \| surface` | exists; gains 3.3–3.6 |
| compute and store | `run <pipeline>` | exists |
| change a node's description | `edit manifold \| dictionary` | **new** (3.6, 3.5) |
| remove | `delete subject --node CLASS:NAME` | changes (3.7) |

## 3. New commands and options

### 3.1 `read subject`
`read subject --dataset D --subject S [--classes manifolds,fields,selections,operators,statistics,dictionaries,spaces,acquisitions] [--related]`.

It returns every node's DESCRIPTION, with no arrays. Each class has its own description:
- **a manifold:** its kind, sizes, topology, coordinate system, and its factors when it is a product;
- **a field:** its domain, value type, components, unit, quantity and lineage;
- **a selection:** its type, the manifold it is on, its members, and its dictionary entries;
- **an operator:** its from, its to, and its kind;
- **a statistic:** its axes.

`--related` adds the subjects this one refers to, such as the dataset's default subject or a registration's source.

The descriptions are nxr-neuro's typed model: named, nested and versioned (`schema: nxr-neuro-subject@1`). They are not table rows.

### 3.2 `list <class>`
`list <class> --dataset D [--subject S] [--name TEXT] [--kind TEXT] [--on MANIFOLD] [--limit N] [--after CURSOR] [--columns a,b,...]`.
- Each row is the class's description, or the columns asked for.
- Paging uses a cursor.
- The filters are named options, never SQL.

`show dataset` gains the count of each class.

### 3.3 `add product`, `add line --coordinates`
- `add product --name N --factors A,B` makes the product manifold of stored factors, such as a line over time or a surface over time.
- `add line --coordinates FILE.npy` makes a line whose points are not evenly spaced, such as an eigenvalue axis. It takes an optional `--kinds FILE.npy` for one code per point.

### 3.4 Options on `add`
- `add field`:
  - `--by-operator NAME` records lineage: the operator that made it;
  - `--contribution SUBJECT:FIELD[:WEIGHT]`, repeatable, names the fields a mapped field is made from;
  - `--producer JSON`.
- `add selection`:
  - `--of-field NAME` names the recording an indices selection belongs to;
  - `--weights FILE.npy`;
  - `--producer JSON`.
- `add operator --layout analytic --method NAME --params JSON --from-selection NAME` adds an operator stored by its method rather than as a matrix.

### 3.5 Dictionaries
- `add dictionary --name N --scope dataset|subject --entries FILE.json`. Each entry has a code, name, label, colour, description and params.
- `edit dictionary --name N --entry CODE [--name-of TEXT] [--colour TEXT] [--description TEXT] [--params JSON]`, or `--entries FILE.json --replace`.

### 3.6 `edit manifold`
`edit manifold --name N [--primary yes|no] [--label TEXT] [--description TEXT]`. Making a surface primary clears the previous primary in the same unit.

### 3.7 Large writes in pieces, and deletes
- **`add field` in pieces.**
  - `--begin` takes the type and the full shape, and returns a write id.
  - `--append --write ID --region …` writes one region.
  - `--commit --write ID` stores the field.
  - The three steps are one unit: until it is committed, the field is not in the subject. An unfinished write is rolled back by `recover dataset`.
  - The handle (`nn.open`, `nxr.open`) carries the write id between calls. The same form serves `add statistic`.
- **Deletes.** `delete subject --node CLASS:NAME` names a node by its class, since names can repeat across classes.
  - io removes what the node owns: a manifold's geometry and topology, and the operator a field was made by.
  - It refuses a node that other nodes depend on, naming them.

### 3.8 Jobs
`run` and `import` record a job row: what ran, when, the program and the outcome. `list jobs` reads them.

## 4. Exceptions and limits

- **Reading a dataset served over HTTP** (the published demo) has no data layer yet. Until nxr-neuro can read remotely (a read-only mode that fetches
  ranges of the database and the shards), that one reader stays outside this rule.
- **The calls are synchronous.** A host puts long ones (imports, runs) on a worker. Progress comes through the `progress` callback, and a run is
  cancelled by ending the worker, after which `recover dataset` rolls the unit back.
- **A client that streams a recording into a GPU** reads it through `read field` with a time range: a copy through the data layer for each chunk, at one chunk per view.

## 5. Plan (scoped tests; synthetic data only)

1. `read subject` and `list <class>`, with `show dataset`'s counts. Test: round trips on a synthetic dataset; paging; filters; no column name of the schema in a result.
2. The `add` options (3.3, 3.4), dictionaries (3.5), `edit manifold` (3.6). Test: one round trip each. **Done (2026-11-20):** `add product`, `add line --coordinates --kinds`, the options of `add field | selection | operator`, `add dictionary`, `edit dictionary`, `edit manifold`; each option is stored in an existing column (docs/reference.md, "The options of `add`, dictionaries and `edit`"); test `test_add_options`.
3. Writes in pieces and deletes by class (3.7), jobs (3.8). Test: an unfinished write is invisible and rolled back; a refused delete names its dependants. **Done (2026-11-20):** `add field | statistic --begin | --append | --commit` (staged beside the database, stored by the one-shot command's unit; regions may not overlap), `delete subject --node CLASS:NAME` (owned arrays go, dependants refuse the delete and are named), job rows for `run` and `import` in the existing `job` table (docs/reference.md, "Writes in pieces, deletes by class and jobs"); test `test_pieces_deletes_jobs`.

Each step adds its commands to the command table, so the command line, Python and JavaScript get them together. The coverage tests already enforce this.

## Checklist (the architecture policy)

- Every row read or written, and every store access, is inside io. The faces convert arguments and results only.
- No computation is added: these are reads, writes and descriptions.
