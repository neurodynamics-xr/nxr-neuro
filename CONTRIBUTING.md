# Contributing to nxr-neuro

Welcome, and thank you for contributing. Read `ARCHITECTURE.md` first: it is the architecture every change follows, and a change that breaks it is not
accepted. `docs/reference.md` describes every verb and method, `docs/specs/` the designs behind them. Contributions are made under the project licence
(BSD-3-Clause, see `LICENSE`); there is no CLA and no sign-off requirement.

## The method rule (how every computation is written; inside the paramount architecture)

1. **Each method has a named reference, and the default reproduces it.** The references are the published methods and established external
   implementations (Brainstorm, MNE-Python, scipy). The project's earlier implementations (nxr in Python, Rheome in MATLAB, the Cortical Flow Studio in
   TypeScript) are not references: their existing goldens stay as regression checks until replaced, and no new method takes one of them as its reference. `docs/reference.md` says which one each method follows. The choice of the default reference is the maintainers';
   record it with its date in the method's section of `docs/reference.md`.
2. **Alternatives are named options, never defaults.**
   - An option has the same name in the library's options struct and on the command line.
   - When a default changes, the old behaviour becomes a named option. No silent aliases.
3. **Every option goes into the provenance and the cache key.**
   - The stored result's `producer_json` records every option, and the cache key includes it, so a different option gives a different cached result.
   - A cache hit must be byte-for-byte the result of the same options.
4. **Parity is proven by tests, not claimed.**
   - Each port has a golden test against its reference (`*/test/golden/`, each golden next to its generator `gen_*.py|.m|.mts`) or a pytest in
     `*/test/parity/`.
   - Use synthetic data only (spheres, lattices, generated signals). Never use a real dataset in a test or a golden.
   - **Real-data checks** are optional and by hand. A real recording is never committed, never a ctest and never a golden. Point `NXR_FIXTURE_DATASET` (an nxr dataset) and, for the Brainstorm side, `NXR_FIXTURE_PROTOCOL` (a Brainstorm protocol) at your own data; checks that need them are skipped, with a reason, when the variables are unset. Nothing measured on such data goes into the repository.
5. **No Brainstorm code (GPLv3).** Write the maths from the papers. Brainstorm's outputs are only reference goldens.
6. **Recordings are streamed, and the span is always the caller's.** Two rules:
   - **Streaming.** A recording is never loaded whole. The engine always computes one chunk at a time through io's chunk service, with bounded memory, whatever
     the span. A CHUNK is 2^15 samples of a recording's time axis, every channel, and is a loading unit only: chunk k is samples [k·2^15, min((k+1)·2^15, n));
     the last is partial. A shard (2^17, four chunks) is storage, never a chunk. Recordings can be hours long.
   - **The span is the caller's.** A computation states its time range (in seconds) or the whole recording explicitly. There is no default span. Chunks are
     io's internal unit and are never something a user chooses (a developer or test option may exist, hidden from users).
   - A chunk is NOT the FFT window. Within a chunk the computation chooses its own windows: a Welch segment's length, taper and overlap set the frequency
     resolution and are free options, and segments are carried across chunk edges.
   - A pipeline reads a recording chunk by chunk, and its result must not depend on the chunk size (test it).
   - State the memory held in the options struct's comment.
7. **State what the instrument cannot resolve.**
   - A method's limits go into `producer_json.warnings` and the method's section of `docs/reference.md` (e.g. optical flow: kinematic, aperture-limited, per frame).
   - Do not claim what the measurement cannot show.

## Where a compute function goes, and what it is called (the toolboxes)

- A compute function lives in the toolbox of what it computes: `compute/include/nxr/<toolbox>/<header>.h` with its source in `compute/src/<toolbox>/`, namespace `nxr::<toolbox>`, flat in it (`nxr::signal::fft`).
  A type whose name two headers of one toolbox share goes in a sub-namespace named after its header. No header at the top of `compute/include/nxr/`; there is no umbrella header.
- Includes between toolboxes follow the table in ARCHITECTURE.md ("The toolboxes"). The test `layering_toolboxes_follow_the_dag` fails on a new include against the table.
  The few that predate the cut are counted in `cmake/toolboxes_allow.txt`; a count may only fall.
- A toolbox's sources, per-file flags and tests are in `compute/cmake/<toolbox>.cmake`.
- **One word per thing across the layers.** The C++ function (`nxr::signal::lowpass`), the registered node (`signal.lowpass`: toolbox, dot, the function's word) and every binding entry
  (MEX `compute.signal.lowpass`, the MATLAB package `+nxr/+compute/+signal/lowpass.m`, the Python module `nxr_neuro.compute.signal.lowpass` and its native `signal_lowpass`) use the same word.
  Add each new entry to `pipelines/test/names.tsv`; the test `test_names_across_layers` fails on a mismatch, on an entry missing from the table and on a table row the sources do not have.
  A renamed entry is refused under its old name with the new name in the message (`retired` rows), never kept as an alias.
- **Every function a user would call is bound.** Every function a user would call on data (a transform and its inverse, a bank's design, a detector, a fit, a measure, a statistic) is bound in every face: as a registered node, or directly while the generated bindings are not built. Only implementation helpers stay internal, each listed by name in the coverage test's exception list. A binding uses the C++ word verbatim (camelCase), in every face.

## Adding a node or a pipeline (inside the paramount architecture and its decided structure)

Every pipeline is a YAML definition of registered nodes (`ARCHITECTURE.md`, "Decided structure"). Do not write a new hand-wired `<x>_compute` C++
pipeline, and do not extend an existing one: it is being converted into a definition.

0. **First, the graph:** write the pipeline as its graph (nodes, typed ports, the inputs it is handed, the outputs it gives back) and place every
   piece in its part (loading · preparation · graph · readback). If any piece does not fit, stop and ask the maintainers.
1. **The maths** goes in `compute/` (or `meeg/`) as pure functions (matrices in, matrices out), with a unit test in `compute/test/test_<topic>.cpp`.
2. **The node:** register each function as a node with its typed ports, its parameters (one comment each: meaning, default, unit) and its traits
   (time behaviour, precision). It takes arrays and parameters only: no database, no file, no lookup of its own inputs (those are bound by the runner).
   A constant it needs beyond its inputs (a surface, a stored statistic, a manifold's type) is asked of `Bound` (`pipelines/include/nxr/pipelines/bound.h`), which the
   runner implements: a node file includes no io header and holds no subject (the test `layering_nodes_without_io` fails otherwise).
   A node made of other nodes is a composite (a graph wrapped as a node).
3. **The pipeline:** a definition in `pipelines/definitions/<name>.yaml`: inputs (found by kind and relation, never by path), parameters (meaning,
   default, unit), steps wired port to port, outputs. Stored names and kinds follow the way the importer or nxr stores the same thing.
4. **Data access:** only through io's task API (find, open, commit, cache, measurements). No SQL, no `database.h`, no `nxr/io/records/` header outside
   io; the test `layering_sql_only_in_io` fails otherwise.
5. **The command line and the bindings** need no new code for a pipeline (`run <name>`) or a node (`compute <node>`): help and the bindings' functions
   come from the registry and the definition. A binding or the command line never holds maths, SQL or logic.
6. **Tests:** the node's unit test; the pipeline on synthetic data (its result independent of the chunk size); goldens against the method's reference.
7. **Docs:** `docs/reference.md`: the method's section (the why, the default and its reference, what is stored, what it cannot resolve).
   **A node or pipeline is not done until `docs/reference.md` lists it.**
8. Larger designs first get a spec in `docs/specs/YYYY-MM-DD-<topic>.md`. A read-only comparison against the references goes in
   `docs/audits/YYYY-MM-DD-<topic>.md`.

## Quantities, never level indices (inside the paramount architecture)

Every selection parameter of a definition, preset or command is a QUANTITY: a number (or range, open-ended as `>=0.4 s`) with a unit in a named chart of the space it lies in (`8-12 Hz`,
`2-5 s`, `20 mm`, `wavelength 40 mm`, `parts 16`, `area 100-400 mm2`). A definition declares `unit` and `chart` for such a parameter; a bare level index is refused outside
the explicit `index` chart (`level index:5`), which is for diagnostics and tests. Grid levels are resolved from the quantity in closed form (`level_of_duration`,
`level_of_diameter`, `level_of_area`, `level_of_wavelength`, `level_of_parts`, `time_level_of_band`), and the run records the stated quantity, its chart, the resolved level
and the snapped quantity. Method constants (a frame density, an origin, a cycle count) are options with their defaults written in the definition, never hard-wired in
compute. Old names are not kept as aliases: a retired name is refused with the name of what replaces it.

## Converting a hand-wired verb into a definition (the recipe)

`<x>_compute` verbs are being converted one at a time; `psd` (`pipelines/definitions/psd.yaml`, `rotary_psd.yaml`; nodes in `pipelines/src/welch_steps.cpp`; `test_psd_golden`, `test_psd_pipeline`) is the worked example. A conversion
is a PR of its own, on the latest main, with these parts in this order. Nothing in it needs a new mechanism: if it does, stop and ask.

1. **Read the verb for what it stores, not how.** List every thing `<x>_compute` writes (Field, statistic, Operator, Selection, line or other manifold, measurement rows) and, for each, what a later reader asks of it: the stored
   names and paths, the axes, the lines and partitions it names, and every `producer_json` key a reader reads (grep the readers: `grep -rn 'producer\.value\|producer\[' pipelines/src`). These stay as they are.
2. **Outputs.** Declare each one in the definition's `outputs:` with its kind (`field`, `statistic`, `operator`, `selection`, `manifold`: `docs/reference.md`, "Declared outputs and presets"): the found input it is of or lies on
   (a link: `recording.manifold.0`, `recording.time`), its name as a template (the verb's default name, `$name`), the names of its line and partition. What the values ARE (measure, method, axes with sizes and a closed-form
   line, the segment tiling, the scaling record, the unit, the options it resolved, the `producer` keys readers ask for) goes in the node's `describe`, never in the definition.
3. **Nodes.** Move each piece of maths the verb holds into `compute/` as a pure function (matrices and parameters in, matrices out) with a unit test and a golden against its reference, then register it as a node
   (grouped name, e.g. `signal.psd`; a file `pipelines/src/<topic>_steps.cpp`, which the layering test `layering_nodes_without_io` scans): ports (streamed input, optional `exclude` for the bad segments), parameters (each with
   meaning, default and unit), traits (time behaviour: an accumulate node over segments carried across blocks is `stateful`; precision), `plan` on types only (its refusals say the port and the reason; `Planned::align` when
   the span must snap to a grid), `prepare`, `start`/`step`/`finish` (accumulate) or `chunk` (pointwise). A node reads no database, file or id: what it needs of a stored operator comes as an array from the runner. A node
   chooses nothing by itself: a mode that changes the graph is another definition, not a switch.
4. **Parameters and presets.** The verb's options become the definition's `parameters:` (long form: meaning, default, unit, choices) and the node's `with:`; each named option set the verb offered (`--set NAME`) becomes a
   preset that names only declared parameters (defaults, then the preset, then the caller's own value). Old flags map to parameters (`--no-x` to `--x ignore`, a pair of flags to a boolean), recorded in the PR.
5. **Delete the hand-wired code**: the `<x>_compute` function and its options and result types, its CLI parser and its entry in `cli/command_table.cpp` (the definition's command `run <x>` is made from the definition), its
   private helpers that nothing else uses. Keep the old spelling refused, naming the new one (`converted_spellings()` in `cli/registry.cpp`). Tests that only used `<x>_compute` to make a fixture call the runner through a helper
   (`pipelines/test/psd_run.h`).
6. **Tests.** Compare with the REFERENCES (the method's goldens: scipy, MNE, Brainstorm; the existing nxr and Rheome goldens), not with the old output: one run per preset and parameter set against the golden, to the stored precision; the result the same to
   the last bit for another storage chunk width; the stored record (axes, lines, partition, scaling, unit, provenance, a cache hit on a rerun, a refused name and `--replace`); the refusals (each names the parameter or port).
   Build only the targets you touch and run only their tests (`tools/tests.py`), never the full suite.
7. **Docs.** `docs/reference.md`: the method's section says it is a definition, lists its parameters and presets, and what is stored; the command table loses the hand-wired row.

## Adding a file format

Pure readers and writers go in `io/formats/` (bytes in, objects out; no store, no database; the layering check enforces it). Recordings are read in
blocks by sample range, so the chunk service can stream them. Parse every header field; keep the file's units and convert to SI. Goldens are synthetic
files written by the reference (MNE-Python, nibabel), each next to its generator; never a real recording. A new dependency needs a compatible licence,
an entry in `THIRD_PARTY_NOTICES.md` and the maintainers' OK.

## Issues: how problems are tracked

Every problem found and not fixed in the same change is a GitHub issue, so nothing lives only in a report, a pull-request description or a chat.
- **Bug** (template `Bug`, label `bug`): wrong behaviour, a wrong refusal, or a page that documents what the code does not do. **Task** (template `Task`, label `task`): known work left (a test to port, a lost check, a missing command).
- One issue per problem; the title says what is wrong in plain words; the body has the smallest reproduction on synthetic data. Add the area labels that apply (`io`, `pipelines`, `cli`, `bindings`, `compute`, `docs`, `tests`, `ci`).
- Never put real data, dataset names, subject identifiers or private paths in an issue: the repository is public.
- A pull request that fixes an issue says `Fixes #N`; a pull request that leaves known work behind opens its issues and links them under *Known*.
- Before starting work, look at the open issues of the area (`gh issue list --label <area>`); a duplicate is closed with a link to the first.

## Test kinds

Every ctest test has exactly one kind, listed in `tests/kinds.tsv` (`name<TAB>kind[<TAB>slow]`); the label is set by `cmake/test_kinds.cmake`, and a test missing from the table fails the configure.

| kind | what it is for |
|---|---|
| `unit` | one function or class on data made in the test (`compute/`, `meeg/`, `io/`, `io/formats/`, the graph executor in `pipelines/graph/`) |
| `golden` | a result against a stored reference: tests that read `*/test/golden`, and the `*golden*` and `*parity*` tests |
| `integration` | a pipeline or the command run end to end (`pipelines/` definition and runner tests, `cli_*`) |
| `guard` | a rule of the repository checked on its sources (`layering_*`, `test_names_across_layers`, the `check_*.cmake` scripts) |

- **Integration tests are small:** one synthetic subject, one chunk, one band, seconds. They check the wiring (the right nodes, in the right order, storing the right
  names); the maths lives in unit and golden tests. A test over about 5 s is also labelled `slow` (CI runs those on main only).
- **A new test is added to `tests/kinds.tsv` with its kind.**
- **A test checks behaviour:** values, shapes, refusals with their reason. Never the internal layout of a provenance record (`producer_json`, `params_json`
  options), log text or private names; a binding test checks the binding (a call, its result, names and errors), not the library again.
- **No refactor gates in the suite:** a comparison of new code against the old code it replaces (a golden recorded from the old code, the old function
  copied into the test) is run during the refactor and removed with it; only references (scipy, MNE, Brainstorm) and the existing nxr, Rheome and Studio goldens stay.
- **Performance is measured in `bench/`** (`NXR_BUILD_BENCH`, off by default; `tools/bench.py` runs it and compares with a baseline), never in a test.
- **A new pipeline:** one integration test, plus a unit test for each new node.
- Select by kind: `tools/tests.py status --kind unit --fast REGEX` (`--kind` is repeatable, `--fast` leaves out `slow`).

## Build and test

CMake 3.20+, Ninja, C++17. Clone with `--recurse-submodules` (geometry-central). Large build trees may live on another disk: make `build/` a link to it before the first configure. The SQLite engine is always the pinned amalgamation (fetched at configure time, compiled with the same options on every platform), never the system library, so a platform SQLite older than the SQL the code uses cannot break a build; zstd and zlib come from the system unless `NXR_NEURO_PORTABLE` is set.

    cmake -S . -B build -G Ninja                         # configure once
    cmake --build build --target <the targets you changed> 2>&1 | tail -20
    tools/tests.py status REGEX         # lists green, stale, failed, new or NOT BUILT; builds nothing and runs nothing (seconds)
    tools/tests.py build REGEX          # the ONLY command that compiles: the matching tests' targets (ask first: no build without the maintainers' OK)
    tools/tests.py run REGEX            # runs the stale, failed and new ones among them and records them; never builds: refuses tests not built
    tools/tests.py run '^(layering_.*|test_names_across_layers|<your tests>)$'   # ALWAYS with the guards: seconds, and they catch what CI finds after minutes
    tools/tests.py show TEST            # a test's last result and what it printed
    tools/tests.py run --suite          # the full build and suite: once, at the end of a round, never during development

**No waste: this is a hard rule, not a preference.** Building and testing cost real time and compute; a brute-force loop is a defect in how
the work was done, whatever the change.
- **Never build everything to test something.** Build the targets you changed (`--target`), never `all`. `tools/tests.py` requires a REGEX
  and builds only the executables of the tests it names (and the `nxr-neuro` command, for a test that calls it).
- **Never rerun a green test.** The record knows what each test ran; an unchanged test is not run again. Do not bypass the record with bare
  `ctest` to dodge a build: the record would no longer be true and the next run repeats your work.
- **Edit, then build once, then test once.** Batch the edits of one item, then one targeted build and one scoped run. Do not iterate
  compile → run → compile over a whole library for each small edit; read the error, fix every instance, then build again.
- **Slow tests are labelled `slow`** (`set_tests_properties(<test> PROPERTIES LABELS slow)`) and run only when the change reaches them.
- **Header-wide renames go first and alone.** A change to a header included everywhere invalidates every build that includes it; land it
  before parallel work starts, never alongside it.
- **The full suite and the full build run once, at the end of a round** (`tools/tests.py run --suite`), when the maintainers say so.

- **The test record.** Results are kept in `test-results/` (in the repo, never in git): `status.json` holds every test's latest result
  and the FINGERPRINT of what it ran (its binary, the `nxr-neuro` command it calls, its library's goldens and data); `runs/` holds each run. A test
  whose fingerprint is unchanged since it passed is still green and is not run again. Run tests through `tools/tests.py`, not bare `ctest`, so the
  record stays true and no green test runs twice.
- **The full suite runs at the end of a development round, when the maintainers say so,** not during it. During development: `status`,
  then `run` on what the change reaches.
- **Continuous integration** (`.github/workflows/ci.yml`; selection by `tools/ci_select.py`, build targets by `tools/tests.py targets`): CI builds only what it tests.
  - **CI runs at milestones only.** Work lands on the integration branch `develop` with scoped local tests and runs no CI there; `main` is updated
    from `develop` by one pull request at a milestone, and that pull request runs CI on Linux and macOS, with the bindings' workflows when their binding
    changed. Branches merged into `develop` open no pull request and wait on no CI.
  - **Milestone pull requests and pushes to main (Linux and macOS)** run the same scoped selection, slow tests left out (a push to main repeats its pull
    request's selection, which already passed, and fills the compiler cache the next milestone starts from). A changed test source (`*/test/<name>.cpp`) selects only that test; a changed library source selects that
    library's tests (every kind) plus the **integration** tests of the libraries that depend on it (not their unit or golden tests); the guards
    (`^layering_`, kind `guard`) always run. Libraries (labels): `compute`, `meeg`, `io_formats`, `io`, `pipelines_graph`, `pipelines`, `cli`.
    A shared file (top-level CMake, `cmake/`, the CI workflow) selects everything; a change to `tools/` selects the guards only (except `tools/gen_record_rows.py`,
    which runs the io tests). Only the selected tests' executables are built (plus `nxr-neuro` for tests that call it); `all` is built only for a scope of all.
    Documentation-only pull requests run nothing.
  - **The full suite** (every test, slow ones included, Linux and macOS) runs **only nightly**, on version tags and on demand, and the nightly run first checks
    whether main changed since the last successful nightly run and does nothing if not.
  - **The bindings** (`matlab.yml`, `wheels.yml`, `wasm.yml`) run only in a milestone pull request that changes their binding (and on tags and on demand), build only the binding
    target and the libraries it links, and have no nightly run.
  - A test that takes minutes is `slow` in `tests/kinds.tsv`.

- **Test only what the change touches.**
  - A small change runs the tests of the function it changed: `tools/tests.py run '<the test>|^layering_'`, or `pytest file::test`.
  - Then stop. Do not sweep for every test that mentions the topic, and do not rerun the whole C++ or parity suite.
  - Run the full suites only for a change across several areas or a shared core (a reader/writer, the schema, a header used everywhere), and say why.
  - The same applies in the sibling repositories.
- **Keep output quiet:** run scoped tests and print failures, not passes.
- **Parity pytests** run with nxr's environment: `uv run --project $NXR python -m pytest -q ...` with `--basetemp` on a scratch disk. The
  environment variables are in the *Tests* section of `docs/reference.md`.
- **MATLAB golden generators and anything heavy** run one at a time and are optional: the committed goldens are what the tests read. A generator finds its reference tools through environment variables (`BRAINSTORM3_HOME`, `RHEOME_HOME`, `NXR_CORTICAL_FLOW`) and says which it needs. Runs on real datasets are by hand, never in a test.

## Cutting a release

A release is a version tag; nothing is built by hand.

1. Set the version in `CMakeLists.txt` (`project(nxr_neuro VERSION X.Y.Z ...)`) and merge that through a pull request.
2. Tag the merge commit and push the tag: `git tag vX.Y.Z && git push origin vX.Y.Z`. The tag must match the version in `CMakeLists.txt`, or the build stops.
3. `.github/workflows/release.yml` then builds the `nxr-neuro` command twice and checks each build:
   - **Linux x86_64**, in the `manylinux_2_28` container (glibc 2.28 baseline), with SQLite (amalgamation), zstd (c-blosc's copy), zlib and libstdc++/libgcc linked
     statically (`-DNXR_NEURO_PORTABLE=ON`). The job fails if `ldd` shows anything but libc, libm, libpthread, libdl and the loader, or if the binary needs a glibc
     newer than 2.28; a second job runs `--version` and the smoke test in a plain `ubuntu:20.04` container.
   - **macOS arm64** on `macos-15`, with no library outside `/usr/lib` and `/System` in `otool -L`.
   - The smoke test (`.github/smoke/smoke.sh`) uses synthetic data only: it creates a dataset, a subject and an icosphere surface with the binary itself and
     computes the surface's eigenmodes. There is no command that makes a synthetic recording, so no `--from/--to` computation is part of it.
4. The `publish` job creates the GitHub release with `nxr-neuro-<version>-linux-x86_64.tar.gz`, `nxr-neuro-<version>-macos-arm64.tar.gz` and `SHA256SUMS`. Each
   tarball holds the binary, `LICENSE`, `THIRD_PARTY_NOTICES.md` and a short `README` (`.github/release/README.txt`).

To try the builds without a tag, run the workflow by hand (Actions, release, Run workflow, or `gh workflow run release.yml --ref <branch>`): it builds, checks and
uploads the tarballs as workflow artefacts and makes no release. Users install from a release as described in the README (*Install a release*). The macOS binary is
not signed or notarised yet; that needs an Apple developer account and is not done.

## Code style

- **Match the code around you:**
  - two-space indent, braces on the same line
  - long lines are fine (up to ~200)
  - `snake_case` for pipeline functions and option fields; `camelCase` in `compute/` where the code already uses it (e.g. `brainstormPsdOptions`, `star1Floor`)
- **Comments say what and why in plain words:**
  - the reference being followed, its units, the decision with its date
  - no change logs in comments (git has them)
- **Errors:**
  - a bad argument is a `cli::Usage`, and the message names the valid values
  - a data problem is an `nxr::io::Error` with a `code::`
  - never fail silently

## Git

- **Every PR description answers the architecture checklist** (the architecture policy, point 5). A PR that breaks the architecture is not opened:
  ask the maintainers first.

- **Branches:** development goes into the integration branch `develop`. Work on a branch from it (`feat/…`, `fix/…`, `docs/…`), one commit per change;
  when its scoped local tests pass, merge it into `develop` (no pull request, no CI) and push.
- **Milestones:** `main` is updated from `develop` by ONE pull request on `neurodynamics-xr/nxr-neuro`, when a piece of work is done. Before opening it,
  run CI's own selection locally once (configure first): `python3 tools/ci_select.py --base origin/main --head develop --exclude-slow` prints the
  tests (`CTEST_NAMES`); run them with `tools/tests.py run '<that regex>'`. The pull request then needs one CI round (Linux, macOS, changed bindings).
- **Commit messages:** one sentence on what changed and why, prefixed by the area when it helps (`flow: …`).
- **Never** force-push, leave work uncommitted at the end of a run, or commit data, build outputs or `Testing/`.
- **`compute/` and `io/` came in by `git subtree`** with their history (`git log -- compute/src` reaches the old repos). The old `nxr-compute` and `nxr-io`
  repos are frozen: make changes here. Bringing in nxr-compute's open branch is described in `docs/reference.md` (*The old repositories*).
