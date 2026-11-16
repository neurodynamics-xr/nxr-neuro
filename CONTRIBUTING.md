# Contributing to nxr-neuro

Welcome, and thank you for contributing. Read `ARCHITECTURE.md` first: it is the architecture every change follows, and a change that breaks it is not
accepted. `docs/reference.md` describes every verb and method, `docs/specs/` the designs behind them. Contributions are made under the project licence
(BSD-3-Clause, see `LICENSE`); there is no CLA and no sign-off requirement.

## The method rule (how every computation is written; inside the paramount architecture)

1. **Each method has a named reference, and the default reproduces it.** The references are nxr (Python), Rheome (MATLAB), the Cortical Flow Studio
   (TypeScript), Brainstorm, MNE-Python and scipy. `docs/reference.md` says which one each method follows. The choice of the default reference is the maintainers';
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

## Adding or changing a pipeline verb (all steps, one PR; inside the paramount architecture)

0. **First, the graph:** write the pipeline as its graph (nodes, typed ports, the inputs it is handed, the outputs it gives back), place every piece in
   its part (loading · preparation · graph · readback), and check it against the Studio's chain. If any piece does not fit, stop and ask the maintainers.
1. The maths goes in `compute/` (or `meeg/`) as pure node functions (matrices in, matrices out), with a unit test in `compute/test/test_<topic>.cpp`.
2. The pipeline goes in `pipelines/include/nxr/pipelines/<area>.h` and `src/`:
   - an `<X>Options` struct with one comment per field giving the meaning, default and unit
   - a `<x>_compute(ds, sub, opt, progress)` function that ONLY prepares the inputs (loading through io, selection, type checks), builds the
     graph, runs it through the executor, and hands the outputs to io: no maths, no loops over samples or chunks
   - stored names and kinds written the way the importer or nxr stores the same thing, so a reader tells them apart only by the producer
3. The command goes in `cli/`:
   - parsing, plus the full option list in the comment at the top of the file
   - one line in `kUsage` (`cli/nxr_neuro.cpp`) and its dispatch
4. Add a pipeline test in `pipelines/test/` and register it in `pipelines/CMakeLists.txt`.
5. Update `docs/reference.md`: the command list, and a section for the method (the why, the default and its reference, what is stored, what it cannot resolve).
   **A verb is not done until `docs/reference.md` lists it.**
6. Larger designs first get a spec in `docs/specs/YYYY-MM-DD-<topic>.md`. A read-only comparison against the references goes in
   `docs/audits/YYYY-MM-DD-<topic>.md`.

## Build and test

CMake 3.20+, Ninja, C++17. Clone with `--recurse-submodules` (geometry-central). Large build trees may live on another disk: make `build/` a link to it before the first configure.

    cmake -S . -B build -G Ninja && cmake --build build 2>&1 | tail -20
    tools/tests.py status [REGEX]       # what is green, stale (what it runs has changed since it passed), failed or new; runs nothing
    tools/tests.py run REGEX            # run the stale, failed and new tests among those matching, and record them
    tools/tests.py show TEST            # a test's last result and what it printed

- **The test record.** Results are kept in `test-results/` (in the repo, never in git): `status.json` holds every test's latest result
  and the FINGERPRINT of what it ran (its binary, the `nxr-neuro` command it calls, its library's goldens and data); `runs/` holds each run. A test
  whose fingerprint is unchanged since it passed is still green and is not run again. Run tests through `tools/tests.py`, not bare `ctest`, so the
  record stays true.
- **The full suite runs at the end of a development round, when the maintainers say so,** not during it. During development: `status`,
  then `run` on what the change reaches.
- **Continuous integration** (`.github/workflows/ci.yml`): on every pull request, Linux builds (compiled objects cached with ccache) and runs
  every test except those labelled `slow`; pushes to main and a nightly run build Linux and macOS and run every test. A test that takes
  minutes gets `set_tests_properties(<test> PROPERTIES LABELS slow)` next to its `add_test`.

- **Test only what the change touches.**
  - A small change runs the tests of the function it changed: the one test file, or `ctest -R <test>`, or `pytest file::test`.
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
   - **macOS arm64** on `macos-14`, with no library outside `/usr/lib` and `/System` in `otool -L`.
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

- **Branch and PR:** work on a branch in this checkout (`feat/…`, `fix/…`, `docs/…`) and merge through a pull request on `neurodynamics-xr/nxr-neuro`.
- **Commit messages:** one sentence on what changed and why, prefixed by the area when it helps (`opticalflow: …`).
- **Never** force-push, leave work uncommitted at the end of a run, or commit data, build outputs or `Testing/`.
- **`compute/` and `io/` came in by `git subtree`** with their history (`git log -- compute/src` reaches the old repos). The old `nxr-compute` and `nxr-io`
  repos are frozen: make changes here. Bringing in nxr-compute's open branch is described in `docs/reference.md` (*The old repositories*).
