# CLAUDE.md

nxr-neuro is a C++ toolkit for measuring the spatiotemporal patterns of cortical activity from MEG/EEG (see `README.md`).

Before changing anything, read:

- `ARCHITECTURE.md`: the architecture every change follows (layers, the dependency rule, reductions belong to the selection system, pipelines as graphs of pure nodes, the chunk service). It is never broken; if a change seems to need breaking it, stop and ask.
- `CONTRIBUTING.md`: the method rule, how to add a verb, building and testing (scoped tests through `tools/tests.py`), code style, git workflow.
- `docs/reference.md`: every verb and method; `docs/specs/`: the designs behind them.
- `docs/vocabulary.md`: one meaning per word (manifold, cell, rank, level, the four classes, reduction and expansion); the target vocabulary every change adopts.

The rules most often broken, all in those documents:
- NO WASTE in builds and tests (CONTRIBUTING.md, *No waste*): build only the targets you changed; `tools/tests.py run REGEX` builds and runs only
  the named tests and never reruns a green one; edit, build once, test once; the full build and suite only with `--suite`, at the end of a round.
- io, orchestration and compute are separate. Only io writes SQL or touches the database; pipelines and the command line use io's task API.
- Every pipeline is a YAML definition of registered nodes. Never write or extend a hand-wired C++ pipeline.
- `compute` = a node or a graph on data handed to it; `run` = a pipeline on a subject (find, stream, store). The same verbs in the command line,
  Python and MATLAB. Bindings are faces: no maths, SQL or logic.
- Reductions are measurements over partitions. Recordings are streamed one chunk at a time; the time range is always the caller's.
- Native file formats in `io/formats/`; references only generate goldens; no GPL code.
- Precision follows the input; names are descriptive; data choices are never defaulted.

Write nothing into this repository that is not meant for the public: no real-data results, no private paths, no names of people, no quoted decisions.
