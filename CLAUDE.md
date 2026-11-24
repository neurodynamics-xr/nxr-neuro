# CLAUDE.md

nxr-neuro is a C++ toolkit for measuring the spatiotemporal patterns of cortical activity from MEG/EEG (see `README.md`).

Before changing anything, read:

- `ARCHITECTURE.md`: the architecture every change follows (layers, the dependency rule, reductions belong to the selection system, pipelines as graphs of pure nodes, the chunk service). It is never broken; if a change seems to need breaking it, stop and ask.
- `CONTRIBUTING.md`: the method rule, how to add a verb, building and testing (scoped tests through `tools/tests.py`), code style, git workflow.
- `docs/reference.md`: every verb and method; `docs/specs/`: the designs behind them.
- `docs/vocabulary.md`: one meaning per word (manifold, cell, rank, level, the four classes, reduction and expansion); the target vocabulary every change adopts.

## Before designing a node or a pipeline

Write it as its graph first (nodes, typed ports, the inputs it is handed, the outputs it gives back), each piece in its part (loading · preparation ·
graph · readback). The spec of the work says what it does; these five rules say how every node and pipeline is built. They are why the Studio's GPU
chain is the model: the same graph must run there unchanged. Check the graph against each before writing code; if a piece fits none, stop and ask.
1. **The four classes, everywhere.** Every input, intermediate and output is a manifold, a field, an operator or a selection, with its full type
   (`docs/vocabulary.md`, *The four classes*). Nothing travels as an untyped array, a JSON attribute or a class of its own.
2. **Spectral first.** Work in frequency and eigenmode coordinates for as much of the pipeline as possible: transform once (the chunk's Fourier
   transform, the analysis onto eigenmodes), do the work there as weightings and gains on bins and modes, which are cheap, diagonal and suited to a
   GPU, and return to time or to the cortex only where an output needs it. Compose linear operators first (`ARCHITECTURE.md`, *Compose first*).
3. **Nodes offer, pipelines choose.** A node takes arrays and offers every choice (a method, a width, a boundary condition, a tolerance) as a named
   parameter with its meaning, default and unit; it hard-codes no value and never picks its own input. A pipeline puts nodes together and makes the
   choices: it may fix a parameter, and every value it fixes is stated in its definition.
4. **The type system.** Every port is typed (the four classes: manifold, rank, level, number type, components, unit) and the runner checks each edge
   before any data flows: a mismatch is refused with its reason, never converted silently.
5. **The vocabulary.** Every name (node, port, parameter, stored kind, document) uses the words of `docs/vocabulary.md` in their one meaning. A new
   concept gets its word there first; a retired word is not used in new work.

## The rules most often broken, all in those documents
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
