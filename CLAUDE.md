# CLAUDE.md

nxr-neuro is a C++ toolkit for measuring the spatiotemporal patterns of cortical activity from MEG/EEG (see `README.md`).

Before changing anything, read:

- `ARCHITECTURE.md`: the architecture every change follows (layers, the dependency rule, reductions belong to the selection system, pipelines as graphs of pure nodes, the chunk service). It is never broken; if a change seems to need breaking it, stop and ask.
- `CONTRIBUTING.md`: the method rule, how to add a verb, building and testing (scoped tests through `tools/tests.py`), code style, git workflow.
- `docs/reference.md`: every verb and method; `docs/specs/`: the designs behind them.
- `docs/vocabulary.md`: one meaning per word (manifold, cell, rank, level, the four classes, reduction and expansion); the target vocabulary every change adopts.

## Find what exists before building or answering

nxr-neuro already has most of what a new piece needs (121 compute headers, the registered nodes, the definitions). Duplicating one is a defect,
however well the copy works. Before writing a compute function, a node, a node option or a pipeline, and before answering a question about how the
code does something:
1. Run `tools/what_exists.py <what you are about to build, in words>` (and its synonyms: `mexican hat`, `hat`, `bank`), and read the entries it
   finds, in the code, not in a commit's diff.
2. Use what exists: call it, offer it as a node's option, or name it in the definition. Extend it only if it lacks something, in its own place.
3. Write a new one only when nothing fits, and say so: the commit (or the brief) names the searches run and why what they found does not fit.
A commit that adds a compute function, a node or a node option without that line is sent back.

## Before designing a node or a pipeline

Write it as its graph first (nodes, typed ports, the inputs it is handed, the outputs it gives back), each piece in its part (loading · preparation ·
graph · readback). The spec of the work says what it does; these five rules say how every node and pipeline is built. They make every graph typed,
cheap and suited to a GPU, so the same graph runs on a CPU or a GPU, in nxr-neuro or in a client. Check the graph against each before writing code; if a piece fits none, stop and ask.
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

## nxr-neuro is the authority; the Studio and nxr are clients

- **nxr-neuro is the authority** for the project's methods, its schema and its stored data. The Cortical Flow Studio, nxr (Python) and the bindings
  are clients.
- **Storage.** nxr-neuro owns the schema, migrations, integrity, the one writer, transactions and queries: one schema, one dataset layout (a catalogue
  plus self-contained subject packages). No client opens the database or the arrays; clients use nxr-neuro's queries (read a subject, list or read a
  class, query measurements, add, run) and receive the four classes with their metadata. The schema stays private: a client adapts to nxr-neuro,
  never the reverse, and no second copy of the schema is kept anywhere.
- **Who computes what.** nxr-neuro builds and solves (eigensolves, factorisations, triangulations); its results are stored and bound as constants.
  A GPU client (the Studio) may run the streamed dense passes and reductions of the same graphs.
- **Methods.** nxr is not a reference. A method's reference is the published method or an external implementation (CONTRIBUTING, the method rule);
  existing nxr, Rheome and Studio goldens are regression checks until replaced.
- **Porting.** Design in nxr-neuro, under its rules, reusing what it already has. Port from another code base (nxr, the Studio, Rheome) only when the
  work asks for a port.
- **The Studio as a reference, for now.** Until the four-class system is consolidated (`docs/vocabulary.md`), consult the Studio's design of the four
  classes, selections and measurements when designing them here; where it differs from nxr-neuro's rules, nxr-neuro's rules win.

## Templates and reference maps

(The design: `docs/specs/2026-11-16-subject-centred-datasets.md`, section 4.)
1. **A template is a subject**: a subject package of kind `template`, opened and read exactly as any subject is. Its reference maps (receptor or
   tracer maps, cortical gradients) are fields on its cortex. No special reader, store or layout for templates or maps.
2. **Never a default template.** The template a dataset uses is the one its data were computed in, named by the caller or the importer and recorded
   by name, version and content hash; no code picks one (not fsaverage, not any other). The search path says only where to look, never which.
3. **Provenance on every item.** Every template item and every reference map carries its `origin` record (source, URL, version, checksum, citation,
   licence, attribution, processing; no local paths), carried into everything derived from it. An item without one is not released.
4. **Released templates are read-only.** A released version is never written again; a change is a new version. A subject moves to a new version only
   when the caller asks (a re-registration).
5. **Compare through registrations, never by converting.** Datasets in different template spaces are compared by carrying their results through a
   registration between the templates (e.g. one template's sphere to the other's). A dataset is never converted to another template (re-imported or
   recomputed there) to make it comparable; it stays in the template it was computed in.

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
