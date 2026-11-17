# The command line: verb first, help everywhere, an explicit time range

Status: built in this change. Section 7 of `2026-11-16-distribution-and-interoperability.md` opened the design; this note records what was decided and built.

## Amendment (2026-11-17): `compute` and `run` redrawn

The maintainers redrew the two verbs on 2026-11-17; the design is `2026-11-17-compute-and-run.md`. In brief, superseding the parts of this note below:
- **`run <pipeline>`** runs every pipeline: each `compute <what>` command of section 3's table becomes `run <what>` with the same options (the hand-wired
  pipelines are dispatched as they are), and the YAML definitions take `--<parameter> VALUE` and `--from/--to`/`--all` (`--set` goes); `run <pipeline> --help`
  is generated, for definitions from their parameters' declared meaning, default and unit.
- **`compute <node or graph>`** applies a registered node (grouped names: `welch.psd`, `filter.bandpass`), a composite node or a graph file to files named on the
  command line (NumPy, MATLAB, GIFTI, NIfTI, the recording formats `io/formats` reads), with a provenance sidecar; `list nodes` lists them.
- **Old spellings** (section 3's hidden aliases, and `compute <what>`) are not kept: nxr-neuro is unreleased, so they are refused with the new spelling named.

## 1. What changes

`nxr-neuro` keeps its name and gets one grammar, **verb first** (`verb what [options]`), instead of the mixed `noun verb` / `verb` / `verb noun` forms it grew.
`cli/` still only parses and prints: every command is the same library call as before (the architecture rule, `ARCHITECTURE.md`). The change is a new front end:

- the argument library is **CLI11** (header-only, BSD-3-Clause; vendored in `cli/extern/CLI11`, listed in `THIRD_PARTY_NOTICES.md`);
- every command, option, meaning, default and unit is **declared once** (`cli/command_table.cpp`); `--help`, the man pages, the shell completion and the old-spelling aliases are all made from that table;
- a developer check, `nxr-neuro __check-options`, compares the table with what each command's parser accepts, and runs as a test, so the help cannot drift from the code;
- the old spellings keep working as hidden aliases for a deprecation period, with a one-line notice on stderr.

## 2. The verbs

| verb | what it does |
|---|---|
| `compute <what>` | every computation: source imaging, geometry, time, spectra, cortical flow and patterns, measurements, nulls |
| `run <pipeline>` | a pipeline definition (YAML) on a subject |
| `list datasets\|subjects\|recordings\|kernels\|results\|pipelines\|spatial-bands` | what exists |
| `show dataset\|subject` | one thing in detail |
| `read dataset\|crossspectrum` | stored arrays, and readings derived from stored cross-spectra |
| `query measurements`, `export measurements` | measurements at any scale, or as a table |
| `import <what>` | from a Brainstorm protocol (today's `convert`) |
| `create\|delete dataset`, `create\|delete subject`, `compact dataset`, `recover dataset`, `add surface` | the dataset's own housekeeping |
| `reduce pyramid` | group reductions |

`compute --help` lists the computations grouped by topic: source imaging, geometry, time, spectra, cortical flow and patterns, measurements, nulls.
The dataset-format verbs of the subject-centred format (`register`, `split`, `join`, the `--where` selection, the template search path) plug into the same
table; they are not part of this change.

## 3. The mapping (old spelling to new)

Every command of the old interface (41 top-level words, 68 spellings) and its new name. The old spelling in the second column keeps working.

| new | old (hidden alias) | topic |
|---|---|---|
| `nxr-neuro create dataset` | `nxr-neuro dataset create` |  |
| `nxr-neuro list datasets` | `nxr-neuro dataset list` |  |
| `nxr-neuro show dataset` | `nxr-neuro dataset show` |  |
| `nxr-neuro delete dataset` | `nxr-neuro dataset delete` |  |
| `nxr-neuro recover dataset` | `nxr-neuro dataset recover` |  |
| `nxr-neuro compact dataset` | `nxr-neuro dataset compact` |  |
| `nxr-neuro create subject` | `nxr-neuro subject create` |  |
| `nxr-neuro list subjects` | `nxr-neuro subject list` |  |
| `nxr-neuro show subject` | `nxr-neuro subject show` |  |
| `nxr-neuro delete subject` | `nxr-neuro subject delete` |  |
| `nxr-neuro add surface` | `nxr-neuro surface add` |  |
| `nxr-neuro list recordings` | (new) |  |
| `nxr-neuro list kernels` | (new) |  |
| `nxr-neuro list results` | (new) |  |
| `nxr-neuro list pipelines` | `nxr-neuro run --list` |  |
| `nxr-neuro list spatial-bands` | `nxr-neuro pyramid spatial-bands` |  |
| `nxr-neuro import list` | `nxr-neuro convert list` |  |
| `nxr-neuro import scan` | `nxr-neuro convert scan` |  |
| `nxr-neuro import subject` | `nxr-neuro convert subject` |  |
| `nxr-neuro import template` | `nxr-neuro convert template` |  |
| `nxr-neuro import group` | `nxr-neuro convert group` |  |
| `nxr-neuro import templates` | `nxr-neuro convert templates` |  |
| `nxr-neuro import surface` | `nxr-neuro convert surface` |  |
| `nxr-neuro import mri` | `nxr-neuro convert mri` |  |
| `nxr-neuro import condition` | `nxr-neuro convert convert` |  |
| `nxr-neuro import inverse` | `nxr-neuro convert inverse` |  |
| `nxr-neuro import map` | `nxr-neuro convert map` |  |
| `nxr-neuro import resample` | `nxr-neuro convert resample` |  |
| `nxr-neuro import badsegments` | `nxr-neuro convert badsegments` |  |
| `nxr-neuro import events` | `nxr-neuro convert events` |  |
| `nxr-neuro read dataset` | `nxr-neuro read` |  |
| `nxr-neuro read crossspectrum` | `nxr-neuro crossspectrum read` |  |
| `nxr-neuro query measurements` | `nxr-neuro measure query` |  |
| `nxr-neuro export measurements` | `nxr-neuro measure export` |  |
| `nxr-neuro reduce pyramid` | `nxr-neuro pyramid reduce` |  |
| `nxr-neuro run` | `nxr-neuro run` |  |
| `nxr-neuro compute noisecov` | `nxr-neuro noisecov compute` | Source imaging |
| `nxr-neuro compute inverse` | `nxr-neuro inverse compute` | Source imaging |
| `nxr-neuro compute headmodel` | `nxr-neuro headmodel compute` | Source imaging |
| `nxr-neuro compute pointspread` | `nxr-neuro pointspread compute` | Source imaging |
| `nxr-neuro compute eigenmodes` | `nxr-neuro eigen solve` | Geometry |
| `nxr-neuro compute frames` | `nxr-neuro frames compute` | Geometry |
| `nxr-neuro compute tiles` | `nxr-neuro tiles compute` | Geometry |
| `nxr-neuro compute constantq` | `nxr-neuro constantq compute` | Time |
| `nxr-neuro compute frametiling` | `nxr-neuro frametiling compute` | Time |
| `nxr-neuro compute joint` | `nxr-neuro joint compute` | Time |
| `nxr-neuro compute lowpass` | `nxr-neuro lowpass` | Time |
| `nxr-neuro compute highpass` | `nxr-neuro highpass` | Time |
| `nxr-neuro compute bandpass` | `nxr-neuro bandpass` | Time |
| `nxr-neuro compute bandstop` | `nxr-neuro bandstop` | Time |
| `nxr-neuro compute notch` | `nxr-neuro notch` | Time |
| `nxr-neuro compute psd` | `nxr-neuro psd compute` | Spectra |
| `nxr-neuro compute sourcepsd` | `nxr-neuro sourcepsd compute` | Spectra |
| `nxr-neuro compute sourcetensor` | `nxr-neuro sourcetensor compute` | Spectra |
| `nxr-neuro compute crossspectrum` | `nxr-neuro crossspectrum compute` | Spectra |
| `nxr-neuro compute quadrature` | `nxr-neuro quadrature compute` | Spectra |
| `nxr-neuro compute hodgepower` | `nxr-neuro hodgepower compute` | Spectra |
| `nxr-neuro compute speedspectrum` | `nxr-neuro speedspectrum compute` | Spectra |
| `nxr-neuro compute phasegradient` | `nxr-neuro phasegradient compute` | Cortical flow and patterns |
| `nxr-neuro compute cyclepattern` | `nxr-neuro cyclepattern compute` | Cortical flow and patterns |
| `nxr-neuro compute opticalflow` | `nxr-neuro opticalflow compute` | Cortical flow and patterns |
| `nxr-neuro compute cellquadrature` | `nxr-neuro cellquadrature compute` | Cortical flow and patterns |
| `nxr-neuro compute patches` | `nxr-neuro patches compute` | Cortical flow and patterns |
| `nxr-neuro compute scaleselection` | `nxr-neuro scaleselection compute` | Cortical flow and patterns |
| `nxr-neuro compute pyramid` | `nxr-neuro pyramid build` | Measurements |
| `nxr-neuro compute measure` | `nxr-neuro measure compute` | Measurements |
| `nxr-neuro compute statistic` | `nxr-neuro statistic compute` | Measurements |
| `nxr-neuro compute wishart` | `nxr-neuro wishart compute` | Nulls |
| `nxr-neuro compute spatialnull` | `nxr-neuro spatialnull compute` | Nulls |
| `nxr-neuro compute null` | `nxr-neuro null compute` | Nulls |
| `nxr-neuro compute repeat` | `nxr-neuro repeat compute` | Nulls |

`list recordings`, `list kernels` and `list results` are new: they read the dataset's rows (`pipelines/listing.h`), nothing is loaded.
`run` keeps its words; only `run --list` moved (to `list pipelines`). `read <folder>` became `read dataset <folder>`.
Options keep their names, with one exception: the one-letter `--h` of `compute null` and `compute spatialnull` (the stopping count of the Besag-Clifford test)
is `--exceedances` (descriptive names everywhere); `--h` still works through the old spelling.

## 4. Help

`nxr-neuro --help` lists the verbs; `nxr-neuro <verb> --help` the commands of a verb (for `compute`, grouped by topic); `nxr-neuro <verb> <what> --help`
one command: its summary, each option with its meaning, default and unit, an example, the time-range rule and the old spelling.
The text of each option is **per command**, taken from that command's own options struct (the field comments: meaning, default and unit); only options that mean the same everywhere (`--dataset`, `--subject`, `--replace`, `--json`, the time range) share one text. A test (`nxr-neuro __check-help`) requires every option of every command to have its own description, and that the help prints the default and the unit when they exist. `nxr-neuro completion bash|zsh|fish`
prints the completion script; `nxr-neuro __man DIR` writes the man pages (one for the command, one per verb, one per command). The build generates both into
`share/` and installs them.

## 5. The time range

**A time range is required for every computation over a recording**, as `--from <s> --to <s>` (seconds, `m:ss` or `h:mm:ss`) or `--all` (the whole recording).
Without one the command refuses and names the options. Two rules, in `ARCHITECTURE.md` and `CONTRIBUTING.md`:

1. **Streaming:** the engine always computes one chunk at a time, whatever the range, with bounded memory.
2. **The span is always stated by the caller:** no default range. `--chunks` (and the old `--start`/`--stop`) are hidden: for developers and tests.

**Library amendment.** The span is a required field the caller sets: `io::resolve_span` refuses a request with neither chunks nor a start and stop (it used to
read chunk 0), so every front end and test states its range. The tests that relied on the default now say `chunks = "0"`, the first chunk's samples.
The range actually used, after snapping to the method's boundaries, is in the result and in the stored provenance (the producer's `span` record: samples,
seconds, chunks read, halo, tile level, and the seconds asked); the readable output prints it.

`compute null` and `compute spatialnull` need a range too (`--from/--to` or `--all`): their options structs have no default span either.

## 6. Output

- In a terminal: a progress bar on stderr (one line per chunk when stderr is not a terminal), then on stdout a **result line** (what was done, `cache hit` or
  `computed`, the time taken, the range used) and what was stored (lists as tables).
- With `--json`, or when stdout is not a terminal: **today's JSON lines, unchanged**, on stdout.
- Progress and diagnostics on stderr, results on stdout. `--quiet` prints only the result line.
- Errors: readable on stderr in a terminal; with `--json`, a `stage: error` object that also carries its `code`; exit status 1 for a failure, 2 for a usage error.
- Colour only in a terminal, never when `NO_COLOR` is set (or `--no-color`).

## 7. Tests

`cli/test/cli_tests.cmake` (ctest `cli_*`): the verbs parse and dispatch; `--help` renders for every command and verb; every old alias still runs and prints its
notice; the time range is refused when missing; the readable and JSON forms switch (the readable one is forced with a hidden `--human` since a test's stdout is
never a terminal); the JSON of a few commands is the same under the old and the new spelling; the table matches the parsers (`__check-options`); man pages and
completion are generated; every option of every command has its own description, printed with its default and unit (`cli_helptext`).
