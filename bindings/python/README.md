# nxr-neuro (Python)

The Python face of nxr-neuro: pure functions over NumPy arrays (`nxr_neuro.compute`), the readers of the field's file formats (`nxr_neuro.io`), and **every command of
the `nxr-neuro` command line as a function** (the dataset API: reading and writing datasets, `run`), as one compiled module over the same C++ library as the command line.
Install a wheel and `import nxr_neuro as nn`. See `docs/reference.md`, section "The Python face", in the repository.

## The dataset API: the command line as functions

`nxr-neuro <verb> <noun> --option VALUE` is `nn.<verb>.<noun>(option=VALUE)` (spec `docs/specs/2026-11-19-io-service.md`):

- the same verbs, nouns and option names, dashes as underscores (`--from-sphere` is `from_sphere`, `export bids-derivatives` is `nn.export.bids_derivatives`);
- a Python keyword takes a trailing underscore: the verb `import` is the module `nn.import_`, the option `--from` is `from_`;
- the command's positional argument (`import subject PROTOCOL`, `read dataset FOLDER`, `compile DEFINITION`) is the one positional parameter; every option is given by name;
- a flag is a boolean, a repeatable option a list (an option the command line takes comma-separated, `--subjects A,B`, also takes a list); required options are required, and
  an option left out takes the command line's default;
- what comes back is the command's result object as an `nn.Result` (a dict). Where the command line writes an array to a file or prints its checksum, the array is in
  `.arrays` (NumPy, in its stored precision). `export ...` writes its files and returns their paths. `progress=callable` receives each progress line as a dict;
- arrays go in as NumPy arrays (any array-like, kept in its precision; or a path to a .npy file): `add field --values FILE.npy` is `nn.add.field(values=array)`;
- refusals raise `nn.Error` subclasses with io's code: `InvalidArgumentError` (the command line's usage error), `NoSuchRow`, `AlreadyExists`, `Ambiguous`,
  `WriterLockHeld`, `NotWritable`, `SchemaVersionError`, `NotADatabaseError`, `NotFound`.

```python
import numpy as np
import nxr_neuro as nn

nn.create.dataset(dataset="study")                                   # a catalogue of subject packages
nn.create.subject(dataset="study", name="sub-01")
nn.list.subjects(dataset="study")["subjects"]
nn.add.points(dataset="study", subject="sub-01", name="electrodes", positions=np.random.rand(8, 3))
nn.add.field(dataset="study", subject="sub-01", name="w", manifold="electrodes", values=np.ones(8, np.float32))
nn.read.field(dataset="study", subject="sub-01", name="w").arrays["values"]          # float32, as stored
nn.read.field(dataset="study", subject="sub-01", name="rest", from_=0.0, to=10.0)  # a recording: its time range is always stated (or all=True)
nn.import_.recording(from_="rec.fif", package="study/sub-01.nxr.zarr")
nn.compile("psd", dataset="study", subject="sub-01", set={"recording": "rest"})    # type-checked, not run
r = nn.run("psd", "study", "sub-01", recording="rest", from_=0.0, to=60.0)         # a pipeline: the command line's `run`
r["stored"], r["cache"]                                                            # what was stored; "computed" | "hit"
nn.list.pipelines()                                                                # what run takes, each parameter with its meaning, default and unit
nn.read.subject(dataset="study", subject="S1")                                  # every node's description (manifolds, fields, ...), no arrays
nn.list.fields(dataset="study", subject="S1", name="rest", limit=20, columns="name,unit")   # the nodes of one class: filtered, paged (`next` -> `after=`), only the columns asked for
```

`nn.run(pipeline, dataset, subject, from_=, to= | all=True, replace=, dry_run=, preset=, **parameters)`: the time range is required (seconds on the recording's clock);
a group definition takes `subjects=[...]` and `into=`. `nn.open(dataset)` gives a handle whose methods are the same functions with the dataset bound
(`d.read.field(subject=..., name=..., all=True)`, `d.run("psd", "sub-01", all=True)`); `nn.open(dataset, mode="write")` takes the dataset's writer lock and holds it until
`close()` (a context manager): the handle's own commands borrow it, another process's write is refused meanwhile (`WriterLockHeld`).

**Changed from the first wheel (refused, naming the new form; no aliases):** `nn.read(dataset, subject, name, ...)` is `nn.read.field | statistic | operator | selection |
manifold(dataset=, subject=, name=, ...)` (a recording's range as `from_`/`to` in seconds, not `start`/`stop` in samples); `nn.run(..., start=, stop=)` is `from_=`, `to=`;
`nn.list.results(dataset, subject, kind)` is `nn.list.results | recordings | kernels(dataset=, subject=)`; `nn.list.subjects(names=)` is `subjects=`; the lists return the
command's result dict (`nn.list.datasets(datastore=d)["datasets"]`). `nn.Stored` and `nn.RunResult` are gone (results are `nn.Result` dicts).

## The surface and its operators

```python
s = nn.compute.geometry.surfaceArrays(V, F)          # made once: the intrinsic Delaunay triangulation and its record, a dict of arrays
c = nn.compute.geometry.complex(s)                   # intrinsic faces, edges (tail -> tip), signed face edges, lengths
dec = nn.compute.operators.assembleDECOperators(s)   # d0, d1, hodge0..2, hodge1Inverse (scipy.sparse)
h = nn.compute.operators.hodgedirac.assemble(s)      # the pencil (K, M), the topology, the star1 counts
m = nn.compute.operators.hodgeDiracModes(h, 60)      # mu, vectors (0-, 1-, 2-form rows), kind
ic = nn.compute.signal.intrinsicComplex(s)          # the intrinsic complex, built once, passed to the forms
v = nn.compute.signal.sharp(ic, w)                   # a 1-form on the intrinsic edges -> tangent vectors at the vertices

md = nn.compute.signal.modesOf(m, h)                 # the modes as the forms read them (P, X2, mu, kind)
J = nn.compute.signal.phaseCurrent(ic, z)             # z [nV, frames] complex -> J [nE, frames]
c = nn.compute.signal.projectOneForm(ic, md, J)       # coefficients [K, frames]; zero rows by kind or by mu to split or band-limit
Phi = nn.compute.signal.exactPotential(md, c)        # the potential of the exact part
Psi = nn.compute.signal.coexactStream(ic, md, c)      # the stream function of the coexact part
```

Every surface function takes the dict `surfaceArrays` made; none triangulates again. See `docs/reference.md` for the full list.

## A local build

`pip install ./bindings/python` (from the repository root) builds the module with CMake and installs it; `-C build-dir=<folder>` keeps the build tree,
so a second install recompiles only what changed.
