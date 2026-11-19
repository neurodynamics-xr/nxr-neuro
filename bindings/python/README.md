# nxr-neuro (Python)

The Python face of nxr-neuro: pure functions over NumPy arrays (`nxr_neuro.compute`), the readers of the field's file formats (`nxr_neuro.io`) and the pipeline
level (`nxr_neuro.run`, `nxr_neuro.list`, `nxr_neuro.read`), as one compiled module over the same C++ library as the `nxr-neuro` command. Install a wheel and `import nxr_neuro as nn`.
See `docs/reference.md`, section "The Python face", in the repository.

## The pipeline level

`run` is `nxr-neuro run`: the same library calls (`pipelines/face.h`), so a result made here is the same cached result the command line makes and finds.

```python
import nxr_neuro as nn

nn.list.pipelines()                       # the pipelines run takes, each parameter with its meaning, default and unit; .help() prints them
nn.list.datasets("study"); nn.list.subjects("study/demo", where="age > 60"); nn.list.results("study/demo", "sub-01")
r = nn.run("psd", "study/demo", "sub-01", recording="rest", set="brainstorm", segment_seconds=2.0, all=True)
r.cache, r.stored                         # "computed" | "hit"; what was stored (output, name, id, path)
psd = r.read()                            # = nn.read("study/demo", "sub-01", "rest_psd_welch"): psd.values [channels, frequencies], psd.meta
```

The time range is always stated: `start` and `stop` in seconds (the command line's `--from` and `--to`; `from` is a Python keyword), or `all=True`; a call without one is
refused. `progress=callable` receives each progress line as a dict. Pipelines named by `nn.list.pipelines()` are `psd` and the shipped definitions or a `.yaml` path; the
other hand-wired pipelines are reached through the command line until they are definitions. `nn.read(dataset, subject, name, measure=, of=)` reads a stored field, statistic,
operator or selection (a selection's measurements with `measure`) as NumPy arrays plus metadata.

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
