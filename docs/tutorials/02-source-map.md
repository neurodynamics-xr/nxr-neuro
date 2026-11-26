# Tutorial 2: from a recording to the cortex

**What this answers.** A pattern of activity lives on the cortex, but a recording is made at the scalp. To say *where* a rhythm is, and later how it moves, you need
three things the toolkit makes explicit instead of hiding: a **head model** (how a source on the cortex shows at each sensor), the **noise** of the recording (what
the sensors see that is not signal) and an **inverse kernel** (the minimum-norm map from sensors back to the cortex). The instrument has a resolution: a source estimate
blurs each source over its neighbours, and the cortex's own geometry limits which spatial patterns can be told apart at all (its eigenmodes). The tutorial prepares each of these on
synthetic data and stops where this build of the command cannot continue; it says so at that point.

Start from tutorial 1's folder (`work/study` with subject `sub-01` and the recording `task-rest`). Synthetic data only.

## 1. A cortex and a scalp

[synthetic_bids.py](synthetic_bids.py) also wrote two spheres in metres: `cortex.obj` (642 vertices, radius 7 cm) and `scalp.obj` (162 vertices, radius 10 cm), and `truth.json` with the
vertices where the 10 Hz and 20 Hz sources were put. Add them to the subject:

    nxr-neuro add surface data/cortex.obj --dataset study --subject sub-01 --name cortex
    nxr-neuro add surface data/scalp.obj --dataset study --subject sub-01 --name scalp --no-primary

    {"stage": "result", "name": "cortex", "folder": "source", "n_vertices": 642, "n_faces": 1280, "atlases": 0, "registration": false, "hemispheres": 0, ...}
    {"stage": "result", "name": "scalp", "folder": "mri", "n_vertices": 162, "n_faces": 320, ...}

`--scale` multiplies the mesh's coordinates to bring a file in millimetres to metres (`--scale 0.001`); the toolkit works in metres. A surface added is made the subject's primary surface by default
(the one other commands use when none is named); `--no-primary` keeps the scalp from taking that place from the cortex. The added surface carries a mesh health report (`mesh_health`: non-manifold
edges, duplicate or degenerate faces, isolated vertices).

## 2. The noise

A minimum-norm inverse needs the covariance of the noise. Here the recording itself stands in for an empty-room or baseline recording; for real data use a recording of the noise alone.

    nxr-neuro run noisecov --dataset study --subject sub-01 --recording task-rest --all

It stores the statistic `task-rest_noise_cov` (channels x channels) and the channels it covers (`task-rest_noise_cov_channels`), computed one chunk at a time over the range you state.
Export it to look at it:

    nxr-neuro export tsv --package study/sub-01.nxr.zarr --statistic task-rest_noise_cov --out out/cov

    {"stage": "result", "files": ["out/cov.tsv"], "sidecar": "out/cov.json", "rows": 1024, "columns": ["channel", "channel", "value"]}

1024 rows is 32 x 32 covariance entries.

## 3. What the cortex allows: its eigenmodes

Whatever the inverse produces, spatial patterns on this cortex can only be resolved as far as the surface and its mesh go. The Hodge-Dirac eigenmodes are the patterns the geometry
allows; the pipelines that measure spatial bands and flow read in this basis.

    nxr-neuro run eigenmodes --dataset study --subject sub-01 --surface cortex --modes 40

    [normalize] M-orthonormality verified: ||U'MU - I||_F = 2.03733e-15
    {"stage": "solved", "exact": 21, "coexact": 15, "seconds": 0.0}
    {"stage": "result", "pipeline": "eigenmodes@1", ..., "stored": [{"output": "modes", "kind": "manifold", "name": "hodgeDirac", ...}, {"output": "complex", ...,
       "name": "cortex_intrinsic", ...}, ...]}

Asking for 40 modes kept 36: 21 exact modes (gradients of scalar patterns) and 15 coexact (curl-like), the constants left out; the orthonormality check is the solve's own. The modes and
the intrinsic mesh are stored as `hodgeDirac*` and `cortex_intrinsic*` (`nxr-neuro list results` names them). The default of 3000 modes is for a real cortex; a 642-vertex sphere
cannot hold that many. See [the eigenmodes page](../pipelines/eigenmodes.md) and [the method](../reference.md#the-hodge-dirac-eigenmodes).

## 4. The head model, the kernel and the source map: not run with this build

The next steps are these, in this order, with the options their pages describe ([headmodel](../pipelines/headmodel.md), [inverse](../pipelines/inverse.md),
[sourcepsd](../pipelines/sourcepsd.md)):

    nxr-neuro run headmodel --dataset study --subject sub-01 --channels task-rest_channels --sources cortex --eeg-surface scalp --name headmodel
    nxr-neuro run inverse   --dataset study --subject sub-01 --forward headmodel --noise-cov task-rest_noise_cov
    nxr-neuro run sourcepsd --dataset study --subject sub-01 --recording task-rest --all

`headmodel` stores the gain (an operator from the cortex's sources to the channels), `inverse` the minimum-norm kernel with its whitener, and `sourcepsd` the band power of every source
(`<recording>_<kernel>_psd`: vertex x band), the map you would then read and compare with where the sources were placed (`truth.json`).

**These three were not run for this tutorial.** The head model needs each channel's position as sensor geometry. In the build used here, the BIDS importer records the electrode table with
the recording but does not store it as a positioned points manifold, and `run headmodel` on the imported recording stops with an error naming the channels manifold
(`channels manifold ...: not among the subject's rows`). Placing a standard electrode montage onto a subject is implemented in the library, but not exposed as a command in this build.
When it is, this section will carry the real output. Until then, a kernel imported from Brainstorm (`import inverse`) or from an MNE inverse operator (`import fif`) can be given to `sourcepsd` with `--kernel`.

## What you learned and where next

- The geometry (surfaces, in metres), the noise and the inverse are stored, named inputs; nothing is guessed (a head model whose surface is not named is refused).
- The cortex's eigenmodes are the stated limit of what spatial patterns can be resolved on a given mesh.

Next: [tutorial 3](03-bursts.md) finds bursts of the 10 Hz rhythm in time. Back: [tutorial 1](01-psd.md).
