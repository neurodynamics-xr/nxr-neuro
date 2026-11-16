# What Rheome has that nxr and nxr-neuro don't (survey, 2026-11-10)

A read-only survey of headers and docstrings, not of function bodies. It starts from every Rheome package's `Contents.m` and greps nxr (Python) and nxr-neuro (C++) for each concept. A "counterpart" here is a docstring or file-name match, so it is evidence, not proof of the same algorithm.

Paths:
- Rheome = `rheome/+rheome/`
- nxr = `nxr/src/nxr/`
- nxr-neuro = `compute/`, `io/`, `pipelines/`

## Summary

**The same in all three:** geometry, operators, eigenmodes, the differential operators, registration, I/O and import, the viewer and snapshot, and vortex-core detection. Rheome's `+cores` is itself a port of nxr's `vortex.py`.

**Rheome has much more** in these areas:
- forward and inverse modelling;
- graph and joint filterbanks;
- dynamics;
- sensor graphs;
- the constant-Q tile and query layer;
- group inference;
- the mode atlas;
- typed plans.

**nxr is not a subset of Rheome.** It has the database-backed dataset store, the CLI and its importers, named pipelines, about 60 analyses and about 70 figure modules.

## 1. Per Rheome package

| Rheome | What it does | nxr | nxr-neuro | Status |
|---|---|---|---|---|
| `view.m`, `snapshot.m` | viewer and print panels (thin callers of the Cortical Flow viewer) | `_viewer.py`, `_snapshot.py` (byte-parity twins) | — | same |
| `+connectome` | HCP-1065 template; resolve a subject's fibre source | `pipelines/atlas/hcp.py`, `convert/fibers.py` | `io/src/bst_fibers.cpp` | same |
| `+cores` | threshold-on-curl vortex cores and tracker, burst stages, band frames, lag-free null, tile crossings, BH | `detect/vortex.py`, `vortex_cores.py`, `cycle_cores.py`, `core_phase.py` | — | same (its origin is nxr) |
| `+demos` | filterbank, sensor, LB and wavelet demos; self-tests | — | — | Rheome only |
| `+detect` | critical points via the Dirac connection (winding, Poincaré–Hopf), ridges, peaks, spindle, rise/fall, chirality, phase singularity, onset spread, occupancy, tile paths | `detect/sleep_events.py`; rise/fall and spindle ported into `detect/vortex.py` | `direction_field.cpp` (partial) | partial |
| `+differential` | curl, divergence, divcurl, helmholtz, `helmholtzbands`, `poisson`, `meancurvature`, `curvaturesource`, `diracparts` | `differential/` (curl, divcurl, divergence, helmholtz) | `poisson_solver.cpp`, `curvatures.cpp`, `hodge.cpp` | partial |
| `+domain`, `+fieldtype` | domain descriptors; field-type registry | — | `field_registry.h`, `facets.h` | partial (nxr-neuro only) |
| `+dynamics` | DMD, dispersion ω(λ), PDE fit per mode, mode coefficients, manifold Horn–Schunck optical flow, flow read-out | envelope-flow estimators (`flow/envelope_flow.py`, `analyses/_estimators.py`), not shown to be the same method | — | Rheome only (DMD, dispersion, PDE); optical flow partial |
| `+eigen` | LB pencil modes, relative-Dirac frame, Hodge–Dirac ladders and package, quaternion lift, connectome eigen, `smallest` | `eigen/hodge_dirac.py`, `eigen/spectra.py` | `eigensolver.cpp`, `eigenproblem.cpp`, `hodge_dirac.cpp`, `pipelines/src/eigen.cpp` | same |
| `+filters` | kernels (heat, mexhat, Gabor, wave, Klein–Gordon, Schrödinger, space–time Matérn, travelling and damped wave, resonator, scatter, steer, spin, sphere), frame analysis, synthesis, bounds and scalograms, `jspectrum`, FIR, resample | `filters/frames.py`, `kernels.py`, `taps.py` | — | partial |
| `+flow` | fused sensor-to-cortex flow and its observables: helicity, enstrophy, energy, the vortex atom/bank/match/pursuit/track, seeded fields, sensitivity, crosstalk, phase, … | `flow/` (envelope flow, packets, sense, stream, time rotation) | — | partial (large Rheome surplus) |
| `+forward` | leadfield, Dirac gain and analysis, project and reconstruct, simulate, sensor banks, observability, `measurable`, lagged pairs | — (kernels handled only as data) | — (`bst_inverse.cpp` stores them) | Rheome only |
| `+geom` | icosphere, geodesics, heat distance, glue, flipOut, Karcher mean, trees, tiles, ladders, meshkey, draw/measure/select | `geom/` | `geodesic*.cpp`, `parametrization.cpp`, `query_measure.cpp`, Karcher mean | same |
| `+graphtransform` | field ↔ coefficients: exact eigenbasis, Chebyshev, validate | — (only the bank *design* is in `filters/frames.py`) | — | Rheome only |
| `+group` | tile-wise GLM (Freedman–Lane, max-t FWER, random intercept), spin test, calibration | `analyses/_grouptests.py`, `_stats.py` (other tests) | — | partial |
| `+import`, `+load` | dataset, surface, cortex, atlas, bases, dirac, inverse, recording | `io/`, `convert/` | `io/` | same |
| `+ingest` | constant-Q tile summaries of a recording (bank, grid, reduce, rollup, spatial, …) | `measure/rollup.py` (not equivalent) | `chunking.cpp` (storage only) | Rheome only |
| `+inverse` | whitened MNE (amplitude/dSPM/sLORETA), Dirac-basis inverse, Hodge weights, synthesis and registration, resolution, whitening | `inverse/fused.py` (Hodge weights), `inverse/roundtrip.py` | — | partial |
| `+io` | readers and writers; viewer bundle `rheome.bundle@1` | `io/`, `figures` bundles | `io/` | same |
| `+jtv` | invertible joint time–vertex filterbanks (bank, bounds, dual, analysis, synthesis) | — (`spectral/joint.py` is a different view) | — | Rheome only |
| `+modes` | resting-state mode atlas: discover, cluster, match, split, fit, lagged, phase randomisation, nulls | — | — | Rheome only |
| `+operators` | mass, LB, connection Laplacian, Diracs, intrinsic Delaunay, face gradient, gauge, connectome Laplacian, `harmonicfill`, registry | `operators/` | `connection_laplacian.cpp`, `dirac_operator.cpp`, `hodge_dirac.cpp`, `spin_dirac.cpp`, `mesh_operators.cpp`, … | same |
| `+pipeline` | a plan as a value: typed arrows checked when built, compile, run with provenance | `pipelines/` (named pipelines, a different design) | `pipelines/`, `operator_registry.h` | partial |
| `+qc` | bad-channel detection and rules (v3), empty-room test | `qc/sensors.py` (v1 rules), `analyses/qc*.py` | — | partial (rule sets not compared) |
| `+register` | registration by vertices or in the Hodge–Dirac modes | `register/` | — | same; nxr-neuro: none |
| `+report` | article with provenance (PDF/HTML/Word); red banner on an uncommitted tree | — | — | Rheome only |
| `+scale` | per-participant driver plus ~40 `measure_*` | `analyses/` (a different, larger set) | — | partial |
| `+select` | tile store as a relational query algebra (two executors that must agree) | `io/database.py`, `schema.py` (not this algebra) | `io/src/database.cpp`, `query_measure.cpp` | partial |
| `+selection` | selection types that merge exactly | — | — | Rheome only |
| `+sensors` | sensor arrays as graphs (linear, grid/Utah/ECoG, MEG, EEG): calibrate, modes, sample, window, tree | import and QC only | import only | Rheome only |
| `+show` | MATLAB graphics fallback | `figures/` | — | partial |
| `+source` | end-to-end Dirac-eigenmode whitened-MNE source mapping | — | — | Rheome only |
| `+spectral` | specparam fit, complementary periodic/aperiodic gains, carrier, 1/f synthesis, Meyer band | `spectral/aperiodic.py`, `chain.py`, `envelope.py`, `rotary.py` | — | partial |
| `+utils` | `hemisphere`, `pysort` | — | — | Rheome only (minor) |
| `+viz` | viewer tiles, points, marks, labellings | `figures/tokens.py`, `scene.py`, `viewer_tiles.py` | — | same |
| `@graphfilterbank`, `@jointfilterbank`, `@timefilterbank` | filterbank classes | `filters/frames.py` (time and graph design only) | — | partial |
| `@flowpage`, `@flowfeatures`, `@flowbrowser` | band pages of cortical vorticity; flow-tensor tables; browser | — | — | Rheome only |
| `@pagedrecording`, `@recordingbrowser` | paged recording reads; recording browser | `io/chunks.py` | `chunking.cpp` | partial |

## 2. Rheome only (no counterpart in nxr or nxr-neuro), by theme

- **Forward and inverse modelling:**
  - `+forward/*`: leadfield, Dirac gain and analysis (they differ by a mass weighting), project, reconstruct, simulate, sensor and analytic banks, observability, `measurable`, lagged pair and rest;
  - `+inverse/mne.m` (whitened MNE, dSPM, sLORETA), `inverse/dirac.m`, `hodge_synthesis.m`, `hodge_register.m`, `resolution.m`, `analytic.m`, `private/whiten.m`;
  - `+source/dirac.m`.
- **Graph and joint filterbanks:** `+graphtransform/*`; `@jointfilterbank`; `+jtv/*`; `+filters` frame operations and scalograms; the kernel library beyond itersine, mexhat and heat.
- **Dynamics:** `+dynamics/dmd.m`, `dispersion.m`, `pde_fit.m`, `mode_coefficients.m`, `flow_readout.m`, `tangent_basis.m`, `opticalflow_scalar.m` (manifold Horn–Schunck).
- **Flow observables and tracking:**
  - helicity, enstrophy, energy;
  - the vortex atom, bank, match, pursuit, spectrum and track; moving vortex and peak;
  - seeded rotor and vortex, synthetic fields;
  - sensitivity, crosstalk, upstream share; alpha states and labels; rate plans and window tables; phase locking, regression and binning; weak forms, sweeps, smoothness;
  - from `+detect`: the critical-point route (`criticalPoints`, `chirality`, `phasesingularity`, `ridges`, tile paths, peaks and tracks, track statistics and nulls, blob scale, charge density, onset spread, occupancy, event-locked analysis).
- **Sensor-graph analysis:** `+sensors/*` and `+demos/sensor_*`.
- **Constant-Q tiling and query:** `+ingest/*`; `+select/*`; `+selection/*`; `@pagedrecording`; `@recordingbrowser`.
- **Mode atlas:** `+modes/*`.
- **Typed plans:** `+pipeline/*` (a different design from nxr's and nxr-neuro's pipelines).
- **Group inference:** `+group/glm.m` (Freedman–Lane, max-t FWER, random intercept), `spin.m`, `calibrate.m`.
- **Other:**
  - the `+scale` driver and its `measure_*` functions;
  - `+report/*`;
  - `+spectral` (`gains`, `carrier`, `synthesise`, `meyerband`);
  - operators `harmonicfill`, `vertex_to_face`, `weak_differential`, `trivial_connection`, `lb_connectome`, `connectome_laplacian`;
  - differential `helmholtzbands`, `curvaturesource`, `diracparts`;
  - `+utils/*`; `+qc` v3 rules; `+demos/*`; `+show/*`.

## 3. In Rheome and nxr, not yet in nxr-neuro

- registration;
- vortex and core detection;
- nxr's differential operators as written (curl, div, Helmholtz per face, averaged to vertices);
- the fused inverse (Hodge weights);
- the spectral and filter chain (aperiodic fit, frames, kernels, envelopes, rotary, packets, envelope flow);
- sensor QC;
- statistics (BH/FDR, permutation);
- sleep events;
- analyses, figures, the viewer twin and snapshot (these stay in Python).

## 4. nxr only

- the database-backed dataset store (`io/database.py`, `model.sql`, `schema.py`, `shared_dataset.py`, `writer_lock.py`, the migrations);
- the CLI;
- importers for EEG, MEG, BIDS-iEEG, FreeSurfer, MRI and templates;
- the named pipelines;
- about 60 analyses and about 70 figure modules;
- `measure/*`.

nxr-neuro only: geometry-central's extras in C++ (stripes, BFF, isolines, streamlines, Cholesky cache, signed and vector heat), and its C++ Zarr/Brainstorm I/O.

## 5. Duplicates to reconcile

- **Hodge–Dirac, spin Dirac, connection Laplacian, face gradient:** three copies each (see `2026-11-10-operators-eigenmodes.md`).
- **Vortex cores:** Rheome `+cores` = nxr `vortex.py`; Rheome's critical-point route is a separate algorithm.
- **Time filter bank:** `@timefilterbank` vs `filters/frames.py itersine_time_frame`. Their gain conventions differ (analytic gain 2 vs 1).
- **Graph filter bank:** `@graphfilterbank` vs `filters/frames.py graph_bank`. Only the design is shared.
- **Optical flow:** Rheome Horn–Schunck vs nxr's envelope-flow estimators, not shown to be the same method. The Studio's version uses the connection Laplacian.
- **Bad-channel QC:** Rheome v3 vs nxr v1, never compared.
- **Spindle and rise/fall:** ported into nxr, but "not compared" with `detect/sleep_events.py`.
- **Aperiodic fits:** Rheome `+spectral`, nxr `spectral/aperiodic.py`, and the slope inside `qc/sensors.py`.
- **Karcher mean:** Rheome `+geom` and nxr-neuro `compute.h`; none in nxr.
- **Registries:** Rheome's field-type and operator registries vs nxr-neuro's `field_registry.h` and `operator_registry.h`.
- **Empty placeholder folders in nxr:** `viz/`, `_port/`, `modes/`, `geometry/`, `steps/` hold only `__pycache__`. nxr-neuro's `neuro/` is a README only.

## For planning nxr-neuro

The largest Rheome-only blocks that fit nxr-neuro:
1. **Forward and inverse modelling**, for `neuro/`: Rheome's `inverse.mne` is already a port of Brainstorm's `bst_inverse_linear_2018`.
2. **The graph and joint filterbanks** (`+graphtransform`, `+jtv`, `@jointfilterbank`, the frame operations), for `compute/`.
3. **Dynamics** (DMD, dispersion, PDE fit, optical flow), for `compute/`.
