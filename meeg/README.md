# meeg — MEG and EEG source imaging (`nxr_meeg`)

Each modality is its own folder (`meeg/` now; `pet/` and others later). This library computes, from arrays:

- the **whitener** of a noise covariance, per channel type (`whitener`; Brainstorm's and MNE-Python's conventions as named options),
- **depth weights** (`depthWeights`),
- the **unconstrained minimum-norm kernel** (imaging kernel, `unconstrainedKernel`): K [3N x M], Ĵ = K b, depth-weighted, with the SNR as Λ,
- the **sample covariance**, accumulated chunk by chunk (`CovarianceAccumulator`), the mean removed per block.

- the **forward model** of sphere head models (`forward.h`, `sphere.h`, `minimise.h`): MEG by Sarvas in one sphere or overlapping spheres, EEG by the three-shell Berg-Scherg sphere, the sphere fits, a Nelder-Mead simplex, the CTF reference correction (spec section 8).

Headers: `include/nxr/meeg/inverse.h`, `covariance.h`, `forward.h`, `sphere.h`, `minimise.h`. Maths, options and reference behaviour: `../docs/specs/2026-11-10-neuro-forward-inverse.md`.

**Dependency rule: Eigen only.** No `compute/`, no `io/`, no file reading (`../io` reads and writes, `../pipelines` joins them: `noisecov compute`, `inverse compute`).
Pure: arrays in, arrays out. The code is written from the published maths (Hämäläinen & Ilmoniemi 1994; Lin et al. 2006; Gramfort et al. 2014), **not translated from Brainstorm** (GPLv3);
Brainstorm and MNE-Python (BSD) are sources of reference outputs only.

Not here yet: dSPM and sLORETA, loose and fixed orientations, BEM/FEM head models (imported), shrinkage covariances. The pipeline `headmodel compute` is in `../pipelines`.

Tests (`ctest -R meeg`): `test_forward` (simplex, sphere fit, Sarvas and EEG-series checks), `test_forward_parity` (Brainstorm and MNE-Python goldens), `test_inverse` (closed form, invariances, errors, full-rank and rank-deficient), `test_inverse_parity` (Brainstorm and MNE-Python goldens), `test_covariance`
(chunking, blocks, divisors, MNE-Python golden). Goldens in `test/golden/` are synthetic; their generators sit beside them (`gen_kernel_golden.py` + `.m`, `gen_cov_golden.py`).
