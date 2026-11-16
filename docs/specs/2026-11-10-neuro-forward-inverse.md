# nxr-neuro: the unconstrained minimum-norm kernel and the noise covariance (2026-11-10)

## Goal and rules

Goal (2026-11-10): compute a source mapping matrix inside nxr-neuro for when none is available through import.

Rules for this step (each modality is its own folder: `meeg/` for MEG and EEG, namespace `nxr::meeg`, library `nxr_meeg`; `pet/` etc. later):
- **Reimplemented from the published maths.** No Brainstorm code is translated, because Brainstorm is GPLv3.
- **Outputs must match** Brainstorm's (the default reference) and MNE-Python's (BSD), each through named options.
- **`meeg/` (MEG and EEG; each modality is its own folder) depends on Eigen only.** It does not depend on `compute/` or `io/`, and does no file I/O.

## 1. The step

| | |
|---|---|
| **In** | Gain G [M × 3N], in T/(A·m) or V/(A·m). Columns 3k..3k+2 are the x, y, z unit dipoles of source k. |
| | Noise covariance C [M × M] for the same M channels. |
| | One type string per channel (`MEG MAG`, `MEG GRAD`, `MEG`, `EEG`, `SEEG`, `ECOG`). |
| **Out** | Kernel K [3N × M], rows x, y, z interleaved per source, in A·m per channel unit. The estimate is Ĵ = K·b. |
| | Also: the whitener W [M × M], Λ, the whitener rank r, and `notes`. |

**Caller's job, already done before the call:**
- G and C are restricted to the good channels.
- When the recording has SSP projectors P: G := PG and C := PCPᵀ.
- When EEG, sEEG or ECoG are present: the average reference A (I − 11ᵀ/n within each of those modalities) is applied, G := AG and C := ACAᵀ.

K is then applied to the recording as read. The whitener's null space removes the projected and common-mode parts.

## 2. Maths

References:
- MNE: Hämäläinen & Ilmoniemi, *Med Biol Eng Comput* 32:35–42, 1994.
- Depth weighting: Lin et al., *NeuroImage* 31:160–171, 2006.
- MNE-Python conventions: Gramfort et al., *NeuroImage* 86:446–460, 2014, and MNE-Python 1.13 (`make_inverse_operator`, `compute_depth_prior`).

**Rank.** Of values s₁ ≥ … ≥ sₙ, keep those with sᵢ > n·ulp₃₂(s₁), where ulp₃₂(x) = 2^(⌊log₂ x⌋ − 23) (`RankTolerance::SinglePrecision`, the reference's rule).

**1. Whitener**, built per channel type. By default (`JointTypes = false`, Brainstorm) the cross-type covariance is dropped and `MEG MAG` and `MEG GRAD` are separate
types. MNE-Python (`JointTypes = true`) whitens the types of one modality together, keeping the mag-grad cross-covariance (MEG types in one block; EEG, sEEG and
ECoG each their own; the cross-covariance between modalities is dropped in both). The MNE ridge is added per type, on the range of that type's block
(ρ_t U_r U_rᵀ, so a projector's null space stays null), and the joint block is then whitened with no diagonal scaling (a scaling would change the null space). For each type block:
- C_t := (C_t + C_tᵀ)/2 = U diag(s²) Uᵀ, eigenvalues descending.
- Truncate to the rank r_t; C_r = U_r diag(s_r²) U_rᵀ.
- Then, by `Regularisation`:

| `Regularisation` | W_t |
|---|---|
| `Ridge` (default; `RidgeFraction` ε = 0.1) | U_r diag(1/√(s_r² + ρ)) U_rᵀ, with ρ = ε · tr(C_t)/n_t (the mean of all n_t eigenvalues; Brainstorm). `RidgeOverRank = true` (MNE-Python): ρ = ε · tr(C_t)/r_t, the mean of the r_t non-zero eigenvalues, which differs from the above only for a projected block |
| `Diagonal` | diag(1/√diag(C_r)) |
| `None` | U_r diag(1/s_r) U_rᵀ |

W = blockdiag(W_t), with rank r = Σ r_t.

If ‖C‖_F − ‖diag C‖_F < ulp₃₂(‖diag C‖_F), the covariance is diagonal, `Diagonal` is used, and this is reported in `notes` (`DetectDiagonal = true`, the reference's
behaviour).

The whitener's form is `WhitenerForm::Symmetric` (U diag Uᵀ, the reference) or `Pca` (diag Uᵀ, MNE). K is the same either way; only the returned W differs.

**2. Depth weighting** (`DepthEnabled = true`). It gives one scalar α_k per source. A larger α_k means a larger prior variance for that source (α_k² is its variance
scale). The norm ν_k of source k is set by `DepthNorm`:
- `Frobenius` (reference): ν_k = ‖G_k‖²_F. It is computed on the unwhitened G, all channels together.
- `Spectral` (MNE): ν_k = the largest eigenvalue of G_kᵀG_k, computed on one channel type (gradiometers if present, else magnetometers, else EEG; `DepthChannels = OneType`).

With exponent p (`DepthExponent`) and limit Λ_d (`DepthLimit` = 10), the cap is applied by `DepthLimitOn`:
- `Weight` (reference):
  - a_k = ν_k^p, then a_k := max(a_k, max a / Λ_d²).
  - α_k = √(max a / a_k), so α_k ∈ [1, Λ_d].
- `Norm` (MNE):
  - u_k = 1/ν_k; û = the smallest u_k exceeding Λ_d²·min u (or max u if there is none).
  - α_k = (min(u_k/û, 1))^{p/2}.

**3. Whitened, weighted gain:** L = W · G · diag(α_k ⊗ 1₃). It is [M × 3N].

**4. Λ from the SNR** (`Snr` = 3, an amplitude ratio). With L Lᵀ = V diag(σ²) Vᵀ:
- `SnrNormalisation::AllChannels` (reference): Λ = Snr² / ((1/M) Σᵢ σᵢ²). This is the mean over all M eigenvalues, zeros included.
- `SnrNormalisation::WhitenerRank` (MNE): Λ = Snr² / ((1/r) Σᵢ σᵢ²). This is equivalent to MNE's λ² = 1/Snr² after it scales the source covariance so that
  tr(L Lᵀ) = r.

The two agree unless W is rank-deficient (SSP, average reference, tSSS).

**5. Kernel:**
K = diag(α_k ⊗ 1₃) · Λ Lᵀ V diag(1/(Λσᵢ² + 1)) Vᵀ · W.

This equals R G̃ᵀ(G̃ R G̃ᵀ + I)⁻¹ W, with G̃ = W G and source covariance R = Λ diag(α_k² ⊗ 1₃): the minimum-norm estimate of Hämäläinen & Ilmoniemi in whitened
data.

**Invariance.** The result does not depend on the eigenbasis chosen (products U f(S) Uᵀ only). The Eigen solver may therefore differ from MATLAB's and NumPy's.

## 3. API

Header `meeg/include/nxr/meeg/inverse.h`, namespace `nxr::meeg`:

```cpp
enum class Regularisation { Ridge, Diagonal, None };
enum class RankTolerance { SinglePrecision, DoublePrecision };
enum class WhitenerForm { Symmetric, Pca };
enum class DepthNorm { Frobenius, Spectral };
enum class DepthLimitOn { Weight, Norm };
enum class DepthChannels { All, OneType };
enum class SnrNormalisation { AllChannels, WhitenerRank };

struct WhitenerOptions {
  Regularisation regularisation = Regularisation::Ridge;
  double ridgeFraction = 0.1;
  RankTolerance rankTolerance = RankTolerance::SinglePrecision;
  bool detectDiagonal = true;
  bool jointTypes = false;        // MNE-Python: the types of a modality whitened together
  bool ridgeOverRank = false;     // MNE-Python: the ridge's mean eigenvalue over the rank
  WhitenerForm form = WhitenerForm::Symmetric;
};
struct DepthOptions {
  bool enabled = true;
  double exponent = 0.5, limit = 10;
  DepthNorm norm = DepthNorm::Frobenius;
  DepthLimitOn limitOn = DepthLimitOn::Weight;
  DepthChannels channels = DepthChannels::All;
};
struct KernelOptions {
  WhitenerOptions whitener;
  DepthOptions depth;
  double snr = 3;
  SnrNormalisation snrNormalisation = SnrNormalisation::AllChannels;
};

struct Whitener { Eigen::MatrixXd W; int rank; std::vector<std::string> notes; };
struct Kernel { Eigen::MatrixXd K, W; Eigen::VectorXd depthWeights; double lambda; int whitenerRank; std::vector<std::string> notes; };

Whitener whitener(const Eigen::MatrixXd& C, const std::vector<std::string>& types, const WhitenerOptions& = {});
Eigen::VectorXd depthWeights(const Eigen::MatrixXd& G, const std::vector<std::string>& types, const DepthOptions& = {});
Kernel unconstrainedKernel(const Eigen::MatrixXd& G, const Eigen::MatrixXd& C, const std::vector<std::string>& types, const KernelOptions& = {});
```

- **Defaults** reproduce Brainstorm's default unconstrained current kernel.
- **MNE-Python's** is reproduced by `mneKernelOptions()` = `{whitener = {form Pca, jointTypes, ridgeOverRank}, depth = {exponent 0.8, norm Spectral, limitOn Norm, channels OneType}, snrNormalisation = WhitenerRank}`.
  `OneType` takes `MEG GRAD` if present, else `MEG MAG`, else `EEG`, else all channels.
- **Errors:** a G with columns not divisible by 3, mismatched sizes, NaN, or types of the wrong length throw `std::invalid_argument`.

Storage (later, through the existing importer's rows `nxr::bst::export_inverse`):
- the kernel is Operator `<k>` kind `inverse kernel`, float32, vector3 to the cortex;
- with `<k>_channels`, `<k>_noise_cov` and `<k>_whitener` beside it;
- `params_json.inverse` records the options above under these names.

## 4. Parity

All inputs are synthetic:
- **Gain:** a real forward model, from MNE-Python's sphere model on synthetic sensors and sources: 12 magnetometers and 12 planar gradiometers on a 0.12 m sphere, 10 EEG
  electrodes at 0.09 m, 42 sources of an icosphere at 0.07 m, free orientation (a size chosen to keep the committed golden under 1 MB; the size of the problem does not enter the maths).
- **Covariance:** a full-rank correlated C with per-type scales. The rank-deficient cases have the EEG average reference and 2 SSP vectors on the magnetometers applied to G and C
  (the caller's job); MNE-Python is given the same projectors and the projected C.
- **Inputs are rounded to 12 significant digits first**, so that both references and nxr read the same numbers as the golden file.

| reference | how | cases | tolerance (max-abs / max) |
|---|---|---|---|
| Brainstorm 2018 linear inverse | MATLAB, `bst_inverse_linear_2018` called headless (its toolbox on the path; no database, no GUI) with synthetic structs (free orientation, amplitude measure); `matlab -batch`, **one MATLAB run at a time** (`operations/scripts/mem-guard.sh`) | default on all four cases (MEG only and MEG+EEG; full and rank-deficient); depth off, `Diagonal` and `None` on MEG+EEG | K 1e-10, W 1e-10 (the 12-digit inputs limit W, whose entries reach 1e13). Λ is not an output of the function, so it is checked through K |
| MNE-Python 1.13 | `mne.cov.regularize(mag=grad=eeg=0.1)`, `make_inverse_operator(loose=1, depth=0.8, fixed=False)` + `prepare_inverse_operator(method='MNE', lambda2=1/9)`, and `_assemble_kernel` (nxr's environment, `uv run`) | the four cases, with depth, and without depth (MEG only and MEG+EEG; full and rank-deficient C) | K 1e-9 with `mneKernelOptions()` |

Goldens are text (`%.17g`) in `meeg/test/golden/` with their generator scripts. The C++ test is `meeg/test/test_inverse.cpp` (`ctest -R meeg`).

## 5. What follows (later specs)

- Forward models still to come: BEM and FEM head models (external programs; they keep coming in by import), homogeneous sEEG, volume source grids, and calling the CTF reference compensation (section 8; its coefficients are not imported yet).
- dSPM and sLORETA measures.
- Orientation priors: fixed and loose.
- Shrinkage and other regularised covariance estimates (the sample covariance of section 7 is the first).
- Loose and fixed orientations; SSP projectors and the average reference read from the dataset (today the average reference is a pipeline option and SSP is the caller's).

## 6. The blocking decision (resolved)

**Depth weighting in step 1.** Brainstorm's default unconstrained kernel is depth-weighted (exponent 0.5, limit 10). Matching that default needs §2.2 in step 1. Should it be
in? *Decided by the maintainers: yes.* It is one scalar per source, and without it the default output cannot equal Brainstorm's. Rheome's `inverse.mne` has none; that
difference is recorded for its later alignment.

## 7. The noise covariance from a recording

**In.** A recording [channels × samples] stored in chunks, a time window [start, stop) in seconds, the good channels of the imaging types.
**Out.** C [M × M] (channel units squared), the number of samples N it holds, the number of blocks B.

**Maths.** The data are cut in B blocks (by default the whole window is one block; `block_seconds` cuts it into blocks of that length). Each block b has its own mean m_b (the DC offset is
removed per block, as the reference removes it per data file):

    C = [ Σ_b Σ_{t in b} (x_t − m_b)(x_t − m_b)ᵀ ] / D

**Chunk by chunk.** Per block the accumulator keeps S = Σ (x − p)(x − p)ᵀ, s = Σ (x − p) and n, with p the block's first sample (a shift, for the numerical stability of recordings with large
offsets); the block's sum about its mean is S − s sᵀ/n. A chunk is seen once, a chunk never crosses a block boundary, and nothing but a chunk and the M × M sums is held.

**Options** (`CovarianceOptions`, and for the pipeline `NoiseCovOptions`):

| option | default | meaning |
|---|---|---|
| `removeMean` | true | remove each block's mean; false: the raw second moment |
| `divisor` | `SamplesMinusBlocks` | D = N − B (the unbiased estimate with a mean removed per block; one block: N − 1). `Samples`: D = N. `SamplesMinusOne`: D = N − 1 |
| `block_seconds` | unset (one block) | the block length in seconds |
| `start`, `stop` | 0, end | the window, on the recording's own clock |
| `chunk_samples` | 16384 | samples read at once (memory: channels × chunk float64) |
| `channel_types` | MEG, MEG MAG, MEG GRAD, EEG, SEEG, ECOG | only good channels of these types (the recording's own flags, else the montage's quality) |

**Reference behaviour.**
- **MNE-Python** (`compute_raw_covariance(tstep=None, method='empirical')`): one block, mean removed, D = N − 1. Parity: max relative difference 9e-12 on a synthetic recording with offsets of 1e3
  (MNE's own rounding; the test tolerance is 1e-9).
- **Brainstorm** (`bst_noisecov`): it is driven by the Brainstorm database (studies, data files), and `brainstorm nogui` did not start headless on this machine, so no Brainstorm golden was made. Its
  divisor is therefore **not verified**: the default `SamplesMinusBlocks` is the standard unbiased choice and is named as an option, to be checked against a Brainstorm-computed covariance (open point).

**Stored** as an Operator of kind "noise covariance" on the channels, `<recording>_noise_cov`, float64 [M × M], from and to the Selection `<name>_channels` (indices of the channels it covers); its
`producer_json` records the window (seconds and samples), N, B, the options, the divisor value, the channel types and the cache key.

## 8. The forward model (sphere head models)

**Goal.** When a dataset has no head model from import, compute one: the unconstrained gain G [M × 3N], channels as rows, x, y, z of each source (a unit dipole, A·m) as columns 3k..3k+2, in SI units
(T/(A·m) for MEG, V/(A·m) for EEG), for sources at the vertices of a surface. Library `meeg/include/nxr/meeg/{forward,sphere,minimise}.h` (Eigen only); pipeline `headmodel_compute`; command
`nxr-neuro headmodel compute`. Not here: OpenMEEG BEM and DUNEuro FEM (external; imported), sEEG, volume grids. The average reference of EEG stays with the inverse step (§1).

**Written from** Sarvas 1987; Huang, Mosher & Leahy 1999; Brody et al. 1973 and Zhang 1995 (the multi-shell series); Berg & Scherg 1994; Lagarias et al. 1998 (the simplex). MNE-Python (BSD) is read as a second
reference. **Brainstorm (GPLv3) was not read**; only its function help headers and its outputs were used.

**Channels.** A channel is a weighted sum of sensor points: G_ch = Σ_k w_k G_point(k) (a magnetometer's integration points, a gradiometer's coils with weights of opposite sign, an electrode's one point). From the
stored sensor geometry: `<channels>_coils` (points), `_coils_orientation`, and the csr operator `<channels>_coil_weights` [M × P]; a channel without coils uses its own point and `<channels>_orientation`.

**1. MEG, single sphere (Sarvas).** Sphere centre c; r, r₀ the sensor point and the source from c; Q the moment; a = r − r₀; F = a(r a + r² − r₀·r);
∇F = (a²/r + a·r/a + 2a + 2r) r − (a + 2r + a·r/a) r₀; B(r) = μ₀/(4πF²) [F Q×r₀ − (Q×r₀·r) ∇F]. With the orientation n, the gain row is the coefficient of Q in n·B.
μ₀ = `SarvasOptions::permeability` (4π·10⁻⁷). A radial dipole gives no field; the radial component is the primary current's Biot–Savart field (both tested). The sphere is fitted to a head surface (default inner skull).

**2. MEG, overlapping spheres (Huang et al.).** Sarvas per channel with the channel's own sphere. The sphere of a channel is fitted to the head surface (default inner skull) by weighted least squares, the weight of a
surface point i being w_i = Σ_k |weight_k| d_ik^(−p) over the channel's sensor points k (`SensorSphereOptions::weightExponent` p = 4). **Up to 3.4 % from Brainstorm's `bst_os`; its fit needs a realistic-head (BEM) lead field; to be revisited when a BEM exists.** The weighting here is inferred, not the reference's: the reference weights the
surface by a sensor-field criterion that is not reproduced by any weight of distance or angle tried (inverse distance, power 1–8; cosine; Gaussians; cones); p = 4 gave the smallest gain mismatch against the reference's
output. On the synthetic case our spheres differ from its by up to 4.3 mm in centre and give a gain within 3.4 % (max) of its overlapping-spheres gain, against 12 % for one sphere. Given the reference's spheres, our Sarvas
gives its gain to 1e-12 (so only the sphere fit differs).

**3. Sphere fit** (`fitSphere`): points xᵢ, weights wᵢ. `SphereCost::Geometric` (default): minimise Σ wᵢ(|xᵢ − c| − R)², R the weighted mean distance, over c by Nelder–Mead (Lagarias et al.; the standard
reflection 1, expansion 2, contraction ½, shrink ½; ordered simplex; start the closed-form algebraic fit; the search is in units of the points' rms spread). `Algebraic`: Σ wᵢ(|xᵢ|² − 2c·xᵢ − k)², R² = k + |c|², linear.

**4. EEG, three-shell sphere.** Shells of relative radii ρ_1 < … < ρ_L (default 0.88, 0.93, 1.00: brain, skull, scalp) and conductivities σ_1..σ_L (0.33, 0.0042, 0.33 S/m): Brainstorm's defaults, read from its
documented options struct; `mneEegSphereOptions()`: 0.90, 0.92, 0.97, 1.00 and 0.33, 1.0, 0.004, 0.33. Centre and radius R of the outer shell: the sphere fitted to the scalp; electrodes are put on it
(`project_electrodes`, radially). The potential at electrode e of a dipole at r₀ is the homogeneous-sphere series with the n-th term multiplied by f_n (Zhang 1995; f_n = 1 for one shell, by the transfer-matrix
recursion `multiShellCoefficients`). Berg–Scherg: f_n ≈ Σ_j λ_j μ_j^(n−1), n = 1..200, J = 3, Σλ_j = f_1; the gain is
V = (1/σ_L) Σ_j λ_j (m₁ μ_j r₀ + m₂ e) / (4π |μ_j r₀|²) · Q, with m₁, m₂ the closed-form terms of a dipole at μ_j r₀ in a homogeneous sphere (`homogeneousPotentialGradient`; its limit at the centre is used within
`centreTolerance` = 10⁻⁶). The fit: μ_j in (−1, 1) by Nelder–Mead from `starts` = 12 deterministic starts (best kept; the fit has local minima), λ_j by linear least squares, weights wₙ = √((2n+1)(3n+1)/n) q^(n−1),
q = ρ_1/ρ_L, n ≥ 2. `EegApproximation::Series` evaluates the series itself (`seriesTerms` = 200). The formula is evaluated as it stands for any source (the reference does the same); the pipeline records
`sources_outside_innermost_shell` and `sources_outside_sphere`. **Parameters.** The fit depends only on the relative radii and conductivities. For Brainstorm's default layers and for MNE-Python's default layers the DEFAULT uses the references' own fitted parameters as
recorded constants (`recordedBergScherg`, `useRecordedBerg = true`), obtained by running `bst_berg` and `make_sphere_model` and copying their output (not from any source): the default gain then equals `bst_eeg_sph`'s and MNE-Python's (1e-12)
with no parameter passed in. For any other layers, or with `useRecordedBerg = false`, our multi-start fit (`fitBergScherg`) runs; it is the more accurate approximation of the exact series (3.5e-4 against 6.1e-3 for Brainstorm's
and 2.3e-3 for MNE-Python's, relative Frobenius) and differs from the references' gains by 0.2-0.6 %.

**5. CTF reference compensation:** `referenceCompensatedGain(G_MEG, C, G_REF)` = G_MEG − C·G_REF. Nothing calls it yet.

**Options** (every constant is one; defaults above): `SarvasOptions{permeability}`, `SphereFitOptions{cost, minimiser}`, `SensorSphereOptions{weightExponent, fit}`, `BergFitOptions{dipoles, terms, starts, minimiser}`,
`EegSphereOptions{radius, conductivity, approximation, berg, useGivenBerg, given, seriesTerms, centreTolerance}`, `NelderMeadOptions{reflection, expansion, contraction, shrink, relativeStep, zeroStep, xTolerance,
fTolerance, maxEvaluations}`; the pipeline's `HeadModelOptions{channels, source_surface, meg_surface, eeg_surface, meg (none | single | overlapping), eeg, name, project_electrodes, replace, …}`.

**Pipeline.** Reads the channels (a points manifold with a `type` labelling), the cortex (or `--source-surface`) vertices as sources, the sensor geometry and the head surfaces; **refuses** when the inner skull
(MEG) or the scalp (EEG) cannot be found, listing the surfaces the subject has and naming `--meg-surface` / `--eeg-surface`; refuses a surface in another coordinate system, a MEG point without an orientation, a NaN.
MEG, EEG-only or both; channels of other types get zero rows (an SEEG channel in a later noise covariance would be rank-deficient: choose types). Stored as `export_forward` stores an imported head model:
Operator `forward` `<name>` (default `headmodel`), float32 [M × 3N], from the cortex manifold (vector3) to the channels (scalar), `<name>_GridLoc`; `producer_json` has the importer's keys (`meg_method` `os_meg` |
`meg_sphere`, `eeg_method` `eeg_3sphereberg`) and `producer` "nxr-neuro … headmodel compute", the options, the fitted spheres and Berg–Scherg parameters; `params_json.forward` = {options, computed}. Cached
(§ pipelines/cache.h). `inverse compute` uses it as any head model (tested: headmodel → noisecov → inverse).

**Parity** (synthetic: 43 sources, 13 MEG channels of 2–8 integration points, 8 electrodes; goldens `meeg/test/golden/forward_golden.txt`, generator `gen_forward_golden.py` + `.m`, MATLAB `-batch`, one run at a time).
Error = max|a − b| / max|b|. `ctest -R meeg_test_forward`.

| model | reference | error |
|---|---|---|
| MEG single sphere | Brainstorm `bst_meg_sph`; MNE-Python `_sphere_field` | 1e-12 · 1e-12 (sources outside the sphere too); tolerance 1e-10 (the 12-digit inputs) |
| MEG overlapping spheres, reference's spheres | `bst_meg_sph` with `bst_os` spheres | 1e-12 |
| MEG overlapping spheres, our spheres | `bst_os` + `bst_meg_sph` | 3.4e-2 (see 2); tolerance 1e-1 |
| EEG f_n | `bst_berg` f; MNE-Python | 4e-13 |
| EEG, given Berg parameters | `bst_eeg_sph`; `_eeg_spherepot_coil` (MNE layers) | 2.5e-12 · 2.3e-12 |
| EEG one shell (series) | `bst_eeg_sph` | 3.7e-6 (the reference's own truncation); tolerance 1e-5 |
| EEG, default options (recorded parameters), Brainstorm / MNE layers | `bst_eeg_sph` / `_eeg_spherepot_coil` | 2.5e-12 / 2.3e-12; tolerance 1e-10 |
| EEG, our own fit (`useRecordedBerg = false`) | `bst_eeg_sph` · the exact series | 6.2e-3 · 3.5e-4; tolerances 2e-2 · 2e-3 |
| EEG, our own fit, MNE layers | `_eeg_spherepot_coil` | 1.5e-3; tolerance 2e-2 |

Unit tests (`test_forward`): the simplex on Rosenbrock, the sphere fit (exact, noisy, errors), Sarvas (radial dipole, Biot–Savart radial field, translation, rotation, channel sums, errors), the EEG series
(one shell = closed form, equal conductivities = 1, Berg–Scherg against the series, the centre limit, errors).

**8a. Why the two reference fits are not reproduced (follow-up, same branch).** Tried, with Brainstorm's function outputs only (`bst_os`, `bst_berg` called with varied inputs; no source read):
- *Overlapping spheres.* `bst_os` returns, besides the sphere, a per-vertex `Weight` that is signed (range -71..64, two exact zeros at the nearest vertices, no correlation with distance) and an `Approx`
  of the same length; the sphere is therefore not a weighted least-squares fit of vertices. This matches the method of Huang et al. (1999), whose sensor-fitted sphere minimises the difference between the sensor's lead
  field of the REAL head (a boundary-element solution: the signed weights are the surface potentials) and the sphere's, over a grid of test dipoles. Reproducing it needs a single-compartment BEM, which is out
  of scope here (BEM is imported). Weights tried against its gain (best Frobenius error): d^-4 3.7 %, the sensor's Sarvas lead-field norm squared 3.8 %, Biot-Savart norm 4.6 %, cones and Gaussians 3-12 %.
  `d^-p` stays the default and the stated approximation. The residual (3.4 % max) is the missing real-head lead field, not the minimiser.
- *Berg-Scherg.* `bst_berg` runs a simplex over (mu_1..3, lambda_2..3) of an objective `zhang_fit(R, f)` that is not published; its result is the same for nmax >= 50. No weighting of the form n^a, q^n, 1/f_n^2, or a truncation
  of the series makes its parameters stationary, so its objective is not one of these. Its fit is also a worse approximation (6.1e-3 against the series; ours 3.5e-4). One start with the documented
  `fminsearchOptions` converges to our optimum, not to its parameters. Hence the recorded constants of section 4.
