// nxr-pipelines — MEG/EEG source imaging: the noise covariance of a recording, and the minimum-norm kernel of a head model, each read with nxr-io, computed
// with nxr-meeg (Eigen only), stored with nxr-io, CACHED (see cache.h). The twins of what the Brainstorm importer stores (`nxr::bst::export_inverse`), so a reader
// cannot tell a computed kernel from an imported one except by its producer string.
//
//   noisecov_compute   a recording's good channels over a time window -> Operator "noise covariance" `<recording>_noise_cov` on the channels (channels -> channels,
//                      on the Selection `<name>_channels` of the channels it covers), float64 [M x M]. The recording is read CHUNK BY CHUNK (nothing but a chunk is held).
//   inverse_compute    a head model (Operator "forward", the gain) + a noise covariance (stored, or computed from a recording first) -> Operator "inverse kernel"
//                      `<name>` float32 [3N x M] (vector3 to the cortex), its Selection `<name>_channels`, and beside it `<name>_noise_cov`, `<name>_whitener` and the
//                      source geometry `<name>_source_rr/_nn`, with `params_json.inverse` in the importer's shape (`nxr::bst::inverse_method_of`).
//   headmodel_compute the sphere head model of a subject: its cortex surface (the sources), the channels' sensor geometry as the importer stores it (the points, their
//                      orientations, the coils and the coil weights) and the head surfaces the spheres are fitted to -> Operator "forward" `<name>`, the gain float32
//                      [M x 3N] (channels <- cortex, vector3 -> scalar), with `<name>_GridLoc`: what `export_forward` stores for an imported head model, the producer saying it
//                      was computed, `params_json.forward` the model, every option and the fitted spheres. MEG: Sarvas (one sphere) or overlapping spheres; EEG: three shells.
//   (the sensor PSD and the rotary spectrum are the definitions `psd` and `rotary_psd` (pipelines/definitions/), run by the runner on the nodes welch.psd, rotary.pair and welch.complex_psd)
//   filter_compute    a temporal filter (lowpass · highpass · bandpass · bandstop · notch; nxr/filter.h) of a recording, read CHUNK BY CHUNK, written as a NEW recording on the same
//                     channels and the same time line, exactly as recordings are stored (the grid of chunks and shards, the envelope; rows and arrays together): `<recording>_<kind>`,
//                     its Field derived from the source, the source's channel flags, events and bad segments (Selections) copied unchanged, its projectors and CTF compensation carried
//                     into `producer_json`, which also records the filter kind, every option, the option set and the design.
//   sourcepsd_compute the band power maps of a recording's SOURCES, Brainstorm's way (the PSD (Welch) computed on sources, then "Group in frequency bands": the mean of the PSD over each band's bins,
//                     optionally "Spectrum normalization: relative"): the recording is read CHUNK BY CHUNK, the sensor cross-spectral density accumulated per band (nxr::welch::CrossWelch), and the
//                     source PSD is diag(K S K^T) with the three orientations of a free kernel summed. One Field per band `<recording>_<kernel>_psd_<band>` (kind band power, scalar, on the kernel's
//                     source manifold, folder source), the band a Selection on the Frequency manifold (`producer_json.band`), as the Brainstorm importer and nxr store a band map.
// The maths and every option: docs/specs/2026-11-10-neuro-forward-inverse.md (the PSD: nxr/welch.h; the filters: nxr/filter.h).
#pragma once
#include "nxr/timebank.h"
#include <Eigen/Dense>
#include <cstdint>
#include <limits>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/meeg/covariance.h"
#include "nxr/meeg/forward.h"
#include "nxr/meeg/inverse.h"
#include "nxr/meeg/sources.h"
#include "nxr/filter.h"
#include "nxr/opticalflow.h"
#include "nxr/opticalflow_methods.h"
#include "nxr/quadrature.h"
#include "nxr/pipelines/cache.h"
#include "nxr/welch.h"

namespace nxr::pipelines {

// The channel types source imaging uses (the default of `channel_types`).
std::vector<std::string> imaging_channel_types();


enum class MegModel { None, SingleSphere, OverlappingSpheres };

struct HeadModelOptions {
  std::optional<std::string> channels;                    // the channels manifold by name; default: the subject's only points manifold with a 'type' labelling
  std::optional<std::string> source_surface;              // the sources: the vertices of this surface; default the primary (cortex) surface
  std::optional<std::string> meg_surface;                 // the surface the MEG spheres are fitted to; default: the one called inner skull
  std::optional<std::string> eeg_surface;                 // the surface the EEG sphere is fitted to; default: the one called scalp (or head)
  MegModel meg = MegModel::OverlappingSpheres;            // Brainstorm's default MEG head model
  bool eeg = true;                                        // compute the EEG channels (three shells)
  std::optional<std::string> name;                        // default `headmodel`
  nxr::meeg::SarvasOptions sarvas;
  nxr::meeg::SphereFitOptions sphere_fit;                 // the single MEG sphere and the EEG sphere
  nxr::meeg::SensorSphereOptions sensor_sphere;           // the overlapping spheres
  nxr::meeg::EegSphereOptions eeg_sphere;                 // Brainstorm's layers, Berg-Scherg
  bool project_electrodes = true;                         // electrodes are moved radially onto the fitted EEG sphere (the formula puts them on the outer shell)
  bool replace = false;
};
struct HeadModelResult {
  bool hit = false;
  double seconds = 0;
  nxr::io::Json info;
  Eigen::MatrixXd gain;                                   // [M x 3N] (zero rows for channels of no computed type)
};
// A missing surface, sensor geometry or orientation is an nxr::io::Error that names what the subject has.
HeadModelResult headmodel_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const HeadModelOptions& opt,
                                  const std::function<void(const nxr::io::Json&)>& progress = {});

struct SourcePsdOptions {
  std::string recording;                                  // the recording Field by name
  std::optional<std::string> kernel;                      // the inverse kernel Operator by name; default: the subject's only one
  std::optional<std::string> name;                        // the stem of the fields' names; default `<recording>_<kernel>_psd`
  std::optional<std::string> chunks;                      // what is loaded: chunk "k", chunks "a-b", or "all" (the whole recording, a loop over its chunks); REQUIRED: the caller states the span (or start and stop)
  std::optional<double> start, stop;                      // or an explicit span [start, stop) in seconds on the recording's clock (both given)
  std::optional<double> segment_seconds = 4.0;
  std::optional<std::int64_t> segment_samples;            // wins when given
  std::optional<int> segment_level;                       // under the set `tiles`: the segment is 2^segment_level samples (wins over seconds)
  nxr::welch::WelchOptions welch = nxr::welch::brainstormPsdOptions(1.0);
  std::string option_set = "tiles";
  std::vector<nxr::meeg::Band> bands = nxr::meeg::brainstormBands();   // Brainstorm's default bands; the mean of the bins from lo to hi inclusive
  std::string normalisation = "none";                     // none · relative (each map divided by the source's total PSD: the sum over all bins, 0 to Nyquist)
  bool exclude_bad_segments = true;
  bool replace = false;
};
struct SourcePsdResult {
  bool hit = false;
  double seconds = 0;
  nxr::io::Json info;                                     // name, fields, bands, sources, segments ...
  Eigen::MatrixXd maps;                                   // [sources x bands] (empty on a cache hit)
};
// The kernel's channels are the ones read from the recording; a kernel on other channels than the recording is an nxr::io::Error.

// ── the cross-spectral TENSOR of sources: the vector analogue of the band power map ──
// Per source and band, the complex 3x3 Hermitian T = K_v (S + i Sq) K_v^T (nxr/meeg/tensor.h): the source current's band cross-spectral density between its
// x y z components (the kernel's ambient axes). Read with the same options as `sourcepsd` (the same segments, window, bands); a free-orientation kernel only.
//   Stored, per band b (the stem default `<recording>_<kernel>_tensor`):
//     `<stem>_<b>`               Field, value type hermitian3, float32 [N x 3 x 3 x 2] (real, imaginary), kind "cross-spectral tensor", unit A²·m²/Hz;
//                                its trace equals the `sourcepsd` band power of the same options exactly
//     `<stem>_stokes_<b>`        Field, quaternion [N x 4] = [S0, S1, S2, S3]: the TANGENTIAL Stokes parameters in the surface's canonical frames
//                                (e1 north, e2 west): S0 tangential power, S1 + i S2 linear polarisation (axis 1/2 arg from north, strength |S1 + i S2|),
//                                S3 circular (> 0 turning from north towards west, counter-clockwise about the outward normal); NaN at the frames' singular vertices
//     `<stem>_normalpower_<b>`   Field, scalar: n^T Re(T) n, the power along the normal (n = e1 x e2; the mesh normal where the frames are singular)
//     `<stem>_normalshare_<b>`   Field, scalar: the normal power over the trace (0 … 1)
//   Needs the surface's canonical frames (`frames compute`). Cached on the options, the kernel, the recording and the frames.

// ── the sensor CROSS-SPECTRUM of a recording: the root of every source cross-spectral quantity ──
// Per band, the Welch cross-spectral density of the channels, C = S + i Sq with C_ab = <X_a conj(X_b)> (X a segment's Fourier coefficients; nxr::welch::CrossWelch with the
// quadrature part), the mean over the band's bins, exactly what `sourcepsd`, `sourcetensor`, `quadrature` and `hodgepower` accumulate. Every source quantity of the band is K C K^T
// read through a kernel: the per-source tensor T_ii, and any block T_ij between two sources (nxr::meeg::crossTensor), without forming the 3N x 3N source cross-spectrum
// (rank <= channels). C does not depend on a kernel: one stored C serves every kernel whose channels it covers (MNE, dSPM, another noise covariance).
//   Stored, per band b (the stem default `<recording>_crossspectrum`), as the noise covariance is: Operator `<stem>_<b>` of kind "cross-spectral density", channels -> channels on the
//   Selection `<stem>_channels` (the channels it covers, in order; one for every band), value type complex, float64 [M x M x 2] (real, imaginary), unit (channel unit)^2/Hz.
//   The cache stamp is on the first band's Operator. Memory held: M x M complex per band, plus one chunk (channels x 2^15 float64).
struct CrossSpectrumOptions {
  std::string recording;                                  // the recording Field by name
  std::optional<std::string> channels_of;                 // an Operator from the recording's channels (an inverse kernel) whose channels (from_selection) C covers;
                                                          // default: the recording's good channels of `channel_types` (as `noisecov compute`)
  std::vector<std::string> channel_types = imaging_channel_types();
  std::optional<std::string> name;                        // the stem of the Operators' names; default `<recording>_crossspectrum`
  std::optional<std::string> chunks;                      // what is loaded: chunk "k", chunks "a-b", or "all" (the whole recording, a loop over its chunks); REQUIRED: the caller states the span (or start and stop)
  std::optional<double> start, stop;                      // or an explicit span [start, stop) in seconds on the recording's clock (both given)
  std::optional<double> segment_seconds = 4.0;
  std::optional<std::int64_t> segment_samples;            // wins when given
  std::optional<int> segment_level;                       // under the set `tiles`: the segment is 2^segment_level samples (wins over seconds)
  nxr::welch::WelchOptions welch = nxr::welch::brainstormPsdOptions(1.0);
  std::string option_set = "tiles";
  std::vector<nxr::meeg::Band> bands = nxr::meeg::brainstormBands();   // the mean of the bins from lo to hi inclusive, as sourcepsd
  bool exclude_bad_segments = true;
  bool replace = false;
  // THE ESTIMATOR (spec docs/specs/2026-11-13-cycle-pattern.md, stage 1, and its time-tile amendment of 2026-11-13): "welch" (the default above: the mean
  // over each band's bins of Welch segments) or "analytic": per band, the Meyer band's analytic signal (timebank, gain 2: Re z is the band-passed signal),
  // decimated by the constant-Q ladder's stride (constantq::strideFor at the band's low edge: a power of two, the samples kept those whose index on the
  // recording's LINE is a multiple of it), and Σ z zᴴ per TILE of level `level` (2^level samples of the line, counted from its first sample;:
  // time is one dyadic partition). A tile holding a bad sample, or cut by the span's end, is dropped. Stored per band as a series `<stem>_<band>_windows`
  // [W x M x M x 2] (the SUMS; producer_json.window_index = the tiles' ordinals on the line) and the level's tiling of the line, the closed-form Selection
  // `<stem>_windows`. A coarser window is a sum at read (crossspectrum_windows_read's `level`). The Welch options do not apply to it.
  // Memory: [bands x W x M x M] complex, W = span / 2^level: chunk 0 at level 10 with 268 channels is 37 MB a band; 10 minutes at 1200 Hz about 0.8 GB a band.
  std::string estimator = "welch";
  // THE FREQUENCY PARTITION of the Welch reduction (spec docs/specs/2026-11-15-band-resolved-cross-spectrum.md): "band" (the default: each band is one member, its bins averaged: the
  // statistic `<stem>_<band>`, unchanged) or "bins" (each bin of the band its own member, nothing reduced over frequency: ALSO writes `<stem>_<band>_bins`, float64 [M x M x F x 2],
  // from the same pass; the band statistic is the mean of its bins). Welch only. Memory: F more M x M complex matrices a band (alpha at 4 s: 17; 268 channels: 20 MB).
  std::string resolve = "band";
  std::optional<int> level;                               // analytic: the window's tile level, 0..15; default: the finest of the bands' constant-Q windows
                                                          // (constantq::windowFor: 6.8 cycles of the low edge rounded up to 2^k; alpha at 1200 Hz: level 10, 0.85 s)
  std::optional<double> window_seconds;                   // analytic: or the window in seconds, snapped to the nearest level (constantq::levelOf); not with `level`
};
// The windows of an analytic cross-spectrum (`crossspectrum compute --estimator analytic`): per stored window its Σ z zᴴ [M x M] and the samples summed.
struct CrossSpectrumWindows {
  std::vector<std::int32_t> channels;                     // the recording channel indices, in the order of the rows
  std::vector<Eigen::MatrixXcd> sum;                      // [W], each [M x M], Hermitian
  std::vector<std::int64_t> count;                        // [W] samples summed (at the decimated rate)
  std::vector<std::int64_t> index;                        // [W] each window's tile ordinal on the recording's line at `level` (dropped windows leave gaps)
  int level = -1;                                         // the tile level of the windows (as read: the stored level, or the coarser level asked); -1: a series stored before the tiles
  nxr::io::Json producer;
  std::vector<Eigen::MatrixXcd> value;                    // [W] each window's cross-spectrum in `scaling` (density · power), when one was asked (empty for stored)
  std::string scaling = "stored", unit;
};
struct CrossSpectrumResult {
  bool hit = false;
  double seconds = 0;
  nxr::io::Json info;                                     // name, operators, channels, bands, segments ...
  std::vector<std::int32_t> channels;                     // the recording's channel indices C covers, in order
  std::vector<Eigen::MatrixXd> S, Sq;                     // per band [M x M] (empty on a cache hit)
  std::vector<CrossSpectrumWindows> windows;                        // analytic: per band (empty on a cache hit)
  std::vector<std::vector<Eigen::MatrixXd>> bins_S, bins_Sq;        // resolve = bins: per band, per bin [M x M] (empty on a cache hit)
};
CrossSpectrumResult crossspectrum_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const CrossSpectrumOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

// One stored band of a cross-spectrum (an Operator "cross-spectral density" by name), optionally restricted to `channels` (recording channel indices, in the order wanted: a kernel's
// channels); a channel C does not cover is an nxr::io::Error.
struct StoredCrossSpectrum {
  std::vector<std::int32_t> channels;
  Eigen::MatrixXd S, Sq;
  nxr::io::Json producer;
  std::string scaling = "stored";                         // the scaling S and Sq are in (as asked), and their unit
  std::string unit;
};
// `scaling` (chosen at READ, nxr::spectral): stored · density · power; a cross-spectrum takes only the linear ones (a factor); one with no scaling record
// (nxr's chain scaling) is refused any but stored.
StoredCrossSpectrum crossspectrum_read(nxr::io::SubjectWriter& sub, const std::string& name, const std::optional<std::vector<std::int32_t>>& channels = std::nullopt,
                                       const std::string& scaling = "stored");
// The band's cross-spectrum bin by bin (`crossspectrum compute --resolve bins`, the statistic `<stem>_<band>_bins`), C(f) per bin as [M x M] complex, on `channels` in that order
// (as crossspectrum_read), with the bins' frequencies in Hz. io reads; the readings (coherency, phase slope, PSI) are nxr::meeg's (meeg/include/nxr/meeg/crossspectrum.h).
struct StoredCrossSpectrumBins {
  std::vector<std::int32_t> channels;
  std::vector<double> frequencies;                        // Hz, one per bin
  std::vector<Eigen::MatrixXcd> C;                        // [F], each [M x M], Hermitian
  nxr::io::Json producer;
  std::string unit;
};
StoredCrossSpectrumBins crossspectrum_bins_read(nxr::io::SubjectWriter& sub, const std::string& name, const std::optional<std::vector<std::int32_t>>& channels = std::nullopt);
// The windows of a stored analytic cross-spectrum (`<stem>_<band>_windows`), restricted to `channels` in that order (as crossspectrum_read).
// `scaling` stored (the sums only) · density · power (each window's value, from its sum: the band power sum / (2 count), then nxr::spectral).
// `level` (≥ the stored level): the windows rolled up to that coarser level, the Studio's envelope roll-up: window M is the sum of the stored tiles m with
// m >> (level − stored) = M, kept only when all 2^(level − stored) of them are stored (whole and free of bad samples).
CrossSpectrumWindows crossspectrum_windows_read(nxr::io::SubjectWriter& sub, const std::string& name, const std::optional<std::vector<std::int32_t>>& channels = std::nullopt,
                                                const std::string& scaling = "stored", std::optional<int> level = std::nullopt);

// ── the PHASE GRADIENT of a band: whether the activity moves across the cortex, which way, how fast ──
// The spectral form of nxr's phase reader (nxr.detect.core_phase: grad phi = Im(conj z grad z) / |z|^2, velocity -omega grad phi / |grad phi|^2), read from a
// STORED band cross-spectrum (`crossspectrum compute`) through a free-orientation kernel and the mesh's P1 gradient (the area-weighted mean of the face gradients,
// nxr::hodge::FaceOperators, projected on the vertex's tangent plane), with power weights over the band's segments and bins (nxr/meeg/wave.h). The signal:
//   normal    z_v = n_v^T J_v, the current along the mesh normal (the column current)
//   ambient   the current's x y z components, each read as a scalar and their sums added (no frame, the normal part included)   (default)
// STORED (the measurement pyramid, spec docs/specs/2026-11-14-measurement-pyramid.md step 5: a reduction over space is a measurement over a partition, never a per-vertex ratio;), per run (one band;
// the stem default `<recording>_<kernel>_phasegradient`, the band from the cross-spectrum): a byte-less Field `<stem>_<band>` (kind `phase gradient`, on the kernel's surface, by_operator the kernel) and, on the
// leaf partition `<surface> tiles L<D>` (`depth`; the ladder is made when the surface has none; `frames compute` first: the current is read in the canonical frames), ARRAY-BACKED measurements [leaf] of it
// (the node `<partition node>/phase/<name>/`), within the band (the Frequency manifold's `bands`, when the band is one of them), each a SUM over the leaf's vertices weighted by the vertex's lumped area a:
//   phase_power            Σ a P            P_v = Σ |z|² the band's power with the phase gradient's weights       (A²·m², /Hz for a density)
//   phase_gradient_power   Σ a Q            Q_v = Σ |∇z|² the power of the tangent gradient                         (A², per m²)
//   phase_current_1, _2    Σ a (H·e1), Σ a (H·e2)   the phase current H = Im(z̄∇z) in the canonical frames (e1 north, e2 west), rad/m × power
//   phase_current_magnitude  Σ a |j|        the length of the vertex's phase current
// Derived at read (`measure query`, `nxr::io::query_measurements`: ratios of the rolled-up sums): phase_gradient_1/2 = Σj / ΣP (rad/m, points AGAINST the travel), phase_gradient_magnitude, phase_gradient_direction_deg,
// travelling_index = |Σj| / √(ΣP ΣQ), phase_coherence = |Σj| / Σ|j|; the speed is ω over the magnitude. A vertex with no canonical frame is out of every sum. The time of a leaf is the cross-spectrum's own span
// (recorded in the Field's producer_json.time): no frame axis. The surface's coordinates must be in m or mm. Cached on the cross-spectrum, the kernel, the surface, the leaves and the signal. The per-vertex Fields of
// earlier runs (`<stem>_<b>`, `_velocity_`, `_speed_`, `_travelling_`, `_power_`, `_gradientpower_`) are no longer written; those already stored are untouched.
//
// AT A SPATIAL SCALE (`spatial_band`; spec docs/specs/2026-11-12-direction-timing.md;: one triangulation): everything on the INTRINSIC complex
// the surface's Hodge–Dirac modes are solved on, with their FULL FORMS (`eigen solve` stores both; nxr::intrinsic):
//   the band     the free kernel on the glued surface, its tangent part as a 1-form on the intrinsic edges (flat) and its normal part as a 2-form on its
//                faces, read into the modes by inner products with their forms, weighted by the band's gain on λ = μ², synthesised back (sharp) to a
//                vertex current (the x y z components: `signal` must be ambient)
//   the phase    a 1-form on the intrinsic edges: each edge's lag between its ends, arg T(tip, tail), T(u, v) = Σ_c k_c(u) C k_c(v)ᵀ from the stored C
//   the split    inner products with the modes' 1-forms: the exact part is d(timing), the coexact part the rotation, the rest outside the modes
// STORED (decision 1 of the measurement pyramid, 2026-11-14), per band and spatial band, stem `<recording>_<kernel>_phasegradient_<spatial band>`, no per-vertex Fields:
//   the fits (global over the modes, or functions of the edge lags' phase: no additive form) are ONE STATISTIC `<stem>_map_<b>`, [vertex of the glued surface, quantity], scalar, of the cross-spectrum statistic
//   (`of_statistic_id`), measure `phase_lag_split`, method `phase_split`, the bridge NaN; quantity: `timing` (s: φ/ω, the exact part's potential; a larger value LEADS: propagation runs from high to low), `propagation_x/y/z`
//   (rad/m: −∇φ, the exact part sharpened), `speed` (m/s: ω / |∇φ|), `stream` (rad: the rotation part's potential; extrema: vortex cores), `inconsistency` (0 … 1: per vertex, the share of ⋆1 |ω|² over its edges the
//   timing map leaves); `<stem>_modes_<b>`: the phase 1-form's coefficients in the modes' 1-forms, on the eigen axis. The leaves (`depth`, as the plain route's) carry array-backed measurements [leaf] of a
//   byte-less Field `<stem>_<b>` within the frequency band: `phase_power` (Σ a P, the band's power), `phase_flux` (the propagation's exact flux out of the leaf, rad) and `phase_circulation` (the phase field's exact
//   circulation round it, rad; a winding × 2π round a defect): the discrete divergence and Stokes theorems on the complex (nxr::intrinsic::tileFlux / tileCirculation; the extrinsic form of nxr's measure/integrals
//   for a vertex field is nxr::integrals), additive up the tile tree; the mean divergence / curl of a tile is the integral over its area.
// and the shares of the phase field's energy (timing · rotation · residual outside the modes) in the provenance and the result.
struct PhaseGradientOptions {
  std::string cross_spectrum;                             // the band's stored cross-spectrum (an Operator "cross-spectral density") by name, e.g. `rest_crossspectrum_alpha`
  std::optional<std::string> kernel;                      // the FREE-orientation inverse kernel by name; default: the subject's only one
  std::string signal = "ambient";                         // ambient (the full current: its x y z components; default) · normal (the normal current n.J)
  std::optional<std::string> spatial_band;                // modes (every mode, gain 1) · weyl:J (Weyl's Meyer octave J on λ = μ²) · weyl:all (each member); unset: the kernel as it is
  std::optional<std::string> package;                     // the Hodge–Dirac package of the spatial band (default: the kernel surface's own, hodgeDirac[__<surface>])
  std::string edge_lag = "frame";                         // each intrinsic edge's lag: frame (in the surface's own frame: the normal parts as scalars, the tip's tangent part carried to
                                                          // the tail along the edge, so folding is not read as phase) · axial (the ambient comparison modulo π: ½ arg T²) · ambient
                                                          // (the x y z components compared as they are: a current turning with a fold reads as a lag near π)
                                                          // · frame-normal / frame-tangent (the frame comparison of the normal part alone / of the tangent part alone)
  std::string fit = "inner";                              // the timing map: inner (inner products with the modes' 1-forms: timing · rotation · residual) · mesh (the weighted fit on the whole
                                                          // intrinsic complex, a sparse solve) · modes (the weighted fit within the package's exact modes, dense)
  std::string phase_of = "current";                       // the signal whose phase is taken: current (the band's current, a vector at each vertex, compared by `edge_lag`) · stream (its
                                                          // stream function, a scalar from the coexact coefficients: no vector, no frame)
  std::string fold_edges = "keep";                        // keep · axial (an edge whose ends' normals are opposed spans an unresolved fold: its lag read modulo π, so the surface frame's
                                                          // reversal across it is not read as phase; counted in the provenance) · all (every edge's lag modulo π: a real lag across one
                                                          // edge is far below π/2)
  std::string defects = "keep";                           // keep · exclude (a weighted fit: edges whose ends' normals are opposed, a fold-over of the mesh, or shorter than 0.1 × the
                                                          // median edge get weight 0; their count and the share of the phase field they carried are reported)
  std::string weight = "power";                           // a weighted fit's edge weight: power (|T|, the edge's cross-power) · coherence (|T| / √(P_tail P_tip))
  std::string field = "gradient";                         // the phase 1-form: gradient (each intrinsic edge's lag arg T; the only one built: a power-weighted fit is a solve, not inner products)
  std::optional<std::string> name;                        // the stem of the fields' names
  std::optional<int> depth;                               // the plain route's leaves: the level D of the surface's tile ladder (default: the finest level every piece reaches), as `pyramid build --depth`
  bool replace = false;
};
struct PhaseGradientResult {
  bool hit = false;
  double seconds = 0;
  nxr::io::Json info;
};
PhaseGradientResult phasegradient_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const PhaseGradientOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

// ── THE CONSISTENT CYCLE PATTERNS of a band at a spatial scale (spec docs/specs/2026-11-13-cycle-pattern.md, stage 2) ──
// From a stored per-window cross-spectrum (`crossspectrum compute --estimator analytic`) and a FREE kernel read into the package's full forms (as phasegradient
// --spatial-band: the normal current a 2-form, the tangent current a 1-form; no orientation assumed): per spatial band, B [n x M] (the band's coexact
// normal rows and every mode's tangent rows, each times its gain, kept where |g| ≥ 1e-3), C_Φ = B C Bᵀ per window and over the span (the window
// partition's sum over its counts), and its q leading eigenvectors: the most consistent cycle patterns, Re(m e^{iφ}) over a cycle. No loop over samples.
// Stored per band b as STATISTICS (step 5 of the measurement pyramid), OF the cross-spectrum statistic it decomposes: `<stem>_<b>` (measure cycle_pattern, method eigendecomposition),
// complex, axes vertex of the glued surface (the bridge NaN) x pattern i x quantity [stream S (the tangent part's divergence-free piece), potential A (its curl-free piece), normal ρ];
// `<stem>_<b>_modes`, complex, axes mode x pattern x part [tangent, normal] (the coefficients; 0 outside the band); with `timing_of`, `<stem>_<b>_lags`. No Field per pattern and quantity. Each pattern's phase origin: its frame of largest variance. The eigenvalues and
// trace per window and over the span, and each window's similarity |⟨m_w, m⟩| to the span's and to the first window's pattern, are in the provenance
// (shares λ / trace are derived at read). Memory: the windows' sums (W x M x M) and, per band, n x n complex.
struct CyclePatternOptions {
  std::string cross_spectrum;                             // a stored cross-spectrum series `<stem>_<band>_windows` by name (crossspectrum compute --estimator analytic)
  std::optional<std::string> kernel;                      // the free-orientation inverse kernel; default: the subject's only one
  std::optional<std::string> package;                     // the Hodge–Dirac package (with full forms) on the kernel's surface; default: as phasegradient
  std::string spatial_band = "weyl:all";                  // modes | weyl:J | weyl:all (every member of the Weyl bank)
  int patterns = 3;                                       // q, the leading patterns kept per band (at most the band's coefficients)
  // THE WINDOW (the time-tile amendment, 2026-11-13): the stored tiles rolled up to a coarser level at read. Default 4 s, experiment 13's window (the reference),
  // snapped to the nearest level (1200 Hz: level 12, 4096 samples, 3.41 s). Neither (both unset): the stored level.
  std::optional<int> level;                               // the tile level read (≥ the stored one)
  std::optional<double> window_seconds = 4.0;             // or seconds, snapped to the nearest level; ignored when `level` is given
  // THE LAGS OF EACH PATTERN (none by default): for each named field of a pattern, its phase 1-form on the intrinsic edges, w_e = arg(x_tip conj x_tail)
  // (the same meaning as phasegradient's arg T), split in the modes (intrinsic::splitPhase): the timing map φ / ω (s; a larger value leads; ω the band's
  // centre), the speed ω / |∇φ| (m/s), the inconsistency per vertex, and the timing · rotation · residual shares (provenance). Fields: stream · potential · normal
  std::vector<std::string> timing_of;
  std::optional<std::string> name;                        // the stem; default `<cross_spectrum without _windows>_cycle`
  bool replace = false;
};
struct CyclePatternResult {
  bool hit = false;
  double seconds = 0;
  nxr::io::Json info;
};
CyclePatternResult cyclepattern_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const CyclePatternOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

// ── the quadrature spectrum of sources ──
// The recording's sensor cross-spectrum per band, S_b = Re + i Im (Welch's, as sourcepsd; nxr::welch::CrossWelch with the quadrature part), its factor L (S_b = L L^H, rank <= channels) read through the
// kernel K as F = K L [(oN) x r complex, never N x N], and from F (nxr/quadrature.h; nxr's lagged.py):
//   quadratureFraction   ||Im C||_F / ||C||_F of the band's source cross-spectrum C = F F^H (0 for any zero-delay mixture): in the params of every field of the band
//   quadratureEnergy     per source, sum_j (Im C_ij)^2 over the other sources j and, for a FREE kernel, over the source's three rows i (x, y, z): Field `<stem>_energy_<band>` (kind quadrature energy)
//   quadraturePairs      the top k (default 3) eigenvalue pairs +- i sigma of Im C: p leads q by a quarter cycle (the pair turns p -> q, counter-clockwise, the rotary spectrum's +f), Im C ⊇ sigma (p q^T - q p^T),
//                        sigma its amplitude, share its share of ||Im C||_F^2 (2 sigma^2 / ||Im C||^2), rotary power at +f and -f (always + : the check): Fields `<stem>_pair<k>_p_<band>` and `_q_`
//                        (kind quadrature pair). The pair is a plane: (p, q) is fixed up to a rotation in it, and the gauge is the strongest row has q = 0, p > 0.
// FREE KERNEL (three orientation rows per source, x y z interleaved as the kernel is stored): the quadrature spectrum is taken over the 3N rows; the energy of a source is the sum of its three rows' energies;
// a pair is one pattern over the 3N rows, so its p and q are stored as vector3 Fields [N x 3] (the source's x, y, z components), unit norm over all 3N numbers. A fixed kernel: scalar Fields [N].
// Memory (free kernel, ~15k sources, ~270 channels): F, [Re F, Im F] and Q are each ~200 MB; one band at a time.
struct QuadratureOptions {
  std::string recording;                                  // the recording Field by name
  std::optional<std::string> kernel;                      // the inverse kernel Operator by name; default: the subject's only one
  std::optional<std::string> name;                        // the stem of the fields' names; default `<recording>_<kernel>_quadrature`
  std::optional<std::string> chunks;                      // what is loaded: chunk "k", chunks "a-b", or "all" (the whole recording, a loop over its chunks); REQUIRED: the caller states the span (or start and stop)
  std::optional<double> start, stop;                      // or an explicit span [start, stop) in seconds on the recording's clock (both given)
  std::optional<double> segment_seconds = 4.0;
  std::optional<std::int64_t> segment_samples;            // wins when given
  std::optional<int> segment_level;                       // under the set `tiles`: the segment is 2^segment_level samples (wins over seconds)
  nxr::welch::WelchOptions welch = nxr::welch::brainstormPsdOptions(1.0);
  std::string option_set = "tiles";
  std::vector<nxr::meeg::Band> bands = nxr::meeg::brainstormBands();   // Brainstorm's default bands; the mean of the bins from lo to hi inclusive
  int pairs = 3;                                          // the k strongest quadrature pairs per band
  bool exclude_bad_segments = true;
  // THE STORED FORM (step 5 of the measurement pyramid): the divergence, curl, speed and velocity are stored as additive SUMS per leaf of `<surface> tiles L<depth>` and per frame of the recording's tower tiling (`frame_level`), never as per-vertex time means
  std::optional<int> depth;                               // the leaves' level (default: the finest level every piece reaches; the ladder is made when the surface has none)
  int frame_level = -18;                                  // the tower level of a frame (86 400 * 2^L s; -18 = 0.33 s)
  std::optional<double> day_start_s;                      // the recording's first sample as seconds from local midnight (anchor day); default: the anchor is the recording
  bool tiles = false;                                     // also write the maps' tile summaries per level (nxr/pipelines/tiles.h): scalars area-weighted, the vector in the frame and the Levi-Civita forms, labelled
  std::vector<int> tile_levels;                           // the levels (empty: 1 .. tile_max_level)
  int tile_max_level = 6;
  bool tile_levi_civita = true;                           // the Levi-Civita form (a transport solve per tile centre)
  bool replace = false;
};
struct QuadratureResult {
  bool hit = false;
  double seconds = 0;
  nxr::io::Json info;                                     // name, fields, bands, per-band fraction and pairs ...
  std::vector<double> fraction;                           // per band (empty on a cache hit)
  Eigen::MatrixXd energy;                                 // [sources x bands]
  std::vector<std::vector<nxr::quadrature::QuadraturePair>> pairs;   // per band; the patterns over the kernel's rows (3N for a free kernel)
  std::vector<std::vector<double>> share;                 // per band, per pair: its share of ||Im C||_F^2
};
QuadratureResult quadrature_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const QuadratureOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

// ── Hodge–Dirac eigenmode power of source maps ──
// The recording's sensor cross-spectrum per band S_b (Welch, as sourcepsd), read through the FUSED weights B [2K x C] of the stored FREE-orientation kernel and the subject's stored
// Hodge–Dirac package (nxr.inverse.fused; nxr::hodge::hodgeWeights): power p_b = diag(B S_b B^T), [2K x bands]. Rows 0..K-1 the normal-current weights a (coexact modes; 0 on exact ones),
// rows K..2K-1 the flow weights b. Stored as TWO fields per band on the package's eigen axis (the mu manifold, K points): `<stem>_<band>_current` and `<stem>_<band>_flow`.
struct HodgePowerOptions {
  std::string recording;                                  // the recording Field by name
  std::optional<std::string> kernel;                      // the FREE-orientation inverse kernel Operator by name; default: the subject's only one
  std::optional<std::string> package;                     // the package's stem (`hodgeDirac` or `hodgeDirac__<surface>`); default: the one of the kernel's surface
  std::optional<std::int64_t> modes;                      // use the first N modes of the package (default: all)
  std::optional<std::string> name;                        // the stem of the fields' names; default `<recording>_<kernel>_hodge`
  std::optional<std::string> chunks;                      // what is loaded: chunk "k", chunks "a-b", or "all" (the whole recording, a loop over its chunks); REQUIRED: the caller states the span (or start and stop)
  std::optional<double> start, stop;                      // or an explicit span [start, stop) in seconds on the recording's clock (both given)
  std::optional<double> segment_seconds = 4.0;
  std::optional<std::int64_t> segment_samples;
  std::optional<int> segment_level;                       // under the set `tiles`: the segment is 2^segment_level samples (wins over seconds)
  nxr::welch::WelchOptions welch = nxr::welch::brainstormPsdOptions(1.0);
  std::string option_set = "tiles";
  std::vector<nxr::meeg::Band> bands = nxr::meeg::brainstormBands();
  bool exclude_bad_segments = true;
  bool replace = false;
};
struct HodgePowerResult {
  bool hit = false;
  double seconds = 0;
  nxr::io::Json info;
  Eigen::MatrixXd power;                                  // [2K x bands] (empty on a cache hit)
};
// A fixed-orientation kernel is an nxr::io::Error: the coefficients need the current's three components.
HodgePowerResult hodgepower_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const HodgePowerOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

struct FilterComputeOptions {
  std::string recording;                                  // the recording Field by name
  std::optional<std::string> name;                        // default `<recording>_<kind>`: _lowpass · _highpass · _bandpass · _bandstop · _notch
  nxr::filter::FilterOptions filter = nxr::filter::brainstormFilterOptions(1.0);   // kind, edges and the reference's design (sfreq is set from the recording)
  std::string option_set = "brainstorm";                  // which named set `filter` started from (recorded): brainstorm
  int harmonics = 0;                                      // notch with one line frequency: 0 = only the frequencies given, N > 0 = the first N multiples, -1 = every multiple below Nyquist
};
struct FilterComputeResult {
  bool hit = false;
  double seconds = 0;
  nxr::io::Json info;                                     // name, id, path, recording, kind, channels, samples, order, look_ahead ...
};
// The new recording keeps all the source's channels (the bad ones too) and all its samples; its bad segments are carried over, not left out of the filtering. A name already stored
// is an nxr::io::Error (choose another name, or remove the node); a rerun with the same options on the same recording is a cache hit.
FilterComputeResult filter_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const FilterComputeOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

// ── The apparent motion of a source map, read in the Hodge–Dirac modes ──
// Rheome's `flow.apparent` for a recording: the window is band-pass filtered (nxr/filter.h, Brainstorm's FIR, read with its margins), made analytic AT THE SENSORS (the kernel is real and
// linear, so hilbert(K x) = K hilbert(x)), carried through the stored kernel and reduced to the amplitude |J| per source (the analytic envelope: `flow.activation`), taken every
// `sfreq / rate` samples; then manifold Horn–Schunck optical flow per frame (`nxr::flow::opticalflowScalar`), and the velocity [3N x frames-1] is read in the subject's stored Hodge–Dirac
// package (`nxr::flow::flowCoefficients`, the fused weights of a field in place of a kernel): the EXACT coefficients (the divergence potential: sources and sinks of the motion) and the
// COEXACT coefficients (the stream function: its rotation).
//   STREAMED: the memory held does not depend on the window's length (apart from the stored outputs, [K x frames-1] and three maps). The wavelet runs twice over the window: once for the
//   max amplitude (Rheome's unit-amplitude scaling is over the whole window), once to solve; per frame only the previous and current maps and one velocity are held, the velocity read in the
//   modes at once and dropped, the divergence, curl and speed kept as running sums.
//   Stored: Field `<stem>_flow_modes` (kind "optical flow", float32 [K x frames-1]) on the product eigen axis x the window's time line `<stem>_time` (frame j at the middle of the step
//   between activation frames j and j+1), and, since step 5 of the measurement pyramid, the divergence, curl, speed and velocity as additive SUMS per leaf of
//   `<surface> tiles L<depth>` and per frame of the recording's tower tiling (`frame_level`): a byte-less Field `<stem>_flow` and the measures flow_divergence, flow_curl, flow_speed (1/s, 1/s, m/s),
//   flow_velocity_1 / _2 (the canonical frames) and flow_velocity_anterior / _second (the anatomical frame) on a joint Selection, `<measure>_density` read as the time mean per unit area (no per-vertex
//   Field of them is written; the Fields of earlier runs are untouched). The per-frame velocity is NOT stored. The warnings of Rheome are in producer_json.warnings and in nxr/opticalflow.h: kinematic not physical · aperture-limited (alpha) · per frame, not per second ·
//   edges of the window (the band-pass margins are read when the record has them) · the record's bad segments are NOT excluded (choose the window free of them).
struct OpticalflowComputeOptions {
  std::string recording;                                  // the recording Field by name
  std::optional<std::string> kernel;                      // the inverse kernel Operator by name (free or fixed orientation); default: the subject's only one
  std::optional<std::string> package;                     // the Hodge–Dirac package's stem; default: the one of the kernel's surface
  std::optional<std::string> name;                        // the stem of the fields' names; default `<recording>_<kernel>_opticalflow`
  std::optional<std::string> chunks;                      // what is loaded: chunk "k", chunks "a-b", or "all" (the whole recording, a loop over its chunks); REQUIRED: the caller states the span (or start and stop)
  std::optional<double> start, stop;                      // or an explicit span [start, stop) in seconds on the recording's clock (both given)
  // the time filterbank (nxr::timebank::Design): family and gain are REQUIRED (no default), sides is Analytic (a real recording); gain 2 makes |c| the physical amplitude
  std::optional<nxr::timebank::Family> family;
  double gain = 0;
  double freq = 0;                                        // the member used: the one whose centre is nearest to this frequency (Hz); required
  std::vector<double> centres;                            // Morlet, Morse, Amor, Bump: the member centres (Hz); empty for Morlet: {freq}
  double cycles = 6;                                      // Morlet: width fc / cycles
  int voices = 4;                                         // Itersine: voices per octave
  double anchor = 1.0;                                    // Itersine: the anchor frequency (Hz)
  std::optional<std::pair<double, double>> limits;        // Itersine, Meyer: the bank's frequency limits (Hz)
  double half_width = 1.0 / 3.0;                          // Meyer: transition half-width in octaves
  double gamma = 3, time_bandwidth = 60;                  // Morse
  double resolution = 0.5;                                // Hz: the bank is designed on a grid of nextFastLen(sfreq / resolution) samples
  double oversample = 2.0;                                // nxr::wt::Options::oversample: the member's own rate = nextFastLen(oversample * bins) fs / nfft
  std::optional<double> tolerance;                        // nxr::wt::Options::tolerance (the filters' impulse-response tail the margin may leave out); unset: from the recording's
                                                          // stored precision — float32 1e-7 (its epsilon, 1.2e-7), float64 1e-12: a stricter margin buys nothing the data hold
  double alpha = std::numeric_limits<double>::quiet_NaN();   // Horn–Schunck smoothness weight; NaN (unset): the regulariser's own default (connection: the Studio's 0.01; ambient: 1). Vary it and report the range
  // THE METHOD (nxr/opticalflow_methods.h): pcg (the default: Rheome's equation by preconditioned conjugate gradients) · direct (Rheome's exact solve, unchanged, for parity) · perscale; the options of the others below. Part of the cache key
  // (a direct run keeps the key it always had; pcg's differs from it). Progress events carry seconds_per_frame (and iterations), the result and the summary solve_seconds_per_frame (and iterations).
  std::string method = "pcg";
  std::string regulariser = "connection";                 // direct, pcg, perscale: the smoothness: connection (THE DEFAULT: the Cortical Flow Studio's connection Laplacian on the signpost gauge, its gauge data term) · ambient (the correct Lefevre and Baillet form; Brainstorm's implementation under-counts its right-hand side) (Lefevre and Baillet 2008: the Dirichlet energy of the ambient 3-vector field; frame-independent) · frame (Rheome's a'Ka + b'Kb of the frame coordinates; Rheome parity)
  std::string datafit = "auto";                           // direct, pcg, perscale: the data term: auto (the regulariser's own: gauge for connection, triangle for ambient) · gauge (the Studio's, connection only) · triangle (per-triangle finite elements, the paper's)
  std::string input = "amplitude";                        // the map: amplitude (|J| of the kernel applied to the complex coefficients) · signed (Re of them through a fixed-orientation kernel: the band signal itself, the paper's input; a free kernel is refused)
  double flow_tolerance = 1e-8;                           // pcg, perscale: relative residual
  int max_iterations = 500;                               // pcg, perscale
  std::string preconditioner = "ldlt";                    // pcg, perscale: ldlt (the exact factor of alpha K + c M, built once) or ic (incomplete Cholesky)
  std::string scale_family;                               // perscale (required): the graph filterbank, mexhat · itersine · logitersine; its band-pass members give one series each
  int scale_voices = 3, scale_filters = 0;                // perscale: the bank's voices per octave / member count (0: from voices)
  std::int64_t progress_frames = 25;                      // a "flow" progress event every this many frames
  // THE STORED FORM (step 5 of the measurement pyramid): the divergence, curl, speed and velocity are stored as additive SUMS per leaf of `<surface> tiles L<depth>` and per frame of the recording's tower tiling (`frame_level`), never as per-vertex time means
  std::optional<int> depth;                               // the leaves' level (default: the finest level every piece reaches; the ladder is made when the surface has none)
  int frame_level = -18;                                  // the tower level of a frame (86 400 * 2^L s; -18 = 0.33 s)
  std::optional<double> day_start_s;                      // the recording's first sample as seconds from local midnight (anchor day); default: the anchor is the recording
  bool tiles = false;                                     // also write the maps' tile summaries per level (nxr/pipelines/tiles.h): scalars area-weighted, the vector in the frame and the Levi-Civita forms, labelled
  std::vector<int> tile_levels;                           // the levels (empty: 1 .. tile_max_level)
  int tile_max_level = 6;
  bool tile_levi_civita = true;                           // the Levi-Civita form (a transport solve per tile centre)
  bool replace = false;
};
// perscale stores, per band-pass member `m00`, `m01`, ...: `<stem>_m00_flow_modes`, `<stem>_m00_divergence`, `_curl`, `_speed` (the per-scale exact / coexact split is in the coefficients and in the
// summary's `series`, with the member's centre wavenumber and mean speed). Every other method stores the four fields as before.
struct OpticalflowComputeResult {
  bool hit = false;
  double seconds = 0;
  nxr::io::Json info;
  Eigen::MatrixXd coefficients;                           // [K x frames-1] flow weights b: exact modes then coexact, in the package's order (empty on a cache hit)
  Eigen::MatrixXd divergence, curl, speed;                // [N x 1] time means (1/s, 1/s, length/s) (empty on a cache hit)
  Eigen::MatrixXd anterior, second;                       // [N x 1] the time-mean velocity along anterior and along the second anatomical axis (m/s); only when the surface has canonical and anatomical frames (nxr-neuro frames compute)
  std::vector<Eigen::MatrixXd> series_coefficients, series_divergence, series_curl, series_speed, series_anterior, series_second;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> series_velocity;   // per series: the time-mean embedded velocity [N x 3] (m/s); only with `tiles`   // per series (one, or one per perscale member); the members above are series 0
  double seconds_per_frame = 0;                           // the flow solve's wall time per frame pair (the filterbank excluded)
  // the stored leaf x frame sums, as written (empty on a cache hit): per series, per row (`frame_rows`: measure, `frame_rel`, `frame_units`) [leaf x frames of the run's span], and the steps each frame holds
  std::vector<std::vector<std::vector<double>>> frame_sums;
  std::vector<std::vector<double>> frame_counts;
  std::vector<std::string> frame_rows, frame_rel, frame_units;
};
OpticalflowComputeResult opticalflow_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const OpticalflowComputeOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines
