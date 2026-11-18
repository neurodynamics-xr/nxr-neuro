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
//   filter_compute    a temporal filter (lowpass · highpass · bandpass · bandstop · notch; nxr/signal/filters.h) of a recording, read CHUNK BY CHUNK, written as a NEW recording on the same
//                     channels and the same time line, exactly as recordings are stored (the grid of chunks and shards, the envelope; rows and arrays together): `<recording>_<kind>`,
//                     its Field derived from the source, the source's channel flags, events and bad segments (Selections) copied unchanged, its projectors and CTF compensation carried
//                     into `producer_json`, which also records the filter kind, every option, the option set and the design.
//   sourcepsd_compute the band power maps of a recording's SOURCES, Brainstorm's way (the PSD (Welch) computed on sources, then "Group in frequency bands": the mean of the PSD over each band's bins,
//                     optionally "Spectrum normalization: relative"): the recording is read CHUNK BY CHUNK, the sensor cross-spectral density accumulated per band (nxr::signal::spectral::CrossWelch), and the
//                     source PSD is diag(K S K^T) with the three orientations of a free kernel summed. One Field per band `<recording>_<kernel>_psd_<band>` (kind band power, scalar, on the kernel's
//                     source manifold, folder source), the band a Selection on the Frequency manifold (`producer_json.band`), as the Brainstorm importer and nxr store a band map.
// The maths and every option: docs/specs/2026-11-10-neuro-forward-inverse.md (the PSD: nxr/signal/spectral.h; the filters: nxr/signal/filters.h).
#pragma once
#include "nxr/signal/timefilterbank.h"
#include <Eigen/Dense>
#include <cstdint>
#include <limits>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "nxr/stats/covariance.h"
#include "nxr/meeg/forward.h"
#include "nxr/meeg/inverse.h"
#include "nxr/meeg/sources.h"
#include "nxr/select/bands.h"
#include "nxr/measurement/measures.h"
#include "nxr/signal/filters.h"
#include "nxr/signal/opticalflow.h"
#include "nxr/signal/opticalflow_methods.h"
#include "nxr/core/algebra.h"
#include "nxr/measurement/congruence_norms.h"
#include "nxr/geometry/mesh.h"
#include "nxr/signal/complex_signal.h"
#include "nxr/signal/crossspectral.h"
#include "nxr/signal/crossspectral_decomposition.h"
#include "nxr/signal/phase_gradient.h"
#include "nxr/stats/circular.h"
#include "nxr/measurement/congruence_norms.h"
#include "nxr/pipelines/cache.h"
#include "nxr/signal/spectral.h"

namespace nxr::pipelines {

/** each value of P [rows x bands] as a share of its row's total: the DERIVED measure `relative_power` (power_sum / total_power; 0 where the total is 0), read from the registry */
inline Eigen::MatrixXd relativeToTotal(const Eigen::MatrixXd& P, const Eigen::VectorXd& total) {
  if (total.size() != P.rows()) throw std::invalid_argument("relative power: one total per row");
  const nxr::measurement::Registry& reg = nxr::measurement::Registry::standard();
  Eigen::MatrixXd out(P.rows(), P.cols());
  for (Eigen::Index i = 0; i < P.rows(); ++i)
    for (Eigen::Index j = 0; j < P.cols(); ++j)
      out(i, j) = reg.derive("relative_power", [&](const std::string& n) { return n == "power_sum" ? P(i, j) : total(i); });
  return out;
}

/** a band preset (`nxr::select::band_preset`) as meeg's bands: the mean of the spectrum's bins from lo to hi inclusive */
inline std::vector<nxr::meeg::Band> bandsOfPreset(const std::string& name) {
  std::vector<nxr::meeg::Band> out;
  for (const nxr::select::BandSpec& b : nxr::select::band_preset(name)) out.push_back({b.name, b.lo, b.hi});
  return out;
}


// The channel types source imaging uses (the default of `channel_types`).
std::vector<std::string> imaging_channel_types();

struct SourcePsdOptions {
  std::string recording;                                  // the recording Field by name
  std::optional<std::string> kernel;                      // the inverse kernel Operator by name; default: the subject's only one
  std::optional<std::string> name;                        // the stem of the fields' names; default `<recording>_<kernel>_psd`
  std::optional<std::string> chunks;                      // what is loaded: chunk "k", chunks "a-b", or "all" (the whole recording, a loop over its chunks); REQUIRED: the caller states the span (or start and stop)
  std::optional<double> start, stop;                      // or an explicit span [start, stop) in seconds on the recording's clock (both given)
  std::optional<double> segment_seconds = 4.0;
  std::optional<std::int64_t> segment_samples;            // wins when given
  nxr::signal::spectral::WelchOptions welch = nxr::signal::spectral::brainstormPsdOptions(1.0);
  std::string option_set = "tiles";
  std::vector<nxr::meeg::Band> bands = nxr::pipelines::bandsOfPreset("brainstorm");   // Brainstorm's default bands; the mean of the bins from lo to hi inclusive
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
// Per band, the Welch cross-spectral density of the channels, C = S + i Sq with C_ab = <X_a conj(X_b)> (X a segment's Fourier coefficients; nxr::signal::spectral::CrossWelch with the
// quadrature part), the mean over the band's bins, exactly what `sourcepsd`, `sourcetensor`, `quadrature` and `hodgepower` accumulate. Every source quantity of the band is K C K^T
// read through a kernel: the per-source tensor T_ii, and any block T_ij between two sources (nxr::core::crossTensor), without forming the 3N x 3N source cross-spectrum
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
  nxr::signal::spectral::WelchOptions welch = nxr::signal::spectral::brainstormPsdOptions(1.0);
  std::string option_set = "tiles";
  std::vector<nxr::meeg::Band> bands = nxr::pipelines::bandsOfPreset("brainstorm");   // the mean of the bins from lo to hi inclusive, as sourcepsd
  bool exclude_bad_segments = true;
  bool replace = false;
  // THE ESTIMATOR (spec docs/specs/2026-11-13-cycle-pattern.md, stage 1, and its time-tile amendment of 2026-11-13): "welch" (the default above: the mean
  // over each band's bins of Welch segments) or "analytic": per band, the Meyer band's analytic signal (timebank, gain 2: Re z is the band-passed signal),
  // decimated by the constant-Q ladder's stride (select::constantq::strideFor at the band's low edge: a power of two, the samples kept those whose index on the
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
                                                          // (select::constantq::windowFor: 6.8 cycles of the low edge rounded up to 2^k; alpha at 1200 Hz: level 10, 0.85 s)
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
// (as crossspectrum_read), with the bins' frequencies in Hz. io reads; the readings (coherency, phase slope, PSI) are nxr::meeg's (meeg/include/nxr/signal/crossspectral.h).
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

// ── Hodge–Dirac eigenmode power of source maps ──
// The recording's sensor cross-spectrum per band S_b (Welch, as sourcepsd), read through the FUSED weights B [2K x C] of the stored FREE-orientation kernel and the subject's stored
// Hodge–Dirac package (nxr.inverse.fused; nxr::operators::hodgeWeights): power p_b = diag(B S_b B^T), [2K x bands]. Rows 0..K-1 the normal-current weights a (coexact modes; 0 on exact ones),
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
  nxr::signal::spectral::WelchOptions welch = nxr::signal::spectral::brainstormPsdOptions(1.0);
  std::string option_set = "tiles";
  std::vector<nxr::meeg::Band> bands = nxr::pipelines::bandsOfPreset("brainstorm");
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


}  // namespace nxr::pipelines
