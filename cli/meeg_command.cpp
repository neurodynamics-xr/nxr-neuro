// nxr-neuro noisecov · inverse — MEG/EEG source imaging (nxr_meeg through the pipelines).
//
//   nxr-neuro noisecov compute --dataset D/N --subject S --recording R [--name N] [--chunks K|A-B|all | --start SEC --stop SEC] [--block-seconds SEC]
//                              [--types "MEG MAG,MEG GRAD,EEG"] [--no-remove-mean] [--divisor samples-minus-blocks|samples|samples-minus-one] [--replace]
//     The sample covariance of the recording's good channels over [start, stop), read chunk by chunk; the mean (DC offset) removed per block (the whole window is
//     one block unless --block-seconds). Stored as a Statistic (measure covariance) `<R>_noise_cov`.
//
//   nxr-neuro inverse compute --dataset D/N --subject S (--noise-cov NAME | --recording R [noise covariance options]) [--forward NAME] [--name NAME] [--replace]
//                             [--no-depth] [--depth-exponent P] [--depth-limit L] [--depth-norm frobenius|spectral] [--depth-limit-on weight|norm] [--depth-channels all|one-type]
//                             [--snr S] [--snr-normalisation all-channels|whitener-rank]
//                             [--regularisation ridge|diagonal|none] [--ridge-fraction E] [--joint-types] [--ridge-over-rank] [--whitener-form symmetric|pca]
//                             [--no-detect-diagonal] [--rank-tolerance single|double] [--no-average-reference] [--mne]
//     The unconstrained minimum-norm kernel with depth weighting. The defaults reproduce Brainstorm's default (exponent 0.5, limit 10, SNR 3, regularisation 'reg' 0.1);
//     --mne selects MNE-Python's conventions (the options after it still apply). Stored as the importer stores a kernel: Operator "inverse kernel" `<name>` with
//     `<name>_channels`, `<name>_noise_cov`, `<name>_whitener`, `<name>_source_rr/_nn`.
//
//   nxr-neuro headmodel compute --dataset D/N --subject S [--name NAME] [--channels NAME] [--source-surface NAME] [--meg overlapping|single|none] [--no-eeg]
//                               [--meg-surface NAME] [--eeg-surface NAME] [--sphere-cost geometric|algebraic] [--sensor-weight-exponent P] [--no-project-electrodes]
//                               [--eeg-radii r1,r2,r3] [--eeg-conductivities s1,s2,s3] [--mne-eeg-layers] [--eeg-series] [--eeg-series-terms N] [--berg-dipoles J]
//                               [--berg-terms N] [--berg-starts N] [--replace]
//     The forward model (gain, unconstrained, T/(A.m) for MEG and V/(A.m) for EEG) of sphere head models, for the cortex surface's vertices: MEG by Sarvas in one sphere or per-sensor
//     overlapping spheres (fitted to the inner skull; Brainstorm's default), EEG by the three-shell Berg-Scherg sphere (fitted to the scalp; Brainstorm's layers). Reads the subject's
//     sensor geometry (points, orientations, coils, coil weights) and surfaces; a missing surface is refused with the subject's surfaces listed (name one with --meg-surface /
//     --eeg-surface). Stored as an imported head model is: Operator "forward" `<name>` (default `headmodel`) + `<name>_GridLoc`; `inverse compute` uses it.
//
//   nxr-neuro psd compute --dataset D/N --subject S --recording R [--name N] [--chunks K|A-B|all | --start SEC --stop SEC]
//                         [--set tiles|brainstorm|scipy|nxr-chain] [--segment-seconds SEC | --segment-samples N] [--overlap F] [--window hann|hamming|blackman|rectangular]
//                         [--periodic | --symmetric] [--detrend none|constant|linear] [--scaling density|spectrum|chain] [--no-bad-segments] [--through OPERATOR] [--replace]
//     --through OPERATOR: the ROTARY SPECTRUM. OPERATOR is a stored [2 x C] Operator over the recording's channels (the subject's `<kernel>_rotationPlane`, rows fx, fy); z = fx.X + i fy.X per block, and the
//     statistic stored is the TWO-SIDED Welch PSD of z over the same segments (axes: pair (1) x frequency on a signed line, -fs/2 ... fs/2; +f = x turning towards y). Default name `<recording>_<OPERATOR>_psd_welch`.
//     The Welch power spectral density of every channel of the recording, read chunk by chunk (the segments cross chunk edges: the result does not depend on the chunk width), the
//     recording's bad segments left out (a segment touching one is skipped). --set picks the reference whose settings are the starting point (default tiles, 2026-11-13: segments
//     are TILES, 2^k samples on the recording's line, k = round(log2(--segment-seconds x fs)) or --segment-level, half overlapping, with Brainstorm's Hamming (symmetric) taper, mean
//     removed, power per Hz, and the span snapped to level k; brainstorm: Brainstorm's exact 4 s (round(4 fs) samples) from the span's start; scipy: Hann (periodic), constant
//     detrend; nxr-chain: nxr.spectral.chain Welch); the options after it override. Under tiles the overlap keeps the hop a tile (0, 0.5, 0.75, ...).
//     Stored as a STATISTIC (docs/specs/2026-11-13-statistics.md): `<R>_psd_welch`, power_density by welch, OF the recording, from its segment partition `<name>_segments`,
//     axes channel x the frequency line `<R>_freq`, float32 [channels x frequencies] beside the recording (`<R>_statistics/`).
//
//   nxr-neuro run surface_frames --dataset D/N --subject S [--surface NAME] [--template SUBJECT [--template-surface NAME]] [--convention scs|ras] [--side auto|left|right] [--min-projection P] [--max-condition C] [--replace]
//     The cortical frames of a surface (nxr/geometry/frames.h): the CANONICAL frames (north / west; read when stored as `<surface>_canonical_frames`, else computed from the registered sphere `<surface>_sphere`)
//     and the ANATOMICAL frame (one angle from north to anterior per vertex, the second axis superior-or-medial, a reliability mask). The anatomical axes are carried from --template (its own frame made by
//     `frames compute` on the template, with --convention for its coordinates) through the registered spheres, or, without it, are the surface's own SCS axes projected on its tangent planes.
//
//   nxr-neuro hodgepower compute --dataset D/N --subject S --recording R [--kernel NAME] [--package STEM] [--modes N] [--name STEM] [--bands name:lo-hi,...] [the psd options] [--replace]
//     The band power of the Hodge-Dirac eigenmode coefficients of a recording's sources: the stored FREE-orientation kernel is fused into the subject's stored Hodge-Dirac package
//     (nxr.inverse.fused: B [2K x C] = [normal-current weights a; flow weights b]), power = diag(B S_b B^T) from the band cross-spectrum. Stored as two Fields per band on the package's
//     eigen axis (K points): `<stem>_<band>_current` (coexact modes) and `<stem>_<band>_flow`. A fixed-orientation kernel is refused.
//
//   nxr-neuro crossspectrum compute --dataset D/N --subject S --recording R [--channels-of OPERATOR | --types "MEG MAG,MEG GRAD,EEG"] [--name STEM] [--chunks K|A-B|all | --start SEC --stop SEC]
//                         [--bands name:lo-hi,...] [the psd options] [--estimator welch|analytic [--level K | --window-seconds SEC]] [--resolve band|bins] [--replace]
//     The sensor cross-spectral density of the recording per band (Welch, as sourcepsd: the mean over the band's bins, with its quadrature part), the root of every source
//     cross-spectral quantity (K C K^T through any kernel). Its channels: those of --channels-of (an inverse kernel), else the good channels of --types. Stored as the noise
//     covariance is: Operator "cross-spectral density" `<stem>_<band>` (stem default `<R>_crossspectrum`), complex, float64 [M x M x 2], on the Selection `<stem>_channels`.
//     --estimator analytic (spec docs/specs/2026-11-13-cycle-pattern.md, as amended 2026-11-13: the time tiles): per band its Meyer analytic signal, decimated by the
//     constant-Q ladder's stride (a power of two, on the recording's line), and Σ z z^H per TILE of --level K (2^K samples of the line; default: the finest of the bands'
//     constant-Q windows; or --window-seconds snapped to the nearest level; at most 15). The span snaps to the tiles; a tile with a bad sample is dropped: Operator
//     "cross-spectrum series" `<stem>_<band>_windows` [W x M x M x 2] (the sums; window_index = the tiles' ordinals) and the closed-form Selection `<stem>_windows` (the
//     level's tiling of the line). A coarser window is a sum at read. The Welch options do not apply.
//     --resolve bins (spec docs/specs/2026-11-15-band-resolved-cross-spectrum.md; Welch only): the same pass also stores, per band, the STATISTIC `<stem>_<band>_bins`, the band's
//     cross-spectrum bin by bin (axes channel x channel x frequency in Hz, float64 [M x M x F x 2]); the band statistic is the mean of its bins. Default band: as before.
//   nxr-neuro crossspectrum read --dataset D/N --subject S --name <stem>_<band>_bins (--pair A,B | --seed A) [--reading coherency|imag|phase|delay|psi] [--weights coherence|none]
//     Derived at read from the stored bins (nxr::signal::coherency, phaseSlope, phase_slope_index), JSON out. A, B are recording channel indices of the statistic's channels. --pair: one
//     pair (row A, column B); --seed: A against every channel. coherency (default): the coherency per bin (re, im, abs); imag: its imaginary part; phase: arg C_AB per bin (wrapped and
//     unwrapped, rad); delay: the phase slope over f as a delay in ms and its residual; psi: Nolte's phase-slope index. Positive delay and psi: the row channel (A) leads.
//
//
//
//   nxr-neuro sourcepsd compute --dataset D/N --subject S --recording R [--kernel NAME] [--name STEM] [--chunks K|A-B|all | --start SEC --stop SEC]
//                               [--bands name:lo-hi,...] [--normalisation none|relative] [the psd options: --set --segment-seconds|--segment-samples --overlap --window --periodic|--symmetric
//                               --detrend --scaling --no-bad-segments] [--replace]
//     The band power of the recording's SOURCES, as Brainstorm makes it: the Welch PSD of the sources (the kernel applied to the recording), "Group in frequency bands" (the MEAN of the PSD over
//     the bins from lo to hi Hz, inclusive; default delta 2-4, theta 5-7, alpha 8-12, beta 15-29, gamma1 30-59, gamma2 60-90) and, with --normalisation relative, "Spectrum normalization:
//     relative" (each map divided by the source's PSD summed over all bins, 0 to Nyquist). A free kernel's three orientations are summed in power. Exact and cheap: the PSD of the sources is
//     diag(K S K^T) with S the sensors' cross-spectral density of each band, accumulated chunk by chunk (equal to Welch on K x). One Field per band `<R>_<kernel>_psd_<band>` (kind band power,
//     on the kernel's source manifold, folder source), the band a Selection on the Frequency manifold in its producer record.
//
//   nxr-neuro lowpass|highpass|bandpass|bandstop|notch --dataset D/N --subject S --recording R [--name N]
//             lowpass: --fpass HZ · highpass: --fpass HZ · bandpass: --fpass LOW,HIGH · bandstop: --fstop LOW,HIGH · notch: --frequencies F1,F2,... | --line F [--harmonics N]
//             [--bandwidth HZ (notch: the 3-dB bandwidth, 1)] [--order N (bandstop: Butterworth prototype order, 4)] [--ripple R (FIR deviation, 1e-3: 60 dB)] [--transition HZ]
//             [--padding mean|zeros] [--transient-samples K] [--no-remove-mean] [--tolerance T]
//     A temporal filter (MATLAB's names), zero phase and no delay, applied chunk by chunk (nxr/signal/filters.h); the defaults are Brainstorm's filters (a linear-phase Kaiser FIR for the
//     first three, forward-backward IIR for the last two). Writes a NEW recording `<R>_<verb>` on the same channels and time line, stored as recordings are, with the source's channel flags,
//     events, bad segments, projectors and CTF compensation carried over, and the verb, every option and the design in its producer_json.
//     One command per verb (not `filter <kind>`): the verbs are MATLAB's own names and each takes its own edges, as `psd`, `noisecov` and `inverse` take theirs.
//
// JSON lines out, as `nxr-neuro run eigenmodes`; the result line adds "cache": "hit" | "miss".
#include <iostream>
#include <map>
#include <sstream>

#include "cli.h"
#include "nxr/pipelines/face.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/frames.h"

using namespace nxr::io;

namespace {

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::istringstream in(s);
  for (std::string t; std::getline(in, t, sep);) out.push_back(t);
  return out;
}
[[noreturn]] void usage(const std::string& prog, const std::string& m) { throw cli::Usage(prog + ": error: " + m); }


std::vector<double> numbers(const std::string& prog, const std::string& flag, const std::string& v) {
  std::vector<double> out;
  for (const auto& t : split(v, ',')) {
    try { out.push_back(std::stod(t)); } catch (...) { usage(prog, flag + " needs comma-separated numbers"); }
  }
  return out;
}

}  // namespace

// the Welch options of `psd compute` and `sourcepsd compute` (--set, --segment-*, --overlap, --window, --periodic|--symmetric, --detrend, --scaling) into any options struct that has
// option_set, welch, segment_seconds and segment_samples
template <class Options>
void welch_args(const std::string& prog, const cli::Args& a, Options& o) {
  using namespace nxr::signal::spectral;
  const std::string set = a.maybe("--set").value_or("tiles");
  o.option_set = set;
  if (set == "tiles" || set == "brainstorm") o.welch = brainstormPsdOptions(1.0);   // tiles: Brainstorm's taper, overlap, detrend and density on 2^k-sample segments
  else if (set == "scipy") o.welch = scipyWelchOptions(1.0);
  else if (set == "nxr-chain") o.welch = nxrChainWelchOptions(1.0, 1.0);
  else usage(prog, "--set must be tiles, brainstorm, scipy or nxr-chain");
  if (const auto v = a.maybe("--segment-seconds")) o.segment_seconds = std::stod(*v);
  if (const auto v = a.maybe("--segment-samples")) o.segment_samples = std::stoll(*v);
  if (a.maybe("--segment-seconds") && a.maybe("--segment-samples")) usage(prog, "--segment-seconds and --segment-samples are alternatives");   // a segment is a duration (s) or a count of samples, never a level index
  if (const auto v = a.maybe("--overlap")) o.welch.overlap = std::stod(*v);
  if (const auto v = a.maybe("--window")) {
    if (*v == "hann") o.welch.window = Window::Hann;
    else if (*v == "hamming") o.welch.window = Window::Hamming;
    else if (*v == "blackman") o.welch.window = Window::Blackman;
    else if (*v == "rectangular") o.welch.window = Window::Rectangular;
    else usage(prog, "--window must be hann, hamming, blackman or rectangular");
  }
  if (a.flag.count("--periodic") && a.flag.count("--symmetric")) usage(prog, "--periodic and --symmetric are alternatives");
  if (a.flag.count("--periodic")) o.welch.periodic = true;
  if (a.flag.count("--symmetric")) o.welch.periodic = false;
  if (const auto v = a.maybe("--detrend")) {
    if (*v == "none") o.welch.detrend = Detrend::None;
    else if (*v == "constant") o.welch.detrend = Detrend::Constant;
    else if (*v == "linear") o.welch.detrend = Detrend::Linear;
    else usage(prog, "--detrend must be none, constant or linear");
  }
  if (const auto v = a.maybe("--scaling")) {
    if (*v == "density") o.welch.scaling = Scaling::Density;
    else if (*v == "spectrum") o.welch.scaling = Scaling::Spectrum;
    else if (*v == "chain") o.welch.scaling = Scaling::Chain;
    else usage(prog, "--scaling must be density, spectrum or chain");
  }
}

// --bands name:lo-hi,name:lo-hi (Hz) into a list of bands
std::vector<nxr::meeg::Band> bands_arg(const std::string& prog, const std::string& v) {
  std::vector<nxr::meeg::Band> out;
  for (const std::string& item : split(v, ',')) {
    const auto colon = item.find(':'), dash = item.find('-', colon == std::string::npos ? 0 : colon);
    if (colon == std::string::npos || dash == std::string::npos) usage(prog, "--bands is name:lo-hi,name:lo-hi (Hz), e.g. alpha:8-12");
    out.push_back({item.substr(0, colon), std::stod(item.substr(colon + 1, dash - colon - 1)), std::stod(item.substr(dash + 1))});
  }
  return out;
}

// crossspectrum read: the readings of a stored `<stem>_<band>_bins` (see the comment at the top of the file)
int crossspectrum_read_main(const std::vector<std::string>& rest) {   // dispatched by run_legacy (nxr_neuro.cpp): `read crossspectrum`
  const std::string prog = "nxr-neuro crossspectrum read";
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--name", "--pair", "--seed", "--reading", "--weights"}, {}, {"--dataset", "--subject", "--name"});
  nxr::pipelines::face::ReadCrossspectrumRequest rq;
  rq.dataset = a.get("--dataset");
  rq.subject = a.get("--subject");
  rq.name = a.get("--name");
  rq.reading = a.maybe("--reading").value_or("coherency");
  rq.weights = a.maybe("--weights").value_or("coherence");
  rq.templates = cli::values_of(a, "--templates");
  rq.created_by = cli::stamp();
  try {
    // the channels, recording channel indices: --pair A,B or --seed A
    if (const auto v = a.maybe("--pair")) {
      const auto ab = split(*v, ',');
      if (ab.size() != 2) usage(prog, "--pair is A,B");
      rq.pair = std::array<std::int32_t, 2>{static_cast<std::int32_t>(std::stol(ab[0])), static_cast<std::int32_t>(std::stol(ab[1]))};
    }
    if (const auto v = a.maybe("--seed")) rq.seed = static_cast<std::int32_t>(std::stol(*v));
    cli::emit_result(cli::call(prog, [&] { return nxr::pipelines::face::read_crossspectrum(rq, cli::progress()); }).result);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}


