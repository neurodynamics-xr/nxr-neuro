// nxr-neuro noisecov · inverse — MEG/EEG source imaging (nxr_meeg through the pipelines).
//
//   nxr-neuro noisecov compute --dataset D/N --subject S --recording R [--name N] [--chunks K|A-B|all | --start SEC --stop SEC] [--block-seconds SEC]
//                              [--types "MEG MAG,MEG GRAD,EEG"] [--no-remove-mean] [--divisor samples-minus-blocks|samples|samples-minus-one] [--replace]
//     The sample covariance of the recording's good channels over [start, stop), read chunk by chunk; the mean (DC offset) removed per block (the whole window is
//     one block unless --block-seconds). Stored as the Operator "noise covariance" `<R>_noise_cov`.
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
//                         [--set tiles|brainstorm|scipy|nxr-chain] [--segment-seconds SEC | --segment-samples N | --segment-level K] [--overlap F] [--window hann|hamming|blackman|rectangular]
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
//   nxr-neuro opticalflow compute --dataset D/N --subject S --recording R [--kernel NAME] [--package STEM] [--name STEM] [--chunks K|A-B|all | --start S --stop S] --family F --gain G --freq HZ [--centres HZ,..] [--cycles C] [--voices V] [--anchor HZ] [--limits LO-HI] [--half-width OCT] [--gamma G --time-bandwidth B] [--resolution HZ] [--oversample X] [--tolerance T] [--alpha A] [--depth D] [--frame-level L] [--start-s S] [--method pcg|direct|perscale] [--regulariser connection|ambient] [--datafit auto|gauge|triangle] [--input amplitude|signed] [--flow-tolerance T] [--max-iterations N] [--preconditioner ldlt|ic] [--scale-family F] [--scale-voices V] [--scale-filters N] [--replace]
//   nxr-neuro tiles compute --dataset D/N --subject S [--surface NAME] [--depth L] [--replace]
//     The dyadic TILE LADDER of a surface (nxr/ladder.h, the Studio's surfaceLadder): 2^l equal-area tiles per connected piece at level l, paired with scale by Weyl's law, stored as Selections (a `ladder` and one `set` per level:
//     the tile code h 2^l + t of every vertex), their areas, centres and adjacency in the Selection's atlas node. `opticalflow compute` and `quadrature compute` take --tiles to write their maps' tile summaries per level
//     (--tile-levels 1,2,3 | --tile-max-level N, --no-levi-civita): scalars area-weighted; vectors in the canonical frame components and in the Levi-Civita (transported) form, labelled; never raw 3-D sums.
//   nxr-neuro frames compute --dataset D/N --subject S [--surface NAME] [--template SUBJECT [--template-surface NAME]] [--convention scs|ras] [--side auto|left|right] [--min-projection P] [--max-condition C] [--replace]
//     The cortical frames of a surface (nxr/frames.h): the CANONICAL frames (north / west; read when stored as `<surface>_canonical_frames`, else computed from the registered sphere `<surface>_sphere`)
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
//     Derived at read from the stored bins (nxr::meeg::coherency, phaseSlope, phaseSlopeIndex), JSON out. A, B are recording channel indices of the statistic's channels. --pair: one
//     pair (row A, column B); --seed: A against every channel. coherency (default): the coherency per bin (re, im, abs); imag: its imaginary part; phase: arg C_AB per bin (wrapped and
//     unwrapped, rad); delay: the phase slope over f as a delay in ms and its residual; psi: Nolte's phase-slope index. Positive delay and psi: the row channel (A) leads.
//
//   nxr-neuro cyclepattern compute --dataset D/N --subject S --cross-spectrum NAME [--kernel NAME] [--package STEM] [--spatial-band modes|weyl:J|weyl:all] [--patterns Q] [--timing-of stream,potential,normal]
//                         [--level K | --window-seconds SEC (default 4, snapped to the nearest level)] [--name STEM] [--replace]
//     The windows read are the stored tiles rolled up to that level (a window kept when all its tiles are stored).
//     THE CONSISTENT CYCLE PATTERNS (spec docs/specs/2026-11-13-cycle-pattern.md, stage 2) from a cross-spectrum series (crossspectrum compute --estimator analytic): a free kernel
//     read into the package's full forms, per spatial band the q leading eigenvectors of B C B^T over the span (and per window, for their consistency); stored per band and pattern
//     on the glued surface as complex S (stream function), A (potential), rho (normal current), and its coefficients on the eigen axis. Default --spatial-band weyl:all, --patterns 3.
//
//   nxr-neuro phasegradient compute --dataset D/N --subject S --cross-spectrum NAME [--kernel NAME] [--signal ambient|normal] [--name STEM] [--replace]
//                                   [--spatial-band modes|weyl:J|weyl:all [--package STEM] [--field gradient] [--edge-lag frame|frame-normal|frame-tangent|axial|ambient] [--fit inner|mesh|modes] [--weight power|coherence] [--defects keep|exclude] [--fold-edges keep|axial|all] [--phase-of current|stream]]
//     --spatial-band: the kernel read in the surface's Hodge–Dirac modes, weighted by a band of them (modes: every mode; weyl:J Weyl's Meyer octave J on μ²;
//     weyl:all each member) and synthesised on the glued surface; with --signal ambient. Adds the phase field's reading in the modes: the timing map, the
//     stream function, divergence, curl, the inconsistency, and the mode coefficients.
//     Whether the band's activity moves across the cortex, which way and how fast: nxr's phase reader (nxr.detect.core_phase, grad phi = Im(conj z grad z) / |z|^2) in its
//     spectral form, from a STORED band cross-spectrum (`crossspectrum compute`) through a free kernel and the mesh's P1 gradient, power-weighted. Stored on the measurement pyramid's leaves
//     (`--depth D`: the tile ladder's level, default the finest every piece reaches; needs `frames compute`): a byte-less Field `<stem>_<band>` and, per leaf of `<surface> tiles L<D>`, the SUMS
//     phase_power, phase_gradient_power, phase_current_1, phase_current_2 (the phase current in the canonical frames), phase_current_magnitude; the phase gradient, its speed, the travelling
//     index and the phase coherence are derived at read (`measure query`: phase_gradient_1, ..., travelling_index). The per-vertex Fields of earlier runs are not written any more (and not touched).
//
//   nxr-neuro quadrature compute --dataset D/N --subject S --recording R [--kernel NAME] [--bands alpha:8-12,...] [--pairs K]   (see nxr/pipelines/meeg.h)
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
//     A temporal filter (MATLAB's names), zero phase and no delay, applied chunk by chunk (nxr/filter.h); the defaults are Brainstorm's filters (a linear-phase Kaiser FIR for the
//     first three, forward-backward IIR for the last two). Writes a NEW recording `<R>_<verb>` on the same channels and time line, stored as recordings are, with the source's channel flags,
//     events, bad segments, projectors and CTF compensation carried over, and the verb, every option and the design in its producer_json.
//     One command per verb (not `filter <kind>`): the verbs are MATLAB's own names and each takes its own edges, as `psd`, `noisecov` and `inverse` take theirs.
//
// JSON lines out, as `nxr-neuro eigen solve`; the result line adds "cache": "hit" | "miss".
#include <iostream>
#include <map>
#include <sstream>

#include "cli.h"
#include "nxr/meeg/crossspectrum.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/tiles.h"

using namespace nxr::io;

namespace {

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::istringstream in(s);
  for (std::string t; std::getline(in, t, sep);) out.push_back(t);
  return out;
}
[[noreturn]] void usage(const std::string& prog, const std::string& m) { throw cli::Usage(prog + ": error: " + m); }

const std::set<std::string> kNoiseValued = {"--recording", "--name", "--chunks", "--start", "--stop", "--block-seconds", "--types", "--divisor"};
const std::set<std::string> kNoiseFlags = {"--no-remove-mean"};

nxr::pipelines::NoiseCovOptions noise_options(const std::string& prog, const cli::Args& a, bool with_name) {
  nxr::pipelines::NoiseCovOptions o;
  o.recording = a.get("--recording");
  if (with_name) o.name = a.maybe("--name");
  if (const auto v = a.maybe("--start")) o.start = std::stod(*v);
  if (const auto v = a.maybe("--stop")) o.stop = std::stod(*v);
  o.chunks = a.maybe("--chunks");
  if (const auto v = a.maybe("--block-seconds")) o.block_seconds = std::stod(*v);
  if (const auto v = a.maybe("--types")) o.channel_types = split(*v, ',');
  o.covariance.removeMean = a.flag.count("--no-remove-mean") == 0;
  if (const auto v = a.maybe("--divisor")) {
    if (*v == "samples-minus-blocks") o.covariance.divisor = nxr::meeg::CovarianceDivisor::SamplesMinusBlocks;
    else if (*v == "samples") o.covariance.divisor = nxr::meeg::CovarianceDivisor::Samples;
    else if (*v == "samples-minus-one") o.covariance.divisor = nxr::meeg::CovarianceDivisor::SamplesMinusOne;
    else usage(prog, "--divisor must be samples-minus-blocks, samples or samples-minus-one");
  }
  return o;
}

std::vector<double> numbers(const std::string& prog, const std::string& flag, const std::string& v) {
  std::vector<double> out;
  for (const auto& t : split(v, ',')) {
    try { out.push_back(std::stod(t)); } catch (...) { usage(prog, flag + " needs comma-separated numbers"); }
  }
  return out;
}

}  // namespace

int noisecov_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro noisecov: error: the following arguments are required: action (compute)");
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  std::set<std::string> valued = kNoiseValued;
  valued.insert({"--dataset", "--subject"});
  std::set<std::string> flags = kNoiseFlags;
  flags.insert("--replace");
  const cli::Args a = cli::parse("nxr-neuro noisecov compute", rest, valued, flags, {"--dataset", "--subject", "--recording"});
  nxr::pipelines::NoiseCovOptions o = noise_options("nxr-neuro noisecov compute", a, true);
  o.replace = a.flag.count("--replace") > 0;
  DatasetWriter ds = open_writer(a.get("--dataset"));
  OpenSubjectOptions oo;
  oo.created_by = cli::stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  const auto r = nxr::pipelines::noisecov_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
  Json out = Json{{"stage", "result"}};
  for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
  out["seconds"] = std::round(r.seconds * 1000) / 1000;
  cli::emit(out);
  return 0;
}

// the Welch options of `psd compute` and `sourcepsd compute` (--set, --segment-*, --overlap, --window, --periodic|--symmetric, --detrend, --scaling) into any options struct that has
// option_set, welch, segment_seconds and segment_samples
template <class Options>
void welch_args(const std::string& prog, const cli::Args& a, Options& o) {
  using namespace nxr::welch;
  const std::string set = a.maybe("--set").value_or("tiles");
  o.option_set = set;
  if (set == "tiles" || set == "brainstorm") o.welch = brainstormPsdOptions(1.0);   // tiles: Brainstorm's taper, overlap, detrend and density on 2^k-sample segments
  else if (set == "scipy") o.welch = scipyWelchOptions(1.0);
  else if (set == "nxr-chain") o.welch = nxrChainWelchOptions(1.0, 1.0);
  else usage(prog, "--set must be tiles, brainstorm, scipy or nxr-chain");
  if (const auto v = a.maybe("--segment-seconds")) o.segment_seconds = std::stod(*v);
  if (const auto v = a.maybe("--segment-samples")) o.segment_samples = std::stoll(*v);
  if (const auto v = a.maybe("--segment-level")) {
    try { o.segment_level = std::stoi(*v); } catch (...) { usage(prog, "--segment-level is a tile level, a whole number (the segment is 2^level samples)"); }
  }
  if ((a.maybe("--segment-seconds") ? 1 : 0) + (a.maybe("--segment-samples") ? 1 : 0) + (a.maybe("--segment-level") ? 1 : 0) > 1)
    usage(prog, "--segment-seconds, --segment-samples and --segment-level are alternatives");
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

int psd_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro psd: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro psd compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::set<std::string> valued = {"--dataset", "--subject", "--recording", "--name", "--chunks", "--start", "--stop", "--set", "--segment-seconds", "--segment-samples", "--segment-level",
                                        "--overlap", "--window", "--detrend", "--scaling", "--through"};
  const std::set<std::string> flags = {"--periodic", "--symmetric", "--no-bad-segments", "--replace"};
  const cli::Args a = cli::parse(prog, rest, valued, flags, {"--dataset", "--subject", "--recording"});
  using namespace nxr::welch;
  nxr::pipelines::PsdOptions o;
  o.through = a.maybe("--through");
  o.recording = a.get("--recording");
  o.name = a.maybe("--name");
  if (const auto v = a.maybe("--start")) o.start = std::stod(*v);
  if (const auto v = a.maybe("--stop")) o.stop = std::stod(*v);
  o.chunks = a.maybe("--chunks");
  welch_args(prog, a, o);
  o.exclude_bad_segments = a.flag.count("--no-bad-segments") == 0;
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_writer(a.get("--dataset"));
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::psd_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    out["seconds"] = std::round(r.seconds * 1000) / 1000;
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
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
static int crossspectrum_read_main(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro crossspectrum read";
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--name", "--pair", "--seed", "--reading", "--weights"}, {}, {"--dataset", "--subject", "--name"});
  const std::string reading = a.maybe("--reading").value_or("coherency");
  if (reading != "coherency" && reading != "imag" && reading != "phase" && reading != "delay" && reading != "psi") usage(prog, "--reading is coherency, imag, phase, delay or psi");
  const std::string weights = a.maybe("--weights").value_or("coherence");
  if (weights != "coherence" && weights != "none") usage(prog, "--weights is coherence or none");
  if (a.maybe("--pair") && a.maybe("--seed")) usage(prog, "--pair and --seed are alternatives");
  if (!a.maybe("--pair") && !a.maybe("--seed")) usage(prog, "name the channels to read: --pair A,B (row A, column B; recording channel indices) or --seed A (A against every channel)");
  try {
    DatasetWriter ds = open_writer(a.get("--dataset"));
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto S = nxr::pipelines::crossspectrum_bins_read(sub, a.get("--name"));
    auto position = [&](const std::string& t) {
      const std::int32_t ch = static_cast<std::int32_t>(std::stol(t));
      for (std::size_t i = 0; i < S.channels.size(); ++i) if (S.channels[i] == ch) return static_cast<Eigen::Index>(i);
      std::string have;
      for (std::size_t i = 0; i < S.channels.size() && i < 8; ++i) have += (i ? "," : "") + std::to_string(S.channels[i]);
      usage(prog, "channel " + t + " is not one of the statistic's " + std::to_string(S.channels.size()) + " channels (recording channel indices, from " + have + (S.channels.size() > 8 ? ", ..." : "") + ")");
      return Eigen::Index(0);
    };
    std::vector<Eigen::Index> rows, cols;                         // the pairs (rows[i], cols[i]) wanted
    if (const auto v = a.maybe("--pair")) {
      const auto ab = split(*v, ',');
      if (ab.size() != 2) usage(prog, "--pair is A,B");
      rows.push_back(position(ab[0])); cols.push_back(position(ab[1]));
    } else {
      const Eigen::Index r = position(*a.maybe("--seed"));
      for (std::size_t j = 0; j < S.channels.size(); ++j) { rows.push_back(r); cols.push_back(static_cast<Eigen::Index>(j)); }
    }
    const Eigen::VectorXd f = Eigen::Map<const Eigen::VectorXd>(S.frequencies.data(), static_cast<Eigen::Index>(S.frequencies.size()));
    Json out = Json{{"stage", "result"}, {"name", a.get("--name")}, {"reading", reading}, {"frequencies_hz", S.frequencies}, {"sign", "positive delay and psi: the row channel leads the column channel"}};
    Json chans = Json::array(), vals = Json::array();
    for (std::size_t i = 0; i < rows.size(); ++i) chans.push_back(Json::array({S.channels[static_cast<std::size_t>(rows[i])], S.channels[static_cast<std::size_t>(cols[i])]}));
    out["pairs"] = chans;
    if (reading == "delay") {
      nxr::meeg::PhaseSlopeOptions po;
      po.weights = weights == "none" ? nxr::meeg::PhaseSlopeOptions::Weights::None : nxr::meeg::PhaseSlopeOptions::Weights::Coherence;
      const nxr::meeg::PhaseSlope ps = nxr::meeg::phaseSlope(S.C, f, po);
      Json res = Json::array();
      for (std::size_t i = 0; i < rows.size(); ++i) { vals.push_back(ps.delayMs(rows[i], cols[i])); res.push_back(ps.residual(rows[i], cols[i])); }
      out["delay_ms"] = vals; out["residual_rad"] = res; out["weights"] = weights;
    } else if (reading == "psi") {
      const Eigen::MatrixXd psi = nxr::meeg::phaseSlopeIndex(S.C);
      for (std::size_t i = 0; i < rows.size(); ++i) vals.push_back(psi(rows[i], cols[i]));
      out["psi"] = vals;
    } else {
      const auto H = nxr::meeg::coherency(S.C);
      for (std::size_t i = 0; i < rows.size(); ++i) {
        Json re = Json::array(), im = Json::array(), ab = Json::array(), ph = Json::array(), un = Json::array();
        double last = 0;
        for (std::size_t k = 0; k < H.size(); ++k) {
          const std::complex<double> h = H[k](rows[i], cols[i]);
          re.push_back(h.real()); im.push_back(h.imag()); ab.push_back(std::abs(h));
          const double p = std::arg(h);
          ph.push_back(p);
          last = k == 0 ? p : last + std::remainder(p - last, 2 * M_PI);
          un.push_back(last);
        }
        if (reading == "imag") vals.push_back(im);
        else if (reading == "phase") vals.push_back(Json{{"phase_rad", ph}, {"unwrapped_rad", un}});
        else vals.push_back(Json{{"re", re}, {"im", im}, {"abs", ab}});
      }
      out["values"] = vals;
    }
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int crossspectrum_main(const std::vector<std::string>& argv) {
  if (argv.empty() || (argv[0] != "compute" && argv[0] != "read")) throw cli::Usage("nxr-neuro crossspectrum: error: the following arguments are required: action (compute | read)");
  if (argv[0] == "read") return crossspectrum_read_main(std::vector<std::string>(argv.begin() + 1, argv.end()));
  const std::string prog = "nxr-neuro crossspectrum compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::set<std::string> valued = {"--resolve", "--dataset", "--subject", "--recording", "--channels-of", "--types", "--name", "--chunks", "--start", "--stop", "--set", "--segment-seconds",
                                        "--segment-samples", "--segment-level", "--overlap", "--window", "--detrend", "--scaling", "--bands", "--estimator", "--level", "--window-seconds"};
  const std::set<std::string> flags = {"--periodic", "--symmetric", "--no-bad-segments", "--replace"};
  const cli::Args a = cli::parse(prog, rest, valued, flags, {"--dataset", "--subject", "--recording"});
  nxr::pipelines::CrossSpectrumOptions o;
  o.recording = a.get("--recording");
  o.channels_of = a.maybe("--channels-of");
  if (const auto v = a.maybe("--types")) o.channel_types = split(*v, ',');
  if (o.channels_of && a.maybe("--types")) usage(prog, "--channels-of and --types are alternatives");
  o.name = a.maybe("--name");
  if (const auto v = a.maybe("--start")) o.start = std::stod(*v);
  if (const auto v = a.maybe("--stop")) o.stop = std::stod(*v);
  o.chunks = a.maybe("--chunks");
  welch_args(prog, a, o);
  if (const auto v = a.maybe("--bands")) o.bands = bands_arg(prog, *v);
  o.exclude_bad_segments = a.flag.count("--no-bad-segments") == 0;
  o.replace = a.flag.count("--replace") > 0;
  if (const auto v = a.maybe("--estimator")) {
    if (*v != "welch" && *v != "analytic") usage(prog, "--estimator is welch or analytic");
    o.estimator = *v;
  }
  if (a.maybe("--level") && a.maybe("--window-seconds")) usage(prog, "--level and --window-seconds are alternatives");
  if ((a.maybe("--level") || a.maybe("--window-seconds")) && o.estimator != "analytic") usage(prog, "--level and --window-seconds apply to --estimator analytic only");
  if (const auto v = a.maybe("--resolve")) {
    if (*v != "band" && *v != "bins") usage(prog, "--resolve is band or bins");
    if (*v == "bins" && o.estimator != "welch") usage(prog, "--resolve bins is the Welch estimator's: the analytic estimator has no frequency bins");
    o.resolve = *v;
  }
  if (const auto v = a.maybe("--level")) {
    try { o.level = std::stoi(*v); } catch (...) { usage(prog, "--level is a tile level, a whole number 0 to 15"); }
  }
  if (const auto v = a.maybe("--window-seconds")) o.window_seconds = std::stod(*v);
  if (o.estimator == "analytic")
    for (const char* w : {"--set", "--overlap", "--window", "--detrend", "--scaling", "--segment-seconds", "--segment-samples", "--segment-level"})
      if (a.maybe(w)) usage(prog, std::string(w) + " is a Welch option: it does not apply to --estimator analytic (its windows are tiles: --level or --window-seconds)");
  try {
    DatasetWriter ds = open_writer(a.get("--dataset"));
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::crossspectrum_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    out["seconds"] = std::round(r.seconds * 1000) / 1000;
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int cyclepattern_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro cyclepattern: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro cyclepattern compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--cross-spectrum", "--kernel", "--package", "--spatial-band", "--patterns", "--name", "--timing-of", "--level", "--window-seconds"},
                                 {"--replace"}, {"--dataset", "--subject", "--cross-spectrum"});
  nxr::pipelines::CyclePatternOptions o;
  o.cross_spectrum = a.get("--cross-spectrum");
  o.kernel = a.maybe("--kernel");
  o.package = a.maybe("--package");
  if (const auto v = a.maybe("--spatial-band")) o.spatial_band = *v;
  if (const auto v = a.maybe("--patterns")) {
    try { o.patterns = std::stoi(*v); } catch (...) { usage(prog, "--patterns is a whole number >= 1"); }
    if (o.patterns < 1) usage(prog, "--patterns is a whole number >= 1");
  }
  o.name = a.maybe("--name");
  if (a.maybe("--level") && a.maybe("--window-seconds")) usage(prog, "--level and --window-seconds are alternatives");
  if (const auto v = a.maybe("--level")) {
    try { o.level = std::stoi(*v); } catch (...) { usage(prog, "--level is a tile level, a whole number"); }
  }
  if (const auto v = a.maybe("--window-seconds")) o.window_seconds = std::stod(*v);
  if (const auto v = a.maybe("--timing-of")) o.timing_of = split(*v, ',');
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_writer(a.get("--dataset"));
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::cyclepattern_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    out["seconds"] = std::round(r.seconds * 1000) / 1000;
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int phasegradient_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro phasegradient: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro phasegradient compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--cross-spectrum", "--kernel", "--signal", "--name", "--spatial-band", "--package", "--field", "--edge-lag", "--fit", "--weight", "--defects", "--fold-edges", "--phase-of", "--depth"}, {"--replace"}, {"--dataset", "--subject", "--cross-spectrum"});
  nxr::pipelines::PhaseGradientOptions o;
  o.cross_spectrum = a.get("--cross-spectrum");
  o.kernel = a.maybe("--kernel");
  o.signal = a.maybe("--signal").value_or("ambient");
  if (o.signal != "normal" && o.signal != "ambient") usage(prog, "--signal must be normal or ambient");
  o.spatial_band = a.maybe("--spatial-band");
  o.package = a.maybe("--package");
  o.field = a.maybe("--field").value_or("gradient");
  o.edge_lag = a.maybe("--edge-lag").value_or("frame");
  o.fit = a.maybe("--fit").value_or("inner");
  o.weight = a.maybe("--weight").value_or("power");
  o.defects = a.maybe("--defects").value_or("keep");
  o.fold_edges = a.maybe("--fold-edges").value_or("keep");
  o.phase_of = a.maybe("--phase-of").value_or("current");
  if (o.phase_of != "current" && o.phase_of != "stream") usage(prog, "--phase-of must be current or stream");
  if (o.fold_edges != "keep" && o.fold_edges != "axial" && o.fold_edges != "all") usage(prog, "--fold-edges must be keep, axial or all");
  if (o.defects != "keep" && o.defects != "exclude") usage(prog, "--defects must be keep or exclude");
  if (o.fit != "inner" && o.fit != "mesh" && o.fit != "modes") usage(prog, "--fit must be inner, mesh or modes");
  if (o.weight != "power" && o.weight != "coherence") usage(prog, "--weight must be power or coherence");
  if (o.edge_lag != "frame" && o.edge_lag != "frame-normal" && o.edge_lag != "frame-tangent" && o.edge_lag != "axial" && o.edge_lag != "ambient")
    usage(prog, "--edge-lag must be frame, frame-normal, frame-tangent, axial or ambient");
  if (o.field != "gradient") usage(prog, "--field must be gradient");
  if (!o.spatial_band && (o.package || a.maybe("--field") || a.maybe("--edge-lag") || a.maybe("--fit") || a.maybe("--weight") || a.maybe("--defects") || a.maybe("--fold-edges") || a.maybe("--phase-of"))) usage(prog, "--package, --field, --edge-lag, --fit and --weight go with --spatial-band");
  o.name = a.maybe("--name");
  if (const auto v = a.maybe("--depth")) {
    try { o.depth = std::stoi(*v); } catch (...) { usage(prog, "--depth is a whole number (the tile ladder's level)"); }
    if (o.spatial_band) usage(prog, "--depth goes with the plain route (the leaves of the pyramid); --spatial-band stores per-vertex Fields on the glued surface");
  }
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_writer(a.get("--dataset"));
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::phasegradient_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    out["seconds"] = std::round(r.seconds * 1000) / 1000;
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

static int source_spectral_main(const std::vector<std::string>& argv, bool tensor) {   // sourcepsd · sourcetensor: the same options
  const std::string verb = tensor ? "sourcetensor" : "sourcepsd";
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro " + verb + ": error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro " + verb + " compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::set<std::string> valued = {"--dataset", "--subject", "--recording", "--kernel", "--name", "--chunks", "--start", "--stop", "--set", "--segment-seconds", "--segment-samples", "--segment-level",
                                        "--overlap", "--window", "--detrend", "--scaling", "--bands", "--normalisation"};
  const std::set<std::string> flags = {"--periodic", "--symmetric", "--no-bad-segments", "--replace"};
  const cli::Args a = cli::parse(prog, rest, valued, flags, {"--dataset", "--subject", "--recording"});
  nxr::pipelines::SourcePsdOptions o;
  o.recording = a.get("--recording");
  o.kernel = a.maybe("--kernel");
  o.name = a.maybe("--name");
  if (const auto v = a.maybe("--start")) o.start = std::stod(*v);
  if (const auto v = a.maybe("--stop")) o.stop = std::stod(*v);
  o.chunks = a.maybe("--chunks");
  welch_args(prog, a, o);
  if (const auto v = a.maybe("--bands")) o.bands = bands_arg(prog, *v);
  o.normalisation = a.maybe("--normalisation").value_or("none");
  if (o.normalisation != "none" && o.normalisation != "relative") usage(prog, "--normalisation must be none or relative");
  o.exclude_bad_segments = a.flag.count("--no-bad-segments") == 0;
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_writer(a.get("--dataset"));
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = tensor ? nxr::pipelines::sourcetensor_compute(ds, sub, o, [](const Json& j) { cli::emit(j); })
                          : nxr::pipelines::sourcepsd_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    out["seconds"] = std::round(r.seconds * 1000) / 1000;
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int sourcepsd_main(const std::vector<std::string>& argv) { return source_spectral_main(argv, false); }
int sourcetensor_main(const std::vector<std::string>& argv) { return source_spectral_main(argv, true); }

// --tiles [--tile-levels 1,2,3] [--tile-max-level N] [--no-levi-civita]: the maps' tile summaries per level
template <class O> void tile_args(const std::string& prog, const cli::Args& a, O& o) {
  o.tiles = a.flag.count("--tiles") > 0;
  if (!o.tiles && (a.maybe("--tile-levels") || a.maybe("--tile-max-level") || a.flag.count("--no-levi-civita"))) usage(prog, "--tile-levels, --tile-max-level and --no-levi-civita need --tiles");
  if (const auto v = a.maybe("--tile-levels")) { std::stringstream ss(*v); for (std::string t; std::getline(ss, t, ',');) o.tile_levels.push_back(std::stoi(t)); }
  if (const auto v = a.maybe("--tile-max-level")) o.tile_max_level = std::stoi(*v);
  o.tile_levi_civita = a.flag.count("--no-levi-civita") == 0;
}

int quadrature_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro quadrature: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro quadrature compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::set<std::string> valued = {"--dataset", "--subject", "--recording", "--kernel", "--name", "--chunks", "--start", "--stop", "--set", "--segment-seconds", "--segment-samples", "--segment-level",
                                        "--overlap", "--window", "--detrend", "--scaling", "--bands", "--pairs", "--tile-levels", "--tile-max-level"};
  const std::set<std::string> flags = {"--periodic", "--symmetric", "--no-bad-segments", "--replace", "--tiles", "--no-levi-civita"};
  const cli::Args a = cli::parse(prog, rest, valued, flags, {"--dataset", "--subject", "--recording"});
  nxr::pipelines::QuadratureOptions o;
  o.recording = a.get("--recording");
  o.kernel = a.maybe("--kernel");
  o.name = a.maybe("--name");
  if (const auto v = a.maybe("--start")) o.start = std::stod(*v);
  if (const auto v = a.maybe("--stop")) o.stop = std::stod(*v);
  o.chunks = a.maybe("--chunks");
  if (const auto v = a.maybe("--pairs")) o.pairs = std::stoi(*v);
  tile_args(prog, a, o);
  welch_args(prog, a, o);
  if (const auto v = a.maybe("--bands")) {       // name:lo-hi,name:lo-hi
    o.bands.clear();
    for (const std::string& item : split(*v, ',')) {
      const auto colon = item.find(':'), dash = item.find('-', colon == std::string::npos ? 0 : colon);
      if (colon == std::string::npos || dash == std::string::npos) usage(prog, "--bands is name:lo-hi,name:lo-hi (Hz), e.g. alpha:8-12");
      o.bands.push_back({item.substr(0, colon), std::stod(item.substr(colon + 1, dash - colon - 1)), std::stod(item.substr(dash + 1))});
    }
  }
  o.exclude_bad_segments = a.flag.count("--no-bad-segments") == 0;
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_writer(a.get("--dataset"));
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::quadrature_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    out["seconds"] = std::round(r.seconds * 1000) / 1000;
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int hodgepower_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro hodgepower: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro hodgepower compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::set<std::string> valued = {"--dataset", "--subject", "--recording", "--kernel", "--package", "--modes", "--name", "--chunks", "--start", "--stop", "--set", "--segment-seconds",
                                        "--segment-samples", "--segment-level", "--overlap", "--window", "--detrend", "--scaling", "--bands"};
  const std::set<std::string> flags = {"--periodic", "--symmetric", "--no-bad-segments", "--replace"};
  const cli::Args a = cli::parse(prog, rest, valued, flags, {"--dataset", "--subject", "--recording"});
  nxr::pipelines::HodgePowerOptions o;
  o.recording = a.get("--recording");
  o.kernel = a.maybe("--kernel");
  o.package = a.maybe("--package");
  if (const auto v = a.maybe("--modes")) o.modes = std::stoll(*v);
  o.name = a.maybe("--name");
  if (const auto v = a.maybe("--start")) o.start = std::stod(*v);
  if (const auto v = a.maybe("--stop")) o.stop = std::stod(*v);
  o.chunks = a.maybe("--chunks");
  welch_args(prog, a, o);
  if (const auto v = a.maybe("--bands")) {       // name:lo-hi,name:lo-hi
    o.bands.clear();
    for (const std::string& item : split(*v, ',')) {
      const auto colon = item.find(':'), dash = item.find('-', colon == std::string::npos ? 0 : colon);
      if (colon == std::string::npos || dash == std::string::npos) usage(prog, "--bands is name:lo-hi,name:lo-hi (Hz), e.g. alpha:8-12");
      o.bands.push_back({item.substr(0, colon), std::stod(item.substr(colon + 1, dash - colon - 1)), std::stod(item.substr(dash + 1))});
    }
  }
  o.exclude_bad_segments = a.flag.count("--no-bad-segments") == 0;
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_writer(a.get("--dataset"));
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::hodgepower_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    out["seconds"] = std::round(r.seconds * 1000) / 1000;
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int opticalflow_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro opticalflow: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro opticalflow compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::set<std::string> valued = {"--dataset", "--subject", "--recording", "--kernel", "--package", "--name", "--depth", "--frame-level", "--start-s", "--chunks", "--start", "--stop", "--family", "--gain", "--freq", "--centres", "--cycles", "--voices", "--anchor", "--limits", "--half-width", "--gamma", "--time-bandwidth", "--resolution", "--oversample", "--tolerance", "--alpha", "--regulariser", "--datafit", "--input", "--method", "--flow-tolerance", "--max-iterations", "--preconditioner", "--scale-family", "--scale-voices", "--scale-filters", "--depth", "--frame-level", "--start-s", "--tile-levels", "--tile-max-level"};
  const std::set<std::string> flags = {"--replace", "--tiles", "--no-levi-civita"};
  const cli::Args a = cli::parse(prog, rest, valued, flags, {"--dataset", "--subject", "--recording"});
  nxr::pipelines::OpticalflowComputeOptions o;
  o.recording = a.get("--recording");
  o.kernel = a.maybe("--kernel");
  o.package = a.maybe("--package");
  o.name = a.maybe("--name");
  if (const auto v = a.maybe("--start")) o.start = std::stod(*v);
  if (const auto v = a.maybe("--stop")) o.stop = std::stod(*v);
  o.chunks = a.maybe("--chunks");
  if (const auto v = a.maybe("--alpha")) o.alpha = std::stod(*v);
  if (const auto v = a.maybe("--method")) o.method = *v;
  if (const auto v = a.maybe("--regulariser")) { if (*v != "connection" && *v != "ambient") usage(prog, "--regulariser is connection (default) or ambient"); o.regulariser = *v; }
  if (const auto v = a.maybe("--datafit")) { if (*v != "auto" && *v != "gauge" && *v != "triangle") usage(prog, "--datafit is auto (default), gauge or triangle"); o.datafit = *v; }
  if (const auto v = a.maybe("--input")) { if (*v != "amplitude" && *v != "signed") usage(prog, "--input is amplitude or signed"); o.input = *v; }
  if (const auto v = a.maybe("--flow-tolerance")) o.flow_tolerance = std::stod(*v);
  if (const auto v = a.maybe("--max-iterations")) o.max_iterations = std::stoi(*v);
  if (const auto v = a.maybe("--preconditioner")) o.preconditioner = *v;
  if (const auto v = a.maybe("--scale-family")) o.scale_family = *v;
  if (const auto v = a.maybe("--scale-voices")) o.scale_voices = std::stoi(*v);
  if (const auto v = a.maybe("--scale-filters")) o.scale_filters = std::stoi(*v);
  tile_args(prog, a, o);
  using F = nxr::timebank::Family;
  if (const auto v = a.maybe("--family")) {
    const std::map<std::string, F> fam = {{"itersine", F::Itersine}, {"meyer", F::Meyer}, {"morlet", F::Morlet}, {"morse", F::Morse}, {"amor", F::Amor}, {"bump", F::Bump}};
    const auto it = fam.find(*v);
    if (it == fam.end()) usage(prog, "--family is itersine, meyer, morlet, morse, amor or bump");
    o.family = it->second;
  } else usage(prog, "--family is required (no default)");
  if (const auto v = a.maybe("--gain")) o.gain = std::stod(*v); else usage(prog, "--gain is required (2 makes |c| the physical amplitude)");
  if (const auto v = a.maybe("--freq")) o.freq = std::stod(*v); else usage(prog, "--freq is required (the member nearest to it is used)");
  if (const auto v = a.maybe("--centres")) {
    std::stringstream ss(*v);
    for (std::string tok; std::getline(ss, tok, ',');) o.centres.push_back(std::stod(tok));
  }
  if (const auto v = a.maybe("--cycles")) o.cycles = std::stod(*v);
  if (const auto v = a.maybe("--voices")) o.voices = std::stoi(*v);
  if (const auto v = a.maybe("--anchor")) o.anchor = std::stod(*v);
  if (const auto v = a.maybe("--half-width")) o.half_width = std::stod(*v);
  if (const auto v = a.maybe("--gamma")) o.gamma = std::stod(*v);
  if (const auto v = a.maybe("--time-bandwidth")) o.time_bandwidth = std::stod(*v);
  if (const auto v = a.maybe("--resolution")) o.resolution = std::stod(*v);
  if (const auto v = a.maybe("--oversample")) o.oversample = std::stod(*v);
  if (const auto v = a.maybe("--tolerance")) o.tolerance = std::stod(*v);
  if (const auto v = a.maybe("--limits")) {       // lo-hi
    const auto dash = v->find('-');
    if (dash == std::string::npos) usage(prog, "--limits is lo-hi (Hz), e.g. 2-40");
    o.limits = std::make_pair(std::stod(v->substr(0, dash)), std::stod(v->substr(dash + 1)));
  }
  try {
    if (const auto v = a.maybe("--depth")) o.depth = std::stoi(*v);
    if (const auto v = a.maybe("--frame-level")) o.frame_level = std::stoi(*v);
    if (const auto v = a.maybe("--start-s")) o.day_start_s = std::stod(*v);
  } catch (...) { usage(prog, "--depth and --frame-level are whole numbers (the tile ladder's level, the tower level), --start-s a number (s)"); }
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_writer(a.get("--dataset"));
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::opticalflow_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    out["seconds"] = std::round(r.seconds * 1000) / 1000;
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int frames_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro frames: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro frames compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--surface", "--template", "--template-surface", "--convention", "--side", "--min-projection", "--max-condition"}, {"--replace"}, {"--dataset", "--subject"});
  nxr::pipelines::FramesComputeOptions o;
  o.surface = a.maybe("--surface");
  o.templ = a.maybe("--template");
  o.template_surface = a.maybe("--template-surface");
  if (const auto v = a.maybe("--convention")) { if (*v != "scs" && *v != "ras") usage(prog, "--convention is scs or ras"); o.convention = *v; }
  if (const auto v = a.maybe("--side")) { if (*v != "auto" && *v != "left" && *v != "right") usage(prog, "--side is auto, left or right"); o.side = *v; }
  if (const auto v = a.maybe("--min-projection")) o.min_projection = std::stod(*v);
  if (const auto v = a.maybe("--max-condition")) o.max_condition = std::stod(*v);
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_writer(a.get("--dataset"));
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::frames_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int tiles_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro tiles: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro tiles compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--surface", "--depth"}, {"--replace"}, {"--dataset", "--subject"});
  nxr::pipelines::TilesComputeOptions o;
  o.surface = a.maybe("--surface");
  if (const auto v = a.maybe("--depth")) o.depth = std::stoi(*v);
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_writer(a.get("--dataset"));
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::tiles_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int inverse_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro inverse: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro inverse compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  std::set<std::string> valued = kNoiseValued;
  valued.insert({"--dataset", "--subject", "--noise-cov", "--forward", "--depth-exponent", "--depth-limit", "--depth-norm", "--depth-limit-on", "--depth-channels", "--snr",
                 "--snr-normalisation", "--regularisation", "--ridge-fraction", "--whitener-form", "--rank-tolerance"});
  std::set<std::string> flags = kNoiseFlags;
  flags.insert({"--replace", "--no-depth", "--joint-types", "--ridge-over-rank", "--no-detect-diagonal", "--no-average-reference", "--mne"});
  const cli::Args a = cli::parse(prog, rest, valued, flags, {"--dataset", "--subject"});
  using namespace nxr::meeg;
  nxr::pipelines::InverseComputeOptions o;
  o.forward = a.maybe("--forward");
  o.noise_cov = a.maybe("--noise-cov");
  if (a.maybe("--recording")) o.from_recording = noise_options(prog, a, false);
  if (!o.noise_cov && !o.from_recording) usage(prog, "give --noise-cov NAME (a stored noise covariance) or --recording R (to compute it from)");
  if (o.noise_cov && o.from_recording) usage(prog, "--noise-cov and --recording are alternatives");
  o.name = a.maybe("--name");
  o.replace = a.flag.count("--replace") > 0;
  o.average_reference = a.flag.count("--no-average-reference") == 0;
  KernelOptions k = a.flag.count("--mne") ? mneKernelOptions() : KernelOptions{};
  if (a.flag.count("--no-depth")) k.depth.enabled = false;
  if (const auto v = a.maybe("--depth-exponent")) k.depth.exponent = std::stod(*v);
  if (const auto v = a.maybe("--depth-limit")) k.depth.limit = std::stod(*v);
  if (const auto v = a.maybe("--depth-norm")) {
    if (*v == "frobenius") k.depth.norm = DepthNorm::Frobenius; else if (*v == "spectral") k.depth.norm = DepthNorm::Spectral; else usage(prog, "--depth-norm must be frobenius or spectral");
  }
  if (const auto v = a.maybe("--depth-limit-on")) {
    if (*v == "weight") k.depth.limitOn = DepthLimitOn::Weight; else if (*v == "norm") k.depth.limitOn = DepthLimitOn::Norm; else usage(prog, "--depth-limit-on must be weight or norm");
  }
  if (const auto v = a.maybe("--depth-channels")) {
    if (*v == "all") k.depth.channels = DepthChannels::All; else if (*v == "one-type") k.depth.channels = DepthChannels::OneType; else usage(prog, "--depth-channels must be all or one-type");
  }
  if (const auto v = a.maybe("--snr")) k.snr = std::stod(*v);
  if (const auto v = a.maybe("--snr-normalisation")) {
    if (*v == "all-channels") k.snrNormalisation = SnrNormalisation::AllChannels;
    else if (*v == "whitener-rank") k.snrNormalisation = SnrNormalisation::WhitenerRank;
    else usage(prog, "--snr-normalisation must be all-channels or whitener-rank");
  }
  if (const auto v = a.maybe("--regularisation")) {
    if (*v == "ridge") k.whitener.regularisation = Regularisation::Ridge;
    else if (*v == "diagonal") k.whitener.regularisation = Regularisation::Diagonal;
    else if (*v == "none") k.whitener.regularisation = Regularisation::None;
    else usage(prog, "--regularisation must be ridge, diagonal or none");
  }
  if (const auto v = a.maybe("--ridge-fraction")) k.whitener.ridgeFraction = std::stod(*v);
  if (a.flag.count("--joint-types")) k.whitener.jointTypes = true;
  if (a.flag.count("--ridge-over-rank")) k.whitener.ridgeOverRank = true;
  if (a.flag.count("--no-detect-diagonal")) k.whitener.detectDiagonal = false;
  if (const auto v = a.maybe("--whitener-form")) {
    if (*v == "symmetric") k.whitener.form = WhitenerForm::Symmetric; else if (*v == "pca") k.whitener.form = WhitenerForm::Pca; else usage(prog, "--whitener-form must be symmetric or pca");
  }
  if (const auto v = a.maybe("--rank-tolerance")) {
    if (*v == "single") k.whitener.rankTolerance = RankTolerance::SinglePrecision;
    else if (*v == "double") k.whitener.rankTolerance = RankTolerance::DoublePrecision;
    else usage(prog, "--rank-tolerance must be single or double");
  }
  o.kernel = k;
  DatasetWriter ds = open_writer(a.get("--dataset"));
  OpenSubjectOptions oo;
  oo.created_by = cli::stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  const auto r = nxr::pipelines::inverse_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
  Json out = Json{{"stage", "result"}};
  for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
  out["seconds"] = std::round(r.seconds * 1000) / 1000;
  cli::emit(out);
  return 0;
}

int headmodel_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro headmodel: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro headmodel compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::set<std::string> valued = {"--dataset", "--subject", "--name", "--channels", "--source-surface", "--meg", "--meg-surface", "--eeg-surface", "--sphere-cost", "--sensor-weight-exponent",
                                        "--eeg-radii", "--eeg-conductivities", "--eeg-series-terms", "--berg-dipoles", "--berg-terms", "--berg-starts"};
  const std::set<std::string> flags = {"--no-eeg", "--no-project-electrodes", "--mne-eeg-layers", "--eeg-series", "--replace"};
  const cli::Args a = cli::parse(prog, rest, valued, flags, {"--dataset", "--subject"});
  using namespace nxr::meeg;
  nxr::pipelines::HeadModelOptions o;
  o.name = a.maybe("--name");
  o.channels = a.maybe("--channels");
  o.source_surface = a.maybe("--source-surface");
  o.meg_surface = a.maybe("--meg-surface");
  o.eeg_surface = a.maybe("--eeg-surface");
  if (const auto v = a.maybe("--meg")) {
    if (*v == "overlapping") o.meg = nxr::pipelines::MegModel::OverlappingSpheres;
    else if (*v == "single") o.meg = nxr::pipelines::MegModel::SingleSphere;
    else if (*v == "none") o.meg = nxr::pipelines::MegModel::None;
    else usage(prog, "--meg must be overlapping, single or none");
  }
  o.eeg = a.flag.count("--no-eeg") == 0;
  o.replace = a.flag.count("--replace") > 0;
  o.project_electrodes = a.flag.count("--no-project-electrodes") == 0;
  if (const auto v = a.maybe("--sphere-cost")) {
    SphereCost c;
    if (*v == "geometric") c = SphereCost::Geometric; else if (*v == "algebraic") c = SphereCost::Algebraic; else usage(prog, "--sphere-cost must be geometric or algebraic");
    o.sphere_fit.cost = c;
    o.sensor_sphere.fit.cost = c;
  }
  if (const auto v = a.maybe("--sensor-weight-exponent")) o.sensor_sphere.weightExponent = std::stod(*v);
  if (a.flag.count("--mne-eeg-layers")) o.eeg_sphere = mneEegSphereOptions();
  auto vec = [&](const char* flag, Eigen::VectorXd& dst) {
    if (const auto v = a.maybe(flag)) { const auto n = numbers(prog, flag, *v); dst = Eigen::Map<const Eigen::VectorXd>(n.data(), static_cast<Eigen::Index>(n.size())); }
  };
  vec("--eeg-radii", o.eeg_sphere.radius);
  vec("--eeg-conductivities", o.eeg_sphere.conductivity);
  if (a.flag.count("--eeg-series")) o.eeg_sphere.approximation = EegApproximation::Series;
  if (const auto v = a.maybe("--eeg-series-terms")) o.eeg_sphere.seriesTerms = std::stoi(*v);
  if (const auto v = a.maybe("--berg-dipoles")) o.eeg_sphere.berg.dipoles = std::stoi(*v);
  if (const auto v = a.maybe("--berg-terms")) o.eeg_sphere.berg.terms = std::stoi(*v);
  if (const auto v = a.maybe("--berg-starts")) o.eeg_sphere.berg.starts = std::stoi(*v);
  DatasetWriter ds = open_writer(a.get("--dataset"));
  OpenSubjectOptions oo;
  oo.created_by = cli::stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  const auto r = nxr::pipelines::headmodel_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
  Json out = Json{{"stage", "result"}};
  for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
  out["seconds"] = std::round(r.seconds * 1000) / 1000;
  cli::emit(out);
  return 0;
}

int filter_main(const std::string& verb, const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro " + verb;
  const std::set<std::string> valued = {"--dataset", "--subject", "--recording", "--name", "--fpass", "--fstop", "--frequencies", "--line", "--harmonics", "--bandwidth",
                                        "--order", "--ripple", "--transition", "--padding", "--transient-samples", "--tolerance"};
  const std::set<std::string> flags = {"--no-remove-mean"};
  const cli::Args a = cli::parse(prog, rest, valued, flags, {"--dataset", "--subject", "--recording"});
  using namespace nxr::filter;
  nxr::pipelines::FilterComputeOptions o;
  o.recording = a.get("--recording");
  o.name = a.maybe("--name");
  FilterOptions& f = o.filter;
  f = brainstormFilterOptions(1.0);
  auto pair = [&](const char* flag) {
    const auto v = a.maybe(flag);
    if (!v) usage(prog, std::string(flag) + " LOW,HIGH is required");
    const auto n = numbers(prog, flag, *v);
    if (n.size() != 2) usage(prog, std::string(flag) + " needs two numbers: LOW,HIGH");
    return n;
  };
  auto single = [&](const char* flag) {
    const auto v = a.maybe(flag);
    if (!v) usage(prog, std::string(flag) + " HZ is required");
    const auto n = numbers(prog, flag, *v);
    if (n.size() != 1) usage(prog, std::string(flag) + " needs one number");
    return n[0];
  };
  if (verb == "lowpass") { f.kind = Kind::Lowpass; f.upperEdge = single("--fpass"); }
  else if (verb == "highpass") { f.kind = Kind::Highpass; f.lowerEdge = single("--fpass"); }
  else if (verb == "bandpass") { f.kind = Kind::Bandpass; const auto n = pair("--fpass"); f.lowerEdge = n[0]; f.upperEdge = n[1]; }
  else if (verb == "bandstop") { f.kind = Kind::Bandstop; const auto n = pair("--fstop"); f.lowerEdge = n[0]; f.upperEdge = n[1]; }
  else {
    f.kind = Kind::Notch;
    if (a.maybe("--frequencies") && a.maybe("--line")) usage(prog, "--frequencies and --line are alternatives");
    if (const auto v = a.maybe("--frequencies")) f.frequencies = numbers(prog, "--frequencies", *v);
    else if (a.maybe("--line")) {
      // the line frequency and its harmonics: N of them, else every multiple below Nyquist (the recording's rate is known only later: marked by a negative count)
      const double line = single("--line");
      const int n = a.maybe("--harmonics") ? std::stoi(*a.maybe("--harmonics")) : -1;
      f.frequencies = {line};
      f.transientSamples.reset();
      o.harmonics = n;
    } else usage(prog, "give --frequencies F1,F2,... or --line F");
  }
  if (const auto v = a.maybe("--bandwidth")) f.notchBandwidth = std::stod(*v);
  if (const auto v = a.maybe("--order")) f.bandstopOrder = std::stoi(*v);
  if (const auto v = a.maybe("--ripple")) f.ripple = std::stod(*v);
  if (const auto v = a.maybe("--transition")) f.transitionBandwidth = std::stod(*v);
  if (const auto v = a.maybe("--padding")) {
    if (*v == "mean") f.edge = Edge::MeanOfTransient; else if (*v == "zeros") f.edge = Edge::Zeros; else usage(prog, "--padding must be mean or zeros");
  }
  if (const auto v = a.maybe("--transient-samples")) f.transientSamples = static_cast<std::size_t>(std::stoll(*v));
  if (const auto v = a.maybe("--tolerance")) f.tolerance = std::stod(*v);
  f.removeMean = a.flag.count("--no-remove-mean") == 0;
  try {
    DatasetWriter ds = open_writer(a.get("--dataset"));
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::filter_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    out["seconds"] = std::round(r.seconds * 1000) / 1000;
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}
