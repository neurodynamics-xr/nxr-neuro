// nxr-neuro speedspectrum compute --dataset D/N --subject S --recording R [--kernel NAME] [--package STEM] [--modes N] [--name STEM] [--chunks K|A-B|all | --start SEC --stop SEC]
//     --graph-family mexhat|heat|itersine|logitersine [--graph-voices V] [--graph-filters N]
//     --family itersine|meyer|morlet|morse|amor|bump --gain G --sides analytic|two-sided [--centres F1,F2,..] [--cycles C] [--voices V] [--anchor HZ] [--limits LO-HI] [--half-width OCT] [--gamma G] [--time-bandwidth B]
//     [--fmin HZ] [--fmax HZ] [--speed-bins N] [--speed-min C] [--speed-max C]
//     [the psd options: --set tiles|brainstorm|scipy|nxr-chain --segment-seconds|--segment-samples|--segment-level --overlap --window --periodic|--symmetric --detrend --scaling --no-bad-segments] [--replace]
// The SPEED SPECTRUM of the recording's cortical activity (nxr/speedspectrum.h): the Hodge-Dirac modes' power per frequency bin, tiled by the joint (eigenvalue x frequency) filterbank, read as
// energy per phase speed (2 pi f = c mu), per spatial scale, for exact and coexact modes apart, with each tile's quadrature fraction (travelling versus standing). The graph bank and the time bank
// have no default family; the time bank's gain and sides are required. A free-orientation kernel only. JSON lines out, as the other pipelines.
#include <iostream>
#include <map>
#include <sstream>

#include "cli.h"
#include "nxr/pipelines/speedspectrum.h"

using namespace nxr::io;

namespace {
std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::istringstream in(s);
  for (std::string t; std::getline(in, t, sep);) out.push_back(t);
  return out;
}
[[noreturn]] void usage(const std::string& prog, const std::string& m) { throw cli::Usage(prog + ": error: " + m); }
}  // namespace

int speedspectrum_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro speedspectrum: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro speedspectrum compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::set<std::string> valued = {"--dataset", "--subject", "--recording", "--kernel", "--package", "--modes", "--name", "--chunks", "--start", "--stop", "--set", "--segment-seconds",
                                        "--segment-samples", "--segment-level", "--overlap", "--window", "--detrend", "--scaling", "--graph-family", "--graph-voices", "--graph-filters", "--family", "--gain", "--sides",
                                        "--centres", "--cycles", "--voices", "--anchor", "--limits", "--half-width", "--gamma", "--time-bandwidth", "--fmin", "--fmax", "--speed-bins", "--speed-min", "--speed-max"};
  const std::set<std::string> flags = {"--periodic", "--symmetric", "--no-bad-segments", "--replace"};
  const cli::Args a = cli::parse(prog, rest, valued, flags, {"--dataset", "--subject", "--recording"});
  nxr::pipelines::SpeedSpectrumOptions o;
  using namespace nxr::welch;
  o.recording = a.get("--recording");
  o.kernel = a.maybe("--kernel");
  o.package = a.maybe("--package");
  if (const auto v = a.maybe("--modes")) o.modes = std::stoll(*v);
  o.name = a.maybe("--name");
  if (const auto v = a.maybe("--start")) o.start = std::stod(*v);
  if (const auto v = a.maybe("--stop")) o.stop = std::stod(*v);
  o.chunks = a.maybe("--chunks");
  // the Welch options (as psd compute)
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
  o.exclude_bad_segments = a.flag.count("--no-bad-segments") == 0;
  // the joint bank
  namespace fb = nxr::filterbank;
  if (const auto v = a.maybe("--graph-family")) {
    const std::map<std::string, fb::Family> fam = {{"mexhat", fb::Family::MexHat}, {"heat", fb::Family::Heat}, {"itersine", fb::Family::Itersine}, {"logitersine", fb::Family::LogItersine}};
    const auto it = fam.find(*v);
    if (it == fam.end()) usage(prog, "--graph-family is mexhat, heat, itersine or logitersine");
    o.graph_family = it->second;
  } else usage(prog, "--graph-family is required (no default)");
  if (const auto v = a.maybe("--graph-voices")) o.graph_voices = std::stoi(*v);
  if (const auto v = a.maybe("--graph-filters")) o.graph_filters = std::stoi(*v);
  using F = nxr::timebank::Family;
  if (const auto v = a.maybe("--family")) {
    const std::map<std::string, F> fam = {{"itersine", F::Itersine}, {"meyer", F::Meyer}, {"morlet", F::Morlet}, {"morse", F::Morse}, {"amor", F::Amor}, {"bump", F::Bump}};
    const auto it = fam.find(*v);
    if (it == fam.end()) usage(prog, "--family is itersine, meyer, morlet, morse, amor or bump");
    o.family = it->second;
  } else usage(prog, "--family is required (no default)");
  if (const auto v = a.maybe("--gain")) o.gain = std::stod(*v); else usage(prog, "--gain is required (the peak in-band response of every time member)");
  if (const auto v = a.maybe("--sides")) {
    if (*v == "analytic") o.sides = nxr::timebank::Sides::Analytic;
    else if (*v == "two-sided") o.sides = nxr::timebank::Sides::TwoSided;
    else usage(prog, "--sides is analytic or two-sided");
  } else usage(prog, "--sides is required (analytic or two-sided)");
  if (const auto v = a.maybe("--centres")) for (const auto& t : split(*v, ',')) o.centres.push_back(std::stod(t));
  if (const auto v = a.maybe("--cycles")) o.cycles = std::stod(*v);
  if (const auto v = a.maybe("--voices")) o.voices = std::stoi(*v);
  if (const auto v = a.maybe("--anchor")) o.anchor = std::stod(*v);
  if (const auto v = a.maybe("--half-width")) o.half_width = std::stod(*v);
  if (const auto v = a.maybe("--gamma")) o.gamma = std::stod(*v);
  if (const auto v = a.maybe("--time-bandwidth")) o.time_bandwidth = std::stod(*v);
  if (const auto v = a.maybe("--limits")) {
    const auto dash = v->find('-');
    if (dash == std::string::npos) usage(prog, "--limits is lo-hi (Hz), e.g. 2-40");
    o.limits = std::make_pair(std::stod(v->substr(0, dash)), std::stod(v->substr(dash + 1)));
  }
  if (const auto v = a.maybe("--fmin")) o.fmin = std::stod(*v);
  if (const auto v = a.maybe("--fmax")) o.fmax = std::stod(*v);
  if (const auto v = a.maybe("--speed-bins")) o.speed.speedBins = std::stoi(*v);
  if (const auto v = a.maybe("--speed-min")) o.speed.speedMin = std::stod(*v);
  if (const auto v = a.maybe("--speed-max")) o.speed.speedMax = std::stod(*v);
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_dataset(a);
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::speedspectrum_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    out["seconds"] = std::round(r.seconds * 1000) / 1000;
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}
