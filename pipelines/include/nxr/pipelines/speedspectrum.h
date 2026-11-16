// nxr-pipelines — the SPEED SPECTRUM of a recording's cortical activity, from the joint (eigenvalue x frequency) analysis: an alternative way to measure
// propagation without optical flow. A wave at phase speed c puts its energy where 2 pi f = c mu (mu = sqrt(lambda), the Hodge-Dirac modes' wavenumber), so
// the modes' power per frequency bin, tiled by the joint filterbank, says how much energy moves at each speed, at each spatial scale. The maths and the
// meaning of each output: nxr/speedspectrum.h.
//
// The recording is read CHUNK BY CHUNK, through the fused weights of a FREE-orientation kernel (a fixed one is refused) and the stored Hodge-Dirac package; the
// flow weights b feed nxr::welch::CrossWelch with one-bin groups (real and imaginary parts), exact and coexact modes kept apart.
//
// Stored (rows and arrays together, CACHED), per kind K in {exact, coexact}, with `<stem>` = `<recording>_<kernel>_speedspectrum` by default:
//   <stem>_<K>_energy       Field "speed spectrum energy"      float32 [graph members x time members]   the joint tiles' energy (sensor power units x the Welch scaling)
//   <stem>_<K>_tile_speed   Field "speed spectrum tile speed"  float32 [graph members x time members]   the tile's centre speed 2 pi f_c / mu_c (length/s)
//   <stem>_<K>_quadrature   Field "speed spectrum quadrature"  float32 [graph members x time members]   ||Im C||_F / ||C||_F of the tile's mode cross-spectrum (travelling vs standing)
//   <stem>_<K>_histogram    Field "speed spectrum histogram"   float32 [graph members x speed bins]     energy per spatial scale on log-spaced phase speeds
// on the product of index lines `<stem>_graph_member` x `<stem>_time_member` (or `<stem>_speed_bin`). The producer record of every field holds every option, the
// bank designs, the member and bin axes (centre wavenumbers, centre frequencies, speed edges and centres) and the energy outside the speed bins.
#pragma once
#include <optional>
#include <string>
#include <vector>

#include "nxr/filterbank.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/speedspectrum.h"
#include "nxr/timebank.h"

namespace nxr::pipelines {

struct SpeedSpectrumOptions {
  std::string recording;                                  // the recording Field by name
  std::optional<std::string> kernel;                      // the FREE-orientation inverse kernel Operator by name; default: the subject's only one
  std::optional<std::string> package;                     // the Hodge-Dirac package's stem; default: the one of the kernel's surface
  std::optional<std::int64_t> modes;                      // use the first N modes of the package (default: all)
  std::optional<std::string> name;                        // the stem of the names; default `<recording>_<kernel>_speedspectrum`
  std::optional<std::string> chunks;                      // what is loaded: chunk "k", chunks "a-b", or "all" (the whole recording, a loop over its chunks); REQUIRED: the caller states the span (or start and stop)
  std::optional<double> start, stop;                      // or an explicit span [start, stop) in seconds on the recording's clock (both given)
  std::optional<double> segment_seconds = 4.0;
  std::optional<std::int64_t> segment_samples;
  std::optional<int> segment_level;                       // under the set `tiles`: the segment is 2^segment_level samples (wins over seconds)
  nxr::welch::WelchOptions welch = nxr::welch::brainstormPsdOptions(1.0);
  std::string option_set = "tiles";
  bool exclude_bad_segments = true;
  double fmin = 1.0, fmax = 40.0;                         // Hz: the frequency bins used; the time bank should cover them
  // the joint bank. The graph bank (over lambda = mu^2) and the time bank (over f) have NO default family; the time bank's gain and sides are required too.
  std::optional<nxr::filterbank::Family> graph_family;
  int graph_voices = 3;                                   // members per octave (graph)
  int graph_filters = 0;                                  // graph member count; 0: from voices
  std::optional<nxr::timebank::Family> family;            // the time bank
  double gain = 0;
  std::optional<nxr::timebank::Sides> sides;              // Analytic: the cross-spectrum is one-sided (f > 0); a TwoSided bank's negative members stay empty
  std::vector<double> centres;                            // Morlet, Morse, Amor, Bump: the member centres (Hz)
  double cycles = 6;
  int voices = 4;                                         // Itersine: voices per octave
  double anchor = 1.0;
  std::optional<std::pair<double, double>> limits;        // Itersine, Meyer: the bank's frequency limits (Hz)
  double half_width = 1.0 / 3.0;                          // Meyer
  double gamma = 3, time_bandwidth = 60;                  // Morse
  nxr::speedspectrum::Options speed;                      // bins (log-spaced) and range
  bool replace = false;
};

struct SpeedSpectrumResult {
  bool hit = false;
  double seconds = 0;
  nxr::io::Json info;
  nxr::speedspectrum::Result result;                      // (empty on a cache hit)
};

// A fixed-orientation kernel is an nxr::io::Error: the coefficients need the current's three components.
SpeedSpectrumResult speedspectrum_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const SpeedSpectrumOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines
