// INTERNAL to nxr-pipelines (not the public API): what a pipeline that reads a recording through the Hodge-Dirac modes needs from meeg.cpp, whose helpers
// are file-static. The sensor cross-spectrum per frequency bin, read through the fused weights of the free kernel (the flow weights b), chunk by chunk:
//   S_b(f) = B_b S_sensor(f) B_b^T   (B_b the K x C flow rows; Welch of B_b x is the same, so the K channels are fed to CrossWelch directly)
#pragma once
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <functional>

#include <Eigen/Dense>

#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/signal/speedspectrum.h"

namespace nxr::pipelines::internal {

// THE STORED PACKAGE'S SCALAR MODES (spatial_band.cpp): a subject's Hodge–Dirac package of a surface holds the Laplace–Beltrami (exact) modes of its GLUED surface
// (the surface itself for one closed piece; a cortex's hemispheres glued by anatomy: the medial walls cut, a bridge grown). Read BY NAME, never recomputed here.
struct StoredLaplace {
  bool found = false;                                  // a complete package of the surface is stored
  std::string package, glued_surface;                  // its stem (hodgeDirac[__<surface>]); the surface its eigenvectors live on
  std::int64_t modes_total = 0, n_exact = 0, n_coexact = 0, glued_vertices = 0, kept = 0;
  double lambda_min = 0, lambda_max = 0;               // of the EXACT modes, lambda = mu^2 (1/m^2); the package is cut at ONE lambda
};
StoredLaplace stored_laplace_info(nxr::io::SubjectWriter& sub, const SurfaceInput& surface);   // no big arrays read
struct LaplaceModes {
  std::string package;
  Eigen::MatrixXd phi;                                 // [V_glued x k] unit-norm in the vertex mass `mass`; the constant (lambda 0) first when wanted
  Eigen::VectorXd lambda, mass;                        // [k] 1/m^2; [V_glued] the glued mesh's mixed-Voronoi mass
  std::vector<std::int32_t> support;                   // [kept] the input surface's vertex of each of the glued surface's first rows (the rest is the bridge)
};
// THE GLUED DOMAIN of a surface: the surface the stored package lives on (the glued mesh of a two-hemisphere cortex: the kept vertices first, in `support` order, then the bridge's;
// the surface itself for one closed piece), and which of the input's vertices its first rows are. With no package: a one-piece surface is its own domain; a cortex of two pieces has none (`why` says what to run).
struct GluedDomain {
  bool found = false;
  SurfaceInput surface;                                // the glued surface (as stored)
  std::vector<std::int32_t> support;                   // [kept] the input surface's vertex of each of the surface's first rows
  std::string why;                                     // not found: the reason
};
GluedDomain glued_domain(nxr::io::SubjectWriter& sub, const SurfaceInput& surface);
// the exact modes whose lambda `want` accepts (a read of the stored eigenvectors; refuses a package that does not fit the surface)
LaplaceModes read_laplace_modes(nxr::io::SubjectWriter& sub, const SurfaceInput& surface, const std::function<bool(double)>& want);

struct HodgeSpectraRequest {
  std::string recording;
  std::optional<std::string> kernel, package;
  std::optional<std::int64_t> modes;
  std::optional<std::string> chunks;                      // what is loaded: chunk "k", chunks "a-b", or "all" (the whole recording, a loop over its chunks); REQUIRED: the caller states the span (or start and stop)
  std::optional<double> start, stop;                      // or an explicit span [start, stop) in seconds on the recording's clock (both given)
  std::optional<double> segment_seconds = 4.0;
  std::optional<std::int64_t> segment_samples;
  std::optional<int> segment_level;                       // under the set `tiles`: the segment is 2^segment_level samples (wins over seconds)
  nxr::signal::spectral::WelchOptions welch = nxr::signal::spectral::brainstormPsdOptions(1.0);
  std::string option_set = "tiles";
  bool exclude_bad_segments = true;
  double fmin = 1.0, fmax = 40.0;                         // Hz: the bins kept (DC and Nyquist never)
  std::string who = "speedspectrum";
};

// What the cache key and the stored rows need, resolved without computing anything.
struct HodgeSpectraPlan {
  std::string recording, recording_id, kernel, kernel_id, package, package_id, axis_id, surface_id;
  std::optional<std::string> acquisition_id;
  std::int64_t modes = 0, s0 = 0, s1 = 0, first_bin = 0, last_bin = 0, chunk_width = 0;   // chunk_width: the recording's (chunks.h)
  double sfreq = 0, t0 = 0, df = 0;
  nxr::signal::spectral::WelchOptions welch;
  nxr::io::Json params;                                   // the choices (cache key part)
  nxr::io::Json input;                                    // the identity of what it is run on (cache key part)
  nxr::io::Json projectors;                               // projector_record of the recording (projectors, applied_in_samples, not_applied, ctf_compensation)
  nxr::io::Json bad_segments;                             // null, or {selection, n_spans}
  std::string span;                                       // the chunks loaded (the chunk service)
  nxr::io::Json chunks_read;                              // [first, last] chunk read
};
struct HodgeSpectraRun {
  Eigen::VectorXd mu;                                     // [K]
  std::vector<int> kind;                                  // [K] 0 exact · 2 coexact
  nxr::signal::speedspectrum::ModeSpectra spectra;
  std::int64_t segments = 0, skipped = 0;
  double excluded_fraction = 0;
};


// THE STATISTICS' COMMON PIECES (docs/specs/2026-11-13-statistics.md), by the recording's name: where a statistic of it lives, the Welch segment partition a
// Welch statistic is computed from (a closed-form spans Selection on its time line; returns its id), and the Welch options as a statistic records them
std::string statistic_path_of(nxr::io::SubjectWriter& sub, const std::string& recording, const std::string& name);
std::string segments_path_of(nxr::io::SubjectWriter& sub, const std::string& recording, const std::string& stem);
std::string welch_segments_of(nxr::io::SubjectWriter& sub, const std::string& recording, const std::string& stem, std::int64_t s0, std::int64_t s1,
                              const nxr::signal::spectral::WelchOptions& wo, std::int64_t segments, std::int64_t skipped);
nxr::io::Json welch_options_of(const nxr::signal::spectral::WelchOptions& wo, const std::string& option_set, bool exclude_bad);
// meeg.cpp's own helpers, for the store step
std::string clear_stored_nodes(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::vector<std::pair<const char*, std::string>>& nodes, bool replace, const std::string& what, bool& replaced);
std::string producer_string(const char* verb);
const char* window_label(nxr::signal::spectral::Window w);
const char* detrend_label(nxr::signal::spectral::Detrend d);
const char* scaling_label(nxr::signal::spectral::Scaling s);

}  // namespace nxr::pipelines::internal
