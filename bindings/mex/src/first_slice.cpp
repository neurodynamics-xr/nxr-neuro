// The first slice of the MATLAB face: the pure functions (`compute.*`) and the dispatch of the readers (`io.*`, first_slice_io.cpp). Each command is one
// call into nxr_compute on arrays; the maths is the library's. Strings name the library's enums (the option names are the library's, as on the
// command line and in the Python wheel). Options arrive complete, positional: the defaults live in the MATLAB functions (+nxr/+compute), from the
// options structs' comments.
#include "first_slice.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>

#include "first_slice_internal.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/transport.h"
#include "nxr/signal/surface_calculus.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/curvature.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/geometry/facets.h"
#include "nxr/operators/cache.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/operators/hodge_decomposition.h"
#include "nxr/operators/assembled.h"
#include "nxr/synthesis/fields.h"
#include "nxr/signal/filters.h"
#ifdef NXR_MEX_WITH_IO
#include "nxr/io/errors.h"
#endif
#include "nxr/stats/pca.h"
#include "nxr/signal/spectral.h"

namespace nxr::mex {
#ifdef NXR_MEX_WITH_IO
bool dispatchIo(const std::string& cmd, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
#endif

namespace {

using namespace nxr::mex::slice;

void need(int nrhs, int n, const char* cmd) {
  if (nrhs != n + 1) throw std::invalid_argument(std::string("nxr_compute('") + cmd + "') takes " + std::to_string(n) + " arguments (use the nxr package functions)");
}

[[noreturn]] void bad_name(const char* what, const std::string& got, const char* allowed) {
  throw std::invalid_argument(std::string(what) + " '" + got + "' is not one of " + allowed);
}

nxr::signal::spectral::Window parse_window(const std::string& s) {
  using nxr::signal::spectral::Window;
  if (s == "hann") return Window::Hann;
  if (s == "hamming") return Window::Hamming;
  if (s == "blackman") return Window::Blackman;
  if (s == "rectangular") return Window::Rectangular;
  bad_name("window", s, "hann, hamming, blackman, rectangular");
}
nxr::signal::spectral::Detrend parse_detrend(const std::string& s) {
  using nxr::signal::spectral::Detrend;
  if (s == "none") return Detrend::None;
  if (s == "constant") return Detrend::Constant;
  if (s == "linear") return Detrend::Linear;
  bad_name("detrend", s, "none, constant, linear");
}
nxr::signal::spectral::Scaling parse_scaling(const std::string& s) {
  using nxr::signal::spectral::Scaling;
  if (s == "density") return Scaling::Density;
  if (s == "spectrum") return Scaling::Spectrum;
  if (s == "chain") return Scaling::Chain;
  bad_name("scaling", s, "density, spectrum, chain");
}
nxr::signal::filters::Method parse_method(const std::string& s) {
  if (s == "fir") return nxr::signal::filters::Method::Fir;
  if (s == "iir") return nxr::signal::filters::Method::Iir;
  bad_name("method", s, "fir, iir");
}
nxr::signal::filters::Edge parse_edge(const std::string& s) {
  if (s == "mean_of_transient") return nxr::signal::filters::Edge::MeanOfTransient;
  if (s == "zeros") return nxr::signal::filters::Edge::Zeros;
  bad_name("edge", s, "mean_of_transient, zeros");
}

// the arguments of a Welch call from a[0]: sfreq, segment_samples, overlap, window, periodic, detrend, scaling
nxr::signal::spectral::WelchOptions welch_options(const mxArray** a) {
  nxr::signal::spectral::WelchOptions o;
  o.sfreq = get_scalar(a[0], "sfreq");
  const double n = get_scalar(a[1], "segment_samples");
  if (n < 1 || n != std::floor(n)) throw std::invalid_argument("segment_samples must be a positive integer");
  o.segmentSamples = static_cast<std::size_t>(n);
  o.overlap = get_scalar(a[2], "overlap");
  o.window = parse_window(get_string(a[3], "window"));
  o.periodic = get_bool(a[4], "periodic");
  o.detrend = parse_detrend(get_string(a[5], "detrend"));
  o.scaling = parse_scaling(get_string(a[6], "scaling"));
  return o;
}

std::vector<std::uint8_t> get_good(const mxArray* a, Eigen::Index samples, bool& given) {
  given = !is_unset(a);
  std::vector<std::uint8_t> g;
  if (!given) return g;
  const Eigen::VectorXd v = get_vector(a, "good");
  if (v.size() != samples) throw std::invalid_argument("good must hold one value per sample");
  g.resize(static_cast<std::size_t>(v.size()));
  for (Eigen::Index i = 0; i < v.size(); ++i) g[static_cast<std::size_t>(i)] = v[i] != 0.0;
  return g;
}

// [frequencies, psd, segments, segments_skipped] = compute.signal.psd(x, sfreq, segment_samples, overlap, window, periodic, detrend, scaling, good)
void welchPsd(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 9, "compute.signal.psd");
  const Matrix x = get_matrix(prhs[1], "x");
  const auto opt = welch_options(prhs + 2);
  bool given = false;
  const auto good = get_good(prhs[9], x.m.cols(), given);
  nxr::signal::spectral::Welch w(static_cast<int>(x.m.rows()), opt);
  if (given) w.add(x.m, good); else w.add(x.m);
  nxr::signal::spectral::WelchResult r = w.result();
  plhs[0] = give_vector(r.frequencies, false, true);
  plhs[1] = give(r.psd, x.single);
  plhs[2] = give_scalar(static_cast<double>(r.segments));
  plhs[3] = give_scalar(static_cast<double>(r.segmentsSkipped));
}

// A stack of equally sized matrices as one [groups, rows, cols] array of the given precision (MATLAB column-major: element (g, i, j) at g + G (i + rows j)).
mxArray* give_stack(const std::vector<Eigen::MatrixXd>& stack, std::size_t rows, std::size_t cols, bool single) {
  const mwSize dims[3] = {static_cast<mwSize>(stack.size()), static_cast<mwSize>(rows), static_cast<mwSize>(cols)};
  mxArray* a = mxCreateNumericArray(3, dims, single ? mxSINGLE_CLASS : mxDOUBLE_CLASS, mxREAL);
  const std::size_t G = stack.size();
  for (std::size_t g = 0; g < G; ++g)
    for (std::size_t j = 0; j < cols; ++j)
      for (std::size_t i = 0; i < rows; ++i) {
        const double v = stack[g](static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        const std::size_t at = g + G * (i + rows * j);
        if (single) static_cast<float*>(mxGetData(a))[at] = static_cast<float>(v);
        else static_cast<double*>(mxGetData(a))[at] = v;
      }
  return a;
}

// [frequencies, real, imag, segments, segments_skipped] = compute.signal.xpsd(x, sfreq, segment_samples, overlap, window, periodic, detrend, scaling,
//                                                                                 bins, mean, quadrature, good)       bins: [G x 2], 1-based
void welchCross(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 12, "compute.signal.xpsd");
  const Matrix x = get_matrix(prhs[1], "x");
  const auto opt = welch_options(prhs + 2);
  const Matrix b = get_matrix(prhs[9], "bins");
  if (b.m.cols() != 2) throw std::invalid_argument("bins is [groups, 2]: first and last bin of each group");
  const bool mean = get_bool(prhs[10], "mean"), quadrature = get_bool(prhs[11], "quadrature");
  std::vector<nxr::signal::spectral::BinGroup> groups;
  for (Eigen::Index g = 0; g < b.m.rows(); ++g) {
    if (b.m(g, 0) < 1 || b.m(g, 1) < b.m(g, 0)) throw std::invalid_argument("each bin group runs from first to last, 1-based (bin 1 is DC)");
    groups.push_back({static_cast<std::size_t>(b.m(g, 0)) - 1, static_cast<std::size_t>(b.m(g, 1)) - 1, mean});
  }
  bool given = false;
  const auto good = get_good(prhs[12], x.m.cols(), given);
  nxr::signal::spectral::CrossWelch w(static_cast<int>(x.m.rows()), opt, groups, quadrature);
  if (given) w.add(x.m, good); else w.add(x.m);
  nxr::signal::spectral::CrossWelchResult r = w.result();
  const std::size_t c = static_cast<std::size_t>(x.m.rows());
  plhs[0] = give_vector(r.frequencies, false, true);
  plhs[1] = give_stack(r.S, c, c, x.single);
  plhs[2] = quadrature ? give_stack(r.Sq, c, c, x.single) : mxCreateDoubleMatrix(0, 0, mxREAL);
  plhs[3] = give_scalar(static_cast<double>(r.segments));
  plhs[4] = give_scalar(static_cast<double>(r.segmentsSkipped));
}

// y = compute.signal.<kind>(x, sfreq, lower, upper, frequencies, method, ripple, transition_bandwidth, edge, transient_energy, transient_samples, remove_mean,
//                       notch_bandwidth, bandstop_order); the kind is the command's last part. [] stands for an option not given.
void filterApply(const std::string& kind, int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 14, "compute.signal.*");
  const Matrix x = get_matrix(prhs[1], "x");
  nxr::signal::filters::FilterOptions o = nxr::signal::filters::brainstormFilterOptions(get_scalar(prhs[2], "sfreq"));
  const double lower = get_scalar(prhs[3], "lower"), upper = get_scalar(prhs[4], "upper");
  const Eigen::VectorXd freqs = get_vector(prhs[5], "frequencies");
  if (!is_unset(prhs[6])) o.method = parse_method(get_string(prhs[6], "method"));
  o.ripple = get_scalar(prhs[7], "ripple");
  if (!is_unset(prhs[8])) o.transitionBandwidth = get_scalar(prhs[8], "transition_bandwidth");
  o.edge = parse_edge(get_string(prhs[9], "edge"));
  o.transientEnergy = get_scalar(prhs[10], "transient_energy");
  if (!is_unset(prhs[11])) {
    const double k = get_scalar(prhs[11], "transient_samples");
    if (k < 0 || k != std::floor(k)) throw std::invalid_argument("transient_samples must be a non-negative integer");
    o.transientSamples = static_cast<std::size_t>(k);
  }
  o.removeMean = get_bool(prhs[12], "remove_mean");
  if (!is_unset(prhs[13])) o.notchBandwidth = get_scalar(prhs[13], "notch_bandwidth");
  o.bandstopOrder = static_cast<int>(get_scalar(prhs[14], "bandstop_order"));
  Eigen::MatrixXd y;
  if (kind == "lowpass") y = nxr::signal::filters::lowpass(x.m, upper, o);
  else if (kind == "highpass") y = nxr::signal::filters::highpass(x.m, lower, o);
  else if (kind == "bandpass") y = nxr::signal::filters::bandpass(x.m, lower, upper, o);
  else if (kind == "bandstop") y = nxr::signal::filters::bandstop(x.m, lower, upper, o);
  else if (kind == "notch") y = nxr::signal::filters::notch(x.m, std::vector<double>(freqs.data(), freqs.data() + freqs.size()), o);
  else throw std::invalid_argument("filter kind '" + kind + "' is not one of lowpass, highpass, bandpass, bandstop, notch");
  plhs[0] = give(y, x.single);
}

// [components, singular_values, explained_variance, explained_variance_ratio, scores, mean, scale, rank] = compute.stats.pca(x, scale)
void pcaFit(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 2, "compute.stats.pca");
  const Matrix x = get_matrix(prhs[1], "x");
  nxr::stats::PcaOptions o;
  const std::string s = get_string(prhs[2], "scale");
  if (s == "none") o.scale = nxr::stats::Scale::None;
  else if (s == "unit_variance") o.scale = nxr::stats::Scale::UnitVariance;
  else bad_name("scale", s, "none, unit_variance");
  nxr::stats::PcaResult r = nxr::stats::pca(x.m, o);
  plhs[0] = give(r.components, x.single);
  plhs[1] = give_vector(r.singularValues, x.single, true);
  plhs[2] = give_vector(r.explainedVariance, x.single, true);
  plhs[3] = give_vector(r.explainedVarianceRatio, x.single, true);
  plhs[4] = give(r.scores, x.single);
  plhs[5] = give_vector(r.mean, x.single, true);
  plhs[6] = give_vector(r.scale, x.single, true);
  plhs[7] = give_scalar(static_cast<double>(r.rank));
}

// y = compute.stats.reconstruct(scores, components, mean, scale, k)
void pcaReconstruct(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 5, "compute.stats.reconstruct");
  const Matrix scores = get_matrix(prhs[1], "scores"), comps = get_matrix(prhs[2], "components");
  const Eigen::VectorXd mean = get_vector(prhs[3], "mean"), scale = get_vector(prhs[4], "scale");
  const double kd = get_scalar(prhs[5], "k");
  const Eigen::Index k = static_cast<Eigen::Index>(kd);
  if (kd < 0 || kd != std::floor(kd) || k > comps.m.rows() || k > scores.m.cols()) throw std::invalid_argument("k is from 0 to the number of components held");
  if (comps.m.cols() != mean.size() || mean.size() != scale.size()) throw std::invalid_argument("mean and scale hold one value per feature");
  if (scores.m.cols() != comps.m.rows()) throw std::invalid_argument("scores and components hold the same number of components");
  plhs[0] = give(nxr::stats::reconstruct(scores.m, comps.m, mean, scale, static_cast<int>(k)), scores.single);
}

// [eigenvalues, eigenvectors, n_converged] = compute.operators.eigen(vertices, faces, k, sigma, normalize, remove_dc, intrinsic_delaunay); faces 1-based
void manifoldEigenmodes(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 7, "compute.operators.eigen");
  const Matrix v = get_matrix(prhs[1], "vertices");
  const Matrix f = get_matrix(prhs[2], "faces");
  if (v.m.cols() != 3 || f.m.cols() != 3) throw std::invalid_argument("vertices and faces are [n, 3]");
  const int nV = static_cast<int>(v.m.rows()), nF = static_cast<int>(f.m.rows());
  std::vector<double> xyz(static_cast<std::size_t>(nV) * 3);
  for (int i = 0; i < nV; ++i)
    for (int j = 0; j < 3; ++j) xyz[static_cast<std::size_t>(3 * i + j)] = v.m(i, j);
  std::vector<std::int32_t> tri(static_cast<std::size_t>(nF) * 3);
  for (int i = 0; i < nF; ++i)
    for (int j = 0; j < 3; ++j) {
      const double idx = f.m(i, j);
      if (idx < 1 || idx > nV || idx != std::floor(idx)) throw std::invalid_argument("a face names a vertex outside the mesh (indices are 1-based)");
      tri[static_cast<std::size_t>(3 * i + j)] = static_cast<std::int32_t>(idx) - 1;
    }
  const int k = static_cast<int>(get_scalar(prhs[3], "k"));
  const bool hasSigma = !is_unset(prhs[4]);
  const double sigma = hasSigma ? get_scalar(prhs[4], "sigma") : 0.0;
  const bool normalize = get_bool(prhs[5], "normalize"), removeDc = get_bool(prhs[6], "remove_dc"), delaunay = get_bool(prhs[7], "intrinsic_delaunay");
  nxr::geometry::Manifold m(xyz.data(), nV, tri.data(), nF, delaunay);
  auto ops = nxr::operators::assembleManifoldOperators(m);
  auto r = nxr::operators::eigen(ops.cotanLaplacian, ops.mass, k, hasSigma ? sigma : nxr::operators::kShiftLadder, normalize, removeDc);
  plhs[0] = give_vector(r.eigenvalues, false, false);
  plhs[1] = give(r.eigenvectors, false);
  plhs[2] = give_scalar(static_cast<double>(r.nConverged));
}

// The commands renamed to the toolbox words (one word per thing across the C++ function, the node and the bindings; the test names_across_layers reads this table):
// an old string is REFUSED with the new name, never answered as an alias.
const char* const kRetiredCommands[][2] = {
    {"compute.welch.psd", "compute.signal.psd"},
    {"compute.welch.cross_spectrum", "compute.signal.xpsd"},
    {"compute.pca.pca", "compute.stats.pca"},
    {"compute.pca.reconstruct", "compute.stats.reconstruct"},
    {"compute.manifold.eigenmodes", "compute.operators.eigen"},
};
// "compute.filter.<kind>" became "compute.signal.<kind>"
bool retired(const std::string& cmd, std::string& renamed) {
  for (const auto& r : kRetiredCommands) if (cmd == r[0]) { renamed = r[1]; return true; }
  if (cmd.rfind("compute.filter.", 0) == 0) { renamed = "compute.signal." + cmd.substr(15); return true; }
  return false;
}

bool dispatchOps(const std::string& cmd, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  if (cmd == "compute.signal.psd") welchPsd(nlhs, plhs, nrhs, prhs);
  else if (cmd == "compute.signal.xpsd") welchCross(nlhs, plhs, nrhs, prhs);
  else if (cmd.rfind("compute.signal.", 0) == 0) filterApply(cmd.substr(15), nlhs, plhs, nrhs, prhs);
  else if (cmd == "compute.stats.pca") pcaFit(nlhs, plhs, nrhs, prhs);
  else if (cmd == "compute.stats.reconstruct") pcaReconstruct(nlhs, plhs, nrhs, prhs);
  else if (cmd == "compute.operators.eigen") manifoldEigenmodes(nlhs, plhs, nrhs, prhs);
  else return false;
  return true;
}

#ifdef NXR_MEX_WITH_IO
// the identifier of a library error code: nxr:formatError, nxr:unsupported, nxr:invalidArgument, nxr:io
std::string identifier(const std::string& code) {
  if (code == "format_error") return "nxr:formatError";
  if (code == "unsupported") return "nxr:unsupported";
  if (code == "invalid_argument") return "nxr:invalidArgument";
  if (code == "io") return "nxr:io";
  return "nxr:error";
}
#endif

}  // namespace

bool dispatchFirstSlice(const std::string& cmd, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  if (cmd.rfind("compute.", 0) != 0 && cmd.rfind("io.", 0) != 0) return false;
  bool known = true;
  std::string renamed;
  if (retired(cmd, renamed)) throw SliceFailure{"nxr:renamed", "The command \"" + cmd + "\" was renamed: use \"" + renamed + "\"."};
  try {
    known = dispatchOps(cmd, nlhs, plhs, nrhs, prhs);
#ifdef NXR_MEX_WITH_IO
    if (!known) known = dispatchIo(cmd, nlhs, plhs, nrhs, prhs);
#endif
#ifdef NXR_MEX_WITH_IO
  } catch (const nxr::io::Error& e) {
    throw SliceFailure{identifier(e.code()), e.what()};
#endif
  } catch (const std::invalid_argument& e) {
    throw SliceFailure{"nxr:invalidArgument", e.what()};
  }
  if (!known) throw SliceFailure{"nxr:unknownCommand", "Unknown command: \"" + cmd + "\"."};
  return true;
}

}  // namespace nxr::mex
