// nxr_neuro._core, the pure functions: Welch's PSD and cross-spectrum, the filters, PCA, the Laplace-Beltrami eigenmodes. Each is one call into nxr_compute
// on arrays; the maths is the library's. Strings name the library's enums (the option names are the library's, as on the command line).
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "arrays.h"
#include "nxr/geometry/transport.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/signal/surface_calculus.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/curvature.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/geometry/facets.h"
#include "nxr/operators/cache.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/operators/hodge_decomposition.h"
#include "nxr/operators/facets.h"
#include "nxr/synthesis/fields.h"
#include "nxr/signal/filters.h"
#include "nxr/stats/pca.h"
#include "nxr/signal/spectral.h"

namespace nxrpy {

namespace {

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

nxr::signal::spectral::WelchOptions welch_options(double sfreq, size_t segment_samples, double overlap, const std::string& window, bool periodic,
                                       const std::string& detrend, const std::string& scaling) {
  nxr::signal::spectral::WelchOptions o;
  o.sfreq = sfreq;
  o.segmentSamples = segment_samples;
  o.overlap = overlap;
  o.window = parse_window(window);
  o.periodic = periodic;
  o.detrend = parse_detrend(detrend);
  o.scaling = parse_scaling(scaling);
  return o;
}

void check_good(const std::optional<Array1<bool>>& good, size_t samples) {
  if (good && good->shape(0) != samples) throw std::invalid_argument("good must hold one value per sample");
}
std::vector<std::uint8_t> good_vector(const Array1<bool>& g) {
  const bool* p = g.data();
  return std::vector<std::uint8_t>(p, p + g.shape(0));
}

// ---- Welch ----

template <class T>
nb::tuple welch_psd(const Array2<T>& x, double sfreq, size_t segment_samples, double overlap, const std::string& window, bool periodic,
                    const std::string& detrend, const std::string& scaling, const std::optional<Array1<bool>>& good) {
  check_good(good, x.shape(1));
  const auto opt = welch_options(sfreq, segment_samples, overlap, window, periodic, detrend, scaling);
  Eigen::MatrixXd X = to_matrix<T>(x);
  nxr::signal::spectral::WelchResult r;
  {
    nb::gil_scoped_release release;
    const std::vector<std::uint8_t> mask = good ? good_vector(*good) : std::vector<std::uint8_t>();
    r = nxr::signal::spectral::psd(X, opt, good ? &mask : nullptr);
  }
  return nb::make_tuple(hand_over(std::move(r.frequencies)), give<T>(std::move(r.psd)), r.segments, r.segmentsSkipped);
}

template <class T>
nb::tuple welch_cross_spectrum(const Array2<T>& x, double sfreq, size_t segment_samples, double overlap, const std::string& window, bool periodic,
                               const std::string& detrend, const std::string& scaling, const std::vector<std::pair<size_t, size_t>>& bins,
                               bool mean, bool quadrature, const std::optional<Array1<bool>>& good) {
  check_good(good, x.shape(1));
  const auto opt = welch_options(sfreq, segment_samples, overlap, window, periodic, detrend, scaling);
  std::vector<nxr::signal::spectral::BinGroup> groups;
  for (const auto& b : bins) groups.push_back({b.first, b.second, mean});
  Eigen::MatrixXd X = to_matrix<T>(x);
  nxr::signal::spectral::CrossWelchResult r;
  {
    nb::gil_scoped_release release;
    const std::vector<std::uint8_t> mask = good ? good_vector(*good) : std::vector<std::uint8_t>();
    r = nxr::signal::spectral::xpsd(X, opt, groups, quadrature, good ? &mask : nullptr);
  }
  const size_t c = static_cast<size_t>(X.rows());
  nb::object imag = quadrature ? give_stack<T>(r.Sq, c, c) : nb::none();
  return nb::make_tuple(hand_over(std::move(r.frequencies)), give_stack<T>(r.S, c, c), imag, r.segments, r.segmentsSkipped);
}

// ---- filters ----

template <class T>
nb::object filter_apply(const std::string& kind, const Array2<T>& x, double sfreq, double lower, double upper, const std::vector<double>& frequencies,
                        const std::optional<std::string>& method, double ripple, const std::optional<double>& transition_bandwidth,
                        const std::string& edge, double transient_energy, const std::optional<size_t>& transient_samples, bool remove_mean,
                        const std::optional<double>& notch_bandwidth, int bandstop_order) {
  nxr::signal::filters::FilterOptions o = nxr::signal::filters::brainstormFilterOptions(sfreq);
  o.sfreq = sfreq;
  if (method) o.method = parse_method(*method);
  o.ripple = ripple;
  o.transitionBandwidth = transition_bandwidth;
  o.edge = parse_edge(edge);
  o.transientEnergy = transient_energy;
  o.transientSamples = transient_samples;
  o.removeMean = remove_mean;
  o.notchBandwidth = notch_bandwidth;
  o.bandstopOrder = bandstop_order;
  Eigen::MatrixXd X = to_matrix<T>(x);
  Eigen::MatrixXd Y;
  {
    nb::gil_scoped_release release;
    if (kind == "lowpass") Y = nxr::signal::filters::lowpass(X, upper, o);
    else if (kind == "highpass") Y = nxr::signal::filters::highpass(X, lower, o);
    else if (kind == "bandpass") Y = nxr::signal::filters::bandpass(X, lower, upper, o);
    else if (kind == "bandstop") Y = nxr::signal::filters::bandstop(X, lower, upper, o);
    else if (kind == "notch") Y = nxr::signal::filters::notch(X, frequencies, o);
    else throw std::invalid_argument("filter kind '" + kind + "' is not one of lowpass, highpass, bandpass, bandstop, notch");
  }
  return give<T>(std::move(Y));
}

// ---- PCA ----

template <class T>
nb::tuple pca_fit(const Array2<T>& x, const std::string& scale) {
  nxr::stats::PcaOptions o;
  if (scale == "none") o.scale = nxr::stats::Scale::None;
  else if (scale == "unit_variance") o.scale = nxr::stats::Scale::UnitVariance;
  else bad_name("scale", scale, "none, unit_variance");
  Eigen::MatrixXd X = to_matrix<T>(x);
  nxr::stats::PcaResult r;
  {
    nb::gil_scoped_release release;
    r = nxr::stats::pca(X, o);
  }
  return nb::make_tuple(give<T>(std::move(r.components)), give<T>(std::move(r.singularValues)), give<T>(std::move(r.explainedVariance)),
                        give<T>(std::move(r.explainedVarianceRatio)), give<T>(std::move(r.scores)), give<T>(std::move(r.mean)),
                        give<T>(std::move(r.scale)), r.rank);
}

template <class T>
nb::object pca_reconstruct(const Array2<T>& scores, const Array2<T>& components, const Array1<T>& mean, const Array1<T>& scale, int k) {
  if (k < 0 || static_cast<size_t>(k) > components.shape(0) || static_cast<size_t>(k) > scores.shape(1))
    throw std::invalid_argument("k is from 0 to the number of components held");
  if (components.shape(1) != mean.shape(0) || mean.shape(0) != scale.shape(0)) throw std::invalid_argument("mean and scale hold one value per feature");
  if (scores.shape(1) != components.shape(0)) throw std::invalid_argument("scores and components hold the same number of components");
  Eigen::MatrixXd S = to_matrix<T>(scores), C = to_matrix<T>(components);
  Eigen::VectorXd mu = to_vector<T>(mean), sc = to_vector<T>(scale);
  Eigen::MatrixXd out;
  {
    nb::gil_scoped_release release;
    out = nxr::stats::reconstruct(S, C, mu, sc, k);
  }
  return give<T>(std::move(out));
}

// ---- eigenmodes of the Laplace-Beltrami operator on a triangle mesh ----

nb::tuple manifold_eigenmodes(const Array2<double>& vertices, const Array2<int32_t>& faces, int k, const std::optional<double>& sigma, bool normalize,
                              bool remove_dc, bool intrinsic_delaunay) {
  if (vertices.shape(1) != 3 || faces.shape(1) != 3) throw std::invalid_argument("vertices and faces are [n, 3]");
  const size_t nV = vertices.shape(0), nF = faces.shape(0);
  const int32_t* f = faces.data();
  for (size_t i = 0; i < nF * 3; ++i)
    if (f[i] < 0 || static_cast<size_t>(f[i]) >= nV) throw std::invalid_argument("a face names a vertex outside the mesh (indices are 0-based)");
  nxr::operators::EigenResult r;
  {
    nb::gil_scoped_release release;
    nxr::geometry::Manifold m(vertices.data(), static_cast<int>(nV), f, static_cast<int>(nF), intrinsic_delaunay);
    auto ops = nxr::operators::assembleManifoldOperators(m);
    r = nxr::operators::eigen(ops.cotanLaplacian, ops.mass, k, sigma ? *sigma : nxr::operators::kShiftLadder, normalize, remove_dc);
  }
  return nb::make_tuple(hand_over(std::move(r.eigenvalues)), hand_over(std::move(r.eigenvectors)), r.nConverged);
}

template <class F1, class F2, class... Extra>
void def_both(nb::module_& m, const char* name, F1 f64, F2 f32, Extra&&... extra) {
  m.def(name, f64, extra...);   // float64 first: integers and other types convert to it
  m.def(name, f32, extra...);
}

}  // namespace

void bind_compute(nb::module_& m) {
  using nb::arg;
  def_both(m, "signal_psd", &welch_psd<double>, &welch_psd<float>, arg("x"), arg("sfreq"), arg("segment_samples"), arg("overlap"), arg("window"),
           arg("periodic"), arg("detrend"), arg("scaling"), arg("good").none());
  def_both(m, "signal_xpsd", &welch_cross_spectrum<double>, &welch_cross_spectrum<float>, arg("x"), arg("sfreq"), arg("segment_samples"),
           arg("overlap"), arg("window"), arg("periodic"), arg("detrend"), arg("scaling"), arg("bins"), arg("mean"), arg("quadrature"), arg("good").none());
  def_both(m, "signal_filter", &filter_apply<double>, &filter_apply<float>, arg("kind"), arg("x"), arg("sfreq"), arg("lower"), arg("upper"), arg("frequencies"),
           arg("method").none(), arg("ripple"), arg("transition_bandwidth").none(), arg("edge"), arg("transient_energy"), arg("transient_samples").none(),
           arg("remove_mean"), arg("notch_bandwidth").none(), arg("bandstop_order"));
  def_both(m, "stats_pca", &pca_fit<double>, &pca_fit<float>, arg("x"), arg("scale"));
  def_both(m, "stats_reconstruct", &pca_reconstruct<double>, &pca_reconstruct<float>, arg("scores"), arg("components"), arg("mean"), arg("scale"), arg("k"));
  m.def("operators_eigen", &manifold_eigenmodes, arg("vertices"), arg("faces"), arg("k"), arg("sigma").none(), arg("normalize"), arg("remove_dc"),
        arg("intrinsic_delaunay"));
}

}  // namespace nxrpy
