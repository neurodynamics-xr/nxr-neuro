// nxr_neuro._core, the temporal events and the wavelet and filterbank functions: median, mad, crossings, merge_spans, keep_spans, detect_bursts, risefall and stages
// (`nxr/select`), and the band, time, graph and joint transforms (`nxr/signal`: band_analytic, band_envelope, time_bank, wt, iwt, graph_bank, gwt, igwt, jwt, ijwt).
// Each is one call into nxr_compute on arrays; the maths is the library's. A bank arrives as a dict of the library's design fields (bindings/common/access.h turns it
// into the library's settings), spans as four arrays. Complex results are complex128; real ones are float64, except band_envelope, which follows the input's precision.
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "access.h"
#include "arrays.h"
#include "nxr/select/events.h"
#include "nxr/select/risefall.h"

namespace nxrpy {

namespace {

namespace sel = nxr::select;
namespace nb_ = nxr::bindings;

using cd = std::complex<double>;
using Array3c = nb::ndarray<nb::numpy, const cd, nb::ndim<3>, nb::c_contig>;
using Index64 = Array1<std::int64_t>;

// ---- arrays ----

Eigen::MatrixXd real_matrix(const Array2<double>& a) { return to_matrix<double>(a); }

/** A [rows, cols] array of float64 or complex128 as a series. */
nb_::Series series(const nb::ndarray<nb::numpy, nb::ndim<2>, nb::c_contig>& a) {
  nb_::Series s;
  const size_t r = a.shape(0), c = a.shape(1);
  const nb::dlpack::dtype dt = a.dtype();
  if (dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float) && dt.bits == 64) {
    s.real = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<const double*>(a.data()), r, c);
  } else if (dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Complex) && dt.bits == 128) {
    s.is_complex = true;
    s.complex = Eigen::Map<const Eigen::Matrix<cd, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(static_cast<const cd*>(a.data()), r, c);
  } else {
    throw std::invalid_argument("the array is float64 or complex128");
  }
  return s;
}

nb_::Blocks blocks(const Array3c& w) {
  nb_::Blocks out;
  const size_t m = w.shape(0), r = w.shape(1), c = w.shape(2);
  for (size_t i = 0; i < m; ++i)
    out.push_back(Eigen::Map<const Eigen::Matrix<cd, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(w.data() + i * r * c, r, c));
  return out;
}

std::vector<Eigen::MatrixXd> real_blocks(const Array3c& w) {
  std::vector<Eigen::MatrixXd> out;
  for (const Eigen::MatrixXcd& b : blocks(w)) out.push_back(b.real());
  return out;
}

/** One C-ordered [members, rows, cols] complex128 array from equally sized blocks. */
nb::object complex_stack(const nb_::Blocks& b) {
  const size_t m = b.size(), r = m ? b[0].rows() : 0, c = m ? b[0].cols() : 0;
  std::vector<cd> v(m * r * c);
  for (size_t i = 0; i < m; ++i)
    Eigen::Map<Eigen::Matrix<cd, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(v.data() + i * r * c, r, c) = b[i];
  return own_array<cd>(std::move(v), {m, r, c});
}

nb::object real_stack(const std::vector<Eigen::MatrixXd>& b) {
  const size_t m = b.size(), r = m ? b[0].rows() : 0, c = m ? b[0].cols() : 0;
  std::vector<double> v(m * r * c);
  for (size_t i = 0; i < m; ++i)
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(v.data() + i * r * c, r, c) = b[i];
  return own_array<double>(std::move(v), {m, r, c});
}

nb::object give_series(nb_::Series&& s) {
  if (!s.is_complex) return hand_over(std::move(s.real));
  const size_t r = s.complex.rows(), c = s.complex.cols();
  std::vector<cd> v(r * c);
  Eigen::Map<Eigen::Matrix<cd, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(v.data(), r, c) = s.complex;
  return own_array<cd>(std::move(v), {r, c});
}

std::vector<std::pair<double, double>> band_list(const Array2<double>& bands) {
  if (bands.shape(1) != 2) throw std::invalid_argument("bands is [bands, 2]: the low and high edge of each, Hz");
  std::vector<std::pair<double, double>> out;
  for (size_t i = 0; i < bands.shape(0); ++i) out.emplace_back(bands.data()[2 * i], bands.data()[2 * i + 1]);
  return out;
}

// ---- spans ----

std::vector<sel::Span> spans_of(const Index64& start, const Index64& stop, const Index64& peak, const Array1<double>& value) {
  const size_t n = start.shape(0);
  if (stop.shape(0) != n || peak.shape(0) != n || value.shape(0) != n) throw std::invalid_argument("start, stop, peak and value hold one entry per span");
  std::vector<sel::Span> out(n);
  for (size_t i = 0; i < n; ++i) {
    out[i].start = start.data()[i];
    out[i].stop = stop.data()[i];
    out[i].peak = peak.data()[i];
    out[i].value = value.data()[i];
  }
  return out;
}

nb::tuple spans_out(const std::vector<sel::Span>& spans) {
  std::vector<std::int64_t> a(spans.size()), b(spans.size()), c(spans.size());
  std::vector<double> d(spans.size());
  for (size_t i = 0; i < spans.size(); ++i) { a[i] = spans[i].start; b[i] = spans[i].stop; c[i] = spans[i].peak; d[i] = spans[i].value; }
  const std::vector<size_t> shape = {spans.size()};
  return nb::make_tuple(own_array<std::int64_t>(std::move(a), shape), own_array<std::int64_t>(std::move(b), shape), own_array<std::int64_t>(std::move(c), shape),
                        own_array<double>(std::move(d), shape));
}

// ---- banks described in a dict ----

template <class T>
void take(const nb::dict& d, const char* key, T& into) {
  if (d.contains(key)) into = nb::cast<T>(d[key]);
}
template <class T>
void take(const nb::dict& d, const char* key, std::optional<T>& into) {
  if (d.contains(key) && !d[key].is_none()) into = nb::cast<T>(d[key]);
}
void only(const nb::dict& d, std::initializer_list<const char*> valid, const char* what, const char* names) {
  for (auto item : d) {
    const std::string k = nb::cast<std::string>(item.first);
    bool ok = false;
    for (const char* v : valid) ok = ok || k == v;
    if (!ok) throw std::invalid_argument(std::string(what) + " has no setting '" + k + "': the settings are " + names);
  }
}

nb_::TimeBankSettings time_settings(const nb::dict& d) {
  only(d, {"family", "sides", "gain", "edge_bins", "voices", "min_bins", "anchor", "half_width", "cycles", "gamma", "time_bandwidth", "limits", "edges", "centres", "bands"},
       "the time bank", "family, sides, gain, edge_bins, voices, min_bins, anchor, half_width, cycles, gamma, time_bandwidth, limits, edges, centres, bands");
  nb_::TimeBankSettings s;
  take(d, "family", s.family); take(d, "sides", s.sides); take(d, "gain", s.gain); take(d, "edge_bins", s.edge_bins);
  take(d, "voices", s.voices); take(d, "min_bins", s.min_bins); take(d, "anchor", s.anchor); take(d, "half_width", s.half_width);
  take(d, "cycles", s.cycles); take(d, "gamma", s.gamma); take(d, "time_bandwidth", s.time_bandwidth); take(d, "limits", s.limits);
  std::optional<std::vector<double>> edges, centres;
  std::optional<std::vector<std::pair<double, double>>> bands;
  take(d, "edges", edges); take(d, "centres", centres); take(d, "bands", bands);
  if (edges) s.edges = *edges;
  if (centres) s.centres = *centres;
  if (bands) s.bands = *bands;
  return s;
}

nb_::GraphBankSettings graph_settings(const nb::dict& d) {
  only(d, {"family", "lmax", "lmin", "num_filters", "voices", "lp_factor", "fine_limit", "scale_limits", "edges", "area", "levels", "half_width"}, "the graph bank",
       "family, lmax, lmin, num_filters, voices, lp_factor, fine_limit, scale_limits, edges, area, levels, half_width");
  nb_::GraphBankSettings s;
  take(d, "family", s.family); take(d, "lmax", s.lmax); take(d, "lmin", s.lmin); take(d, "num_filters", s.num_filters); take(d, "voices", s.voices);
  take(d, "lp_factor", s.lp_factor); take(d, "fine_limit", s.fine_limit); take(d, "scale_limits", s.scale_limits); take(d, "area", s.area);
  take(d, "levels", s.levels); take(d, "half_width", s.half_width);
  std::optional<std::vector<double>> edges;
  take(d, "edges", edges);
  if (edges) s.edges = *edges;
  return s;
}

std::vector<nb_::KernelSettings> kernel_settings(const nb::list& kernels) {
  std::vector<nb_::KernelSettings> out;
  for (nb::handle h : kernels) {
    const nb::dict d = nb::cast<nb::dict>(h);
    only(d, {"type", "c", "d", "v", "width", "sense"}, "a kernel", "type, c, d, v, width, sense");
    nb_::KernelSettings k;
    take(d, "type", k.type); take(d, "c", k.c); take(d, "d", k.d); take(d, "v", k.v); take(d, "width", k.width); take(d, "sense", k.sense);
    out.push_back(k);
  }
  return out;
}

Eigen::SparseMatrix<double> sparse_of(const nb::tuple& t) {
  // (row, col, value, n_rows, n_cols): the matrix's entries; the wrapper makes them from a scipy sparse matrix or a diagonal
  const auto rows = nb::cast<Index64>(t[0]);
  const auto cols = nb::cast<Index64>(t[1]);
  const auto vals = nb::cast<Array1<double>>(t[2]);
  std::vector<Eigen::Triplet<double>> trips;
  for (size_t i = 0; i < rows.shape(0); ++i) trips.emplace_back(static_cast<int>(rows.data()[i]), static_cast<int>(cols.data()[i]), vals.data()[i]);
  Eigen::SparseMatrix<double> m(nb::cast<int>(t[3]), nb::cast<int>(t[4]));
  m.setFromTriplets(trips.begin(), trips.end());
  return m;
}

nb_::TransformSettings transform_settings(const std::optional<Array1<double>>& eigenvalues, const std::optional<Array2<double>>& eigenvectors,
                                          const std::optional<nb::tuple>& mass, const std::optional<nb::tuple>& laplacian, int order, double lmax) {
  nb_::TransformSettings t;
  if (eigenvalues) t.eigenvalues = to_vector<double>(*eigenvalues);
  if (eigenvectors) t.eigenvectors = to_matrix<double>(*eigenvectors);
  if (mass) t.mass = sparse_of(*mass);
  if (laplacian) t.laplacian = sparse_of(*laplacian);
  t.order = order;
  t.lmax = lmax;
  return t;
}

// ---- select ----

double select_median(const Array1<double>& x) {
  const Eigen::VectorXd v = to_vector<double>(x);
  return sel::median(v);
}
double select_mad(const Array1<double>& x) {
  const Eigen::VectorXd v = to_vector<double>(x);
  return sel::mad(v);
}
nb::tuple select_crossings(const Array1<double>& x, double entry, double exit) {
  const Eigen::VectorXd v = to_vector<double>(x);
  return spans_out(sel::crossings(v, entry, exit));
}
nb::tuple select_merge_spans(const Index64& start, const Index64& stop, const Index64& peak, const Array1<double>& value, std::int64_t gap) {
  return spans_out(sel::mergeSpans(spans_of(start, stop, peak, value), gap));
}
nb::tuple select_keep_spans(const Index64& start, const Index64& stop, const Index64& peak, const Array1<double>& value, std::int64_t min_length,
                            const std::optional<std::int64_t>& max_length) {
  return spans_out(sel::keepSpans(spans_of(start, stop, peak, value), min_length, max_length ? *max_length : std::numeric_limits<std::int64_t>::max()));
}
nb::tuple select_detect_bursts(const Array1<double>& amplitude, double step, double enter_sd, double exit_sd, double merge_s, double min_s, double max_s,
                               const std::optional<std::pair<double, double>>& baseline) {
  const Eigen::VectorXd v = to_vector<double>(amplitude);
  sel::BurstOptions o;
  o.enterSd = enter_sd; o.exitSd = exit_sd; o.mergeS = merge_s; o.minS = min_s; o.maxS = max_s;
  std::optional<sel::Baseline> b;
  if (baseline) b = sel::Baseline{baseline->first, baseline->second};
  sel::Bursts r;
  {
    nb::gil_scoped_release release;
    r = sel::detectBursts(v, step, o, b);
  }
  return nb::make_tuple(spans_out(r.spans), r.median, r.sd, r.enter, r.exit, r.good, r.gap, r.minLength, r.maxLength, r.skipped);
}
nb::tuple select_risefall(const Array1<double>& amplitude, double sfreq, double t0, double t1, const std::string& model, double margin, double smooth, double tau_max,
                          const std::optional<double>& min_r2) {
  const Eigen::VectorXd v = to_vector<double>(amplitude);
  sel::Options o;
  o.model = sel::parseModel(model);
  o.margin = margin; o.smooth = smooth; o.tauMax = tau_max;
  sel::Fit f;
  {
    nb::gil_scoped_release release;
    f = sel::fit(v, sfreq, t0, t1, o);
  }
  nb::object kept = min_r2 ? nb::cast(sel::passesGate(f, *min_r2)) : nb::none();
  return nb::make_tuple(f.tOn, f.tOff, f.tauR, f.tauF, f.b, f.h, f.r2, f.ok, f.atBound, std::make_pair(f.buildup[0], f.buildup[1]), std::make_pair(f.plateau[0], f.plateau[1]),
                        std::make_pair(f.decay[0], f.decay[1]), f.hasPlateau, kept);
}
nb::tuple select_stages(double t_on, double t_off, double tau_rise, double tau_fall) {
  const sel::Stages s = sel::stagesOf(t_on, t_off, tau_rise, tau_fall);
  return nb::make_tuple(std::make_pair(s.onset[0], s.onset[1]), std::make_pair(s.within[0], s.within[1]), std::make_pair(s.offset[0], s.offset[1]), s.hasWithin, s.valid);
}

// ---- signal ----

nb::object signal_band_analytic(const Array2<double>& x, double sfreq, const Array2<double>& bands, const std::string& shape, double half_width) {
  const Eigen::MatrixXd X = real_matrix(x);
  const auto b = band_list(bands);
  nb_::Blocks out;
  {
    nb::gil_scoped_release release;
    out = nb_::band_analytic(X, sfreq, b, shape, half_width);
  }
  return complex_stack(out);
}
template <class T>
nb::object signal_band_envelope(const Array2<T>& x, double sfreq, const Array2<double>& bands, const std::string& shape, double half_width) {
  const Eigen::MatrixXd X = to_matrix<T>(x);
  const auto b = band_list(bands);
  std::vector<Eigen::MatrixXd> out;
  {
    nb::gil_scoped_release release;
    out = nb_::band_envelope(X, sfreq, b, shape, half_width);
  }
  return give_stack<T>(out, out.empty() ? 0 : out[0].rows(), out.empty() ? 0 : out[0].cols());
}
nb::tuple signal_time_bank(double sfreq, size_t n, const nb::dict& bank) {
  const auto s = time_settings(bank);
  nb_::TimeBankReport r;
  {
    nb::gil_scoped_release release;
    r = nb_::time_bank_report(s, sfreq, n);
  }
  return nb::make_tuple(hand_over(std::move(r.frequencies)), hand_over(std::move(r.gains)), hand_over(std::move(r.centres)), r.A, r.B);
}
nb::object signal_wt(const nb::ndarray<nb::numpy, nb::ndim<2>, nb::c_contig>& x, double sfreq, const nb::dict& bank) {
  const nb_::Series X = series(x);
  const auto s = time_settings(bank);
  nb_::Blocks out;
  {
    nb::gil_scoped_release release;
    out = nb_::wt(X, sfreq, s);
  }
  return complex_stack(out);
}
nb::object signal_iwt(const Array3c& w, double sfreq, size_t n, const nb::dict& bank, const std::string& synthesis) {
  const nb_::Blocks W = blocks(w);
  const auto s = time_settings(bank);
  nb_::Series r;
  {
    nb::gil_scoped_release release;
    r = nb_::iwt(W, sfreq, n, s, synthesis);
  }
  return give_series(std::move(r));
}
nb::tuple signal_graph_bank(const nb::dict& bank, const std::optional<Array1<double>>& eigenvalues) {
  const auto s = graph_settings(bank);
  std::optional<Eigen::VectorXd> ev;
  if (eigenvalues) ev = to_vector<double>(*eigenvalues);
  nb_::GraphBankReport r;
  {
    nb::gil_scoped_release release;
    r = nb_::graph_bank_report(s, ev);
  }
  nb::object gains = r.gains ? hand_over(std::move(*r.gains)) : nb::none();
  return nb::make_tuple(gains, r.roles, hand_over(std::move(r.scales)), hand_over(std::move(r.centres)), r.A, r.B);
}
nb::object signal_gwt(const Array2<double>& F, const nb::dict& bank, const std::optional<Array1<double>>& eigenvalues, const std::optional<Array2<double>>& eigenvectors,
                      const std::optional<nb::tuple>& mass, const std::optional<nb::tuple>& laplacian, int order, double lmax) {
  const Eigen::MatrixXd f = to_matrix<double>(F);
  const auto s = graph_settings(bank);
  const auto t = transform_settings(eigenvalues, eigenvectors, mass, laplacian, order, lmax);
  std::vector<Eigen::MatrixXd> out;
  {
    nb::gil_scoped_release release;
    out = nb_::gwt(f, s, t);
  }
  return real_stack(out);
}
nb::object signal_igwt(const Array3c& w, const nb::dict& bank, const std::optional<Array1<double>>& eigenvalues, const std::optional<Array2<double>>& eigenvectors,
                       const std::optional<nb::tuple>& mass, const std::optional<nb::tuple>& laplacian, int order, double lmax, const std::string& synthesis) {
  const std::vector<Eigen::MatrixXd> W = real_blocks(w);
  const auto s = graph_settings(bank);
  const auto t = transform_settings(eigenvalues, eigenvectors, mass, laplacian, order, lmax);
  Eigen::MatrixXd out;
  {
    nb::gil_scoped_release release;
    out = nb_::igwt(W, s, t, synthesis);
  }
  return hand_over(std::move(out));
}
nb::tuple signal_jwt(const nb::ndarray<nb::numpy, nb::ndim<2>, nb::c_contig>& F, double sfreq, const nb::dict& graph_bank, const nb::dict& time_bank,
                     const std::optional<Array1<double>>& eigenvalues, const std::optional<Array2<double>>& eigenvectors, const std::optional<nb::tuple>& mass,
                     const std::optional<nb::tuple>& laplacian, int order, double lmax, const nb::list& kernels) {
  const nb_::Series f = series(F);
  const auto g = graph_settings(graph_bank);
  const auto tb = time_settings(time_bank);
  const auto t = transform_settings(eigenvalues, eigenvectors, mass, laplacian, order, lmax);
  const auto ks = kernel_settings(kernels);
  nb_::JointResult r;
  {
    nb::gil_scoped_release release;
    r = nb_::jwt(f, sfreq, g, tb, t, ks);
  }
  std::vector<std::int64_t> index;
  for (const auto& i : r.index) index.insert(index.end(), {static_cast<std::int64_t>(i[0]), static_cast<std::int64_t>(i[1]), static_cast<std::int64_t>(i[2])});
  return nb::make_tuple(complex_stack(r.W), own_array<std::int64_t>(std::move(index), {r.index.size(), 3}));
}
nb::object signal_ijwt(const Array3c& w, double sfreq, size_t n, const nb::dict& graph_bank, const nb::dict& time_bank, const std::optional<Array1<double>>& eigenvalues,
                       const std::optional<Array2<double>>& eigenvectors, const std::optional<nb::tuple>& mass, const std::optional<nb::tuple>& laplacian, int order,
                       double lmax, const nb::list& kernels, const std::string& synthesis) {
  const nb_::Blocks W = blocks(w);
  const auto g = graph_settings(graph_bank);
  const auto tb = time_settings(time_bank);
  const auto t = transform_settings(eigenvalues, eigenvectors, mass, laplacian, order, lmax);
  const auto ks = kernel_settings(kernels);
  nb_::Series r;
  {
    nb::gil_scoped_release release;
    r = nb_::ijwt(W, sfreq, n, g, tb, t, ks, synthesis);
  }
  return give_series(std::move(r));
}

}  // namespace

void bind_access(nb::module_& m) {
  using nb::arg;
  m.def("select_median", &select_median, arg("x"));
  m.def("select_mad", &select_mad, arg("x"));
  m.def("select_crossings", &select_crossings, arg("x"), arg("entry"), arg("exit"));
  m.def("select_merge_spans", &select_merge_spans, arg("start"), arg("stop"), arg("peak"), arg("value"), arg("gap"));
  m.def("select_keep_spans", &select_keep_spans, arg("start"), arg("stop"), arg("peak"), arg("value"), arg("min_length"), arg("max_length").none());
  m.def("select_detect_bursts", &select_detect_bursts, arg("amplitude"), arg("step"), arg("enter_sd"), arg("exit_sd"), arg("merge_s"), arg("min_s"), arg("max_s"),
        arg("baseline").none());
  m.def("select_risefall", &select_risefall, arg("amplitude"), arg("sfreq"), arg("t0"), arg("t1"), arg("model"), arg("margin"), arg("smooth"), arg("tau_max"),
        arg("min_r2").none());
  m.def("select_stages", &select_stages, arg("t_on"), arg("t_off"), arg("tau_rise"), arg("tau_fall"));
  m.def("signal_band_analytic", &signal_band_analytic, arg("x"), arg("sfreq"), arg("bands"), arg("shape"), arg("half_width"));
  m.def("signal_band_envelope", &signal_band_envelope<double>, arg("x"), arg("sfreq"), arg("bands"), arg("shape"), arg("half_width"));
  m.def("signal_band_envelope", &signal_band_envelope<float>, arg("x"), arg("sfreq"), arg("bands"), arg("shape"), arg("half_width"));
  m.def("signal_time_bank", &signal_time_bank, arg("sfreq"), arg("n"), arg("bank"));
  m.def("signal_wt", &signal_wt, arg("x"), arg("sfreq"), arg("bank"));
  m.def("signal_iwt", &signal_iwt, arg("w"), arg("sfreq"), arg("n"), arg("bank"), arg("synthesis"));
  m.def("signal_graph_bank", &signal_graph_bank, arg("bank"), arg("eigenvalues").none());
  m.def("signal_gwt", &signal_gwt, arg("f"), arg("bank"), arg("eigenvalues").none(), arg("eigenvectors").none(), arg("mass").none(), arg("laplacian").none(), arg("order"),
        arg("lmax"));
  m.def("signal_igwt", &signal_igwt, arg("w"), arg("bank"), arg("eigenvalues").none(), arg("eigenvectors").none(), arg("mass").none(), arg("laplacian").none(), arg("order"),
        arg("lmax"), arg("synthesis"));
  m.def("signal_jwt", &signal_jwt, arg("f"), arg("sfreq"), arg("graph_bank"), arg("time_bank"), arg("eigenvalues").none(), arg("eigenvectors").none(), arg("mass").none(),
        arg("laplacian").none(), arg("order"), arg("lmax"), arg("kernels"));
  m.def("signal_ijwt", &signal_ijwt, arg("w"), arg("sfreq"), arg("n"), arg("graph_bank"), arg("time_bank"), arg("eigenvalues").none(), arg("eigenvectors").none(),
        arg("mass").none(), arg("laplacian").none(), arg("order"), arg("lmax"), arg("kernels"), arg("synthesis"));
}

}  // namespace nxrpy
