// The temporal events and the wavelet and filterbank functions of the MATLAB face: `compute.select.*` and `compute.signal.{band_analytic, band_envelope, time_bank, wt, iwt,
// graph_bank, gwt, igwt, jwt, ijwt}`. Each command is one call into nxr_compute on arrays (through bindings/common/access.h, which the Python face shares); the maths is the
// library's. Options arrive complete and positional: the defaults live in the MATLAB functions (+nxr/+compute). A bank is a struct of the library's design fields, a field not
// given keeping the library's default. Spans are a struct of columns: start and peak 1-based, stop the last sample (inclusive), value; the library's are 0-based, stop exclusive,
// and the conversion is made here, once.
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "access.h"
#include "first_slice_internal.h"
#include "nxr/select/events.h"
#include "nxr/select/risefall.h"

namespace nxr::mex::access {

namespace {

using namespace nxr::mex::slice;
namespace nb_ = nxr::bindings;
namespace sel = nxr::select;
using cd = std::complex<double>;

void need(int nrhs, int n, const char* cmd) {
  if (nrhs != n + 1) throw std::invalid_argument(std::string("nxr_compute('") + cmd + "') takes " + std::to_string(n) + " arguments (use the nxr package functions)");
}

// ---- structs ----

const mxArray* field(const mxArray* s, const char* name) {
  const mxArray* f = mxGetField(s, 0, name);
  return (f && !mxIsEmpty(f)) ? f : nullptr;
}
void check_struct(const mxArray* s, const char* what) {
  if (!mxIsStruct(s) || mxGetNumberOfElements(s) != 1) throw std::invalid_argument(std::string(what) + " must be a scalar struct");
}
void only(const mxArray* s, std::initializer_list<const char*> valid, const char* what, const char* names) {
  check_struct(s, what);
  for (int i = 0; i < mxGetNumberOfFields(s); ++i) {
    const std::string k = mxGetFieldNameByNumber(s, i);
    bool ok = false;
    for (const char* v : valid) ok = ok || k == v;
    if (!ok) throw std::invalid_argument(std::string(what) + " has no setting '" + k + "': the settings are " + names);
  }
}
void take(const mxArray* s, const char* name, std::string& into) { if (const mxArray* f = field(s, name)) into = get_string(f, name); }
void take(const mxArray* s, const char* name, std::optional<std::string>& into) { if (const mxArray* f = field(s, name)) into = get_string(f, name); }
void take(const mxArray* s, const char* name, double& into) { if (const mxArray* f = field(s, name)) into = get_scalar(f, name); }
void take(const mxArray* s, const char* name, std::optional<double>& into) { if (const mxArray* f = field(s, name)) into = get_scalar(f, name); }
void take(const mxArray* s, const char* name, std::optional<int>& into) {
  if (const mxArray* f = field(s, name)) {
    const double v = get_scalar(f, name);
    if (v != std::floor(v)) throw std::invalid_argument(std::string(name) + " must be an integer");
    into = static_cast<int>(v);
  }
}
void take(const mxArray* s, const char* name, std::optional<std::pair<double, double>>& into) {
  if (const mxArray* f = field(s, name)) {
    const Eigen::VectorXd v = get_vector(f, name);
    if (v.size() != 2) throw std::invalid_argument(std::string(name) + " is a pair: low and high");
    into = std::make_pair(v[0], v[1]);
  }
}
void take(const mxArray* s, const char* name, std::vector<double>& into) {
  if (const mxArray* f = field(s, name)) {
    const Eigen::VectorXd v = get_vector(f, name);
    into.assign(v.data(), v.data() + v.size());
  }
}
std::vector<std::pair<double, double>> pairs_of(const mxArray* a, const char* name) {
  const Matrix m = get_matrix(a, name);
  if (m.m.cols() != 2) throw std::invalid_argument(std::string(name) + " is [bands x 2]: the low and high edge of each, Hz");
  std::vector<std::pair<double, double>> out;
  for (Eigen::Index i = 0; i < m.m.rows(); ++i) out.emplace_back(m.m(i, 0), m.m(i, 1));
  return out;
}

nb_::TimeBankSettings time_settings(const mxArray* s) {
  only(s, {"family", "sides", "gain", "edge_bins", "voices", "min_bins", "anchor", "half_width", "cycles", "gamma", "time_bandwidth", "limits", "edges", "centres", "bands"},
       "the time bank", "family, sides, gain, edge_bins, voices, min_bins, anchor, half_width, cycles, gamma, time_bandwidth, limits, edges, centres, bands");
  nb_::TimeBankSettings t;
  take(s, "family", t.family); take(s, "sides", t.sides); take(s, "gain", t.gain); take(s, "edge_bins", t.edge_bins);
  take(s, "voices", t.voices); take(s, "min_bins", t.min_bins); take(s, "anchor", t.anchor); take(s, "half_width", t.half_width);
  take(s, "cycles", t.cycles); take(s, "gamma", t.gamma); take(s, "time_bandwidth", t.time_bandwidth); take(s, "limits", t.limits);
  take(s, "edges", t.edges); take(s, "centres", t.centres);
  if (const mxArray* b = field(s, "bands")) t.bands = pairs_of(b, "bands");
  return t;
}

nb_::GraphBankSettings graph_settings(const mxArray* s) {
  only(s, {"family", "lmax", "lmin", "num_filters", "voices", "lp_factor", "fine_limit", "scale_limits", "edges", "area", "levels", "half_width"}, "the graph bank",
       "family, lmax, lmin, num_filters, voices, lp_factor, fine_limit, scale_limits, edges, area, levels, half_width");
  nb_::GraphBankSettings g;
  take(s, "family", g.family); take(s, "lmax", g.lmax); take(s, "lmin", g.lmin); take(s, "num_filters", g.num_filters); take(s, "voices", g.voices);
  take(s, "lp_factor", g.lp_factor); take(s, "fine_limit", g.fine_limit); take(s, "scale_limits", g.scale_limits); take(s, "edges", g.edges);
  take(s, "area", g.area); take(s, "levels", g.levels); take(s, "half_width", g.half_width);
  return g;
}

std::vector<nb_::KernelSettings> kernel_settings(const mxArray* a) {
  std::vector<nb_::KernelSettings> out;
  if (is_unset(a)) return out;
  if (!mxIsCell(a)) throw std::invalid_argument("kernels is a cell array of structs");
  for (mwSize i = 0; i < mxGetNumberOfElements(a); ++i) {
    const mxArray* s = mxGetCell(a, i);
    only(s, {"type", "c", "d", "v", "width", "sense"}, "a kernel", "type, c, d, v, width, sense");
    nb_::KernelSettings k;
    take(s, "type", k.type); take(s, "c", k.c); take(s, "d", k.d); take(s, "v", k.v); take(s, "width", k.width); take(s, "sense", k.sense);
    out.push_back(k);
  }
  return out;
}

Eigen::SparseMatrix<double> get_sparse(const mxArray* a, const char* name) {
  std::vector<Eigen::Triplet<double>> trips;
  if (mxIsSparse(a)) {
    if (!mxIsDouble(a) || mxIsComplex(a)) throw std::invalid_argument(std::string(name) + " must be a real double sparse matrix");
    const mwIndex* jc = mxGetJc(a);
    const mwIndex* ir = mxGetIr(a);
    const double* pr = mxGetDoubles(a);
    const mwSize cols = mxGetN(a);
    for (mwSize c = 0; c < cols; ++c)
      for (mwIndex k = jc[c]; k < jc[c + 1]; ++k) trips.emplace_back(static_cast<int>(ir[k]), static_cast<int>(c), pr[k]);
    Eigen::SparseMatrix<double> m(static_cast<int>(mxGetM(a)), static_cast<int>(cols));
    m.setFromTriplets(trips.begin(), trips.end());
    return m;
  }
  const Matrix d = get_matrix(a, name);
  if (d.m.rows() == 1 || d.m.cols() == 1) {   // a vector is the diagonal
    const int n = static_cast<int>(d.m.size());
    for (int i = 0; i < n; ++i) trips.emplace_back(i, i, d.m.data()[i]);
    Eigen::SparseMatrix<double> m(n, n);
    m.setFromTriplets(trips.begin(), trips.end());
    return m;
  }
  return d.m.sparseView();
}

nb_::TransformSettings transform_settings(const mxArray* eigenvalues, const mxArray* eigenvectors, const mxArray* mass, const mxArray* laplacian, const mxArray* order,
                                          const mxArray* lmax) {
  nb_::TransformSettings t;
  if (!is_unset(eigenvalues)) t.eigenvalues = get_vector(eigenvalues, "eigenvalues");
  if (!is_unset(eigenvectors)) t.eigenvectors = get_matrix(eigenvectors, "eigenvectors").m;
  if (!is_unset(mass)) t.mass = get_sparse(mass, "mass");
  if (!is_unset(laplacian)) t.laplacian = get_sparse(laplacian, "laplacian");
  t.order = static_cast<int>(get_scalar(order, "order"));
  t.lmax = get_scalar(lmax, "lmax");
  return t;
}

// ---- arrays ----

mxArray* complex_array(mwSize m, mwSize r, mwSize c, bool is3d = true) {
  const mwSize dims[3] = {m, r, c};
  return is3d ? mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX) : mxCreateUninitNumericMatrix(r, c, mxDOUBLE_CLASS, mxCOMPLEX);
}

/** [members x rows x cols], member varying fastest, complex double. */
mxArray* give_complex_stack(const nb_::Blocks& b) {
  const mwSize m = b.size(), r = m ? b[0].rows() : 0, c = m ? b[0].cols() : 0;
  mxArray* a = complex_array(m, r, c);
  mxComplexDouble* p = mxGetComplexDoubles(a);
  for (mwSize i = 0; i < m; ++i)
    for (mwSize k = 0; k < c; ++k)
      for (mwSize j = 0; j < r; ++j) {
        const cd v = b[i](static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(k));
        p[i + m * (j + r * k)] = {v.real(), v.imag()};
      }
  return a;
}

mxArray* give_real_stack(const std::vector<Eigen::MatrixXd>& b, bool single) {
  const mwSize m = b.size(), r = m ? b[0].rows() : 0, c = m ? b[0].cols() : 0;
  const mwSize dims[3] = {m, r, c};
  mxArray* a = mxCreateNumericArray(3, dims, single ? mxSINGLE_CLASS : mxDOUBLE_CLASS, mxREAL);
  for (mwSize i = 0; i < m; ++i)
    for (mwSize k = 0; k < c; ++k)
      for (mwSize j = 0; j < r; ++j) {
        const double v = b[i](static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(k));
        const mwSize at = i + m * (j + r * k);
        if (single) static_cast<float*>(mxGetData(a))[at] = static_cast<float>(v);
        else static_cast<double*>(mxGetData(a))[at] = v;
      }
  return a;
}

mxArray* give_series(const nb_::Series& s) {
  if (!s.is_complex) return give(s.real, false);
  mxArray* a = mxCreateUninitNumericMatrix(s.complex.rows(), s.complex.cols(), mxDOUBLE_CLASS, mxCOMPLEX);
  mxComplexDouble* p = mxGetComplexDoubles(a);
  for (Eigen::Index k = 0; k < s.complex.cols(); ++k)
    for (Eigen::Index j = 0; j < s.complex.rows(); ++j) p[j + s.complex.rows() * k] = {s.complex(j, k).real(), s.complex(j, k).imag()};
  return a;
}

/** A real or complex 2-D array as a series. */
nb_::Series get_series(const mxArray* a, const char* name) {
  nb_::Series s;
  if (!mxIsComplex(a)) { s.real = get_matrix(a, name).m; return s; }
  if ((!mxIsDouble(a) && !mxIsSingle(a)) || mxIsSparse(a) || mxGetNumberOfDimensions(a) != 2) throw std::invalid_argument(std::string(name) + " must be a dense 2-D array");
  s.is_complex = true;
  const Eigen::Index r = static_cast<Eigen::Index>(mxGetM(a)), c = static_cast<Eigen::Index>(mxGetN(a));
  s.complex.resize(r, c);
  if (mxIsDouble(a)) {
    const mxComplexDouble* p = mxGetComplexDoubles(a);
    for (Eigen::Index i = 0; i < r * c; ++i) s.complex.data()[i] = cd(p[i].real, p[i].imag);
  } else {
    const mxComplexSingle* p = mxGetComplexSingles(a);
    for (Eigen::Index i = 0; i < r * c; ++i) s.complex.data()[i] = cd(p[i].real, p[i].imag);
  }
  return s;
}

/** A 3-D array [members x rows x cols] (real or complex, double) as one block per member. */
nb_::Blocks get_blocks(const mxArray* a, const char* name) {
  if (!mxIsDouble(a) || mxIsSparse(a) || mxGetNumberOfDimensions(a) > 3) throw std::invalid_argument(std::string(name) + " must be a dense double array [members x rows x samples]");
  const mwSize* d = mxGetDimensions(a);
  const mwSize m = d[0], r = d[1], c = mxGetNumberOfDimensions(a) == 3 ? d[2] : 1;
  nb_::Blocks out(m, Eigen::MatrixXcd(r, c));
  const mwSize n = m * r * c;
  for (mwSize at = 0; at < n; ++at) {
    const mwSize i = at % m, j = (at / m) % r, k = at / (m * r);
    out[i](static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(k)) = mxIsComplex(a) ? cd(mxGetComplexDoubles(a)[at].real, mxGetComplexDoubles(a)[at].imag) : cd(mxGetDoubles(a)[at], 0.0);
  }
  return out;
}
std::vector<Eigen::MatrixXd> real_blocks(const mxArray* a, const char* name) {
  std::vector<Eigen::MatrixXd> out;
  for (const Eigen::MatrixXcd& b : get_blocks(a, name)) out.push_back(b.real());
  return out;
}

// ---- spans ----

mxArray* column(const std::vector<double>& v) {
  mxArray* a = mxCreateDoubleMatrix(v.size(), 1, mxREAL);
  if (!v.empty()) std::memcpy(mxGetDoubles(a), v.data(), v.size() * sizeof(double));
  return a;
}
mxArray* spans_out(const std::vector<sel::Span>& spans) {
  std::vector<double> start, stop, peak, value;
  for (const sel::Span& s : spans) { start.push_back(static_cast<double>(s.start) + 1); stop.push_back(static_cast<double>(s.stop)); peak.push_back(static_cast<double>(s.peak) + 1); value.push_back(s.value); }
  return make_struct({{"start", column(start)}, {"stop", column(stop)}, {"peak", column(peak)}, {"value", column(value)}});
}
std::vector<sel::Span> spans_of(const mxArray* s) {
  check_struct(s, "spans");
  const char* names[4] = {"start", "stop", "peak", "value"};
  Eigen::VectorXd v[4];
  for (int i = 0; i < 4; ++i) {
    const mxArray* f = mxGetField(s, 0, names[i]);
    if (!f) throw std::invalid_argument("spans is a struct with start, stop, peak and value");
    v[i] = get_vector(f, names[i]);
  }
  if (v[1].size() != v[0].size() || v[2].size() != v[0].size() || v[3].size() != v[0].size()) throw std::invalid_argument("start, stop, peak and value hold one entry per span");
  std::vector<sel::Span> out(static_cast<size_t>(v[0].size()));
  for (size_t i = 0; i < out.size(); ++i) {
    out[i].start = static_cast<std::int64_t>(v[0][i]) - 1;
    out[i].stop = static_cast<std::int64_t>(v[1][i]);
    out[i].peak = static_cast<std::int64_t>(v[2][i]) - 1;
    out[i].value = v[3][i];
  }
  return out;
}

mxArray* pair_row(double a, double b) {
  mxArray* r = mxCreateDoubleMatrix(1, 2, mxREAL);
  mxGetDoubles(r)[0] = a;
  mxGetDoubles(r)[1] = b;
  return r;
}

}  // namespace

// ---- select ----

void selectMedian(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 1, "compute.select.median");
  const Eigen::VectorXd x = get_vector(prhs[1], "x");
  plhs[0] = give_scalar(sel::median(x));
}
void selectMad(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 1, "compute.select.mad");
  const Eigen::VectorXd x = get_vector(prhs[1], "x");
  plhs[0] = give_scalar(sel::mad(x));
}
// spans = compute.select.crossings(x, entry, exit)
void selectCrossings(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 3, "compute.select.crossings");
  const Eigen::VectorXd x = get_vector(prhs[1], "x");
  plhs[0] = spans_out(sel::crossings(x, get_scalar(prhs[2], "entry"), get_scalar(prhs[3], "exit")));
}
// spans = compute.select.merge_spans(spans, gap)
void selectMergeSpans(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 2, "compute.select.merge_spans");
  plhs[0] = spans_out(sel::mergeSpans(spans_of(prhs[1]), static_cast<std::int64_t>(get_scalar(prhs[2], "gap"))));
}
// spans = compute.select.keep_spans(spans, min_length, max_length); max_length [] or Inf: no upper bound
void selectKeepSpans(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 3, "compute.select.keep_spans");
  const double mx = is_unset(prhs[3]) ? INFINITY : get_scalar(prhs[3], "max_length");
  plhs[0] = spans_out(sel::keepSpans(spans_of(prhs[1]), static_cast<std::int64_t>(get_scalar(prhs[2], "min_length")),
                                     std::isinf(mx) ? std::numeric_limits<std::int64_t>::max() : static_cast<std::int64_t>(mx)));
}
// r = compute.select.detect_bursts(amplitude, step, enter_sd, exit_sd, merge_s, min_s, max_s, baseline); baseline [] or [median sd]
void selectDetectBursts(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 8, "compute.select.detect_bursts");
  const Eigen::VectorXd x = get_vector(prhs[1], "amplitude");
  sel::BurstOptions o;
  o.enterSd = get_scalar(prhs[3], "enter_sd"); o.exitSd = get_scalar(prhs[4], "exit_sd"); o.mergeS = get_scalar(prhs[5], "merge_s");
  o.minS = get_scalar(prhs[6], "min_s"); o.maxS = get_scalar(prhs[7], "max_s");
  std::optional<sel::Baseline> b;
  if (!is_unset(prhs[8])) {
    const Eigen::VectorXd v = get_vector(prhs[8], "baseline");
    if (v.size() != 2) throw std::invalid_argument("baseline is [median sd]");
    b = sel::Baseline{v[0], v[1]};
  }
  const sel::Bursts r = sel::detectBursts(x, get_scalar(prhs[2], "step"), o, b);
  plhs[0] = make_struct({{"spans", spans_out(r.spans)}, {"median", give_scalar(r.median)}, {"sd", give_scalar(r.sd)}, {"enter", give_scalar(r.enter)}, {"exit", give_scalar(r.exit)},
                         {"good", give_scalar(static_cast<double>(r.good))}, {"gap", give_scalar(static_cast<double>(r.gap))},
                         {"min_length", give_scalar(static_cast<double>(r.minLength))}, {"max_length", give_scalar(static_cast<double>(r.maxLength))},
                         {"skipped", give_string(r.skipped)}});
}
// r = compute.select.risefall(amplitude, sfreq, t0, t1, model, margin, smooth, tau_max, min_r2); min_r2 [] : kept is []
void selectRisefall(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 9, "compute.select.risefall");
  const Eigen::VectorXd x = get_vector(prhs[1], "amplitude");
  sel::Options o;
  o.model = sel::parseModel(get_string(prhs[5], "model"));
  o.margin = get_scalar(prhs[6], "margin"); o.smooth = get_scalar(prhs[7], "smooth"); o.tauMax = get_scalar(prhs[8], "tau_max");
  const sel::Fit f = sel::fit(x, get_scalar(prhs[2], "sfreq"), get_scalar(prhs[3], "t0"), get_scalar(prhs[4], "t1"), o);
  mxArray* kept = is_unset(prhs[9]) ? mxCreateDoubleMatrix(0, 0, mxREAL) : give_bool(sel::passesGate(f, get_scalar(prhs[9], "min_r2")));
  plhs[0] = make_struct({{"t_on", give_scalar(f.tOn)}, {"t_off", give_scalar(f.tOff)}, {"tau_rise", give_scalar(f.tauR)}, {"tau_fall", give_scalar(f.tauF)},
                         {"baseline", give_scalar(f.b)}, {"height", give_scalar(f.h)}, {"r2", give_scalar(f.r2)}, {"ok", give_bool(f.ok)}, {"at_bound", give_bool(f.atBound)},
                         {"buildup", pair_row(f.buildup[0], f.buildup[1])}, {"plateau", pair_row(f.plateau[0], f.plateau[1])}, {"decay", pair_row(f.decay[0], f.decay[1])},
                         {"has_plateau", give_bool(f.hasPlateau)}, {"kept", kept}});
}
// r = compute.select.stages(t_on, t_off, tau_rise, tau_fall)
void selectStages(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 4, "compute.select.stages");
  const sel::Stages s = sel::stagesOf(get_scalar(prhs[1], "t_on"), get_scalar(prhs[2], "t_off"), get_scalar(prhs[3], "tau_rise"), get_scalar(prhs[4], "tau_fall"));
  plhs[0] = make_struct({{"onset", pair_row(s.onset[0], s.onset[1])}, {"within", pair_row(s.within[0], s.within[1])}, {"offset", pair_row(s.offset[0], s.offset[1])},
                         {"has_within", give_bool(s.hasWithin)}, {"valid", give_bool(s.valid)}});
}

// ---- signal ----

// z = compute.signal.band_analytic(x, sfreq, bands, shape, half_width): complex [bands x channels x samples]
void signalBandAnalytic(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 5, "compute.signal.band_analytic");
  const Matrix x = get_matrix(prhs[1], "x");
  plhs[0] = give_complex_stack(nb_::band_analytic(x.m, get_scalar(prhs[2], "sfreq"), pairs_of(prhs[3], "bands"), get_string(prhs[4], "shape"), get_scalar(prhs[5], "half_width")));
}
// e = compute.signal.band_envelope(x, sfreq, bands, shape, half_width): real [bands x channels x samples], in x's precision
void signalBandEnvelope(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 5, "compute.signal.band_envelope");
  const Matrix x = get_matrix(prhs[1], "x");
  plhs[0] = give_real_stack(nb_::band_envelope(x.m, get_scalar(prhs[2], "sfreq"), pairs_of(prhs[3], "bands"), get_string(prhs[4], "shape"), get_scalar(prhs[5], "half_width")), x.single);
}
// [frequencies, gains, centre_frequencies, A, B] = compute.signal.time_bank(sfreq, n, bank)
void signalTimeBank(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 3, "compute.signal.time_bank");
  const double n = get_scalar(prhs[2], "n");
  if (n < 1 || n != std::floor(n)) throw std::invalid_argument("n must be a positive integer");
  const nb_::TimeBankReport r = nb_::time_bank_report(time_settings(prhs[3]), get_scalar(prhs[1], "sfreq"), static_cast<std::size_t>(n));
  plhs[0] = give_vector(r.frequencies, false, true);
  plhs[1] = give(r.gains, false);
  plhs[2] = give_vector(r.centres, false, true);
  plhs[3] = give_scalar(r.A);
  plhs[4] = give_scalar(r.B);
}
// W = compute.signal.wt(x, sfreq, bank): complex [members x channels x samples]
void signalWt(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 3, "compute.signal.wt");
  plhs[0] = give_complex_stack(nb_::wt(get_series(prhs[1], "x"), get_scalar(prhs[2], "sfreq"), time_settings(prhs[3])));
}
// x = compute.signal.iwt(W, sfreq, n, bank, synthesis): [channels x n], real for an analytic bank, complex for a two-sided one
void signalIwt(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 5, "compute.signal.iwt");
  const double n = get_scalar(prhs[3], "n");
  if (n < 1 || n != std::floor(n)) throw std::invalid_argument("n must be a positive integer");
  plhs[0] = give_series(nb_::iwt(get_blocks(prhs[1], "W"), get_scalar(prhs[2], "sfreq"), static_cast<std::size_t>(n), time_settings(prhs[4]), get_string(prhs[5], "synthesis")));
}
// [gains, roles, scales, centre_wavenumbers, A, B] = compute.signal.graph_bank(bank, eigenvalues); gains [] without eigenvalues
void signalGraphBank(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 2, "compute.signal.graph_bank");
  std::optional<Eigen::VectorXd> ev;
  if (!is_unset(prhs[2])) ev = get_vector(prhs[2], "eigenvalues");
  const nb_::GraphBankReport r = nb_::graph_bank_report(graph_settings(prhs[1]), ev);
  plhs[0] = r.gains ? give(*r.gains, false) : mxCreateDoubleMatrix(0, 0, mxREAL);
  plhs[1] = give_cellstr(r.roles);
  plhs[2] = give_vector(r.scales, false, true);
  plhs[3] = give_vector(r.centres, false, true);
  plhs[4] = give_scalar(r.A);
  plhs[5] = give_scalar(r.B);
}
// W = compute.signal.gwt(F, bank, eigenvalues, eigenvectors, mass, laplacian, order, lmax): real [members x vertices x samples]
void signalGwt(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 8, "compute.signal.gwt");
  const Matrix f = get_matrix(prhs[1], "F");
  plhs[0] = give_real_stack(nb_::gwt(f.m, graph_settings(prhs[2]), transform_settings(prhs[3], prhs[4], prhs[5], prhs[6], prhs[7], prhs[8])), false);
}
// F = compute.signal.igwt(W, bank, eigenvalues, eigenvectors, mass, laplacian, order, lmax, synthesis): [vertices x samples]
void signalIgwt(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 9, "compute.signal.igwt");
  plhs[0] = give(nb_::igwt(real_blocks(prhs[1], "W"), graph_settings(prhs[2]), transform_settings(prhs[3], prhs[4], prhs[5], prhs[6], prhs[7], prhs[8]), get_string(prhs[9], "synthesis")), false);
}
// [W, index] = compute.signal.jwt(F, sfreq, graph_bank, time_bank, eigenvalues, eigenvectors, mass, laplacian, order, lmax, kernels): index [members x 3] (g, t, k; 1-based)
void signalJwt(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 11, "compute.signal.jwt");
  const nb_::JointResult r = nb_::jwt(get_series(prhs[1], "F"), get_scalar(prhs[2], "sfreq"), graph_settings(prhs[3]), time_settings(prhs[4]),
                                     transform_settings(prhs[5], prhs[6], prhs[7], prhs[8], prhs[9], prhs[10]), kernel_settings(prhs[11]));
  plhs[0] = give_complex_stack(r.W);
  Eigen::MatrixXd index(static_cast<Eigen::Index>(r.index.size()), 3);
  for (size_t i = 0; i < r.index.size(); ++i)
    for (int j = 0; j < 3; ++j) index(static_cast<Eigen::Index>(i), j) = static_cast<double>(r.index[i][static_cast<size_t>(j)]) + 1;
  plhs[1] = give(index, false);
}
// x = compute.signal.ijwt(W, sfreq, n, graph_bank, time_bank, eigenvalues, eigenvectors, mass, laplacian, order, lmax, kernels, synthesis): [rows x n]
void signalIjwt(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  need(nrhs, 13, "compute.signal.ijwt");
  const double n = get_scalar(prhs[3], "n");
  if (n < 1 || n != std::floor(n)) throw std::invalid_argument("n must be a positive integer");
  plhs[0] = give_series(nb_::ijwt(get_blocks(prhs[1], "W"), get_scalar(prhs[2], "sfreq"), static_cast<std::size_t>(n), graph_settings(prhs[4]), time_settings(prhs[5]),
                                  transform_settings(prhs[6], prhs[7], prhs[8], prhs[9], prhs[10], prhs[11]), kernel_settings(prhs[12]), get_string(prhs[13], "synthesis")));
}

}  // namespace nxr::mex::access
