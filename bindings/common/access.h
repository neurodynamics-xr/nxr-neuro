#pragma once
// What the Python and MATLAB faces share for the event, wavelet and filterbank functions: the library's own settings structs filled from the plain data a face
// hands over (a bank described by names and numbers, a transform's operators, a list of kernels), and the call of the library function on the face's layout
// ([channels x samples] for a signal, as the other bindings have it). No file, no maths of its own: every line either converts a setting to the library's type
// or calls the library. A face converts its arrays, dicts and structs to these types and the results back.
#include <array>
#include <limits>
#include <complex>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/graphtransforms.h"
#include "nxr/signal/graphwavelet.h"
#include "nxr/signal/jointfilterbank.h"
#include "nxr/signal/jointwavelet.h"
#include "nxr/signal/timefilterbank.h"
#include "nxr/signal/wavelet.h"

namespace nxr::bindings {

namespace tfb = nxr::signal::timefilterbank;
namespace gfb = nxr::signal::graphfilterbank;
namespace jfb = nxr::signal::jointfilterbank;
namespace gt = nxr::signal::graphtransforms;

using Blocks = std::vector<Eigen::MatrixXcd>;

[[noreturn]] inline void bad_name(const char* what, const std::string& got, const char* allowed) {
  throw std::invalid_argument(std::string(what) + " '" + got + "' is not one of " + allowed);
}

inline gfb::Synthesis parse_synthesis(const std::string& s) {
  if (s == "dual") return gfb::Synthesis::Dual;
  if (s == "tight") return gfb::Synthesis::Tight;
  bad_name("synthesis", s, "dual, tight");
}

/** A series that is real or complex (an analytic bank's input is real, a two-sided bank's may be complex). */
struct Series {
  bool is_complex = false;
  Eigen::MatrixXd real;
  Eigen::MatrixXcd complex;
  Eigen::Index rows() const { return is_complex ? complex.rows() : real.rows(); }
  Eigen::Index cols() const { return is_complex ? complex.cols() : real.cols(); }
  Eigen::MatrixXcd as_complex() const { return is_complex ? complex : Eigen::MatrixXcd(real.cast<std::complex<double>>()); }
};

// ── the settings of a bank, described in data ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** A time bank: the family and sides by name, and the Design's fields; a field not given keeps the library's default. */
struct TimeBankSettings {
  std::string family, sides = "analytic";
  double gain = 1.0;
  std::optional<std::string> edge_bins;                       // "single" | "drop"
  std::optional<int> voices, min_bins;
  std::optional<double> anchor, half_width, cycles, gamma, time_bandwidth;
  std::optional<std::pair<double, double>> limits;
  std::vector<double> edges, centres;
  std::vector<std::pair<double, double>> bands;
};

inline tfb::Design time_design(const TimeBankSettings& s) {
  if (s.family.empty()) throw std::invalid_argument("a time bank needs a family: itersine, meyer, morlet, morse, amor, bump or power_partition");
  tfb::Design d(tfb::familyOf(s.family), s.gain, tfb::sidesOf(s.sides));
  if (s.edge_bins) {
    if (*s.edge_bins == "single") d.edgeBins = tfb::EdgeBins::Single;
    else if (*s.edge_bins == "drop") d.edgeBins = tfb::EdgeBins::Drop;
    else bad_name("edge_bins", *s.edge_bins, "single, drop");
  }
  if (s.voices) d.voices = *s.voices;
  if (s.min_bins) d.minBins = *s.min_bins;
  if (s.anchor) d.anchor = *s.anchor;
  if (s.half_width) d.halfWidth = *s.half_width;
  if (s.cycles) d.cycles = *s.cycles;
  if (s.gamma) d.gamma = *s.gamma;
  if (s.time_bandwidth) d.timeBandwidth = *s.time_bandwidth;
  d.limits = s.limits;
  d.edges = s.edges;
  d.centres = s.centres;
  d.bands = s.bands;
  return d;
}

/** The bank on a grid of n samples at fs (the caller's window: nothing is cut or padded beyond what the design says). */
inline tfb::TimeFilterbank time_bank(const TimeBankSettings& s, std::size_t n, double fs) { return tfb::design(time_design(s), n, fs); }

/** A graph bank: the family by name and the NamedBank's settings; a setting not given keeps the library's default. */
struct GraphBankSettings {
  std::string family;
  std::optional<double> lmax, lmin, lp_factor, area, half_width;
  std::optional<int> num_filters, voices, levels;
  std::optional<std::string> fine_limit;                      // "basis" | "usable"
  std::optional<std::pair<double, double>> scale_limits;
  std::vector<double> edges;
};

/** The bank, designed on the range the settings give: lmin and lmax, or lmax, or the spectrum handed in (the exact eigenvalues), or the operator's lmax. */
inline gfb::GraphFilterbank graph_bank(const GraphBankSettings& s, const std::optional<Eigen::VectorXd>& spectrum = std::nullopt,
                                       std::optional<double> operator_lmax = std::nullopt) {
  if (s.family.empty()) throw std::invalid_argument("a graph bank needs a family: mexhat, heat, itersine, logitersine, meyer, weyl or rank");
  gfb::NamedBank n;
  n.family = s.family;
  if (s.lmax && s.lmin) n.range = gfb::Range::fromLimits(*s.lmin, *s.lmax);
  else if (s.lmax) n.range = gfb::Range::fromLmax(*s.lmax);
  else if (spectrum && spectrum->size() > 0) n.range = gfb::Range::fromSpectrum(*spectrum);
  else if (operator_lmax && *operator_lmax > 0) n.range = gfb::Range::fromLmax(*operator_lmax);
  if (s.num_filters) n.options.numFilters = *s.num_filters;
  if (s.voices) n.options.voices = *s.voices;
  if (s.lp_factor) n.options.lpFactor = *s.lp_factor;
  if (s.fine_limit) {
    if (*s.fine_limit == "basis") n.options.fineLimit = gfb::FineLimit::Basis;
    else if (*s.fine_limit == "usable") n.options.fineLimit = gfb::FineLimit::Usable;
    else bad_name("fine_limit", *s.fine_limit, "basis, usable");
  }
  n.options.scaleLimits = s.scale_limits;
  n.edges = s.edges;
  if (s.area) n.area = *s.area;
  if (s.levels) n.levels = *s.levels;
  if (s.half_width) n.halfWidth = *s.half_width;
  return gfb::make(n);
}

/** A joint kernel: "speed" (c, d) or "ridge" (v, width, sense). */
struct KernelSettings {
  std::string type;
  std::optional<double> c, d, v, width;
  std::optional<int> sense;
};

inline jfb::Kernel kernel(const KernelSettings& k) {
  const auto need = [&](const std::optional<double>& v, const char* name) {
    if (!v) throw std::invalid_argument("a " + k.type + " kernel needs '" + name + "'");
    return *v;
  };
  if (k.type == "speed") return jfb::speedkernel(need(k.c, "c"), need(k.d, "d"));
  if (k.type == "ridge") {
    if (!k.sense) throw std::invalid_argument("a ridge kernel needs 'sense'");
    return jfb::ridgeKernel(need(k.v, "v"), need(k.width, "width"), *k.sense);
  }
  bad_name("kernel type", k.type, "speed, ridge");
}

/** What a field is transformed through: the operator's modes (eigenvalues, with eigenvectors for a field on vertices) or a Chebyshev series in (laplacian, mass). */
struct TransformSettings {
  std::optional<Eigen::VectorXd> eigenvalues;
  std::optional<Eigen::MatrixXd> eigenvectors;
  std::optional<Eigen::SparseMatrix<double>> mass, laplacian;
  int order = 40;
  double lmax = 0;                                            // the series' bound; 0: estimated
  bool eigen() const { return eigenvectors.has_value(); }
  bool series() const { return !eigenvectors && laplacian.has_value(); }
  bool modes() const { return !eigenvectors && !laplacian && eigenvalues.has_value(); }
};

inline gt::EigenTransform eigen_transform(const TransformSettings& t, Eigen::Index rows) {
  if (!t.eigenvalues) throw std::invalid_argument("eigenvectors come with their eigenvalues");
  if (t.eigenvectors->cols() != t.eigenvalues->size()) throw std::invalid_argument("eigenvectors hold one column per eigenvalue");
  if (t.eigenvectors->rows() != rows) throw std::invalid_argument("eigenvectors hold one row per vertex of the field");
  gt::EigenTransform e;
  e.phi = *t.eigenvectors;
  e.lambda = *t.eigenvalues;
  if (t.mass) e.mass = *t.mass;
  return e;
}

inline gt::ChebyshevOperator chebyshev_operator(const TransformSettings& t, Eigen::Index rows) {
  if (t.laplacian->rows() != rows || t.laplacian->cols() != rows) throw std::invalid_argument("the laplacian is [vertices x vertices], one row per vertex of the field");
  return gt::ChebyshevOperator(*t.laplacian, t.mass ? *t.mass : Eigen::SparseMatrix<double>(), t.lmax, t.order);
}

[[noreturn]] inline void no_route() {
  throw std::invalid_argument("give the transform: eigenvalues and eigenvectors (the operator's modes), or a laplacian (with its mass) for the Chebyshev series");
}

// ── the band functions ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

inline tfb::Family band_shape(const std::string& s) {
  if (s != "morlet" && s != "meyer" && s != "power_partition") bad_name("shape", s, "morlet, meyer, power_partition");
  return tfb::familyOf(s);
}

/** x [channels x samples] -> one [channels x samples] complex block per band. */
inline Blocks band_analytic(const Eigen::MatrixXd& x, double fs, const std::vector<std::pair<double, double>>& bands, const std::string& shape, double half_width) {
  const Eigen::MatrixXd xt = x.transpose();
  Blocks out;
  for (const Eigen::MatrixXcd& b : nxr::signal::bandAnalytic(xt, fs, bands, band_shape(shape), half_width)) out.push_back(b.transpose());
  return out;
}
inline std::vector<Eigen::MatrixXd> band_envelope(const Eigen::MatrixXd& x, double fs, const std::vector<std::pair<double, double>>& bands, const std::string& shape,
                                                  double half_width) {
  const Eigen::MatrixXd xt = x.transpose();
  std::vector<Eigen::MatrixXd> out;
  for (const Eigen::MatrixXd& b : nxr::signal::bandEnvelope(xt, fs, bands, band_shape(shape), half_width)) out.push_back(b.transpose());
  return out;
}

struct TimeBankReport {
  Eigen::VectorXd frequencies, centres;
  Eigen::MatrixXd gains;                                      // [members x bins]
  double A = 0, B = 0;
};
inline TimeBankReport time_bank_report(const TimeBankSettings& s, double fs, std::size_t n) {
  const tfb::TimeFilterbank bank = time_bank(s, n, fs);
  TimeBankReport r;
  r.frequencies = tfb::bins(bank);
  r.gains = tfb::gains(bank).transpose();
  r.centres = tfb::centerFrequencies(bank);
  const gfb::Bounds b = tfb::framebounds(bank);
  r.A = b.A;
  r.B = b.B;
  return r;
}

/** wt of x [channels x samples] on a bank designed on its own n samples -> one [channels x n] block per member. */
inline Blocks wt(const Series& x, double fs, const TimeBankSettings& s) {
  const tfb::TimeFilterbank bank = time_bank(s, static_cast<std::size_t>(x.cols()), fs);
  tfb::Coefficients w;
  if (bank.sides == tfb::Sides::Analytic) {
    if (x.is_complex) throw std::invalid_argument("an analytic bank takes a real signal; use sides=\"two_sided\" for a complex one");
    w = nxr::signal::wt(bank, Eigen::MatrixXd(x.real.transpose()));
  } else {
    w = nxr::signal::wt(bank, Eigen::MatrixXcd(x.as_complex().transpose()));
  }
  Blocks out;
  for (const Eigen::MatrixXcd& b : w) out.push_back(b.transpose());
  return out;
}

/** iwt of blocks [channels x n] (n = the samples the bank was designed on) -> the record [channels x n]: real for an analytic bank, complex for a two-sided one. */
inline Series iwt(const Blocks& W, double fs, std::size_t n, const TimeBankSettings& s, const std::string& synthesis) {
  const tfb::TimeFilterbank bank = time_bank(s, n, fs);
  const gfb::Synthesis mode = parse_synthesis(synthesis);
  tfb::Coefficients w;
  for (const Eigen::MatrixXcd& b : W) {
    if (static_cast<std::size_t>(b.cols()) != n) throw std::invalid_argument("each member's coefficients hold n samples (the bank is designed on n)");
    w.push_back(b.transpose());
  }
  if (static_cast<int>(w.size()) != bank.size()) throw std::invalid_argument("the coefficients hold one block per member of the bank");
  Series r;
  if (bank.sides == tfb::Sides::Analytic) r.real = nxr::signal::iwt(bank, w, n, mode).transpose();
  else { r.is_complex = true; r.complex = nxr::signal::iwtComplex(bank, w, n, mode).transpose(); }
  return r;
}

// ── the graph bank and its transforms ─────────────────────────────────────────────────────────────────────────────────────────────────────────

struct GraphBankReport {
  std::optional<Eigen::MatrixXd> gains;                       // [members x K] at the eigenvalues, when given
  std::vector<std::string> roles;
  Eigen::VectorXd scales, centres;                            // wavenumbers
  double A = 0, B = 0;
};
inline const char* role_name(gfb::Role r) {
  switch (r) {
    case gfb::Role::Scaling: return "scaling";
    case gfb::Role::Wavelet: return "wavelet";
    case gfb::Role::Top: return "top";
    default: return "member";
  }
}
inline GraphBankReport graph_bank_report(const GraphBankSettings& s, const std::optional<Eigen::VectorXd>& eigenvalues) {
  const gfb::GraphFilterbank bank = graph_bank(s, eigenvalues);
  GraphBankReport r;
  for (const gfb::Member& m : bank.members) r.roles.push_back(role_name(m.role));
  r.scales = gfb::scales(bank);
  // a bank with no spectral range (meyer, weyl, rank: edges, not a range) has no centre wavenumbers: NaN, as `scales` is NaN where a family has none
  r.centres = bank.lmax > 0 ? gfb::centerWavenumbers(bank) : Eigen::VectorXd::Constant(static_cast<Eigen::Index>(bank.members.size()), std::numeric_limits<double>::quiet_NaN());
  gfb::Bounds b;
  if (eigenvalues) {
    r.gains = gfb::gains(bank, *eigenvalues).transpose();
    b = gfb::framebounds(bank, *eigenvalues);
  } else if (bank.lmax > 0) {
    b = gfb::framebounds(bank);
  } else {                                                    // no range and no spectrum (meyer, weyl, rank): the bounds are of a spectrum, so NaN until one is given
    b.A = b.B = std::numeric_limits<double>::quiet_NaN();
  }
  r.A = b.A;
  r.B = b.B;
  return r;
}

inline std::optional<double> operator_bound(const gt::ChebyshevOperator& op) { return op.lmax(); }

/** gwt of F [n x T] -> one [n x T] block per member. */
inline std::vector<Eigen::MatrixXd> gwt(const Eigen::MatrixXd& F, const GraphBankSettings& s, const TransformSettings& t) {
  if (t.eigen()) {
    const gt::EigenTransform e = eigen_transform(t, F.rows());
    return nxr::signal::gwt(graph_bank(s, e.lambda), e, F);
  }
  if (t.series()) {
    const gt::ChebyshevOperator op = chebyshev_operator(t, F.rows());
    return nxr::signal::gwt(graph_bank(s, std::nullopt, operator_bound(op)), op, F);
  }
  no_route();
}
inline Eigen::MatrixXd igwt(const std::vector<Eigen::MatrixXd>& W, const GraphBankSettings& s, const TransformSettings& t, const std::string& synthesis) {
  if (W.empty()) throw std::invalid_argument("the coefficients hold no member");
  const gfb::Synthesis mode = parse_synthesis(synthesis);
  if (t.eigen()) {
    const gt::EigenTransform e = eigen_transform(t, W[0].rows());
    return nxr::signal::igwt(graph_bank(s, e.lambda), e, W, mode);
  }
  if (t.series()) {
    const gt::ChebyshevOperator op = chebyshev_operator(t, W[0].rows());
    return nxr::signal::igwt(graph_bank(s, std::nullopt, operator_bound(op)), op, W, mode);
  }
  no_route();
}

// ── the joint bank and its transforms ─────────────────────────────────────────────────────────────────────────────────────────────────────────

struct JointResult {
  Blocks W;                                                   // [rows x samples] per member
  std::vector<std::array<std::size_t, 3>> index;              // (g, t, k) 0-based per member, g fastest
};

inline jfb::JointFilterbank joint_bank(const gfb::GraphFilterbank& graph, const TimeBankSettings& time, std::size_t n, double fs, const std::vector<KernelSettings>& kernels) {
  std::vector<jfb::Kernel> ks;
  for (const KernelSettings& k : kernels) ks.push_back(kernel(k));
  return jfb::make(graph, time_bank(time, n, fs), std::move(ks));
}

/** jwt of F: a [modes x T] series of mode coefficients (eigenvalues only), or a field [vertices x T] (eigenvalues and eigenvectors, or a laplacian). The time bank is designed on F's T samples. */
inline JointResult jwt(const Series& F, double fs, const GraphBankSettings& g, const TimeBankSettings& tb, const TransformSettings& t, const std::vector<KernelSettings>& kernels) {
  const std::size_t n = static_cast<std::size_t>(F.cols());
  const auto run = [&](const jfb::JointFilterbank& bank, const auto& domain) {
    const bool two = bank.time.sides == tfb::Sides::TwoSided;
    return (F.is_complex || two) ? nxr::signal::jwt(bank, domain, F.as_complex()) : nxr::signal::jwt(bank, domain, F.real);
  };
  JointResult r;
  const auto take = [&](const jfb::JointFilterbank& bank, nxr::signal::JointCoefficients&& w) {
    r.W = std::move(w);
    for (const jfb::Index& i : bank.index) r.index.push_back({i.g, i.t, i.k});
  };
  if (t.eigen()) {
    const gt::EigenTransform e = eigen_transform(t, F.rows());
    const jfb::JointFilterbank bank = joint_bank(graph_bank(g, e.lambda), tb, n, fs, kernels);
    take(bank, run(bank, e));
  } else if (t.series()) {
    const gt::ChebyshevOperator op = chebyshev_operator(t, F.rows());
    const jfb::JointFilterbank bank = joint_bank(graph_bank(g, std::nullopt, operator_bound(op)), tb, n, fs, kernels);
    take(bank, run(bank, op));
  } else if (t.modes()) {
    if (F.rows() != t.eigenvalues->size()) throw std::invalid_argument("a mode series holds one row per eigenvalue");
    const jfb::JointFilterbank bank = joint_bank(graph_bank(g, *t.eigenvalues), tb, n, fs, kernels);
    take(bank, run(bank, *t.eigenvalues));
  } else {
    no_route();
  }
  return r;
}

/** ijwt: the series back, [rows x n]: real for an analytic time bank, complex for a two-sided one. */
inline Series ijwt(const Blocks& W, double fs, std::size_t n, const GraphBankSettings& g, const TimeBankSettings& tb, const TransformSettings& t,
                   const std::vector<KernelSettings>& kernels, const std::string& synthesis) {
  if (W.empty()) throw std::invalid_argument("the coefficients hold no member");
  const gfb::Synthesis mode = parse_synthesis(synthesis);
  for (const Eigen::MatrixXcd& b : W)
    if (static_cast<std::size_t>(b.cols()) != n) throw std::invalid_argument("each member's coefficients hold n samples (the time bank is designed on n)");
  const auto run = [&](const jfb::JointFilterbank& bank, const auto& domain) {
    if (static_cast<int>(W.size()) != bank.size()) throw std::invalid_argument("the coefficients hold one block per member of the joint bank");
    Series r;
    if (bank.time.sides == tfb::Sides::Analytic) r.real = nxr::signal::ijwt(bank, domain, W, n, mode);
    else { r.is_complex = true; r.complex = nxr::signal::ijwtComplex(bank, domain, W, n, mode); }
    return r;
  };
  if (t.eigen()) {
    const gt::EigenTransform e = eigen_transform(t, W[0].rows());
    return run(joint_bank(graph_bank(g, e.lambda), tb, n, fs, kernels), e);
  }
  if (t.series()) {
    const gt::ChebyshevOperator op = chebyshev_operator(t, W[0].rows());
    return run(joint_bank(graph_bank(g, std::nullopt, operator_bound(op)), tb, n, fs, kernels), op);
  }
  if (t.modes()) {
    if (W[0].rows() != t.eigenvalues->size()) throw std::invalid_argument("a mode series holds one row per eigenvalue");
    return run(joint_bank(graph_bank(g, *t.eigenvalues), tb, n, fs, kernels), *t.eigenvalues);
  }
  no_route();
}

}  // namespace nxr::bindings
