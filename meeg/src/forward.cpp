#include "nxr/meeg/forward.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nxr::meeg {

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {
constexpr double kPi = 3.14159265358979323846;

void checkCoils(const Coils& c, bool needOrientation) {
  const Eigen::Index P = c.position.rows();
  if (c.position.cols() != 3 || c.weight.size() != P || static_cast<Eigen::Index>(c.channel.size()) != P || c.channels < 1)
    throw std::invalid_argument("coils: positions [P x 3], weights [P], channels [P] and a channel count are needed");
  if (needOrientation && (c.orientation.rows() != P || c.orientation.cols() != 3)) throw std::invalid_argument("coils: orientations [P x 3] are needed for MEG");
  if (!c.position.allFinite() || !c.weight.allFinite()) throw std::invalid_argument("coils: a NaN in the positions or weights");
  if (needOrientation && !c.orientation.allFinite()) throw std::invalid_argument("coils: a sensor point has no orientation (NaN)");
  for (int ch : c.channel)
    if (ch < 0 || ch >= c.channels) throw std::invalid_argument("coils: a point belongs to a channel outside 0..channels-1");
}
void checkSources(const MatrixXd& s) {
  if (s.cols() != 3) throw std::invalid_argument("sources must be [N x 3]");
  if (!s.allFinite()) throw std::invalid_argument("a NaN in the sources");
}
}  // namespace

// ---- MEG ----------------------------------------------------------------------------------------------------------------------------------------------------

MatrixXd sarvasPointGain(const MatrixXd& sources, const Vector3d& position, const Vector3d& orientation, const Vector3d& centre, const SarvasOptions& opt) {
  checkSources(sources);
  const Vector3d r = position - centre;
  const double rn = r.norm();
  const double scale = opt.permeability / (4 * kPi);
  MatrixXd out(sources.rows(), 3);
  for (Eigen::Index k = 0; k < sources.rows(); ++k) {
    const Vector3d r0 = sources.row(k).transpose() - centre;
    const Vector3d a = r - r0;
    const double an = a.norm();
    const double F = an * (rn * an + rn * rn - r0.dot(r));
    if (!(F > 0)) throw std::invalid_argument("sarvasPointGain: a source coincides with the sensor point");
    const double ar = a.dot(r);
    const Vector3d gradF = (an * an / rn + ar / an + 2 * an + 2 * rn) * r - (an + 2 * rn + ar / an) * r0;
    // n . B = scale / F^2 [ F n.(Q x r0) - (Q x r0 . r)(grad F . n) ] and n.(Q x r0) = Q.(r0 x n),  (Q x r0).r = Q.(r0 x r)
    out.row(k) = (scale / (F * F) * (F * r0.cross(orientation) - gradF.dot(orientation) * r0.cross(r))).transpose();
  }
  return out;
}

namespace {
MatrixXd sarvasWith(const MatrixXd& sources, const Coils& coils, const std::vector<Vector3d>& centres, bool perChannel, const SarvasOptions& opt) {
  checkSources(sources);
  checkCoils(coils, true);
  if (perChannel && static_cast<int>(centres.size()) != coils.channels) throw std::invalid_argument("sarvasGain: one sphere centre per channel is needed");
  MatrixXd G = MatrixXd::Zero(coils.channels, 3 * sources.rows());
  for (Eigen::Index p = 0; p < coils.position.rows(); ++p) {
    const int ch = coils.channel[static_cast<std::size_t>(p)];
    const Vector3d& c = perChannel ? centres[static_cast<std::size_t>(ch)] : centres[0];
    const MatrixXd g = sarvasPointGain(sources, coils.position.row(p).transpose(), coils.orientation.row(p).transpose(), c, opt);
    for (Eigen::Index k = 0; k < sources.rows(); ++k) G.block<1, 3>(ch, 3 * k) += coils.weight(p) * g.row(k);
  }
  return G;
}
}  // namespace

MatrixXd sarvasGain(const MatrixXd& sources, const Coils& coils, const Vector3d& centre, const SarvasOptions& opt) { return sarvasWith(sources, coils, {centre}, false, opt); }
MatrixXd sarvasGain(const MatrixXd& sources, const Coils& coils, const std::vector<Vector3d>& centres, const SarvasOptions& opt) { return sarvasWith(sources, coils, centres, true, opt); }

MatrixXd referenceCompensatedGain(const MatrixXd& megGain, const MatrixXd& coefficients, const MatrixXd& referenceGain) {
  if (coefficients.rows() != megGain.rows() || coefficients.cols() != referenceGain.rows() || referenceGain.cols() != megGain.cols())
    throw std::invalid_argument("referenceCompensatedGain: sizes disagree (G_MEG [M x 3N], C [M x R], G_REF [R x 3N])");
  return megGain - coefficients * referenceGain;
}

// ---- EEG ----------------------------------------------------------------------------------------------------------------------------------------------------

namespace {
void checkLayers(const VectorXd& radius, const VectorXd& sigma) {
  if (radius.size() < 1 || radius.size() != sigma.size()) throw std::invalid_argument("layers: one radius and one conductivity per shell are needed");
  for (Eigen::Index i = 0; i < radius.size(); ++i) {
    if (!(radius(i) > 0) || !(sigma(i) > 0)) throw std::invalid_argument("layers: radii and conductivities must be positive");
    if (i > 0 && !(radius(i) > radius(i - 1))) throw std::invalid_argument("layers: radii must increase from the innermost to the outermost shell");
  }
}
}  // namespace

VectorXd multiShellCoefficients(const VectorXd& radius, const VectorXd& conductivity, int terms) {
  checkLayers(radius, conductivity);
  if (terms < 1) throw std::invalid_argument("multiShellCoefficients: terms must be at least 1");
  const int L = static_cast<int>(radius.size());
  VectorXd f = VectorXd::Ones(terms);
  if (L == 1) return f;
  for (int n = 1; n <= terms; ++n) {
    Eigen::Matrix2d M = Eigen::Matrix2d::Identity();
    const double n1 = n + 1.0;
    for (int k = L - 2; k >= 0; --k) {
      const double c1 = conductivity(k) / conductivity(k + 1), c2 = c1 - 1;
      const double cr = std::pow(radius(k) / radius(L - 1), 2.0 * n + 1);
      Eigen::Matrix2d A;
      A << n + n1 * c1, n1 * c2 / cr, n * c2 * cr, n1 + n * c1;
      M = A * M;
    }
    f(n - 1) = n * std::pow(2.0 * n + 1, L - 1) / (n * M(1, 1) + n1 * M(1, 0));
  }
  return f;
}

namespace {
// For the eccentricities mu: the magnitudes lambda by linear least squares under sum lambda = f_1, and the weighted squared residual.
double bergMagnitudes(const VectorXd& mu, const VectorXd& f, const VectorXd& w, VectorXd& lambda) {
  const int J = static_cast<int>(mu.size()), T = static_cast<int>(f.size());
  VectorXd y(T - 1);
  MatrixXd M(T - 1, J - 1);
  for (int n = 2; n <= T; ++n) {
    const double m0 = std::pow(mu(0), n - 1);
    y(n - 2) = w(n - 1) * (f(n - 1) - m0 * f(0));
    for (int j = 1; j < J; ++j) M(n - 2, j - 1) = w(n - 1) * (std::pow(mu(j), n - 1) - m0);
  }
  VectorXd l = VectorXd::Zero(J - 1);
  if (J > 1) l = M.completeOrthogonalDecomposition().solve(y);
  lambda.resize(J);
  lambda.tail(J - 1) = l;
  lambda(0) = f(0) - l.sum();
  const VectorXd r = y - (J > 1 ? VectorXd(M * l) : VectorXd::Zero(T - 1));
  return r.squaredNorm();
}
}  // namespace

BergScherg fitBergScherg(const VectorXd& radius, const VectorXd& conductivity, const BergFitOptions& opt) {
  checkLayers(radius, conductivity);
  if (opt.dipoles < 1 || opt.terms < 3 || opt.starts < 1) throw std::invalid_argument("fitBergScherg: dipoles >= 1, terms >= 3 and starts >= 1 are needed");
  const int J = opt.dipoles, T = opt.terms;
  const VectorXd f = multiShellCoefficients(radius, conductivity, T);
  const double q = radius(0) / radius(radius.size() - 1);
  VectorXd w(T);
  for (int n = 1; n <= T; ++n) w(n - 1) = std::sqrt((2.0 * n + 1) * (3.0 * n + 1) / n) * std::pow(q, n - 1);
  auto cost = [&](const VectorXd& mu) {
    if (mu.cwiseAbs().maxCoeff() >= 1) return 1e3;
    VectorXd lambda;
    return bergMagnitudes(mu, f, w, lambda);
  };
  MinimiseResult best;
  best.value = 1e300;
  for (int s = 0; s < opt.starts; ++s) {
    VectorXd x0(J);
    for (int j = 0; j < J; ++j) {
      const double u = std::fmod(0.6180339887498949 * (s * J + j + 1), 1.0);
      x0(j) = -0.5 + 1.45 * u;
    }
    const MinimiseResult r = nelderMead(cost, x0, opt.minimiser);
    if (r.value < best.value) best = r;
  }
  VectorXd mu = best.x, lambda;
  const double rss = bergMagnitudes(mu, f, w, lambda);
  std::vector<int> order(static_cast<std::size_t>(J));
  for (int j = 0; j < J; ++j) order[static_cast<std::size_t>(j)] = j;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return mu(a) > mu(b); });
  BergScherg out;
  out.eccentricity.resize(J);
  out.magnitude.resize(J);
  for (int j = 0; j < J; ++j) { out.eccentricity(j) = mu(order[static_cast<std::size_t>(j)]); out.magnitude(j) = lambda(order[static_cast<std::size_t>(j)]); }
  // the relative residual is over n >= 2 (n = 1 is exact by the constraint)
  double fw = 0;
  for (int n = 2; n <= T; ++n) fw += w(n - 1) * w(n - 1) * f(n - 1) * f(n - 1);
  out.residual = std::sqrt(rss / fw);
  return out;
}

bool recordedBergScherg(const VectorXd& radius, const VectorXd& conductivity, BergScherg& out) {
  auto is = [&](std::initializer_list<double> r, std::initializer_list<double> c) {
    if (radius.size() != static_cast<Eigen::Index>(r.size()) || conductivity.size() != radius.size()) return false;
    Eigen::Index i = 0;
    for (double v : r) { if (std::abs(radius(i) / radius(radius.size() - 1) - v) > 1e-12) return false; ++i; }
    i = 0;
    for (double v : c) { if (std::abs(conductivity(i) / c.begin()[0] - v / c.begin()[0]) > 1e-12) return false; ++i; }
    return true;
  };
  // Brainstorm, layers (0.88, 0.93, 1) x R and (0.33, 0.0042, 0.33) S/m: the output of bst_berg(R, sigma) (nmax 200, 3 dipoles), Brainstorm 3, November 2026.
  if (is({0.88, 0.93, 1.0}, {0.33, 0.0042, 0.33})) {
    out.eccentricity = (VectorXd(3) << -0.14449562967270013, 0.5659916459705662, 0.9434179691066816).finished();
    out.magnitude = (VectorXd(3) << 0.013511003384285458, 0.5356722778041332, 0.15119749656746304).finished();
    out.residual = 0;
    return true;
  }
  // MNE-Python 1.13, make_sphere_model(relative_radii=(0.90, 0.92, 0.97, 1.0), sigmas=(0.33, 1.0, 0.004, 0.33)): mu and lambda x sigma_L of the model (any head radius).
  if (is({0.90, 0.92, 0.97, 1.0}, {0.33, 1.0, 0.004, 0.33})) {
    out.eccentricity = (VectorXd(3) << 0.9432112879770956, 0.6631031404159992, 0.11394344322830123).finished();
    out.magnitude = (VectorXd(3) << 0.14096400576239754, 0.6884444499005578, -0.019038852994738415).finished();
    out.residual = 0;
    return true;
  }
  return false;
}

EegSphereOptions mneEegSphereOptions() {
  EegSphereOptions o;
  o.radius = (VectorXd(4) << 0.90, 0.92, 0.97, 1.0).finished();
  o.conductivity = (VectorXd(4) << 0.33, 1.0, 0.004, 0.33).finished();
  return o;
}

namespace {
// The gradient (coefficient of Q) of the potential at electrode e (from the centre) of a dipole at rd in a homogeneous sphere, per unit 1/sigma, times 4 pi: (m1 rd + m2 e) / |rd|^2.
Vector3d homogeneousPotentialGradient(const Vector3d& rd, const Vector3d& e, double centreTolerance) {
  const double r2 = e.squaredNorm(), r = std::sqrt(r2);
  const double rd2 = rd.squaredNorm();
  if (rd2 <= centreTolerance * centreTolerance * r2) return 3 * e / (r2 * r);        // the limit at the centre
  const Vector3d av = e - rd;
  const double a = av.norm();
  const double a3 = 2 / (a * a * a);
  const double rrd = rd.dot(e);
  const double ra = r2 - rrd, rda = rrd - rd2;
  const double F = a * (r * a + ra);
  const double c1 = a3 * rda + 1 / a - 1 / r;
  const double c2 = a3 + (a + r) / (r * F);
  const double m1 = c1 - c2 * rrd, m2 = c2 * rd2;
  return (m1 * rd + m2 * e) / rd2;
}
// The same by the series: sum_n f_n (2n+1)/n (rho/a)^(n-1) [ n P_n(c) q^ + P_n'(c) (e^ - c q^) ] / (4 pi sigma a^2) (without the 4 pi sigma): Q . g.
Vector3d seriesGradient(const Vector3d& rq, const Vector3d& e, const VectorXd& f) {
  const double a = e.norm(), rho = rq.norm();
  const Vector3d eh = e / a;
  if (rho <= 1e-12 * a) return 3 * f(0) * eh / (a * a);
  const Vector3d qh = rq / rho;
  const double c = qh.dot(eh);
  double Pm = 1, P = c, dPm = 0, dP = 1;                // P_0, P_1 and their derivatives
  Vector3d g = Vector3d::Zero();
  const int N = static_cast<int>(f.size());
  double pw = 1;                                         // (rho/a)^(n-1)
  for (int n = 1; n <= N; ++n) {
    g += f(n - 1) * (2.0 * n + 1) / n * pw * (n * P * qh + dP * (eh - c * qh));
    const double Pn = ((2.0 * n + 1) * c * P - n * Pm) / (n + 1);
    const double dPn = (n + 1) * P + c * dP;
    Pm = P; P = Pn; dPm = dP; dP = dPn;
    pw *= rho / a;
  }
  (void)dPm;
  return g / (a * a);
}
}  // namespace

MatrixXd eegSphereGain(const MatrixXd& sources, const Coils& electrodes, const Vector3d& centre, const EegSphereOptions& opt, BergScherg* usedParameters) {
  checkSources(sources);
  Coils e = electrodes;
  if (e.orientation.rows() == 0) e.orientation = MatrixXd::Zero(e.position.rows(), 3);
  checkCoils(e, false);
  checkLayers(opt.radius, opt.conductivity);
  const double sigmaOuter = opt.conductivity(opt.conductivity.size() - 1);
  BergScherg berg;
  VectorXd f;
  if (opt.approximation == EegApproximation::BergScherg) {
    if (opt.useGivenBerg) berg = opt.given;
    else if (!(opt.useRecordedBerg && recordedBergScherg(opt.radius, opt.conductivity, berg))) berg = fitBergScherg(opt.radius, opt.conductivity, opt.berg);
    if (berg.eccentricity.size() < 1 || berg.eccentricity.size() != berg.magnitude.size()) throw std::invalid_argument("eegSphereGain: Berg-Scherg parameters need one magnitude per eccentricity");
    if (usedParameters) *usedParameters = berg;
  } else {
    f = multiShellCoefficients(opt.radius, opt.conductivity, opt.seriesTerms);
  }
  MatrixXd G = MatrixXd::Zero(e.channels, 3 * sources.rows());
  for (Eigen::Index p = 0; p < e.position.rows(); ++p) {
    const int ch = e.channel[static_cast<std::size_t>(p)];
    const Vector3d ep = e.position.row(p).transpose() - centre;
    for (Eigen::Index k = 0; k < sources.rows(); ++k) {
      const Vector3d r0 = sources.row(k).transpose() - centre;
      Vector3d g = Vector3d::Zero();
      if (opt.approximation == EegApproximation::BergScherg) {
        for (Eigen::Index j = 0; j < berg.eccentricity.size(); ++j) g += berg.magnitude(j) * homogeneousPotentialGradient(berg.eccentricity(j) * r0, ep, opt.centreTolerance);
      } else {
        g = seriesGradient(r0, ep, f);
      }
      G.block<1, 3>(ch, 3 * k) += e.weight(p) * (g / (4 * kPi * sigmaOuter)).transpose();
    }
  }
  return G;
}

}  // namespace nxr::meeg
