// nxr::signal: the cross-spectral measures, decompositions, complex-signal and phase-gradient functions of the former quadrature.h (a port of nxr.spectral.lagged); conventions in the headers.
#include "nxr/signal/complex_signal.h"
#include "nxr/signal/crossspectral.h"
#include "nxr/signal/crossspectral_decomposition.h"
#include "nxr/signal/phase_gradient.h"
#include "nxr/core/algebra.h"
#include "nxr/geometry/mesh.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
constexpr double kPi = 3.14159265358979323846;
using cd = std::complex<double>;
}  // namespace

namespace nxr::signal {
using Eigen::MatrixXcd; using Eigen::MatrixXd; using Eigen::VectorXcd; using Eigen::VectorXd;

MatrixXd cospectrum(const MatrixXcd& C) { return C.real(); }

MatrixXd quadspectrum(const MatrixXcd& C) { return C.imag(); }

}  // namespace nxr::signal

namespace nxr::signal {
using Eigen::MatrixXcd; using Eigen::MatrixXd; using Eigen::VectorXcd; using Eigen::VectorXd;

double quadratureFractionFromFactor(const MatrixXcd& F) {
    const MatrixXcd GH = F.adjoint() * F;
    const double c2 = GH.squaredNorm();
    if (c2 == 0) return 0;
    const MatrixXcd G = F.transpose() * F;
    const double tr = (G * G.conjugate()).trace().real();
    return std::sqrt(std::max(0.0, (c2 - tr) / 2) / c2);
}

double quadratureFraction(const MatrixXcd& C) {
    const double n = C.norm();
    return n == 0 ? 0 : C.imag().norm() / n;
}

}  // namespace nxr::signal

namespace nxr::signal {
using Eigen::MatrixXcd; using Eigen::MatrixXd; using Eigen::VectorXcd; using Eigen::VectorXd;

namespace {
// the gauge of a pair: rotate (a, b) in their plane so that the row of largest a^2 + b^2 has q = 0 and p > 0 (the 2-form p q^T - q p^T is unchanged)
QuadraturePair gauge(const VectorXd& a, const VectorXd& b, double sigma) {
    Eigen::Index i;
    (a.array().square() + b.array().square()).maxCoeff(&i);
    const double th = std::atan2(b[i], a[i]);
    QuadraturePair r;
    r.p = std::cos(th) * a + std::sin(th) * b;
    r.q = -std::sin(th) * a + std::cos(th) * b;
    r.sigma = sigma;
    return r;
}
}  // namespace

std::vector<QuadraturePair> planes(const core::QuadratureFactor& f, int k) {
    std::vector<QuadraturePair> out;
    if (k < 1 || f.S.size() == 0) return out;
    const MatrixXcd H = f.S.cast<cd>() * cd(0, -1);   // H = -iS, Hermitian: S z = i sigma z  <=>  H z = sigma z
    Eigen::SelfAdjointEigenSolver<MatrixXcd> es(H);
    const VectorXd& w = es.eigenvalues();       // ascending
    for (Eigen::Index i = w.size() - 1; i >= 0 && static_cast<int>(out.size()) < k; --i) {
        if (!(w[i] > 1e-10 * f.reference) || !(w[i] > 0)) break;
        const VectorXcd v = f.Q * es.eigenvectors().col(i);
        const VectorXd re = v.real(), im = v.imag();
        const double nre = re.norm();
        if (nre == 0) continue;
        const VectorXd a = re / nre;
        VectorXd b = im - im.dot(a) * a;
        const double nb = b.norm();
        if (nb == 0) continue;
        out.push_back(gauge(a, b / nb, w[i]));
    }
    return out;
}

QuadraturePair top_plane(const core::QuadratureFactor& f) {
    const auto p = planes(f, 1);
    return p.empty() ? QuadraturePair{} : p[0];
}

std::vector<QuadraturePair> planes(const MatrixXcd& F, int k) { return planes(core::skewFactor(F), k); }

double lagged_share(const core::QuadratureFactor& f, const std::vector<QuadraturePair>& pairs) {
    const double total = f.S.squaredNorm();
    if (pairs.empty() || total == 0) return 0;
    MatrixXd P(f.Q.rows(), 2 * static_cast<Eigen::Index>(pairs.size()));
    for (std::size_t i = 0; i < pairs.size(); ++i) { P.col(2 * i) = pairs[i].p; P.col(2 * i + 1) = pairs[i].q; }
    const MatrixXd B = Eigen::HouseholderQR<MatrixXd>(P).householderQ() * MatrixXd::Identity(P.rows(), P.cols());
    const MatrixXd BQ = B.transpose() * f.Q;
    return (BQ * f.S * BQ.transpose()).squaredNorm() / total;
}

RotaryPower rotaryPower(const MatrixXcd& F, const VectorXd& p, const VectorXd& q) {
    VectorXcd w(p.size());
    for (Eigen::Index i = 0; i < p.size(); ++i) w[i] = cd(p[i], q[i]);
    RotaryPower r;
    r.positive = (F.transpose() * w).squaredNorm();
    r.negative = (F.adjoint() * w).squaredNorm();
    r.sense = r.positive >= r.negative ? 1 : -1;
    return r;
}

double impropriety(const VectorXcd& u) {
    const double n = u.squaredNorm();
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    cd s(0, 0);
    for (Eigen::Index i = 0; i < u.size(); ++i) s += u[i] * u[i];
    return std::abs(s) / n;
}

RotarySpectrum rotarySpectrum(const VectorXcd& z, double sfreq, const std::vector<double>& freqs, double ncycles) {
    if (!(sfreq > 0)) throw std::invalid_argument("rotarySpectrum: the sampling frequency must be positive");
    const Eigen::Index T = z.size();
    RotarySpectrum out;
    out.positive = MatrixXd::Zero(static_cast<Eigen::Index>(freqs.size()), T);
    out.negative = MatrixXd::Zero(static_cast<Eigen::Index>(freqs.size()), T);
    for (std::size_t fi = 0; fi < freqs.size(); ++fi) {
        const double f = freqs[fi];
        if (!(std::abs(f) > 0)) throw std::invalid_argument("rotarySpectrum: frequencies must be nonzero");
        const double sd = ncycles / (2 * kPi * std::abs(f));
        const Eigen::Index L = static_cast<Eigen::Index>(std::ceil(8 * sd * sfreq));      // numpy.arange(-4 sd, 4 sd, 1/fs)
        std::vector<double> g(static_cast<std::size_t>(L));
        double gsum = 0;
        for (Eigen::Index j = 0; j < L; ++j) {
            const double t = -4 * sd + static_cast<double>(j) / sfreq;
            g[static_cast<std::size_t>(j)] = std::exp(-t * t / (2 * sd * sd));
            gsum += g[static_cast<std::size_t>(j)];
        }
        const Eigen::Index off = (L - 1) / 2;
        for (int k = 0; k < 2; ++k) {
            const double sgn = k == 0 ? 1.0 : -1.0;
            std::vector<cd> w(static_cast<std::size_t>(L));
            for (Eigen::Index j = 0; j < L; ++j) {
                const double t = -4 * sd + static_cast<double>(j) / sfreq;
                w[static_cast<std::size_t>(j)] = g[static_cast<std::size_t>(j)] / gsum * std::polar(1.0, 2 * kPi * sgn * f * t);
            }
            for (Eigen::Index n = 0; n < T; ++n) {
                cd acc(0, 0);
                const Eigen::Index jlo = std::max<Eigen::Index>(0, n + off - (T - 1)), jhi = std::min<Eigen::Index>(L - 1, n + off);
                for (Eigen::Index j = jlo; j <= jhi; ++j) acc += w[static_cast<std::size_t>(j)] * z[n + off - j];
                (k == 0 ? out.positive : out.negative)(static_cast<Eigen::Index>(fi), n) = std::norm(acc);
            }
        }
    }
    return out;
}

QuadraturePairFilters rotationPlaneFilters(const MatrixXd& W, const MatrixXcd& S) {
    if (W.cols() != S.rows() || S.rows() != S.cols()) throw std::invalid_argument("rotationPlaneFilters: W is [R x C], S is [C x C]");
    const MatrixXcd L = core::hermitianFactor(S);
    QuadraturePairFilters r;
    r.fx = VectorXd::Zero(W.cols());
    r.fy = VectorXd::Zero(W.cols());
    if (L.cols() == 0) return r;
    const MatrixXcd F = W.cast<cd>() * L;
    const QuadraturePair pr = top_plane(core::skewFactor(F));
    if (pr.p.size() == 0) return r;
    r.sigma = pr.sigma;
    r.fx = W.transpose() * pr.p;
    r.fy = W.transpose() * pr.q;
    VectorXcd w(r.fx.size());
    for (Eigen::Index i = 0; i < w.size(); ++i) w[i] = cd(r.fx[i], r.fy[i]);
    r.positive = (w.transpose() * S * w.conjugate())(0, 0).real();
    r.negative = (w.adjoint() * S * w)(0, 0).real();
    if (r.positive < r.negative) { r.fy = -r.fy; r.sense = -1; std::swap(r.positive, r.negative); }
    return r;
}

}  // namespace nxr::signal

namespace nxr::signal {
using Eigen::MatrixXcd; using Eigen::MatrixXd; using Eigen::VectorXcd; using Eigen::VectorXd;

namespace {
PhaseGradient phaseGradientOverDisc(const VectorXcd& z, const operators::FaceOperators& ops, const operators::FaceMat& faces, const VectorXd& areas, const geometry::ChartKit& kit, int anchor,
                                    const std::vector<int>& disc, const Eigen::VectorXd& X, const AnatomicalFrame* anatomy) {
    if (z.size() != ops.nV) throw std::invalid_argument("weightedPhaseGradient: z needs one value per vertex");
    if (faces.rows() != ops.nF || areas.size() != ops.nF) throw std::invalid_argument("weightedPhaseGradient: faces and areas must be the operators' faces");
    if (kit.n() != ops.nV) throw std::invalid_argument("weightedPhaseGradient: the chart kit is of another mesh");
    if (anchor < 0 || anchor >= ops.nV) throw std::invalid_argument("weightedPhaseGradient: the anchor is outside the mesh");
    std::vector<char> in(static_cast<std::size_t>(ops.nV), 0);
    for (int v : disc) in[static_cast<std::size_t>(v)] = 1;
    // X: the anchor's e1 carried to every vertex, (re, im) in that vertex's basis (the caller carried it: geometry::transportedEach)
    const VectorXd zr = z.real(), zi = z.imag();
    VectorXd gr[3], gi[3];
    for (int d = 0; d < 3; ++d) { gr[d] = ops.grad[d] * zr; gi[d] = ops.grad[d] * zi; }
    cd S(0, 0);
    double den = 0;
    for (int f = 0; f < ops.nF; ++f) {
        const cd zf = (z[faces(f, 0)] + z[faces(f, 1)] + z[faces(f, 2)]) / 3.0;
        double P[3], g2 = 0;
        for (int d = 0; d < 3; ++d) {
            const cd g(gr[d][f], gi[d][f]);
            P[d] = (std::conj(zf) * g).imag();
            g2 += std::norm(g);
        }
        const double len = std::sqrt(P[0] * P[0] + P[1] * P[1] + P[2] * P[2]), mag = std::abs(zf) * std::sqrt(g2);
        for (int k = 0; k < 3; ++k) {
            const int w = faces(f, k);
            if (!in[static_cast<std::size_t>(w)]) continue;
            const double wa = areas[f] / 3.0;
            den += wa * mag;
            if (!(len > 0)) continue;
            const double xr = X[2 * w], xi = X[2 * w + 1];
            if (!std::isfinite(xr) || !std::isfinite(xi)) continue;
            const double h = kit.gaugeHeading(w, P);                     // P_f at w, as an angle in w's basis
            if (!std::isfinite(h)) continue;
            S += wa * len * std::polar(1.0, h - std::atan2(xi, xr));       // carried back to the anchor's gauge
        }
    }
    PhaseGradient r;
    r.direction = std::atan2(-S.imag(), -S.real());
    r.phaseGradientAlignment = den > 0 ? std::abs(S) / den : 0.0;
    double t[3];
    kit.headingVector(anchor, std::arg(S), t);
    r.weightedSum = std::abs(S) * Eigen::Vector3d(t[0], t[1], t[2]);
    if (anatomy) {                                                         // the travel direction -S read in the anatomical frame at the anchor
        geometry::Mat3 vec = geometry::Mat3::Zero(ops.nV, 3);
        vec.row(anchor) = -r.weightedSum.transpose();
        const auto comp = geometry::anatomicalComponents(*anatomy->anatomical, *anatomy->e1, *anatomy->e2, vec);
        Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> one = comp.row(anchor);
        r.directionFromAnterior = geometry::angleFromAnterior(one)[0];
    }
    return r;
}
}  // namespace

std::vector<PhaseGradient> weightedPhaseGradients(const VectorXcd& z, const operators::FaceOperators& ops, const operators::FaceMat& faces, const VectorXd& areas, const geometry::ChartKit& kit,
                                                  const std::vector<int>& anchors, double radius, const AnatomicalFrame* anatomy, double maxRadius) {
    if (kit.n() != ops.nV) throw std::invalid_argument("weightedPhaseGradient: the chart kit is of another mesh");
    for (int a : anchors) if (a < 0 || a >= ops.nV) throw std::invalid_argument("weightedPhaseGradient: the anchor is outside the mesh");
    std::vector<std::vector<int>> disc;
    try { disc = geometry::discs(kit, anchors, radius, maxRadius); } catch (const std::exception& e) { throw std::invalid_argument(e.what()); }
    std::vector<PhaseGradient> out(anchors.size());
    geometry::transportedEach(kit, anchors, {}, [&](std::size_t k, const VectorXd& X) { out[k] = phaseGradientOverDisc(z, ops, faces, areas, kit, anchors[k], disc[k], X, anatomy); });
    return out;
}

PhaseGradient weightedPhaseGradient(const VectorXcd& z, const operators::FaceOperators& ops, const operators::FaceMat& faces, const VectorXd& areas, const geometry::ChartKit& kit, int anchor, double radius,
                                    const AnatomicalFrame* anatomy, double maxRadius) {
    return weightedPhaseGradients(z, ops, faces, areas, kit, {anchor}, radius, anatomy, maxRadius)[0];
}

PhaseGradient weightedPhaseGradient(const VectorXcd& z, const operators::FaceOperators& ops, const operators::FaceMat& faces, const VectorXd& areas, const geometry::ChartKit& kit, int anchor,
                                    const selection::Elements& disc, const AnatomicalFrame* anatomy) {
    if (disc.members() != 1) throw std::invalid_argument("weightedPhaseGradient: the neighbourhood is one member (the disc of a distance selection), got " + std::to_string(disc.members()));
    if (disc.size != ops.nV) throw std::invalid_argument("weightedPhaseGradient: the neighbourhood lies on another manifold than the mesh");
    if (anchor < 0 || anchor >= ops.nV) throw std::invalid_argument("weightedPhaseGradient: the anchor is outside the mesh");
    std::vector<int> v;
    for (std::int64_t i = disc.offsets[0]; i < disc.offsets[1]; ++i) v.push_back(static_cast<int>(disc.elements[static_cast<std::size_t>(i)]));
    return phaseGradientOverDisc(z, ops, faces, areas, kit, anchor, v, geometry::transported(kit, anchor, 0.0), anatomy);
}

VectorXcd standingSurrogate(const VectorXcd& z) {
    cd s(0, 0);
    for (Eigen::Index i = 0; i < z.size(); ++i) s += z[i] * z[i];
    const double th = 0.5 * std::arg(s);
    const cd ph = std::polar(1.0, th);
    VectorXcd out(z.size());
    for (Eigen::Index i = 0; i < z.size(); ++i) out[i] = (z[i] * std::conj(ph)).real() * ph;
    return out;
}

}  // namespace nxr::signal

namespace nxr::signal {
using Eigen::MatrixXcd; using Eigen::MatrixXd; using Eigen::VectorXcd; using Eigen::VectorXd;

MatrixXd phaseSpectrum(const MatrixXcd& C) { return C.unaryExpr([](const cd& c) { return std::arg(c); }); }
double phaseSpectrum(std::complex<double> c) { return std::arg(c); }
MatrixXd timeDelay(const MatrixXcd& C, double f) { return phaseSpectrum(C) / (2 * kPi * f); }
double timeDelay(std::complex<double> c, double f) { return std::arg(c) / (2 * kPi * f); }

}  // namespace nxr::signal

