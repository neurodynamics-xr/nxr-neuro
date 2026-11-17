// nxr::quadrature — see include/nxr/quadrature.h for the conventions and the formulas. A port of nxr.spectral.lagged.
#include "nxr/quadrature.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace nxr::quadrature {

namespace {
constexpr double kPi = 3.14159265358979323846;
using cd = std::complex<double>;

MatrixXd antisymmetricPart(const MatrixXd& S) { return (S - S.transpose()) * 0.5; }
}  // namespace

MatrixXd coSpectrum(const MatrixXcd& C) { return C.real(); }

MatrixXd quadratureSpectrum(const MatrixXcd& C) { return C.imag(); }

MatrixXcd hermitianFactor(const MatrixXcd& S, double tolerance) {
    if (S.rows() != S.cols()) throw std::invalid_argument("hermitianFactor: the matrix must be square");
    Eigen::SelfAdjointEigenSolver<MatrixXcd> es(S);
    const VectorXd& e = es.eigenvalues();
    const double top = std::max(0.0, e.size() ? e.maxCoeff() : 0.0);
    std::vector<Eigen::Index> keep;
    for (Eigen::Index i = e.size() - 1; i >= 0; --i) if (e[i] > tolerance * top && e[i] > 0) keep.push_back(i);
    MatrixXcd L(S.rows(), static_cast<Eigen::Index>(keep.size()));
    for (std::size_t k = 0; k < keep.size(); ++k) L.col(static_cast<Eigen::Index>(k)) = es.eigenvectors().col(keep[k]) * std::sqrt(e[keep[k]]);
    return L;
}

QuadratureFactor quadratureFactor(const MatrixXcd& F) {
    const Eigen::Index N = F.rows(), n = F.cols();
    MatrixXd RJ(N, 2 * n);
    RJ.leftCols(n) = F.real();
    RJ.rightCols(n) = F.imag();
    const Eigen::Index m = std::min(N, 2 * n);
    Eigen::HouseholderQR<MatrixXd> qr(RJ);
    QuadratureFactor f;
    f.Q = qr.householderQ() * MatrixXd::Identity(N, m);
    const MatrixXd Rq = qr.matrixQR().topRows(m).triangularView<Eigen::Upper>();
    MatrixXd M = MatrixXd::Zero(2 * n, 2 * n);
    M.bottomLeftCorner(n, n).setIdentity();
    M.topRightCorner(n, n) = -MatrixXd::Identity(n, n);        // [R J] M [R J]^T = J R^T - R J^T
    f.S = antisymmetricPart(Rq * M * Rq.transpose());
    f.reference = F.squaredNorm();
    return f;
}

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

VectorXd quadratureEnergy(const MatrixXcd& Fr, const MatrixXcd& F) {
    if (Fr.cols() != F.cols()) throw std::invalid_argument("quadratureEnergy: Fr and F need the same number of columns");
    const MatrixXcd G = F.adjoint() * F;                         // sum_j conj(F_ja) F_jb
    const MatrixXcd G2 = F.adjoint() * F.conjugate();            // sum_j conj(F_ja) conj(F_jb)
    // sum_j |C_ij|^2 = f_i G f_i^H ;  sum_j C_ij^2 = f_i G2 f_i^T ;  (Im c)^2 = (|c|^2 - Re c^2) / 2
    const MatrixXcd A = Fr * G, B = Fr * G2;
    VectorXd e(Fr.rows());
    for (Eigen::Index i = 0; i < Fr.rows(); ++i) {
        const double n2 = (A.row(i).array() * Fr.row(i).conjugate().array()).sum().real();
        const double c2 = (B.row(i).array() * Fr.row(i).array()).sum().real();
        e[i] = std::max(0.0, (n2 - c2) / 2);
    }
    return e;
}

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

std::vector<QuadraturePair> quadraturePairs(const QuadratureFactor& f, int k) {
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

QuadraturePair quadraturePair(const QuadratureFactor& f) {
    const auto p = quadraturePairs(f, 1);
    return p.empty() ? QuadraturePair{} : p[0];
}

std::vector<QuadraturePair> quadraturePairs(const MatrixXcd& F, int k) { return quadraturePairs(quadratureFactor(F), k); }

double quadratureShare(const QuadratureFactor& f, const std::vector<QuadraturePair>& pairs) {
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

QuadraturePairFilters quadraturePairFilters(const MatrixXd& W, const MatrixXcd& S) {
    if (W.cols() != S.rows() || S.rows() != S.cols()) throw std::invalid_argument("quadraturePairFilters: W is [R x C], S is [C x C]");
    const MatrixXcd L = hermitianFactor(S);
    QuadraturePairFilters r;
    r.fx = VectorXd::Zero(W.cols());
    r.fy = VectorXd::Zero(W.cols());
    if (L.cols() == 0) return r;
    const MatrixXcd F = W.cast<cd>() * L;
    const QuadraturePair pr = quadraturePair(quadratureFactor(F));
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

VectorXd faceAreas(const hodge::RowMat3& positions, const hodge::FaceMat& faces) {
    VectorXd a(faces.rows());
    for (Eigen::Index f = 0; f < faces.rows(); ++f) {
        const Eigen::Vector3d p0 = positions.row(faces(f, 0)), p1 = positions.row(faces(f, 1)), p2 = positions.row(faces(f, 2));
        a[f] = 0.5 * (p1 - p0).cross(p2 - p0).norm();
    }
    return a;
}

namespace {
PhaseGradient phaseGradientOverDisc(const VectorXcd& z, const hodge::FaceOperators& ops, const hodge::FaceMat& faces, const VectorXd& areas, chart::ChartKit& kit, int anchor,
                                    const std::vector<int>& disc, const AnatomicalFrame* anatomy) {
    if (z.size() != ops.nV) throw std::invalid_argument("weightedPhaseGradient: z needs one value per vertex");
    if (faces.rows() != ops.nF || areas.size() != ops.nF) throw std::invalid_argument("weightedPhaseGradient: faces and areas must be the operators' faces");
    if (kit.n() != ops.nV) throw std::invalid_argument("weightedPhaseGradient: the chart kit is of another mesh");
    if (anchor < 0 || anchor >= ops.nV) throw std::invalid_argument("weightedPhaseGradient: the anchor is outside the mesh");
    std::vector<char> in(static_cast<std::size_t>(ops.nV), 0);
    for (int v : disc) in[static_cast<std::size_t>(v)] = 1;
    const Eigen::VectorXd X = kit.transported(anchor, 0.0);          // the anchor's e1 carried to every vertex, (re, im) in that vertex's basis
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
        frames::Mat3 vec = frames::Mat3::Zero(ops.nV, 3);
        vec.row(anchor) = -r.weightedSum.transpose();
        const auto comp = frames::anatomicalComponents(*anatomy->anatomical, *anatomy->e1, *anatomy->e2, vec);
        Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> one = comp.row(anchor);
        r.directionFromAnterior = frames::angleFromAnterior(one)[0];
    }
    return r;
}
}  // namespace

PhaseGradient weightedPhaseGradient(const VectorXcd& z, const hodge::FaceOperators& ops, const hodge::FaceMat& faces, const VectorXd& areas, chart::ChartKit& kit, int anchor, double radius,
                                    const AnatomicalFrame* anatomy, double maxRadius) {
    if (anchor < 0 || anchor >= ops.nV) throw std::invalid_argument("weightedPhaseGradient: the anchor is outside the mesh");
    if (kit.n() != ops.nV) throw std::invalid_argument("weightedPhaseGradient: the chart kit is of another mesh");
    std::vector<int> disc;
    try { disc = kit.disc(anchor, radius, maxRadius); } catch (const std::exception& e) { throw std::invalid_argument(e.what()); }
    return phaseGradientOverDisc(z, ops, faces, areas, kit, anchor, disc, anatomy);
}

PhaseGradient weightedPhaseGradient(const VectorXcd& z, const hodge::FaceOperators& ops, const hodge::FaceMat& faces, const VectorXd& areas, chart::ChartKit& kit, int anchor,
                                    const selection::Elements& disc, const AnatomicalFrame* anatomy) {
    if (disc.members() != 1) throw std::invalid_argument("weightedPhaseGradient: the neighbourhood is one member (the disc of a distance selection), got " + std::to_string(disc.members()));
    if (disc.size != ops.nV) throw std::invalid_argument("weightedPhaseGradient: the neighbourhood lies on another manifold than the mesh");
    std::vector<int> v;
    for (std::int64_t i = disc.offsets[0]; i < disc.offsets[1]; ++i) v.push_back(static_cast<int>(disc.elements[static_cast<std::size_t>(i)]));
    return phaseGradientOverDisc(z, ops, faces, areas, kit, anchor, v, anatomy);
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

double meanResultantLength(const VectorXd& a) {
    if (a.size() == 0) return 0;
    cd m(0, 0);
    for (Eigen::Index i = 0; i < a.size(); ++i) m += std::polar(1.0, a[i]);
    return std::abs(m / static_cast<double>(a.size()));
}

double meanDirection(const VectorXd& a) {
    cd m(0, 0);
    for (Eigen::Index i = 0; i < a.size(); ++i) m += std::polar(1.0, a[i]);
    return std::arg(m);
}

Rayleigh rayleighTest(const VectorXd& a) {
    Rayleigh r;
    r.n = a.size();
    r.R = meanResultantLength(a);
    r.p = std::exp(-static_cast<double>(r.n) * r.R * r.R);
    r.meanDirection = meanDirection(a);
    return r;
}

MatrixXd phaseLag(const MatrixXcd& C) { return C.unaryExpr([](const cd& c) { return std::arg(c); }); }
double phaseLag(std::complex<double> c) { return std::arg(c); }
MatrixXd timeLag(const MatrixXcd& C, double f) { return phaseLag(C) / (2 * kPi * f); }
double timeLag(std::complex<double> c, double f) { return std::arg(c) / (2 * kPi * f); }

namespace {
// (‖T‖², ‖Im T‖²) of T = B C Bᵀ from the Gram G = BᵀB: tr(X G Xᵀ G) = Σ_ij (X G)_ij (G X)_ji... with X symmetric (Cr) or antisymmetric (Cq) after the symmetrisation, one product each
CongruenceNorms normsFromGram(const MatrixXd& G, const MatrixXcd& C) {
    const MatrixXcd H = 0.5 * (C + C.adjoint());                    // Hermitian: Re symmetric, Im antisymmetric
    const MatrixXd Cr = H.real(), Cq = H.imag();
    const MatrixXd P = Cr * G, Q = Cq * G;
    double re = 0, im = 0;
    for (Eigen::Index i = 0; i < P.rows(); ++i)
        for (Eigen::Index j = 0; j < P.cols(); ++j) { re += P(i, j) * P(j, i); im -= Q(i, j) * Q(j, i); }
    return {re + im, im};
}
}  // namespace

CongruenceNorms congruenceNorms(const MatrixXd& B, const MatrixXcd& C) {
    if (C.rows() != C.cols() || C.rows() != B.cols()) throw std::invalid_argument("congruenceNorms: C must be square with as many rows as B has columns");
    return normsFromGram(B.transpose() * B, C);
}

namespace {
struct GroupGrams {
    std::vector<std::vector<Eigen::Index>> rows;
    std::vector<MatrixXd> G;          // [groups] Bᵀ W B (m x m), empty for a group with no row
};
GroupGrams groupGrams(const MatrixXd& A, const std::vector<int>& rowGroup, int groups, const std::vector<MatrixXcd>& Cs, const VectorXd& w, const char* who) {
    const std::string name = who;
    if (static_cast<Eigen::Index>(rowGroup.size()) != A.rows()) throw std::invalid_argument(name + ": one group per row of A");
    if (groups < 0) throw std::invalid_argument(name + ": a negative number of groups");
    if (w.size() != 0 && w.size() != A.rows()) throw std::invalid_argument(name + ": one weight per row of A");
    for (const MatrixXcd& C : Cs)
        if (C.size() != 0 && (C.rows() != C.cols() || C.rows() != A.cols())) throw std::invalid_argument(name + ": every C is square with as many rows as A has columns");
    GroupGrams g;
    g.rows.assign(static_cast<std::size_t>(groups), {});
    g.G.assign(static_cast<std::size_t>(groups), MatrixXd());
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        const int k = rowGroup[static_cast<std::size_t>(i)];
        if (k >= groups || k < -1) throw std::invalid_argument(name + ": a row's group is outside [0, groups)");
        if (k >= 0) g.rows[static_cast<std::size_t>(k)].push_back(i);
    }
    for (int k = 0; k < groups; ++k) {
        const auto& r = g.rows[static_cast<std::size_t>(k)];
        if (r.empty()) continue;
        MatrixXd Bg(static_cast<Eigen::Index>(r.size()), A.cols());
        for (std::size_t q = 0; q < r.size(); ++q) Bg.row(static_cast<Eigen::Index>(q)) = (w.size() ? std::sqrt(w[r[q]]) : 1.0) * A.row(r[q]);     // W^½ B: G = (W^½ B)ᵀ (W^½ B) = Bᵀ W B
        g.G[static_cast<std::size_t>(k)] = Bg.transpose() * Bg;                     // once per group
    }
    return g;
}
}  // namespace

GroupedCongruenceNorms groupedCongruenceNorms(const MatrixXd& A, const std::vector<int>& rowGroup, int groups, const std::vector<MatrixXcd>& Cs, const VectorXd& rowWeights) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const GroupGrams gg = groupGrams(A, rowGroup, groups, Cs, rowWeights, "groupedCongruenceNorms");
    GroupedCongruenceNorms out;
    out.total = MatrixXd::Constant(groups, static_cast<Eigen::Index>(Cs.size()), nan);
    out.quadrature = out.total;
    for (int g = 0; g < groups; ++g) {
        if (gg.rows[static_cast<std::size_t>(g)].empty()) continue;
        for (std::size_t t = 0; t < Cs.size(); ++t) {
            if (Cs[t].size() == 0) continue;
            const CongruenceNorms n = normsFromGram(gg.G[static_cast<std::size_t>(g)], Cs[t]);
            out.total(g, static_cast<Eigen::Index>(t)) = n.total;
            out.quadrature(g, static_cast<Eigen::Index>(t)) = n.quadrature;
        }
    }
    return out;
}

PairCongruenceNorms pairCongruenceNorms(const MatrixXd& A, const std::vector<int>& rowGroup, int groups, const std::vector<std::pair<int, int>>& pairs, const std::vector<MatrixXcd>& Cs, const VectorXd& rowWeights) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const GroupGrams gg = groupGrams(A, rowGroup, groups, Cs, rowWeights, "pairCongruenceNorms");
    for (const auto& p : pairs) if (p.first < 0 || p.second < 0 || p.first >= groups || p.second >= groups) throw std::invalid_argument("pairCongruenceNorms: a pair is outside [0, groups)");
    const Eigen::Index nC = static_cast<Eigen::Index>(Cs.size()), nP = static_cast<Eigen::Index>(pairs.size());
    PairCongruenceNorms out;
    out.tiles.total = MatrixXd::Constant(groups, nC, nan);
    out.tiles.quadrature = out.tiles.total;
    out.pairTotal = MatrixXd::Constant(nP, nC, nan);
    out.pairQuadrature = out.pairTotal;
    std::vector<char> used(static_cast<std::size_t>(groups), 0);
    for (int g = 0; g < groups; ++g) if (!gg.rows[static_cast<std::size_t>(g)].empty()) used[static_cast<std::size_t>(g)] = 1;
    std::vector<MatrixXd> Pr(static_cast<std::size_t>(groups)), Pq(static_cast<std::size_t>(groups));
    for (Eigen::Index t = 0; t < nC; ++t) {
        if (Cs[static_cast<std::size_t>(t)].size() == 0) continue;
        const MatrixXcd H = 0.5 * (Cs[static_cast<std::size_t>(t)] + Cs[static_cast<std::size_t>(t)].adjoint());
        const MatrixXd Cr = H.real(), Cq = H.imag();
        for (int g = 0; g < groups; ++g) {
            if (!used[static_cast<std::size_t>(g)]) continue;
            Pr[static_cast<std::size_t>(g)] = Cr * gg.G[static_cast<std::size_t>(g)];
            Pq[static_cast<std::size_t>(g)] = Cq * gg.G[static_cast<std::size_t>(g)];
            const double re = Pr[static_cast<std::size_t>(g)].cwiseProduct(Pr[static_cast<std::size_t>(g)].transpose()).sum();
            const double im = -Pq[static_cast<std::size_t>(g)].cwiseProduct(Pq[static_cast<std::size_t>(g)].transpose()).sum();
            out.tiles.total(g, t) = re + im;
            out.tiles.quadrature(g, t) = im;
        }
        for (Eigen::Index k = 0; k < nP; ++k) {
            const int a = pairs[static_cast<std::size_t>(k)].first, b = pairs[static_cast<std::size_t>(k)].second;
            if (!used[static_cast<std::size_t>(a)] || !used[static_cast<std::size_t>(b)]) continue;
            // tr(X G_b X G_a) = Σ_ij (X G_b)_ij (X G_a)_ji
            const double re = Pr[static_cast<std::size_t>(b)].cwiseProduct(Pr[static_cast<std::size_t>(a)].transpose()).sum();
            const double im = -Pq[static_cast<std::size_t>(b)].cwiseProduct(Pq[static_cast<std::size_t>(a)].transpose()).sum();
            out.pairTotal(k, t) = re + im;
            out.pairQuadrature(k, t) = im;
        }
    }
    return out;
}

namespace {
std::vector<int> groups_of(const selection::MemberOfElement& g) { return std::vector<int>(g.member_of.begin(), g.member_of.end()); }
}  // namespace

GroupedCongruenceNorms groupedCongruenceNorms(const MatrixXd& A, const selection::MemberOfElement& groups, const std::vector<MatrixXcd>& Cs, const VectorXd& rowWeights) {
    return groupedCongruenceNorms(A, groups_of(groups), static_cast<int>(groups.members), Cs, rowWeights);
}

PairCongruenceNorms pairCongruenceNorms(const MatrixXd& A, const selection::MemberOfElement& groups, const std::vector<std::pair<int, int>>& pairs, const std::vector<MatrixXcd>& Cs, const VectorXd& rowWeights) {
    return pairCongruenceNorms(A, groups_of(groups), static_cast<int>(groups.members), pairs, Cs, rowWeights);
}

}  // namespace nxr::quadrature
