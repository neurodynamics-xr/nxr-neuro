// Hodge–Dirac d + δ on the intrinsic Delaunay triangulation, and its eigenmodes by the exact/coexact split.
// Reference: nxr-cortical-flow hodgeDirac.ts (+ solve-lb-hodge.py, make-hodge-package.py); ⋆1 floor optional (default off, as nxr; Rheome PR #94 used 1e-6).
#include "nxr/compute.h"
#include "geometrycentral/surface/intrinsic_geometry_interface.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace nxr::manifold::ops::hodgedirac {

// The Hodge–Dirac's triangulation is THE one intrinsic construction (nxr/intrinsic_surface.h, on geometry-central): the input's edge lengths, mollified
// (`mollify`), the mixed-Voronoi mass of the mollified input (⋆0), flipped to Delaunay (`flipEps`). The Manifold's intrinsic surface is used when it is
// the intrinsic one built with these very options; otherwise the same construction is run on the Manifold's input. d0, d1, ⋆1 and ⋆2 are
// geometry-central's DEC operators of that triangulation.
HodgeDirac assemble(Manifold& m, double star1Floor, double mollify, double flipEps) {
    std::unique_ptr<IntrinsicSurface> own;
    IntrinsicSurface* sp = &m.intrinsicSurface();
    if (!(m.isIntrinsicDelaunay() && m.options().delaunay.mollify == mollify && m.options().delaunay.flipEps == flipEps)) {
        const Eigen::MatrixXd& Pm = m.vertexPositions();
        const Eigen::MatrixXi& Fm = m.faces();
        std::vector<double> P(static_cast<size_t>(3 * m.nV()));
        std::vector<int32_t> F(static_cast<size_t>(3 * Fm.rows()));
        for (int i = 0; i < m.nV(); ++i) for (int k = 0; k < 3; ++k) P[static_cast<size_t>(3 * i + k)] = Pm(i, k);
        for (int f = 0; f < Fm.rows(); ++f) for (int k = 0; k < 3; ++k) F[static_cast<size_t>(3 * f + k)] = Fm(f, k);
        IntrinsicOptions o;
        o.intrinsic = true; o.mollify = mollify; o.flipEps = flipEps;
        own = std::make_unique<IntrinsicSurface>(P.data(), m.nV(), F.data(), static_cast<int>(Fm.rows()), o);
        sp = own.get();
    }
    IntrinsicSurface& surf = *sp;
    HodgeDirac h;
    h.record = surf.record();
    h.mollifyEps = h.record.epsilon;
    const std::vector<double> mass = surf.massDiagonal();
    h.star0 = Eigen::Map<const Eigen::VectorXd>(mass.data(), static_cast<Eigen::Index>(mass.size()));

    auto& g = surf.geometry();
    g.requireDECOperators();
    const int nV = surf.nV(), nE = static_cast<int>(g.d0.rows()), nF = static_cast<int>(g.d1.rows());
    h.nV = nV; h.nE = nE; h.nF = nF;
    h.d0 = g.d0; h.d1 = g.d1;
    h.star1 = g.hodge1.diagonal();
    h.star2 = g.hodge2.diagonal();
    g.unrequireDECOperators();
    // ⋆1 (the cotan weights of a Delaunay triangulation) is non-negative, and the pencil's mass carries it (and the 2-form ladder its inverse): an edge whose weight is zero within rounding (exactly cocircular
    // quads, as on a symmetric mesh: -1e-17 or 0) makes the pencil indefinite and the shift-invert factorization fail. With no floor asked for, such an edge is raised to ROUNDING level (1e-9 of the mean weight),
    // counted in `star1Clamped` (stored with the solve); a floor the caller names (Rheome's 1e-6) is applied as before and replaces it.
    if (!(star1Floor > 0.0) && nE > 0) {
        double mean = 0.0;
        for (int e = 0; e < nE; ++e) mean += std::abs(h.star1(e));
        mean /= nE;
        const double tiny = 1e-9 * mean;
        for (int e = 0; e < nE; ++e)
            if (h.star1(e) < tiny) { ++h.star1Clamped; h.star1(e) = tiny; }
    }
    if (star1Floor > 0.0)
        for (int e = 0; e < nE; ++e)
            if (h.star1(e) < star1Floor) { ++h.star1Clamped; h.star1(e) = star1Floor; }
    h.complex = surf.complex();
    h.faces.resize(nF, 3);
    for (int f = 0; f < nF; ++f) for (int k = 0; k < 3; ++k) h.faces(f, k) = h.complex.faces[static_cast<size_t>(3 * f + k)];

    const int N = nV + nE + nF;
    Eigen::SparseMatrix<double> A = h.star1.asDiagonal() * h.d0;      // ⋆1 d0  [E×V]
    Eigen::SparseMatrix<double> B = h.star2.asDiagonal() * h.d1;      // ⋆2 d1  [F×E]
    std::vector<Eigen::Triplet<double>> tk;
    for (int o2 = 0; o2 < A.outerSize(); ++o2)
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, o2); it; ++it) {
            tk.emplace_back(nV + static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
            tk.emplace_back(static_cast<int>(it.col()), nV + static_cast<int>(it.row()), it.value());
        }
    for (int o2 = 0; o2 < B.outerSize(); ++o2)
        for (Eigen::SparseMatrix<double>::InnerIterator it(B, o2); it; ++it) {
            tk.emplace_back(nV + nE + static_cast<int>(it.row()), nV + static_cast<int>(it.col()), it.value());
            tk.emplace_back(nV + static_cast<int>(it.col()), nV + nE + static_cast<int>(it.row()), it.value());
        }
    h.K.resize(N, N); h.K.setFromTriplets(tk.begin(), tk.end()); h.K.makeCompressed();
    h.M.resize(N);
    h.M << h.star0, h.star1, h.star2;
    return h;
}

Eigen::SparseMatrix<double> massMatrix(const HodgeDirac& h) {
    Eigen::SparseMatrix<double> M(h.M.size(), h.M.size());
    M = h.M.asDiagonal();
    return M;
}

}  // namespace nxr::manifold::ops::hodgedirac

namespace nxr::manifold::solve {

using ops::hodgedirac::HodgeDirac;

Eigen::VectorXi classifyHodgeModes(const HodgeDirac& h, const Eigen::MatrixXd& X, const Eigen::VectorXd& mu, double tol) {
    const int k = static_cast<int>(X.cols());
    Eigen::VectorXi kind(k);
    const double mmax = mu.size() ? mu.cwiseAbs().maxCoeff() : 0.0;
    for (int c = 0; c < k; ++c) {
        double e0 = 0, e2 = 0;
        for (int i = 0; i < h.nV; ++i) e0 += h.M(i) * X(i, c) * X(i, c);
        for (int i = 0; i < h.nF; ++i) e2 += h.M(h.nV + h.nE + i) * X(h.nV + h.nE + i, c) * X(h.nV + h.nE + i, c);
        kind(c) = (std::abs(mu(c)) <= tol * mmax) ? 1 : (e0 >= e2 ? 0 : 2);
    }
    return kind;
}

EigenResult hodgeLadder(const Eigen::SparseMatrix<double>& S, const Eigen::SparseMatrix<double>& M, int k, const HodgeSolveOptions& opt,
                        const CancellationToken& cancel, const ProgressObserver& progress) {
    EigenResult r;
    if (opt.slice > 0) {
        r = eigenSliced(S, M, k, opt.slice, opt.normalize, cancel, progress);
    } else if (opt.shift == HodgeShift::RelativeLadder) {
        r = eigenSmallest(S, M, k, opt.normalize, cancel, progress);
    } else {
        EigenControls ctl;
        ctl.ncv = opt.ncv; ctl.maxit = opt.maxit; ctl.tol = opt.tol; ctl.requireAll = true;
        r = eigen(S, M, k, opt.sigma, opt.normalize, /*removeDC=*/false, cancel, progress, ctl);
    }
    for (Eigen::Index i = 0; i < r.eigenvalues.size(); ++i) r.eigenvalues(i) = std::max(r.eigenvalues(i), 0.0);
    return r;
}

HodgeLadders hodgeLadders(const Eigen::SparseMatrix<double>& L0, const Eigen::SparseMatrix<double>& M0,
                          const Eigen::SparseMatrix<double>& S2, const Eigen::SparseMatrix<double>& M2,
                          int count, const HodgeSolveOptions& opt, const CancellationToken& cancel, const ProgressObserver& progress) {
    if (count < 1) throw Error(ErrorCode::EigensolveInvalidK, "hodgeLadders: count must be >= 1");
    const int nV = static_cast<int>(M0.rows()), nF = static_cast<int>(M2.rows());
    const int want = count / 2 + 1;
    const int k0 = std::min(want + 1, nV - 2), k2 = std::min(want + 1, nF - 2);
    if (k0 < 1 || k2 < 1) throw Error(ErrorCode::EigensolveInvalidK, "hodgeLadders: the surface is too small for the solve");
    HodgeLadders out;
    out.exact = hodgeLadder(L0, M0, k0, opt, cancel, progress);
    out.coexact = hodgeLadder(S2, M2, k2, opt, cancel, progress);
    const Eigen::VectorXd& l0 = out.exact.eigenvalues;
    const Eigen::VectorXd& l2 = out.coexact.eigenvalues;
    // the constants (μ = 0) out, per ladder: λ > 1e-6 · max λ of the ladder
    const double c0 = 1e-6 * l0.maxCoeff(), c2 = 1e-6 * l2.maxCoeff();
    std::vector<int> i0, i2;
    for (int i = 0; i < l0.size(); ++i) if (l0(i) > c0) i0.push_back(i);
    for (int i = 0; i < l2.size(); ++i) if (l2(i) > c2) i2.push_back(i);
    if (i0.empty() || i2.empty()) throw Error(ErrorCode::EigensolveInvalidK, "hodgeLadders: a ladder has no positive mode (the surface is too small for the solve)");
    double m0 = 0, m2 = 0;
    for (int i : i0) m0 = std::max(m0, l0(i));
    for (int i : i2) m2 = std::max(m2, l2(i));
    out.top = std::min(m0, m2);                                  // ONE λ for both ladders: the band is complete
    std::vector<HodgeLadders::Mode> all;
    for (int i : i0) if (l0(i) <= out.top) all.push_back({std::sqrt(l0(i)), 0, i});
    for (int i : i2) if (l2(i) <= out.top) all.push_back({std::sqrt(l2(i)), 2, i});
    double mmax = 0;
    for (const auto& m : all) mmax = std::max(mmax, m.mu);
    all.erase(std::remove_if(all.begin(), all.end(), [&](const HodgeLadders::Mode& m) { return !(m.mu > 1e-6 * mmax); }), all.end());
    std::stable_sort(all.begin(), all.end(), [](const HodgeLadders::Mode& a, const HodgeLadders::Mode& b) { return a.mu < b.mu; });
    for (const auto& m : all) (m.kind == 0 ? out.nExact : out.nCoexact)++;
    out.modes = std::move(all);
    return out;
}

EigenResult hodgeDiracModes(const HodgeDirac& h, int count, bool bothSigns, const HodgeSolveOptions& opt,
                            const CancellationToken& cancel, const ProgressObserver& progress) {
    if (count < 1) throw Error(ErrorCode::EigensolveInvalidK, "hodgeDiracModes: count must be >= 1");
    const int nV = h.nV, nE = h.nE, nF = h.nF, N = nV + nE + nF;
    Eigen::SparseMatrix<double> s1(nE, nE), s1inv(nE, nE);
    s1 = h.star1.asDiagonal();
    s1inv = h.star1.cwiseInverse().asDiagonal();
    // Δ0 = d0ᵀ⋆1 d0 (mass ⋆0);  Δ2 = (⋆2d1) ⋆1⁻¹ (⋆2d1)ᵀ (mass ⋆2)
    Eigen::SparseMatrix<double> L0 = h.d0.transpose() * s1 * h.d0;
    Eigen::SparseMatrix<double> B = h.star2.asDiagonal() * h.d1;
    Eigen::SparseMatrix<double> S2 = B * s1inv * B.transpose();
    Eigen::SparseMatrix<double> M0(nV, nV), M2(nF, nF);
    M0 = h.star0.asDiagonal(); M2 = h.star2.asDiagonal();

    HodgeLadders lad = hodgeLadders(L0, M0, S2, M2, count, opt, cancel, progress);
    const EigenResult& r0 = lad.exact;
    const EigenResult& r2 = lad.coexact;
    std::vector<HodgeLadders::Mode> cand = lad.modes;
    if (opt.limit > 0 && static_cast<int>(cand.size()) > opt.limit) cand.resize(opt.limit);
    const int kp = static_cast<int>(cand.size());
    const int kOut = bothSigns ? 2 * kp : kp;

    EigenResult out;
    out.eigenvectors = Eigen::MatrixXd::Zero(N, kOut);
    out.eigenvalues.resize(kOut);
    out.kind.resize(kOut);
    const double r2h = 1.0 / std::sqrt(2.0);
    for (int c = 0; c < kp; ++c) {
        const double mu = cand[c].mu;
        Eigen::VectorXd x = Eigen::VectorXd::Zero(N);
        if (cand[c].kind == 0) {
            Eigen::VectorXd psi = r0.eigenvectors.col(cand[c].col);
            x.head(nV) = psi * r2h;
            x.segment(nV, nE) = (h.d0 * psi) * (r2h / mu);
        } else {
            Eigen::VectorXd chi = r2.eigenvectors.col(cand[c].col);
            x.tail(nF) = chi * r2h;
            x.segment(nV, nE) = (s1inv * (B.transpose() * chi)) * (r2h / mu);
        }
        // ascending in μ: with bothSigns the −μ mirrors come first (−μ_max … −μ_min), then +μ_min … +μ_max
        const int pc = bothSigns ? kp + c : c;
        out.eigenvectors.col(pc) = x;
        out.eigenvalues(pc) = mu;
        out.kind(pc) = cand[c].kind;
        if (bothSigns) {
            Eigen::VectorXd y = x;
            y.segment(nV, nE) *= -1.0;                     // (ψ, −ω, χ) has eigenvalue −μ
            const int nc = kp - 1 - c;
            out.eigenvectors.col(nc) = y;
            out.eigenvalues(nc) = -mu;
            out.kind(nc) = cand[c].kind;
        }
    }
    out.k = kOut; out.nConverged = kOut;
    return out;
}

}  // namespace nxr::manifold::solve
