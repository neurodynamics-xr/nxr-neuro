// nxr::flow methods (opticalflow_methods.h), on synthetic icospheres:
//  * pcg is the default method and equals direct to its tolerance (both preconditioners); iterations per frame; warm start helps
//  * perscale separates two rotating patterns of different scale (opposite senses of rotation land in different members)
//  * FlowEngine(direct) == opticalflowScalar bit for bit
// Set NXR_BENCH=1 for the per-frame timings on larger spheres.
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include "mesh_helpers.h"
#include "nxr/signal/opticalflow_methods.h"

using namespace nxr::signal;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct Sphere {
    RowMat3 P; nxr::operators::FaceMat F; RowMat3 N;
    VectorXd mass;
};

static Sphere makeSphere(int level) {
    std::vector<double> V; std::vector<int32_t> Ff;
    icosphere(level, V, Ff);
    Sphere s;
    s.P = Eigen::Map<RowMat3>(V.data(), V.size() / 3, 3);
    s.F = Eigen::Map<nxr::operators::FaceMat>(Ff.data(), Ff.size() / 3, 3);
    s.N = vertexNormals(s.P, s.F);
    Eigen::SparseMatrix<double> L;
    embeddedLaplacian(s.P, s.F, L, s.mass);
    return s;
}

// Gaussians rotating about z by theta per frame
static MatrixXd rotating(const RowMat3& P, const std::vector<Eigen::Vector3d>& c0, double w2, double theta, int T, double base = 0.1) {
    MatrixXd A = MatrixXd::Constant(P.rows(), T, base);
    for (int t = 0; t < T; ++t)
        for (const Eigen::Vector3d& c0i : c0) {
            const Eigen::Vector3d c = Eigen::AngleAxisd(theta * t, Eigen::Vector3d::UnitZ()) * c0i;
            for (Eigen::Index i = 0; i < P.rows(); ++i) A(i, t) += std::exp(-(P.row(i).transpose() - c).squaredNorm() / w2);
        }
    return A;
}

static FlowMethodOptions directOptions() { FlowMethodOptions o; o.method = FlowMethod::Direct; return o; }

static double rel(const VectorXd& a, const VectorXd& b) { return (a - b).norm() / std::max(b.norm(), 1e-300); }

// the mean cosine between a velocity and the rotation about z (the motion of a pattern turning about z), weighted by speed^2
static double rotationSense(const RowMat3& P, const VectorXd& v) {
    double num = 0, den = 0;
    for (Eigen::Index i = 0; i < P.rows(); ++i) {
        const Eigen::Vector3d truth = Eigen::Vector3d::UnitZ().cross(Eigen::Vector3d(P.row(i))), est = v.segment<3>(3 * i);
        num += est.dot(truth); den += est.norm() * truth.norm();
    }
    return num / std::max(den, 1e-300);
}

static std::vector<VectorXd> run(FlowEngine& e, const MatrixXd& A, std::vector<FlowStep>* steps = nullptr) {
    for (Eigen::Index t = 0; t < A.cols(); ++t) e.observe(A.col(t));
    std::vector<VectorXd> v;
    for (Eigen::Index t = 0; t + 1 < A.cols(); ++t) {
        FlowStep s = e.step(A.col(t), A.col(t + 1));
        v.push_back(s.velocity[0]);
        if (steps) steps->push_back(std::move(s));
    }
    return v;
}

int main() {
    const std::vector<Eigen::Vector3d> two = {Eigen::Vector3d(1, 0.3, 0.5).normalized(), Eigen::Vector3d(-0.5, -0.8, -0.2).normalized()};
    const bool bench = std::getenv("NXR_BENCH") != nullptr;
    EXPECT(FlowMethodOptions{}.method == FlowMethod::Pcg && std::string(flowMethodName(FlowMethod::Pcg)) == "pcg", "pcg is the default method");
    bool pyramidGone = false;
    try { parseFlowMethod("pyramid"); } catch (const std::invalid_argument&) { pyramidGone = true; }
    EXPECT(pyramidGone, "pyramid is not a method");
    // ---- direct is opticalflowScalar, bit for bit
    {
        const Sphere s = makeSphere(3);
        const MatrixXd A = rotating(s.P, two, 0.3, 0.06, 5);
        const MatrixXd ref = opticalflowScalar(A, s.P, s.F, s.N, 0.05);
        FlowEngine e(directOptions(), s.P, s.F, s.N, 0.05);
        const auto v = run(e, A);
        double d = 0;
        for (std::size_t t = 0; t < v.size(); ++t) d = std::max(d, (v[t] - ref.col(static_cast<Eigen::Index>(t))).cwiseAbs().maxCoeff());
        EXPECT(d == 0.0, "FlowEngine(direct) equals opticalflowScalar exactly");
    }
    // ---- pcg vs direct
    {
        const Sphere s = makeSphere(3);
        const int T = 8;
        const MatrixXd A = rotating(s.P, two, 0.3, 0.06, T);
        FlowEngine d(directOptions(), s.P, s.F, s.N, 0.05);
        const auto vd = run(d, A);
        for (int pc = 0; pc < 2; ++pc) {
            FlowMethodOptions o; o.method = FlowMethod::Pcg;
            o.preconditioner = pc == 0 ? PcgOptions::Preconditioner::Ldlt : PcgOptions::Preconditioner::IncompleteCholesky;
            FlowEngine e(o, s.P, s.F, s.N, 0.05);
            std::vector<FlowStep> st;
            const auto v = run(e, A, &st);
            double worst = 0;
            for (std::size_t t = 0; t < v.size(); ++t) worst = std::max(worst, rel(v[t], vd[t]));
            std::printf("  pcg (%s): max relative difference from direct %.2e; iterations per frame:", pc == 0 ? "ldlt" : "ic", worst);
            for (const FlowStep& f : st) std::printf(" %d", f.iterations[0]);
            std::printf("\n");
            EXPECT(worst < 1e-6, (pc == 0 ? "pcg (ldlt preconditioner) equals direct to its tolerance" : "pcg (incomplete Cholesky) equals direct to its tolerance"));
        }
    }
    // ---- perscale: two scales turning in opposite senses
    {
        const Sphere s = makeSphere(4);
        std::mt19937 rng(3);
        std::normal_distribution<double> nd;
        std::vector<Eigen::Vector3d> fine, mid;
        for (int i = 0; i < 60; ++i) fine.push_back(Eigen::Vector3d(nd(rng), nd(rng), nd(rng)).normalized());
        for (int i = 0; i < 14; ++i) mid.push_back(Eigen::Vector3d(nd(rng), nd(rng), nd(rng)).normalized());
        const int T = 4;
        const double th = 0.012;
        const MatrixXd coarse = rotating(s.P, mid, 0.012, th, T, 0.0), fineA = rotating(s.P, fine, 0.0015, -th, T, 0.0);
        const MatrixXd A = coarse + fineA;
        FlowMethodOptions o; o.method = FlowMethod::Perscale; o.scaleFamily = "logitersine"; o.scaleVoices = 2;
        FlowEngine e(o, s.P, s.F, s.N, 0.05);
        for (int t = 0; t < T; ++t) e.observe(A.col(t));
        const FlowStep st = e.step(A.col(1), A.col(2));
        const auto cw = e.seriesCentres();
        int lo = -1, hi = -1;
        double best = 0, worst = 0;
        std::printf("  perscale: %d members; rotation sense / centre wavenumber:", e.series());
        for (int m = 0; m < e.series(); ++m) {
            const double r = rotationSense(s.P, st.velocity[static_cast<std::size_t>(m)]);
            std::printf(" [%.2f @ %.1f, %d it]", r, cw[static_cast<std::size_t>(m)], st.iterations[static_cast<std::size_t>(m)]);
            if (r > best) { best = r; lo = m; }
            if (r < worst) { worst = r; hi = m; }
        }
        std::printf("\n");
        EXPECT(lo >= 0 && hi >= 0 && best > 0.3 && worst < -0.3 && cw[static_cast<std::size_t>(lo)] < cw[static_cast<std::size_t>(hi)], "per-scale flows separate: the coarse pattern turns one way in a low-wavenumber member, the fine one the other way in a higher one");
    }
    // ---- timings
    {
        std::printf("  seconds per frame (frame 4 of a rotating-bump window):\n");
        for (int level : bench ? std::vector<int>{3, 4, 5} : std::vector<int>{3}) {
            const Sphere s = makeSphere(level);
            const int T = 6;
            const MatrixXd A = rotating(s.P, two, 0.3, 0.03, T);
            std::vector<std::pair<std::string, FlowMethodOptions>> ms;
            FlowMethodOptions o = directOptions();
            ms.push_back({"direct", o});
            o.method = FlowMethod::Pcg; ms.push_back({"pcg-ldlt", o});
            o.preconditioner = PcgOptions::Preconditioner::IncompleteCholesky; ms.push_back({"pcg-ic", o});
            o.preconditioner = PcgOptions::Preconditioner::Ldlt;
            o.method = FlowMethod::Perscale; o.scaleFamily = "logitersine"; o.scaleVoices = 2; ms.push_back({"perscale", o});
            for (auto& [name, mo] : ms) {
                FlowEngine e(mo, s.P, s.F, s.N, 0.05);
                std::vector<FlowStep> st;
                const auto t0 = std::chrono::steady_clock::now();
                run(e, A, &st);
                double sec = 0; int it = 0;
                for (std::size_t i = 1; i < st.size(); ++i) { sec += st[i].seconds; if (!st[i].iterations.empty()) it += st[i].iterations[0]; }
                std::printf("    V=%6d %-9s %.4f s/frame%s (setup+all %.2f s)", static_cast<int>(s.P.rows()), name.c_str(), sec / (st.size() - 1), mo.method == FlowMethod::Perscale ? (" for " + std::to_string(e.series()) + " members").c_str() : "",
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                if (!st.back().iterations.empty()) std::printf(", %.1f it/frame", static_cast<double>(it) / (st.size() - 1));
                std::printf("\n");
            }
        }
    }
    std::cout << (g_failures ? "FAILED\n" : "ok\n");
    return g_failures ? 1 : 0;
}
