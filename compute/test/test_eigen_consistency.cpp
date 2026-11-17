// test_eigen_consistency.cpp: the eigensolver gives the same answer however it is reached.
//
//  1. one-shot (assemble + eigen with the default ladder, normalised, null space removed: what the one-shot command does)
//     equals step-by-step (assemble, solve, normalize, removeDC as separate calls) to 1e-10;
//  2. an explicit shift clear of the null space (sigma = -1e-3) gives the same eigenpairs as the default;
//  3. the normalisation flag does not change the eigenvalues, and the eigenvectors agree up to the M-orthonormalisation.
// Eigenvalues are compared one by one; eigenvectors as subspaces, cluster by cluster, where eigenvalues repeat (the sphere's
// degree-l multiplets). Meshes: an icosphere (repeated eigenvalues) and a jittered one (none).

#include "nxr/compute.h"
#include "synthetic_meshes.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace nxr::manifold;
using namespace nxr::manifold::solve;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; std::printf("  FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static double maxLambdaDiff(const EigenResult& a, const EigenResult& b) {
    if (a.eigenvalues.size() != b.eigenvalues.size()) return INFINITY;
    return (a.eigenvalues - b.eigenvalues).cwiseAbs().maxCoeff();
}

// Largest M-distance between the subspaces spanned by the eigenvector clusters of a and b (clusters = runs of eigenvalues closer than
// `gap`; the sign and the basis inside a cluster are free). Both are M-orthonormal.
static double maxSubspaceDiff(const EigenResult& a, const EigenResult& b, const Eigen::SparseMatrix<double>& M, double gap = 1e-6) {
    const int n = (int)a.eigenvalues.size();
    double worst = 0; int s = 0;
    while (s < n) {
        int e = s + 1;
        while (e < n && a.eigenvalues(e) - a.eigenvalues(e - 1) < gap * (1 + std::abs(a.eigenvalues(e)))) ++e;
        const Eigen::MatrixXd A = a.eigenvectors.middleCols(s, e - s), B = b.eigenvectors.middleCols(s, e - s);
        const Eigen::MatrixXd R = A - B * (B.transpose() * (M * A));      // A minus its M-projection on span(B)
        worst = std::max(worst, std::sqrt((R.transpose() * (M * R)).trace() / (e - s)));
        s = e;
    }
    return worst;
}

static double mOrthoDeviation(const EigenResult& r, const Eigen::SparseMatrix<double>& M) {
    const Eigen::MatrixXd G = r.eigenvectors.transpose() * (M * r.eigenvectors);
    return (G - Eigen::MatrixXd::Identity(G.rows(), G.cols())).cwiseAbs().maxCoeff();
}

static void runMesh(const std::string& name, const Mesh& mesh, int k) {
    std::printf("[%s] nV=%d k=%d\n", name.c_str(), mesh.nV(), k);
    const double tol = 1e-10;

    // one-shot: what the one-shot command does
    Manifold m1(mesh.V.data(), mesh.nV(), mesh.F.data(), mesh.nF());
    auto ops1 = ops::assembleManifoldOperators(m1);
    const EigenResult oneShot = eigen(ops1.cotanLaplacian, ops1.mass, k, kShiftLadder, /*normalize=*/true, /*removeDC=*/true);

    // step by step, on a fresh Manifold: assemble, solve, normalize, removeDC
    Manifold m2(mesh.V.data(), mesh.nV(), mesh.F.data(), mesh.nF());
    auto ops2 = ops::assembleManifoldOperators(m2);
    EigenResult raw = eigen(ops2.cotanLaplacian, ops2.mass, k);
    raw.eigenvectors = normalize(raw.eigenvectors, ops2.mass);
    const EigenResult stepped = removeDC(raw);

    CHECK(oneShot.k == stepped.k, "one-shot k %d vs step-by-step %d", oneShot.k, stepped.k);
    CHECK(oneShot.k == k - 1, "removeDC should drop exactly one mode (k=%d)", oneShot.k);
    const double dl = maxLambdaDiff(oneShot, stepped), dv = maxSubspaceDiff(oneShot, stepped, ops1.mass);
    std::printf("  one-shot vs step-by-step: max|dlambda| %.2e, subspace %.2e\n", dl, dv);
    CHECK(dl < tol, "one-shot vs step-by-step eigenvalues differ by %g", dl);
    CHECK(dv < 1e-7, "one-shot vs step-by-step eigenvectors differ by %g", dv);
    CHECK(mOrthoDeviation(oneShot, ops1.mass) < 1e-10, "one-shot not M-orthonormal");

    // the ladder entry point (eigenSmallest) agrees too
    const EigenResult sm = eigenSmallest(ops2.cotanLaplacian, ops2.mass, k, true);
    const EigenResult smNoDC = removeDC(sm);
    const double ds = maxLambdaDiff(oneShot, smNoDC);
    std::printf("  eigenSmallest vs one-shot: max|dlambda| %.2e\n", ds);
    CHECK(ds < tol, "eigenSmallest differs from one-shot by %g", ds);

    // an explicit shift clear of the null space gives the same eigenpairs as the default
    const EigenResult shifted = eigen(ops2.cotanLaplacian, ops2.mass, k, -1e-3, true, true);
    const double dsh = maxLambdaDiff(oneShot, shifted), dshv = maxSubspaceDiff(oneShot, shifted, ops1.mass);
    std::printf("  sigma=-1e-3 vs default: max|dlambda| %.2e, subspace %.2e\n", dsh, dshv);
    CHECK(dsh < tol, "sigma=-1e-3 eigenvalues differ from the default by %g", dsh);
    CHECK(dshv < 1e-7, "sigma=-1e-3 eigenvectors differ from the default by %g", dshv);

    // the normalisation flag does not change the eigenvalues; the vectors agree once M-orthonormalised
    const EigenResult unnorm = eigen(ops2.cotanLaplacian, ops2.mass, k, kShiftLadder, /*normalize=*/false, /*removeDC=*/true);
    const double dn = maxLambdaDiff(oneShot, unnorm);
    std::printf("  normalize false vs true: max|dlambda| %.2e\n", dn);
    CHECK(dn < tol, "normalisation flag changes the eigenvalues by %g", dn);
    EigenResult unnormO = unnorm;
    unnormO.eigenvectors = normalize(unnorm.eigenvectors, ops2.mass);
    const double dnv = maxSubspaceDiff(oneShot, unnormO, ops1.mass);
    CHECK(dnv < 1e-7, "normalisation flag changes the eigenvector subspaces by %g", dnv);
}

int main() {
    runMesh("icosphere level 2 (repeated eigenvalues)", icosphere(2), 9);   // 1 + 3 + 5: ends on a multiplet boundary
    runMesh("jittered icosphere level 2", jitterSphere(icosphere(2), 0.25, 7u), 8);
    if (g_fail) { std::printf("test_eigen_consistency: %d FAILED\n", g_fail); return 1; }
    std::printf("test_eigen_consistency: all passed\n");
    return 0;
}
