#include "nxr/operators/assembled.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/transport.h"

#include <string>
#include <vector>

namespace nxr::operators {

using core::Error;
using core::ErrorCode;

Eigen::SparseMatrix<std::complex<double>> vertexConnectionLaplacian(const geometry::SurfaceArrays& surface, geometry::ConnectionType type,
                                                                   const std::map<int,double>& singularities) {
    geometry::checkSingularities(surface, type, singularities);
    namespace cl = laplacian::connection;
    cl::ConnectionLaplacianOptions o;
    o.domain = cl::ConnectionDomain::Vertex;
    o.nSym   = 1;
    o.format = cl::ConnectionLaplacianFormat::Complex;
    cl::ConnectionLaplacian r =
        (type == geometry::ConnectionType::Trivial)
          ? [&] { DECOperators dec = embeddedDEC(surface); const std::unique_ptr<CholeskyFactor> exact = exactFormFactor(dec); return cl::assembleTrivialConnectionLaplacian(surface, singularities, dec, *exactFormFactor(dec), o); }()
          : cl::assembleConnectionLaplacian(surface, o);
    return std::move(r.K_complex);
}

Eigen::SparseMatrix<double> covariantLaplacian(const geometry::SurfaceArrays& surface, geometry::ConnectionType type, const std::map<int,double>& singularities,
                                               laplacian::connection::CovariantCoupling coupling) {
    geometry::checkSingularities(surface, type, singularities);
    namespace cl = laplacian::connection;
    Eigen::MatrixXcd g = connectionGrid(surface, type, singularities);   // realized active-gauge frame
    const Eigen::SparseMatrix<double> cotanL = cotanLaplacian(surface);                                // real cotan Laplacian
    // The Ambient covariant (the default 3D-frame view) is built purely from the frames + scalar weights and does NOT use the connection Laplacian K
    // (frame-conjugate of kron(I3, cotanL)). Only the Product coupling — which puts the intrinsic connection on the tangent block — needs K, so the
    // connection assembly (incl. the trivial-gauge Poisson solve) is skipped otherwise.
    if (coupling == cl::CovariantCoupling::Product) {
        const Eigen::SparseMatrix<std::complex<double>> K = vertexConnectionLaplacian(surface, type, singularities);
        return cl::assembleCovariantLaplacian(coupling, K, g, cotanL);
    }
    const Eigen::SparseMatrix<std::complex<double>> noK;  // unused by Ambient
    return cl::assembleCovariantLaplacian(coupling, noK, g, cotanL);
}

// τ arrives unmodified from the binding-layer dispatch (a literal or a single mxGetScalar), never from arithmetic, so the τ==0.0 / τ==1.0 fast paths use exact equality deliberately.
Eigen::SparseMatrix<double> diracFamily(const geometry::SurfaceArrays& surface, double tau) {
    // Negated form so NaN is rejected too.
    if (!(tau >= 0.0 && tau <= 1.0))
        throw Error(ErrorCode::InvalidInput, "dirac: tau must be in [0,1]", "Got tau=" + std::to_string(tau) + ".");
    const int N = surface.nV();
    Eigen::SparseMatrix<double> L4;
    if (tau < 1.0) {
        const Eigen::SparseMatrix<double> cotanL = cotanLaplacian(surface);    // the operator mesh
        std::vector<Eigen::Triplet<double>> T;
        T.reserve(static_cast<size_t>(cotanL.nonZeros()) * 4);
        for (int k = 0; k < cotanL.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(cotanL, k); it; ++it)
                for (int c = 0; c < 4; ++c)
                    T.emplace_back(4 * static_cast<int>(it.row()) + c, 4 * static_cast<int>(it.col()) + c, it.value());
        L4.resize(4 * N, 4 * N);
        L4.setFromTriplets(T.begin(), T.end());
    }
    if (tau == 0.0) { L4.makeCompressed(); return L4; }
    const Eigen::SparseMatrix<double> E = dirac::extrinsicBlock(surface);
    if (tau == 1.0) { Eigen::SparseMatrix<double> out = E; out.makeCompressed(); return out; }
    Eigen::SparseMatrix<double> L = (1.0 - tau) * L4 + tau * E;
    L.makeCompressed();
    return L;
}

Eigen::SparseMatrix<double> diracFaceFamily(const geometry::SurfaceArrays& surface, double tau) {
    if (!(tau >= 0.0 && tau <= 1.0))
        throw Error(ErrorCode::InvalidInput, "diracFace: tau must be in [0,1]", "Got tau=" + std::to_string(tau) + ".");
    const int Fn = surface.nF();
    Eigen::SparseMatrix<double> K4;
    if (tau < 1.0) {
        // K̃ = d₁⋆₁⁻¹d₁ᵀ lives on the INPUT's faces (the face Dirac Ẽ is embedded); d1t is an explicit temporary so the chain has no Eigen aliasing.
        const DECOperators dec = embeddedDEC(surface);
        Eigen::SparseMatrix<double> d1t = dec.d1.transpose();
        const Eigen::SparseMatrix<double> Ktilde = dec.d1 * dec.hodge1Inverse * d1t;
        std::vector<Eigen::Triplet<double>> T;
        T.reserve(static_cast<size_t>(Ktilde.nonZeros()) * 4);
        for (int k = 0; k < Ktilde.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(Ktilde, k); it; ++it)
                for (int c = 0; c < 4; ++c)
                    T.emplace_back(4 * static_cast<int>(it.row()) + c, 4 * static_cast<int>(it.col()) + c, it.value());
        K4.resize(4 * Fn, 4 * Fn);
        K4.setFromTriplets(T.begin(), T.end());
    }
    if (tau == 0.0) { K4.makeCompressed(); return K4; }
    const Eigen::SparseMatrix<double> E = dirac::extrinsicBlockFace(surface);
    if (tau == 1.0) { Eigen::SparseMatrix<double> out = E; out.makeCompressed(); return out; }
    Eigen::SparseMatrix<double> L = (1.0 - tau) * K4 + tau * E;
    L.makeCompressed();
    return L;
}

Eigen::MatrixXcd connectionGrid(const geometry::SurfaceArrays& surface, geometry::ConnectionType type, const std::map<int,double>& singularities) {
    Eigen::MatrixXcd g = geometry::vertexGrid(surface);    // LC / euclidean base frame: the signposts' gauge
    if (type != geometry::ConnectionType::Trivial) return g;
    DECOperators dec = embeddedDEC(surface); const std::unique_ptr<CholeskyFactor> exact = exactFormFactor(dec);
    connection::GaugeRotations gr = connection::integrateTrivialGaugeRotations(surface, dec, *exact, singularities);
    for (int v = 0; v < g.rows(); ++v) g.row(v) *= gr.vertex(v);  // combing rotation .* grid
    return g;
}

Eigen::MatrixXcd connectionFaceGrid(const geometry::SurfaceArrays& surface, geometry::ConnectionType type, const std::map<int,double>& singularities) {
    Eigen::MatrixXcd g = geometry::faceGrid(surface);
    if (type != geometry::ConnectionType::Trivial) return g;
    DECOperators dec = embeddedDEC(surface); const std::unique_ptr<CholeskyFactor> exact = exactFormFactor(dec);
    connection::GaugeRotations gr = connection::integrateTrivialGaugeRotations(surface, dec, *exact, singularities);
    for (int f = 0; f < g.rows(); ++f) g.row(f) *= gr.face(f);
    return g;
}

} // namespace nxr::operators
