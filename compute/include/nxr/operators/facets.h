#pragma once
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/facets.h"

namespace nxr::geometry::facet {

class ConnectionFacet {
public:
    ConnectionFacet(Manifold& m, ConnectionType type, std::map<int,double> singularities)
        : m_(m), type_(type), sing_(std::move(singularities)) {}
    ConnectionType type() const { return type_; }
    const std::map<int,double>& singularities() const { return sing_; }
    // Realized per-vertex frame in this gauge: [nV,3] complex c = e1 + i e2.
    // Levi-Civita/Euclidean: the raw vertex grid. Trivial: rotation .* grid —
    // the combed frame (real part == the trivial parallel field). See
    // GaugeRotations in compute.h for the convention.
    Eigen::MatrixXcd grid() const;
    // Realized per-face frame in this gauge: [nF,3] complex. Same combing
    // convention as grid(), via the dual-graph face rotations.
    Eigen::MatrixXcd faceGrid() const;
private:
    Manifold& m_;
    ConnectionType type_;
    std::map<int,double> sing_;
};

class OperatorsFacet {
public:
    explicit OperatorsFacet(Manifold& m) : m_(m) {}

    // Views hold Manifold& (not OperatorsFacet&) so a stored view
    // (auto v = m.operators().laplacian()) does not dangle when the
    // OperatorsFacet temporary expires. Manifold befriends OperatorsFacet,
    // and per CWG 45 its nested views inherit that access to the private
    // *Cached_() helpers.
    struct LaplacianView {
        Manifold& m;
        const Eigen::SparseMatrix<double>& cotan() const;   // real, intrinsic (directly from GC cotan cache)
        const Eigen::SparseMatrix<double>& graph() const;   // real, topological

        // Declared; implemented in Task C3.
        const Eigen::SparseMatrix<std::complex<double>>& connection() const;
        // Default coupling is Ambient — the full 3D-frame embedded covariant.
        // Product (intrinsic connection on the tangent block) is opt-in.
        const Eigen::SparseMatrix<double>& covariant(
            ops::laplacian::connection::CovariantCoupling coupling =
                ops::laplacian::connection::CovariantCoupling::Ambient) const;
    };

    // ── Task C2: dec / mass / hodge ──────────────────────────────────────────
    // Views hold Manifold& m (not OperatorsFacet&) — same C1 pattern;
    // a stored view (auto v = m.operators().mass()) does not dangle.

    struct MassView {
        Manifold& m;
        const Eigen::SparseMatrix<double>& mixedVoronoi() const;   // the default mass
        const Eigen::SparseMatrix<double>& barycentric()  const;   // A/3 per vertex (formerly "lumped")
        const Eigen::SparseMatrix<double>& galerkin()     const;
    };

    struct HodgeView {
        Manifold& m;
        const Eigen::SparseMatrix<double>& h0()    const;
        const Eigen::SparseMatrix<double>& h1()    const;
        const Eigen::SparseMatrix<double>& h2()    const;
        const Eigen::SparseMatrix<double>& h1inv() const;
    };

    // dec(): returns the full DECOperators bundle {d0,d1,hodge0..2,hodge1Inverse}.
    const ops::DECOperators& dec()  const;
    MassView  mass()  const { return MassView{m_}; }
    HodgeView hodge() const { return HodgeView{m_}; }

    LaplacianView laplacian() const { return LaplacianView{m_}; }

    // gradient3D(): the covariant gradient operator G (3E x 3N), cached. Artifact-free
    // covariant difference of a 3-vector cortical field (differential::covariantGradient).
    const Eigen::SparseMatrix<double>& gradient3D() const;

    // extrinsicWeitzenbock(): the ambient (ℝ³) extrinsic vector Weitzenböck
    // Δ₃ + W_ext [3V×3V], cached. World-frame component-major (c·N+v). The
    // imaginary (vector) block of the immersion squared Dirac 2·dirac(0.5);
    // ambient sibling of dirac(1). See differential::assembleExtrinsicWeitzenbock.
    const Eigen::SparseMatrix<double>& extrinsicWeitzenbock() const;

    // dirac(tau): the relative-Dirac family L(τ) = (1−τ)(cotanL⊗I₄) + τ·E, a
    // [4V×4V] real symmetric sparse matrix, returned BY VALUE (τ-dependent blend).
    // τ=0 ⇒ block cotan-Laplacian; τ=1 ⇒ pure relative Dirac E. τ ∈ [0,1].
    Eigen::SparseMatrix<double> dirac(double tau) const;

    // diracFace(tau): the FACE-domain (dual) relative-Dirac family
    // L̃(τ) = (1−τ)(K̃⊗I₄) + τ·Ẽ, a [4F×4F] real symmetric sparse matrix, by value.
    // τ=0 ⇒ DEC 2-form Laplacian ⊗ I₄; τ=1 ⇒ pure extrinsic face Dirac Ẽ. τ ∈ [0,1].
    Eigen::SparseMatrix<double> diracFace(double tau) const;

    // diracD(): the first-order (rectangular) extrinsic Dirac operator D [4F×4V],
    // cached, returned by const-ref (τ-free, geometry-only). Its area-weighted
    // Galerkin square DᵀW_F D is the τ=1 block of dirac(τ).
    const Eigen::SparseMatrix<double>& diracD() const;

    // diracFaceD(): the first-order FACE-domain (dual) Dirac operator D̃ [4V×4F],
    // cached, by const-ref. Its Galerkin square D̃ᵀW_V D̃ is the τ=1 block of
    // diracFace(τ). Closed-mesh v1: throws on an open boundary (as diracFace).
    const Eigen::SparseMatrix<double>& diracFaceD() const;

    // diracIntrinsicD(): the first-order INTRINSIC Dirac operator D_int [4F×4V],
    // cached, by const-ref (τ-free, geometry-only). Built from the immersion (edge
    // vectors) rather than the Gauss map; its area-weighted Galerkin square is the
    // intrinsic Dirac² (the spin-connection root, scalar part = cotan Laplacian).
    const Eigen::SparseMatrix<double>& diracIntrinsicD() const;

    // diracFaceIntrinsicD(): the first-order INTRINSIC face Dirac D̃_int [4V×4F],
    // cached, by const-ref. The centroid (immersion) dual of diracFaceD (face centroids
    // instead of the Gauss map), mirroring diracIntrinsicD on the vertex side. Closed-mesh v1.
    const Eigen::SparseMatrix<double>& diracFaceIntrinsicD() const;

    // connectionGradient(nSym): the first-order covariant gradient d^∇ [E×V] complex,
    // built in the active gauge; (d^∇)ᴴ⋆₁d^∇ == the connection Laplacian. Cached per nSym.
    const Eigen::SparseMatrix<std::complex<double>>& connectionGradient(int nSym = 1) const;

    // gradFace(): barycentric dual-mesh gradient of a per-face scalar [3F×F], cached.
    // Green–Gauss / DEC; annihilates constants, output tangent to each face. The dual
    // of the vertex FEM gradient. Closed-mesh v1.
    const Eigen::SparseMatrix<double>& gradFace() const;
    // lapFace(): face Laplacian K̃ = gradFaceᵀ ⋆_F gradFace [F×F], cached. Built from
    // the same gradFace (never drift); symmetric PSD; kernel = constants (genus-0).
    const Eigen::SparseMatrix<double>& lapFace() const;

private:
    Manifold& m_;
};


} // namespace nxr::geometry::facet
