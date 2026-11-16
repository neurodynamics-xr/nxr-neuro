// THE ONE INTRINSIC DELAUNAY CONSTRUCTION.
//
// A line port of the Cortical Flow Studio's `intrinsicDelaunay.ts` (nxr-cortical-flow
// studio/src/engine/toolbox/differential/intrinsicDelaunay.ts), the reference; nxr's `nxr.operators.intrinsic_delaunay` and Rheome's
// `+operators/intrinsic_delaunay.m` are ports of the same file. Every operator of nxr-compute that works on an intrinsic mesh
// (cotan Laplacian, masses, DEC, connection Laplacian, Hodge–Dirac) is built on THIS mesh and no other.
//
// What the construction does (each step names the Studio function it matches):
//   makeMesh      `intrinsicMesh`   the halfedge mesh (halfedge h = 3f + k runs from F[h] to F[3f + (k+1)%3]) and the edge lengths
//   mollify       `mollify`         Sharp & Crane 2020: δ = `mollify` × the mean edge length; ε = max over corners of (l_C − l_A − l_B + δ),
//                                   starting at 0; if ε > 0 it is added to EVERY edge
//   addSignposts  `addSignposts`    each halfedge's angle at its tail, from each vertex's first outgoing halfedge in face order (a boundary
//                                   vertex: its twinless halfedge), carried through the flips modulo the cone angle
//   flipToDelaunay `flipToDelaunay` the mixed-Voronoi mass (Meyer et al. 2003) of the mollified INPUT is recorded BEFORE the first flip; a FIFO
//                                   queue of every edge; an edge whose cotan weight is below −`flipEps` is flipped and its four neighbours queued
//   flip          `flip`            the reference's refusals (boundary; a degree-1 vertex; a new length not finite or ≤ 0; an outer twin that
//                                   is the flipped edge itself); self-edges are allowed; the new length from the reference's planar layout
//
// The two numerical constants are NAMED, RECORDED options with the names the Studio, nxr and Rheome share: `mollify` (relative δ,
// default 1e-6) and `flipEps` (flip tolerance, default 1e-6). What was told and done is on `Mesh::record`.
//
// Pure: arrays in, arrays out. NaN handling follows JavaScript's Math.min / Math.max (a NaN propagates), as the reference does.
#pragma once

#include <Eigen/Sparse>

#include <complex>
#include <cstdint>
#include <vector>

namespace nxr::manifold::delaunay {

/** The two numerical choices of the construction (named alike in the Studio, nxr and Rheome). */
struct DelaunayOptions {
    double mollify  = 1e-6;   // the mollification margin δ relative to the mean edge length (0 = no margin; ε is still added where a triangle is invalid)
    double flipEps  = 1e-6;   // an edge is flipped while its cotangent weight is below −flipEps
};

/** What the construction was told and did. */
struct DelaunayRecord {
    double mollify = 0.0;     // the option used
    double epsilon = 0.0;     // ε, the length added to every edge by mollify (0 = nothing needed)
    double flipEps = 0.0;     // the option used
    int    flips   = 0;       // edge flips made
};

/** The halfedge mesh: halfedge h = 3f + k runs from `F[h]` to `F[3f + (k+1)%3]`. Flips rewrite F, twin, edgeOf, edgeHe and length in place;
 *  the vertices never change. */
struct Mesh {
    int nV = 0, nF = 0;
    std::vector<int32_t> F;          // [3 nF] the faces' vertices (after flips, the INTRINSIC triangulation)
    std::vector<int32_t> twin;       // [3 nF] each halfedge's twin, −1 on the boundary
    std::vector<int32_t> edgeOf;     // [3 nF] each halfedge's edge
    std::vector<int32_t> edgeHe;     // [nE]   one halfedge of each edge (the edge's orientation: tail → tip)
    std::vector<uint8_t> boundary;   // [nV]   1 on a vertex with a twinless halfedge
    std::vector<double>  length;     // [nE]   edge lengths (input units)

    // the signposts, once added (`addSignposts`)
    bool hasSignposts = false;
    std::vector<double>  angle;      // [3 nF] each halfedge's angle at its tail, counter-clockwise from the vertex's reference halfedge, in [0, angleSum)
    std::vector<double>  angleSum;   // [nV]   each vertex's cone angle Θ_v
    std::vector<int32_t> ref;        // [nV]   each vertex's reference neighbour: the head of its angle-0 halfedge (an INPUT edge)

    // the mixed-Voronoi mass of the (mollified) INPUT triangulation, recorded before the first flip; empty until then
    std::vector<double>  mass;       // [nV]

    DelaunayRecord record;

    int nE() const { return static_cast<int>(length.size()); }
};

inline int nextHe(int h) { return h - (h % 3) + ((h % 3) + 1) % 3; }

/** `intrinsicMesh`: the halfedge mesh of an oriented manifold triangle mesh (closed or with a boundary), lengths from the positions
 *  (`positions` [3 nV] row-major, `faces` [3 nF]). Throws core::Error on a non-manifold or inconsistently oriented edge. */
Mesh makeMesh(const double* positions, int nV, const int32_t* faces, int nF);

/** Heron's area from lengths, clamped at 0 (geometry-central's `triangleArea`). */
double triangleArea(double a, double b, double c);
/** The corner angle at the tail of halfedge h (law of cosines). */
double cornerAngle(const Mesh& m, int h);
/** ½ cot of the angle opposite halfedge h in its face. */
double halfCot(const Mesh& m, int h);
/** An edge's cotangent weight ½(cot α + cot β); a boundary edge's its one triangle's ½ cot α. */
double edgeWeight(const Mesh& m, int e);

/** `mollify`: add ε to every length (Sharp & Crane 2020). Records mollify and ε. Returns ε. */
double mollify(Mesh& m, const DelaunayOptions& o = {});
/** `addSignposts`: call after `mollify`, BEFORE `flipToDelaunay`, so every reference is an input edge. */
void addSignposts(Mesh& m);
/** `flip`: flip edge e. Returns false where the reference refuses. */
bool flip(Mesh& m, int e);
/** `flipToDelaunay`: records the mass first; returns the number of flips and records flipEps and flips. */
int flipToDelaunay(Mesh& m, const DelaunayOptions& o = {});
/** `lumpedMass` of the Studio (named for what it is, NOT geometry-central's barycentric "lumped"): each vertex's MIXED VORONOI area (Meyer,
 *  Desbrun, Schröder & Barr 2003) of the triangulation as it is; `Mesh::mass` holds the INPUT's once recorded. */
std::vector<double> mixedVoronoiMass(const Mesh& m);
/** Barycentric mass: a third of each triangle's area, on the triangulation as it is. */
std::vector<double> barycentricMass(const Mesh& m);
/** The number of edges whose cotangent weight is below −eps. */
int nonDelaunayEdges(const Mesh& m, double eps = 1e-6);

/** THE preparation every operator starts from: halfedge mesh → mollify → signposts → mixed-Voronoi mass of the mollified input recorded →
 *  flip to Delaunay. With `intrinsic = false` (the named embedded option) the mesh is left as it is: no mollification, no flips; the
 *  signposts and the mass are of the input. */
Mesh prepare(const double* positions, int nV, const int32_t* faces, int nF, const DelaunayOptions& o = {}, bool intrinsic = true);

// ── operators of a prepared mesh ──

/** The cotangent Laplacian L (positive semidefinite): L_ij = −w_ij, L_ii = Σ w_ij; a self-edge contributes nothing, parallel edges add. */
Eigen::SparseMatrix<double> cotanLaplacian(const Mesh& m);
/** Per-edge cotangent weights ⋆1 (raw, unfloored), [nE]. */
Eigen::VectorXd edgeCotanWeights(const Mesh& m);
/** Per-face areas, [nF]. */
Eigen::VectorXd faceAreas(const Mesh& m);
/** The consistent (Galerkin) P1 mass of the triangulation as it is. */
Eigen::SparseMatrix<double> galerkinMass(const Mesh& m);
/** A halfedge's RESCALED signpost angle ψ = coneTarget × angle / angleSum (coneTarget: 2π, or on the boundary the vertex's own angle sum). */
double rescaledAngle(const Mesh& m, int h);
/** The rescaled angle of the edge of halfedge h at its HEAD, pointing back to its tail. */
double reverseAngle(const Mesh& m, int h);
/** The vertex connection Laplacian on the n-RoSy bundle, complex Hermitian [nV×nV] (the Studio's `connectionLaplacian`, complex storage). */
Eigen::SparseMatrix<std::complex<double>> connectionLaplacian(const Mesh& m, int nSym = 1);

/** The DEC of the mesh: d0 [E×V] (edge orientation tail → tip), d1 [F×E], ⋆1 the cotan weights, ⋆2 = 1/area; ⋆0 is the caller's mass. */
struct Dec {
    Eigen::SparseMatrix<double> d0, d1;
    Eigen::VectorXd star1, star2;
};
Dec dec(const Mesh& m);

}  // namespace nxr::manifold::delaunay
