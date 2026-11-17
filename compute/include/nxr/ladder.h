// nxr::ladder — the dyadic TILE LADDER of a surface and the tile summaries (Step 3 of the cortical coordinate system, docs/audits/2026-11-11-cortical-coordinates.md).
//
// A line port of the Cortical Flow Studio's `engine/ladder.ts` (Weyl pairing, `vertexAreas`, `onePiece`, `surfaceLadder`), which nxr (`nxr.geom.ladder`, `nxr.geom.trees`) and Rheome
// (`+geom/surfaceladder.m`) reproduce bit for bit: each tile is halved by EQUAL MASS across its geodesic diameter (Dijkstra on the input edge graph, two sweeps, a stable sort by d_a - d_b,
// the cut at half the mass), every tile ONE PIECE, mass the barycentric lumped vertex area. Level l has 2^l tiles; tile t of level l has children 2t, 2t+1; level l of a vertex is code >> (D - l).
// The arithmetic uses Math.hypot as V8 computes it (jsHypot), so equal-length paths tie as they do in the app; the source is compiled without fused multiply-add.
//
//   weylLambda / weylModes / wavelengthOf     level l (m = 2^l tiles) pairs with lambda(m) = 4 pi m / A, the wavelength 2 pi / sqrt(lambda), tau = 1 / lambda
//   surfaceLadder(mesh, params)               ONE connected piece: its levels, the tile of every vertex per level, tile areas and vertex counts (`SurfaceLadder`)
//   subjectTree(positions, faces)             the SUBJECT tree of nxr (`nxr.geom.trees.subject_tree`): the ladder per connected piece (a hemisphere), a vertex's code at depth D is h 2^D + t
//   tileCentres / tileAdjacency / tileAreas   a tile's centre (the vertex nearest its mass centroid), the tiles joined by a mesh edge, a tile's mass
//
// SUMMARIES over tiles. A scalar map: Sum a x, Sum a x^2, Sum a, count, min, max (nxr `measure.scalars.map_stats`; mean = s1 / w). A VECTOR map is summed in a frame, never as raw 3-D vectors (on folded
// cortex the 3-D vectors of one tile point every way and cancel):
//   FRAME form        (nxr `measure.vectors.vector_stats`) Sum a c1, a c2, a cn, second moments, in the canonical frames (nxr/frames.h); mergeable, rolls up like a scalar
//   LEVI-CIVITA form  (nxr `levi_civita`, `lc_resultant`) per level: each tile's centre has e1 carried (ChartKit::transported) to every vertex of the tile, the frame angle rho it arrives with is the
//                     vertex's offset from the centre's frame, and the tile's resultant is Sum a (c1 + i c2) e^{-i rho}: the field carried to the centre. The two differ by the curvature the tile encloses.
//                     The connection between centres, Omega[a, b], is the same solves read at the other centres.
#pragma once
#include <Eigen/Dense>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "nxr/chart.h"
#include "nxr/selection/dyadic.h"
#include "nxr/frames.h"

namespace nxr::ladder {

using frames::Mat3;

// ── the partition (moved to the selection module, nxr/selection/dyadic.h; these names stay visible here) ─────────────
using namespace nxr::selection::dyadic;

// ── tiles ────────────────────────────────────────────────────────────────────────────────
/** A tile's mass (area) per node of `nodes` entries. */
std::vector<double> tileAreas(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area);
/** Per node, the (non-singular) member vertex nearest the area-weighted centroid (the Studio's `tileCentres`, nxr `node_centres`); -1 for an empty node. */
std::vector<int> tileCentres(const double* positions, const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const std::vector<char>& singular = {});
/** The pairs (a < b) of nodes joined by a mesh edge (nxr `tile_adjacency`). */
std::vector<std::pair<int, int>> tileAdjacency(const int32_t* faces, int nF, const std::vector<int32_t>& node);

// ── scalar summaries (nxr measure.scalars.map_stats) ─────────────────────────────────────
struct ScalarSummary {
    Eigen::VectorXd w, s1, s2, n, min, max;     // per node: Sum a (finite x), Sum a x, Sum a x^2, count, min, max
    Eigen::VectorXd mean() const;               // s1 / w (NaN where w = 0)
};
ScalarSummary scalarSummary(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const double* x);

// ── vector summaries ─────────────────────────────────────────────────────────────────────
/** FRAME form (nxr measure.vectors.vector_stats): the components c1 = v.e1, c2 = v.e2, cn = v.n of a [V x 3] field, summed per node with area weights; singular / non-finite vertices out. */
struct FrameSummary { Eigen::VectorXd w, c1, c2, cn, c11, c22, c12, mag; };
FrameSummary frameSummary(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const Mat3& e1, const Mat3& e2, const Mat3& normal, const Mat3& v);

/** LEVI-CIVITA connection of one level: the centres, rho (the frame angle at each vertex of e1(centre of its node) carried there; NaN off the node's centre component), Omega[a, b]. */
struct LeviCivita {
    std::vector<int> centres;                   // [nodes] (-1: empty)
    Eigen::VectorXd rho;                        // [V]
    Eigen::MatrixXd omega;                      // [nodes x nodes] (NaN across pieces and where a centre is missing)
};
/** `kit` the surface's charts; `e1`, `e2` the canonical frames (NaN at singular vertices); `gauge` frames::gaugeAngle(kit, e1). The transport is chart::ChartKit::transported (geometry-central's vector heat method, in its gauge), calibrated to the identity at its source; rho is the angle of the transported e1 seen from each vertex's own e1, so it does not depend on the gauge. */
LeviCivita leviCivita(chart::ChartKit& kit, const Mat3& e1, const Mat3& e2, const Eigen::VectorXd& gauge, const std::vector<int32_t>& node, int nodes, const std::vector<int>& centres);
/** Omega made ANTISYMMETRIC (nxr `joint.antisymmetric`, the same formula since nxr #68): the round trip a -> b -> a misses the identity by d = wrap(Omega_ab + Omega_ba), and half of d is
 *  taken from each direction, so Omega_ba = -Omega_ab. (Halving an angle on the circle, angle(e^{i Omega_ab} e^{-i Omega_ba}) / 2, is ambiguous by pi: it turned every |Omega| > 90 degrees by 180.) NaN stays NaN. */
Eigen::MatrixXd antisymmetric(const Eigen::MatrixXd& omega);
/** Per node, Sum a (c1 + i c2) e^{-i rho}: the field carried to the centre (complex, in the centre's frame). */
Eigen::VectorXcd lcResultant(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const Mat3& e1, const Mat3& e2, const Eigen::VectorXd& rho, const Mat3& v,
                             Eigen::VectorXd* weight = nullptr, Eigen::VectorXd* magnitude = nullptr);   // weight: Sum a, magnitude: Sum a |(c1, c2)| over the vertices counted

/** Both forms of one vector map at one level, labelled. */
struct VectorSummary {
    std::string gauge = "canonical frames";
    FrameSummary frame;                         // reduction "frame"
    Eigen::VectorXcd transported;               // reduction "levi-civita"
    Eigen::VectorXd weight;                     // Sum a over the vertices the transported form counts
    Eigen::VectorXd resultantLength;            // |transported| / Sum a |(c1, c2)| (1: the vectors agree after transport; NaN for an empty node)
};
/** Both forms at one level: `rho` from leviCivita(...).rho. */
VectorSummary vectorSummary(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const Mat3& e1, const Mat3& e2, const Mat3& normal, const Eigen::VectorXd& rho, const Mat3& v);

}  // namespace nxr::ladder
