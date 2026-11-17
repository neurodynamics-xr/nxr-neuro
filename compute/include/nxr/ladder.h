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
//   tileCentres / tileAdjacency   a tile's centre (the vertex nearest its mass centroid), the tiles joined by a mesh edge (a tile's mass and every summary of a map over tiles is a MEASURE:
//                     `nxr::measurement::reduce` over the dyadic grid's members, spec docs/specs/2026-11-17-consolidation.md)
//
// THE LEVI-CIVITA FORM (nxr `levi_civita`, `lc_resultant`) per level: each tile's centre has e1 carried (ChartKit::transported) to every vertex of the tile, the frame angle rho it arrives with is the
//                   vertex's offset from the centre's frame, and a vector map is brought to the centre as (c1 + i c2) e^{-i rho} (`carriedToCentre`, pointwise); the tile's resultant is its SUM over the
//                   tile's vertices (measure `sum` of `reduce`, weighted by the vertex areas). The connection between centres, Omega[a, b], is the same solves read at the other centres: a value over PAIRS of
//                   tiles (a measurement over the product of the level with itself, the measure `levi_civita`).
// Summaries over tiles (a scalar map's sums and extremes, a vector map's frame form, the resultant) are measures, read by `nxr::measurement::reduce`: nothing here reduces.
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
using namespace nxr::selection::bisection;

// ── tiles ────────────────────────────────────────────────────────────────────────────────
/** Per node, the (non-singular) member vertex nearest the area-weighted centroid (the Studio's `tileCentres`, nxr `node_centres`); -1 for an empty node. */
std::vector<int> tileCentres(const double* positions, const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const std::vector<char>& singular = {});
/** The pairs (a < b) of nodes joined by a mesh edge (nxr `tile_adjacency`). */
std::vector<std::pair<int, int>> tileAdjacency(const int32_t* faces, int nF, const std::vector<int32_t>& node);

// ── the Levi-Civita connection ─────────────────────────────────────────────────────────
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
/** Per vertex, a vector map `v` [V x 3] carried to its tile centre: (c1 + i c2) e^{-i rho}, c1 = v.e1, c2 = v.e2 (complex, in the centre's frame; the field is real in 3-D and complex in the frame). A vertex with no frame or no
 *  transport (a non-finite rho, c1 or c2) is NaN and 0 in `finite` (when asked): the caller leaves it out of the tile's members, as the sum of the tile's resultant always did. The tile's resultant is the SUM of these over
 *  its vertices (`nxr::measurement::reduce`, area weights). */
Eigen::VectorXcd carriedToCentre(const Mat3& e1, const Mat3& e2, const Eigen::VectorXd& rho, const Mat3& v, std::vector<char>* finite = nullptr);

}  // namespace nxr::ladder
