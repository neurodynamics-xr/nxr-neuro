// nxr::connectome — the STRUCTURAL CONNECTOME ON THE CORTEX, as pure functions (docs/specs/2026-11-16-structural-connectome.md). Matrices in, matrices out; no file,
// no subject, no knowledge of where the fibers came from. Precision follows the input: every function is for float32 and float64 and never casts.
//
// The pointwise object is each fiber's pair of END VERTICES on an end surface (fiberEnds, nearestVertices, withinGate; for a template's fibers carried to a subject
// through registrationMatch and carryEnds). Counts are measurements over a PARTITION of the vertices: smoothMemberships pushes the smoothing onto the memberships and
// partitionCounts sums over the fibers (the vertex matrix is never formed). The vertex connectome (endConnectome, connectionWeights, pruneBelow, componentsOf,
// largestComponent) is the finest partition, formed only on request.
//
// Words, never single letters: "fiber counts" are the counts of fibers; "connection weights" are the smoothed (and pruned) result. The helpers the reference takes from
// Brainstorm (GPLv3) are written here from the mathematics: Shepard weights (Franke & Nielson 1980, nxr/sphere_registration.h), the nearest-vertex search
// (nxr/nearest.h), connected components (a union-find), Eigen sparse algebra.
//
// ORDER OF ENDS. A bundle of nF fibers has 2 nF ends in BLOCK order: the first end of fiber f is row f, its second end row nF + f (fiberEnds). `interleaveEnds`
// converts to the stored order (fiber f's ends at 2f and 2f + 1) and back.
#pragma once
#include <Eigen/Sparse>
#include <cstdint>
#include <vector>

#include "nxr/sphere_registration.h"

namespace nxr::connectome {

template <class S>
using Sparse = Eigen::SparseMatrix<S>;

// ── the ends ──────────────────────────────────────────────────────────────────────────────────────────────────
/** The first and the last point of each fiber, [2nF x 3] (row f: first end; row nF + f: last end). `points` is [nF·nP x 3], fiber-major; nP >= 2. */
template <class S>
std::vector<S> fiberEnds(const std::vector<S>& points, std::int64_t pointsPerFiber);

/** Block order [first ends…, last ends…] to the stored order (fiber f at 2f and 2f + 1), and back. */
template <class T>
std::vector<T> interleaveEnds(const std::vector<T>& block);
template <class T>
std::vector<T> blockEnds(const std::vector<T>& interleaved);

/** THE GATE: per fiber, 1 when both ends are within `maxDistance` of the end surface (<=), 0 when either is farther. `endDistance` is [2nF] in block order, in the unit of
 *  `maxDistance`. A negative or NaN `maxDistance` is refused; use `keepAll` for no gate. */
template <class S>
std::vector<std::uint8_t> withinGate(const std::vector<S>& endDistance, S maxDistance);
inline std::vector<std::uint8_t> keepAll(std::int64_t nFibers) { return std::vector<std::uint8_t>(static_cast<std::size_t>(nFibers), 1); }

// ── template ends on the subject (registration of the spheres) ───────────────────────────────────────────────────
struct MatchReport {
  std::int64_t empty_columns = 0;   // template vertices no subject vertex takes weight from: each is sent to its nearest subject vertex on the sphere
};
/** For each TEMPLATE vertex, the subject vertex that takes the most weight from it in `w` [subject x template] (the largest weight in the vertex's column; the lowest
 *  subject index on a tie); -1 for an empty column. */
template <class S>
std::vector<std::int32_t> strongestDestination(const nxr::registration::Csr<S>& w);

/** The match template vertex -> subject vertex: the Shepard resampling operator between the registered spheres (K nearest template vertices of each subject vertex,
 *  Franke & Nielson weights), `strongestDestination`, and, FOR A TEMPLATE VERTEX WITH NO SUBJECT NEIGHBOUR (an empty column), its NEAREST SUBJECT VERTEX ON THE SPHERE (the
 *  unit-normalised sphere positions, within the vertex's own piece), never subject vertex 0 (the reference's defect, fixed: decided 2026-11-16). Positions are [n x 3]. */
template <class S>
std::vector<std::int32_t> registrationMatch(const std::vector<S>& templateSphere, const std::vector<nxr::registration::Piece>& templatePieces, const std::vector<S>& subjectSphere,
                                            const std::vector<nxr::registration::Piece>& subjectPieces, int neighbours = 8, MatchReport* report = nullptr);

/** The subject vertex of each end: match[endVertex]. std::invalid_argument for a vertex outside the match. */
std::vector<std::int32_t> carryEnds(const std::vector<std::int32_t>& endVertex, const std::vector<std::int32_t>& match);

// ── counting over a partition ─────────────────────────────────────────────────────────────────────────────────
/** The mesh's smoothing step: row u holds u and its neighbours with equal weights summing to 1 (the random-walk normalisation of the adjacency with self-loops). */
template <class S>
Sparse<S> smoothingStep(const std::vector<std::int32_t>& faces, std::int64_t nVertices);
/** The step raised to `hops` by sparse products (0: the identity). */
template <class S>
Sparse<S> smoothingOperator(const std::vector<std::int32_t>& faces, std::int64_t nVertices, int hops);

/** One-hot memberships [V x R] from a label per vertex (a region's ordinal 0 … R-1; -1 in no region). std::invalid_argument for a label >= R. */
template <class S>
Sparse<S> memberships(const std::vector<std::int32_t>& label, std::int64_t nRegions);

/** The memberships pushed through the smoothing: (stepᵀ)^hops · memberships [V x R], the ADJOINT of the smoothing applied to the one-hot memberships, hop by hop: the
 *  smoothed membership of a vertex in a region is the weight with which the region's smoothing receives it (a vertex's end is spread over its neighbours, so the part of it
 *  that lands in the region is what counts). This is the exact linear push of the smoothing onto the partition: counting with it equals summing the smoothed vertex matrix
 *  P · counts · Pᵀ over the region pairs (the rows of P sum to 1, its columns do not on an irregular mesh). No vertex matrix is formed. */
template <class S>
Sparse<S> smoothMemberships(const Sparse<S>& step, const Sparse<S>& memberships, int hops);

/** THE FIBER COUNTS OVER A PARTITION [R x R]: for each region pair, the sum over the kept fibers of (membership of the first end in a)(membership of the second end in b)
 *  plus the same with the ends swapped, with `membershipOfVertex` the (smoothed) memberships [V x R]. Symmetric. A fiber with both ends in one region adds 2 at that
 *  region's diagonal (both orders, as the vertex matrix has them): the within-region count. `endVertex` is [2nF] in block order, `keep` is [nF] (empty: every fiber). */
template <class S>
Sparse<S> partitionCounts(const std::vector<std::int32_t>& endVertex, const std::vector<std::uint8_t>& keep, const Sparse<S>& membershipOfVertex);

// ── the materialised vertex connectome ───────────────────────────────────────────────────────────────────────────
/** The sparse symmetric matrix of fiber counts [V x V]: each kept fiber adds 1 at its two end vertices, both orders; a fiber with both ends on one vertex leaves no entry. */
template <class S>
Sparse<S> endConnectome(const std::vector<std::int32_t>& endVertex, const std::vector<std::uint8_t>& keep, std::int64_t nVertices);
/** The connection weights: P · counts · Pᵀ with P = smoothingOperator(hops), the diagonal removed. */
template <class S>
Sparse<S> connectionWeights(const Sparse<S>& counts, const std::vector<std::int32_t>& faces, int hops);
/** The largest entry (0 for none). */
template <class S>
S strongest(const Sparse<S>& weights);
/** Entries below `fraction` x the strongest entry are removed from the pattern (nothing is removed from a matrix with no entry; fraction 0 removes nothing). */
template <class S>
Sparse<S> pruneBelow(const Sparse<S>& weights, S fraction);
/** The connected components of the graph whose edges are the non-zero entries, over all V vertices (an isolated vertex is its own component). Components are numbered by
 *  their lowest vertex in order of first appearance: label[v] is the ordinal of v's component. */
template <class S>
std::vector<std::int32_t> componentsOf(const Sparse<S>& weights);
/** The vertices (ascending) of the biggest component (the lowest-numbered on a tie). `sizes` (optional) receives each component's size. */
std::vector<std::int32_t> largestComponent(const std::vector<std::int32_t>& label, std::vector<std::int64_t>* sizes = nullptr);

}  // namespace nxr::connectome
