// nxr::selection::bisection — THE DYADIC EQUAL-AREA PARTITION OF A SURFACE: a line port of the Cortical Flow Studio's `engine/ladder.ts` (`surfaceLadder`,
// `vertexAreas`, `onePiece`, the Weyl pairing), which nxr (`nxr.geom.ladder`, `nxr.geom.trees`) and Rheome (`+geom/surfaceladder.m`) reproduce bit for bit.
//
// Each tile is halved by EQUAL MASS across its geodesic diameter (Dijkstra on the input edge graph, two sweeps, a stable sort by d_a - d_b, the cut at half
// the mass), every tile ONE PIECE, mass the barycentric lumped vertex area. Level l has 2^l tiles a piece; tile t of level l has children 2t, 2t+1; the tile
// of a vertex at level l is code >> (D - l). Weyl's law pairs level l (m = 2^l tiles) with the eigenvalue lambda(m) = 4 pi m / A. The arithmetic uses
// Math.hypot as V8 computes it (jsHypot) and is compiled without fused multiply-add, so the tiles are the Studio's. The public face is
// nxr/selection/partition.h (`partition(surface, Dyadic)`); these are its parts.
#pragma once
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace nxr::selection::bisection {

// ── Weyl ─────────────────────────────────────────────────────────────────────────────────
double weylLambda(double modes, double area);      // 4 pi m / A
double weylModes(double lambda, double area);      // A lambda / 4 pi
double wavelengthOf(double lambda);                // 2 pi / sqrt(lambda); Infinity at 0

/** Math.hypot of three numbers as V8 computes it (scaled by the largest, Kahan sum of squares). */
double jsHypot(double a, double b, double c);

// ── the mesh ─────────────────────────────────────────────────────────────────────────────
struct Mesh {
    std::vector<double> positions;      // [n x 3] row-major
    std::vector<int32_t> faces;         // [f x 3]
    int n = 0;
};
/** The vertices [from, from + n) and the faces wholly inside them, re-indexed from zero (`componentMesh`). */
Mesh componentMesh(const double* positions, const int32_t* faces, int nF, int from, int n);
/** The barycentric lumped vertex areas: a third of each incident triangle; they total the surface area. */
std::vector<double> vertexAreas(const Mesh& mesh);

/** ONE PIECE (decision 98b0f3df): the children of `parity` keep their heaviest connected piece (ties: the one holding the lowest vertex), the rest goes to the sibling (code ^ 1). */
std::vector<int32_t> onePiece(const std::vector<int32_t>& lab, const std::vector<double>& mass, const std::vector<int32_t>& to, const std::vector<int32_t>& start, int parity);

// ── the ladder ───────────────────────────────────────────────────────────────────────────
struct SurfaceLadderParams {
    int maxTiles = 0;                   // a ceiling on the finest level, in tiles (0: 2^floor(log2 n)); the ladder stops there OR at the finest level whose tiles all hold a vertex
    int exactDiameterUpTo = 1024;       // the geodesic diameter is measured at levels of at most this many tiles; above it the disc estimate 2 sqrt(area / pi)
    std::vector<double> eigenvalues;    // the stored eigenvalues ascending, for lambdaStored
};
struct Range { double min = 0, mean = 0, max = 0, cv = 0; };
struct SurfaceLevel {
    int level = 0, tiles = 1;
    Range verticesPerTile, areaPerTile; // areaPerTile in m^2, cv the spread that makes the tiling equal-area
    double diameter = 0; bool diameterMeasured = false;
    double lambdaMin = 0, lambdaMax = 0;   // the band this level ADDS, by Weyl
    double wavelength = 0;              // 2 pi / sqrt(lambdaMax)
    double diameterInWavelengths = 0;
    double tau = 0;                     // 1 / lambdaMax
    int modes = 1;
    double lambdaStored = std::numeric_limits<double>::quiet_NaN();
    int finestTiles = 1;                // how many of the finest level's tiles one tile of this level contains
    double resolutionSpace = 0, resolutionLambda = 0;
};
struct SurfaceLadder {
    double area = 0;
    int nVertices = 0;
    std::vector<SurfaceLevel> levels;
    std::vector<std::vector<int32_t>> labels;       // per level: the tile of every vertex
    std::vector<std::vector<double>> tileAreas;     // per level: each tile's area
    std::vector<std::vector<int32_t>> tileVertices; // per level: each tile's vertex count
    std::vector<double> mass;                       // the vertex areas
};
/** The ladder of ONE connected piece (`surfaceLadder`). Throws on an empty mesh. */
SurfaceLadder surfaceLadder(const Mesh& mesh, const SurfaceLadderParams& params = {});

// ── the tree over a whole surface (nxr geom.trees.subject_tree) ─────────────────────────
struct Tree {
    int depth = 0;                      // D: codes are h 2^D + t
    int finest = 0;                     // the deepest level the ladder reached on every piece
    int slots = 2;                      // hemisphere slots (nxr: 2); the leaves are slots 2^depth
    std::vector<int32_t> codes;         // [nVertices] the tile LABEL of every vertex (h 2^D + t), not a member number: partition.h's `dyadic` numbers the members
    std::vector<std::pair<int, int>> runs;   // (start, length) of every connected piece
    std::vector<double> pieceArea;
    int nLeaves() const { return slots << depth; }
    int nodes(int level) const { return slots << level; }
    /** the node of every vertex at `level` <= depth: code >> (depth - level) */
    std::vector<int32_t> at(int level) const;
};
/** The connected pieces as contiguous vertex runs (a hemisphere is one block of vertices); throws if a piece is not contiguous. */
std::vector<std::pair<int, int>> pieceRuns(const int32_t* faces, int nF, int nV);
/** The SUBJECT tree: the ladder per connected piece, at `depth` (< 0: the finest level every piece reaches). */
Tree subjectTree(const double* positions, int nV, const int32_t* faces, int nF, int depth = -1);

// ── the group tree (nxr geom.trees Template.tree_for; D127/D128) ─────────────────────────
/** A subject's vertices on the GROUP tree: each vertex takes the code of the TEMPLATE vertex nearest to it on the registered sphere, hemisphere (piece) by piece.
 *  `tplSphere` [nT x 3] and `subSphere` [nV x 3] row-major registered spheres (normalised here); `tplCodes` [nT] the template's tile codes at its `finest` level, `h 2^finest + t` (h the piece);
 *  `tplRuns`, `subRuns` the (start, length) of each piece, in the same order (left, right). The code at `depth` (0 <= depth <= finest) is `tplCode >> (finest - depth)`.
 *  A pure function of arrays: where the template came from is the caller's. Throws std::invalid_argument on a piece-count mismatch or a depth outside 0..finest.
 *  `matchMedian` / `matchMax` (optional): the median and maximum chord distance to the matched template vertex (unit sphere). Ties go to the lowest template index. */
Tree groupTreeFor(const double* tplSphere, const int32_t* tplCodes, const std::vector<std::pair<int, int>>& tplRuns, int finest,
                  const double* subSphere, int nV, const std::vector<std::pair<int, int>>& subRuns, int depth, double* matchMedian = nullptr, double* matchMax = nullptr);

/** The connected pieces of a mesh: a piece number per vertex, numbered in order of their lowest vertex. */
std::vector<int> pieces(const int32_t* faces, int nF, int nV);

}  // namespace nxr::selection::bisection
