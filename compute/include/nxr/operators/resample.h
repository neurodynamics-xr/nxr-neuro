// nxr::registration — the resampling operator between two registered spheres (FreeSurfer's `sphere.reg`, or any sphere on which two surfaces are put in
// correspondence): a pure function of matrices (positions and faces in, a sparse matrix out); no file, no subject, no knowledge of where the spheres come from.
//
//   a registered sphere puts a vertex of one surface at a place on a common sphere; the operator W [to x from] carries a field on the `from` surface to the `to`
//   surface: (W x)_i = sum_j W_ij x_j, one row per destination vertex, one column per source vertex, stored CSR.
//
// Methods (named options; the reference's is the default):
//   shepard      the K nearest source vertices on the sphere with the modified Shepard weights of Franke & Nielson (1980): with squared chord distances
//                d_1 <= ... <= d_K between the unit-normalised points, weight k = ((d_K - d_k) / (d_K d_k))^2 for k = 1 ... K-1, normalised to sum 1 (the K-th neighbour
//                sets the radius and takes no weight; a zero distance is replaced by the smallest positive number of the type; a row whose weights are all 0 gives its
//                nearest neighbour weight 1). Written from the paper's formula with these choices stated.
//   nearest      the nearest source vertex, weight 1.
//   barycentric  the three vertices of the source triangle that holds the destination point, with its spherical barycentric coordinates (a point on no triangle
//                found among the faces of its nearest vertices takes its nearest vertex).
// Pieces (a cortex's two hemispheres, glued as runs of vertices, left then right) are matched one to one, in order: a destination vertex looks only among the source
// vertices of its own piece. A sphere of one piece has one piece.
//
// What it cannot resolve: the correspondence is the spheres', not anatomy's; every method smooths (a destination value is a mixture of source values), and a
// field carried back and forth is not the field. Precision follows the input: positions in float32 give float32 weights, float64 float64.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace nxr::operators::registration {

struct Piece {
  std::int64_t first = 0, count = 0;   // a run of vertices
};

enum class Method { Shepard, Nearest, Barycentric };
// "shepard" | "nearest" | "barycentric"; std::invalid_argument naming the choices otherwise.
Method parse_method(const std::string& name);
std::string method_name(Method method);

struct Options {
  Method method = Method::Shepard;
  int neighbours = 8;                  // shepard: K, the neighbours looked at (the K-th sets the radius); at most the piece's size
};

template <class S>
struct Csr {
  std::int64_t rows = 0, cols = 0;               // [destination vertices x source vertices]
  std::vector<std::int32_t> indptr;              // rows + 1
  std::vector<std::int32_t> indices;             // source vertex of each entry, ascending within a row
  std::vector<S> data;                           // its weight; the rows sum to 1
  std::int64_t nnz() const { return static_cast<std::int64_t>(data.size()); }
};

struct Report {
  std::int64_t fallback_to_nearest = 0;          // barycentric: destination points that lay on no triangle searched
  std::int64_t empty_columns = 0;                // source vertices no destination vertex takes anything from
  double max_chord = 0;                          // the largest chord distance (unit sphere) from a destination point to its nearest source vertex
};

// The connected runs of a surface's faces as contiguous vertex ranges, ordered by their first vertex (a cortex: left then right). std::invalid_argument when a
// piece is not one run of vertices, or a vertex is on no face.
std::vector<Piece> contiguous_pieces(const std::vector<std::int32_t>& faces, std::int64_t n_vertices);

// The operator carrying a field on `from` to `to`. Positions [n x 3] row-major (the spheres' radius is irrelevant: points are normalised); `from_faces` [m x 3]
// (used by `barycentric` only). std::invalid_argument for a piece count that differs, pieces that do not tile their sphere, or a point at the origin.
template <class S>
Csr<S> resample_operator(const std::vector<S>& from_positions, const std::vector<std::int32_t>& from_faces, const std::vector<Piece>& from_pieces,
                         const std::vector<S>& to_positions, const std::vector<Piece>& to_pieces, const Options& options, Report* report = nullptr);

// y = W x for a field [cols] (one value per source vertex): the operator applied.
template <class S>
std::vector<S> carry(const Csr<S>& w, const std::vector<S>& x);

}  // namespace nxr::registration
