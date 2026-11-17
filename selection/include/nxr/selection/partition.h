// nxr::selection — A PARTITION OF A MANIFOLD BY A METHOD. A partition divides a manifold's elements (a line's samples, a surface's vertices) into members
// at one or more nested levels; the manifold says what is divided, the method how. A partition made here and one io loads from file are the same thing and
// follow the same rules.
//
//   on a Line     ConstantQ   levels of tiles of 2^k samples from the first sample (level k), the ones a constant-Q band design needs, each with its band
//   on a Surface  Dyadic      equal-area bisection, 2^l tiles a connected piece (level l), each level paired with its eigenvalue band by Weyl's law
//                 Transfer    another surface's partition carried through the registered spheres (each vertex takes its nearest vertex's member)
//
// The member of an element at a level is arithmetic: element >> k on a line; codes[element] >> (depth - l) on a surface.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nxr::selection {

// ── the manifolds ────────────────────────────────────────────────────────────────────────
/** a recording's time line: `samples` samples at `sfreq` Hz */
struct Line {
  std::int64_t samples = 0;
  double sfreq = 0;
};
/** a triangle mesh: positions [vertices x 3] and faces [face_count x 3], row-major; each connected piece (a hemisphere) one contiguous block of vertices */
struct Surface {
  const double* positions = nullptr;
  std::int32_t vertices = 0;
  const std::int32_t* faces = nullptr;
  std::int32_t face_count = 0;
};

// ── the partition ────────────────────────────────────────────────────────────────────────
/** one level: `count` members; [band_lo, band_hi] the band of the manifold's spectrum it resolves (a line: Hz; a surface: Laplacian eigenvalues, in the
 *  inverse square of the positions' unit) */
struct Level {
  int level = 0;
  std::int64_t count = 0;
  double band_lo = 0, band_hi = 0;
};
/** one band of a constant-Q design and the level whose tiles resolve it */
struct Band {
  double lo = 0, hi = 0;
  int level = 0;
};
struct Partition {
  std::string on;                    // line | surface
  std::string method;                // constant-q | dyadic | transfer
  std::int64_t elements = 0;         // samples or vertices
  std::vector<Level> levels;         // by level, ascending
  std::vector<Band> bands;           // constant-Q's bands, highest first (other methods: none)
  int depth = 0;                     // a surface: the finest level
  std::vector<std::int32_t> codes;   // a surface: each vertex's member at `depth`, piece * 2^depth + tile (-1: none); a line: none (arithmetic)
};

/** the member of `element` at `level`; std::invalid_argument for a level or element the partition does not have */
std::int64_t member(const Partition& partition, int level, std::int64_t element);

// ── the methods ──────────────────────────────────────────────────────────────────────────
/** CONSTANT-Q (the Cortical Flow Studio's backend/src/db/constant-q.ts): bands of `voices` per octave down from f_max; a band's window is `cycles` periods of
 *  its low edge rounded UP to 2^k samples; bands sharing a window share level k. */
struct ConstantQ {
  std::string method = "hertz";            // hertz: cycles from the rate (about 6.8 at 1200 Hz), f_max 64 Hz (capped by Nyquist), f_min 0.5 Hz · cycles: 8 cycles, f_max Nyquist, f_min 0
  int voices = 6;                          // bands per octave
  std::optional<double> cycles;            // cycles of a band's low edge in its window; default: the method's
  std::optional<double> f_min, f_max;      // Hz; default: the method's (f_max at most Nyquist)
};
/** DYADIC (the Studio's engine/ladder.ts): each level halves every tile of the one above by equal area across its geodesic diameter, each tile one piece. */
struct Dyadic {
  std::optional<int> depth;                // the finest level; default: the finest at which every tile of every piece still holds a vertex
};
/** TRANSFER: `from` (a partition of another surface with `from_vertices` vertices) carried onto this surface through the registered spheres: each vertex takes
 *  the member of the nearest `from` vertex, piece by piece (the pieces of both in the same order). */
struct Transfer {
  const Partition* from = nullptr;
  const double* from_sphere = nullptr;     // [from_vertices x 3]
  const std::int32_t* from_faces = nullptr;
  std::int32_t from_vertices = 0, from_face_count = 0;
  const double* sphere = nullptr;          // this surface's registered sphere, [vertices x 3]
};

Partition partition(const Line& line, const ConstantQ& method);
Partition partition(const Surface& surface, const Dyadic& method);
Partition partition(const Surface& surface, const Transfer& method);

}  // namespace nxr::selection
