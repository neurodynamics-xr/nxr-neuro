// nxr::selection (compute) — CREATING SELECTIONS: A PARTITION OF A MANIFOLD BY A METHOD, in memory; io stores, loads and updates them, and a loaded one
// follows the same rules. A partition divides a manifold's elements (a line's points, a surface's vertices) into members at one or more nested levels, every
// element in one member; the manifold says what is divided, the method how. A partition may carry a lookup naming its codes (a parcellation's regions).
//
//   on a time Line      ConstantQ   levels of tiles of 2^k samples from the first sample (level k), the ones a constant-Q band design needs
//   on a Surface        Dyadic      equal-area bisection, 2^l tiles a connected piece (level l)
//                       Transfer    another surface's partition carried through the registered spheres (each vertex takes its nearest vertex's member)
//   any manifold        labels      a code per element and a lookup (one level)
//
// Each level of a constant-Q or dyadic partition resolves a band of the spectrum (Hz; Laplacian eigenvalues by Weyl's law), and those bands partition the
// spectrum's own line: `spectrum_partition` gives the frequency partition of a constant-Q one, the eigenvalue partition of a dyadic one.
//
// A level's members are found by arithmetic (point >> k on a time line), by codes (codes[vertex] >> (depth - l)), or by edges (a line divided at
// boundaries: member i is [edges[i], edges[i + 1])).
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nxr/selection/lookup.h"

namespace nxr::selection {

// ── the manifolds ────────────────────────────────────────────────────────────────────────
/** a line of `count` points `spacing` apart from `origin`, in `unit` (a time line: spacing = 1 / sfreq, unit s); `quantity` says what it measures */
struct Line {
  std::string quantity = "time";   // time · frequency · eigenvalue · …
  std::string unit = "s";
  std::int64_t count = 0;
  double origin = 0, spacing = 0;
};
/** a triangle mesh: positions [vertices x 3] and faces [face_count x 3], row-major; each connected piece (a hemisphere) one contiguous block of vertices */
struct Surface {
  const double* positions = nullptr;
  std::int32_t vertices = 0;
  const std::int32_t* faces = nullptr;
  std::int32_t face_count = 0;
};

// ── the partition ────────────────────────────────────────────────────────────────────────
/** one level: `count` members; [band_lo, band_hi] the band of the spectrum it resolves (none: 0, 0); `edges` (count + 1, ascending) when its members are
 *  intervals of a line in the line's unit */
struct Level {
  int level = 0;
  std::int64_t count = 0;
  double band_lo = 0, band_hi = 0;
  std::vector<double> edges;
};
/** one band of the spectrum a method designed, and the level whose members resolve it */
struct Band {
  double lo = 0, hi = 0;
  int level = 0;
};
struct Partition {
  std::string on;                    // the manifold's quantity: time · frequency · eigenvalue · surface · …
  std::string unit;                  // a line's unit (its edges and coordinates)
  std::string method;                // constant-q · dyadic · transfer · labels · spectrum
  std::int64_t elements = 0;         // points or vertices (a partition of a line by edges: 0, it divides the line's values)
  double origin = 0, spacing = 0;    // a line's points (arithmetic levels)
  std::vector<Level> levels;         // by level, ascending
  std::vector<Band> bands;           // the spectrum's bands the method designed, highest first (constant-q: one a band; dyadic: one a level)
  std::string spectrum, spectrum_unit;   // the line those bands lie on: frequency, Hz · eigenvalue, the inverse square of the positions' unit
  int depth = 0;                     // codes: the finest level
  std::vector<std::int32_t> codes;   // a code per element at `depth` (-1: none); a surface's dyadic code is piece * 2^depth + tile
  Lookup lookup;                     // the codes' names (empty: none)
};

/** the member of element `element` at `level` (-1: none); std::invalid_argument for a level or element the partition does not have */
std::int64_t member(const Partition& partition, int level, std::int64_t element);
/** the member holding the line coordinate `value` at `level` (-1: none) — a partition of a line, by arithmetic or edges */
std::int64_t member_at(const Partition& partition, int level, double value);

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
 *  the code of the nearest `from` vertex, piece by piece (the pieces of both in the same order); the lookup comes with it. */
struct Transfer {
  const Partition* from = nullptr;
  const double* from_sphere = nullptr;     // [from_vertices x 3]
  const std::int32_t* from_faces = nullptr;
  std::int32_t from_vertices = 0, from_face_count = 0;
  const double* sphere = nullptr;          // this surface's registered sphere, [vertices x 3]
};

Partition partition(const Line& time, const ConstantQ& method);
Partition partition(const Surface& surface, const Dyadic& method);
Partition partition(const Surface& surface, const Transfer& method);

/** LABELS: the partition of the manifold `on` given by a code per element (-1: none) and the lookup naming the codes (a parcellation, a flag per channel);
 *  one level. std::invalid_argument for a code the lookup lacks. */
Partition labels(std::string on, std::vector<std::int32_t> codes, Lookup lookup);

/** THE SPECTRUM'S PARTITION: the line the bands of `p` lie on (frequency for constant-Q, eigenvalue for dyadic) divided at the bands' edges, one member a
 *  band, lowest first, each named by its range; one level. std::invalid_argument for a partition with no bands. */
Partition spectrum_partition(const Partition& p);

}  // namespace nxr::selection
