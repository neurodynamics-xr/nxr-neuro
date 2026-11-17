// nxr::selection (compute) — THE PARTITION METHODS: selections whose members are disjoint and cover a manifold, made in memory (selection.h); io stores them.
// Compute takes arrays: a line is a count and a spacing, a surface its positions and faces.
//
//   constant_q        a time line   a set with LEVELS, each a spans selection in closed form (tiles of 2^k samples from the first sample, level k), the levels a
//                                   constant-Q band design needs, each with the band (Hz) it resolves
//   constant_q_bands  a time line   the same design's FREQUENCY partition: spans on the frequency line, one row a band, labelled by its range
//   dyadic            a surface     a set with LEVELS by codes: equal-area bisection, 2^l tiles a connected piece (level l), each level with its eigenvalue band
//                                   by Weyl's law (1/unit^2)
//   eigenvalue_bands  a set with levels   the EIGENVALUE partition: spans on the eigenvalue line, one row a level's band
//   transfer          a surface     a set by codes carried from another surface through the registered spheres (nearest vertex, piece by piece)
#pragma once
#include <cstdint>
#include <optional>
#include <string>

#include "nxr/selection/selection.h"

namespace nxr::selection {

/** a line of `count` points `spacing` apart from `origin`, in `unit` (a time line: spacing = 1 / sfreq, unit s) */
struct Line {
  std::string unit = "s";
  std::int64_t count = 0;
  double origin = 0, spacing = 0;
};
/** a triangle mesh: positions [vertices x 3] in `unit` and faces [face_count x 3], row-major; each connected piece (a hemisphere) one contiguous block of
 *  vertices. The unit is the caller's (never defaulted): it gives the eigenvalues theirs, 1/unit^2. */
struct Surface {
  std::string unit;
  const double* positions = nullptr;
  std::int32_t vertices = 0;
  const std::int32_t* faces = nullptr;
  std::int32_t face_count = 0;
};

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
/** TRANSFER: `from` (a set by codes on another surface of `from_vertices` vertices) carried through the registered spheres; its levels and lookup come with it */
struct Transfer {
  const Selection* from = nullptr;
  const double* from_sphere = nullptr;     // [from_vertices x 3]
  const std::int32_t* from_faces = nullptr;
  std::int32_t from_vertices = 0, from_face_count = 0;
  const double* sphere = nullptr;          // this surface's registered sphere, [vertices x 3]
};

Selection constant_q(const Line& time, const ConstantQ& method);
Selection constant_q_bands(const Line& time, const ConstantQ& method);
Selection dyadic(const Surface& surface, const Dyadic& method);
Selection eigenvalue_bands(const Selection& levels);
Selection transfer(const Surface& surface, const Transfer& method);

}  // namespace nxr::selection
