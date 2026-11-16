// nxr::glue — two closed pieces (a cortex's hemispheres) made ONE closed surface by anatomy: each piece's MEDIAL WALL is cut out as one disk and a
// BRIDGE of new vertices is grown between the two rims. A line port of the Cortical Flow Studio's `glueMesh` and `sourcesOf` (studio/src/studio/glue.ts),
// the reference (nxr's `geom.glue` is the same algorithm): the iteration orders, the tie rules and JavaScript's rounding are kept, so the faces come out
// identical (test_glue, golden from the Studio's own function). Compiled without fused multiply-add (JavaScript has none).
//
//   glueMesh(P, F, wall)   P [n x 3] row-major, F [f x 3], wall [n] (1 on the medial wall: the Desikan-Killiany atlas' `unassigned`)
//     → positions: the KEPT vertices (ascending original index), then the bridge's R rings of N vertices each
//       faces:     the two cut hemispheres' faces (re-indexed), then the bridge's
//       keep:      each glued vertex's original index; −1 on the bridge (new vertices: not cortex)
//       bridge:    its vertex count, rings, area and longest edge
//   Checked, as the reference does: every half-edge once, every edge on two faces, one piece, χ = χ₁ + χ₂ − 2.
//   sourcesOf(P, g)        where each glued vertex's values come from: a kept vertex its own; a bridge vertex the inverse-distance blend of its 3 nearest
//                          CUT wall vertices (the Studio carries an estimate over the bridge this way; a forward model is zero there: not cortex)
#pragma once
#include <cstdint>
#include <vector>

namespace nxr::glue {

struct Bridge { std::int64_t vertices = 0, rings = 0; double area = 0, longestEdge = 0; };
struct Glued {
  std::vector<double> positions;           // [total x 3]
  std::vector<std::int32_t> faces;         // [faces x 3]
  std::vector<std::int32_t> keep;          // [total]: the original vertex, −1 on the bridge
  Bridge bridge;
};
Glued glueMesh(const std::vector<double>& P, const std::vector<std::int32_t>& F, const std::vector<std::uint8_t>& wall);

struct Sources { int k = 3; std::vector<std::int32_t> idx; std::vector<float> w; };   // [total x k]; idx −1: no source
Sources sourcesOf(const std::vector<double>& P, const Glued& g);

}  // namespace nxr::glue
