// nxr-graph — the gauge's nodes (G5; spec docs/specs/2026-11-17-geometry-central-switch.md, sections 5.2 and 5.3). Pure, pointwise, on arrays: no surface, no file.
//
//   frame.rotate   x [V x time] complex spin k, to [V] real (radians)   -> y = x e^(-i k to): the value in the frame whose reference is at angle `to` in x's gauge
//   frame.embed    x [V x time] complex spin 1, grid [V x 3] complex     -> y [3V x time]: z = a + ib becomes a e1 + b e2 (the embedded 3-vector, gauge-free)
//   frame.project  x [3V x time] real vectors, grid [V x 3] complex      -> y [V x time] complex: z = (v . e1) + i (v . e2), spin 1 in the grid's gauge
//
// `grid` is the surface's `vertexGrid` (c = e1 + i e2, [V x 3]); as an Array it is rows V, cols 3, comps 2 (re = e1, im = e2). `to` is an Array of V doubles.
// A value inside the loop is a Block: complex rows V; a 3-vector's components consecutive (rows 3V). The arithmetic runs in double, the stored precision follows the input (float32 blocks).
#pragma once
#include "nxr/graph/node.h"

namespace nxr::graph {

NodeSpec frame_rotate_node();
NodeSpec frame_embed_node();
NodeSpec frame_project_node();
/** the three, registered by `registry()` */
void register_frame_nodes(Registry& r);

}  // namespace nxr::graph
