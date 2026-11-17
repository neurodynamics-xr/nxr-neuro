// nxr-pipelines — what a pipeline reads from a subject's rows, with nxr-io: a stored surface (its row, positions and faces).
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io.h"
#include "nxr/io/subject_package.h"

namespace nxr::pipelines {

struct SurfaceInput {
  nxr::io::Row manifold;                   // the surface's manifold row
  std::string id, name;
  bool primary = false;
  std::vector<double> vertices;            // [nV x 3] float32 values widened to float64, as the Studio receives them (nxr.pipelines.hodge.modes_input)
  std::vector<std::int32_t> faces;         // [nF x 3]
  std::int64_t n_vertices() const { return static_cast<std::int64_t>(vertices.size() / 3); }
  std::int64_t n_faces() const { return static_cast<std::int64_t>(faces.size() / 3); }
};

// A subject's surface read from its rows: the primary one, or the surface called `name`. KeyError (no_such_row) when there is none.
// `as_stored`: the positions at the precision they are stored in (float64), as the Studio's backend (entities/manifold.ts loadPositions) and nxr read
// them for the dyadic ladder; otherwise (the default) rounded to float32 first, as the Studio's GPU-side pipelines receive them (eigen, flow, …).
SurfaceInput read_surface(nxr::io::SubjectWriter& sub, const std::optional<std::string>& name = std::nullopt, bool as_stored = false);

// The same, from a package opened alone (a released template is never opened by a writer): the surface called `name`; `no_such_row` when there is none.
SurfaceInput read_surface(const nxr::io::SubjectPackage& package, const std::string& name);

// How many connected pieces the faces make (a cortex: its two hemispheres; a mesh or a sphere: one).
int connected_pieces(const std::vector<std::int32_t>& faces, std::int64_t n_vertices);

// A surface's MEDIAL WALL [nV] (1 on it): the vertices its stored Desikan–Killiany atlas (`<surface>_atlases/Desikan-Killiany`) codes `unassigned` (or
// `unknown`), as nxr's `nxr.pipelines.hodge.medial_wall` and the Studio's `medialWall` read it. A surface with no such atlas is refused by name (the
// hemispheres are glued by anatomy only).
std::vector<std::uint8_t> medial_wall(nxr::io::SubjectWriter& sub, const SurfaceInput& surface);

}  // namespace nxr::pipelines
