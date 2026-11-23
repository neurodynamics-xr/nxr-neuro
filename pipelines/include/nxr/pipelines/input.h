// nxr-pipelines — what a pipeline reads from a subject's rows, with nxr-io: a stored surface (its row, positions and faces).
#pragma once

#include "nxr/operators/hodge_synthesis_weights.h"
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
  bool primary = false;                    // the stored flag, a fact of the row; never a reason to pick this surface
  std::vector<double> vertices;            // [nV x 3] float32 values widened to float64, as the Studio receives them (nxr.pipelines.hodge.modes_input)
  std::vector<std::int32_t> faces;         // [nF x 3]
  std::int64_t n_vertices() const { return static_cast<std::int64_t>(vertices.size() / 3); }
  std::int64_t n_faces() const { return static_cast<std::int64_t>(faces.size() / 3); }
};

// A subject's surface read from its rows: the surface called `name`; with no name, the subject's one surface. WHICH SURFACE is the caller's choice, never a "primary" default: a subject with several surfaces
// and no name is refused, naming them; a subject with none, no_such_row.
// `as_stored`: the positions at the precision they are stored in (float64), as the Studio's backend (entities/manifold.ts loadPositions) and nxr read
// them for the dyadic ladder; otherwise (the default) rounded to float32 first, as the Studio's GPU-side pipelines receive them (eigen, flow, …).
SurfaceInput read_surface(nxr::io::SubjectWriter& sub, const std::optional<std::string>& name = std::nullopt, bool as_stored = false);

// The same, from a package opened alone (a released template is never opened by a writer): the surface called `name`; `no_such_row` when there is none.
SurfaceInput read_surface(const nxr::io::SubjectPackage& package, const std::string& name);

// A recording's channels as facts a node may read on its port type: {types, quality, units, n_channels} (per channel; quality from the recording's flags, else the montage's).
nxr::io::Json channel_facts(nxr::io::SubjectWriter& sub, const std::string& recording);
// The sensor geometry of a channels manifold (what a head model reads): positions [M x 3], types, orientations when stored, and the coils with their weights.
struct SensorGeometry {
  std::string id, name, cs_id;
  std::vector<std::string> types;
  std::vector<double> positions;                          // [M x 3]
  std::optional<std::vector<double>> orientation;         // [M x 3], `<name>_orientation`
  bool coils = false;                                     // `<name>_coils` and `<name>_coil_weights` stored
  std::vector<double> coil_positions, coil_orientation;   // [Q x 3]
  std::vector<std::int32_t> indptr, indices;              // the weights' CSR [channels x coils]
  std::vector<float> weights;
};
SensorGeometry sensor_geometry(nxr::io::SubjectWriter& sub, const std::string& channels_id);
// The coordinate system a manifold's geometry is in ("" when it records none).
std::string coordinate_system_of(nxr::io::SubjectWriter& sub, const std::string& manifold_id);
// The length unit of a manifold's coordinate system ("m", "mm"; "" when it records none).
std::string coordinate_unit_of(nxr::io::SubjectWriter& sub, const std::string& manifold_id);
// A HODGE-DIRAC PACKAGE as a node reads it: the basis of its first `modes` modes (null: all), the input surface's positions, the package's identity.
struct HodgePackageData {
  nxr::operators::Basis basis;
  std::vector<double> input_positions;    // [V_in x 3] the kernel's sources
  std::string input_surface_id, stem, axis_id, id;
};
HodgePackageData hodge_package_data(nxr::io::SubjectWriter& sub, const std::string& eigenvectors_name, const std::optional<std::int64_t>& modes);
/** the package's FULL forms beyond its eigenvectors: the 2-form parts χ/√2 of its first `K` modes on the intrinsic faces (`<stem>_forms2`, [nF x K], float32 as stored, row-major) and the
 *  options its intrinsic complex was built with (the solve's provenance: mollify, flip_eps, star1_floor; null floor: none named), so a node rebuilds the same complex from the glued surface */
struct HodgePackageForms {
  std::vector<float> forms2;
  std::vector<float> forms1;                 // the 1-form parts [nE x K] of `<stem>_forms1` (float32 as stored, row-major); empty when the package has none (a reading makes them)
  std::int64_t nF = 0, K = 0, nE = 0;
  double mollify = 1e-6, flip_eps = 1e-6, star1_floor = 0.0;
};
HodgePackageForms hodge_package_forms(nxr::io::SubjectWriter& sub, const HodgePackageData& package);
// The type of each channel of a channels manifold (its 'type' labelling); empty when there is none.
/** the positions of a points manifold with no channel types (the landmarks of a template: nasion, LPA, RPA): [n x 3] and each point's name from its `<manifold>_names` labelling (empty when it has none) */
struct NamedPoints { std::string id, name; std::vector<double> positions; std::vector<std::string> names; };
NamedPoints named_points(nxr::io::SubjectWriter& sub, const std::string& points_id);
std::vector<std::string> channel_types_of(nxr::io::SubjectWriter& sub, const std::string& channels_id);
// The name of each channel of a channels manifold (its `<manifold>_names` labelling); empty when there is none.
std::vector<std::string> channel_names_of(nxr::io::SubjectWriter& sub, const std::string& channels_id);

// How many connected pieces the faces make (a cortex: its two hemispheres; a mesh or a sphere: one).
int connected_pieces(const std::vector<std::int32_t>& faces, std::int64_t n_vertices);
/** the stem of the Hodge-Dirac package a surface's eigenmodes are stored under: its own (`hodgeDirac` for the primary surface, `hodgeDirac__<name>`), or, for a cortex of two pieces, the package of its glued
 *  surface `<name>_glued` (the surface `glue` makes, the one the modes are solved on) */
std::string hodge_package_stem(const SurfaceInput& surface);

// A surface's MEDIAL WALL [nV] (1 on it): the vertices its stored Desikan–Killiany atlas (`<surface>_atlases/Desikan-Killiany`) codes `unassigned` (or
// `unknown`), as nxr's `nxr.pipelines.hodge.medial_wall` and the Studio's `medialWall` read it. A surface with no such atlas is refused by name (the
// hemispheres are glued by anatomy only).
std::vector<std::uint8_t> medial_wall(nxr::io::SubjectWriter& sub, const SurfaceInput& surface);

// THE GLUED CORTEX. The imported two-hemisphere surface is the manifold of record; the glued surface is a DERIVED manifold made by a `glue` definition, which stores its MAP (glued vertex → original
// vertex, -1 on the bridge) in the manifold's producer record under `glue: {original: <manifold id>}`; the map is the field `<glued name>_map` of the glued surface. Analysis runs on the glued surface; every result is RESTRICTED BACK to the original for
// storage, export and visualisation: exact (the glued → original transfer), the bridge dropped like the medial wall.
struct GlueMap {
  std::string glued_id, original_id;
  std::vector<std::int32_t> original_of;   // [n_glued]: the original vertex, or -1 (the bridge)
  std::int64_t n_glued() const { return static_cast<std::int64_t>(original_of.size()); }
};
// The glue map of a manifold when it is a derived glued surface (made by `glue`), else nothing.
std::optional<GlueMap> glue_map_of(nxr::io::SubjectWriter& sub, const std::string& manifold_id);
// Vertex-major values [n_glued x comps] restricted to the original [n_original x comps]: each original vertex takes its glued counterpart's values; an original vertex with no counterpart (the medial wall,
// when the glue left it out) is NaN, never a guess. The bridge's values are dropped.
std::vector<double> restrict_to_original(const std::vector<double>& glued, std::int64_t comps, const GlueMap& map, std::int64_t n_original);
// A value given on the original surface carried to the glued one (the original → glued transfer for an input): copied where the vertex is kept; the bridge vertices take the mean of their `sources` neighbours
// when `sources` (per glued vertex, the original vertices it interpolates from; empty for a copied one) is given, else NaN.
std::vector<double> carry_to_glued(const std::vector<double>& original, std::int64_t comps, const GlueMap& map, const std::vector<std::vector<std::int32_t>>& sources = {});

}  // namespace nxr::pipelines
