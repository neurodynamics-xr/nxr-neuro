// nxr-pipelines — the dyadic TILE LADDER of a surface and the tile summaries of maps, stored as nxr stores them (Step 3 of the cortical coordinate system; maths in nxr/ladder.h).
//
//   Tiles are SELECTIONS on the surface's manifold, never a manifold or a folder of their own (nxr's atlas layout):
//     ladder        `<surface> tiles`        Selection type `ladder`, dictionary = the levels (code = level; measurement = the level's wavelength, attributes = tiles, Weyl band, tau, ...); params: weyl, area, tree "subject"
//     level l       `<surface> tiles L<l>`   Selection type `set`, member l of the ladder: one tile code per vertex (h 2^l + t, h the connected piece: a hemisphere), its dictionary `<surface> tiles L<l>` (a tile's name;
//                                            entries up to 1024 tiles, as the Studio writes them). Arrays in the Selection's atlas node (`source/<surface>_tiles_L<l>_atlas`, nxr.io.layout.atlas_node):
//                                              codes [V] int32 · w [tiles] float64 (area) · lc/L<l>/centre [tiles] int32 (the vertex nearest the area-weighted centroid; -1 empty) · lc/L<l>/adjacency [E,2] int32
//   SUMMARIES of a map over the tiles of a level (each recorded in the level Selection's params_json "summaries"; the level's params_json gains nxr's `atlas` block {tree, level, finest, gauge, areas, node}):
//     scalar        maps/<name>/{w,s1,s2,n,min,max}       Sum a, Sum a x, Sum a x^2, count, min, max   (nxr map_stats, nxr's key `<node>/maps/<field>/...`; area-weighted) as ARRAY-BACKED MEASUREMENT ROWS of the Selection
//                   (step 5 of the measurement pyramid, 2026-11-14): measures weight, sum, sum_square, count, min, max of the map's Field (`of_field_id`; a statistic's slice: its byte-less Field, below);
//                   the mean (sum / weight), mean_square and variance are DERIVED AT READ by `measure query`, which also rolls the rows up the ladder. `mean` is not stored.
//                   A map of a STATISTIC's slice (`quadrature --tiles`' energy maps) gets a BYTE-LESS Field `<slice> tile map` (kind `tile map`, no array; the pattern of the source estimate and phasegradient) that its
//                   rows are of; its producer_json names the statistic and the slice (decision 4, 2026-11-14). The vector forms below stay arrays in the node, named `summary/<stem>/<map>/...`:
//     vector        frame/{w,c1,c2,cn,c11,c22,c12,mag}    components in the CANONICAL frames (nxr vector_stats; reduction "frame", gauge "canonical frames"); rolls up like a scalar
//                   levi_civita/{re,im,w,length}          the field carried along the Levi-Civita connection to the tile centre (nxr lc_resultant; reduction "levi-civita"); length = |z| / Sum a |(c1,c2)|
//                   lc/L<l>/{rho,omega}                   the frame angle of each vertex's tile centre, carried there, and the connection between centres (per level, never rolled up)
//                   Raw 3-D sums across a tile are never written. Without canonical frames only the Levi-Civita form is written, in the signpost gauge (labelled).
#pragma once
#include <Eigen/Dense>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/frames.h"
#include "nxr/io.h"

namespace nxr::pipelines {

struct TilesComputeOptions {
  std::optional<std::string> surface;        // the subject's surface (default: the primary one)
  int depth = -1;                            // the finest level written (< 0: the finest level every piece reaches)
  bool replace = false;                      // an existing ladder on the surface is removed and rewritten
  std::string tree = "subject";              // "subject" (equal-area on this cortex) or "group" (the default subject's: the GROUP TREE every subject reaches through its sphere)
  std::optional<nxr::io::Json> atlas;        // the dataset's default atlas definitions, carried in the ladder's params_json "atlas" (the group tree only; D131)
};
struct TilesComputeResult {
  nxr::io::Json info;
  bool existed = false;                      // the stored ladder was kept
};
TilesComputeResult tiles_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const TilesComputeOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

// THE GROUP TREE (D127/D128, nxr.geom.trees): the dyadic ladder of the default subject's primary cortex (`<surface> tiles`, Selection type `ladder`, D85),
// written as `tiles_compute` writes any ladder, with `tree: "group"` and the dataset's default ATLAS DEFINITIONS in its params_json "atlas" — the settings
// every subject's atlas and the group sums must share (nxr's atlas default-subject): the tree, the atlas leaf depth, the resolution, the frames (the id of the
// pial's canonical frames when `frames compute` stored them first, as `dataset create` does; else null), the time convention (frames are cycles of the day-anchored tower at `frame_level`: 86 400·2^frame_level s; a level-m tile is tower level
// frame_level + m), the default bands (octaves 1–64 Hz), the Levi-Civita levels and the default trajectory scales.
// With the canonical frames stored it also writes the group tree's LEVI-CIVITA CONNECTION at `lc_levels` (nxr's default subject): the Selection `<surface>_lc_centres_L<l>`
// (each tile's centre vertex, -1 empty) and Omega[a, b] as `levi-civita connection` measurements (rad) on level l's Selection, code a, within_code b; its record (transport,
// antisymmetrisation, warnings) in the ladder's params_json "levi_civita". The antisymmetrisation is NOT nxr's (which turns |Omega| > 90 degrees by 180; see tiles.cpp).
struct GroupTreeOptions {
  std::string resolution = "fsaverage5";     // recorded in the definitions
  int atlas_depth = 8;                       // the default leaf depth of the dataset's atlases (nxr's --depth)
  int frame_level = -18;                     // the base frame's tower level (nxr's --frame-level; ≈ 0.33 s)
  std::vector<int> lc_levels = {4, 6};       // the levels of the Levi-Civita connection (nxr's --lc-levels)
};
nxr::io::Json group_tree_definitions(const GroupTreeOptions& opt);
TilesComputeResult group_tree_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& default_subject, const GroupTreeOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

// Maps to summarise over tiles: scalars [V], vectors [V x 3] (ambient 3-vectors, read in the frames).
// what a scalar map is OF: its summaries are measurement rows of a Field, or of a statistic's slice (one of the two)
struct TileMapSource {
  std::string field_id;                      // the Field the map is (a stored Field with values)
  std::string statistic_id;                  // or the statistic ...
  std::int64_t index = -1;                   // ... and its slice along the second axis (a map of a statistic: the band)
  std::string name;                          // the map's own name (the Field's or the slice's: nxr's `maps/<name>`); default `<stem>_<map>`
  std::optional<std::string> unit;
};
struct TileMaps {
  std::string stem;                          // the maps' common name (e.g. the optical flow's stem)
  std::vector<std::pair<std::string, Eigen::VectorXd>> scalars;
  std::map<std::string, TileMapSource> sources;   // by scalar map name: REQUIRED for every scalar (the rows say what they are of)
  std::vector<std::pair<std::string, nxr::frames::Mat3>> vectors;
};
struct TileSummaryOptions {
  std::optional<std::string> surface;
  std::vector<int> levels;                   // levels to summarise (empty: 1 .. min(depth, max_level))
  int max_level = 6;                         // the deepest default level (the Levi-Civita form carries a tile centre per tile)
  bool levi_civita = true;
  bool replace = false;                      // summaries of this stem already stored are rewritten
};
// The ladder is made first when the surface has none. Returns {levels, summaries: [...], gauge, ...}.
nxr::io::Json tile_summaries_write(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const TileMaps& maps, const TileSummaryOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

// A stored Field of the subject read as a map [n x components] (float32 / float64 / int32 values), or else a map of a statistic of that name (its band slice,
// io::map_slices); nullopt when there is neither or its size is not a multiple of n.
std::optional<Eigen::MatrixXd> stored_map(nxr::io::SubjectWriter& sub, const std::string& field_name, std::int64_t n, TileMapSource* source = nullptr);

}  // namespace nxr::pipelines
