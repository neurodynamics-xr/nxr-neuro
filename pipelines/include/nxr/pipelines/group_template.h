// nxr-pipelines — what the GROUP TREE of a subject is carried from: the dataset's default subject's primary cortex, its registered sphere, and its group ladder (the finest level's tile
// codes and the dictionaries its levels use). Read by the runner (Bound::group_template) for the node `group_tile_grid`; the node itself holds none of it.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "nxr/io.h"

namespace nxr::pipelines {

struct GroupTemplate {
  std::string subject, surface;                  // the default subject and its primary surface
  std::int64_t n_vertices = 0;                   // the template surface's vertices
  std::vector<std::int32_t> faces;               // [faces x 3] of the template surface
  std::vector<double> sphere;                    // the template's registered sphere [n_vertices x 3]
  std::vector<double> subject_sphere;            // the run's subject's registered sphere [vertices x 3]
  std::vector<std::int32_t> codes;               // the template's tile code per vertex at its finest level (h 2^finest + t)
  int finest = 0;                                // that level
  int slots = 2;                                 // the piece slots the codes reserve (level l has slots << l members)
  std::string ladder_dictionary_id;              // the template ladder's dictionary (its levels) ...
  std::vector<std::string> level_dictionary_ids; // ... and each level's (the tiles' names), reused by the subject's group tree
};

/** the default subject's group tree as the subject `sub` (a regular subject of `ds`) is carried onto it through the registered spheres `<surface>_sphere`; refuses (no_such_row,
 *  invalid_argument) when the dataset has no default subject, either sphere is missing, or the template's ladder is not a group tree */
GroupTemplate read_group_template(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::string& surface_name);

}  // namespace nxr::pipelines
