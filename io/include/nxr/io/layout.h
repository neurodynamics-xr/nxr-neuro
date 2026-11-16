// nxr-io — THE LAYOUT: where a node lives in a subject's store (`nxr.subject@6.0`, register D157). A FIELD'S FOLDER IS THE MANIFOLD IT IS DEFINED ON.
//
//     sensor · time · frequency · eigenvalue · source · mri · fibre     a manifold's folder
//     a product                                                         its factors' folders joined by `-`, in axis order (sensor-time, source-source)
//
// Only the manifold's row decides (its type, its role, its operator, its factors), never a path. nxr.io.layout.
#pragma once
#include <optional>
#include <string>

#include "nxr/io/database.h"

namespace nxr::io {

// the folder of a manifold (and of the fields and selections on it), from its row
std::string folder_of(const Database& db, const std::string& manifold_id);
// a line's role (SPEC §6, derived from its coordinate system): "time" · "spectral" · "eigen" · "" (none, or not a line)
std::string line_role(const Database& db, const std::string& manifold_id);
// the store path of a node named `name` that lives with manifold `manifold_id`
std::string node_path(const Database& db, const std::string& manifold_id, const std::string& name);
// the folder of a manifold about to be written (its row not there yet); `role`: time · spectral · eigen (lines)
std::string folder_for(const std::string& type, const std::string& name = "", const std::optional<std::string>& role = std::nullopt,
                       const std::optional<std::string>& op = std::nullopt);
// a Selection's name as a node name: runs of anything but letters, digits, `_ . -` become `_`
std::string safe_name(const std::string& s);
// the node of an atlas's arrays: in the folder of the Selection's manifold, named `<name>_atlas`
std::string atlas_node(const Database& db, const std::string& manifold_id, const std::string& name);

}  // namespace nxr::io
