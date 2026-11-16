// nxr-io — YAML read as JSON: a pipeline definition (nxr's format, docs/specs/2026-11-12-chain-graph.md §2) and any other YAML.
// Parsed with rapidyaml (vendored: io/extern/ryml, v0.15.2;). Plain scalars are typed as PyYAML's
// `safe_load` types them (nxr reads the same files with it): null (`~`, `null`, empty), booleans (`true`/`false`, any case, and yes/no/on/off), integers,
// floats (`.inf`, `.nan` too), else strings; a quoted scalar is always a string. Anchors and aliases are resolved; tags are not supported.
#pragma once
#include <string>

#include "nxr/io/json_text.h"

namespace nxr::io {

/** the YAML document `text` as JSON; a syntax error is an Error (invalid_argument) naming `where` and the line */
Json parse_yaml(const std::string& text, const std::string& where = "yaml");

}  // namespace nxr::io
