// WHICH STORE PATHS THE ROWS NAME (register D164) — what keeps a delete from taking an array another row still needs. The Studio's
// `backend/src/db/named-paths.ts`: a path is "named" when any remaining row's column holds it; two paths conflict when they are equal or one is
// inside the other, compared as the file system compares names (here: ASCII case-folded; node names are `safe_name`d, so NFC never differs).
#pragma once
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/io/database.h"

namespace nxr::io {

// every store path a row of the subject names -> the first row naming it ("<table> <id>"), in the order found.
std::vector<std::pair<std::string, std::string>> named_paths(const Database& db, const std::string& subject_id);
bool overlaps(const std::string& a, const std::string& b);
std::optional<std::pair<std::string, std::string>> named_overlapping(const std::vector<std::pair<std::string, std::string>>& named,
                                                                     const std::string& path);
// whether a deleted node's location (`<dataset>/<subject>/<path>`) is named, or overlapped, by a row that remains.
bool named_by_remaining(const Database& db, const std::optional<std::string>& dataset, const std::string& location);

}  // namespace nxr::io
