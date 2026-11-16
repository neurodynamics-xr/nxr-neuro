// nxr-io — CELLS (D92): which elements of a manifold a Field sits on, a Selection indexes, an Operator maps between. A port of the Cortical Flow
// Studio's `backend/src/db/cells.ts` (the reference): a cell key '0'…'3' (a rank), '1*' (a dual cell), '0,1' (one part per factor of a product);
// '0' is the vertices on a product too. A count is the manifold's f-vector entry (a product's: its factors' product; a dual k-cell counts as a
// primal (n − k)-cell).
#pragma once
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/database.h"

namespace nxr::io {

struct CellPart { int rank = 0; bool dual = false; };
// Throws invalid_argument on a malformed key.
std::vector<CellPart> parse_cell(const std::string& key);
std::vector<CellPart> cell_parts(const std::string& key, std::size_t factors);
// How many cells `key` addresses on manifold `manifold_id`; nullopt where the f-vector does not know (the Studio's cellCountOf).
std::optional<std::int64_t> cell_count_of(const Database& db, const std::string& manifold_id, const std::string& key);
// REFUSE a key that does not fit its manifold (one part per factor, none above its factor's top rank): throws invalid_argument "<what>: …" (the Studio's checkCell).
void check_cell(const Database& db, const std::string& manifold_id, const std::string& key, const std::string& what);

}  // namespace nxr::io
