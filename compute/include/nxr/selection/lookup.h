// nxr::selection (compute) — A LOOKUP: the names a selection's codes are read with (a region's name and colour in a parcellation, an event's name, a band's
// name). io stores it as the selection's dictionary.
#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nxr::selection {

/** one code and what it means */
struct Entry {
  std::int64_t code = 0;
  std::string name;
  std::optional<std::array<int, 3>> color;   // 0..255 each
};

/** the entries of a lookup, one per code; empty: the codes are their own names */
using Lookup = std::vector<Entry>;

/** the entry of `code`; std::invalid_argument when the lookup has none */
const Entry& entry(const Lookup& lookup, std::int64_t code);

/** std::invalid_argument when two entries share a code or a name is empty */
void check(const Lookup& lookup);

}  // namespace nxr::selection
