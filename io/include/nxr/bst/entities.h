// nxr-io — the Brainstorm readers' shapes as ENTITIES, composed on a SubjectWriter: `nxr.convert.entities` (labelling, label_set, flag_levels).
//
// A LABELLING is a partition Selection (D48): one code per element, its codes read with a DICTIONARY (D34): an atlas's is app-wide, a subject's
// own names (its channels, its head points) are the subject's. A LABEL SET is a `set` Selection whose members are labellings (O9).
#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/io/writer.h"

namespace nxr::bst {

using Rgb = std::array<int, 3>;

struct LabellingOptions {
  std::string manifold_id;
  std::string dictionary;                       // its name
  std::string scope = "subject";                // app · dataset · subject
  std::vector<std::optional<Rgb>> colors;       // per level (empty: the dictionary's own)
  std::optional<std::string> acquisition;      // a label (found or made) ...
  std::optional<std::string> acquisition_id;    // ... or an id (kept)
  std::optional<std::string> description;
  std::optional<std::string> of_field_id;       // the Field the labelling describes
  std::optional<std::pair<std::string, int>> member_of;   // (set id, ordinal)
  bool element_rows = true;
};

// One PARTITION Selection at `path`: its codes array, its dictionary (made or extended by name in its scope). `codes` index `levels`.
// Returns the selection's id.
std::string labelling(nxr::io::SubjectWriter& sub, const std::string& path, const std::vector<std::int64_t>& codes, const std::vector<std::string>& levels,
                      const LabellingOptions& opt);

struct LabelMember {
  std::string family;
  std::vector<std::int64_t> codes;
  std::vector<std::string> levels;
  std::vector<std::optional<Rgb>> colors;
};

// A SET of labellings: the set at `set_path` (a group node), each member its own partition `<set_path>/<family>`. `dictionary(family)` is (name, scope).
// Returns (set id, member ids).
std::pair<std::string, std::vector<std::string>> label_set(nxr::io::SubjectWriter& sub, const std::string& set_path, const std::vector<LabelMember>& members,
                                                           const std::string& manifold_id, const std::string& description, const std::optional<std::string>& acquisition,
                                                           const std::function<std::pair<std::string, std::string>(const std::string&)>& dictionary);

// Brainstorm's channel flags (1 good, -1 bad) as a labelling: a code per channel, the levels present (good first), their colours. ValueError otherwise.
struct FlagLevels { std::vector<std::int64_t> codes; std::vector<std::string> levels; std::vector<std::optional<Rgb>> colors; };
// `where` prefixes the error message (the recording's path).
FlagLevels flag_levels(const std::vector<std::int64_t>& flags, const std::string& where = "");

// Brainstorm's [0, 1] colour as 0-255 (a dictionary entry's `color_json`); a colour with fewer than 3 values is grey.
Rgb rgb255(const std::vector<double>& c);

// A `spans` Selection on a time Line (D87/D143): a member per interval [start, stop) in seconds (an instantaneous one start = stop), `code` = its type in the subject's
// dictionary named after the selection. `types` are the dictionary's names (each span's `code` indexes them); `colors` (Brainstorm's 0-1 colours) one per type, or none.
struct TimeSpan { double start = 0, stop = 0; std::int64_t type = 0; };
struct TimeSpansOptions {
  std::string line_id, of_field_id;
  std::optional<std::string> acquisition, acquisition_id;
  std::vector<std::vector<double>> colors;
  std::string description;
  std::optional<nxr::io::Json> params;
};
std::string time_spans(nxr::io::SubjectWriter& sub, const std::string& path, const std::vector<TimeSpan>& spans, const std::vector<std::string>& types, const TimeSpansOptions& opt);

}  // namespace nxr::bst
