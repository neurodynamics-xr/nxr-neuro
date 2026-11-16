// nxr-io — the Frequency manifold and the dataset's bands: `nxr.convert.frequency` (register D158). No manifold is made from a Selection: a frequency band is a
// Selection on a Frequency manifold. The manifold lives once, in the dataset's DEFAULT SUBJECT (`kind = 'template'`), at `frequency/Frequency`: a `line` in Hz,
// origin 0, spacing 0.5 Hz, 0-500 Hz (1001 samples). On it one `spans` Selection, `bands`: a span [lo, hi) per entry of the `frequency bands` dictionary.
//
//   auto ids = bst::ensure_frequency(default_subject_writer);          // get-or-create both (the default-subject builder calls it)
//   Json band = bst::band_ref(db, "alpha", 8, 13);                     // what a field records in producer_json["band"]
//   std::string node = bst::band_field_name("timefreq_srcpow_N2", "alpha");
//
// A writer in a dataset with no default subject records the band WITHOUT a selection id (null); it never makes a Frequency manifold elsewhere.
#pragma once
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/writer.h"

namespace nxr::bst {

constexpr const char* kFrequencyName = "Frequency";
constexpr const char* kBandsName = "bands";
constexpr double kFrequencySpacingHz = 0.5;
constexpr std::int64_t kFrequencySamples = 1001;   // 0-500 Hz

struct BandEntry { std::int64_t code = 0; std::string name; double lo_hz = 0, hi_hz = 0; };

// the folder of the Frequency manifold (`frequency`)
std::string frequency_folder();
// the dictionary's bands (code, name, edges), in order
std::vector<BandEntry> band_entries(nxr::io::Database& db);

struct FrequencyRows { nxr::io::Row manifold, bands; };
// the manifold and the `bands` Selection of the dataset's default subject, or nullopt when it has none
std::optional<FrequencyRows> find_frequency(nxr::io::Database& db);

struct FrequencyIds { std::string manifold_id, selection_id; };
// GET-OR-CREATE the Frequency manifold and its `bands` Selection in `sub`, which must be the default subject (a template).
FrequencyIds ensure_frequency(nxr::io::SubjectWriter& sub);

// a band as a field records it (`producer_json["band"]`): name, lo_hz, hi_hz, selection_id, code (the last two null when the dataset has no such band)
nxr::io::Json band_ref(nxr::io::Database& db, const std::string& name, double lo_hz, double hi_hz);
// `<stem>_<band>` as a node name
std::string band_field_name(const std::string& stem, const std::string& band);

}  // namespace nxr::bst
