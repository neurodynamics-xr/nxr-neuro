// nxr-io — the operators an imported recording brings with it (docs/audits/2026-11-17-format-mapping-and-coordinates.md, gaps 4 and 5): its SSP projectors and its CTF
// compensation, as Operators on the channels, in the shapes the Brainstorm importer writes (register D168, `nxr/bst/projectors.h`).
//
//   <channels>_projector_<k>     kind `projector`, dense [M x M]: x -> P x, one per projector in the file's order; params_json status (1 active, 0 not) · method · kind · components
//   <channels>_ctf_compensation  kind `ctf compensation`, csr [M x M]: entry (MEG channel, reference channel) = the file's coefficient, for the grade the data are at; params_json grade
//
//   <channels>_ica_unmixing      kind `ica unmixing`, dense [components x M]: x -> W S x (EEGLAB's icaweights times icasphere), the columns of the channels the ICA was run on
//   <channels>_ica_mixing        kind `ica mixing`, dense [M x components]: components -> x (EEGLAB's icawinv), the rows of those channels; neither is applied
//
// The samples are stored AS RECORDED: no projector is applied by the import. The chunk service applies the ACTIVE projectors to every block it loads (nxr/io/chunk_loop.h),
// and the recording's provenance lists each with its status. The compensation is NOT applied: it belongs to the forward model (G - C G_ref).
#pragma once
#include <cstdint>
#include <optional>
#include <utility>
#include <string>
#include <vector>

#include "nxr/bst/projectors.h"
#include "nxr/formats/recording.h"
#include "nxr/io/json_text.h"
#include "nxr/io/writer.h"

namespace nxr::convert {

struct CtfCompensation {
  int grade = 0;                                   // the grade the data are at (1, 2, 3)
  bool calibrated = true;                          // FIF: the file's `save_calibrated` flag
  std::int64_t n_meg = 0, n_ref = 0;
  std::vector<std::int32_t> indptr, indices;       // csr [M x M] over the channels
  std::vector<double> values;
};

struct RecordingOperators {
  std::vector<nxr::bst::ProjectorEntry> projectors;
  std::optional<CtfCompensation> compensation;
  std::optional<std::pair<std::int64_t, std::vector<double>>> ica_unmixing, ica_mixing;   // (components, dense row-major [components x M] / [M x components])
  nxr::io::Json record = nxr::io::Json::object();  // `projectors` and `ctf_compensation`, for the recording's producer record (after `write_operators`)
};

// What the file holds. Throws when a projector or a compensation cannot be mapped onto the recording's channels (an uncalibrated compensation is refused).
RecordingOperators operators_of(const nxr::formats::RecordingReader& reader);


// Writes the operators on the sensor manifold `channels_id` (inside the compose that writes it) and fills `ops.record`.
void write_operators(nxr::io::SubjectWriter& sub, RecordingOperators& ops, const std::string& channels_name, const std::string& channels_id, const std::string& acquisition);

}  // namespace nxr::convert
