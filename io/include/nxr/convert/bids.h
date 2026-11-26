// nxr-io — IMPORT OF A BIDS DATASET (docs/specs/2026-11-17-interoperability.md, section 3.2): a BIDS folder (MEG, EEG, iEEG) becomes a DATASET (a catalogue) of subject
// PACKAGES, one per `sub-<label>`, made with the pure BIDS layer (`nxr/formats/bids.h`) and the recording importer (`nxr/convert/recording.h`).
//
//   nxr::convert::BidsImport opt;  opt.registered_by = "nxr-neuro";
//   Json out = nxr::convert::import_bids("/data/ds-rest", "/work/ds-rest.nxr", opt);
//
//   participants.tsv / participants.json   the dataset's participant variables (`participants_import`: age, sex, group, ... so `--where` works on them)
//   a recording (meg, eeg, ieeg)           imported through `import_recording` (copied, or linked with `in_place`), named by its entities (`ses-01_task-rest_run-01`), in the
//                                          acquisition `ses-<label>` (`main` for a subject without sessions)
//   <rec>_channels.tsv                     the channels' TYPES and STATUS: a channel whose status is `bad` is a bad channel (the recording's channel flags)
//   <rec>_events.tsv                       the events: `onset`, `duration` and the label (`trial_type`, else `value`) become the recording's events, as Selections on the time line;
//                                          they replace the data file's own markers (bad-segment markers, labelled BAD..., are kept)
//   <rec>_electrodes.tsv + coordsystem     the channels' POSITIONS (the unit the coordinate system states) on the sensor manifold; the coordinate system's record is kept
//                                          VERBATIM in the recording's producer record (`bids.coordinate_system`): this importer does not interpret or transform a frame
//   the merged sidecar                     kept in the producer record (`bids.sidecar`), without the institution and device-serial keys when `anonymize` is set
// The dataset's licence and authors are the origin record of every item. A recording of a format this build does not read, or a split recording, is listed in `skipped`
// with its reason; everything else is imported or the whole import is refused (a participant table that does not match the subjects, a malformed table).
#pragma once
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "nxr/bst/session.h"
#include "nxr/formats/recording.h"
#include "nxr/formats/tsv.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/json_text.h"

namespace nxr::convert {

struct BidsImport {
  std::string registered_by = "nxr-neuro";   // recorded with each catalogue entry
  bool in_place = false;                     // link the recordings' files instead of copying their samples
  bool anonymize = false;                    // drop the identifying header fields (see import_recording) and the sidecar's institution and device keys
  std::vector<std::string> subjects;         // labels (without `sub-`); empty: all
  bool participants = true;                  // import participants.tsv (needs `participants.tsv` at the BIDS root)
  nxr::bst::Emit progress;
};

// The header of a recording the way the BIDS tables describe it: channel types and status from `channels`, positions from `electrodes` (in `position_unit`), events from
// `events` (the data file's own non-BAD markers dropped). Any table may be absent (null). `import_bids` hands it to `import_recording` (RecordingImport::header); exposed for the tests.
nxr::formats::RecordingHeader bids_header(const nxr::formats::RecordingHeader& base, const nxr::formats::tsv::Table* channels, const nxr::formats::tsv::Table* events,
                                          const nxr::formats::tsv::Table* electrodes, const std::string& position_unit,
                                          const std::string& position_space = std::string());   // `position_space`: the coordinate system's name (the electrodes' `space` entity); the positions stay in it (frame `bids-<space>`)

// Import `bids_root` into the new dataset folder `dataset_folder` (refused when it holds a catalogue). Returns {dataset, subjects, recordings, skipped, participants, packages}.
nxr::io::Json import_bids(const std::filesystem::path& bids_root, const std::filesystem::path& dataset_folder, const BidsImport& opt);

}  // namespace nxr::convert
