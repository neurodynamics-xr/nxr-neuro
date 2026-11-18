// nxr-io — where an imported recording's channel positions are, in a TRUE coordinate system (docs/audits/2026-11-17-format-mapping-and-coordinates.md, gaps 1 and 2).
//
// A reader gives positions in the frame its file holds them in: a MEG array's own (device) frame, the Neuromag head, CTF's head, a unit sphere (BrainVision), EEGLAB's
// axes. `place_sensors` names that frame, carries every channel into ONE frame of the subject's head space when the file's own transform and landmarks allow it, and says
// which coordinate systems and transition operators the import must make with it:
//
//   - `device:<acquisition>` -> the file's head frame (`head`: Neuromag; `ctf-head`): the device-to-head transform the file holds (FIF dev_head_t, CTF's fit of its head
//     coils, KIT's fit of its marker coils to the digitised HPI points);
//   - the head frame -> `scs`, from the digitised nasion, left and right ear points (Brainstorm's definition of the subject coordinate system, which needs no MRI);
//   - with those, the positions are written in `scs` (one frame, comparable to the anatomy). A file without them keeps its own frame, in a system of its own that has no
//     transition: explicitly not comparable to anything.
//   - BrainVision's unit sphere and EEGLAB's axes are systems of their own with no transition. An empty position unit is never taken as metres: the system's unit is then unknown.
// Every system a link needs is made with its transition operator in the same compose (never an `affine` with no parameters).
#pragma once
#include <array>
#include <optional>
#include <string>
#include <vector>

#include "nxr/formats/fif.h"
#include "nxr/formats/recording.h"
#include "nxr/io/json_text.h"
#include "nxr/io/writer.h"

namespace nxr::convert {

struct SensorPlacement {
  struct System {
    std::string space, name;                   // the space (`head`: the subject's head space) and the system on it
    std::optional<std::string> unit;
    std::string description, space_description;
  };
  struct Link {                                // a transition operator between two systems of `systems` (or `scs`)
    std::string from, to, name, description;
    std::array<double, 12> matrix{};           // 3 x 4, row-major, metres to metres
    nxr::io::Json producer;
  };
  std::string cs_key;                          // the system the positions are written in; empty: no channel has a position
  std::vector<std::optional<std::array<double, 3>>> location, orientation;   // per channel, metres, in `cs_key`
  std::vector<System> systems;
  std::vector<Link> links;
  // the head frame the file digitised in (`head` or `ctf-head`), its landmarks (nasion, left, right) in it, and `scs` from them
  struct DigitisedPoint { std::array<double, 3> r; std::string type, label; };   // type: CARDINAL · HPI · EEG · EXTRA; metres, in the head frame
  std::vector<DigitisedPoint> digitisation;    // everything the file digitised (fiducials, HPI coils, EEG electrodes, head-shape points), as digitised
  std::string headpoints_cs_key;               // the head system the digitisation is in; empty when the placement made none (the points are then not imported)
  std::string head_key;
  std::optional<std::array<std::array<double, 3>, 3>> landmarks;
  std::optional<std::array<double, 16>> head_to_scs;
  nxr::io::Json record;                        // what was decided, for the producer record
};

// `base`: the channels' name (it prefixes the operators' names); `acquisition`: the acquisition's label, the device system's suffix.
// `header`: the header whose channels carry the positions when it is not the reader's own (BIDS: the electrodes table's positions, in a frame named `bids-<space>`); null: the reader's.
SensorPlacement place_sensors(const nxr::formats::RecordingReader& reader, const std::string& base, const std::string& acquisition, const nxr::formats::RecordingHeader* header = nullptr);

// Makes the systems of `p` and writes its links as transition operators from the sensor manifold `from_manifold_id`; call inside the compose, after the manifold is written.
void make_frames(nxr::io::SubjectWriter& sub, const SensorPlacement& p);
void write_links(nxr::io::SubjectWriter& sub, const SensorPlacement& p, const std::string& from_manifold_id, const std::optional<std::string>& acquisition);

// A head -> MRI transform (an MNE `-trans.fif`: head to MRI, translation in metres) as a transition operator from the head frame `head_key` to `freesurfer-tkr` (the MRI
// frame is FreeSurfer's surface RAS in metres), written from the sensor manifold `from_manifold_id`. Either direction in the file is accepted. Returns the operator's id.
std::string import_head_mri_transition(nxr::io::SubjectWriter& sub, const nxr::formats::fif::CoordTrans& t, const std::string& name, const std::string& from_manifold_id,
                                       const std::string& head_key = "head");

}  // namespace nxr::convert
