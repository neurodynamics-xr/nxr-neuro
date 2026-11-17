// nxr-io — the sensors of a Brainstorm channel file as entities: `nxr.convert.sensors.export_sensors` with the full geometry of register D166.
//
// The sensor MANIFOLD stays one point per channel (`Loc(:,1)`; a bipolar electrode channel at the midpoint of its two contacts, `_contact1/2`). Around it,
// all in `sensor/` (the operator, which maps between two folders, at the subject's root): Field `<ch>_orientation` (`Orient(:,1)`), Manifold `<ch>_coils`
// (every `Loc` column of every non-bipolar channel) with Field `_coils_orientation` and Operator `<ch>_coil_weights` (csr [M x P], coils -> sensors,
// `Weight`), Manifold `<ch>_headpoints` with its label and type Selections, and the transforms (`TransfMeg`, `TransfEeg`, `SCS`) in the sensor manifold's
// `producer_json["transforms"]`. The channel file's variables are `read_all_variables`' map.
//
//   auto mat = bst::read_all_variables("channel.mat");
//   SensorInfo si = bst::export_sensors(sub, mat, flags, {.name = "channel", .acquisition = "rest"});
#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/bst/mat.h"
#include "nxr/io/writer.h"

namespace nxr::bst {

struct SensorOptions {
  std::string name;
  std::string acquisition;
  std::optional<std::string> source_sha1;
  std::string cs_key = "scs";                       // the coordinate system the positions are in (metres)
  std::optional<std::string> headpoints_cs_key;     // the system the digitised head points are in, when it is not the channels' (a head frame; default: `cs_key`)
  bool placed = true;                               // false: the channels have no positions, so the manifold has no geometry and stores none
  std::optional<nxr::io::Json> producer;            // rides in the manifold's provenance
  std::map<std::string, std::string> units;         // a unit for a channel type other than the default one (Neuromag planar gradiometers: T/m)
};

struct SensorInfo {
  std::string name, id;
  std::int64_t n_channels = 0, bipolar = 0;
  std::vector<std::string> modalities, names;
};

// `channel_flag`: Brainstorm's ChannelFlag (1 good, -1 bad), one per channel; empty (or another length) writes no `quality` family.
SensorInfo export_sensors(nxr::io::SubjectWriter& sub, const std::map<std::string, Value>& channel_mat, const std::vector<std::int64_t>& channel_flag,
                          const SensorOptions& opt);

// The unit of each channel type (`channel_units_for_types`).
std::vector<std::string> channel_units_for_types(const std::vector<std::string>& types, const std::map<std::string, std::string>& override_ = {});

}  // namespace nxr::bst
