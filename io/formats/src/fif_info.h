// The measurement-info and annotation readers shared by the FIF sources (not installed).
#pragma once
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/formats/fif.h"

namespace nxr::formats::fif {

// The measurement info of the (single) measurement block `meas` of `f`.
Info read_info(const File& f, const Block& meas);
// The first SSP projector block found at or below `scope` (names mapped through `renamed`, MNE's long-name map); empty when there is none.
std::vector<Projector> read_projectors(const File& f, const Block& scope, const std::map<std::string, std::string>& renamed = {});
// MNE's extended channel information (long names, one block per channel) applied to `channels`; returns the old-name to new-name map.
std::map<std::string, std::string> apply_extended_channel_info(const File& f, const Block& scope, std::vector<ChannelInfo>& channels);
// The coordinate frame a channel of this kind is in when its own tag does not say (MEG: device; EEG and the like: head).
std::int32_t channel_coord_frame(std::int32_t kind);
// The first annotation block anywhere in the tree, if any.
std::optional<AnnotationSet> read_annotations(const File& f, const Block& root);

}  // namespace nxr::formats::fif
