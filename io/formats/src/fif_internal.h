// Shared by the FIF sources only (not installed): the decoders of the structured tag types and a few helpers.
#pragma once
#include <string>
#include <vector>

#include "nxr/formats/fif.h"

namespace nxr::formats::fif::detail {

Id decode_id(const std::vector<unsigned char>& b);
ChannelInfo decode_ch_info(const std::vector<unsigned char>& b);
DigPoint decode_dig_point(const unsigned char* p);                           // 20 bytes
std::vector<DigPoint> decode_dig_string(const std::vector<unsigned char>& b);   // kind, ident, n, then n points
CoordTrans decode_coord_trans(const std::vector<unsigned char>& b);
CoordTrans invert(const CoordTrans& t);
std::string latin1_to_utf8(const unsigned char* p, std::size_t n);
std::vector<std::string> split_name_list(const std::string& s);              // ':'-separated, "{COLON}" a literal colon (MNE's name lists)

}  // namespace nxr::formats::fif::detail
