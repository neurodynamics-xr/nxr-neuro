// nxr::formats: the failures a pure format reader or writer reports, as nxr::io::Error with a stable code.
//
//   code::format       the bytes are not the format they claim to be (a bad magic, a truncated header, a field out of range)
//   code::refused      a valid file this reader does not take, with the reason (an epoched recording, an ASCII BrainVision file)
//   code::out_of_range a block read asks for samples or channels the recording does not have
//   code::io           the file cannot be opened, read or written
//
// `refused` is nxr::io's `unsupported`, so a caller that already handles it handles a refusal here.
#pragma once
#include <string>

#include "nxr/io/errors.h"   // the one nxr/io header a pure format may include (cmake/check_formats_layering.cmake)

namespace nxr::formats {

namespace code {
constexpr const char* format = "format_error";
constexpr const char* refused = nxr::io::code::unsupported;
constexpr const char* out_of_range = nxr::io::code::invalid_argument;
constexpr const char* io = nxr::io::code::io;
}  // namespace code

inline nxr::io::Error format_error(const std::string& what) { return nxr::io::Error(code::format, what); }
inline nxr::io::Error refused(const std::string& what) { return nxr::io::Error(code::refused, what); }
inline nxr::io::Error range_error(const std::string& what) { return nxr::io::Error(code::out_of_range, what); }
inline nxr::io::Error io_error(const std::string& what) { return nxr::io::Error(code::io, what); }

}  // namespace nxr::formats
