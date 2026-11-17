// zlib and gzip streams as plain byte vectors (the compressed forms of NIfTI `.nii.gz` and GIFTI `GZipBase64Binary`), through zlib.
#pragma once
#include <cstddef>
#include <vector>

namespace nxr::formats {

enum class DeflateWrapper { zlib, gzip, detect };   // detect: gzip when the data start 1f 8b, else zlib

// Inflates the whole stream. `size_hint` pre-sizes the output (0 = unknown); `max_size` refuses a stream that inflates beyond it (a bomb guard).
// Throws code::format on a corrupt or truncated stream.
std::vector<unsigned char> inflate_bytes(const unsigned char* data, std::size_t n, DeflateWrapper w = DeflateWrapper::detect,
                                         std::size_t size_hint = 0, std::size_t max_size = std::size_t(1) << 33);

std::vector<unsigned char> deflate_bytes(const unsigned char* data, std::size_t n, DeflateWrapper w = DeflateWrapper::zlib, int level = 6);

}  // namespace nxr::formats
