// crc32c (Castagnoli), table-driven — the `crc32c` codec and a shard index's checksum.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nxr::io::detail {

inline std::uint32_t crc32c(const std::uint8_t* p, std::size_t n) {
  static const auto table = [] {
    std::vector<std::uint32_t> t(256);
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
      t[i] = c;
    }
    return t;
  }();
  std::uint32_t c = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

}  // namespace nxr::io::detail
