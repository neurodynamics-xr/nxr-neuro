// Fixed-width loads and stores in a stated byte order, independent of the host's (memcpy, no aliasing).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace nxr::formats {

inline bool host_little() {
  const std::uint16_t one = 1;
  unsigned char b;
  std::memcpy(&b, &one, 1);
  return b == 1;
}

// Unsigned integer of `N` bytes (1..8) at `p`, little or big endian.
template <std::size_t N>
inline std::uint64_t load_uint(const unsigned char* p, bool little) {
  static_assert(N >= 1 && N <= 8, "1 to 8 bytes");
  std::uint64_t v = 0;
  if (little) for (std::size_t i = 0; i < N; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
  else        for (std::size_t i = 0; i < N; ++i) v = (v << 8) | p[i];
  return v;
}

inline std::uint16_t load_u16(const void* p, bool little) { return static_cast<std::uint16_t>(load_uint<2>(static_cast<const unsigned char*>(p), little)); }
inline std::uint32_t load_u32(const void* p, bool little) { return static_cast<std::uint32_t>(load_uint<4>(static_cast<const unsigned char*>(p), little)); }
inline std::uint64_t load_u64(const void* p, bool little) { return load_uint<8>(static_cast<const unsigned char*>(p), little); }
inline std::int16_t load_i16(const void* p, bool little) { return static_cast<std::int16_t>(load_u16(p, little)); }
inline std::int32_t load_i32(const void* p, bool little) { return static_cast<std::int32_t>(load_u32(p, little)); }
inline std::int64_t load_i64(const void* p, bool little) { return static_cast<std::int64_t>(load_u64(p, little)); }
inline float load_f32(const void* p, bool little) { const std::uint32_t u = load_u32(p, little); float f; std::memcpy(&f, &u, 4); return f; }
inline double load_f64(const void* p, bool little) { const std::uint64_t u = load_u64(p, little); double d; std::memcpy(&d, &u, 8); return d; }

// A signed 24-bit integer (BDF), sign-extended.
inline std::int32_t load_i24(const void* p, bool little) {
  std::uint32_t u = static_cast<std::uint32_t>(load_uint<3>(static_cast<const unsigned char*>(p), little));
  if (u & 0x800000u) u |= 0xFF000000u;
  return static_cast<std::int32_t>(u);
}

template <std::size_t N>
inline void store_uint(unsigned char* p, std::uint64_t v, bool little) {
  static_assert(N >= 1 && N <= 8, "1 to 8 bytes");
  if (little) for (std::size_t i = 0; i < N; ++i) p[i] = static_cast<unsigned char>(v >> (8 * i));
  else        for (std::size_t i = 0; i < N; ++i) p[N - 1 - i] = static_cast<unsigned char>(v >> (8 * i));
}
inline void store_u16(void* p, std::uint16_t v, bool little) { store_uint<2>(static_cast<unsigned char*>(p), v, little); }
inline void store_u32(void* p, std::uint32_t v, bool little) { store_uint<4>(static_cast<unsigned char*>(p), v, little); }
inline void store_u64(void* p, std::uint64_t v, bool little) { store_uint<8>(static_cast<unsigned char*>(p), v, little); }
inline void store_f32(void* p, float f, bool little) { std::uint32_t u; std::memcpy(&u, &f, 4); store_u32(p, u, little); }
inline void store_f64(void* p, double d, bool little) { std::uint64_t u; std::memcpy(&u, &d, 8); store_u64(p, u, little); }

}  // namespace nxr::formats
