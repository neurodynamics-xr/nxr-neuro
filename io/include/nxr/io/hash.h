// nxr-io — SHA-256 as lowercase hex (hashlib.sha256(b).hexdigest()): a pipeline definition's identity in a producer record, a key's digest in a subject package.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace nxr::io {
std::string sha256_hex(const std::string& data);

// The same hash taken piece by piece (a shard streamed to its key is hashed as it is written, never held whole).
class Sha256 {
 public:
  Sha256();
  void update(const std::uint8_t* data, std::size_t n);
  void update(const std::string& s) { update(reinterpret_cast<const std::uint8_t*>(s.data()), s.size()); }
  std::uint64_t size() const { return total_; }
  // The digest of everything given so far; the object is spent after it.
  std::string hex();

 private:
  void block(const std::uint8_t* p);
  std::array<std::uint32_t, 8> h_;
  std::array<std::uint8_t, 64> buf_{};
  std::size_t fill_ = 0;
  std::uint64_t total_ = 0;
};
}  // namespace nxr::io
