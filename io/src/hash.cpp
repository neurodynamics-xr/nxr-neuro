// nxr-io — hash.h: the internal SHA-256 (sha256.h) made public.
#include "nxr/io/hash.h"

#include "sha256.h"

namespace nxr::io {
std::string sha256_hex(const std::string& data) { return detail::sha256_hex(data); }
}  // namespace nxr::io
