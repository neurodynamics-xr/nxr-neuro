// nxr-io — SHA-256 of a byte string as lowercase hex (hashlib.sha256(b).hexdigest()): a pipeline definition's identity in a producer record.
#pragma once
#include <string>

namespace nxr::io {
std::string sha256_hex(const std::string& data);
}  // namespace nxr::io
