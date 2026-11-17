// nxr-io — deterministic ids and clock, for the conformance and parity tests only (no database needed to switch them on).
#pragma once
#include <cstdint>

namespace nxr::io::testing {
// Deterministic ids (00000000-0000-7000-8000-<counter>) and a fixed clock, so two writers' output can be compared byte for byte.
// Used only by the conformance tests.
void deterministic(bool on, std::uint64_t start = 0);   // `start`: the counter's first value (ids continue after it)
}  // namespace nxr::io::testing
