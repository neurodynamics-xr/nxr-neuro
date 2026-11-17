// Shared by the recording-format tests: the golden's raw float32 files and the checks every reader must pass (block reads independent of the
// block size, channel subsets, and values equal to the reference's).
#pragma once
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

#include "check.h"
#include "nxr/formats/errors.h"
#include "nxr/formats/recording.h"

namespace rg {

inline bool throws_code(const std::function<void()>& f, const char* code) {
  try { f(); } catch (const nxr::io::Error& e) { return e.code() == code; }
  return false;
}

// channel after channel, samples in order
inline std::vector<float> load_f32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::printf("  cannot open golden %s\n", path.c_str()); ++nxrtest::failures(); return {}; }
  std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  std::vector<float> v(b.size() / 4);
  std::memcpy(v.data(), b.data(), v.size() * 4);
  return v;
}

inline bool close(float a, float b, double rel = 2e-6, double abs_ = 1e-12) { return std::fabs(static_cast<double>(a) - b) <= rel * std::fabs(static_cast<double>(b)) + abs_; }

// Reads the whole recording, then in blocks of several sizes and over channel subsets, and compares with `want` (channel rows of n_samples).
inline void check_against(const nxr::formats::RecordingReader& r, const std::vector<float>& want, const char* what, double rel = 2e-6, double abs_ = 1e-12) {
  using namespace nxr::formats;
  const RecordingHeader& h = r.header();
  const std::size_t nc = h.channels.size();
  const std::uint64_t n = h.n_samples;
  if (want.size() != nc * n) { std::printf("  %s: golden has %zu values, expected %zu x %llu\n", what, want.size(), nc, static_cast<unsigned long long>(n)); ++nxrtest::failures(); return; }
  for (std::uint64_t block : {std::uint64_t(1), std::uint64_t(7), std::uint64_t(100), std::uint64_t(333), n ? n : std::uint64_t(1)}) {
    std::vector<float> got(nc * n, -777.0f);
    for_each_block(r, 0, n, 0, nc, block, [&](std::uint64_t first, std::uint64_t m, const float* rows) {
      for (std::size_t c = 0; c < nc; ++c)
        for (std::uint64_t i = 0; i < m; ++i) got[c * n + first + i] = rows[c * m + i];
    });
    bool ok = true;
    for (std::size_t i = 0; ok && i < got.size(); ++i) ok = close(got[i], want[i], rel, abs_);
    if (!ok) std::printf("  %s: values differ from the reference (block %llu)\n", what, static_cast<unsigned long long>(block));
    CHECK(ok);
  }
  // a channel subset at an offset
  if (nc >= 2 && n >= 10) {
    const std::uint64_t s0 = n / 3, m = n / 2;
    std::vector<float> got((nc - 1) * m);
    r.read(s0, m, 1, nc, got.data());
    bool ok = true;
    for (std::size_t c = 1; ok && c < nc; ++c)
      for (std::uint64_t i = 0; ok && i < m; ++i) ok = close(got[(c - 1) * m + i], want[c * n + s0 + i], rel, abs_);
    CHECK(ok);
  }
  // outside the recording
  std::vector<float> junk(4);
  CHECK(throws_code([&] { r.read(n, 1, 0, 1, junk.data()); }, code::out_of_range));
  CHECK(throws_code([&] { r.read(0, 1, 0, nc + 1, junk.data()); }, code::out_of_range));
}

}  // namespace rg
