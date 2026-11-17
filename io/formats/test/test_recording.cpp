// The recording header and the block-read contract, with a toy reader over a synthetic signal: a block read gives the same values whatever
// the block size, and a range outside the recording is refused.
#include <cmath>
#include <functional>
#include <vector>

#include "check.h"
#include "nxr/formats/recording.h"

using namespace nxr::formats;

namespace {
class Toy final : public RecordingReader {
 public:
  Toy(std::size_t nch, std::uint64_t n) {
    h_.format = "toy";
    h_.sampling_rate = 100.0;
    h_.n_samples = n;
    for (std::size_t c = 0; c < nch; ++c) { Channel ch; ch.name = "c" + std::to_string(c); ch.unit = "V"; h_.channels.push_back(ch); }
  }
  const RecordingHeader& header() const override { return h_; }
  void read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, float* out) const override {
    check_range(s0, n, c0, c1);
    for (std::size_t c = c0; c < c1; ++c)
      for (std::uint64_t i = 0; i < n; ++i) out[(c - c0) * n + i] = static_cast<float>(c * 1000 + (s0 + i));
  }
 private:
  RecordingHeader h_;
};

bool throws_code(const std::function<void()>& f, const char* code) {
  try { f(); } catch (const nxr::io::Error& e) { return e.code() == code; }
  return false;
}
}  // namespace

int main() {
  Toy r(3, 1000);
  std::vector<float> whole(2 * 1000);
  r.read(0, 1000, 1, 3, whole.data());
  CHECK_EQ(whole[0], 1000.0f);
  CHECK_EQ(whole[1000 + 999], 2999.0f);
  for (std::uint64_t block : {1ull, 7ull, 64ull, 999ull, 1000ull, 5000ull}) {
    std::vector<float> got(2 * 1000, -1.0f);
    for_each_block(r, 0, 1000, 1, 3, block, [&](std::uint64_t first, std::uint64_t m, const float* rows) {
      for (std::size_t c = 0; c < 2; ++c)
        for (std::uint64_t i = 0; i < m; ++i) got[c * 1000 + first + i] = rows[c * m + i];
    });
    CHECK(got == whole);   // independent of the block size
  }
  std::vector<float> buf(10);
  CHECK(throws_code([&] { r.read(995, 10, 0, 1, buf.data()); }, code::out_of_range));
  CHECK(throws_code([&] { r.read(0, 5, 2, 4, buf.data()); }, code::out_of_range));
  CHECK(throws_code([&] { for_each_block(r, 0, 5, 0, 1, 0, [](std::uint64_t, std::uint64_t, const float*) {}); }, code::out_of_range));
  r.read(1000, 0, 0, 0, buf.data());   // an empty read at the end is fine
  return nxrtest::finish("formats_recording");
}
