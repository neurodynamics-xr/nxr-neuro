// .stc source estimates against MNE-Python (test/golden/gen_stc.py): the reader reads MNE's files, the writer writes the same bytes, the streaming writer equals the whole-file
// writer, and the refusals.
#include <fstream>
#include <functional>
#include <iterator>

#include "check.h"
#include "nlohmann/json.hpp"
#include "nxr/formats/errors.h"
#include "nxr/formats/stc.h"

using namespace nxr::formats;
using nlohmann::json;

static const std::string kData = std::string(NXR_FORMATS_TEST_DIR) + "/data/";

static bool throws_code(const std::function<void()>& f, const char* code) {
  try { f(); } catch (const nxr::io::Error& e) { return e.code() == code; }
  return false;
}
static std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); return std::string(std::istreambuf_iterator<char>(f), {}); }

int main() {
  std::ifstream gf(std::string(NXR_FORMATS_TEST_DIR) + "/golden/stc_golden.json");
  const json g = json::parse(gf);
  const stc::Estimate left = stc::read_file(kData + "stc_basic-lh.stc");
  const stc::Estimate right = stc::read_file(kData + "stc_basic-rh.stc");
  CHECK_EQ(left.tmin, g["tmin"].get<double>());
  CHECK_EQ(left.tstep, g["tstep"].get<double>());
  CHECK_EQ(left.n_times, 4u);
  CHECK_EQ(left.n_vertices(), g["left"].size());
  CHECK_EQ(right.n_vertices(), g["right"].size());
  for (std::size_t i = 0; i < left.n_vertices(); ++i) CHECK_EQ(left.vertices[i], g["left"][i].get<std::uint32_t>());
  for (std::size_t i = 0; i < right.n_vertices(); ++i) CHECK_EQ(right.vertices[i], g["right"][i].get<std::uint32_t>());
  for (std::size_t v = 0; v < left.n_vertices(); ++v)
    for (std::size_t t = 0; t < 4; ++t) CHECK_EQ(left.at(v, t), static_cast<float>(g["data"][v][t].get<double>()));
  for (std::size_t v = 0; v < right.n_vertices(); ++v)
    for (std::size_t t = 0; t < 4; ++t) CHECK_EQ(right.at(v, t), static_cast<float>(g["data"][left.n_vertices() + v][t].get<double>()));

  // the writer writes MNE's bytes
  MemorySink sink;
  stc::write(sink, left);
  const std::string mine(reinterpret_cast<const char*>(sink.data().data()), sink.data().size());
  CHECK(mine == slurp(kData + "stc_basic-lh.stc"));

  // frame by frame, the same file
  MemorySink streamed;
  {
    stc::Writer w(streamed, left.tmin, left.tstep, left.vertices, left.n_times);
    for (std::uint32_t t = 0; t < left.n_times; ++t) w.frame(left.data.data() + t * left.n_vertices());
    w.finish();
  }
  CHECK(streamed.data() == sink.data());

  // refusals
  {
    MemorySink s;
    stc::Writer w(s, 0.0, 0.1, {0, 1}, 2);
    const float f[2] = {1, 2};
    w.frame(f);
    CHECK(throws_code([&] { w.finish(); }, code::out_of_range));                      // one frame short
    w.frame(f);
    CHECK(throws_code([&] { w.frame(f); }, code::out_of_range));                      // one too many
    CHECK(throws_code([&] { stc::Writer(s, 0.0, 0.1, {2, 1}, 1); }, code::out_of_range));       // vertices not increasing
    CHECK(throws_code([&] { stc::Writer(s, 0.0, 0.0, {0}, 1); }, code::out_of_range));          // no time step
    CHECK(throws_code([&] { stc::Writer(s, 0.0, 0.1, {}, 1); }, code::out_of_range));           // no vertices
  }
  {
    const std::string cut = slurp(kData + "stc_basic-lh.stc").substr(0, 30);
    CHECK(throws_code([&] { stc::read(MemorySource(cut)); }, code::format));
    CHECK(throws_code([&] { stc::read(MemorySource(std::string("abc"))); }, code::format));
  }
  return nxrtest::finish("formats_stc");
}
