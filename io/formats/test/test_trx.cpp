// TRX (nxr/formats/trx.h) on archives made here: the three position types and both offset conventions (N and N + 1), the half-precision conversion for every half, the entries
// that are not read (named), and the refusals. There is no reference reader for TRX in the test tools: the archives are written by this writer and, for the odd cases, byte by byte.
#include <cmath>
#include <cstring>

#include "check.h"
#include "nxr/formats/errors.h"
#include "nxr/formats/trx.h"
#include "nxr/formats/zip.h"

using namespace nxr::formats;
using Bytes = std::vector<unsigned char>;

static std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const nxr::io::Error& e) { return e.code(); } catch (const std::exception&) { return "other"; }
  return "";
}

static trx::Streamlines make() {
  trx::Streamlines s;
  s.offsets = {0, 4, 7, 12};
  for (int i = 0; i < 12; ++i) for (int k = 0; k < 3; ++k) s.points.push_back(0.37f * static_cast<float>(i) - 1.9f * static_cast<float>(k) + 0.0123f * static_cast<float>(i * k));
  s.header.dimensions = {90, 108, 90};
  s.header.voxel_to_rasmm = {-2, 0, 0, 90, 0, 2, 0, -126, 0, 0, 2, -72, 0, 0, 0, 1};
  return s;
}

static Bytes written(const trx::Streamlines& s, const trx::WriteOptions& o) {
  MemorySink sink;
  trx::write(sink, s, o);
  return sink.data();
}

static trx::Streamlines read_bytes(const Bytes& b) { return trx::read(std::make_shared<MemorySource>(b)); }

int main() {
  // every half is exactly a single, and back
  for (std::uint32_t h = 0; h < 65536; ++h) {
    const std::uint16_t x = static_cast<std::uint16_t>(h);
    if (((x >> 10) & 0x1F) == 31 && (x & 0x3FF)) continue;     // a NaN's payload is not kept
    CHECK_EQ(trx::float_to_half(trx::half_to_float(x)), x);
  }
  CHECK(trx::half_to_float(0x3C00) == 1.0f && trx::half_to_float(0xC000) == -2.0f && trx::half_to_float(0x7BFF) == 65504.0f);
  CHECK(trx::half_to_float(0x0001) == std::ldexp(1.0f, -24));
  CHECK(trx::float_to_half(1e6f) == 0x7C00 && trx::float_to_half(-1e6f) == 0xFC00);                  // overflow: infinity
  CHECK(trx::float_to_half(1.0f + std::ldexp(1.0f, -11)) == 0x3C00);                                   // a tie rounds to even
  CHECK(trx::float_to_half(1.0f + 3 * std::ldexp(1.0f, -11)) == 0x3C02);

  const trx::Streamlines s = make();
  for (const char* pd : {"float32", "float16", "float64"}) {
    for (const char* od : {"uint64", "uint32"}) {
      for (const bool with_end : {true, false}) {
        trx::WriteOptions o;
        o.positions_dtype = pd;
        o.offsets_dtype = od;
        o.offsets_with_end = with_end;
        const trx::Streamlines r = read_bytes(written(s, o));
        CHECK_EQ(r.header.n_streamlines, static_cast<std::int64_t>(3));
        CHECK_EQ(r.header.n_vertices, static_cast<std::int64_t>(12));
        CHECK(r.offsets == s.offsets);
        CHECK_EQ(r.positions_dtype, std::string(pd));
        CHECK_EQ(r.offsets_dtype, std::string(od));
        CHECK_EQ(r.points.size(), s.points.size());
        for (std::size_t i = 0; i < s.points.size(); ++i) {
          const float want = std::string(pd) == "float16" ? trx::half_to_float(trx::float_to_half(s.points[i])) : s.points[i];
          CHECK(r.points[i] == want);
        }
        CHECK(r.header.dimensions == s.header.dimensions && r.header.voxel_to_rasmm == s.header.voxel_to_rasmm);
        CHECK(r.other_entries.empty());
      }
    }
  }

  // entries that are not read are named
  {
    MemorySink sink;
    trx::write(sink, s, {});
    auto z = zip::Reader::open(std::make_shared<MemorySource>(sink.data()));
    std::vector<zip::NewEntry> all;
    for (const auto& e : z.entries()) all.push_back({e.name, z.read(e)});
    all.push_back({"dps/weight.float32", Bytes(12, 0)});
    all.push_back({"groups/bundle1.uint32", Bytes(8, 0)});
    MemorySink out;
    zip::write(out, all);
    const trx::Streamlines r = read_bytes(out.data());
    CHECK_EQ(r.other_entries.size(), static_cast<std::size_t>(2));
    CHECK(r.other_entries[0] == "dps/weight.float32");
  }

  // refusals
  auto with = [&](const std::function<void(std::vector<zip::NewEntry>&)>& edit) {
    MemorySink sink;
    trx::write(sink, s, {});
    auto z = zip::Reader::open(std::make_shared<MemorySource>(sink.data()));
    std::vector<zip::NewEntry> all;
    for (const auto& e : z.entries()) all.push_back({e.name, z.read(e)});
    edit(all);
    MemorySink out;
    zip::write(out, all);
    return out.data();
  };
  auto set_header = [](std::vector<zip::NewEntry>& all, const std::function<void(nlohmann::json&)>& f) {
    nlohmann::json j = nlohmann::json::parse(all[0].data.begin(), all[0].data.end());
    f(j);
    const std::string t = j.dump();
    all[0].data.assign(t.begin(), t.end());
  };
  CHECK_EQ(code_of([&] { read_bytes(with([](auto& a) { a.erase(a.begin()); })); }), std::string(code::format));                      // no header.json
  CHECK_EQ(code_of([&] { read_bytes(with([](auto& a) { a.erase(a.begin() + 1); })); }), std::string(code::format));                  // no positions
  CHECK_EQ(code_of([&] { read_bytes(with([](auto& a) { a.erase(a.begin() + 2); })); }), std::string(code::format));                  // no offsets
  CHECK_EQ(code_of([&] { read_bytes(with([&](auto& a) { set_header(a, [](auto& j) { j["NB_VERTICES"] = 13; }); })); }), std::string(code::format));
  CHECK_EQ(code_of([&] { read_bytes(with([&](auto& a) { set_header(a, [](auto& j) { j["NB_STREAMLINES"] = 5; }); })); }), std::string(code::format));
  CHECK_EQ(code_of([&] { read_bytes(with([&](auto& a) { set_header(a, [](auto& j) { j.erase("VOXEL_TO_RASMM"); }); })); }), std::string(code::format));
  CHECK_EQ(code_of([&] { read_bytes(with([](auto& a) { a[1].name = "positions.3.int8"; })); }), std::string(code::refused));
  CHECK_EQ(code_of([&] { read_bytes(with([](auto& a) { std::uint64_t v = 99; std::memcpy(&a[2].data[8], &v, 8); })); }), std::string(code::format));   // decreasing offsets
  CHECK_EQ(code_of([&] { read_bytes(with([](auto& a) { std::uint64_t v = 5; std::memcpy(&a[2].data[0], &v, 8); })); }), std::string(code::format));    // the first is not 0
  CHECK_EQ(code_of([&] { read_bytes(with([](auto& a) { a[1].data[3] = 0x7F; a[1].data[2] = 0x80; })); }), std::string(code::format));               // a position that is not finite
  CHECK_EQ(code_of([&] { read_bytes(Bytes(64, 1)); }), std::string(code::format));                                                  // not a ZIP archive
  CHECK_EQ(code_of([&] { trx::Streamlines bad = s; bad.offsets.back() = 11; MemorySink k; trx::write(k, bad, {}); }), std::string(code::format));   // the writer refuses what disagrees
  return nxrtest::finish("formats_trx");
}
