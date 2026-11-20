// `read <class> --region` and `--budget` (data-layer queries spec, 2: one node, a region at a time): a part of each array the read gives, [start, stop) per leading axis, a stop past an
// array's extent cut to it; a read above its budget refused before anything is read, naming its size and the region that fits; a time-resolved field read by its rows and time range,
// never by a region. And `read selection` of a tiling: its spans as start, stop and code, made from its rule. Synthetic data only.
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/npy.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/task/task.h"
#include "nxr/pipelines/face.h"

using namespace nxr::io;
namespace face = nxr::pipelines::face;
namespace fs = std::filesystem;

namespace {
template <class T>
std::string npy(const fs::path& file, const std::string& dtype, const std::vector<std::int64_t>& shape, const std::vector<T>& v) {
  std::vector<std::uint8_t> b(v.size() * sizeof(T));
  if (!v.empty()) std::memcpy(b.data(), v.data(), b.size());
  write_npy(file, dtype, shape, b);
  return file.string();
}
std::string thrown(const std::function<void()>& f) {
  try { f(); } catch (const task::Refused& e) { return std::string("refused: ") + e.what(); } catch (const Error& e) { return e.code() + std::string(": ") + e.what(); }
  return "";
}
bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
template <class T>
std::vector<T> values(const task::CommandArray& a) {
  std::vector<T> v(a.bytes.size() / sizeof(T));
  if (!v.empty()) std::memcpy(v.data(), a.bytes.data(), a.bytes.size());
  return v;
}
const task::CommandArray* array(const task::CommandResult& r, const std::string& name) {
  for (const auto& a : r.arrays) if (a.name == name) return &a;
  return nullptr;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_read_region_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "in");
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const fs::path folder = face::synthetic_dataset(root, "demo", "S1", "rest", 3, 100.0, 40.0);
  const std::string D = folder.string(), program = face::stamp();
  auto base = [&](auto& r, const std::string& name) { r.dataset = D; r.subject = "S1"; r.name = name; r.created_by = program; };

  // a cloud of 1000 points (positions [1000 x 3], stored as float64: 24000 bytes), a statistic, a field, a partition, and a tiling of the recording's time line
  const std::int64_t N = 1000;
  std::vector<double> pos(static_cast<std::size_t>(3 * N));
  for (std::size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<double>(i);
  {
    task::AddPointsRequest p;
    base(p, "cloud");
    p.positions = npy(root / "in" / "cloud.npy", "float64", {N, 3}, pos);
    task::add_points(p);
    task::AddLineRequest fl;
    base(fl, "freq");
    fl.n = 4;
    fl.spacing = 1.0;
    fl.unit = "Hz";
    fl.dimension = "frequency";
    task::add_line(fl);
    task::AddStatisticRequest st;
    base(st, "psd");
    st.measure = "power_density";
    st.method = "welch";
    st.axes = {"channel:sensors", "frequency:freq"};
    st.of_field = "rest";
    std::vector<float> psd(12);
    for (int i = 0; i < 12; ++i) psd[static_cast<std::size_t>(i)] = static_cast<float>(i);
    st.values = npy(root / "in" / "psd.npy", "float32", {3, 4}, psd);
    task::add_statistic(st);
    task::AddFieldRequest f;
    base(f, "w");
    f.manifold = "sensors";
    f.values = npy(root / "in" / "w.npy", "float64", {3}, std::vector<double>{1, 2, 3});
    task::add_field(f);
    // the tiling: 8 tiles of 2 s, one every 1 s from 0, the line ending at 10 s
    DatasetWriter ds = open_writer(folder);
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    const task::Subject ts = task::Subject::of(sub);
    std::string line;
    if (const auto m = ts.by_id_or_name(task::Cls::manifold, "rest_time")) line = m->id;   // the recording's time line
    CHECK(!line.empty());
    task::Unit u = ts.unit(task::By{"unit test", Json::object()});
    task::SelectionOut tiles;
    tiles.name = "segments";
    tiles.output = "segments";
    tiles.type = "tiling";
    tiles.manifold_id = line;
    tiles.tiling = task::Tiling{0.0, 2.0, 1.0, 10.0, 8, 0};
    u.selection(tiles);
    u.commit();
  }
  auto rq = [&](const std::string& name) { face::ReadNodeRequest r; r.dataset = D; r.subject = "S1"; r.name = name; return r; };

  {   // a manifold: its positions, a part of them, and its budget
    const auto whole = face::read_manifold(rq("cloud"));
    CHECK(array(whole, "positions") && array(whole, "positions")->shape == (std::vector<std::int64_t>{N, 3}));
    auto r = rq("cloud");
    r.region = {{10, 20}};
    const auto part = face::read_manifold(r);
    const auto* a = array(part, "positions");
    CHECK(a && a->shape == (std::vector<std::int64_t>{10, 3}));
    if (a) CHECK(values<double>(*a) == std::vector<double>(pos.begin() + 30, pos.begin() + 60));
    CHECK(part.result["regions"]["positions"] == Json::parse("[[10, 20], [0, 3]]"));
    r.region = {{990, 5000}};   // a stop past the extent is cut to it
    const auto tail = face::read_manifold(r);
    CHECK(array(tail, "positions") && array(tail, "positions")->shape == (std::vector<std::int64_t>{10, 3}));
    r.region = {{0, 1000}, {1, 3}};
    const auto cols = face::read_manifold(r);
    CHECK(array(cols, "positions") && array(cols, "positions")->shape == (std::vector<std::int64_t>{N, 2}));
    r.region = {{5, 2}};
    CHECK(has(thrown([&] { face::read_manifold(r); }), "START <= STOP"));
    r.region = {{0, 1}, {0, 1}, {0, 1}};
    CHECK(has(thrown([&] { face::read_manifold(r); }), "names 3 axes"));
    auto b = rq("cloud");
    b.budget = 1000;   // the whole cloud is 24000 bytes
    const std::string refused = thrown([&] { face::read_manifold(b); });
    CHECK(has(refused, "refused: 'cloud' as asked is 24000 bytes"));
    CHECK(has(refused, "positions [1000 x 3] float64"));
    CHECK(has(refused, "--region 0:41"));   // 1000 bytes hold 41 rows of 24 bytes
    b.region = {{0, 41}};
    CHECK(array(face::read_manifold(b), "positions") != nullptr);
    b.budget = 0;
    CHECK(has(thrown([&] { face::read_manifold(b); }), "--budget"));
  }
  {   // a statistic in its stored precision, and a part of it
    auto r = rq("psd");
    r.region = {{1, 3}};
    const auto part = face::read_statistic(r);
    const auto* a = array(part, "values");
    CHECK(a && a->dtype == "float32" && a->shape == (std::vector<std::int64_t>{2, 4}));
    if (a) CHECK(values<float>(*a) == (std::vector<float>{4, 5, 6, 7, 8, 9, 10, 11}));
    const auto whole_read = face::read_statistic(rq("psd"));
    const auto* whole = array(whole_read, "values");
    CHECK(whole && whole->dtype == "float32" && whole->shape == (std::vector<std::int64_t>{3, 4}));
    auto b = rq("psd");
    b.budget = 10;
    CHECK(has(thrown([&] { face::read_statistic(b); }), "above the read's budget"));
  }
  {   // a field with no time line; a time-resolved field takes rows and a time range, never a region, and its budget names the time range that fits
    auto r = rq("w");
    r.region = {{1, 2}};
    const auto got = face::read_field(r);
    const auto* a = array(got, "values");
    CHECK(a && a->shape == (std::vector<std::int64_t>{1}) && values<double>(*a) == std::vector<double>{2});
    auto s = rq("rest");
    s.all = true;
    s.region = {{0, 1}};
    CHECK(has(thrown([&] { face::read_field(s); }), "not --region"));
    s.region.clear();
    s.budget = 1000;   // 3 rows x 4000 samples of float32
    const std::string refused = thrown([&] { face::read_field(s); });
    CHECK(has(refused, "48000 bytes"));
    CHECK(has(refused, "shorter time range"));
  }
  {   // a tiling: its spans made from its rule, and a part of them
    const auto r = face::read_selection(rq("segments"));
    const auto* start = array(r, "start");
    const auto* stop = array(r, "stop");
    const auto* code = array(r, "code");
    CHECK(start && stop && code);
    if (start && stop && code) {
      CHECK(values<double>(*start) == (std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7}));
      CHECK(values<double>(*stop) == (std::vector<double>{2, 3, 4, 5, 6, 7, 8, 9}));
      CHECK(values<std::int64_t>(*code) == (std::vector<std::int64_t>{0, 1, 2, 3, 4, 5, 6, 7}));
    }
    CHECK(r.result["params"]["tiling"]["count"] == 8);
    auto p = rq("segments");
    p.region = {{2, 5}};
    const auto part = face::read_selection(p);
    CHECK(array(part, "start") && values<double>(*array(part, "start")) == (std::vector<double>{2, 3, 4}));
    CHECK(part.result["regions"]["stop"] == Json::parse("[[2, 5]]"));
    auto b = rq("segments");
    b.budget = 16;
    CHECK(has(thrown([&] { face::read_selection(b); }), "above the read's budget"));
  }
  return nxrtest::finish("test_read_region");
}
