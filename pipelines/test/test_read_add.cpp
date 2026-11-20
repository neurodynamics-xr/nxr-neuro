// `read <class>` and `add <class>` (docs/specs/2026-11-19-io-service.md, section 4): each read verb's result and its .npy files; each add verb stores, and the read that
// follows gives the same arrays back (byte for byte where the stored dtype is the input's); the refusals (no time range, a length that does not fit, an unknown manifold, a name
// used without --replace); a reader leaves every file of the dataset as it was. Synthetic data only (face::synthetic_dataset).
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/npy.h"
#include "nxr/io/task/commands.h"
#include "nxr/pipelines/face.h"

using namespace nxr::io;
namespace face = nxr::pipelines::face;
namespace fs = std::filesystem;

namespace {
std::string bytes_of(const fs::path& f) { std::ifstream in(f, std::ios::binary); return std::string(std::istreambuf_iterator<char>(in), {}); }
std::map<std::string, std::string> tree(const fs::path& folder) {
  std::map<std::string, std::string> out;
  for (const auto& e : fs::recursive_directory_iterator(folder)) if (e.is_regular_file()) out[fs::relative(e.path(), folder).string()] = bytes_of(e.path());
  return out;
}
// what a call throws: "refused: <message>", "<code>: <message>", or "" when it returns
std::string thrown(const std::function<void()>& f) {
  try { f(); } catch (const task::Refused& e) { return std::string("refused: ") + e.what(); } catch (const Error& e) { return e.code() + std::string(": ") + e.what(); }
  return "";
}
bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
template <class T>
std::vector<std::uint8_t> raw(const std::vector<T>& v) {
  std::vector<std::uint8_t> b(v.size() * sizeof(T));
  if (!v.empty()) std::memcpy(b.data(), v.data(), b.size());
  return b;
}
template <class T>
fs::path npy(const fs::path& file, const std::string& dtype, const std::vector<std::int64_t>& shape, const std::vector<T>& v) {
  write_npy(file, dtype, shape, raw(v));
  return file;
}
const task::CommandArray& array_named(const task::CommandResult& r, const std::string& name) {
  for (const auto& a : r.arrays) if (a.name == name) return a;
  static task::CommandArray none;
  std::printf("  no array '%s'\n", name.c_str());
  return none;
}
int exit_of(const std::string& command) {
  const int s = std::system(command.c_str());
  return WIFEXITED(s) ? WEXITSTATUS(s) : -1;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_read_add_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "in");
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const fs::path folder = face::synthetic_dataset(root, "demo", "S1", "rest", 3, 100.0, 40.0);
  const std::string D = folder.string();
  auto rq = [&](const std::string& name) { face::ReadNodeRequest r; r.dataset = D; r.subject = "S1"; r.name = name; return r; };

  // ---- read <class> on what the dataset holds; nothing is written
  const auto before = tree(folder);
  {
    face::ReadNodeRequest r = rq("rest");
    CHECK(has(thrown([&] { face::read_field(r); }), "refused: 'rest' is a time-resolved field: a time range is required"));
    r.from = 0.0;
    r.to = 1.0;
    const task::CommandResult f = face::read_field(r);
    CHECK_EQ(f.result["class"].get<std::string>(), std::string("field"));
    CHECK(!f.result["manifold_id"].is_null());                       // the field names its manifold and its cells (#155)
    CHECK_EQ(f.result["cell"].get<std::string>(), std::string("0"));
    CHECK_EQ(f.result["time"].get<std::string>(), std::string("rest_time"));
    CHECK_EQ(f.arrays.size(), std::size_t(1));
    CHECK(f.arrays[0].shape == (std::vector<std::int64_t>{3, 100}));
    CHECK_EQ(f.arrays[0].dtype, std::string("float32"));
    r.from.reset();
    r.to.reset();
    r.all = true;
    r.rows = {2, 0};
    r.out = (root / "rest.npy").string();
    const task::CommandResult all = face::read_field(r);
    CHECK(all.arrays[0].shape == (std::vector<std::int64_t>{2, 4000}));
    const NpyFile back = NpyFile::open(root / "rest.npy");            // the .npy file holds the same array
    CHECK(back.shape() == all.arrays[0].shape);
    CHECK(raw(back.all<float>()) == all.arrays[0].bytes);
    CHECK_EQ(all.result["files"]["values"].get<std::string>(), (root / "rest.npy").string());
    r.all = false;
    r.rows.clear();
    r.out.reset();
    r.shape_only = true;                                              // the description alone needs no range
    CHECK(face::read_field(r).arrays.empty());
  }
  {
    const task::CommandResult m = face::read_manifold(rq("sensors"));
    CHECK(array_named(m, "positions").shape == (std::vector<std::int64_t>{3, 3}));
    CHECK_EQ(m.result["coordinate_system"]["name"].get<std::string>(), std::string("scs"));   // the manifold's coordinate system (#155)
    CHECK_EQ(m.result["coordinate_system"]["space"].get<std::string>(), std::string("head"));
    CHECK_EQ(m.result["names"].size(), std::size_t(3));
    face::ReadNodeRequest fam = rq("sensors");
    fam.family = "quality";
    CHECK(array_named(face::read_manifold(fam), "codes").shape == (std::vector<std::int64_t>{3}));
    face::ReadNodeRequest sp = rq("rest_bad_segments");
    sp.out = (root / "bad.npy").string();
    const task::CommandResult s = face::read_selection(sp);
    CHECK_EQ(s.result["type"].get<std::string>(), std::string("spans"));
    CHECK(fs::exists(root / "bad.start.npy") && fs::exists(root / "bad.stop.npy") && fs::exists(root / "bad.code.npy"));
    CHECK(has(thrown([&] { face::read_field(rq("sensors")); }), "it is a manifold (read manifold)"));
    CHECK(has(thrown([&] { face::read_statistic(rq("nothing")); }), "no_such_row"));
    face::ReadNodeRequest wrong = rq("sensors");
    wrong.all = true;
    CHECK(has(thrown([&] { face::read_manifold(wrong); }), "refused: a time range"));
  }
  CHECK(tree(folder) == before);                                      // readers leave every file byte for byte

  // ---- add <class>, then read it back
  const std::string by = face::stamp();
  auto base = [&](auto& r, const std::string& name) { r.dataset = D; r.subject = "S1"; r.name = name; r.created_by = by; };
  {
    task::AddFieldRequest f;
    base(f, "w");
    f.manifold = "sensors";
    f.values = npy(root / "in" / "w.npy", "float32", {3}, std::vector<float>{0.5f, -1.25f, 3.0f}).string();
    f.unit = "mm";
    f.description = "a test map";
    const task::CommandResult a = task::add_field(f);
    CHECK_EQ(a.result["class"].get<std::string>(), std::string("field"));
    const task::CommandResult r = face::read_field(rq("w"));
    CHECK_EQ(r.arrays.at(0).dtype, std::string("float32"));          // precision follows the input
    CHECK(r.arrays.at(0).bytes == raw(std::vector<float>{0.5f, -1.25f, 3.0f}));
    CHECK_EQ(r.result["manifold"].get<std::string>(), std::string("sensors"));
    const std::string producer = r.result["producer"].dump();       // the producer names the command and the input file, never its path
    CHECK(has(producer, "add field") && has(producer, "\"w.npy\"") && !has(producer, root.string()));
    CHECK(has(thrown([&] { task::add_field(f); }), "already_exists"));   // a name used is refused without --replace
    f.replace = true;
    CHECK(!task::add_field(f).result["replaced"].get<std::string>().empty());   // moved aside, never deleted
    task::AddFieldRequest bad = f;
    bad.name = "w4";
    bad.values = npy(root / "in" / "w4.npy", "float64", {4}, std::vector<double>{1, 2, 3, 4}).string();
    const std::string why = thrown([&] { task::add_field(bad); });
    CHECK(has(why, "refused: --values w4.npy is [4]") && has(why, "is [3]"));
    bad.manifold = "nowhere";
    CHECK(has(thrown([&] { task::add_field(bad); }), "no manifold named 'nowhere'"));
  }
  {
    task::AddLineRequest l;
    base(l, "epoch_time");
    l.n = 50;
    l.spacing = 0.01;
    l.unit = "s";
    task::add_line(l);
    CHECK(has(thrown([&] { task::add_line(l); }), "already_exists"));
    task::AddLineRequest fl;
    base(fl, "freq");
    fl.n = 4;
    fl.spacing = 1.0;
    fl.origin = 1.0;
    fl.unit = "Hz";
    fl.dimension = "frequency";
    task::add_line(fl);
    std::vector<double> series(3 * 50);
    for (std::size_t i = 0; i < series.size(); ++i) series[i] = 0.25 * static_cast<double>(i);
    task::AddFieldRequest f;
    base(f, "series");
    f.manifold = "sensors";
    f.time = "epoch_time";
    f.values = npy(root / "in" / "series.npy", "float64", {3, 50}, series).string();
    task::add_field(f);
    face::ReadNodeRequest r = rq("series");
    CHECK(has(thrown([&] { face::read_field(r); }), "a time range is required"));
    r.all = true;
    const task::CommandResult back = face::read_field(r);
    CHECK_EQ(back.arrays.at(0).dtype, std::string("float64"));
    CHECK(back.arrays.at(0).bytes == raw(series));
  }
  {
    task::AddSelectionRequest s;
    base(s, "parts");
    s.manifold = "sensors";
    s.type = "partition";
    s.members = npy(root / "in" / "parts.npy", "int32", {3}, std::vector<std::int32_t>{0, 1, 0}).string();
    s.entries = {"left", "right"};
    CHECK_EQ(task::add_selection(s).result["members"].get<std::int64_t>(), std::int64_t(2));
    const task::CommandResult back = face::read_selection(rq("parts"));
    CHECK(array_named(back, "values").bytes == raw(std::vector<std::int32_t>{0, 1, 0}));
    CHECK_EQ(back.result["dictionary"].size(), std::size_t(2));
    task::AddSelectionRequest sp;
    base(sp, "marks");
    sp.manifold = "epoch_time";
    sp.type = "spans";
    sp.spans = npy(root / "in" / "marks.npy", "float64", {2, 3}, std::vector<double>{0.05, 0.10, 0, 0.2, 0.35, 1}).string();
    task::add_selection(sp);
    const task::CommandResult spans = face::read_selection(rq("marks"));
    CHECK(array_named(spans, "start").bytes == raw(std::vector<double>{0.05, 0.2}));
    CHECK(array_named(spans, "stop").bytes == raw(std::vector<double>{0.10, 0.35}));
    CHECK(array_named(spans, "code").bytes == raw(std::vector<std::int64_t>{0, 1}));
    task::AddSelectionRequest pick;
    base(pick, "pick");
    pick.manifold = "sensors";
    pick.type = "indices";
    pick.indices = npy(root / "in" / "pick.npy", "int32", {2}, std::vector<std::int32_t>{2, 0}).string();
    task::add_selection(pick);
    CHECK_EQ(face::read_selection(rq("pick")).arrays.size(), std::size_t(1));
    pick.name = "pick5";
    pick.indices = npy(root / "in" / "pick5.npy", "int32", {1}, std::vector<std::int32_t>{5}).string();
    CHECK(has(thrown([&] { task::add_selection(pick); }), "element 5 is not one of the 3"));

    task::AddMeasurementsRequest m;
    m.dataset = D; m.subject = "S1"; m.created_by = by;
    m.selection = "parts";
    m.field = "w";
    m.measure = {"mean=" + npy(root / "in" / "mean.npy", "float64", {2}, std::vector<double>{1.5, -2.0}).string()};
    task::add_measurements(m);
    face::ReadNodeRequest mr = rq("parts");
    mr.measure = "mean";
    mr.field = "w";
    CHECK(array_named(face::read_selection(mr), "value").bytes == raw(std::vector<double>{1.5, -2.0}));
    CHECK(has(thrown([&] { task::add_measurements(m); }), "already_exists"));
    m.replace = true;
    task::add_measurements(m);
    m.measure = {"max=" + npy(root / "in" / "max3.npy", "float64", {3}, std::vector<double>{1, 2, 3}).string()};
    CHECK(has(thrown([&] { task::add_measurements(m); }), "is [3]; one value a member of 'parts' is [2]"));
  }
  {
    task::AddStatisticRequest st;
    base(st, "psd");
    st.measure = "power_density";
    st.method = "welch";
    st.axes = {"channel:sensors", "frequency:freq"};
    st.of_field = "rest";
    std::vector<float> v(12);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.5f * static_cast<float>(i);
    st.values = npy(root / "in" / "psd.npy", "float32", {3, 4}, v).string();
    task::add_statistic(st);
    const task::CommandResult back = face::read_statistic(rq("psd"));
    CHECK_EQ(back.arrays.at(0).dtype, std::string("float32"));
    CHECK(back.arrays.at(0).bytes == raw(v));
  }
  {
    task::AddOperatorRequest o;
    base(o, "mix");
    o.from_manifold = "sensors";
    o.to_manifold = "sensors";
    const std::vector<double> dense = {1, 0, 0, 0, 2, 0, 0.5, 0, 3};
    o.values = npy(root / "in" / "mix.npy", "float64", {3, 3}, dense).string();
    task::add_operator(o);
    CHECK(face::read_operator(rq("mix")).arrays.at(0).bytes == raw(dense));
    task::AddOperatorRequest sp;
    base(sp, "sparse");
    sp.from_manifold = "sensors";
    sp.to_manifold = "sensors";
    sp.data = npy(root / "in" / "d.npy", "float64", {2}, std::vector<double>{4.0, 5.0}).string();
    sp.indices = npy(root / "in" / "i.npy", "int32", {2}, std::vector<std::int32_t>{0, 2}).string();
    sp.indptr = npy(root / "in" / "p.npy", "int32", {4}, std::vector<std::int32_t>{0, 1, 1, 2}).string();
    task::add_operator(sp);
    face::ReadNodeRequest r = rq("sparse");
    r.out = (root / "sparse.npy").string();
    const task::CommandResult back = face::read_operator(r);
    CHECK(array_named(back, "data").bytes == raw(std::vector<double>{4.0, 5.0}));
    CHECK(array_named(back, "indices").bytes == raw(std::vector<std::int32_t>{0, 2}));
    CHECK(NpyFile::open(root / "sparse.indptr.npy").all<std::int64_t>() == (std::vector<std::int64_t>{0, 1, 1, 2}));
    sp.name = "sparse_bad";
    sp.indptr = npy(root / "in" / "p3.npy", "int32", {3}, std::vector<std::int32_t>{0, 1, 2}).string();
    CHECK(has(thrown([&] { task::add_operator(sp); }), "has 3 offsets"));
  }
  {
    task::AddPointsRequest p;
    base(p, "electrodes");
    const std::vector<double> pos = {0, 0, 1, 0, 1, 0, 1, 0, 0, 0.5, 0.5, 0.5};
    p.positions = npy(root / "in" / "pos.npy", "float64", {4, 3}, pos).string();
    task::add_points(p);
    CHECK(array_named(face::read_manifold(rq("electrodes")), "positions").bytes == raw(pos));
    task::AddSurfaceRequest s;
    s.dataset = D; s.subject = "S1"; s.created_by = by;
    s.name = "tetra";
    s.vertices = npy(root / "in" / "v.npy", "float64", {4, 3}, pos).string();
    s.faces = npy(root / "in" / "f.npy", "int32", {4, 3}, std::vector<std::int32_t>{0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3}).string();
    s.no_primary = true;
    CHECK_EQ(task::add_surface(s).result["n_vertices"].get<std::int64_t>(), std::int64_t(4));
    const task::CommandResult m = face::read_manifold(rq("tetra"));
    CHECK(array_named(m, "cells_2").shape == (std::vector<std::int64_t>{4, 3}));
    CHECK(has(thrown([&] { task::AddSurfaceRequest no = s; no.name.reset(); task::add_surface(no); }), "named by --name"));
  }

#ifdef NXR_TEST_CLI
  {   // the command line: the same functions, a refusal is a usage error (exit 2), and a field written with --out comes back from add field as the same file
    const std::string cli = std::string(NXR_TEST_CLI) + " ", ds = " --dataset " + D + " --subject S1 ", quiet = " > /dev/null 2>&1";
    CHECK_EQ(exit_of(cli + "read field" + ds + "--name rest" + quiet), 2);
    const fs::path w = root / "in" / "w.npy", back = root / "cli_back.npy";
    CHECK_EQ(exit_of(cli + "add field" + ds + "--name cli_w --manifold sensors --values " + w.string() + quiet), 0);
    CHECK_EQ(exit_of(cli + "add field" + ds + "--name cli_w --manifold sensors --values " + w.string() + quiet), 1);
    CHECK_EQ(exit_of(cli + "read field" + ds + "--name cli_w --out " + back.string() + quiet), 0);
    CHECK(bytes_of(back) == bytes_of(w));
    CHECK_EQ(exit_of(cli + "add line" + ds + "--name cli_line --n 10 --spacing 0.1 --unit s" + quiet), 0);
    CHECK_EQ(exit_of(cli + "add surface" + ds + "--name cli_tetra --no-primary --vertices " + (root / "in" / "v.npy").string() + " --faces " + (root / "in" / "f.npy").string() + quiet), 0);
    CHECK_EQ(exit_of(cli + "read manifold" + ds + "--name cli_tetra --out " + (root / "tetra.npy").string() + quiet), 0);
    CHECK(fs::exists(root / "tetra.positions.npy") && fs::exists(root / "tetra.cells_2.npy"));
  }
#endif
  return nxrtest::finish("test_read_add");
}
