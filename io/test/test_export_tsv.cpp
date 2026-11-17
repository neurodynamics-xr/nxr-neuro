// export tsv (nxr/exporters/tsv.h) on a synthetic package: measurements of a map over a labelling's members (stored and derived measures), and statistics in long form
// (a frequency axis, a bin axis, a selection axis). Each table is read back with the pure TSV reader and its sidecar checked.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>

#include "check.h"
#include "nxr/exporters/tsv.h"
#include "nxr/formats/tsv.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/surface.h"
#include "nxr/io/task/task.h"
#include "nxr/io/writer.h"

using namespace nxr::io;
using namespace nxr::io::task;
namespace fs = std::filesystem;
namespace ex = nxr::exporters;
namespace tsv = nxr::formats::tsv;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception&) { return "other"; }
  return "";
}
Json read_json(const std::string& path) { std::ifstream f(path); return Json::parse(f); }
ex::Context ctx_of() {
  ex::Context c;
  c.version = "9.9.9";
  c.command = "nxr-neuro export tsv test";
  return c;
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_export_tsv_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  const fs::path pkg = root / "S.nxr.zarr";

  std::vector<double> xyz;
  for (int h = 0; h < 2; ++h)
    for (const auto& p : std::vector<std::array<double, 3>>{{0, 0, 0}, {0.01, 0, 0}, {0, 0.01, 0}, {0, 0, 0.01}}) { xyz.push_back(p[0] + (h ? 0.1 : 0.0)); xyz.push_back(p[1]); xyz.push_back(p[2]); }
  std::vector<std::int32_t> faces;
  for (int h = 0; h < 2; ++h)
    for (const auto& f : std::vector<std::array<int, 3>>{{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}}) for (int c = 0; c < 3; ++c) faces.push_back(4 * h + f[c]);
  std::vector<float> map(8, 1.0f);

  std::string cortex_id, field_id, atlas_id;
  {
    CreatePackageOptions po;
    po.home_dataset = "demo";
    DatasetWriter w = create_package(pkg, "S", po);
    SubjectWriter sub = SubjectWriter::open(w, "S");
    SurfaceOptions so;
    so.name = "cortex";
    AtlasMember atlas;
    atlas.family = "demo-atlas";
    atlas.codes = {0, 1, 1, 1, 2, 2, 2, 0};
    atlas.levels = {"unassigned", "alpha", "beta"};
    atlas.colors = {{0, 0, 0}, {255, 0, 128}, {0, 0, 255}};
    so.atlases = {atlas};
    const SurfaceInfo si = add_surface(sub, so, xyz, faces);
    cortex_id = si.id;
    FieldSpec fm;
    fm.name = "map"; fm.path = "source/map"; fm.kind = "map"; fm.manifold_id = si.id; fm.data = view(map, {8});
    fm.cols.set("unit", "V");
    field_id = sub.field(fm);
    set_origin(w.db(), "field", field_id, OriginRecord{"fixture:map", "CC-BY-4.0", "A. Author", true});
    sub.complete();
  }
  {
    DatasetWriter w = open_package_writer(pkg);
    SubjectWriter sub = SubjectWriter::open(w, "S");
    Subject t = Subject::of(sub);
    atlas_id = t.require({.cls = Cls::selection, .name = "demo-atlas"}, "the atlas").id;
    // measurements: mean and mean_square per member (the codes of the atlas's dictionary: 0 unassigned, 1 alpha, 2 beta)
    Json rows = Json::array();
    const double mean[3] = {0.0, 1.0, 2.0}, msq[3] = {0.0, 2.0, 5.0};
    for (int c = 1; c <= 2; ++c) {
      rows.push_back(Json{{"selection_id", atlas_id}, {"code", c}, {"of_field_id", field_id}, {"measure", "mean"}, {"value", mean[c]}});
      rows.push_back(Json{{"selection_id", atlas_id}, {"code", c}, {"of_field_id", field_id}, {"measure", "mean_square"}, {"value", msq[c]}});
    }
    CHECK_EQ(t.write_measurements(rows, std::string("test")), static_cast<std::int64_t>(4));
    // statistics
    Unit u = t.unit(By{"test", Json::object()});
    LineOut ln;
    ln.name = "freq"; ln.n = 5; ln.spacing = 0.5; ln.origin = 1.0;
    const std::string line_id = u.line(ln);
    StatisticOut st;
    st.name = "psd"; st.output = "psd"; st.measure = "power_density"; st.method = "welch"; st.of_field_id = field_id; st.unit = "V\xc2\xb2/Hz";
    st.axes = Json::array({Json{{"name", "vertex"}, {"manifold_id", cortex_id}, {"size", 8}}, Json{{"name", "frequency"}, {"manifold_id", line_id}, {"size", 5}, {"unit", "Hz"}}});
    st.data.shape = {8, 5};
    for (int v = 0; v < 8; ++v) for (int f = 0; f < 5; ++f) st.data.f32.push_back(static_cast<float>(v * 10 + f));
    u.statistic(st);
    StatisticOut hist;
    hist.name = "hist"; hist.output = "hist"; hist.measure = "power"; hist.method = "histogram"; hist.of_field_id = field_id;
    hist.axes = Json::array({Json{{"name", "bin"}, {"edges", Json::array({0.0, 1.0, 2.0, 4.0})}, {"size", 3}}});
    hist.data.shape = {3};
    hist.data.f32 = {5.0f, 7.0f, 1.0f};
    u.statistic(hist);
    StatisticOut bym;
    bym.name = "bymember"; bym.output = "bymember"; bym.measure = "power"; bym.method = "welch"; bym.of_field_id = field_id;
    bym.axes = Json::array({Json{{"name", "member"}, {"selection_id", atlas_id}, {"size", 3}}});
    bym.data.shape = {3};
    bym.data.f32 = {0.5f, 1.5f, 2.5f};
    u.statistic(bym);
    u.commit();
  }
  CHECK(SubjectPackage::open(pkg).verify().ok);

  // ---- measurements: stored and derived columns, the members by name
  {
    ex::TsvExport o;
    o.partition = "demo-atlas"; o.field = "map"; o.measures = {"mean", "mean_square", "sd"}; o.out = (root / "m").string(); o.ctx = ctx_of();
    const Json r = ex::export_tsv(pkg, o);
    const tsv::Table t = tsv::read_file((root / "m.tsv").string());
    CHECK_EQ(t.columns, std::vector<std::string>({"member", "code", "mean", "mean_square", "sd"}));
    CHECK_EQ(t.rows.size(), static_cast<std::size_t>(2));
    CHECK_EQ(t.cell(0, "member"), std::string("alpha"));
    CHECK_EQ(t.number(0, "mean"), 1.0);
    CHECK_EQ(t.number(1, "mean_square"), 5.0);
    CHECK(std::abs(t.number(0, "sd") - 1.0) < 1e-12);                       // derived at read: sqrt(2 - 1)
    CHECK(std::abs(t.number(1, "sd") - 1.0) < 1e-12);                       // sqrt(5 - 4)
    const Json side = read_json((root / "m.json").string());
    CHECK(side["sd"]["Derived"].get<bool>());
    CHECK(!side["mean"]["Derived"].get<bool>());
    CHECK_EQ(side["member"]["Levels"]["1"].get<std::string>(), std::string("alpha"));
    CHECK_EQ(side["Partition"]["dictionary"].size(), static_cast<std::size_t>(3));
    CHECK_EQ(side["mean"]["FieldUnits"].get<std::string>(), std::string("V"));
    const Json& gen = side["GeneratedBy"][0];
    CHECK_EQ(gen["producer"]["version"].get<std::string>(), std::string("9.9.9"));
    CHECK_EQ(gen["outputs"][0]["sha256"].get<std::string>(), ex::file_sha256((root / "m.tsv").string()));
    CHECK_EQ(gen["terms"]["licence"].get<std::string>(), std::string("CC-BY-4.0 + unknown"));   // the field states one, the atlas none: both are listed
    CHECK_EQ(r["rows"].get<int>(), 2);
    // a measure that is not stored, and refusals
    o.measures = {"max"};
    CHECK_EQ(code_of([&] { ex::export_tsv(pkg, o); }), std::string(code::invalid_argument));
    o.measures = {"mean"}; o.partition = "nope";
    CHECK_EQ(code_of([&] { ex::export_tsv(pkg, o); }), std::string(code::no_such_row));
    ex::TsvExport both;
    both.statistic = "psd"; both.partition = "demo-atlas"; both.out = (root / "b").string();
    CHECK_EQ(code_of([&] { ex::export_tsv(pkg, both); }), std::string(code::invalid_argument));
    ex::TsvExport none;
    none.out = (root / "n").string();
    CHECK_EQ(code_of([&] { ex::export_tsv(pkg, none); }), std::string(code::invalid_argument));
  }

  // ---- statistics in long form
  {
    ex::TsvExport o;
    o.statistic = "psd"; o.out = (root / "psd").string(); o.ctx = ctx_of();
    ex::export_tsv(pkg, o);
    const tsv::Table t = tsv::read_file((root / "psd.tsv").string());
    CHECK_EQ(t.columns, std::vector<std::string>({"vertex", "frequency", "value"}));
    CHECK_EQ(t.rows.size(), static_cast<std::size_t>(40));
    CHECK_EQ(t.number(0, "frequency"), 1.0);
    CHECK_EQ(t.number(7, "frequency"), 2.0);                                // row 7 = vertex 1, frequency index 2
    CHECK_EQ(t.number(7, "vertex"), 1.0);
    CHECK_EQ(t.number(7, "value"), 12.0);                                   // 1 * 10 + 2
    CHECK_EQ(t.number(39, "value"), 74.0);
    const Json side = read_json((root / "psd.json").string());
    CHECK_EQ(side["frequency"]["Units"].get<std::string>(), std::string("Hz"));
    CHECK_EQ(side["value"]["Units"].get<std::string>(), std::string("V\xc2\xb2/Hz"));
    CHECK_EQ(side["Statistic"]["method"].get<std::string>(), std::string("welch"));
    o.statistic = "hist"; o.out = (root / "hist").string();
    ex::export_tsv(pkg, o);
    const tsv::Table h = tsv::read_file((root / "hist.tsv").string());
    CHECK_EQ(h.columns, std::vector<std::string>({"bin_low", "bin_high", "value"}));
    CHECK_EQ(h.number(2, "bin_low"), 2.0);
    CHECK_EQ(h.number(2, "bin_high"), 4.0);
    CHECK_EQ(h.number(1, "value"), 7.0);
    o.statistic = "bymember"; o.out = (root / "bym").string();
    ex::export_tsv(pkg, o);
    const tsv::Table b = tsv::read_file((root / "bym.tsv").string());
    CHECK_EQ(b.cell(1, "member"), std::string("alpha"));
    CHECK_EQ(b.number(2, "value"), 2.5);
    o.statistic = "nope";
    CHECK_EQ(code_of([&] { ex::export_tsv(pkg, o); }), std::string(code::no_such_row));
  }
  fs::remove_all(root);
  return nxrtest::finish("export_tsv");
}
