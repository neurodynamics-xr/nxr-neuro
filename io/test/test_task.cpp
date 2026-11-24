// The record API's read side: every repository call equals the SQL it replaces, on a synthetic subject package.
#include <unistd.h>

#include <filesystem>

#include "check.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/task/task.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/testing/probe.h"

using namespace nxr::io;
using namespace nxr::io::task;
namespace fs = std::filesystem;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); }
  return "";
}

ManifoldSpec triangle(const std::string& name) {
  ManifoldSpec m;
  m.name = name;
  m.path = "source/" + name;
  m.type = "surface";
  m.n_vertices = 4;
  m.topology = {"simplicial", 2, 1, std::string("ccw-outward"), {{2, "source/" + name + "/faces"}}};
  GeometrySpec g;
  g.form = "stored";
  g.cs_key = "scs";
  g.positions_path = "source/" + name + "/vertices";
  m.geometry = g;
  return m;
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_task_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  CreatePackageOptions opt;
  opt.home_dataset = "demo";
  std::string sid, mid, fid, opid, setid, member0, member1;
  {
    DatasetWriter w = create_package(root / "S.nxr.zarr", "S", opt);
    SubjectWriter sub = SubjectWriter::open(w, "S");
    sid = sub.id();
    const std::vector<double> v(12, 1.0);
    const std::vector<std::int32_t> f = {0, 1, 2, 0, 2, 3};
    ManifoldSpec m = triangle("cortex");
    m.populate = [&](Store&, const std::string&) {
      sub.write_array("source/cortex/vertices", view(v, {4, 3}));
      sub.write_array("source/cortex/faces", view(f, {2, 3}));
    };
    mid = sub.manifold(m);
    const std::vector<float> a(4, 2.0f), b(4, 3.0f);
    FieldSpec fa;
    fa.name = "a"; fa.path = "source/a"; fa.kind = "map"; fa.manifold_id = mid; fa.data = view(a, {4});
    fid = sub.field(fa);
    FieldSpec fb;
    fb.name = "b"; fb.path = "source/b"; fb.kind = "other"; fb.manifold_id = mid; fb.data = view(b, {4});
    sub.field(fb);
    FieldSpec fv;                      // a virtual field: no path
    fv.name = "virtual"; fv.kind = "map"; fv.manifold_id = mid;
    fv.meta = std::nullopt;
    OperatorSpec op;
    op.name = "analytic"; op.kind = "gradient"; op.from_manifold_id = mid; op.layout = "analytic"; op.n_rows = 4; op.n_cols = 4;
    op.cols.set("method", "finite-difference");
    opid = sub.operator_(op);
    SelectionSpec set;
    set.name = "tiles"; set.type = "set"; set.manifold_id = mid; set.no_elements = true;
    set.cols.json("params_json", Json{{"atlas", Json{{"tree", "group"}, {"default", true}}}, {"joint", 3}});
    setid = sub.selection(set);
    for (int i = 0; i < 2; ++i) {
      SelectionSpec mem;
      mem.name = "tile" + std::to_string(i); mem.type = "spans"; mem.manifold_id = mid; mem.no_elements = true;
      mem.cols.set("member_of_id", setid).set("member_ordinal", i).set("cell", "1");
      mem.spans = std::vector<SpanSpec>{{0.5 * i, 1.0 + i, std::nullopt}, {3.0 + i, 4.0 + i, 7}};
      (i == 0 ? member0 : member1) = sub.selection(mem);
    }
    sub.complete();
  }
  {   // the write side: a cached node found by its key, parameters replaced, measurements written and removed, a unit that fails stores nothing
    DatasetWriter w = open_package_writer(root / "S.nxr.zarr");
    SubjectWriter sub = SubjectWriter::open(w, "S");
    Subject t = Subject::of(sub);
    const Json key = Json{{"pipeline", "demo"}, {"params", Json{{"k", 1}}}, {"input", Json::object()}, {"compute", "x"}};
    CHECK(!t.cached(Cls::selection, key));
    t.set_params(setid, Json{{"cache", key}, {"joint", 3}});
    const auto hit = t.cached(Cls::selection, key);
    CHECK(hit && hit->id == setid);
    CHECK(!t.cached(Cls::selection, key, std::string("no/such/path")));
    CHECK(!t.cached(Cls::selection, Json{{"pipeline", "other"}}));
    const Json rows = Json::array({Json{{"selection_id", member0}, {"code", 1}, {"measure", "area"}, {"value", 2.5}}});
    CHECK_EQ(t.write_measurements(rows, std::string("task test")), static_cast<std::int64_t>(1));
    CHECK(t.has_measure(member0, "area"));
    CHECK(!t.has_measure(member0, "area", std::string("no-such-field")));
    CHECK_EQ(t.measured_kinds(member0).size(), static_cast<std::size_t>(1));
    CHECK_EQ(t.remove_measurements(member0, std::nullopt, std::string("area")), static_cast<std::int64_t>(1));
    CHECK(!t.has_measure(member0, "area"));
    CHECK_EQ(code_of([&] { t.commit([&] { t.write_measurements(rows, std::nullopt); throw Error(code::invalid_argument, "stop"); }); }), std::string(code::invalid_argument));
    CHECK(!t.has_measure(member0, "area"));    // all or nothing
    t.commit([&] { t.write_measurements(rows, std::nullopt); });
    CHECK(t.has_measure(member0, "area"));
    {   // a Unit: rows and parameters together, or nothing
      t.remove_measurements(member0);
      const Json params_before = t.require_id(Cls::selection, setid).params;
      {
        Unit u = t.unit(By{"unit test", Json::object()});
        CHECK_EQ(u.write_measurements(rows), static_cast<std::int64_t>(1));
        u.set_params(setid, Json{{"changed", true}});
        CHECK(t.has_measure(member0, "area"));
      }   // destroyed without commit: rolled back
      CHECK(!t.has_measure(member0, "area"));
      CHECK(t.require_id(Cls::selection, setid).params == params_before);
      Unit u = t.unit(By{"unit test", Json::object()});
      u.write_measurements(rows);
      u.set_params(setid, Json{{"changed", true}});
      u.commit();
      CHECK_EQ(code_of([&] { u.commit(); }), std::string(code::not_writable));
      CHECK(t.has_measure(member0, "area"));
      CHECK(t.require_id(Cls::selection, setid).params.contains("changed"));
    }
    t.set_params(setid, Json{{"atlas", Json{{"tree", "group"}}}, {"joint", 3}});   // back as it was
    {   // THE TYPED WRITES (spec 2026-11-17-definition-outputs, section 4): every kind of result in one unit, with the unit's producer record; all or nothing
      const Json key{{"cache", Json{{"pipeline", "toy@1"}}}};
      LineOut ln;
      ln.name = "toy_freq"; ln.alt = "toy_freq_alt"; ln.n = 5; ln.spacing = 0.5;
      FieldOut fo;
      fo.name = "toy_map"; fo.output = "map"; fo.kind = "map"; fo.manifold_id = mid; fo.path = "source/toy_map"; fo.unit = "V"; fo.data.shape = {4}; fo.data.f32 = {1, 2, 3, 4};
      SelectionOut tiles;
      tiles.name = "toy_segments"; tiles.output = "psd"; tiles.type = "tiling"; tiles.tiling = Tiling{0.0, 2.0, 1.0, 10.0, 8, 1};
      SelectionOut part;
      part.name = "toy_parts"; part.output = "parts"; part.type = "partition"; part.manifold_id = mid; part.member_of = {0, 0, 1, -1}; part.member_codes = {0, 1}; part.entries = {"left", "right"};
      OperatorOut op;
      op.name = "toy_op"; op.output = "op"; op.kind = "toy operator"; op.from_manifold_id = mid; op.to_manifold_id = mid; op.data.shape = {4, 4}; op.data.f64.assign(16, 0.25);
      op.params = Json{{"k", 1}};
      StatisticOut st;
      st.name = "toy_psd"; st.output = "psd"; st.measure = "power_density"; st.method = "welch"; st.of_field_id = fid; st.n_members = 8; st.members_skipped = 1;
      st.unit = "V\xc2\xb2/Hz"; st.data.shape = {4, 5}; st.data.f32.assign(20, 1.0f); st.producer = Json{{"segments", 8}};
      {
        Unit u = t.unit(By{"unit test", key});
        CHECK(!u.line(ln).empty());
        CHECK(!u.field(fo).empty());
      }   // rolled back
      CHECK(!t.at_path(Cls::field, "source/toy_map"));
      Unit u = t.unit(By{"unit test", key});
      const std::string line_id = u.line(ln);
      CHECK_EQ(u.line(ln), line_id);                                     // found again: it fits
      LineOut other = ln;
      other.spacing = 0.25;
      CHECK(u.line(other) != line_id);                                   // the name is taken by another grid: the alternative name
      SurfaceOut so;
      so.name = "toy_mesh"; so.output = "mesh"; so.derived_from_id = mid;
      so.vertices.shape = {4, 3}; so.vertices.f64 = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
      so.faces = {0, 1, 2, 0, 2, 3};
      const std::string mesh_id = u.surface(so);
      CHECK_EQ(u.surface(so), mesh_id);                                  // found again
      SurfaceOut broken = so;
      broken.name = "broken"; broken.faces = {0, 1, 9};
      CHECK_EQ(code_of([&] { u.surface(broken); }), std::string(code::invalid_argument));
      CHECK_EQ(code_of([&] { u.product(ProductOut{{mesh_id}, std::nullopt}); }), std::string(code::invalid_argument));
      const std::string prod_id = u.product(ProductOut{{mesh_id, line_id}, std::nullopt});   // a factor made in this unit
      CHECK(!prod_id.empty());
      CHECK_EQ(u.replace(Cls::field, u.path_of(fo), false, "the toy map"), std::string());   // nothing there
      const std::string field_id = u.field(fo);
      tiles.manifold_id = line_id;
      const std::string seg_id = u.selection(tiles);
      const std::string part_id = u.selection(part);
      const std::string op_id = u.operator_(op);
      st.partition_selection_id = seg_id;
      st.axes = Json::array({Json{{"name", "vertex"}, {"manifold_id", mid}, {"size", 4}}, Json{{"name", "frequency"}, {"manifold_id", line_id}, {"size", 5}, {"unit", "Hz"}}});
      CHECK_EQ(u.path_of(st), std::string("source/a_statistics/toy_psd"));   // beside the field it is of
      const std::string st_id = u.statistic(st);
      SelectionOut bad = part;
      bad.name = "bad"; bad.member_of.clear();
      CHECK_EQ(code_of([&] { u.selection(bad); }), std::string(code::invalid_argument));
      FieldOut wrong = fo;
      wrong.name = "wrong"; wrong.path = "source/wrong"; wrong.data.shape = {5};
      CHECK_EQ(code_of([&] { u.field(wrong); }), std::string(code::invalid_argument));
      u.commit();
      const Node fn = t.require_id(Cls::field, field_id);
      CHECK_EQ(fn.producer["output"].get<std::string>(), std::string("map"));
      CHECK(t.cached(Cls::field, key["cache"], std::string("source/toy_map")));
      CHECK_EQ(*t.require_id(Cls::manifold, mesh_id).n_vertices, static_cast<std::int64_t>(4));
      CHECK_EQ(t.factors(prod_id), std::vector<std::string>({mesh_id, line_id}));
      CHECK_EQ(t.require_id(Cls::statistic, st_id).producer["segments"].get<int>(), 8);
      CHECK(t.require_id(Cls::selection, seg_id).params["tiling"]["count"] == Json(9));
      CHECK_EQ(t.require_id(Cls::selection, part_id).params["producer"]["output"].get<std::string>(), std::string("parts"));
      CHECK_EQ(*t.require_id(Cls::operator_, op_id).to_manifold_id, mid);
      CHECK(t.cached(Cls::selection, key["cache"], t.require_id(Cls::selection, seg_id).path));
      // a result already there: refused without replace, moved aside with it
      Unit r = t.unit(By{"unit test", key});
      CHECK_EQ(code_of([&] { r.replace(Cls::field, "source/toy_map", false, "the toy map"); }), std::string(code::already_exists));
      r.replace(Cls::field, "source/toy_map", true, "the toy map");
      CHECK(!t.at_path(Cls::field, "source/toy_map"));
      r.field(fo);
      r.commit();
      CHECK(t.at_path(Cls::field, "source/toy_map"));
    }
    {   // THE FREQUENCY LINE OF A SPAN, keyed by (time line, n, fs) and made by the FFT operator: two lengths are two lines, the same length again is the same line, an old per-recording line of the grid is found
      Unit u = t.unit(By{"unit test", Json::object()});
      LineOut tl;
      tl.name = "rec_time"; tl.alt = "rec_time_alt"; tl.n = 4096; tl.spacing = 0.01; tl.unit = "s"; tl.dimension = "time"; tl.folder = "time"; tl.cs_key = "clock";
      const std::string time_id = u.line(tl);
      const std::string l256 = u.frequency_line(time_id, 256, 100.0), l128 = u.frequency_line(time_id, 128, 100.0);
      CHECK(!l256.empty() && !l128.empty() && l256 != l128);
      CHECK_EQ(u.frequency_line(time_id, 256, 100.0), l256);                 // the same span length: the same line, no second operator
      CHECK_EQ(u.frequency_line(time_id, 128, 100.0), l128);
      const std::string l200 = u.frequency_line(time_id, 200, 100.0);        // not a power of two: named by n
      CHECK(l200 != l256 && l200 != l128);
      LineOut old;                                                           // a line an older run stored under the recording's name, of the grid of n = 64 at 100 Hz
      old.name = "rec_freq"; old.alt = "rec_freq_alt"; old.n = 33; old.spacing = 100.0 / 64.0;
      const std::string old_id = u.line(old);
      CHECK_EQ(u.frequency_line(time_id, 64, 100.0, "rec_freq"), old_id);    // found by its old name: nothing renamed, no operator made
      CHECK(u.frequency_line(time_id, 32, 100.0, "rec_freq") != old_id);     // another grid is not that line
      CHECK_EQ(code_of([&] { u.frequency_line(time_id, 1, 100.0); }), std::string(code::invalid_argument));
      u.commit();
    }
    sub.complete();
  }
  SubjectPackage pkg = SubjectPackage::open(root / "S.nxr.zarr");
  const Subject s = Subject::of(pkg);
  CHECK_EQ(s.id(), sid);

  {   // the frequency lines made above: each with an analytic fft operator row from the time line, params {n, fs, sides, window}; the old line has none
    auto line_of = [&](const std::string& name) { return s.require({.cls = Cls::manifold, .name = name}, "the line"); };
    const Node time = line_of("rec_time");
    for (const auto& [name, n] : std::vector<std::pair<std::string, int>>{{"rec_time_freq_L8", 256}, {"rec_time_freq_L7", 128}, {"rec_time_freq_n200", 200}, {"rec_time_freq_L5", 32}}) {
      const Node line = line_of(name);
      CHECK_EQ(*line.n_vertices, static_cast<std::int64_t>(n / 2 + 1));
      const std::string stem = name.substr(std::string("rec_time_freq_").size());
      const Node op = s.require({.cls = Cls::operator_, .name = "rec_time_fft_" + stem}, "the fft operator");
      CHECK(op.from_manifold_id && *op.from_manifold_id == time.id && op.to_manifold_id && *op.to_manifold_id == line.id);
      CHECK(op.params["n"] == Json(n) && op.params["fs"] == Json(100.0) && op.params["sides"] == Json("one-sided") && op.params["window"] == Json("none"));
    }
    CHECK(!s.find({.cls = Cls::operator_, .name = "rec_time_fft_L6"}));      // the old line of n = 64 got no operator row
  }

  // find: one, none, several; the Want's class decides what it may filter on
  const Node cortex = s.require({.cls = Cls::manifold, .name = "cortex"}, "the surface");
  CHECK_EQ(cortex.id, mid);
  CHECK(cortex.type && *cortex.type == "surface");
  CHECK_EQ(*cortex.n_vertices, static_cast<std::int64_t>(4));
  CHECK(!s.find({.cls = Cls::manifold, .name = "nope"}));
  CHECK_EQ(code_of([&] { s.require({.cls = Cls::manifold, .name = "nope"}, "the surface"); }), std::string(code::no_such_row));
  CHECK_EQ(code_of([&] { s.find({.cls = Cls::field}); }), std::string(code::ambiguous));
  CHECK_EQ(code_of([&] { s.find({.cls = Cls::field, .type = "set"}); }), std::string(code::invalid_argument));
  const Node a = s.require({.cls = Cls::field, .name = "a", .kind = "map"}, "a");
  CHECK_EQ(a.id, fid);
  CHECK(a.path && *a.path == "source/a");
  CHECK(a.manifold_id && *a.manifold_id == mid);
  CHECK_EQ(s.find_all({.cls = Cls::field, .manifold_id = mid}).size(), static_cast<std::size_t>(3));   // a, b and the typed writes' toy_map
  CHECK_EQ(s.find_all({.cls = Cls::field, .stored = false}).size(), static_cast<std::size_t>(0));
  CHECK_EQ(s.by_id(Cls::field, fid)->name, std::string("a"));
  CHECK(!s.by_id(Cls::field, "no-such-id"));
  CHECK_EQ(s.find_all({.cls = Cls::statistic}).size(), static_cast<std::size_t>(1));   // the typed writes' toy_psd

  // operators, selections by parameters, members, spans
  const Node op = s.require({.cls = Cls::operator_, .kind = "gradient"}, "the gradient");
  CHECK(op.method && *op.method == "finite-difference");
  CHECK(op.from_manifold_id && *op.from_manifold_id == mid);
  const Node tiles = s.require({.cls = Cls::selection, .name = "tiles", .type = "set"}, "the tiles");
  CHECK(tiles.params["atlas"]["tree"] == "group");
  CHECK_EQ(s.find_all({.cls = Cls::selection, .params = {{"atlas.tree", "group"}}}).size(), static_cast<std::size_t>(1));
  const auto mem = s.members(tiles);
  CHECK_EQ(mem.size(), static_cast<std::size_t>(2));
  CHECK_EQ(mem[1].id, member1);
  CHECK_EQ(s.member(tiles, 0)->id, member0);
  CHECK(!s.member(tiles, 9));
  const auto sp = s.spans(mem[1]);
  CHECK_EQ(sp.size(), static_cast<std::size_t>(2));
  CHECK(sp[1].code && *sp[1].code == 7);
  CHECK_EQ(sp[0].start, 0.5);
  CHECK_EQ(s.manifold_of(a)->id, mid);

  // geometry
  const Geometry g = s.geometry(mid);
  CHECK(g.positions_path && *g.positions_path == "source/cortex/vertices");
  CHECK_EQ(g.cells.size(), static_cast<std::size_t>(1));
  CHECK_EQ(g.cells[0].first, static_cast<std::int64_t>(2));

  // measured kinds, the cache lookup (nothing cached here), node paths
  CHECK(s.measured_kinds(setid).empty());
  CHECK(s.has_measure(member0, "area"));
  CHECK(!s.cached(Cls::field, Json{{"pipeline", "x"}}));
  CHECK(!s.node_path(mid, "abc").empty());
  CHECK_EQ(s.at_path(Cls::field, "source/a")->id, fid);
  CHECK(!s.at_path(Cls::field, "source/none"));
  CHECK_EQ(s.factors(mid), std::vector<std::string>({mid}));              // not a product: itself
  CHECK(s.has_measures(member0, fid) == false);
  CHECK(s.measured_selections(fid, "area").empty());
  CHECK_EQ(s.describe(a).manifolds, std::vector<std::string>({mid}));
  CHECK(s.fields_on_axis(mid, 1).empty());
  CHECK(!s.map_slice("a"));
  CHECK_EQ(code_of([&] { s.commit([] {}); }), std::string(code::not_writable));   // a package opened for reading cannot commit
  fs::remove_all(root);
  return nxrtest::finish("task");
}
