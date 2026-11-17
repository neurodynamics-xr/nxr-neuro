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
    sub.complete();
  }
  SubjectPackage pkg = SubjectPackage::open(root / "S.nxr.zarr");
  const Subject s = Subject::of(pkg);
  CHECK_EQ(s.id(), sid);

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
  CHECK_EQ(s.find_all({.cls = Cls::field, .manifold_id = mid}).size(), static_cast<std::size_t>(2));
  CHECK_EQ(s.find_all({.cls = Cls::field, .stored = false}).size(), static_cast<std::size_t>(0));
  CHECK_EQ(s.by_id(Cls::field, fid)->name, std::string("a"));
  CHECK(!s.by_id(Cls::field, "no-such-id"));
  CHECK_EQ(s.find_all({.cls = Cls::statistic}).size(), static_cast<std::size_t>(0));

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
