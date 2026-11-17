// The record API's read side: every repository call equals the SQL it replaces, on a synthetic subject package.
#include <unistd.h>

#include <filesystem>

#include "check.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/records/records.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/testing/probe.h"

using namespace nxr::io;
using namespace nxr::io::records;
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
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_records_" + std::to_string(::getpid()));
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
  SubjectPackage pkg = SubjectPackage::open(root / "S.nxr.zarr");
  const Database& db = pkg.db();
  const Records rec = Records::of(pkg);

  // ---- manifolds
  {
    const auto by_sql = db.all("SELECT * FROM manifold WHERE subject_id = ? AND type = 'surface' AND status = 'complete' ORDER BY name, id", {V(sid)});
    const auto got = rec.manifolds().list({.subject_id = sid, .type = "surface"});
    CHECK_EQ(got.size(), by_sql.size());
    CHECK_EQ(got.size(), static_cast<std::size_t>(1));
    CHECK(got[0].to_json() == ManifoldRow::from_row(by_sql[0]).to_json());
    CHECK_EQ(got[0].id, mid);
    CHECK(rec.manifolds().find({.subject_id = sid, .name = "cortex"})->id == mid);
    CHECK(!rec.manifolds().find({.subject_id = sid, .name = "nope"}));
    const auto r = rec.manifolds().read(mid);
    CHECK(r && r->topology && r->geometry && r->coordinate_system);
    CHECK_EQ(r->cells.size(), static_cast<std::size_t>(1));
    CHECK_EQ(r->row.n_vertices, static_cast<std::int64_t>(4));
    const auto cs = db.one("SELECT cs.* FROM manifold m JOIN geometry g ON g.id = m.geometry_id JOIN coordinate_system cs ON cs.id = g.coordinate_system_id WHERE m.id = ?", {V(mid)});
    CHECK(r->coordinate_system->to_json() == CoordinateSystemRow::from_row(*cs).to_json());
    CHECK(!rec.manifolds().read("no-such-id"));
  }
  // ---- fields
  {
    const auto by_sql = db.all("SELECT * FROM field WHERE subject_id = ? AND status = 'complete' ORDER BY name, id", {V(sid)});
    const auto got = rec.fields().list({.subject_id = sid});
    CHECK_EQ(got.size(), by_sql.size());
    for (std::size_t i = 0; i < got.size(); ++i) CHECK(got[i].to_json() == FieldRow::from_row(by_sql[i]).to_json());
    CHECK_EQ(rec.fields().find({.subject_id = sid, .name = "a"})->id, fid);
    CHECK_EQ(rec.fields().list({.subject_id = sid, .kind = "other"}).size(), static_cast<std::size_t>(1));
    CHECK_EQ(rec.fields().list({.subject_id = sid, .stored = false}).size(), static_cast<std::size_t>(0));
    CHECK_EQ(code_of([&] { rec.fields().find({.subject_id = sid}); }), std::string(code::ambiguous));    // several match
    CHECK_EQ(rec.fields().read(fid)->name, std::string("a"));
    CHECK(rec.fields().read(fid)->to_json() == FieldRow::from_json(rec.fields().read(fid)->to_json()).to_json());   // the JSON round trip
  }
  // ---- operators
  {
    const auto got = rec.operators().find({.subject_id = sid, .kind = "gradient", .from_manifold_id = mid});
    CHECK(got && got->id == opid);
    CHECK(got->method && *got->method == "finite-difference");
    CHECK_EQ(rec.operators().list({.subject_id = sid, .stored = false}).size(), static_cast<std::size_t>(1));
    CHECK_EQ(rec.operators().list({.subject_id = sid, .stored = true}).size(), static_cast<std::size_t>(0));
    CHECK(!rec.operators().find({.subject_id = sid, .kind = "kernel"}));
  }
  // ---- selections
  {
    const auto tiles = rec.selections().find({.subject_id = sid, .type = "set", .name = "tiles"});
    CHECK(tiles && tiles->id == setid);
    CHECK(tiles->params_json["atlas"]["tree"] == "group");
    CHECK_EQ(rec.selections().list({.subject_id = sid, .params = {{"atlas.tree", "group"}}}).size(), static_cast<std::size_t>(1));
    CHECK_EQ(rec.selections().list({.subject_id = sid, .params = {{"$.atlas.default", true}}}).size(), static_cast<std::size_t>(1));
    CHECK_EQ(rec.selections().list({.subject_id = sid, .params = {{"atlas.tree", "single"}}}).size(), static_cast<std::size_t>(0));
    CHECK_EQ(rec.selections().list({.subject_id = sid, .params_present = {"joint"}}).size(), static_cast<std::size_t>(1));
    CHECK_EQ(rec.selections().list({.subject_id = sid, .params_absent = {"joint"}}).size(), static_cast<std::size_t>(2));
    const auto sql_members = db.all("SELECT * FROM selection WHERE member_of_id = ? AND status = 'complete' ORDER BY member_ordinal", {V(setid)});
    const auto mem = rec.selections().members(setid);
    CHECK_EQ(mem.size(), sql_members.size());
    CHECK_EQ(mem.size(), static_cast<std::size_t>(2));
    CHECK_EQ(mem[0].id, member0);
    CHECK_EQ(rec.selections().member(setid, 1)->id, member1);
    CHECK(!rec.selections().member(setid, 5));
    const auto sp = rec.selections().spans(member1);
    const auto sql_spans = db.all("SELECT * FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {V(member1)});
    CHECK_EQ(sp.size(), sql_spans.size());
    CHECK_EQ(sp.size(), static_cast<std::size_t>(2));
    CHECK(sp[1].code && *sp[1].code == 7);
    CHECK_EQ(sp[1].start, 4.0);
    CHECK_EQ(rec.selections().list({.subject_id = sid, .name_like = "tile%"}).size(), static_cast<std::size_t>(3));
  }
  // ---- the generic read, the status default, and the probe
  {
    CHECK(rec.read({Kind::manifold, mid})->at("name") == "cortex");
    CHECK(rec.read({Kind::selection, setid})->at("params_json")["joint"] == 3);
    CHECK(!rec.read({Kind::field, "no-such-id"}));
    const testing::Probe probe(pkg);
    CHECK_EQ(probe.count("field"), static_cast<std::int64_t>(2));
    CHECK_EQ(probe.count("field", {{"name", V("a")}}), static_cast<std::int64_t>(1));
    CHECK_EQ(probe.count("selection", {{"member_of_id", V(nullptr)}}), static_cast<std::int64_t>(1));
    CHECK(probe.integrity().empty());
    CHECK_EQ(probe.rows("manifold").size(), static_cast<std::size_t>(1));
  }
  fs::remove_all(root);
  return nxrtest::finish("records");
}
