// The task API's command-line calls (task::Dataset and the Subject's lookups by name, counts and read-time measurements): each equals the SQL it replaced, on a
// synthetic one-database dataset (a surface, a time line, a recording on their product) and a subject package.
#include <unistd.h>

#include <filesystem>

#include "check.h"
#include "nxr/io/database.h"
#include "nxr/io/measures.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/task/dataset.h"
#include "nxr/io/task/task.h"
#include "nxr/io/writer.h"

using namespace nxr::io;
namespace task = nxr::io::task;
namespace fs = std::filesystem;

namespace {
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
ManifoldSpec line(const std::string& name, std::int64_t n) {
  ManifoldSpec t;
  t.name = name; t.path = "time/" + name; t.type = "line"; t.n_vertices = n;
  t.topology = {"path", 1, 1, std::nullopt, {}};
  GeometrySpec g; g.form = "closed_form"; g.cs_key = "clock";
  DimensionSpec d; d.name = "time"; d.n_vertices = n; d.spacing = 0.01; d.origin = 0.0; d.unit = "s";
  g.dimensions.push_back(d); t.geometry = g;
  t.cols.set("n_edges", n - 1);
  return t;
}
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); }
  return "";
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_task_dataset_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  DatasetWriter ds = create_dataset(root, "demo");
  std::string mid, tid, pid, fid, rid, setid;
  {
    SubjectWriter t = SubjectWriter::create(ds, "T", SubjectOptions{.kind = "template"});
    t.compose([] {});
    SubjectWriter s2 = SubjectWriter::create(ds, "S2");
    s2.compose([] {});
  }
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    const std::vector<double> v(12, 1.0);
    const std::vector<std::int32_t> f = {0, 1, 2, 0, 2, 3};
    const std::vector<float> a(4, 2.0f);
    sub.compose([&] {
      ManifoldSpec m = triangle("cortex");
      m.populate = [&](Store&, const std::string&) {
        sub.write_array("source/cortex/vertices", view(v, {4, 3}));
        sub.write_array("source/cortex/faces", view(f, {2, 3}));
      };
      mid = sub.manifold(m);
      tid = sub.manifold(line("rest_time", 100));
      sub.manifold(line("other_time", 50));
      pid = sub.product({mid, tid});
      FieldSpec fa;
      fa.name = "a"; fa.path = "source/a"; fa.kind = "map"; fa.manifold_id = mid; fa.data = view(a, {4});
      fid = sub.field(fa);
      FieldSpec rec;                     // a recording on the product (virtual here: the lookups read its rows only)
      rec.name = "rest"; rec.kind = "recording"; rec.manifold_id = pid; rec.meta = std::nullopt;
      rid = sub.field(rec);
      SelectionSpec set;
      set.name = "tiles"; set.type = "set"; set.manifold_id = mid; set.no_elements = true;
      setid = sub.selection(set);
    });
  }
  const Database& db = ds.db();
  const task::Dataset d = task::Dataset::of(ds);

  // subjects: the listing, the kinds, by name, by name or path
  {
    const auto rows = db.all("SELECT name, kind, status, path FROM subject WHERE dataset_id = ? ORDER BY name", {V(ds.id())});
    const auto got = d.subjects();
    CHECK_EQ(got.size(), rows.size());
    CHECK_EQ(got.size(), static_cast<std::size_t>(3));
    for (std::size_t i = 0; i < std::min(got.size(), rows.size()); ++i)
      CHECK(got[i].name == text(rows[i], "name") && got[i].kind == text(rows[i], "kind") && got[i].status == text(rows[i], "status") && got[i].path == text(rows[i], "path"));
    const auto kinds = db.all("SELECT kind, count(*) AS n FROM subject WHERE dataset_id = ? AND status = 'complete' GROUP BY kind ORDER BY kind", {V(ds.id())});
    const auto gk = d.subject_kinds();
    CHECK_EQ(gk.size(), kinds.size());
    for (std::size_t i = 0; i < std::min(gk.size(), kinds.size()); ++i) CHECK(gk[i].kind == text(kinds[i], "kind") && gk[i].n == integer(kinds[i], "n"));
    const auto s1 = db.one("SELECT id, path FROM subject WHERE name = ?", {V("S1")});
    CHECK(d.subject_named("S1") && d.subject_named("S1")->id == text(*s1, "id"));
    CHECK(!d.subject_named("nobody"));
    CHECK(task::Dataset::of(db).subject_named("S1")->id == text(*s1, "id"));
    CHECK(d.find_subject(text(*s1, "path"))->name == "S1");
    CHECK(d.find_subject("S1")->path == text(*s1, "path"));
    CHECK(!d.find_subject("nowhere"));
  }

  // the arrays the rows name, as the read command listed them
  {
    std::vector<std::string> want;
    const char* sqls[] = {
        "SELECT s.path AS sp, s.name AS sn, r.path AS p FROM field r JOIN subject s ON s.id = r.subject_id WHERE r.path IS NOT NULL",
        "SELECT s.path AS sp, s.name AS sn, r.array_path AS p FROM selection r JOIN subject s ON s.id = r.subject_id WHERE r.array_path IS NOT NULL",
        "SELECT s.path AS sp, s.name AS sn, r.path AS p FROM operator r JOIN subject s ON s.id = r.subject_id WHERE r.path IS NOT NULL",
        "SELECT s.path AS sp, s.name AS sn, r.positions_path AS p FROM geometry r JOIN subject s ON s.id = r.subject_id WHERE r.positions_path IS NOT NULL",
        "SELECT s.path AS sp, s.name AS sn, r.path AS p FROM topology_cell r JOIN topology t ON t.id = r.topology_id JOIN subject s ON s.id = t.subject_id WHERE r.path IS NOT NULL",
        "SELECT s.path AS sp, s.name AS sn, r.array_path AS p FROM selection_measurement r JOIN selection x ON x.id = r.selection_id JOIN subject s ON s.id = x.subject_id WHERE r.array_path IS NOT NULL"};
    for (const char* q : sqls) for (const Row& r : db.all(q)) want.push_back(text(r, "sn") + "|" + text(r, "sp") + "|" + text(r, "p"));
    std::vector<std::string> got;
    for (const auto& a : d.stored_arrays()) got.push_back(a.subject_name + "|" + a.subject_path + "|" + a.path);
    CHECK(got == want);
    CHECK(got.size() >= static_cast<std::size_t>(3));   // the field, the positions, the faces
    const task::Dataset ro = task::Dataset::open_readonly(root / "demo");
    std::vector<std::string> got_ro;
    for (const auto& a : ro.stored_arrays()) got_ro.push_back(a.subject_name + "|" + a.subject_path + "|" + a.path);
    CHECK(got_ro == want);
    CHECK_EQ(ro.subjects().size(), static_cast<std::size_t>(3));
    CHECK_EQ(code_of([&] { ro.drain(); }), std::string(code::not_writable));
  }

  // the subject's lookups: time lines, a line by name, a node by id or name, counts
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  const task::Subject s = task::Subject::of(sub);
  {
    const std::string q_time = "SELECT gd.manifold_id AS line FROM field f JOIN manifold m ON m.id = f.manifold_id JOIN geometry_dimension gd ON gd.geometry_id = m.geometry_id "
                               "JOIN manifold t ON t.id = gd.manifold_id WHERE f.subject_id = ? AND f.kind = 'recording' AND f.status = 'complete' AND t.type = 'line'";
    std::vector<std::string> want;
    for (const Row& r : db.all(q_time, {V(sub.id())})) want.push_back(text(r, "line"));
    CHECK(s.recording_time_lines() == want);
    CHECK(want == std::vector<std::string>({tid}));
    CHECK(s.recording_time_line("rest") == text(*db.one(q_time + " AND f.name = ?", {V(sub.id()), V("rest")}), "line"));
    CHECK(!s.recording_time_line("a"));
    CHECK(s.line_named("other_time") == text(*db.one("SELECT id FROM manifold WHERE subject_id = ? AND name = ? AND type = 'line'", {V(sub.id()), V("other_time")}), "id"));
    CHECK(!s.line_named("cortex"));
    CHECK_EQ(s.by_id_or_name(task::Cls::field, "a")->id, fid);
    CHECK_EQ(s.by_id_or_name(task::Cls::field, fid)->id, fid);
    CHECK_EQ(s.by_id_or_name(task::Cls::selection, "tiles")->id, setid);
    CHECK(!s.by_id_or_name(task::Cls::field, "tiles"));
    const task::NodeCounts n = s.node_counts();
    auto count = [&](const char* t) { return integer(*db.one(std::string("SELECT count(*) AS n FROM ") + t + " WHERE subject_id = ?", {V(sub.id())}), "n"); };
    CHECK(n.manifolds == count("manifold") && n.fields == count("field") && n.operators == count("operator") && n.selections == count("selection"));
    CHECK_EQ(n.fields, static_cast<std::int64_t>(2));
  }

  // read-time measurements: as io's query
  {
    s.write_measurements(Json::array({Json{{"selection_id", setid}, {"code", 1}, {"measure", "area"}, {"value", 2.5}}}), std::string("task test"));
    const Json q = Json{{"selectionId", setid}, {"measure", "area"}};
    CHECK(s.read_measurements(q) == query_measurements(db, sub.store(), q));
    CHECK_EQ(s.read_measurements(q).size(), static_cast<std::size_t>(1));
  }

  // removal and recovery
  {
    d.remove(task::Cls::field, fid);
    CHECK(!s.by_id_or_name(task::Cls::field, "a"));
    const task::Healed h = d.recover();
    CHECK(h.rolled_back == 0 && h.finished == 0);
    d.drain();
    const std::string s2 = d.subject_named("S2")->id;
    d.remove_subject(s2);
    CHECK(!d.subject_named("S2"));
  }

  // a subject package: its subject from the package record
  {
    CreatePackageOptions opt;
    opt.home_dataset = "demo";
    DatasetWriter w = create_package(root / "P.nxr.zarr", "P", opt);
    CHECK_EQ(task::Dataset::of(w).package_subject_id(), read_package_record(w.db()).subject_id);
    CHECK(!task::Dataset::of(w).package_subject_id().empty());
  }
  fs::remove_all(root);
  return nxrtest::finish("task_dataset");
}
