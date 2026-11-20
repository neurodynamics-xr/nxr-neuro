// What `read subject` and `list <class>` describe of each node, and what `add <class>` records of who made it (data-layer queries spec, 3.1, 3.2 and 3.4): every node's producer record in
// its provenance, as `read <class>` gives it; the author `--by` as created_by (the producer record still naming the program); `add operator --producer`; the members of a set described as
// the selections they are; a line's coordinate (origin, spacing, unit); a component's from_index; every dictionary the subject sees, app-scope ones included, each with its scope.
// Synthetic data only: two tetrahedra as a two-component surface with a parcellation, and the `add <class>` commands.
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/npy.h"
#include "nxr/io/surface.h"
#include "nxr/io/task/commands.h"
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
const Json* named(const Json& rows, const std::string& name) {
  for (const auto& r : rows) if (r["name"] == name) return &r;
  return nullptr;
}
std::string thrown(const std::function<void()>& f) {
  try { f(); } catch (const task::Refused& e) { return std::string("refused: ") + e.what(); } catch (const Error& e) { return e.code() + std::string(": ") + e.what(); }
  return "";
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_subject_descriptions_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "in");
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};

  // ---- a subject with a surface of two components (two tetrahedra) and its parcellation set, written by io's surface writer, and an app dictionary no selection reads
  {
    DatasetWriter ds = create_dataset(root, "demo");
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      const std::vector<double> P = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 5, 0, 0, 6, 0, 0, 5, 1, 0, 5, 0, 1};
      std::vector<std::int32_t> F = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
      for (int i = 0; i < 12; ++i) F.push_back(F[static_cast<std::size_t>(i)] + 4);
      SurfaceOptions so;
      so.name = "patch";
      so.primary = false;
      so.hemispheres = {Hemisphere{"left", 0, 4}, Hemisphere{"right", 4, 4}};
      so.atlases.push_back(AtlasMember{"Lobes", {1, 1, 2, 0, 1, 2, 2, 0}, {"unassigned", "front", "back"}, {{{0, 0, 0}}, {{200, 0, 0}}, {{0, 0, 200}}}});
      add_surface(sub, so, P, F);
      DictionarySpec d;
      d.name = "app quality";
      d.scope = "app";
      d.names = {"good", "bad"};
      sub.dictionary(d);
    });
  }
  const std::string D = (root / "demo").string(), program = face::stamp();
  auto base = [&](auto& r, const std::string& name) { r.dataset = D; r.subject = "S1"; r.name = name; r.created_by = program; };

  // ---- G1, G2, G12: the producer record and the author
  {
    task::AddFieldRequest f;
    base(f, "w");
    f.manifold = "patch";
    f.values = npy(root / "in" / "w.npy", "float32", {8}, std::vector<float>(8, 1.5f));
    f.producer = R"({"studio": {"layer": "L"}})";
    f.by = "a person";
    task::add_field(f);
    task::AddFieldRequest f2;
    base(f2, "w2");
    f2.manifold = "patch";
    f2.values = f.values;
    task::add_field(f2);
    task::AddSelectionRequest p;
    base(p, "halves");
    p.manifold = "patch";
    p.type = "partition";
    p.members = npy(root / "in" / "halves.npy", "int32", {8}, std::vector<std::int32_t>{0, 0, 0, 0, 1, 1, 1, 1});
    p.producer = R"({"studio": {"tool": "lasso"}})";
    p.by = "an app";
    task::add_selection(p);
    task::AddOperatorRequest o;
    base(o, "mix");
    o.from_manifold = "patch";
    o.to_manifold = "patch";
    std::vector<double> eye(64, 0.0);
    for (int i = 0; i < 8; ++i) eye[static_cast<std::size_t>(9 * i)] = 1.0;
    o.values = npy(root / "in" / "mix.npy", "float64", {8, 8}, eye);
    o.producer = R"({"studio": {"kernel": "identity"}})";
    o.by = "a person";
    task::add_operator(o);
    task::AddOperatorRequest bad = o;
    bad.name = "mix2";
    bad.producer = R"({"program": "mine"})";
    CHECK(thrown([&] { task::add_operator(bad); }).find("'program' is the command's own record") != std::string::npos);
    task::AddLineRequest l;
    base(l, "epoch");
    l.n = 10;
    l.spacing = 0.5;
    l.origin = 2.0;
    l.unit = "s";
    l.by = "a script";
    task::add_line(l);
    task::AddDictionaryRequest dd;
    dd.dataset = D;
    dd.subject = "S1";
    dd.name = "marks";
    dd.scope = "subject";
    dd.entries = R"([{"code": 1, "name": "one"}])";
    dd.created_by = program;
    dd.by = "a person";
    task::add_dictionary(dd);
  }

  task::ReadSubjectRequest q;
  q.dataset = D;
  q.subject = "S1";
  const Json sub = face::read_subject(q).result;

  {   // G1: the producer record in each description, as `read <class>` has it
    const Json* w = named(sub["fields"], "w");
    CHECK(w != nullptr);
    if (w) {
      const Json& prov = (*w)["provenance"];
      CHECK(prov["producer"].is_object());
      CHECK_EQ(prov["producer"]["studio"]["layer"].get<std::string>(), std::string("L"));
      CHECK_EQ(prov["producer"]["command"].get<std::string>(), std::string("add field"));
      CHECK_EQ(prov["producer"]["program"].get<std::string>(), program);          // G2: the producer record names the program
      CHECK_EQ(prov["created_by"].get<std::string>(), std::string("a person"));    // G2: the author
      task::ListClassRequest lq;
      lq.cls = "fields";
      lq.dataset = D;
      lq.subject = "S1";
      lq.name = "w";
      const Json rows = face::list_class(lq).result["rows"];
      CHECK(named(rows, "w") && (*named(rows, "w"))["provenance"]["producer"] == prov["producer"]);   // list <class> gives the same description
      face::ReadNodeRequest rn;
      rn.dataset = D;
      rn.subject = "S1";
      rn.name = "w";
      CHECK(face::read_field(rn).result["producer"] == prov["producer"]);                               // as read <class> gives it
    }
    const Json* w2 = named(sub["fields"], "w2");
    CHECK(w2 && (*w2)["provenance"]["created_by"].get<std::string>() == program);     // G2: without --by, the program
    const Json* h = named(sub["selections"], "halves");
    CHECK(h && (*h)["provenance"]["producer"]["studio"]["tool"] == "lasso");          // a selection's record, kept in its parameters
    CHECK(h && (*h)["provenance"]["created_by"] == "an app");
    const Json* m = named(sub["operators"], "mix");
    CHECK(m && (*m)["provenance"]["producer"]["studio"]["kernel"] == "identity");     // G12
    CHECK(m && (*m)["provenance"]["producer"]["command"] == "add operator");
    CHECK(m && (*m)["provenance"]["created_by"] == "a person");
    const Json* e = named(sub["manifolds"], "epoch");
    CHECK(e && (*e)["provenance"]["created_by"] == "a script");
    const Json* mk = named(sub["dictionaries"], "marks");
    CHECK(mk && (*mk)["provenance"]["created_by"] == "a person");
  }
  {   // G3: the members of the set of parcellations are selections of their own
    const Json* set = named(sub["selections"], "patch_atlases");
    const Json* lobes = named(sub["selections"], "Lobes");
    CHECK(set != nullptr);
    CHECK(lobes != nullptr);
    if (lobes) {
      CHECK_EQ((*lobes)["member_of"].get<std::string>(), std::string("patch_atlases"));
      CHECK_EQ((*lobes)["member_ordinal"].get<std::int64_t>(), std::int64_t(0));
      CHECK_EQ((*lobes)["dictionary"].get<std::string>(), std::string("Lobes"));
    }
    if (set) CHECK((*set)["member_of"].is_null());
    task::ListClassRequest lq;
    lq.cls = "selections";
    lq.dataset = D;
    lq.subject = "S1";
    lq.on = "patch";
    CHECK(named(face::list_class(lq).result["rows"], "Lobes") != nullptr);
  }
  {   // G8: a line's coordinate; G9: a component's from_index
    const Json* e = named(sub["manifolds"], "epoch");
    CHECK(e != nullptr);
    if (e) {
      const Json& axes = (*e)["coordinates"]["axes"];
      CHECK(axes.is_array() && axes.size() == 1);
      if (axes.is_array() && axes.size() == 1) {
        CHECK_EQ(axes[0]["origin"].get<double>(), 2.0);
        CHECK_EQ(axes[0]["spacing"].get<double>(), 0.5);
        CHECK_EQ(axes[0]["unit"].get<std::string>(), std::string("s"));
        CHECK_EQ(axes[0]["size"].get<std::int64_t>(), std::int64_t(10));
      }
    }
    const Json* left = named(sub["manifolds"], "patch_1");
    const Json* right = named(sub["manifolds"], "patch_2");
    CHECK(left && (*left)["component_of"] == "patch" && (*left)["from_index"] == 0);
    CHECK(right && (*right)["component_of"] == "patch" && (*right)["from_index"] == 4);
    const Json* whole = named(sub["manifolds"], "patch");
    CHECK(whole && (*whole)["from_index"].is_null() && (*whole)["coordinates"]["axes"].empty());   // a stored geometry has no closed-form axis
  }
  {   // G13: every dictionary the subject sees, the app's included (one no selection reads), each with its scope
    const Json* app = named(sub["dictionaries"], "app quality");
    CHECK(app && (*app)["scope"] == "app");
    const Json* lobes = named(sub["dictionaries"], "Lobes");
    CHECK(lobes && (*lobes)["scope"] == "app");
    const Json* marks = named(sub["dictionaries"], "marks");
    CHECK(marks && (*marks)["scope"] == "subject");
  }
  return nxrtest::finish("test_subject_descriptions");
}
