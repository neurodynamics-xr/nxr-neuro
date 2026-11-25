// `add operator` as a TRANSITION between two coordinate systems (kind transition, from/to_coordinate_system `SPACE:NAME`, one affine [3 x 4]), as the importers store one:
// stored, read back (`read operator`, `list operators`, `read subject`: the matrix and both systems), told apart from a map between manifolds, every refusal, and deleted as any operator.
// Synthetic data only (face::synthetic_dataset).
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>

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
task::ArrayInput npy(const fs::path& file, const std::string& dtype, const std::vector<std::int64_t>& shape, const std::vector<T>& v) {
  write_npy(file, dtype, shape, raw(v));
  return file.string();
}
const Json* named(const Json& rows, const std::string& name) {
  for (const auto& r : rows) if (r["name"] == name) return &r;
  return nullptr;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_add_transition_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "in");
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const std::string D = face::synthetic_dataset(root, "demo", "S1", "rest", 3, 100.0, 4.0).string();
  const std::string by = face::stamp();
  auto rq = [&](const std::string& name) { face::ReadNodeRequest r; r.dataset = D; r.subject = "S1"; r.name = name; return r; };
  auto input = [&](const std::string& file, const std::vector<std::int64_t>& shape, const std::vector<double>& v) { return npy(root / "in" / file, "float64", shape, v); };

  {   // a second coordinate system of the head space to map to: the points' own (the subject holds head:scs already)
    task::AddPointsRequest p;
    p.dataset = D; p.subject = "S1"; p.created_by = by;
    p.name = "landmarks";
    p.coordinate_system = "ncs";
    p.positions = input("pos.npy", {2, 3}, {0, 0, 1, 0, 1, 0});
    task::add_points(p);
  }
  // a rotation of 90 degrees about z, a scale of 2 on z and a translation: a general invertible affine [3 x 4]
  const std::vector<double> affine = {0, -1, 0, 10, 1, 0, 0, -20, 0, 0, 2, 5};
  auto transition = [&](const std::string& name) {
    task::AddOperatorRequest o;
    o.dataset = D; o.subject = "S1"; o.created_by = by;
    o.name = name;
    o.from_manifold = "sensors";
    o.from_coordinate_system = "head:scs";
    o.to_coordinate_system = "head:ncs";
    o.values = input(name + ".npy", {3, 4}, affine);
    return o;
  };

  {   // stored, and read back: the matrix, kind transition, both systems, no target manifold; the convention in the row and its producer record
    task::AddOperatorRequest o = transition("scs_to_ncs");
    o.producer = R"({"source":"synthetic landmarks"})";
    const task::CommandResult added = task::add_operator(o);
    CHECK_EQ(added.result["shape"][0].get<std::int64_t>(), std::int64_t(3));
    CHECK_EQ(added.result["shape"][1].get<std::int64_t>(), std::int64_t(4));
    const task::CommandResult back = face::read_operator(rq("scs_to_ncs"));
    CHECK_EQ(back.result["kind"].get<std::string>(), std::string("transition"));
    CHECK_EQ(back.result["from_coordinate_system"].get<std::string>(), std::string("head:scs"));
    CHECK_EQ(back.result["to_coordinate_system"].get<std::string>(), std::string("head:ncs"));
    CHECK(back.result["to_manifold"].is_null());
    CHECK_EQ(back.result["from_manifold"].get<std::string>(), std::string("sensors"));
    CHECK_EQ(back.arrays.at(0).name, std::string("values"));
    CHECK(back.arrays.at(0).shape == (std::vector<std::int64_t>{3, 4}));
    CHECK(back.arrays.at(0).bytes == raw(affine));
    CHECK(back.result["producer"]["convention"].get<std::string>().find("head:ncs = A [x y z 1]^T") != std::string::npos);
    CHECK_EQ(back.result["producer"]["source"].get<std::string>(), std::string("synthetic landmarks"));
    CHECK_EQ(back.result["producer"]["command"].get<std::string>(), std::string("add operator"));
  }
  {   // a map between manifolds has no coordinate systems; the listings tell the two apart and agree with `read subject`
    task::AddOperatorRequest m;
    m.dataset = D; m.subject = "S1"; m.created_by = by;
    m.name = "mix";
    m.from_manifold = "sensors";
    m.to_manifold = "sensors";
    m.values = input("mix.npy", {3, 3}, {1, 0, 0, 0, 2, 0, 0.5, 0, 3});
    task::add_operator(m);
    CHECK(face::read_operator(rq("mix")).result["from_coordinate_system"].is_null());
    task::ListClassRequest l;
    l.cls = "operators"; l.dataset = D; l.subject = "S1";
    const Json rows = face::list_class(l).result["rows"];
    const Json* t = named(rows, "scs_to_ncs");
    const Json* x = named(rows, "mix");
    CHECK(t && x);
    if (t && x) {
      CHECK_EQ((*t)["kind"].get<std::string>(), std::string("transition"));
      CHECK_EQ((*t)["from"]["coordinate_system"].get<std::string>(), std::string("head:scs"));
      CHECK_EQ((*t)["to"]["coordinate_system"].get<std::string>(), std::string("head:ncs"));
      CHECK((*t)["to"]["manifold"].is_null());
      CHECK((*x)["from"]["coordinate_system"].is_null() && (*x)["to"]["coordinate_system"].is_null());
      CHECK_EQ((*t)["shape"]["rows"].get<std::int64_t>(), std::int64_t(3));
      CHECK_EQ((*t)["shape"]["columns"].get<std::int64_t>(), std::int64_t(4));
    }
    task::ReadSubjectRequest s;
    s.dataset = D; s.subject = "S1";
    const Json subject = face::read_subject(s).result;
    const Json* in_subject = named(subject["operators"], "scs_to_ncs");
    CHECK(in_subject && (*in_subject)["from"]["coordinate_system"] == "head:scs");
  }
  {   // float32 is kept; a target manifold may be given (recorded, its size not used); the name is used without --replace, replaced with it
    task::AddOperatorRequest o = transition("scs_to_ncs_f32");
    o.values = npy(root / "in" / "f32.npy", "float32", {3, 4}, std::vector<float>(affine.begin(), affine.end()));
    o.to_manifold = "sensors";
    task::add_operator(o);
    const task::CommandResult back = face::read_operator(rq("scs_to_ncs_f32"));
    CHECK_EQ(back.arrays.at(0).dtype, std::string("float32"));
    CHECK_EQ(back.result["to_manifold"].get<std::string>(), std::string("sensors"));
    CHECK(has(thrown([&] { task::add_operator(transition("scs_to_ncs")); }), "already_exists"));
    task::AddOperatorRequest again = transition("scs_to_ncs");
    again.replace = true;
    task::add_operator(again);
    CHECK(face::read_operator(rq("scs_to_ncs")).arrays.at(0).bytes == raw(affine));
  }
  {   // the refusals
    auto refusal = [&](task::AddOperatorRequest o) { o.name = "refused"; return thrown([&] { task::add_operator(o); }); };
    task::AddOperatorRequest o = transition("refused");
    { auto x = o; x.to_coordinate_system.reset(); CHECK(has(refusal(x), "go together")); }
    { auto x = o; x.from_coordinate_system.reset(); CHECK(has(refusal(x), "go together")); }
    { auto x = o; x.to_coordinate_system = "nowhere:scs";
      const std::string t = refusal(x);
      CHECK(has(t, "no_such_row") && has(t, "nowhere:scs") && has(t, "head:ncs") && has(t, "head:scs")); }     // names what exists
    { auto x = o; x.from_coordinate_system = "head:nope"; CHECK(has(refusal(x), "head:nope")); }
    { auto x = o; x.from_coordinate_system = "scs"; CHECK(has(refusal(x), "SPACE:NAME")); }
    { auto x = o; x.values = input("sq.npy", {3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1}); CHECK(has(refusal(x), "[3 x 3]") && has(refusal(x), "[3 x 4]")); }
    { auto x = o; x.values = input("sq4.npy", {4, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}); CHECK(has(refusal(x), "[4 x 4]")); }
    { auto x = o; x.values = input("flat.npy", {12}, affine); CHECK(has(refusal(x), "[12]")); }
    { std::vector<double> bad = affine; bad[5] = std::numeric_limits<double>::quiet_NaN();
      auto x = o; x.values = input("nan.npy", {3, 4}, bad); CHECK(has(refusal(x), "not finite")); }
    { std::vector<double> bad = affine; bad[11] = std::numeric_limits<double>::infinity();
      auto x = o; x.values = input("inf.npy", {3, 4}, bad); CHECK(has(refusal(x), "not finite")); }
    { auto x = o; x.values = input("sing.npy", {3, 4}, {1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0}); CHECK(has(refusal(x), "singular")); }       // a zero column
    { auto x = o; x.values = input("sing2.npy", {3, 4}, {1, 2, 3, 0, 2, 4, 6, 0, 0, 0, 1, 0}); CHECK(has(refusal(x), "singular")); }      // rows 1 and 2 dependent
    { auto x = o; x.values.reset(); CHECK(has(refusal(x), "takes --values")); }
    { auto x = o; x.data = input("d.npy", {1}, {1.0}); x.indices = input("i.npy", {1}, {0.0}); x.indptr = input("p.npy", {2}, {0.0, 1.0}); CHECK(has(refusal(x), "one dense affine")); }
    { auto x = o; x.layout = "csr"; CHECK(has(refusal(x), "one dense affine")); }
    { auto x = o; x.from_manifold = "nothing"; CHECK(has(refusal(x), "nothing")); }
    { auto x = o; x.from_coordinate_system.reset(); x.to_coordinate_system.reset(); CHECK(has(refusal(x), "--to-manifold")); }   // without systems the manifold map still needs both manifolds
    CHECK(has(thrown([&] { face::read_operator(rq("refused")); }), "no_such_row"));                                                   // nothing was stored by a refusal
  }
  {   // `delete subject --node operator:NAME` removes it, as any operator; the coordinate systems stay
    task::DeleteSubjectRequest d;
    d.dataset = D; d.subject = "S1"; d.node = "operator:scs_to_ncs"; d.created_by = by;
    task::delete_subject(d);
    CHECK(has(thrown([&] { face::read_operator(rq("scs_to_ncs")); }), "no_such_row"));
    task::ListClassRequest l;
    l.cls = "operators"; l.dataset = D; l.subject = "S1";
    const Json left = face::list_class(l).result["rows"];
    CHECK(!named(left, "scs_to_ncs"));
    CHECK(named(left, "scs_to_ncs_f32") != nullptr);
    CHECK(face::read_manifold(rq("landmarks")).result["coordinate_system"]["name"] == "ncs");
    task::add_operator(transition("scs_to_ncs"));                                                                                    // and the name is free again
  }
  return nxrtest::finish("test_add_transition");
}
