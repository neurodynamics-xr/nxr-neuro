// THE GLUED CORTEX'S RELATION, STATED AND CARRIED (synthetic two-hemisphere brain, never a real dataset):
//   - `run glue` records the relation in the glued cortex's description (its producer's `glue`): the original manifold, the hemispheres labelling of the glued cortex, the medial wall's
//     selection on the original (the Desikan-Killiany member, its `unassigned` entry and code), the counts and the rule;
//   - `read field|operator --onto <glued>` gives a node's rows selected onto the glued cortex: each glued vertex the rows of its original vertex (k rows a vertex), a bridge vertex's rows zero;
//     a node whose rows are not the original's vertices, a manifold that is not a glued cortex, and --onto with --region are refused;
//   - `migrate glue` states the relation of a glue whose record names only the original (an earlier version's), and is idempotent.
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "brain_helpers.h"
#include "check.h"
#include "hemispheres.h"
#include "nxr/io/task/task.h"
#include "nxr/pipelines/eigen_migrate.h"
#include "nxr/pipelines/face.h"
#include "nxr/pipelines/input.h"
#include "run_definition.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
namespace tk = nxr::io::task;

namespace {
std::string refusal(const std::function<void()>& f) { try { f(); } catch (const std::exception& e) { return e.what(); } return ""; }
template <class T>
std::vector<T> values_of(const face::Stored& s) {
  std::vector<T> v(s.arrays.at(0).bytes.size() / sizeof(T));
  std::memcpy(v.data(), s.arrays.at(0).bytes.data(), v.size() * sizeof(T));
  return v;
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_glue_onto_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(3, false);
  DatasetWriter ds = create_dataset(root, "demo");
  std::vector<std::int32_t> dk(static_cast<std::size_t>(b.n), 1);
  for (int i = 0; i < b.n; ++i) if (std::abs(b.P[static_cast<std::size_t>(3 * i + 1)]) < 0.6) dk[static_cast<std::size_t>(i)] = 0;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      so.atlases.push_back(AtlasMember{"Desikan-Killiany", dk, {"unassigned", "cortex"}, {{{0, 0, 0}}, {{200, 200, 200}}}});
      so.atlases.push_back(nxrtest::structures_of(b.P, b.F));
      add_surface(sub, so, b.P, b.F);
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  nxrtest::run_definition(ds, sub, "glue", Json{{"surface", "cortex"}});
  const tk::Subject ts = tk::Subject::of(sub);
  const auto glued = ts.find({.cls = tk::Cls::manifold, .name = "cortex_glued", .type = "surface"});
  const auto cortex = ts.find({.cls = tk::Cls::manifold, .name = "cortex", .type = "surface"});
  CHECK(glued && cortex);
  const std::int64_t V = b.n, Vg = glued->n_vertices.value_or(0);
  std::vector<std::int32_t> off_wall;
  for (std::int64_t v = 0; v < V; ++v) if (dk[static_cast<std::size_t>(v)] != 0) off_wall.push_back(static_cast<std::int32_t>(v));
  const std::int64_t m = static_cast<std::int64_t>(off_wall.size());

  // THE RELATION, STATED in the glued cortex's description
  {
    const Json& g = glued->producer["glue"];
    std::printf("  the glue's relation: %s\n", g.dump().substr(0, 400).c_str());
    CHECK(g.value("original", std::string()) == cortex->id && g.value("original_name", std::string()) == "cortex");
    CHECK(g.value("hemispheres", std::string()) == "cortex_glued_hemispheres");
    CHECK(ts.find({.cls = tk::Cls::selection, .name = "cortex_glued_hemispheres"}).has_value());
    CHECK(g["medial_wall"].value("labelling", std::string()) == "cortex_atlases" && g["medial_wall"].value("member", std::string()) == "Desikan-Killiany");
    CHECK(g["medial_wall"].value("entry", std::string()) == "unassigned" && g["medial_wall"].value("code", -1) == 0 && !g["medial_wall"].value("id", std::string()).empty());
    CHECK(g.value("cortex_vertices", std::int64_t(-1)) == m && g.value("bridge_vertices", std::int64_t(-1)) == Vg - m);
    CHECK(g.value("rule", std::string()).find("ascending index") != std::string::npos);
    CHECK(text(*sub.db().read("manifold", glued->id), "derived_from_id") == cortex->id);
  }

  // a field on the original [V x 2] and an operator to it [3V x 4] (a free orientation's three rows a vertex)
  std::vector<double> fv(static_cast<std::size_t>(V * 2));
  for (std::int64_t v = 0; v < V; ++v) for (int c = 0; c < 2; ++c) fv[static_cast<std::size_t>(v * 2 + c)] = 10.0 * static_cast<double>(v) + c + 1;
  const std::int64_t C = 4;
  std::vector<float> ov(static_cast<std::size_t>(3 * V * C));
  for (std::size_t i = 0; i < ov.size(); ++i) ov[i] = static_cast<float>(i + 1);
  {
    tk::Unit u = ts.unit(tk::By{"test", Json::object()});
    tk::FieldOut f; f.name = "values"; f.output = "values"; f.kind = "test"; f.manifold_id = cortex->id;
    f.data.shape = {V, 2}; f.data.f64 = fv; u.field(f);
    tk::OperatorOut o; o.name = "kernel"; o.output = "kernel"; o.kind = "inverse kernel"; o.from_manifold_id = cortex->id; o.to_manifold_id = cortex->id;
    o.data.shape = {3 * V, C}; o.data.f32 = ov; u.operator_(o);
    tk::FieldOut g; g.name = "on_glued"; g.output = "on_glued"; g.kind = "test"; g.manifold_id = glued->id;
    g.data.shape = {Vg}; g.data.f64.assign(static_cast<std::size_t>(Vg), 1.0); u.field(g);
    u.commit();
  }
  const fs::path D = ds.folder();

  // CARRIED: the field's rows, each glued vertex its original vertex's, the bridge's zero
  {
    face::ReadRequest rq; rq.cls = "field"; rq.onto = std::string("cortex_glued");
    const face::Stored s = face::read(D, "S1", "values", rq);
    CHECK(s.arrays.size() == 1 && s.arrays[0].shape == std::vector<std::int64_t>({Vg, 2}));
    const std::vector<double> x = values_of<double>(s);
    bool ok = true;
    for (std::int64_t i = 0; i < Vg; ++i) for (int c = 0; c < 2; ++c)
      ok = ok && x[static_cast<std::size_t>(i * 2 + c)] == (i < m ? fv[static_cast<std::size_t>(off_wall[static_cast<std::size_t>(i)] * 2 + c)] : 0.0);
    CHECK(ok);
    CHECK(s.meta["onto"].value("rows_per_vertex", 0) == 1 && s.meta["onto"].value("bridge_rows", std::string()) == "zero");
  }
  // CARRIED: the operator's rows, three a vertex
  {
    face::ReadNodeRequest rq; rq.dataset = D.string(); rq.subject = "S1"; rq.name = "kernel"; rq.onto = std::string("cortex_glued");
    const auto r = face::read_operator(rq);
    CHECK(r.arrays.size() == 1 && r.arrays[0].shape == std::vector<std::int64_t>({3 * Vg, C}));
    std::vector<float> x(r.arrays[0].bytes.size() / sizeof(float));
    std::memcpy(x.data(), r.arrays[0].bytes.data(), r.arrays[0].bytes.size());
    bool ok = true;
    for (std::int64_t i = 0; i < Vg; ++i) for (int d = 0; d < 3; ++d) for (std::int64_t c = 0; c < C; ++c)
      ok = ok && x[static_cast<std::size_t>((3 * i + d) * C + c)] == (i < m ? ov[static_cast<std::size_t>((3 * off_wall[static_cast<std::size_t>(i)] + d) * C + c)] : 0.0f);
    CHECK(ok);
    CHECK(r.result["onto"].value("rows_per_vertex", 0) == 3);
  }
  // REFUSED: a node not on the original, a manifold that is not a glued cortex, a region with onto
  {
    face::ReadRequest rq; rq.cls = "field"; rq.onto = std::string("cortex_glued");
    const std::string a = refusal([&] { face::read(D, "S1", "on_glued", rq); });
    CHECK(a.find("are not the vertices of cortex") != std::string::npos);
    rq.onto = std::string("cortex");
    const std::string b2 = refusal([&] { face::read(D, "S1", "values", rq); });
    CHECK(b2.find("is not a glued cortex") != std::string::npos);
    rq.onto = std::string("cortex_glued"); rq.region = {{0, 2}};
    const std::string c = refusal([&] { face::read(D, "S1", "values", rq); });
    CHECK(c.find("not with a region") != std::string::npos);
    std::printf("  refused: %s | %s | %s\n", a.substr(0, 90).c_str(), b2.substr(0, 90).c_str(), c.substr(0, 90).c_str());
  }

  // MIGRATE GLUE: a record of an earlier version (the original only) gets the relation stated; idempotent
  {
    tk::Unit u = ts.unit(tk::By{"an earlier version's glue record", Json::object()});
    u.edit_manifold_producer(glued->id, "glue", Json{{"original", cortex->id}, {"form", "labelling"}}, "as an earlier version recorded it");
    u.commit();
  }
  const Json r1 = migrate_glue(sub, std::string("cortex_glued"), "Structures", "Cortex L", "Cortex R");
  std::printf("  migrate glue: %s\n", r1.dump().c_str());
  CHECK(r1["glued"][0]["status"] == "migrated" && r1["glued"][0]["relation"] == "stated");
  {
    const auto g2 = ts.find({.cls = tk::Cls::manifold, .name = "cortex_glued", .type = "surface"});
    const Json& g = g2->producer["glue"];
    CHECK(g.value("hemispheres", std::string()) == "cortex_glued_hemispheres" && g.contains("rule") && g["medial_wall"].value("entry", std::string()) == "unassigned");
    CHECK(g2->producer.contains("edits") && g2->producer["edits"].size() == 2);
    CHECK(opt_text(*sub.db().read("manifold", glued->id), "comment").value_or("").find("off the wall (ascending index)") != std::string::npos);
  }
  CHECK(migrate_glue(sub, std::string("cortex_glued"), "Structures", "Cortex L", "Cortex R")["glued"][0]["status"] == "already in the new form");
  return nxrtest::finish("test_glue_onto");
}
