// `copy nodes` (io/copy_nodes.h): a subject's stored solves taken from another package of the same subject, never solved again. Synthetic: package A holds a closed surface
// (an icosphere) and a channels manifold; its intrinsic triangulation and Hodge-Dirac eigensolve, a sensor mesh of the channels and the mesh's own eigensolve (an open disc, *1
// floor named) are made by the definitions. Package B, a re-import of the same subject, holds the same surface and channels and nothing solved. Checks: a dry run writes nothing;
// the copy keeps the surface B holds, finds the channels there, copies the rest with B's ids; every chunk's digest equals A's and B verifies; the readers take the copy (the
// stored eigensolve's options: mollify, flip_eps, star1_floor; its rows; the vertex potentials, which check the triangulation against the surface); the same eigenmodes run on B is
// a cache hit (nothing solved); a second copy keeps everything; refused: a target whose surface differs, a target without the channels a copy names.
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>

#include "check.h"
#include "mesh_helpers.h"
#include "nxr/bst/entities.h"
#include "nxr/io/copy_nodes.h"
#include "nxr/io/measures.h"
#include "nxr/io/rollup.h"
#include "nxr/io/subject_api.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/surface.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/input.h"
#include "run_definition.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {

std::string refusal(const std::function<void()>& f) { try { f(); } catch (const std::exception& e) { return e.what(); } return ""; }
bool has(const std::string& t, const std::string& p) { return t.find(p) != std::string::npos; }

constexpr int kChannels = 60;
// the subject as an import makes it: the surface `sphere` (an icosphere, `scale` its radius) and the channels `sensors` on a cap with their type and quality labellings
void make_subject(const fs::path& package, const std::string& home, double scale, bool with_sensors) {
  CreatePackageOptions po;
  po.home_dataset = home;
  DatasetWriter w = create_package(package, "S1", po);
  SubjectWriter sub = SubjectWriter::open(w, "S1");
  sub.compose([&] {
    std::vector<double> V;
    std::vector<std::int32_t> F;
    icosphere(2, V, F, 0.08 * scale);
    SurfaceOptions so;
    so.name = "sphere";
    add_surface(sub, so, V, F);
    if (!with_sensors) return;
    std::vector<float> pos;
    const double golden = M_PI * (3.0 - std::sqrt(5.0)), z0 = std::cos(1.4);
    for (int i = 0; i < kChannels; ++i) {
      const double z = 1.0 - (1.0 - z0) * (i + 0.5) / kChannels + 0.001 * std::sin(7.0 * i), rho = std::sqrt(std::max(0.0, 1.0 - z * z)), phi = golden * i;
      for (const double u : {rho * std::cos(phi), rho * std::sin(phi), z}) pos.push_back(static_cast<float>(0.1 * u));
    }
    const std::string cid = uuid7();
    ManifoldSpec m;
    m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = kChannels;
    m.topology = {"none", 0, 1, std::nullopt, {}};
    GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/sensors"; m.geometry = g;
    m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {kChannels, 3}), {}, std::nullopt, "manifold", cid); };
    sub.manifold(std::move(m));
    std::vector<std::int64_t> type(kChannels, 0), quality(kChannels, 0);
    nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", type, {"MEG GRAD"}, {}}, {"quality", quality, {"good"}, {}}}, cid, "the montage families", std::nullopt,
                        [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
  });
  w.close();
}
// the stem of the eigensolve of a surface (`hodgeDirac`, or `hodgeDirac__<name>` for a surface that is not the primary one)
std::string stem_of(SubjectWriter& sub, const std::string& surface) { return nxr::pipelines::hodge_package_stem(nxr::pipelines::read_surface(sub, surface)); }
std::string id_of(SubjectWriter& sub, const std::string& name) {
  const auto m = find_manifold(sub, name);
  return m ? text(*m, "id") : std::string();
}
Json copied_names(const Json& list) {
  Json names = Json::array();
  for (const auto& x : list) names.push_back(x["name"]);
  return names;
}
bool lists(const Json& list, const std::string& name) {
  for (const auto& x : list) if (x["name"] == name) return true;
  return false;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_copy_nodes_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const fs::path A = root / "old" / "S1.nxr.zarr", B = root / "new" / "S1.nxr.zarr";

  // A: the solves, made once by the definitions
  make_subject(A, "old", 1.0, true);
  std::string sphere_stem, helmet_stem;
  std::vector<double> mu_sphere, mu_helmet;
  std::vector<float> rows_sphere;
  {
    DatasetWriter w = open_package_writer(A);
    SubjectWriter sub = SubjectWriter::open(w, "S1");
    nxr::pipelines::EigenSolveOptions eo;
    eo.surface = "sphere";
    eo.modes = 16;
    nxr::pipelines::eigen_solve(w, sub, eo, {});
    CHECK(!nxrtest::run_definition(w, sub, "sensor_mesh", Json{{"channels", "sensors"}, {"sensor_type", "MEG"}, {"name", "helmet"}}).empty());
    CHECK(!nxrtest::run_definition(w, sub, "eigenmodes", Json{{"surface", "helmet"}, {"modes", 12}, {"star1_floor", 1e-6}}).empty());
    sphere_stem = stem_of(sub, "sphere");
    helmet_stem = stem_of(sub, "helmet");
    const auto es = nxr::pipelines::stored_eigensolve(sub, sphere_stem);
    mu_sphere = es.mu;
    rows_sphere = nxr::pipelines::stored_eigensolve_rows(sub, es, 0, es.V + es.E + es.F);
    mu_helmet = nxr::pipelines::stored_eigensolve(sub, helmet_stem).mu;
    // a measurement of a DECLARED measure on the sphere (issue #280): a set of two members, a measure the registry does not know, declared `sum`
    sub.compose([&] {
      SelectionSpec s;
      s.name = "halves"; s.type = "set"; s.manifold_id = id_of(sub, "sphere"); s.no_elements = true;
      s.cols.set("n_members", static_cast<std::int64_t>(2));
      const std::string sel = sub.selection(std::move(s));
      MeasureDeclaration d;
      d.name = "copied_flow_thing"; d.merge = "sum"; d.description = "a synthetic additive measure";
      declare_measure(sub.db(), d, std::string("test"));
      Json rows = Json::array();
      for (int m = 0; m < 2; ++m) rows.push_back(Json{{"selection_id", sel}, {"code", m}, {"measure", "copied_flow_thing"}, {"value", 1.5 + m}});
      write_measurements(sub.db(), rows, std::string("test"));
    });
    w.close();
  }
  CHECK(!sphere_stem.empty() && !helmet_stem.empty());
  const std::string hash_a = SubjectPackage::open(A).record().content_hash;

  // B: the same subject imported again, nothing solved
  make_subject(B, "new", 1.0, true);
  std::string sphere_b;
  {
    DatasetWriter w = open_package_writer(B);
    SubjectWriter sub = SubjectWriter::open(w, "S1");
    sphere_b = id_of(sub, "sphere");
    const std::string before = SubjectPackage::open(B).content_hash();
    CopyNodesOptions dry;
    dry.manifolds = {"sphere", "helmet"};
    dry.dry_run = true;
    const Json plan = copy_nodes(A, sub, dry);
    std::printf("  dry run: copies %s\n", copied_names(plan["copied"]).dump().c_str());
    CHECK(plan["dry_run"].get<bool>());
    CHECK(lists(plan["kept"], "sphere") && lists(plan["found"], "sensors"));
    CHECK(lists(plan["copied"], "sphere_intrinsic") && lists(plan["copied"], sphere_stem + "_eigenvectors") && lists(plan["copied"], "helmet") &&
          lists(plan["copied"], "helmet_map") && lists(plan["copied"], "helmet_channels") && lists(plan["copied"], helmet_stem + "_kinds"));
    CHECK(plan["keys"].get<std::int64_t>() > 0);
    CHECK(SubjectPackage::open(B).content_hash() == before);   // nothing written
    CHECK(id_of(sub, "sphere_intrinsic").empty());

    CopyNodesOptions o;
    o.manifolds = {"sphere", "helmet"};
    o.by = "test";
    const Json r = copy_nodes(A, sub, o);
    CHECK(r["copied"].size() == plan["copied"].size() && r["keys"] == plan["keys"]);
    CHECK_EQ(id_of(sub, "sphere"), sphere_b);   // kept: the target's own row
    // the copies are B's rows: new ids, B's subject, the surface B holds, and the record of where they came from
    const auto tri = find_manifold(sub, "sphere_intrinsic");
    CHECK(tri.has_value());
    if (tri) {
      CHECK(opt_text(*tri, "derived_from_id") == std::optional<std::string>(sphere_b));
      const Json pj = Json::parse(opt_text(*tri, "producer_json").value_or("{}"));
      CHECK(pj["copied"]["from_content_hash"] == hash_a && pj["copied"]["from_package"] == "S1.nxr.zarr");
      for (const auto& x : r["copied"]) CHECK(x["id"] != x["from_id"]);
    }
    // the readers take the copy: the solve's options and spectrum, its rows byte for byte, the vertex potentials (the triangulation checked against B's surface)
    const auto es = nxr::pipelines::stored_eigensolve(sub, sphere_stem);
    CHECK(es.mu == mu_sphere);
    CHECK(es.mollify == 1e-6 && es.flip_eps == 1e-6);
    CHECK(nxr::pipelines::stored_eigensolve_rows(sub, es, 0, es.V + es.E + es.F) == rows_sphere);
    CHECK_EQ(text(es.surface, "id"), sphere_b);
    CHECK(!nxr::pipelines::stored_vertex_potentials(sub, es, nxr::pipelines::read_surface(sub, "sphere")).empty());
    const auto eh = nxr::pipelines::stored_eigensolve(sub, helmet_stem);
    CHECK(eh.mu == mu_helmet && eh.star1_floor == 1e-6);
    // the measure's declaration came with its measurement (issue #280): the target rolls it up by the same rule
    const auto copied_decl = declared_measure(sub.db(), "copied_flow_thing");
    CHECK(copied_decl.has_value() && (*copied_decl)["merge"] == "sum");
    CHECK(measure_merge(sub.db(), "copied_flow_thing") == MergeRule::sum);
    CHECK(!nxr::pipelines::stored_vertex_potentials(sub, eh, nxr::pipelines::read_surface(sub, "helmet")).empty());
    // the same runs asked again on B: cache hits, nothing solved
    const auto again = nxrtest::run_definition(w, sub, "eigenmodes", Json{{"surface", "helmet"}, {"modes", 12}, {"star1_floor", 1e-6}});
    CHECK(!again.empty() && again[0].value("hit", false));
    nxr::pipelines::EigenSolveOptions eo;
    eo.surface = "sphere";
    eo.modes = 16;
    CHECK(nxr::pipelines::eigen_solve(w, sub, eo, {}).hit);
    // a second copy: everything is there, the same: kept, nothing written
    const Json r2 = copy_nodes(A, sub, o);
    CHECK(r2["copied"].empty() && r2["keys"] == 0 && r2["kept"].size() == r["copied"].size() + 1);
    w.close();
  }
  {
    const VerifyReport v = SubjectPackage::open(B).verify();
    CHECK(v.ok);
    for (const auto& p : v.problems) std::printf("  verify B: %s\n", p.c_str());
  }

  // refused, nothing written: a target whose surface differs (another radius); a target without the channels the sensor mesh names
  {
    const fs::path C = root / "other" / "S1.nxr.zarr";
    make_subject(C, "other", 1.5, true);
    DatasetWriter w = open_package_writer(C);
    SubjectWriter sub = SubjectWriter::open(w, "S1");
    CopyNodesOptions o;
    o.manifolds = {"sphere"};
    const std::string e = refusal([&] { copy_nodes(A, sub, o); });
    std::printf("  another surface: %s\n", e.substr(0, 160).c_str());
    CHECK(has(e, "differ"));
    CHECK(id_of(sub, "sphere_intrinsic").empty());
  }
  {
    const fs::path D = root / "bare" / "S1.nxr.zarr";
    make_subject(D, "bare", 1.0, false);
    DatasetWriter w = open_package_writer(D);
    SubjectWriter sub = SubjectWriter::open(w, "S1");
    CopyNodesOptions o;
    o.manifolds = {"helmet"};
    const std::string e = refusal([&] { copy_nodes(A, sub, o); });
    std::printf("  no channels: %s\n", e.substr(0, 160).c_str());
    CHECK(has(e, "sensors") && has(e, "does not hold"));
  }
  {
    // a target that declares the measure with another merge rule: refused (one name, one meaning), nothing copied
    const fs::path E = root / "declared" / "S1.nxr.zarr";
    make_subject(E, "declared", 1.0, true);
    DatasetWriter w = open_package_writer(E);
    SubjectWriter sub = SubjectWriter::open(w, "S1");
    sub.compose([&] { MeasureDeclaration d; d.name = "copied_flow_thing"; d.merge = "none"; declare_measure(sub.db(), d, std::string("test")); });
    CopyNodesOptions o;
    o.manifolds = {"sphere"};
    const std::string e = refusal([&] { copy_nodes(A, sub, o); });
    std::printf("  another declaration: %s\n", e.substr(0, 160).c_str());
    CHECK(has(e, "copied_flow_thing") && has(e, "merge"));
    CHECK(id_of(sub, "sphere_intrinsic").empty());
  }
  return nxrtest::finish("test_copy_nodes");
}
