// `run level_complex` on a synthetic subject (never a real dataset; spec docs/specs/2026-11-26-levels-of-one-manifold.md, section 7): the surface's dyadic grid, then the navigation
// operators of one level stated as a quantity, stored ON THE SURFACE (no manifold row), keyed by the grid and the level: their shapes and structure equal compute's levelComplex, their
// params name the grid, the level, the mesh and the metric, the record says whether the level is a valid complex; a rerun is a cache hit; another level is another set of operators.
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "brain_helpers.h"
#include "check.h"
#include "make_grids.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/io/task/task.h"
#include "nxr/operators/dec.h"
#include "nxr/select/dyadic.h"
#include "nxr/select/levels.h"
#include "run_definition.h"

using namespace nxr::io;
namespace fs = std::filesystem;
namespace tk = nxr::io::task;

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_level_complex_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(3, false);
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] { SurfaceOptions so; so.name = "cortex"; add_surface(sub, so, b.P, b.F); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  CHECK(!nxrtest::make_tiles(ds, sub, "parts 8", std::string("cortex")).empty());
  const auto ts = tk::Subject::of(sub);
  const auto grid = ts.find({.cls = tk::Cls::selection, .name = "cortex tiles", .type = "ladder"});
  CHECK(grid.has_value());
  const std::size_t manifolds = ts.find_all({.cls = tk::Cls::manifold}).size();

  for (const char* mesh : {"input", "operator"}) {
    nxrtest::run_definition(ds, sub, "level_complex", Json{{"surface", "cortex"}, {"tiles", "cortex tiles"}, {"spatial_scale", "parts 4"}, {"mesh", mesh}}, true);
    const auto d0 = ts.find({.cls = tk::Cls::operator_, .name = nxr::io::safe_name("cortex tiles L2 d0")});
    const auto s1 = ts.find({.cls = tk::Cls::operator_, .name = nxr::io::safe_name("cortex tiles L2 star1")});
    const auto cr = ts.find({.cls = tk::Cls::operator_, .name = nxr::io::safe_name("cortex tiles L2 crossing")});
    CHECK(d0 && s1 && cr);
    if (!(d0 && s1 && cr)) return 1;
    // compute's level on the same mesh: the stored shapes and counts are its
    const std::vector<double>& P = b.P;
    const auto S = nxr::geometry::surfaceArrays(P.data(), b.n, b.F.data(), static_cast<int>(b.F.size() / 3), nxr::geometry::intrinsicOptions(std::string(mesh) == "operator"));
    const auto dec = std::string(mesh) == "operator" ? nxr::operators::assembleDECOperators(S) : nxr::operators::embeddedDEC(S);
    const auto tree = nxr::select::subjectTree(P.data(), b.n, b.F.data(), static_cast<int>(b.F.size() / 3), 3);
    nxr::select::FineComplex fine;
    fine.d0 = dec.d0; fine.d1 = dec.d1; fine.star0 = dec.hodge0.diagonal(); fine.star1 = dec.hodge1.diagonal();
    const nxr::select::Level L = nxr::select::dyadicLevel(fine, tree.codes, tree.depth, tree.slots, 2);
    CHECK(d0->fact("n_rows").get<std::int64_t>() == L.edges.rows() && d0->fact("n_cols").get<std::int64_t>() == L.cells);
    CHECK(d0->fact("nnz").get<std::int64_t>() == 2 * L.edges.rows());
    CHECK(cr->fact("n_rows").get<std::int64_t>() == fine.d0.rows() && cr->fact("nnz").get<std::int64_t>() == L.crossing.nonZeros());
    CHECK(s1->fact("n_rows").get<std::int64_t>() == L.edges.rows());
    CHECK(d0->from_selection_id == grid->id && d0->from_manifold_id == grid->manifold_id);
    const Json pj = d0->fact("params_json");
    const Json params = pj.is_string() ? Json::parse(pj.get<std::string>()) : pj;
    CHECK(params["level"] == 2 && params["mesh"] == mesh && params["metric"] == "galerkin");
    CHECK(d0->producer.contains("valid_complex"));
    std::cout << "  mesh " << mesh << ": level 2 has " << L.cells << " cells, " << L.edges.rows() << " edges, " << L.faces.rows() << " junctions; valid " << L.valid << (L.valid ? "" : " (" + L.why + ")") << "\n";
  }
  // a rerun is a cache hit (the same operator), another level is other operators; no manifold row is written
  const std::string id0 = ts.find({.cls = tk::Cls::operator_, .name = nxr::io::safe_name("cortex tiles L2 d0")})->id;
  nxrtest::run_definition(ds, sub, "level_complex", Json{{"surface", "cortex"}, {"tiles", "cortex tiles"}, {"spatial_scale", "parts 4"}, {"mesh", "operator"}});
  CHECK(ts.find({.cls = tk::Cls::operator_, .name = nxr::io::safe_name("cortex tiles L2 d0")})->id == id0);
  nxrtest::run_definition(ds, sub, "level_complex", Json{{"surface", "cortex"}, {"tiles", "cortex tiles"}, {"spatial_scale", "parts 8"}});
  CHECK(ts.find({.cls = tk::Cls::operator_, .name = nxr::io::safe_name("cortex tiles L3 d0")}).has_value());
  CHECK(ts.find_all({.cls = tk::Cls::manifold}).size() == manifolds);
  return nxrtest::finish("test_level_complex");
}
