// The selection MAKER definitions on a synthetic two-hemisphere brain (never a real dataset; spec docs/specs/2026-11-17-consolidation.md section 1):
//   - `select_dyadic_surface` (rule subject): the grid's finest tile is a QUANTITY; "parts N" resolves to level log2(N) exactly, any other power of two is refused with the reason, a bare number (a level
//     index) is refused; the grid is stored once as `<surface> tiles` and a rerun is a cache hit
//   - `select_bands` (rule levels): one span of the Eigenvalue line per level of the grid, contiguous octaves of lambda
//   - `glue` makes the glued surface as a manifold of its own, with its map and its bridge selection
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "brain_helpers.h"
#include "check.h"
#include "make_grids.h"
#include "nxr/bst/frequency.h"
#include "nxr/io/spatial_bands.h"
#include "nxr/io/task/task.h"
#include "run_definition.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
namespace tk = nxr::io::task;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_select_makers_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(3, false);
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter def = SubjectWriter::create(ds, "default_template", [] { SubjectOptions o; o.kind = "template"; return o; }());
    def.compose([&] { nxr::bst::ensure_frequency(def); ensure_eigenvalue(def); });
  }
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    // the Desikan-Killiany atlas with its medial wall (code 0, `unassigned`): the caps facing each other, which the glue's bridge replaces
    std::vector<std::int32_t> dk(static_cast<std::size_t>(b.n), 1);
    for (int i = 0; i < b.n; ++i) if (std::abs(b.P[static_cast<std::size_t>(3 * i + 1)]) < 0.6) dk[static_cast<std::size_t>(i)] = 0;
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      so.atlases.push_back(AtlasMember{"Desikan-Killiany", dk, {"unassigned", "cortex"}, {{{0, 0, 0}}, {{200, 200, 200}}}});
      add_surface(sub, so, b.P, b.F);
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  nxrtest::run_definition(ds, sub, "surface_frames", Json{{"surface", "cortex"}});

  // the grid: a quantity, never a level index
  CHECK(!nxrtest::make_tiles(ds, sub, "parts 8", std::string("cortex")).empty());
  const auto ts = tk::Subject::of(sub);
  const auto grid = ts.find({.cls = tk::Cls::selection, .name = "cortex tiles", .type = "ladder"});
  CHECK(grid.has_value());
  CHECK(grid->n_members.value_or(0) == 4);                                   // levels 0..3: parts 8 per hemisphere is level 3 (2^3)
  CHECK(code_of([&] { nxrtest::make_tiles(ds, sub, "parts 6", std::string("cortex"), true); }) == "invalid_argument");   // parts are exact powers of two
  CHECK(code_of([&] { nxrtest::make_tiles(ds, sub, "3", std::string("cortex"), true); }) == "invalid_argument");           // a bare number is a level index: refused (the index chart is explicit: "index:3")
  const std::string id0 = grid->id;
  nxrtest::make_tiles(ds, sub, "parts 8", std::string("cortex"));                // a rerun: a cache hit, the same grid
  CHECK(ts.find({.cls = tk::Cls::selection, .name = "cortex tiles", .type = "ladder"})->id == id0);

  // the band set of the grid's levels, on the Eigenvalue line
  CHECK(!nxrtest::make_spatial_bands(ds, sub, "cortex").empty());
  const auto bank = ts.find({.cls = tk::Cls::selection, .name = "cortex spatial bands", .type = "spans"});
  CHECK(bank && bank->n_members.value_or(0) == grid->n_members.value_or(-1));

  // the glued cortex: its own manifold, its map to the original (-1 on the bridge) and the bridge as a labelled selection
  nxrtest::run_definition(ds, sub, "glue", Json{{"surface", "cortex"}});
  const auto glued = ts.find({.cls = tk::Cls::manifold, .name = "cortex_glued", .type = "surface"});
  const auto bridge = ts.find({.cls = tk::Cls::selection, .name = "cortex_glued_bridge"});
  CHECK(glued.has_value() && bridge.has_value());
  CHECK(glued->n_vertices.value_or(0) > 0);


  // THE PRODUCT with COUPLED LEVELS (`select_product`, `couple`): a surface factor's level from a stated scale (parts per hemisphere), a time factor's from a frequency band, in closed form; the product records each
  // factor's level and snapped quantity, and the stored selection has the surface level's tiles x the band's time tiles members
  {
    const int M = 2, T = 4096;
    const double sfreq = 250.0;
    std::vector<float> xf(static_cast<std::size_t>(M) * T, 0.0f);
    sub.compose([&] {
      const std::string cid = uuid7();
      std::vector<float> pos(static_cast<std::size_t>(M) * 3, 0.1f);
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec gs; gs.form = "stored"; gs.cs_key = "scs"; gs.positions_path = "sensor/sensors"; m.geometry = gs;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      ManifoldSpec tm;
      tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      tm.cols.set("n_edges", T - 1);
      GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
      DimensionSpec dm; dm.name = "time"; dm.n_vertices = T; dm.spacing = 1.0 / sfreq; dm.origin = 0.0; dm.unit = "s";
      tg.dimensions.push_back(dm); tm.geometry = tg;
      const std::string tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      FieldSpec rec;
      rec.name = "rest"; rec.path = tk::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, std::int64_t(4096)}; rec.array.compress = false; rec.cols.set("unit", "A");
      sub.field(std::move(rec));
    });
    nxrtest::make_time_grids(ds, sub, "band", Json{{"band", "8-16 Hz"}});               // the band's time tiles (the Studio's fitLevelOf), named by their rule
    const auto tl = ts.find({.cls = tk::Cls::selection, .name = "rest time grid band 8-16 Hz"});
    CHECK(tl.has_value());
    nxrtest::run_definition(ds, sub, "select_product", Json{{"recording", "rest"}, {"surface", "cortex"}, {"time_grid", "rest time grid band 8-16 Hz"},
                                                               {"couple", Json::array({Json{{"factor", 0}, {"scale", "parts 4"}}})}});
    const auto pr = ts.find({.cls = tk::Cls::selection, .name = "rest cortex product"});
    CHECK(pr.has_value());
    if (pr) {
      const Json f = pr->params["product"];                                               // io's record of the factors: [{selection, members, level?}]
      CHECK(f.size() == 2 && f[0]["level"] == 2 && f[0]["members"] == 8);                 // parts 4 per hemisphere: level 2, 4 tiles a piece, two pieces
      CHECK(!f[1].contains("level") && f[1]["members"] == 32);                            // the band's tiles: 4096 samples in tiles of 128 (4 cycles of 12 Hz at 250 Hz, a power of two)
      CHECK(pr->params["radices"] == Json::array({8, 32}) && pr->params["factors_snapped"][0]["snapped"]["parts_per_hemisphere"] == 4);
    }
  }

  std::printf("test_select_makers: ok\n");
  return 0;
}
