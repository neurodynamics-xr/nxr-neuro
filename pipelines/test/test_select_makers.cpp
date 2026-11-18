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

  std::printf("test_select_makers: ok\n");
  return 0;
}
