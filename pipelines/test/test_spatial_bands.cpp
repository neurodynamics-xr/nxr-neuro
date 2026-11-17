// The spatial-scale (Weyl) axis of the measurement pyramid on a synthetic two-hemisphere brain (never a real dataset; spec docs/specs/2026-11-14-measurement-pyramid.md, step 4):
//   - the Eigenvalue line (default subject; a line in 1/m^2) and the surface's `<surface> spatial bands` spans Selection on it: one span per ladder level, code = level, edges = the ladder's
//     Weyl band of the level (level 0 from 0, level l from half the λ of l to the λ of l: contiguous), the same as the library's `weyl(mean piece area, l + 1)` member l; a rerun keeps it
// The band set is made by the maker definition `select_bands` (rule levels), the surface's grid by `select_dyadic_surface`. The read-time roll-up of a measurement over a band's spatial level is no longer
// io's (rule 35, one definition of each measure's merge rule): it is tested by the parity test of compute's roll_up and io's query, and the pipelines' query face test.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/bst/frequency.h"
#include "nxr/filterbank.h"
#include "nxr/io/measures.h"
#include "nxr/io/spatial_bands.h"
#include "nxr/io/task/task.h"
#include "nxr/pipelines/frames.h"
#include "make_grids.h"
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
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_spatial_bands_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(3, false);
  DatasetWriter ds = create_dataset(root, "demo");
  std::string def_id;
  {
    SubjectWriter def = SubjectWriter::create(ds, "default_template", [] { SubjectOptions o; o.kind = "template"; return o; }());
    def.compose([&] { nxr::bst::ensure_frequency(def); ensure_eigenvalue(def); });
    def_id = def.id();
  }
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] { SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph; add_surface(sub, so, b.P, b.F); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  nxrtest::run_definition(ds, sub, "surface_frames", Json{{"surface", "cortex"}});
  nxrtest::make_tiles(ds, sub, "", std::string("cortex"));

  // the line
  const auto tdef = tk::Subject::of(ds, def_id);
  const auto line = tdef.find({.cls = tk::Cls::manifold, .name = kEigenvalueName, .type = "line"});
  CHECK(line && line->type.value_or("") == "line" && line->n_vertices.value_or(0) == kEigenvalueSamples);
  CHECK(code_of([&] { SubjectWriter s2 = SubjectWriter::open(ds, "S1"); ensure_eigenvalue(s2); }) == "invalid_argument");        // not in a subject that is not the default

  // the bank
  const auto made = nxrtest::make_spatial_bands(ds, sub, "cortex");
  CHECK(!made.empty());
  const auto ts0 = tk::Subject::of(sub);
  const auto bank0 = ts0.find({.cls = tk::Cls::selection, .name = "cortex spatial bands", .type = "spans"});
  CHECK(bank0.has_value());
  const int levels = static_cast<int>(bank0->n_members.value_or(0));
  std::printf("  %d spatial bands of cortex\n", levels);
  const auto ts = tk::Subject::of(sub);
  const auto bank = ts.find({.cls = tk::Cls::selection, .name = "cortex spatial bands", .type = "spans"});
  CHECK(bank && bank->n_members.value_or(0) == levels && levels >= 4 && bank->manifold_id.value_or("") == line->id);
  const auto spans = ts.spans(*bank);
  CHECK(static_cast<int>(spans.size()) == levels);
  const auto lad = ts.find({.cls = tk::Cls::selection, .name = "cortex tiles", .type = "ladder"});
  const double area = ts.members(*lad)[1].params.value("piece_area", 0.0);
  (void)area;
  bool contiguous = spans[0].start == 0.0;
  for (int l = 0; l < levels; ++l) {
    contiguous = contiguous && spans[l].code.value_or(-1) == l;
    if (l > 0) contiguous = contiguous && std::abs(spans[l].start - spans[l - 1].stop) <= 1e-9 * spans[l].start;
    contiguous = contiguous && spans[l].stop > spans[l].start;
    if (l > 0) contiguous = contiguous && std::abs(spans[l].stop / spans[l].start - 2.0) < 1e-9;      // an octave of λ
  }
  CHECK(contiguous);
  // level l IS member l of the library's Weyl bank of the mean piece area (the ladder's own record, paired)
  {
    const auto members = ts.members(*lad);
    const Json lr = members[levels - 1].params;
    const double lam_top = lr["lambda_max"].get<double>();
    CHECK(std::abs(spans[levels - 1].stop - lam_top) <= 1e-12 * lam_top);
  }
  const std::string bank_id = bank->id;
  nxrtest::make_spatial_bands(ds, sub, "cortex");   // a rerun is a cache hit: the band set keeps its id
  CHECK(ts.find({.cls = tk::Cls::selection, .name = "cortex spatial bands", .type = "spans"})->id == bank_id);

  // every band is stated in a chart of the Eigenvalue line: eigenvalue (1/m^2, the reference) and wavelength (mm, 2 pi / sqrt(lambda)); the wavelength of a band's edge is read back in mm
  {
    const double lam = spans[levels - 1].stop;                          // 1/m^2
    const double wavelength_mm = 1000.0 * 2.0 * M_PI / std::sqrt(lam);
    CHECK(wavelength_mm > 0.0 && std::isfinite(wavelength_mm));
  }
  std::printf("test_spatial_bands: ok\n");
  return 0;
}
