// The spatial-scale (Weyl) axis of the measurement pyramid on a synthetic two-hemisphere brain (never a real dataset; spec docs/specs/2026-11-14-measurement-pyramid.md, step 4):
//   - the Eigenvalue line (default subject; a line in 1/m^2) and the surface's `<surface> spatial bands` spans Selection on it: one span per ladder level, code = level, edges = the ladder's
//     Weyl band of the level (level 0 from 0, level l from half the λ of l to the λ of l: contiguous), the same as the library's `weyl(mean piece area, l + 1)` member l; a rerun keeps it
//   - a measurement's spatial band is its `within` key, and the lookup reads it: `spatialBand` selects the row, rolls it up in space exactly as the brute-force sum over leaves
//   - a measurement with a frequency band AND a spatial band takes the joint bands Selection (code = band * n_spatial + spatial band): `band` and `spatialBand` together pick one row, either alone
//     with several of the other is refused naming it; a band not stored is refused; no schema change (the rows are ordinary selection_measurement rows)
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
#include "nxr/pipelines/pyramid.h"

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
  { FramesComputeOptions fo; frames_compute(ds, sub, fo); }

  // the line
  const auto tdef = tk::Subject::of(ds, def_id);
  const auto line = tdef.find({.cls = tk::Cls::manifold, .name = kEigenvalueName, .type = "line"});
  CHECK(line && line->type.value_or("") == "line" && line->n_vertices.value_or(0) == kEigenvalueSamples);
  CHECK(code_of([&] { SubjectWriter s2 = SubjectWriter::open(ds, "S1"); ensure_eigenvalue(s2); }) == "invalid_argument");        // not in a subject that is not the default

  // the bank
  const SpatialBandsResult r = pyramid_spatial_bands(ds, sub, SpatialBandsOptions{});
  const int levels = r.info["levels"].get<int>();
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
  CHECK(pyramid_spatial_bands(ds, sub, SpatialBandsOptions{}).info["selection_id"].get<std::string>() == bank_id);     // a rerun keeps it

  // measurements with a spatial band as the within key: arrays [leaves] on the leaf partition
  const auto ladder_members = ts.members(*lad);
  const auto part = ladder_members.back();
  const int D = levels - 1;
  Json pp = part.params;
  const std::int64_t L = part.n_members.value_or(0);
  pp["atlas"] = Json{{"tree", "subject"}, {"level", D}, {"areas", pp["areas"]}, {"node", pp["node"]}};
  ts.set_params(part.id, pp);
  // the measured field: a scalar map on the surface (the rows are of a field)
  const auto surf = ts.find({.cls = tk::Cls::manifold, .name = "cortex", .type = "surface"});
  std::vector<float> map(static_cast<std::size_t>(b.n), 1.0f);
  std::string fid;
  sub.compose([&] { FieldSpec f; f.name = "probe"; f.path = "source/probe"; f.kind = "map"; f.manifold_id = surf->id; f.value_type = "scalar"; f.data = view(map, {b.n}); fid = sub.field(std::move(f)); });
  auto value = [&](int band, int sb, std::int64_t leaf) { return 1000.0 * band + 10.0 * sb + 0.5 * static_cast<double>(leaf) + 1.0; };
  const int NB = 3, NS = levels;
  sub.compose([&] {
    for (int sb = 0; sb < NS; ++sb) {            // spatial band only
      std::vector<double> arr(static_cast<std::size_t>(L));
      for (std::int64_t l = 0; l < L; ++l) arr[static_cast<std::size_t>(l)] = value(0, sb, l);
      const std::string path = pp["node"].get<std::string>() + "/probe_sb" + std::to_string(sb);
      sub.write_array(path, view(arr, {L}));
      Cols c; c.set("measure", "sum").set("of_field_id", fid).set("array_path", path).set("within_selection_id", bank_id).set("within_code", static_cast<std::int64_t>(sb)).set("computed_by", "test");
      sub.measurement(part.id, c);
    }
  });
  auto ask = [&](const Json& q) { return query_measurements(sub.db(), sub.store(), q); };
  for (const int sb : {0, 2, NS - 1}) {
    for (const int lv : {D, 1, -1}) {
      const Json rows = ask(Json{{"selectionId", part.id}, {"measure", "sum"}, {"ofFieldId", fid}, {"spatialBand", sb}, {"spaceLevel", lv}});
      std::map<std::int64_t, double> want;
      for (std::int64_t l = 0; l < L; ++l) want[lv < 0 ? 0 : l >> (D - lv)] += value(0, sb, l);
      bool ok = rows.size() == want.size();
      for (const Json& rw : rows) ok = ok && std::abs(rw["value"].get<double>() - want[rw["code"].get<std::int64_t>()]) <= 1e-9 && rw["spatial_band"] == sb && rw["band"].is_null();
      CHECK(ok);
    }
  }
  CHECK(code_of([&] { ask(Json{{"selectionId", part.id}, {"measure", "sum"}, {"ofFieldId", fid}, {"spatialBand", NS + 3}}); }) == "no_such_row");
  CHECK(code_of([&] { ask(Json{{"selectionId", part.id}, {"measure", "sum"}, {"ofFieldId", fid}, {"band", 1}}); }) == "no_such_row");     // these rows have no frequency band

  // the joint bands: frequency band x spatial band as one within key
  const auto fband = tdef.find({.cls = tk::Cls::selection, .name = nxr::bst::kBandsName, .type = "spans"});
  CHECK(fband.has_value());
  const std::string key_id = band_key_selection(sub, fband->id, bank_id, "cortex bands x spatial bands");
  CHECK(band_key_selection(sub, fband->id, bank_id, "cortex bands x spatial bands") == key_id);
  const WithinKind wk = within_kind(sub.db(), key_id);
  CHECK(wk.kind == WithinKind::BandKey && wk.n_spatial == NS);
  CHECK(within_kind(sub.db(), bank_id).kind == WithinKind::SpatialBands && within_kind(sub.db(), fband->id).kind == WithinKind::Bands && within_kind(sub.db(), std::nullopt).kind == WithinKind::None);
  std::string fid2;
  sub.compose([&] { FieldSpec f; f.name = "probe2"; f.path = "source/probe2"; f.kind = "map"; f.manifold_id = surf->id; f.value_type = "scalar"; f.data = view(map, {b.n}); fid2 = sub.field(std::move(f)); });
  sub.compose([&] {
    for (int bd = 0; bd < NB; ++bd)
      for (int sb = 0; sb < NS; ++sb) {
        std::vector<double> arr(static_cast<std::size_t>(L));
        for (std::int64_t l = 0; l < L; ++l) arr[static_cast<std::size_t>(l)] = value(bd, sb, l);
        const std::string path = pp["node"].get<std::string>() + "/probe2_b" + std::to_string(bd) + "_s" + std::to_string(sb);
        sub.write_array(path, view(arr, {L}));
        Cols c; c.set("measure", "sum").set("of_field_id", fid2).set("array_path", path).set("within_selection_id", key_id).set("within_code", band_key_code(bd, sb, NS)).set("computed_by", "test");
        sub.measurement(part.id, c);
      }
  });
  for (const auto& [bd, sb] : std::vector<std::pair<int, int>>{{0, 0}, {1, 2}, {2, NS - 1}}) {
    const Json rows = ask(Json{{"selectionId", part.id}, {"measure", "sum"}, {"ofFieldId", fid2}, {"band", bd}, {"spatialBand", sb}, {"spaceLevel", 1}});
    std::map<std::int64_t, double> want;
    for (std::int64_t l = 0; l < L; ++l) want[l >> (D - 1)] += value(bd, sb, l);
    bool ok = rows.size() == want.size();
    for (const Json& rw : rows) ok = ok && std::abs(rw["value"].get<double>() - want[rw["code"].get<std::int64_t>()]) <= 1e-9 && rw["band"] == bd && rw["spatial_band"] == sb;
    CHECK(ok);
  }
  // one of the two alone, with several of the other stored: refused (naming what to give)
  CHECK(code_of([&] { ask(Json{{"selectionId", part.id}, {"measure", "sum"}, {"ofFieldId", fid2}, {"band", 1}}); }) == "invalid_argument");
  CHECK(code_of([&] { ask(Json{{"selectionId", part.id}, {"measure", "sum"}, {"ofFieldId", fid2}, {"spatialBand", 1}}); }) == "invalid_argument");
  CHECK(code_of([&] { ask(Json{{"selectionId", part.id}, {"measure", "sum"}, {"ofFieldId", fid2}, {"band", NB}, {"spatialBand", 0}}); }) == "no_such_row");
  std::printf("test_spatial_bands: ok\n");
  return 0;
}
