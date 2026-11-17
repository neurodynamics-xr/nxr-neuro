// the definition `measure` on a synthetic two-hemisphere brain: Fields measured over a labelling (a `set` Selection with a code per vertex and a dictionary) by the registry's measures.
// The values equal an independent computation for each weight rule (none, barycentric, voronoi), a constant Field measures as itself under every rule, the members' own areas add up to the
// surface's, a re-run is a cache hit, a weight that is not stated is refused, the lookup derives a standard deviation, the shape, impulse and clearance factors, the skewness and the kurtosis
// from the stored ingredients, and the median (merge none) is the member's own, refused under a weight. Written through the one definition (run_shipped), read through the one query.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <functional>
#include <sstream>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/intrinsic_surface.h"
#include "nxr/io/measures.h"
#include "nxr/io/task/task.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/run_shipped.h"
#include "nxr/stats.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_measure_pipeline_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const Brain b = brain(3, false);
  const int nV = b.n;
  // the partition: 7 members (codes 0..6), a few vertices in none (-1); and three Fields
  std::vector<std::int32_t> codes(static_cast<size_t>(nV));
  for (int i = 0; i < nV; ++i) codes[i] = i % 23 == 0 ? -1 : (i < b.nHalf ? i % 4 : 4 + i % 3);
  std::vector<float> xs(static_cast<size_t>(nV)), two(static_cast<size_t>(nV), 2.0f), other(static_cast<size_t>(nV), 9.0f);
  for (int i = 0; i < nV; ++i) xs[i] = static_cast<float>(b.P[3 * i]);
  DatasetWriter ds = create_dataset(root, "demo");
  std::string mid, xid, cid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      mid = add_surface(sub, so, b.P, b.F).id;
      DictionarySpec dsp; dsp.name = "parcels"; dsp.names = {"a", "b", "c", "d", "e", "f", "g"};
      const DictionaryResult dict = sub.dictionary(dsp);
      SelectionSpec sel; sel.name = "parcels"; sel.path = "source/parcels"; sel.type = "set"; sel.manifold_id = mid; sel.data = view(codes, {nV});
      sel.cols.set("dictionary_id", dict.id).set("cell", "0");
      sub.selection(sel);
      auto field = [&](const std::string& name, const std::vector<float>& v) {
        FieldSpec f; f.name = name; f.path = "source/" + name; f.kind = "map"; f.manifold_id = mid; f.data = view(v, {nV});
        f.cols.set("unit", "a.u.");
        if (name == "map_x") f.cols.json("producer_json", nxr::io::Json{{"spectral_scaling", nxr::io::Json{{"stored", "density"}, {"degree", 1}, {"band_width_hz", 4.0}, {"bin_enbw_hz", 0.375}, {"power_unit", "a.u.\xc2\xb2"}}}});   // a spectrum: a density over a 4 Hz band
        return sub.field(std::move(f));
      };
      xid = field("map_x", xs); cid = field("map_two", two); field("other", other);
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S");
  const std::string sid = task::Subject::of(sub).require({.cls = task::Cls::selection, .name = "parcels"}, "the parcels labelling").id;
  {   // a labelling holding -1 (in no member): its members are the codes 0 .. 6, the -1 is not one
    const auto node = task::Subject::of(sub).by_id(task::Cls::selection, sid);
    CHECK(node.has_value() && node->n_members == 7 && node->n_elements == nV);
  }
  // a measure's value for a member, read through the one query (an ingredient as stored, a derived measure from its ingredients)
  auto value = [&](const std::string& fid, int code, const std::string& m) {
    Json q = Json{{"selectionId", sid}, {"measure", m}};
    if (!fid.empty()) q["ofFieldId"] = fid;
    const Json r = query_measurements(sub.db(), sub.store(), q);
    for (const auto& row : r) if (row["code"].get<std::int64_t>() == code) return row["value"].get<double>();
    return std::nan("");
  };
  // the measure definition on one Field of the labelling, the weight stated; the report says whether it was a cache hit
  auto run = [&](const std::string& field, const std::string& weight, const std::vector<std::string>& measures, bool replace) {
    Json set = Json{{"field", field}, {"over", "parcels"}, {"measures", measures}};
    if (!weight.empty()) set["weight"] = weight;
    return nxr::pipelines::run_shipped(ds, sub, "measure", set, replace);
  };
  auto throws = [&](const std::function<void()>& f) { try { f(); } catch (const Error&) { return true; } return false; };
  // the independent computation: the weighted mean of map_x over each member
  auto expected = [&](const std::vector<double>& w, int code) {
    double sw = 0, sx = 0;
    for (int i = 0; i < nV; ++i) if (codes[i] == code) { const double wi = w.empty() ? 1.0 : w[i]; sw += wi; sx += wi * static_cast<double>(xs[i]); }
    return sx / sw;
  };
  nxr::ladder::Mesh lm; lm.positions = b.P; lm.faces = b.F; lm.n = nV;
  const std::vector<double> bary = nxr::ladder::vertexAreas(lm);
  nxr::manifold::IntrinsicSurface isurf(b.P.data(), nV, b.F.data(), b.nF);
  const std::vector<double> voro = isurf.inputMixedVoronoiMass();

  const std::vector<std::string> basic = {"mean", "max", "min", "mean_square", "abs_max", "area", "n_elements"};

  // each weight rule: the weighted mean of map_x over each member, a constant Field as itself, a re-run a hit, the weight changed replaces
  struct Rule { const char* name; const std::vector<double>* w; };
  for (const Rule& r : {Rule{"barycentric", &bary}, Rule{"none", nullptr}, Rule{"voronoi", &voro}}) {
    const std::vector<double> none;
    const std::vector<double>& w = r.w ? *r.w : none;
    run("map_x", r.name, basic, true);
    run("map_two", r.name, basic, true);
    double worst = 0;
    for (int k = 0; k < 7; ++k) {
      worst = std::max(worst, std::abs(value(xid, k, "mean") - expected(w, k)) / std::max(1e-12, std::abs(expected(w, k))));
      CHECK(std::abs(value(cid, k, "mean") - 2.0) < 1e-12);
      CHECK(value("", k, "n_elements") > 0);
    }
    std::printf("  %s: mean max rel err %.2e\n", r.name, worst);
    CHECK(worst < 1e-12);
    if (r.w) {   // the members' areas add up to the labelled part of the surface
      double area_sum = 0, total = 0;
      for (int k = 0; k < 7; ++k) area_sum += value("", k, "area");
      for (int i = 0; i < nV; ++i) if (codes[i] >= 0) total += (*r.w)[i];
      CHECK(std::abs(area_sum - total) < 1e-12 * total);
    }
  }
  {   // a re-run with the same parameters is a hit
    const auto again = run("map_x", "voronoi", basic, false);
    CHECK(!again.empty() && again[0].value("hit", false));
  }
  {   // barycentric and voronoi differ on a real mesh
    double differs = 0;
    for (int k = 0; k < 7; ++k) differs = std::max(differs, std::abs(expected(voro, k) - expected(bary, k)));
    CHECK(differs > 1e-6);
  }
  const std::vector<std::string> features = {"mean", "mean_square", "abs_max", "mean_abs", "mean_sqrt_abs", "mean_cube", "mean_fourth", "median"};
  run("map_x", "none", features, true);
  // the DERIVED measures of the stored ingredients: shape, impulse and clearance factors, skewness and kurtosis within each member, the median as the member's own value
  {
    double worst = 0;
    for (const char* m : {"shape_factor", "impulse_factor", "clearance_factor"}) {
      const Json q = query_measurements(sub.db(), sub.store(), Json{{"selectionId", sid}, {"measure", m}, {"ofFieldId", xid}});
      CHECK(q.size() == 7);
      for (const auto& row : q) {
        const int k = static_cast<int>(row["code"].get<std::int64_t>());
        double sa = 0, ss = 0, sq = 0, pk = 0; int n = 0;
        for (int i = 0; i < nV; ++i) if (codes[i] == k) { const double v = xs[i]; sa += std::abs(v); ss += std::sqrt(std::abs(v)); sq += v * v; pk = std::max(pk, std::abs(v)); ++n; }
        const double want = std::string(m) == "shape_factor" ? std::sqrt(sq / n) / (sa / n) : std::string(m) == "impulse_factor" ? pk / (sa / n) : pk / ((ss / n) * (ss / n));
        worst = std::max(worst, std::abs(row["value"].get<double>() / want - 1));
      }
    }
    std::printf("  shape, impulse, clearance factors vs direct: %.2e\n", worst);
    CHECK(worst < 1e-9);
    double wm = 0, wd = 0;
    for (const char* m : {"skewness", "kurtosis", "median"}) {
      const Json q = query_measurements(sub.db(), sub.store(), Json{{"selectionId", sid}, {"measure", m}, {"ofFieldId", xid}});
      CHECK(q.size() == 7);
      for (const auto& row : q) {
        const int k = static_cast<int>(row["code"].get<std::int64_t>());
        std::vector<double> x; for (int i = 0; i < nV; ++i) if (codes[i] == k) x.push_back(xs[i]);
        const std::string name = m;
        if (name == "median") { wd = std::max(wd, std::abs(row["value"].get<double>() - nxr::stats::median(x))); continue; }
        const double want = name == "skewness" ? nxr::stats::skewness(x) : nxr::stats::kurtosis(x);
        wm = std::max(wm, std::abs(row["value"].get<double>() - want) / std::max(1.0, std::abs(want)));
      }
    }
    std::printf("  skewness, kurtosis within members vs direct: %.2e; median: %.2e\n", wm, wd);
    CHECK(wm < 1e-9 && wd < 1e-12);
    CHECK(throws([&] { run("map_x", "barycentric", {"median"}, true); }));   // a weighted median does not exist (merge none)
  }
  {   // the lookup derives sd from the stored mean and mean_square: zero for the constant, positive for the map
    run("map_x", "barycentric", basic, true);
    const Json sd = query_measurements(sub.db(), sub.store(), Json{{"selectionId", sid}, {"measure", "sd"}, {"ofFieldId", cid}});
    CHECK(sd.is_array() && sd.size() == 7);
    for (const auto& row : sd) CHECK(std::abs(row["value"].get<double>()) < 1e-6);
    const Json sdx = query_measurements(sub.db(), sub.store(), Json{{"selectionId", sid}, {"measure", "sd"}, {"ofFieldId", xid}});
    CHECK(sdx.size() == 7 && sdx[0]["value"].get<double>() > 0);
  }
  // refusals: an unstated weight, a labelling that does not exist, a field that does not exist
  CHECK(throws([&] { run("map_x", "", basic, true); }));
  CHECK(throws([&] { nxr::pipelines::run_shipped(ds, sub, "measure", Json{{"field", "map_x"}, {"over", "nope"}, {"weight", "none"}, {"measures", basic}}, true); }));
  CHECK(throws([&] { run("nothing", "none", basic, true); }));
  fs::remove_all(root);
  return nxrtest::finish("test_measure_pipeline");
}
