// measure compute on a synthetic two-hemisphere brain: Fields measured over a partition (a `set` Selection with a code per vertex and a dictionary).
// The rows are the Studio's (one per member, Field and measure, with the member's weight), their means equal an independent computation for each weight
// rule (none · barycentric · voronoi), a constant Field measures as itself under every rule, the members' areas add up to the surface's, a re-run is a
// cache hit, another recipe's rows are refused unless --replace, the recipe is an entry of the dataset's `recipes` dictionary, the lookup derives a
// standard deviation from the stored rows (zero for the constant), and a weight that is not stated is refused.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/intrinsic_surface.h"
#include "nxr/io/measures.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/measure.h"
#include "nxr/io/statistic.h"
#include "nxr/pipelines/statistics.h"
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
  auto value = [&](const std::string& fid, int code, const std::string& m) {
    const Value of = fid.empty() ? Value{std::monostate{}} : Value{fid};
    const auto found = testing::Probe(sub).rows("selection_measurement", {{"selection_id", sid}, {"of_field_id", of}, {"code", static_cast<std::int64_t>(code)}, {"measure", m}});
    const auto r = found.empty() ? std::optional<Row>() : std::optional<Row>(found.front());
    if (!r) return std::nan(""); const Value& x = r->at("value"); return std::holds_alternative<double>(x) ? std::get<double>(x) : static_cast<double>(std::get<std::int64_t>(x));
  };
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

  MeasureComputeOptions o;
  o.labelling = "parcels"; o.fields = "map_.*"; o.weight = "barycentric";
  const auto r1 = measure_compute(ds, sub, o);
  CHECK_EQ(r1.info["members"].get<int>(), 7);
  CHECK_EQ(static_cast<int>(r1.info["fields"].size()), 2);            // map_x and map_two, not `other`
  CHECK_EQ(r1.info["rows"].get<std::int64_t>(), 7 * 2 + 2 * 7 * 5);    // n_elements and area per member; 5 measures per member and Field
  double worst = 0, area_sum = 0, total = 0;
  for (int k = 0; k < 7; ++k) {
    worst = std::max(worst, std::abs(value(xid, k, "mean") - expected(bary, k)) / std::max(1e-12, std::abs(expected(bary, k))));
    CHECK(std::abs(value(cid, k, "mean") - 2.0) < 1e-12);
    area_sum += value("", k, "area");
  }
  for (int i = 0; i < nV; ++i) if (codes[i] >= 0) total += bary[i];
  std::printf("  barycentric: mean max rel err %.2e; members' areas %.6g of %.6g m^2\n", worst, area_sum, total);
  CHECK(worst < 1e-12);
  CHECK(std::abs(area_sum - total) < 1e-12 * total);
  const testing::Probe probe(sub);
  const auto recipe_entries = [&] {   // the entries of the dataset's recipes dictionary
    std::vector<Row> out;
    for (const Row& d : probe.rows("dictionary", {{"name", "recipes"}, {"scope", "dataset"}}))
      for (const Row& e : probe.rows("dictionary_entry", {{"dictionary_id", text(d, "id")}})) out.push_back(e);
    return out;
  };
  const auto recipes0 = recipe_entries();
  const auto rec = recipes0.empty() ? std::optional<Row>() : std::optional<Row>(recipes0.front());
  CHECK(rec.has_value() && text(*rec, "name") == "partition-barycentric@1" && text(*rec, "function") == "reduction");
  if (rec) { const Json p = Json::parse(text(*rec, "params_json")); CHECK(p["measure"]["kind"] == "reduction" && p["measure"]["weight"] == "barycentric"); }
  const auto by_rows = probe.rows("selection_measurement", {{"selection_id", sid}});
  const auto by = by_rows.empty() ? std::optional<Row>() : std::optional<Row>(by_rows.front());
  CHECK(by.has_value() && text(*by, "computed_by").rfind("recipe partition-barycentric@1 \xc2\xb7 nxr-neuro ", 0) == 0);

  // the LOOKUP (the Studio's query, ported): sd derived from the stored mean and mean_square; zero for the constant
  const Json sd = query_measurements(sub.db(), Json{{"selectionId", sid}, {"measure", "sd"}, {"ofFieldId", cid}});
  CHECK(sd.is_array() && sd.size() == 7);
  for (const auto& row : sd) CHECK(std::abs(row["value"].get<double>()) < 1e-6);
  const Json sdx = query_measurements(sub.db(), Json{{"selectionId", sid}, {"measure", "sd"}, {"ofFieldId", xid}});
  CHECK(sdx.size() == 7 && sdx[0]["value"].get<double>() > 0);

  // a re-run is a cache hit; another recipe is refused, then replaces with --replace
  const auto r2 = measure_compute(ds, sub, o);
  CHECK(r2.info["fields"][0]["cache"] == "hit" && r2.info["fields"][1]["cache"] == "hit");
  o.weight = "none";
  bool refused = false;
  try { measure_compute(ds, sub, o); } catch (const Error&) { refused = true; }
  CHECK(refused);
  o.replace = true;
  measure_compute(ds, sub, o);
  worst = 0;
  for (int k = 0; k < 7; ++k) worst = std::max(worst, std::abs(value(xid, k, "mean") - expected({}, k)) / std::max(1e-12, std::abs(expected({}, k))));
  std::printf("  none: mean max rel err %.2e\n", worst);
  CHECK(worst < 1e-12);
  CHECK(std::isnan(value("", 0, "area")));                             // no area without area weights (the barycentric one is gone with its recipe)
  CHECK(value("", 0, "n_elements") > 0);
  o.weight = "voronoi";
  measure_compute(ds, sub, o);
  worst = 0;
  double differs = 0;
  for (int k = 0; k < 7; ++k) {
    worst = std::max(worst, std::abs(value(xid, k, "mean") - expected(voro, k)) / std::max(1e-12, std::abs(expected(voro, k))));
    differs = std::max(differs, std::abs(expected(voro, k) - expected(bary, k)));
    CHECK(std::abs(value(cid, k, "mean") - 2.0) < 1e-12);
  }
  std::printf("  voronoi: mean max rel err %.2e (barycentric differs by up to %.2e)\n", worst, differs);
  CHECK(worst < 1e-12 && differs > 1e-6);
  CHECK_EQ(static_cast<int>([&] { std::size_t n = 0; for (const Row& d : probe.rows("dictionary", {{"name", "recipes"}})) n += probe.rows("dictionary_entry", {{"dictionary_id", text(d, "id")}}).size(); return n; }()), 3);

  // MATLAB'S TIME-DOMAIN FEATURES over a partition: mean_abs and mean_sqrt_abs stored, shape, impulse and clearance factors derived at read (unweighted here)
  {
    MeasureComputeOptions fo = o; fo.fields = "map_x"; fo.weight = "none"; fo.measures = {"mean", "mean_square", "abs_max", "mean_abs", "mean_sqrt_abs", "mean_cube", "mean_fourth", "median"}; fo.replace = true; fo.recipe = std::string("features");
    measure_compute(ds, sub, fo);
    double worst = 0;
    for (const char* m : {"shape_factor", "impulse_factor", "clearance_factor"}) {
      const Json q = query_measurements(sub.db(), Json{{"selectionId", sid}, {"measure", m}, {"ofFieldId", xid}});
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
    // SKEWNESS and KURTOSIS WITHIN each member (1/n, MATLAB's), derived at read from the stored raw moments, against the direct central moments
    double wm = 0;
    for (const char* m : {"skewness", "kurtosis"}) {
      const Json q = query_measurements(sub.db(), Json{{"selectionId", sid}, {"measure", m}, {"ofFieldId", xid}});
      CHECK(q.size() == 7);
      for (const auto& row : q) {
        const int k = static_cast<int>(row["code"].get<std::int64_t>());
        std::vector<double> x; for (int i = 0; i < nV; ++i) if (codes[i] == k) x.push_back(xs[i]);
        const double want = std::string(m) == "skewness" ? nxr::stats::skewness(x) : nxr::stats::kurtosis(x);
        wm = std::max(wm, std::abs(row["value"].get<double>() - want) / std::max(1.0, std::abs(want)));
      }
    }
    std::printf("  skewness, kurtosis within members vs direct: %.2e\n", wm);
    CHECK(wm < 1e-9);
    // the MEDIAN within each member (exact, unweighted; numpy's), stored with reduction none; a weighted median is refused
    {
      const Json q = query_measurements(sub.db(), Json{{"selectionId", sid}, {"measure", "median"}, {"ofFieldId", xid}});
      CHECK(q.size() == 7);
      double wd = 0;
      for (const auto& row : q) {
        const int k = static_cast<int>(row["code"].get<std::int64_t>());
        std::vector<double> x; for (int i = 0; i < nV; ++i) if (codes[i] == k) x.push_back(xs[i]);
        wd = std::max(wd, std::abs(row["value"].get<double>() - nxr::stats::median(x)));
      }
      std::printf("  median within members vs direct: %.2e\n", wd);
      CHECK(wd < 1e-12);
      MeasureComputeOptions wmed = fo; wmed.weight = "barycentric"; wmed.measures = {"median"}; wmed.recipe = std::string("wmedian");
      bool refusedw = false;
      try { measure_compute(ds, sub, wmed); } catch (const Error&) { refusedw = true; }
      CHECK(refusedw);
    }
    MeasureComputeOptions back = o; back.replace = true;
    measure_compute(ds, sub, back);                                     // the test's recipe again, for what follows
  }
  // SUMMARIES of SCALAR ROWS (statistic compute, docs/specs/2026-11-13-statistic-summaries.md): map_x's parcel means summarised over the 7 parcels
  {
    StatisticComputeOptions so;
    so.over = "parcels"; so.of = "map_x"; so.measure = "mean"; so.summaries = {"median", "mean", "percentile:75", "sd"};
    const auto r = statistic_compute(ds, sub, so);
    CHECK(!r.hit && r.info["rows"] == 1 && r.info["members"] == 7);
    std::vector<double> means;
    for (int k = 0; k < 7; ++k) means.push_back(value(xid, k, "mean"));
    const StoredStatistic st = read_statistic(sub, r.info["name"].get<std::string>());
    CHECK(text(st.row, "method") == "summary" && st.shape == (std::vector<std::int64_t>{1, 4}));
    CHECK(std::abs(st.values[0] - nxr::stats::median(means)) < 1e-12 && std::abs(st.values[1] - nxr::stats::mean(means)) < 1e-12 &&
          std::abs(st.values[2] - nxr::stats::percentile(means, 75)) < 1e-12 && std::abs(st.values[3] - nxr::stats::sd(means)) < 1e-12);
    CHECK(statistic_compute(ds, sub, so).hit);
    StatisticComputeOptions bad = so; bad.measure = "sd"; bad.name = "x";
    bool derived = false;
    try { statistic_compute(ds, sub, bad); } catch (const Error&) { derived = true; }
    CHECK(derived);                                                    // a derived measure is not stored: refused
  }
  // A MAP STATISTIC (docs/specs/2026-11-13-statistics.md; the maintainers: statistics read as maps): axes vertex x band; each band measured as a Field would be, its
  // rows naming the statistic and the band (of_statistic_id, of_index); a re-run a hit
  std::string stid;
  {
    std::vector<float> stv(static_cast<std::size_t>(nV) * 2);
    for (int i = 0; i < nV; ++i) { stv[static_cast<std::size_t>(i) * 2] = xs[i]; stv[static_cast<std::size_t>(i) * 2 + 1] = 2.0f; }
    sub.compose([&] {
      StatisticSpec st;
      st.name = "stat"; st.measure = "power_density"; st.method = "welch"; st.of_field_id = xid; st.unit = "a.u.";
      st.axes = Json::array({Json{{"name", "vertex"}, {"manifold_id", mid}, {"size", nV}},
                             Json{{"name", "band"}, {"size", 2}, {"values", Json::array({Json{{"name", "lo"}}, Json{{"name", "hi"}}})}}});
      st.data = view(stv, {nV, 2});
      stid = sub.statistic(std::move(st));
    });
    MeasureComputeOptions so = o; so.fields = "stat_.*"; so.replace = false; so.measures = {"mean"};
    const auto rs = measure_compute(ds, sub, so);
    CHECK(rs.info["fields"].size() == 2 && rs.info["fields"][0]["name"] == "stat_lo" && rs.info["fields"][0]["statistic"] == "stat");
    const Json lo = query_measurements(sub.db(), Json{{"selectionId", sid}, {"measure", "mean"}, {"ofStatisticId", stid}, {"ofIndex", 0}});
    const Json hi = query_measurements(sub.db(), Json{{"selectionId", sid}, {"measure", "mean"}, {"ofStatisticId", stid}, {"ofIndex", 1}});
    double w2 = 0;
    for (std::size_t k = 0; k < lo.size(); ++k) w2 = std::max(w2, std::abs(lo[k]["value"].get<double>() - value(xid, static_cast<int>(lo[k]["code"].get<std::int64_t>()), "mean")));
    CHECK(lo.size() == 7 && hi.size() == 7 && w2 < 1e-6 && std::abs(hi[0]["value"].get<double>() - 2.0) < 1e-9);   // the band "lo" is map_x (stored float32)
    CHECK(query_measurements(sub.db(), Json{{"selectionId", sid}, {"measure", "mean"}, {"ofFieldId", xid}}).size() == 7);   // the Field's own rows, apart
    CHECK(measure_compute(ds, sub, so).info["fields"][1]["cache"] == "hit");
  }

  // refusals: an unstated weight, a partition that does not exist, a pattern that matches nothing
  auto throws = [&](MeasureComputeOptions bad) { try { measure_compute(ds, sub, bad); } catch (const Error&) { return true; } return false; };
  MeasureComputeOptions bad = o; bad.weight = "";
  CHECK(throws(bad));
  bad = o; bad.labelling = "nope";
  CHECK(throws(bad));
  bad = o; bad.fields = "nothing";
  CHECK(throws(bad));

  // THE EXPORT: two subjects; T's dictionary names the members in another order (codes differ) and lacks member "g"; S is measured with voronoi
  // above, T now — first with barycentric (refused: one column, one method), then alike
  std::vector<std::int32_t> tcodes(static_cast<size_t>(nV));
  for (int i = 0; i < nV; ++i) tcodes[i] = codes[i] < 0 ? -1 : codes[i] == 6 ? 5 : 5 - codes[i];   // a→5 … f→0; g's vertices go to "a"
  std::string tx;
  {
    SubjectWriter t = SubjectWriter::create(ds, "T");
    t.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      const std::string tm = add_surface(t, so, b.P, b.F).id;
      DictionarySpec dsp; dsp.name = "parcels"; dsp.names = {"f", "e", "d", "c", "b", "a"};
      const DictionaryResult dict = t.dictionary(dsp);
      SelectionSpec sel; sel.name = "parcels"; sel.path = "source/parcels"; sel.type = "set"; sel.manifold_id = tm; sel.data = view(tcodes, {nV});
      sel.cols.set("dictionary_id", dict.id).set("cell", "0");
      t.selection(sel);
      FieldSpec f; f.name = "map_x"; f.path = "source/map_x"; f.kind = "map"; f.manifold_id = tm; f.data = view(xs, {nV}); f.cols.set("unit", "a.u.");
      f.cols.json("producer_json", nxr::io::Json{{"spectral_scaling", nxr::io::Json{{"stored", "density"}, {"degree", 1}, {"band_width_hz", 4.0}, {"bin_enbw_hz", 0.375}, {"power_unit", "a.u.\xc2\xb2"}}}});
      tx = t.field(std::move(f));
    });
  }
  {
    SubjectWriter t = SubjectWriter::open(ds, "T");
    MeasureComputeOptions to; to.labelling = "parcels"; to.fields = "map_x"; to.weight = "barycentric";
    measure_compute(ds, t, to);
  }
  const fs::path out = root / "tables";
  MeasureExportOptions eo; eo.labelling = "parcels"; eo.fields = "map_(x)"; eo.table = "maps"; eo.out = out.string(); eo.statistics = {"mean", "sd", "area"}; eo.layout = "wide";
  bool mixed = false;
  try { measure_export(root / "demo", eo); } catch (const Error&) { mixed = true; }
  CHECK(mixed);                                                      // S voronoi, T barycentric: refused
  {
    SubjectWriter t = SubjectWriter::open(ds, "T");
    MeasureComputeOptions to; to.labelling = "parcels"; to.fields = "map_x"; to.weight = "voronoi"; to.replace = true;
    measure_compute(ds, t, to);
  }
  const Json ex = measure_export(root / "demo", eo);
  auto lines = [](const fs::path& f) { std::vector<std::string> v; std::ifstream in(f); for (std::string l; std::getline(in, l);) v.push_back(l); return v; };
  auto cells = [](const std::string& l) { std::vector<std::string> v; std::stringstream ss(l); for (std::string c; std::getline(ss, c, ',');) v.push_back(c); if (!l.empty() && l.back() == ',') v.emplace_back(); return v; };
  const auto tab = lines(out / "maps.csv");
  CHECK_EQ(static_cast<int>(tab.size()), 3);                         // the header, S, T
  const auto head = cells(tab[0]), srow = cells(tab[1]), trow = cells(tab[2]);
  auto col = [&](const std::string& name) { for (size_t i = 0; i < head.size(); ++i) if (head[i] == name) return static_cast<int>(i); return -1; };
  const int ca = col("x_a_mean"), cg = col("x_g_mean"), cs = col("x_c_sd"), car = col("a_area");
  CHECK(ca > 0 && cg > 0 && cs > 0 && car > 0 && head[0] == "subject" && srow[0] == "S" && trow[0] == "T");
  if (ca > 0 && cg > 0 && cs > 0 && car > 0) {
    // member "a" by NAME: S's code 0 and T's code 5 hold different vertices (T's "a" also has g's), so the two means are each its own subject's
    CHECK(std::abs(std::stod(srow[ca]) - expected(voro, 0)) < 1e-12 * std::abs(expected(voro, 0)));
    std::vector<double> tw = voro; double sw = 0, sx = 0;
    for (int i = 0; i < nV; ++i) if (tcodes[i] == 5) { sw += tw[i]; sx += tw[i] * static_cast<double>(xs[i]); }
    CHECK(std::abs(std::stod(trow[ca]) - sx / sw) < 1e-12 * std::abs(sx / sw));
    CHECK(trow[cg].empty() && !srow[cg].empty());                    // T has no member "g"
  }
  // THE MAPS OF A STATISTIC EXPORTED as a Field's are: the band is the condition (`stat_(.*)`), member "a"'s mean of band lo is S's lookup value
  {
    MeasureExportOptions es; es.labelling = "parcels"; es.fields = "stat_(.*)"; es.table = "stats"; es.out = out.string(); es.statistics = {"mean"}; es.layout = "wide";
    measure_export(root / "demo", es);
    const auto st = lines(out / "stats.csv");
    const auto sh = cells(st[0]);
    int cl = -1, chh = -1;
    for (size_t i = 0; i < sh.size(); ++i) { if (sh[i] == "lo_a_mean") cl = static_cast<int>(i); if (sh[i] == "hi_a_mean") chh = static_cast<int>(i); }
    CHECK(cl > 0 && chh > 0);
    if (cl > 0 && chh > 0) {
      const auto sr = cells(st[1]);
      const Json q = query_measurements(sub.db(), Json{{"selectionId", sid}, {"measure", "mean"}, {"ofStatisticId", stid}, {"ofIndex", 0}});
      CHECK(sr[0] == "S" && std::abs(std::stod(sr[cl]) - q[0]["value"].get<double>()) < 1e-9 * std::abs(q[0]["value"].get<double>()) && std::abs(std::stod(sr[chh]) - 2.0) < 1e-9);
    }
  }
  // THE SCALING CHOSEN AT READ: --scaling power multiplies a density's mean and sd by the band's width (4 Hz); the area is not a spectrum; a non-linear
  // target is refused for sd; a Field without a scaling record is refused any conversion
  {
    MeasureExportOptions ep = eo; ep.table = "maps_power"; ep.scaling = "power";
    measure_export(root / "demo", ep);
    const auto pt = lines(out / "maps_power.csv");
    const auto ph = cells(pt[0]), ps = cells(pt[1]);
    auto pc = [&](const std::string& name) { for (size_t i = 0; i < ph.size(); ++i) if (ph[i] == name) return static_cast<int>(i); return -1; };
    const int pa = pc("x_a_mean"), psd = pc("x_c_sd"), par = pc("a_area");
    CHECK(pa > 0 && psd > 0 && par > 0);
    if (pa > 0 && psd > 0 && par > 0) {
      CHECK(std::abs(std::stod(ps[pa]) - 4.0 * std::stod(srow[ca])) < 1e-9 * std::abs(std::stod(ps[pa])));
      CHECK(std::abs(std::stod(ps[psd]) - 4.0 * std::stod(srow[cs])) < 1e-9 * std::abs(std::stod(ps[psd])));
      CHECK(ps[par] == srow[car]);
    }
    bool nonlinear = false, norecord = false;
    { MeasureExportOptions e = eo; e.table = "maps_db"; e.scaling = "db"; try { measure_export(root / "demo", e); } catch (const Error&) { nonlinear = true; } }
    { MeasureExportOptions e = eo; e.table = "maps_two"; e.fields = "map_(two)"; e.statistics = {"mean"}; e.scaling = "density"; try { measure_export(root / "demo", e); } catch (const Error&) { norecord = true; } }
    CHECK(nonlinear && norecord);
  }
  {
    MeasureExportOptions ex2 = eo; ex2.table = "maps_no_b"; ex2.exclude = {"b"};
    measure_export(root / "demo", ex2);
    const auto h2 = cells(lines(out / "maps_no_b.csv")[0]);
    bool hasb = false; for (const auto& c : h2) if (c.find("_b_") != std::string::npos || c.rfind("b_", 0) == 0) hasb = true;
    CHECK(!hasb && h2.size() < head.size());
    CHECK(std::stod(srow[cs]) > 0);                                  // sd, derived from the stored rows
  }
  const auto dict = lines(out / "dictionary.csv");
  int rows_of_maps = 0; for (const auto& l : dict) if (l.find(",maps,") != std::string::npos) ++rows_of_maps;
  CHECK_EQ(rows_of_maps, static_cast<int>(head.size()));
  CHECK(dict[0].rfind("index,data_table,column_number,variable,description,units,possible_data_values,value_type,notes,labelling,member,condition,statistic,field_kind,recipe", 0) == 0);
  CHECK(dict[2].find("partition-voronoi@1") != std::string::npos);
  // a second table in the same folder: its rows are added, the first table's kept
  MeasureExportOptions eo2 = eo; eo2.table = "maps_sizes"; eo2.statistics = {"n_elements"};
  measure_export(root / "demo", eo2);
  const auto dict2 = lines(out / "dictionary.csv");
  int first = 0, second = 0;
  for (const auto& l : dict2) { if (l.find(",maps,") != std::string::npos) ++first; if (l.find(",maps_sizes,") != std::string::npos) ++second; }
  CHECK(first == static_cast<int>(head.size()) && second > 1);
  const Json prov = Json::parse(std::ifstream(out / "maps.json"));
  CHECK(prov["recipes"].contains("partition-voronoi@1") && prov["subjects"].size() == 2);
  // an ambiguous request: two Fields of one condition in a subject
  MeasureExportOptions amb = eo; amb.fields = "map_.*";              // no group: the condition is the kind, "map", for map_x and map_two
  bool ambiguous = false;
  try { measure_export(root / "demo", amb); } catch (const Error&) { ambiguous = true; }
  CHECK(ambiguous);
  // THE SPLIT LAYOUT (the default): a table a condition × statistic, a column a member; the members' own sizes in their own table; a re-export with
  // fewer statistics drops the tables (and dictionary rows) it no longer makes
  MeasureExportOptions sp = eo; sp.table = "split"; sp.layout = "split";
  const Json sx = measure_export(root / "demo", sp);
  CHECK_EQ(static_cast<int>(sx["tables"].size()), 3);                // split_x_mean, split_x_sd, split_area
  const auto smean = lines(out / "split_x_mean.csv"), sarea = lines(out / "split_area.csv");
  CHECK(smean.size() == 3 && sarea.size() == 3 && fs::exists(out / "split_x_sd.csv"));
  if (smean.size() == 3) {
    const auto h = cells(smean[0]), sr = cells(smean[1]);
    auto at = [&](const std::string& c) { for (size_t i = 0; i < h.size(); ++i) if (h[i] == c) return static_cast<int>(i); return -1; };
    CHECK(h[0] == "subject" && at("a_mean") > 0 && at("g_mean") > 0 && h.size() == 8);   // subject + 7 members
    if (at("a_mean") > 0 && ca > 0) CHECK(sr[at("a_mean")] == srow[ca]);                   // the same value as the wide table's
  }
  auto dict_rows = [&](const std::string& table) { int k = 0; for (const auto& l : lines(out / "dictionary.csv")) if (l.find("," + table + ",") != std::string::npos) ++k; return k; };
  CHECK(dict_rows("split_x_mean") == 8 && dict_rows("split_area") == 8 && dict_rows("maps") == static_cast<int>(head.size()));
  sp.statistics = {"mean"};
  measure_export(root / "demo", sp);
  CHECK(!fs::exists(out / "split_x_sd.csv") && !fs::exists(out / "split_area.csv") && fs::exists(out / "maps.csv"));
  CHECK(dict_rows("split_x_sd") == 0 && dict_rows("split_area") == 0 && dict_rows("split_x_mean") == 8 && dict_rows("maps") == static_cast<int>(head.size()));
  std::printf("  export: %d columns, %d subjects; members joined by name\n", static_cast<int>(head.size()), ex["subjects"].get<int>());


  // VECTORS: the canonical frames, then (1) the GRADIENT of the map x = z — it points up, so in the canonical frames (north towards the sphere's +z)
  // its direction is near 0 degrees, with no normal component; (2) a TENSOR Field, constant z z^T (a current swinging along z): each member's normal
  // share, anisotropy and circularity from its tensor_* sums equal an independent computation from the frames; each family has its own recipe
  {
    SubjectWriter vs = SubjectWriter::open(ds, "S");
    FramesComputeOptions fo;
    frames_compute(ds, vs, fo);
    const StoredFrames fr = read_stored_frames(vs, "cortex", nV);
    CHECK(fr.canonical);
    std::vector<float> zf(static_cast<size_t>(nV)), herm(static_cast<size_t>(nV) * 18, 0.0f);
    for (int i = 0; i < nV; ++i) { zf[i] = static_cast<float>(b.P[3 * i + 2]); herm[static_cast<size_t>(18 * i + 2 * 8)] = 1.0f; }   // T = z z^T: re[2][2] = 1
    vs.compose([&] {
      FieldSpec f; f.name = "map_z"; f.path = "source/map_z"; f.kind = "map"; f.manifold_id = mid; f.data = view(zf, {nV}); f.cols.set("unit", "m");
      vs.field(std::move(f));
      FieldSpec t; t.name = "tens_z"; t.path = "source/tens_z"; t.kind = "cross-spectral tensor"; t.manifold_id = mid; t.value_type = "hermitian3"; t.data = view(herm, {nV, 3, 3, 2});
      t.cols.set("unit", "a.u.");
      vs.field(std::move(t));
    });
    MeasureComputeOptions g; g.labelling = "parcels"; g.fields = "map_z|tens_z"; g.weight = "barycentric"; g.gradient = true; g.replace = true;
    const auto rg = measure_compute(ds, vs, g);
    // the same tensor as a STATISTIC whose values are hermitian3 (one band, "z"): measured as the Field, the same tensor_* sums, its rows naming the slice
    {
      std::string tsid;
      vs.compose([&] {
        StatisticSpec st;
        st.name = "tstat"; st.measure = "cross_spectral_density"; st.method = "welch"; st.value_type = "hermitian3"; st.unit = "a.u.";
        st.of_field_id = task::Subject::of(vs).require({.cls = task::Cls::field, .name = "tens_z"}, "the tensor field").id;
        st.axes = Json::array({Json{{"name", "vertex"}, {"manifold_id", mid}, {"size", nV}}, Json{{"name", "band"}, {"size", 1}, {"values", Json::array({Json{{"name", "z"}}})}}});
        st.data = view(herm, {nV, 1, 3, 3, 2});
        tsid = vs.statistic(std::move(st));
      });
      MeasureComputeOptions gt = g; gt.fields = "tstat_z"; gt.gradient = false;
      measure_compute(ds, vs, gt);
      const std::string tfid = task::Subject::of(vs).require({.cls = task::Cls::field, .name = "tens_z"}, "the tensor field").id;
      double tw = 0;
      for (const char* m : {"tensor_total", "tensor_nn", "tensor_11", "tensor_12_im"}) {
        const Json a = query_measurements(vs.db(), Json{{"selectionId", sid}, {"measure", m}, {"ofFieldId", tfid}});
        const Json c = query_measurements(vs.db(), Json{{"selectionId", sid}, {"measure", m}, {"ofStatisticId", tsid}, {"ofIndex", 0}});
        CHECK(a.size() == c.size() && !a.empty());
        for (std::size_t k = 0; k < std::min(a.size(), c.size()); ++k) tw = std::max(tw, std::abs(a[k]["value"].get<double>() - c[k]["value"].get<double>()));
      }
      CHECK(tw < 1e-9);
    }
    const Json recs = rg.info["recipes"];
    CHECK(recs.size() == 3 && recs[1].get<std::string>().rfind("partition-barycentric-gradient@", 0) == 0 && recs[2].get<std::string>().rfind("partition-barycentric-tensor@", 0) == 0);
    const fs::path vout = root / "vtables";
    MeasureExportOptions ve; ve.labelling = "parcels"; ve.out = vout.string(); ve.table = "v";
    ve.fields = "map_(z)"; ve.statistics = {"gradient_direction_deg", "gradient_strength", "gradient_coherence", "gradient_n"};
    measure_export(root / "demo", ve);
    ve.fields = "tens_(z)"; ve.table = "t"; ve.statistics = {"normal_share", "anisotropy", "circularity"};
    measure_export(root / "demo", ve);
    auto row1 = [&](const std::string& table) { const auto l = lines(vout / (table + ".csv")); return std::make_pair(cells(l[0]), cells(l[1])); };
    // (1) the gradient
    const auto [hd, vd] = row1("v_z_gradient_direction_deg");
    const auto [hn, vn] = row1("v_z_gradient_n");
    const auto [hc, vc] = row1("v_z_gradient_coherence");
    double worst_dir = 0, worst_n = 0, worst_c = 0;
    for (size_t i = 1; i < hd.size(); ++i) worst_dir = std::max(worst_dir, std::abs(std::stod(vd[i])));
    for (size_t i = 1; i < hn.size(); ++i) worst_n = std::max(worst_n, std::abs(std::stod(vn[i])));
    for (size_t i = 1; i < hc.size(); ++i) worst_c = std::max(worst_c, std::stod(vc[i]));
    std::printf("  gradient of z: |direction| <= %.1f deg from north; |gradient_n| <= %.1e; coherence <= %.3f\n", worst_dir, worst_n, worst_c);
    CHECK(worst_dir < 30.0 && worst_n < 1e-9 && worst_c <= 1.0 + 1e-12);
    // (2) the tensor: the independent computation, member by member (barycentric weights over the frames' regular vertices)
    std::vector<double> s11(7, 0), s22(7, 0), snn(7, 0), s12(7, 0);
    for (int i = 0; i < nV; ++i) {
      const int k = codes[i];
      if (k < 0 || !std::isfinite(fr.e1(i, 0)) || !std::isfinite(fr.e2(i, 0))) continue;
      const double z1 = fr.e1(i, 2), z2 = fr.e2(i, 2);
      const double nz = fr.e1(i, 0) * fr.e2(i, 1) - fr.e1(i, 1) * fr.e2(i, 0);           // (e1 x e2)_z
      s11[k] += bary[i] * z1 * z1; s22[k] += bary[i] * z2 * z2; snn[k] += bary[i] * nz * nz; s12[k] += bary[i] * z1 * z2;
    }
    const auto [hs, vsh] = row1("t_z_normal_share");
    const auto [ha, va] = row1("t_z_anisotropy");
    const auto [hci, vci] = row1("t_z_circularity");
    double worst = 0, circ = 0;
    const char* names[7] = {"a", "b", "c", "d", "e", "f", "g"};
    for (int k = 0; k < 7; ++k) {
      auto at = [&](const std::vector<std::string>& h, const std::string& c) { for (size_t i = 0; i < h.size(); ++i) if (h[i] == c) return static_cast<int>(i); return -1; };
      const int is = at(hs, std::string(names[k]) + "_normal_share"), ia = at(ha, std::string(names[k]) + "_anisotropy"), ic = at(hci, std::string(names[k]) + "_circularity");
      if (is < 0 || ia < 0 || ic < 0) { CHECK(false); continue; }
      const double share = snn[k] / (s11[k] + s22[k] + snn[k]), aniso = std::hypot(s11[k] - s22[k], 2 * s12[k]) / (s11[k] + s22[k]);
      worst = std::max({worst, std::abs(std::stod(vsh[is]) - share), std::abs(std::stod(va[ia]) - aniso)});
      circ = std::max(circ, std::abs(std::stod(vci[ic])));
    }
    std::printf("  tensor z z^T: normal share and anisotropy vs the independent computation: %.2e; |circularity| <= %.1e\n", worst, circ);
    CHECK(worst < 1e-6 && circ < 1e-9);
  }
  fs::remove_all(root);
  return nxrtest::finish("test_measure_pipeline");
}
