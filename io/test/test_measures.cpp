// The measure registry: the Studio's entries followed by nxr-neuro's own (schema/measures_nxr.json, codes 1000-1999), with no code or name in both; the
// dataset's `measure` dictionary carries both; and the phase family's derived measures computed at read from the stored sums (the lookup), on members
// with known answers (a plane wave: travelling index 1; a standing pattern: 0; no power: no phase gradient) and on a roll-up (ratios of the rolled-up sums,
// never sums of ratios).
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <set>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/measures.h"

using namespace nxr::io;
namespace fs = std::filesystem;

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_measures_" + std::to_string(::getpid()));
  fs::remove_all(root);

  // THE REGISTRY: codes and names unique; nxr-neuro's in its range, after the Studio's
  const Json& reg = measure_registry();
  std::set<std::int64_t> codes;
  std::set<std::string> names;
  int own = 0;
  for (const auto& m : reg["measures"]) {
    CHECK(codes.insert(m["code"].get<std::int64_t>()).second);
    CHECK(names.insert(m["name"].get<std::string>()).second);
    if (m["code"].get<std::int64_t>() >= 1000) ++own;
  }
  CHECK(own == 112 && measure_reduction("phase_current_x") == "mean" && measure_is_derived("phase_gradient_z") && !measure_is_derived("phase_current_y") && measure_reduction("scale_peak") == "none" && !measure_is_derived("scale_response") && measure_reduction("scale_event") == "none" && measure_reduction("n_cells") == "sum" && measure_reduction("path_length") == "none" && measure_reduction("burst_ordinal") == "none" && measure_reduction("t_on") == "none" && measure_reduction("n_tiles") == "sum" && measure_reduction("centre_vertex") == "none" && measure_reduction("median") == "none" && measure_reduction("duration") == "sum" && measure_reduction("interval_next") == "none" && measure_is_derived("travelling_index") && measure_is_derived("shape_factor") && measure_is_derived("kurtosis") && !measure_is_derived("mean_fourth") && !measure_is_derived("mean_abs") && !measure_is_derived("phase_power") && measure_reduction("phase_current_1") == "sum" && measure_is_derived("quadrature_fraction") && measure_reduction("cross_spectrum_norm2") == "sum" && !measure_is_derived("cross_spectrum_quadrature_norm2") && measure_reduction("t_on") == "none" && measure_reduction("n_tiles") == "sum" && measure_reduction("centre_vertex") == "none" && measure_reduction("median") == "none" && measure_reduction("duration") == "sum" && measure_reduction("interval_next") == "none" && measure_is_derived("travelling_index") && measure_is_derived("shape_factor") && measure_is_derived("kurtosis") && !measure_is_derived("mean_fourth") && !measure_is_derived("mean_abs") && !measure_is_derived("phase_power") && measure_reduction("phase_current_1") == "sum" && measure_reduction("n_cells") == "sum" && measure_reduction("path_length") == "none" && measure_reduction("burst_ordinal") == "none" && measure_reduction("t_on") == "none" && measure_reduction("n_tiles") == "sum" && measure_reduction("centre_vertex") == "none" && measure_reduction("median") == "none" && measure_reduction("duration") == "sum" && measure_reduction("interval_next") == "none" && measure_is_derived("travelling_index") && measure_is_derived("shape_factor") && measure_is_derived("kurtosis") && !measure_is_derived("mean_fourth") && !measure_is_derived("mean_abs") && !measure_is_derived("phase_power") && measure_reduction("phase_current_1") == "sum" && measure_is_derived("quadrature_fraction") && measure_reduction("cross_spectrum_norm2") == "sum" && !measure_is_derived("cross_spectrum_quadrature_norm2") && measure_reduction("t_on") == "none" && measure_reduction("n_tiles") == "sum" && measure_reduction("centre_vertex") == "none" && measure_reduction("median") == "none" && measure_reduction("duration") == "sum" && measure_reduction("interval_next") == "none" && measure_is_derived("travelling_index") && measure_is_derived("shape_factor") && measure_is_derived("kurtosis") && !measure_is_derived("mean_fourth") && !measure_is_derived("mean_abs") && !measure_is_derived("phase_power") && measure_reduction("phase_current_1") == "sum");
  CHECK(measure_spec("sd").has_value() && measure_spec("phase_gradient_magnitude").has_value());

  DatasetWriter ds = create_dataset(root, "demo");
  std::string sid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      ManifoldSpec t;
      t.name = "line"; t.path = "time/line"; t.type = "line"; t.n_vertices = 100;
      t.topology = {"path", 1, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "closed_form"; g.cs_key = "clock";
      DimensionSpec d; d.name = "time"; d.n_vertices = 100; d.spacing = 0.01; d.origin = 0.0; d.unit = "s";
      g.dimensions.push_back(d); t.geometry = g;
      t.cols.set("n_edges", 99);
      const std::string tid = sub.manifold(t);
      SelectionSpec s;
      s.name = "windows"; s.type = "spans"; s.manifold_id = tid;
      s.cols.set("cell", "1").set("n_elements", static_cast<std::int64_t>(100)).set("n_members", static_cast<std::int64_t>(8));
      s.cols.json("params_json", Json{{"tiling", Json{{"origin", 0.0}, {"width", 0.125}, {"hop", 0.125}, {"count", 8}}}});
      sid = sub.selection(std::move(s));
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  // THE DICTIONARY carries both lists
  const std::string did = ensure_measure_dictionary(sub.db(), std::string("test"));
  CHECK(sub.db().one("SELECT code FROM dictionary_entry WHERE dictionary_id = ? AND name = 'travelling_index' AND code = 1104", {V(did)}).has_value());
  CHECK(sub.db().one("SELECT code FROM dictionary_entry WHERE dictionary_id = ? AND name = 'sd'", {V(did)}).has_value());

  // THE STORED SUMS: {code: {pc1, pc2, pcm, pp, pgp}}
  //   0: a plane wave, j = k|z|², |∇z|² = |k|²|z|² with k = (1.5, 2): gradient (1.5, 2), length 2.5, 53.13°, travelling index 1, coherence 1
  //   1: a standing pattern (no current): length 0, travelling index 0, no direction, no coherence
  //   3: no power: no phase gradient
  //   4, 5: two opposite waves, rolled up into 2 (shift 1): their currents cancel
  const std::map<std::int64_t, std::array<double, 5>> rows = {{0, {3, 4, 5, 2, 12.5}}, {1, {0, 0, 0, 1, 4}}, {3, {0, 0, 0, 0, 0}}, {4, {1, 0, 1, 1, 1}}, {5, {-1, 0, 1, 1, 1}}};
  const char* comp[5] = {"phase_current_1", "phase_current_2", "phase_current_magnitude", "phase_power", "phase_gradient_power"};
  Json w = Json::array();
  for (const auto& [code, v] : rows)
    for (int c = 0; c < 5; ++c) w.push_back(Json{{"selection_id", sid}, {"code", code}, {"measure", comp[c]}, {"value", v[static_cast<std::size_t>(c)]}, {"weight", 1.0}});
  write_measurements(sub.db(), w, std::string("test"));

  auto get = [&](const std::string& m, std::int64_t shift = 0) {
    std::map<std::int64_t, double> out;
    for (const auto& r : query_measurements(sub.db(), Json{{"selectionId", sid}, {"measure", m}, {"shift", shift}})) out[r["code"].get<std::int64_t>()] = r["value"].get<double>();
    return out;
  };
  auto near = [](double a, double b) { return std::abs(a - b) < 1e-12; };
  const auto g1 = get("phase_gradient_1"), g2 = get("phase_gradient_2"), gm = get("phase_gradient_magnitude"), dir = get("phase_gradient_direction_deg"),
             ti = get("travelling_index"), coh = get("phase_coherence");
  CHECK(near(g1.at(0), 1.5) && near(g2.at(0), 2.0) && near(gm.at(0), 2.5) && near(dir.at(0), std::atan2(4.0, 3.0) * 180.0 / M_PI) && near(ti.at(0), 1.0) && near(coh.at(0), 1.0));
  CHECK(near(gm.at(1), 0.0) && near(ti.at(1), 0.0) && !dir.count(1) && !coh.count(1));
  CHECK(!g1.count(3) && !gm.count(3) && !ti.count(3));
  CHECK(near(dir.at(5), 180.0) && near(ti.at(4), 1.0));
  // THE ROLL-UP: the sums rolled up first, then the ratio (members 4 and 5 into 2: the currents cancel)
  const auto ti1 = get("travelling_index", 1), coh1 = get("phase_coherence", 1), gm1 = get("phase_gradient_magnitude", 1);
  CHECK(near(ti1.at(2), 0.0) && near(coh1.at(2), 0.0) && near(gm1.at(2), 0.0) && near(ti1.at(0), 5.0 / std::sqrt(3.0 * 16.5)));   // 0 and 1 into 0: 0.711, not the mean of their indices (0.5)
  // THE MEDIAN does not roll up (merge rule none): shift 0 gives each member's own value, a shift that groups members is refused
  {
    Json mw = Json::array();
    for (const auto& [code, val] : std::map<std::int64_t, double>{{0, 2.5}, {1, 7.0}, {4, -1.0}})
      mw.push_back(Json{{"selection_id", sid}, {"code", code}, {"measure", "median"}, {"value", val}, {"weight", 1.0}});
    write_measurements(sub.db(), mw, std::string("test"));
    const auto med = get("median");
    CHECK(med.size() == 3 && near(med.at(0), 2.5) && near(med.at(1), 7.0) && near(med.at(4), -1.0));
    bool refused = false;
    try { get("median", 1); } catch (const Error& e) { refused = e.code() == code::unsupported; }
    CHECK(refused);
  }
  // the kinds list offers the derived measures where their ingredients are stored
  std::set<std::string> kinds;
  for (const auto& k : measurement_kinds(sub.db(), sid)) kinds.insert(k["measure"].get<std::string>());
  CHECK(kinds.count("travelling_index") && kinds.count("phase_gradient_direction_deg") && kinds.count("phase_coherence") && !kinds.count("sd"));

  ds.close();
  fs::remove_all(root);
  return nxrtest::finish("test_measures");
}
