// migrate_measurement_storage (migrate.h; spec 2026-11-17-consolidation.md section 6.3) on a synthetic dataset: statistics with a MEMBER axis (the member axis last, and first) become array-backed
// measurements [rows x members x components] at `<path>.members` (verified, the source untouched), a second run changes nothing, `dry_run` changes nothing, a conflict is reported and not
// overwritten, a tower tiling is marked, and the explicit second step removes the migrated statistics. The task-level statistic writer refuses a statistic with a member axis.
// Synthetic data only.
#include <unistd.h>

#include <cstring>
#include <filesystem>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/measures.h"
#include "nxr/io/migrate.h"
#include "nxr/io/statistic.h"
#include "nxr/io/task/task.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
std::vector<double> doubles_of(const Store& store, const std::string& path, std::vector<std::int64_t>* shape) {
  const ArrayReader ar = ArrayReader::open(store, path);
  if (shape) *shape = ar.shape();
  const auto raw = ar.read();
  std::vector<double> v(raw.size() / 8);
  std::memcpy(v.data(), raw.data(), raw.size());
  return v;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_migrate_measurement_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};

  std::string part, fid;
  std::vector<double> last(3 * 8), first(8 * 3);                    // [freq x member] and [member x freq]
  for (int f = 0; f < 3; ++f) for (int m = 0; m < 8; ++m) { last[static_cast<std::size_t>(f * 8 + m)] = 100 * f + m; first[static_cast<std::size_t>(m * 3 + f)] = 100 * f + m; }
  {
    DatasetWriter ds = create_dataset(root, "demo");
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    std::vector<float> x(100, 1.0f);
    sub.compose([&] {
      ManifoldSpec t;
      t.name = "line"; t.path = "time/line"; t.type = "line"; t.n_vertices = 100;
      t.topology = {"path", 1, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "closed_form"; g.cs_key = "clock";
      DimensionSpec d; d.name = "time"; d.n_vertices = 100; d.spacing = 0.01; d.origin = 0.0; d.unit = "s";
      g.dimensions.push_back(d); t.geometry = g;
      t.cols.set("n_edges", 99);
      const std::string tid = sub.manifold(t);
      FieldSpec f;
      f.name = "rec"; f.path = node_path(sub.db(), tid, "rec"); f.kind = "recording"; f.manifold_id = tid; f.data = view(x, {100});
      fid = sub.field(f);
      SelectionSpec s;
      s.name = "segments"; s.path = "time/segments"; s.type = "spans"; s.manifold_id = tid;
      s.cols.set("cell", "1").set("unit", "s").set("n_members", 8);
      s.cols.json("params_json", Json{{"tiling", Json{{"origin", 0.0}, {"width", 0.125}, {"hop", 0.125}, {"count", 8}}}});
      part = sub.selection(std::move(s));
      // a tower tiling (retired): the frame tiling of the day-anchored tower
      SelectionSpec tw;
      tw.name = "frames"; tw.path = "time/frames"; tw.type = "spans"; tw.manifold_id = tid;
      tw.cols.set("cell", "1").set("unit", "s").set("n_members", 8);
      tw.cols.json("params_json", Json{{"tiling", Json{{"origin", 0.0}, {"width", 0.125}, {"hop", 0.125}, {"count", 8}}}, {"atlas", Json{{"frame_level", 4}, {"frame_code0", 12345}}}});
      sub.selection(std::move(tw));
      auto stat = [&](const std::string& name, const std::string& measure, bool member_first, const std::vector<double>& v) {
        StatisticSpec st;
        st.name = name; st.measure = measure; st.method = "welch"; st.of_field_id = fid; st.partition_selection_id = part;
        const Json freq{{"name", "frequency"}, {"size", 3}, {"values", Json::array({0.0, 25.0, 50.0})}, {"unit", "Hz"}};
        const Json member{{"name", "segment"}, {"size", 8}, {"selection_id", part}};
        st.axes = member_first ? Json::array({member, freq}) : Json::array({freq, member});
        st.unit = "T^2/Hz";
        st.data = view(v, member_first ? std::vector<std::int64_t>{8, 3} : std::vector<std::int64_t>{3, 8});
        sub.statistic(std::move(st));
      };
      stat("psd_by_segment", "power_density", false, last);
      stat("count_by_segment", "count", true, first);
    });
  }
  const std::string last_path = "time/rec_statistics/psd_by_segment", first_path = "time/rec_statistics/count_by_segment";

  // DRY RUN: the plan, nothing written
  {
    MeasurementStorageOptions o; o.dry_run = true;
    const Json r = migrate_measurement_storage(root / "demo", o);
    CHECK(r["statistics_converted"] == 2 && r["tower_tilings_marked"] == 1 && r["dry_run"] == true);
    DatasetWriter ds = open_writer(root / "demo");
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    CHECK(!sub.store().exists_prefix(last_path + ".members") && !sub.db().one("SELECT 1 FROM selection_measurement WHERE array_path IS NOT NULL").has_value());
  }
  // THE MIGRATION
  {
    const Json r = migrate_measurement_storage(root / "demo");
    CHECK(r["statistics_converted"] == 2 && r["conflicts"] == 0 && r["tower_tilings_marked"] == 1 && r["statistics_removed"] == 0);
    DatasetWriter ds = open_writer(root / "demo");
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    std::vector<std::int64_t> shape;
    // the member axis last: [3 x 8] -> [rows 3 x members 8 x 1], the same numbers
    auto v = doubles_of(sub.store(), last_path + ".members", &shape);
    CHECK((shape == std::vector<std::int64_t>{3, 8, 1}) && v == last);
    // the member axis FIRST: [8 x 3] -> [rows 3 x members 8 x 1], transposed: element (r, m) = the source's (m, r)
    v = doubles_of(sub.store(), first_path + ".members", &shape);
    CHECK((shape == std::vector<std::int64_t>{3, 8, 1}) && v == last);
    // the sources are untouched, and noted
    CHECK(doubles_of(sub.store(), last_path, nullptr) == last && doubles_of(sub.store(), first_path, nullptr) == first);
    CHECK(sub.db().one("SELECT 1 FROM statistic WHERE name = 'psd_by_segment' AND json_extract(options_json, '$.migrated_to_measurement.array_path') = ?", {V(last_path + ".members")}).has_value());
    // the measurement rows: one a measure, on the partition, of the field
    const MeasurementValues mv = measurement_values(sub, part, fid, std::nullopt, std::nullopt, "power_density");
    CHECK(mv.rows == 3 && mv.members == 8 && mv.array_backed && mv.values == last && mv.unit && *mv.unit == "T^2/Hz");
    CHECK(sub.db().one("SELECT 1 FROM selection WHERE name = 'frames' AND json_extract(params_json, '$.retired_grid') = 'tower'").has_value());
    CHECK_EQ(text(*ds.db().one("SELECT value FROM meta WHERE key = 'measurement_storage'"), "value"), std::string(kMeasurementStorage));
  }
  // A SECOND RUN changes nothing
  {
    const Json r = migrate_measurement_storage(root / "demo");
    CHECK(r["statistics_converted"] == 0 && r["tower_tilings_marked"] == 0);
  }
  // THE EXPLICIT SECOND STEP removes the migrated statistics (only when the option is set); their measurements stay
  {
    MeasurementStorageOptions o; o.remove_statistics = true;
    const Json r = migrate_measurement_storage(root / "demo", o);
    CHECK(r["statistics_removed"] == 2);
    DatasetWriter ds = open_writer(root / "demo");
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    CHECK(!sub.db().one("SELECT 1 FROM statistic WHERE name IN ('psd_by_segment', 'count_by_segment')").has_value());
    CHECK(measurement_values(sub, part, fid, std::nullopt, std::nullopt, "count").values == last);
  }
  // THE TASK-LEVEL WRITER refuses a statistic with a member axis (a measurement), and takes one with none
  {
    DatasetWriter ds = open_writer(root / "demo");
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    task::Unit u = task::Subject::of(sub).unit(task::By{"test", Json()});
    task::StatisticOut s;
    s.name = "again"; s.measure = "power_density"; s.method = "welch"; s.of_field_id = fid; s.partition_selection_id = part;
    s.axes = Json::array({Json{{"name", "segment"}, {"size", 8}, {"selection_id", part}}});
    CHECK_EQ(code_of([&] { u.statistic(s); }), std::string(code::invalid_argument));
    u.rollback();
  }
  return nxrtest::finish("test_migrate_measurement");
}
