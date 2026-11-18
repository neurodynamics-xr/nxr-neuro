// PARITY of the two roll-ups (roll-ups exist in io's query AND in compute, ONE definition of each measure's merge rule, as data in the measures registry).
// io's SQL roll-up (nxr::io::query::members: member number `code >> shift`, the whole, across subjects) and compute's in-memory roll-up (nxr::measurement::roll_values over the groups of
// coarser_groups-style member sets) are run on the SAME synthetic members, with a registry built from io's table (measure_merge_table / measure_registry: what pipelines hand compute), and must give the
// same numbers for every stored merge rule (sum, max, min, a weighted mean), at every level of the hierarchy, to the whole, and across subjects. A median (merge none) is refused by both.
// Synthetic only: a time line with a tiling of 8 tiles, two subjects, unequal weights (so a mean that forgot its weights would be caught).
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>

#include <Eigen/Dense>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/measurement_query.h"
#include "nxr/io/task/task.h"
#include "nxr/measurement/measures.h"
#include "nxr/measurement/roll_up.h"

using namespace nxr::io;
namespace nm = nxr::measurement;
namespace fs = std::filesystem;

namespace {
template <class F> bool refused(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }
bool near(double a, double b) { return std::abs(a - b) <= 1e-12 * (1.0 + std::abs(b)); }

// the registry compute reads: io's table, row for row (the merge rule and the weight measure are DATA there; `derived` merges nothing)
nm::Registry registry_from_io() {
  std::vector<nm::MeasureSpec> specs;
  for (const auto& m : measure_registry()["measures"]) {
    nm::MeasureSpec s;
    s.name = m["name"].get<std::string>();
    s.form = m.value("form", std::string("none"));
    const std::string merge = m.value("merge", std::string("none"));
    s.merge = nm::merge_from(merge == "derived" ? "none" : merge);
    if (m.contains("weight") && m["weight"].is_string()) s.weight = m["weight"].get<std::string>();
    if (m.contains("derived_from") && m["derived_from"].is_array()) for (const auto& d : m["derived_from"]) s.derived_from.push_back(d.get<std::string>());
    specs.push_back(std::move(s));
  }
  return nm::Registry(std::move(specs));
}

// a subject with a time line of 100 samples and a tiling of 8 tiles; returns the tiling's selection id
std::string make_subject(DatasetWriter& ds, const std::string& name) {
  SubjectWriter sub = SubjectWriter::create(ds, name);
  std::string sid;
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
    s.name = "tiles"; s.type = "spans"; s.manifold_id = tid;
    s.cols.set("cell", "1").set("unit", "s").set("n_elements", static_cast<std::int64_t>(100)).set("n_members", static_cast<std::int64_t>(8));
    s.cols.json("params_json", Json{{"tiling", Json{{"origin", 0.0}, {"width", 0.125}, {"hop", 0.125}, {"count", 8}}}});
    sid = sub.selection(std::move(s));
  });
  return sid;
}

// compute's groups for a roll-up by `shift` levels of 8 members: coarse member j holds the fine members m with m >> shift == j; `copies` stacked subjects (across: group m holds m of every copy)
nxr::selection::Elements groups_of(std::int64_t members, int shift, bool whole) {
  nxr::selection::Elements g;
  g.size = members;
  const std::int64_t n = whole ? 1 : ((members - 1) >> shift) + 1;
  g.offsets.assign(1, 0);
  for (std::int64_t j = 0; j < n; ++j) {
    for (std::int64_t m = 0; m < members; ++m) if (whole || (m >> shift) == j) g.elements.push_back(m);
    g.offsets.push_back(static_cast<std::int64_t>(g.elements.size()));
  }
  return g;
}
nxr::selection::Elements across_groups(std::int64_t members, int subjects) {
  nxr::selection::Elements g;
  g.size = members * subjects;
  g.offsets.assign(1, 0);
  for (std::int64_t m = 0; m < members; ++m) {
    for (int s = 0; s < subjects; ++s) g.elements.push_back(s * members + m);
    g.offsets.push_back(static_cast<std::int64_t>(g.elements.size()));
  }
  return g;
}
Eigen::MatrixXd row(const std::vector<double>& v) {
  Eigen::MatrixXd m(1, static_cast<Eigen::Index>(v.size()));
  for (std::size_t i = 0; i < v.size(); ++i) m(0, static_cast<Eigen::Index>(i)) = v[i];
  return m;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_rollup_parity_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};

  DatasetWriter ds = create_dataset(root, "demo");
  const std::string s1 = make_subject(ds, "S1"), s2 = make_subject(ds, "S2");
  const std::vector<double> a = {3, 1, 4, 1, 5, 9, 2, 6}, b = {2, 7, 1, 8, 2, 8, 1, 8};
  const std::vector<double> wa = {1, 2, 0.5, 3, 1, 1, 4, 2}, wb = {2, 1, 1, 0.25, 3, 2, 1, 1};
  const std::vector<std::string> measures = {"sum", "max", "min", "mean", "mean_square", "abs_max"};
  auto put = [&](const std::string& subject, const std::string& sel, const std::vector<double>& v, const std::vector<double>& w, const std::string& measure) {
    SubjectWriter sub = SubjectWriter::open(ds, subject);
    const task::Subject ts = task::Subject::of(sub);
    Json rows = Json::array();
    for (std::size_t i = 0; i < v.size(); ++i) rows.push_back(Json{{"selection_id", sel}, {"code", static_cast<std::int64_t>(i)}, {"measure", measure}, {"value", v[i]}, {"weight", w[i]}});
    ts.write_member_rows(rows, "test");
  };
  for (const std::string& m : measures) { put("S1", s1, a, wa, m); put("S2", s2, b, wb, m); }
  put("S1", s1, a, wa, "median");

  const nm::Registry registry = registry_from_io();
  // the two sides read ONE rule: every stored measure's compute merge is io's `merge`
  for (const std::string& m : measures) CHECK(measure_merge(m) == MergeRule::sum || measure_merge(m) == MergeRule::max || measure_merge(m) == MergeRule::min || measure_merge(m) == MergeRule::mean);
  CHECK(std::string(nm::merge_name(registry.at("mean").merge)) == "mean" && std::string(nm::merge_name(registry.at("max").merge)) == "max" &&
        std::string(nm::merge_name(registry.at("sum").merge)) == "sum" && std::string(nm::merge_name(registry.at("min").merge)) == "min");

  SubjectWriter sub1 = SubjectWriter::open(ds, "S1");
  const task::Subject db = task::Subject::of(sub1);
  auto sql_rows = [&](const std::string& measure, int shift, bool whole) {
    query::Query q;
    q.measure = measure;
    q.filter.subjects = {"S1"};
    q.roll.shift = shift;
    q.roll.whole = whole;
    std::map<std::int64_t, double> out;
    for (const auto& r : db.query_members(q)) out[r["code"].get<std::int64_t>()] = r["value"].get<double>();
    return out;
  };
  const Eigen::MatrixXd va = row(a), wva = row(wa);
  // every merge rule, every level of the hierarchy (as stored, 1, 2 up, the whole)
  int compared = 0;
  for (const std::string& m : measures) {
    for (const int shift : {0, 1, 2, 3}) {
      const auto sql = sql_rows(m, shift, false);
      const Eigen::MatrixXd mem = nm::roll_values(registry, m, va, &wva, groups_of(8, shift, false));
      CHECK(static_cast<Eigen::Index>(sql.size()) == mem.cols());
      for (const auto& [code, v] : sql) { CHECK(near(v, mem(0, code))); ++compared; }
    }
    const auto sql = sql_rows(m, 0, true);
    const Eigen::MatrixXd mem = nm::roll_values(registry, m, va, &wva, groups_of(8, 0, true));
    CHECK(sql.size() == 1 && near(sql.at(0), mem(0, 0)));
    ++compared;
  }
  // ACROSS SUBJECTS: the same member of both, code for code (S1's members then S2's are the stacked values)
  for (const std::string& m : measures) {
    query::Query q;
    q.measure = m; q.filter.selections = {"tiles"}; q.roll.across_subjects = true;
    const Json sql = db.query_members(q);
    std::vector<double> both = a, wboth = wa;
    both.insert(both.end(), b.begin(), b.end());
    wboth.insert(wboth.end(), wb.begin(), wb.end());
    const Eigen::MatrixXd wm = row(wboth);
    const Eigen::MatrixXd mem = nm::roll_values(registry, m, row(both), &wm, across_groups(8, 2));
    CHECK(sql.size() == 8 && mem.cols() == 8);
    for (std::size_t i = 0; i < sql.size() && static_cast<Eigen::Index>(i) < mem.cols(); ++i) { CHECK(near(sql[i]["value"].get<double>(), mem(0, static_cast<Eigen::Index>(i)))); ++compared; }
  }
  // a median (merge none) is refused by both, when members would be merged
  {
    query::Query q;
    q.measure = "median"; q.filter.subjects = {"S1"}; q.roll.shift = 1;
    CHECK(refused([&] { db.query_members(q); }));
    CHECK(refused([&] { nm::roll_values(registry, "median", va, nullptr, groups_of(8, 1, false)); }));
  }
  std::printf("  io SQL roll-up and compute roll_values agree on %d values (sum, max, min, weighted mean; levels, the whole, across subjects)\n", compared);
  CHECK(compared == static_cast<int>(measures.size()) * (15 + 1 + 8));   // 8 + 4 + 2 + 1 members up the levels, the whole, 8 across
  return nxrtest::finish("test_rollup_parity");
}
