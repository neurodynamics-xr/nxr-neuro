// The measurement query API (measurement_query.h): filters and roll-ups in SQL over scalar member rows, merged by each measure's registry rule (measures.json `merge`, the one definition
// compute shares), and the same rule in memory (rollup.h) for arrays. Synthetic members with known answers: a grid of 8 tiles in two subjects.
//   - roll-up through the closed-form hierarchy (code >> shift) by sum, max, min and a weighted mean; the whole; across the two subjects, code for code;
//   - a derived measure (variance) from its rolled-up ingredients, never a roll-up of the ratio; a `none` measure (median) is refused when members would be merged;
//   - filters: by member number, by subject, by LEVEL IN PHYSICAL UNITS (the tiling's tile width in seconds);
//   - the SQL roll-up agrees with the in-memory one (rollup::members / across) on the same numbers (the parity the spec asks for);
//   - the merge table is the registry's data; the product-factor roll-up's SQL carries the radix arithmetic.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <map>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/measurement_query.h"
#include "nxr/io/rollup.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
bool near(double a, double b) { return std::abs(a - b) < 1e-12 * (1 + std::abs(b)); }

// a subject with a time line and a tiling of 8 tiles of 0.125 s; returns the selection's id
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
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_mquery_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};

  // THE MERGE RULE IS DATA: the registry's table
  CHECK(measure_merge("sum") == MergeRule::sum && measure_merge("max") == MergeRule::max && measure_merge("min") == MergeRule::min && measure_merge("mean") == MergeRule::mean);
  CHECK(measure_merge("median") == MergeRule::none && measure_merge("variance") == MergeRule::derived && measure_merge("envelope_max") == MergeRule::max && measure_merge("a_name_of_a_subject") == MergeRule::sum);
  const Json table = measure_merge_table();
  CHECK(table.at("sum") == "sum" && table.at("mean_square") == "mean" && table.at("median") == "none" && table.at("sd") == "derived");
  for (const auto& m : measure_registry()["measures"]) CHECK(m.contains("merge") && m["merge"].is_string());
  CHECK(rollup::merge_of("max") == rollup::Merge::Max && rollup::merge_of("mean") == rollup::Merge::Mean && rollup::merge_of("median") == rollup::Merge::None);

  DatasetWriter ds = create_dataset(root, "demo");
  const std::string s1 = make_subject(ds, "S1"), s2 = make_subject(ds, "S2");
  // member values 0..7 in S1, 10 + 0..7 doubled in S2; weights 1; mean_square and mean for the variance; a median
  const std::vector<double> a = {3, 1, 4, 1, 5, 9, 2, 6}, b = {2, 7, 1, 8, 2, 8, 1, 8};
  auto put = [&](const std::string& subject, const std::string& sel, const std::vector<double>& v, const std::string& measure, double weight = 1.0) {
    SubjectWriter sub = SubjectWriter::open(ds, subject);
    Json w = Json::array();
    for (std::size_t i = 0; i < v.size(); ++i) w.push_back(Json{{"selection_id", sel}, {"code", static_cast<std::int64_t>(i)}, {"measure", measure}, {"value", v[i]}, {"weight", weight}});
    write_measurements(sub.db(), w, std::string("test"));
  };
  put("S1", s1, a, "sum"); put("S2", s2, b, "sum");
  put("S1", s1, a, "max"); put("S2", s2, b, "max");
  put("S1", s1, a, "mean"); put("S2", s2, b, "mean");
  put("S1", s1, a, "median");
  std::vector<double> sq1, sq2;
  for (double x : a) sq1.push_back(x * x);
  for (double x : b) sq2.push_back(x * x);
  put("S1", s1, sq1, "mean_square"); put("S2", s2, sq2, "mean_square");

  SubjectWriter sub1 = SubjectWriter::open(ds, "S1");
  const Database& db = sub1.db();
  query::Query q;
  q.measure = "sum";
  q.filter.subjects = {"S1"};
  auto rows_of = [&](const query::Query& qq) {
    std::map<std::int64_t, double> out;
    for (const auto& r : query::members(db, qq)) out[r["code"].get<std::int64_t>()] = r["value"].get<double>();
    return out;
  };
  // as stored
  auto r0 = rows_of(q);
  CHECK(r0.size() == 8 && near(r0.at(5), 9.0));
  // SUM one level up (shift 1): pairs (3+1, 4+1, 5+9, 2+6)
  q.roll.shift = 1;
  auto r1 = rows_of(q);
  CHECK(r1.size() == 4 && near(r1.at(0), 4) && near(r1.at(1), 5) && near(r1.at(2), 14) && near(r1.at(3), 8));
  // PARITY: the in-memory roll-up of the same numbers (rollup::members) gives the same
  std::int64_t groups = 0;
  const auto mem = rollup::members(a, 1, 8, 1, rollup::Merge::Sum, &groups);
  CHECK(groups == 4);
  for (std::int64_t g = 0; g < groups; ++g) CHECK(near(mem[static_cast<std::size_t>(g)], r1.at(g)));
  // MAX and MIN by the registry's rule; the whole
  q.measure = "max";
  auto mx = rows_of(q);
  CHECK(near(mx.at(0), 3) && near(mx.at(2), 9));
  q.roll = query::RollUp{}; q.roll.whole = true;
  CHECK(near(rows_of(q).at(0), 9));
  // MEAN through weights: equal weights, the mean of the four
  q.measure = "mean"; q.roll = query::RollUp{}; q.roll.shift = 2;
  auto mn = rows_of(q);
  CHECK(near(mn.at(0), (3 + 1 + 4 + 1) / 4.0) && near(mn.at(1), (5 + 9 + 2 + 6) / 4.0));
  // a DERIVED measure from rolled-up ingredients: variance = mean_square - mean^2 of the MERGED members, not a roll-up of the member variances
  q.measure = "variance"; q.roll = query::RollUp{}; q.roll.shift = 2;
  auto vr = rows_of(q);
  CHECK(vr.size() == 2 && near(vr.at(0), 6.75 - 2.25 * 2.25) && near(vr.at(1), (25 + 81 + 4 + 36) / 4.0 - 5.5 * 5.5));
  // a MEDIAN never rolls up: as stored it is each member's own, grouped it is refused naming the reason
  q.measure = "median"; q.roll = query::RollUp{};
  CHECK(rows_of(q).size() == 8);
  q.roll.shift = 1;
  CHECK_EQ(code_of([&] { query::members(db, q); }), std::string(code::unsupported));
  // ACROSS SUBJECTS: the same member of both subjects, code for code (a selection NAME names the same selection in each subject)
  query::Query across;
  across.measure = "sum"; across.filter.selections = {"tiles"}; across.roll.across_subjects = true;
  const auto ar = query::members(db, across);
  CHECK(ar.size() == 8 && ar[0]["subject_id"].is_null() && near(ar[0]["value"].get<double>(), 3 + 2) && near(ar[5]["value"].get<double>(), 9 + 8) && ar[5]["members"].get<std::int64_t>() == 2);
  const std::vector<double>* both[2] = {&a, &b};
  const auto am = rollup::across({both[0], both[1]}, rollup::Merge::Sum);
  for (std::size_t i = 0; i < 8; ++i) CHECK(near(am[i], ar[i]["value"].get<double>()));   // parity with the in-memory merge
  // across subjects AND a level up
  across.roll.shift = 1;
  const auto ar1 = query::members(db, across);
  CHECK(ar1.size() == 4 && near(ar1[0]["value"].get<double>(), 3 + 1 + 2 + 7));
  // the long table: one row per subject, selection and member (what a mixed model reads)
  query::Query long_table;
  long_table.measure = "sum"; long_table.filter.selections = {"tiles"};
  const auto lt = query::members(db, long_table);
  CHECK(lt.size() == 16 && !lt[0]["subject_id"].is_null() && !lt[0]["selection_id"].is_null());
  // FILTERS: by member number, by subject, by value
  query::Query f;
  f.measure = "sum"; f.filter.members = {1, 2}; f.filter.subjects = {"S2"};
  const auto fr = rows_of(f);
  CHECK(fr.size() == 2 && near(fr.at(1), 7) && near(fr.at(2), 1));
  f.filter.subjects.clear(); f.filter.min_value = 4; f.filter.members = {0, 1, 2, 3};
  CHECK(query::members(db, f).size() == 2 + 1);                 // S1: 4 (code 2); S2: 7 (code 1), 8 (code 3)
  // by LEVEL IN PHYSICAL UNITS: the tiles are 0.125 s long
  query::Query lv;
  lv.measure = "sum";
  lv.filter.level = query::LevelRange{"s", 0.1, 0.2, std::nullopt};
  CHECK(query::members(db, lv).size() == 16);
  lv.filter.level = query::LevelRange{"s", 1.0, 2.0, std::nullopt};
  CHECK(query::members(db, lv).empty());
  lv.filter.level = query::LevelRange{"mm", 0.1, 0.2, std::nullopt};
  CHECK(query::members(db, lv).empty());                        // another unit is another size
  CHECK(query::level_size(db, s1) && near(query::level_size(db, s1)->size, 0.125) && query::level_size(db, s1)->unit == "s");
  // the statement a query runs: the SQL carries the closed-form shift and the registry's aggregate
  query::Query st; st.measure = "mean"; st.roll.shift = 3;
  const auto stmt = query::statement(db, st, {s1});
  CHECK(stmt.sql.find(">> 3") != std::string::npos && stmt.sql.find("sum(m.value * coalesce(m.weight, 1.0))") != std::string::npos);
  // the in-memory roll-ups of a product's factor (mixed radix, first factor fastest): radices {4, 2}, factor 0 by one level: [4 x 2] -> [2 x 2]
  {
    std::vector<double> x(8);
    for (int i = 0; i < 8; ++i) x[static_cast<std::size_t>(i)] = i;           // code = c0 + 4 c1
    std::vector<std::int64_t> out_radices;
    const auto y = rollup::product_factor(x, 1, {4, 2}, 0, 1, rollup::Merge::Sum, &out_radices);
    CHECK(out_radices == std::vector<std::int64_t>({2, 2}) && y.size() == 4 && near(y[0], 0 + 1) && near(y[1], 2 + 3) && near(y[2], 4 + 5) && near(y[3], 6 + 7));
  }
  // an unknown shift is refused
  query::Query bad; bad.measure = "sum"; bad.roll.shift = -1;
  CHECK_EQ(code_of([&] { query::members(db, bad); }), std::string(code::invalid_argument));
  bool threw = false;
  try { rollup::members({1, 2}, 1, 2, 1, rollup::Merge::None); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);


  ds.close();
  return nxrtest::finish("test_measurement_query");
}
