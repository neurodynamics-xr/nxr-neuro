// Declared measures (measures.h, "declared measures"; docs/reference.md): the registry is a vocabulary, not a gate. A measure the registry does not know is stored
// and DECLARED by its writer (its merge rule, form, unit, description), kept in the subject's measure_declaration table. On a grid of 8 tiles with known answers:
//   - a new measure declared `sum` rolls up by its declared rule (SQL query and the array read agree);
//   - a p value declared `none` is read as stored and refused when members would merge;
//   - a measure neither declared nor registered is read as stored, refused at any roll-up (never summed by default), and has no rule of its own;
//   - a declaration that contradicts the registry, or the subject's earlier one, is refused; `derived` only for a registered name;
//   - a statistic's measure and method: registered, or declared (a description), else refused; a declared unit rule that contradicts the registry is refused.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <map>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/measurement_query.h"
#include "nxr/io/measures.h"
#include "nxr/io/rollup.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/statistic.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
template <class F> std::string message_of(F f) { try { f(); } catch (const Error& e) { return e.what(); } return ""; }
bool near(double a, double b) { return std::abs(a - b) < 1e-12 * (1 + std::abs(b)); }
bool has(const std::string& t, const std::string& part) { return t.find(part) != std::string::npos; }

// a subject with a time line and a tiling of 8 tiles; returns the selection's id
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
MeasureDeclaration decl(const std::string& name, const std::optional<std::string>& merge) {
  MeasureDeclaration d;
  d.name = name;
  d.merge = merge;
  return d;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_mdecl_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};

  // a name the registry does not know has NO rule of its own (no fallback to sum); a registered one keeps its rule
  CHECK_EQ(code_of([] { measure_merge("flow_thing"); }), std::string(code::unsupported));
  CHECK(has(message_of([] { measure_merge("flow_thing"); }), "neither declared nor registered"));
  CHECK_EQ(code_of([] { rollup::merge_of("flow_thing_max"); }), std::string(code::unsupported));
  CHECK(measure_merge("median") == MergeRule::none && measure_merge("sum") == MergeRule::sum);

  DatasetWriter ds = create_dataset(root, "demo");
  const std::string sel = make_subject(ds, "S1");
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  Database& db = sub.db();
  const std::vector<double> a = {3, 1, 4, 1, 5, 9, 2, 6};
  auto put = [&](const std::string& measure) {
    Json w = Json::array();
    for (std::size_t i = 0; i < a.size(); ++i) w.push_back(Json{{"selection_id", sel}, {"code", static_cast<std::int64_t>(i)}, {"measure", measure}, {"value", a[i]}});
    write_measurements(db, w, std::string("test"));
  };
  auto rows_of = [&](const std::string& measure, int shift) {
    query::Query q;
    q.measure = measure;
    q.roll.shift = shift;
    std::map<std::int64_t, double> out;
    for (const auto& r : query::members(db, q)) out[r["code"].get<std::int64_t>()] = r["value"].get<double>();
    return out;
  };

  // A NEW MEASURE, DECLARED `sum`: rolls up by its declared rule
  const Json stored = declare_measure(db, MeasureDeclaration{"flow_thing", std::string("sum"), std::string("mode"), std::nullopt, std::nullopt, std::string("a coefficient summed over bins")}, std::string("test"));
  CHECK(stored["merge"] == "sum" && stored["form"] == "mode" && stored["declared_by"] == "test");
  put("flow_thing");
  CHECK(measure_merge(db, "flow_thing") == MergeRule::sum);
  const auto up = rows_of("flow_thing", 1);
  CHECK(up.size() == 4 && near(up.at(0), 4) && near(up.at(2), 14));
  {
    Json q = Json{{"selectionId", sel}, {"measure", "flow_thing"}, {"shift", 2}};
    const Json r = query_measurements(db, q);
    double total = 0;
    for (const auto& x : r) total += x["value"].get<double>();
    CHECK(r.size() == 2 && near(total, 31));
  }
  // the same declaration again is accepted (it says the same); another rule for the same name is refused
  CHECK(declare_measure(db, decl("flow_thing", "sum"), std::nullopt)["merge"] == "sum");
  CHECK(has(message_of([&] { declare_measure(db, decl("flow_thing", "max"), std::nullopt); }), "already declares merge 'sum'"));

  // A P VALUE DECLARED `none`: read as stored, refused when members merge
  declare_measure(db, decl("p_value_of_test", "none"), std::nullopt);
  put("p_value_of_test");
  CHECK(rows_of("p_value_of_test", 0).size() == 8);
  CHECK_EQ(code_of([&] { rows_of("p_value_of_test", 1); }), std::string(code::unsupported));
  CHECK_EQ(code_of([&] { query_measurements(db, Json{{"selectionId", sel}, {"measure", "p_value_of_test"}, {"shift", 1}}); }), std::string(code::unsupported));

  // AN UNDECLARED, UNKNOWN MEASURE: read as stored; any roll-up is refused naming the missing declaration (never summed)
  put("mystery");
  CHECK(rows_of("mystery", 0).size() == 8);
  CHECK(has(message_of([&] { rows_of("mystery", 1); }), "neither declared nor registered"));
  CHECK(has(message_of([&] { query_measurements(db, Json{{"selectionId", sel}, {"measure", "mystery"}, {"shift", 1}}); }), "neither declared nor registered"));
  CHECK(!measure_read_rule(db, "mystery").known);

  // CONTRADICTIONS WITH THE REGISTRY are refused (one name, one meaning); a declaration may leave out what the registry gives
  CHECK(has(message_of([&] { declare_measure(db, decl("median", "sum"), std::nullopt); }), "the registry 'none'"));
  CHECK(declare_measure(db, decl("median", std::nullopt), std::nullopt)["merge"] == "none");
  MeasureDeclaration wrong_unit = decl("duration", std::nullopt);
  wrong_unit.unit = "ms";
  CHECK(has(message_of([&] { declare_measure(db, wrong_unit, std::nullopt); }), "unit 'ms'"));
  // no rule at all for a new name; `derived` only for a registered one (io holds the formulas); a bad rule names the rules
  CHECK(has(message_of([&] { declare_measure(db, decl("another", std::nullopt), std::nullopt); }), "gives its merge rule"));
  CHECK(has(message_of([&] { declare_measure(db, decl("a_ratio", "derived"), std::nullopt); }), "only registered measures"));
  CHECK(has(message_of([&] { declare_measure(db, decl("odd", "median"), std::nullopt); }), "not a merge rule"));
  // the JSON form a request carries: an unknown key is refused
  CHECK_EQ(code_of([] { measure_declaration_of(Json{{"name", "x"}, {"merge", "sum"}, {"rule", "sum"}}); }), std::string(code::invalid_argument));
  CHECK(measure_declaration_of(Json{{"name", "x"}, {"merge", "max"}}).merge == std::optional<std::string>("max"));

  // A STATISTIC: a measure and a method the registries do not know are declared, else refused; a declared unit rule that contradicts the registry is refused
  CHECK(has(message_of([&] { check_statistic_declaration(db, "flow_state_centroid", "kmeans", Json(), "statistic 'x'"); }), "declare what it is"));
  CHECK(has(message_of([&] { check_statistic_declaration(db, "flow_state_centroid", "kmeans", Json{{"measure", Json{{"description", "a centroid"}}}}, "statistic 'x'"); }), "declare how the values were estimated"));
  CHECK(message_of([&] { check_statistic_declaration(db, "flow_state_centroid", "kmeans", Json{{"measure", Json{{"description", "a centroid"}}}, {"method", Json{{"description", "k-means"}}}}, "s"); }).empty());
  CHECK(message_of([&] { check_statistic_declaration(db, "power_density", "welch", Json(), "s"); }).empty());
  CHECK(has(message_of([&] { check_statistic_declaration(db, "power_density", "welch", Json{{"measure", Json{{"unit_rule", "Hz"}}}}, "s"); }), "the registry"));

  return nxrtest::finish("test_measure_declarations");
}
