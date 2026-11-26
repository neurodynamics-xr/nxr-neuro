// nxr-io — the measurement query API (measurement_query.h): filters and roll-ups of what is measured, in SQL for scalar member rows and by the same merge rule in memory for arrays.
#include "nxr/io/measurement_query.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>

#include "nxr/io/bundle.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/dataset_index.h"
#include "nxr/io/errors.h"
#include "nxr/io/rollup.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/statistic.h"
#include "nxr/io/writer.h"

namespace nxr::io::query {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, "measurement query: " + m); }

std::string marks(std::size_t n) {
  std::string s;
  for (std::size_t i = 0; i < n; ++i) s += (i ? ", ?" : "?");
  return s;
}
double number(const Value& v) {
  if (std::holds_alternative<double>(v)) return std::get<double>(v);
  if (std::holds_alternative<std::int64_t>(v)) return static_cast<double>(std::get<std::int64_t>(v));
  return kNaN;
}
Json json_of(const Value& v) {
  if (std::holds_alternative<std::monostate>(v)) return Json(nullptr);
  if (std::holds_alternative<std::int64_t>(v)) return Json(std::get<std::int64_t>(v));
  if (std::holds_alternative<double>(v)) return Json(std::get<double>(v));
  return Json(std::get<std::string>(v));
}

// the SQL aggregate of a merge rule over the rows of one group (the registry's data, measures.h): the ONE definition; compute applies the same rule in memory
std::string aggregate(MergeRule r) {
  switch (r) {
    case MergeRule::sum: return "sum(m.value)";
    case MergeRule::max: return "max(m.value)";
    case MergeRule::min: return "min(m.value)";
    case MergeRule::mean: return "sum(m.value * coalesce(m.weight, 1.0)) / sum(coalesce(m.weight, 1.0))";   // a mean through its sums and its weights, never a mean of means
    case MergeRule::none: return "max(m.value)";                                                              // a member's own value; grouping members is refused after the read
    case MergeRule::derived: break;
  }
  refuse("a derived measure has no aggregate of its own (its ingredients are merged first)");
}

// the member number a roll-up puts a stored code in: `code >> shift`, one factor of a mixed-radix product (first factor fastest: unpack, shift that factor's member, repack with its radix rolled to
// ceil(radix / 2^shift): the arithmetic of selection::coarser_product_code), or 0 for the whole
std::string code_expression(const RollUp& roll, const std::vector<std::int64_t>& radices) {
  if (roll.whole) return "0";
  if (roll.factor) {
    if (*roll.factor >= radices.size()) refuse("the product has " + std::to_string(radices.size()) + " factors, not " + std::to_string(*roll.factor + 1));
    if (roll.shift < 0 || roll.shift > 62) refuse("the shift is 0 to 62");
    std::vector<std::int64_t> rolled = radices;
    rolled[*roll.factor] = ((radices[*roll.factor] - 1) >> roll.shift) + 1;
    std::string e;
    std::int64_t from = 1, to = 1;
    for (std::size_t k = 0; k < radices.size(); ++k) {
      std::string part = "((m.code / " + std::to_string(from) + ") % " + std::to_string(radices[k]) + ")";
      if (k == *roll.factor && roll.shift > 0) part = "(" + part + " >> " + std::to_string(roll.shift) + ")";
      e += (k ? " + " : "") + part + " * " + std::to_string(to);
      from *= radices[k];
      to *= rolled[k];
    }
    return "(" + e + ")";
  }
  if (roll.shift < 0) refuse("the shift is 0 or more");
  if (roll.shift == 0) return "m.code";
  return "(m.code >> " + std::to_string(std::min(roll.shift, 62)) + ")";
}

// the ingredients of a derived measure, or the measure itself
std::vector<std::string> ingredients_of(const std::string& measure) {
  const auto spec = measure_spec(measure);
  if (!spec || (*spec)["derived_from"].is_null()) return {measure};
  std::vector<std::string> out;
  for (const auto& x : (*spec)["derived_from"]) out.push_back(x.get<std::string>());
  return out;
}

// the member numbers a filter's `labels` are, in one selection: the labels array through the Store (selection::member_labels), the label of member i at position i
std::vector<std::int64_t> numbers_of_labels(const Database& db, const Store* store, const std::string& selection_id, const std::vector<std::int64_t>& labels) {
  if (!store) refuse("a filter by member label needs the subject's Store (the labels are an array beside the selection)");
  const std::vector<std::int64_t> all = selection::member_labels(db, *store, selection_id);
  std::vector<std::int64_t> out;
  for (std::size_t i = 0; i < all.size(); ++i)
    if (std::find(labels.begin(), labels.end(), all[i]) != labels.end()) out.push_back(static_cast<std::int64_t>(i));
  return out;
}

struct Built {
  std::string sql;
  std::vector<Value> params;
};

Built build(const Database& db, const std::string& measure, const Query& q, const std::vector<std::string>& ids, const Store* store, const std::vector<std::int64_t>& radices) {
  const MergeRule rule = measure_read_rule(db, measure).rule;   // a measure neither declared nor registered is read as stored (`none`) and refused when members merge (below)
  const std::string agg = aggregate(rule);
  const std::string grp = code_expression(q.roll, radices);
  Built b;
  const bool across = q.roll.across_subjects;
  std::string w = "m.measure = ? AND m.code IS NOT NULL";
  b.params.push_back(V(measure));
  // the selections, each with the members the filter restricts it to (numbers, or labels through its labels array)
  if (q.filter.members.empty() && q.filter.labels.empty()) {
    w += " AND m.selection_id IN (" + marks(ids.size()) + ")";
    for (const auto& id : ids) b.params.push_back(V(id));
  } else {
    std::string any;
    for (const auto& id : ids) {
      std::set<std::int64_t> nums(q.filter.members.begin(), q.filter.members.end());
      if (!q.filter.labels.empty()) for (const std::int64_t n : numbers_of_labels(db, store, id, q.filter.labels)) nums.insert(n);
      if (nums.empty()) continue;
      any += (any.empty() ? "" : " OR ") + std::string("(m.selection_id = ? AND m.code IN (") + marks(nums.size()) + "))";
      b.params.push_back(V(id));
      for (const std::int64_t n : nums) b.params.push_back(V(n));
    }
    w += any.empty() ? " AND 0" : " AND (" + any + ")";
  }
  if (q.filter.of_field) {
    w += " AND m.of_field_id IN (SELECT id FROM field WHERE id = ? OR name = ?)";
    b.params.push_back(V(*q.filter.of_field)); b.params.push_back(V(*q.filter.of_field));
    if (has_column(db, "selection_measurement", "of_statistic_id")) w += " AND m.of_statistic_id IS NULL";
  }
  if (q.filter.within_selection) {
    w += " AND m.within_selection_id = ?";
    b.params.push_back(V(*q.filter.within_selection));
    if (q.filter.within_code) { w += " AND m.within_code = ?"; b.params.push_back(V(*q.filter.within_code)); }
  } else {
    w += " AND m.within_selection_id IS NULL";
  }
  const std::string keys = across ? "" : "s.subject_id, m.selection_id, ";
  b.sql = std::string("SELECT ") + (across ? "" : "s.subject_id AS subject_id, m.selection_id AS selection_id, ") + grp + " AS grp, " + agg + " AS value, sum(coalesce(m.weight, 1.0)) AS weight, count(*) AS members " +
          "FROM selection_measurement m JOIN selection s ON s.id = m.selection_id WHERE " + w + " GROUP BY " + keys + "grp ORDER BY " + keys + "grp";
  return b;
}

std::vector<std::int64_t> common_radices(const Database& db, const std::vector<std::string>& ids, const RollUp& roll) {
  if (!roll.factor) return {};
  std::vector<std::int64_t> radices;
  for (const auto& id : ids) {
    const auto r = selection::product_radices(db, id);
    if (radices.empty()) radices = r;
    else if (radices != r) refuse("the selections are products of different shapes: a factor is rolled in one product at a time");
  }
  return radices;
}
}  // namespace

std::optional<LevelSize> level_size(const Database& db, const std::string& selection_id, std::optional<std::size_t> factor) {
  const auto row = db.read("selection", selection_id);
  if (!row) throw Error(code::no_such_row, "selection " + selection_id + ": no such row");
  const std::string type = text(*row, "type");
  const Json p = Json::parse(opt_text(*row, "params_json").value_or("{}"), nullptr, false);
  auto of_level = [&](const std::string& ladder_id, std::int64_t code) -> std::optional<LevelSize> {
    for (const auto& g : selection::grid_level_sizes(db, ladder_id)) if (g.code == code) return LevelSize{g.size, g.unit};
    return std::nullopt;
  };
  if (factor) {
    const auto factors = db.all("SELECT factor_selection_id FROM selection_factor WHERE selection_id = ? ORDER BY ordinal", {V(selection_id)});
    if (*factor >= factors.size()) return std::nullopt;
    const std::vector<int> levels = selection::product_factor_levels(db, selection_id);
    const std::string fid = text(factors[*factor], "factor_selection_id");
    if (levels[*factor] >= 0) {
      for (const auto& g : selection::grid_level_sizes(db, fid)) if (g.level == levels[*factor]) return LevelSize{g.size, g.unit};   // a factor's stored level is the level's IDENTITY (its ordinal in the grid's level dictionary), never its label (the power of two of a constant-Q level))
      return std::nullopt;
    }
    return level_size(db, fid);
  }
  if (!is_null(*row, "member_of_id") && !is_null(*row, "code")) {            // a level row of a grid
    if (const auto s = of_level(text(*row, "member_of_id"), integer(*row, "code"))) return s;
  }
  if (type == "spans" && p.is_object() && p.contains("tiling") && p["tiling"].contains("width") && !is_null(*row, "unit"))   // a tiling: its tile's width in the selection's unit
    return LevelSize{p["tiling"]["width"].get<double>(), text(*row, "unit")};
  return std::nullopt;
}

std::vector<std::string> selections(const Database& db, const std::string& measure, const Filter& f) {
  std::string sql = "SELECT DISTINCT s.id AS id, s.subject_id AS subject_id, s.name AS name FROM selection s JOIN selection_measurement m ON m.selection_id = s.id WHERE m.measure = ? AND s.status = 'complete'";
  std::vector<Value> a{V(measure)};
  if (!f.selections.empty()) {
    sql += " AND (s.id IN (" + marks(f.selections.size()) + ") OR s.name IN (" + marks(f.selections.size()) + "))";
    for (int twice = 0; twice < 2; ++twice) for (const auto& x : f.selections) a.push_back(V(x));
  }
  if (!f.subjects.empty()) {
    sql += " AND s.subject_id IN (SELECT id FROM subject WHERE id IN (" + marks(f.subjects.size()) + ") OR name IN (" + marks(f.subjects.size()) + "))";
    for (int twice = 0; twice < 2; ++twice) for (const auto& x : f.subjects) a.push_back(V(x));
  }
  if (f.of_field) {
    sql += " AND m.of_field_id IN (SELECT id FROM field WHERE id = ? OR name = ?)";
    a.push_back(V(*f.of_field)); a.push_back(V(*f.of_field));
  }
  if (f.within_selection) {
    sql += " AND m.within_selection_id = ?";
    a.push_back(V(*f.within_selection));
    if (f.within_code) { sql += " AND m.within_code = ?"; a.push_back(V(*f.within_code)); }
  }
  sql += " ORDER BY s.subject_id, s.name, s.id";
  std::vector<std::string> out;
  for (const Row& r : db.all(sql, a)) {
    if (f.level) {
      const auto sz = level_size(db, text(r, "id"), f.level->factor);
      if (!sz || sz->unit != f.level->unit) continue;
      if (f.level->min && sz->size < *f.level->min) continue;
      if (f.level->max && sz->size > *f.level->max) continue;
    }
    out.push_back(text(r, "id"));
  }
  return out;
}

Statement statement(const Database& db, const Query& q, const std::vector<std::string>& selection_ids) {
  const std::string first = ingredients_of(q.measure).front();
  Built b = build(db, first, q, selection_ids, nullptr, common_radices(db, selection_ids, q.roll));
  return Statement{b.sql, b.params};
}

Json members(const Database& db, const Query& q, const Store* store) {
  const std::vector<std::string> from = ingredients_of(q.measure);
  const std::vector<std::string> ids = selections(db, from.front(), q.filter);
  if (!q.filter.subjects.empty() && ids.empty()) return Json::array();
  if (ids.empty()) return Json::array();
  const std::vector<std::int64_t> radices = common_radices(db, ids, q.roll);
  struct Got { Json subject, selection; std::int64_t grp = 0; double value = kNaN, weight = 0; std::int64_t members = 0; };
  auto key_of = [](const Got& g) { return g.subject.dump() + "|" + g.selection.dump() + "|" + std::to_string(g.grp); };
  std::vector<std::map<std::string, Got>> by_key;
  std::vector<Got> first_rows;
  for (std::size_t i = 0; i < from.size(); ++i) {
    const Built b = build(db, from[i], q, ids, store, radices);
    const ReadRule read = measure_read_rule(db, from[i]);
    const MergeRule rule = read.rule;
    std::map<std::string, Got> m;
    for (const Row& r : db.all(b.sql, b.params)) {
      Got g;
      g.subject = r.count("subject_id") ? json_of(r.at("subject_id")) : Json(nullptr);
      g.selection = r.count("selection_id") ? json_of(r.at("selection_id")) : Json(nullptr);
      g.grp = integer(r, "grp"); g.value = number(r.at("value")); g.weight = number(r.at("weight")); g.members = integer(r, "members");
      if (!read.known && g.members > 1) throw Error(code::unsupported, undeclared_refusal(from[i]));
      if (rule == MergeRule::none && g.members > 1)
        throw Error(code::unsupported, "measure " + from[i] + " does not roll up: its value is a member's own (a median of two members together is no combination of their medians), so members cannot be merged");
      m[key_of(g)] = g;
      if (i == 0) first_rows.push_back(g);
    }
    by_key.push_back(std::move(m));
  }
  std::vector<Json> out;
  for (const Got& g : first_rows) {
    double value = g.value;
    if (from.size() > 1 || from.front() != q.measure) {                // a derived measure: its formula over the merged ingredients (ratios of sums, never sums of ratios)
      std::map<std::string, double> ing;
      bool ok = true;
      for (std::size_t i = 0; i < from.size(); ++i) {
        const auto it = by_key[i].find(key_of(g));
        if (it == by_key[i].end() || !std::isfinite(it->second.value)) { ok = false; break; }
        ing[from[i]] = it->second.value;
      }
      if (!ok) continue;
      value = measure_formula(q.measure, ing);
      if (!std::isfinite(value)) continue;
    }
    if (q.filter.min_value && value < *q.filter.min_value) continue;
    if (q.filter.max_value && value > *q.filter.max_value) continue;
    out.push_back(Json{{"subject_id", g.subject}, {"selection_id", g.selection}, {"code", g.grp}, {"value", value}, {"weight", g.weight}, {"members", g.members}});
  }
  if (!q.order.empty()) {
    const bool asc = q.order == "asc";
    std::stable_sort(out.begin(), out.end(), [&](const Json& a, const Json& b) { return asc ? a["value"].get<double>() < b["value"].get<double>() : a["value"].get<double>() > b["value"].get<double>(); });
  }
  if (q.limit > 0 && static_cast<std::int64_t>(out.size()) > q.limit) out.resize(static_cast<std::size_t>(q.limit));
  return Json(out);
}

std::vector<std::string> subjects_where(const std::filesystem::path& dataset_folder, const std::string& filter_text) {
  SubjectSelection sel;
  sel.where = filter_text;
  std::vector<std::string> out;
  for (const auto& m : select_subjects(dataset_folder, sel).members) out.push_back(m.subject_id);
  return out;
}

Database open_index(const std::filesystem::path& dataset_folder) {
  if (!index_is_current(dataset_folder)) build_index(dataset_folder);
  return Database::open(dataset_folder / kIndexFile, false, "", true);
}

// ── arrays ──────────────────────────────────────────────────────────────────────────────────────────────────────

namespace {
std::vector<double> roll_array(const Database& db, const std::string& selection_id, const std::vector<double>& x, std::int64_t rows, std::int64_t n, const RollUp& roll, rollup::Merge op, std::int64_t* groups) {
  try {
    if (roll.whole) return rollup::members(x, rows, n, 62, op, groups);
    if (roll.factor) {
      const std::vector<std::int64_t> radices = selection::product_radices(db, selection_id);
      std::vector<std::int64_t> rolled;
      std::vector<double> y = rollup::product_factor(x, rows, radices, *roll.factor, roll.shift, op, &rolled);
      std::int64_t total = 1;
      for (const std::int64_t r : rolled) total *= r;
      if (groups) *groups = total;
      return y;
    }
    return rollup::members(x, rows, n, roll.shift, op, groups);
  } catch (const std::invalid_argument& e) {
    throw Error(code::invalid_argument, std::string("measurement query: ") + e.what());
  }
}

// one ingredient, merged: the stored array of the measure by its rule (a mean through the stored weights)
MeasurementValues merged_stored(SubjectWriter& sub, const std::string& selection_id, const std::optional<std::string>& of_field_id, const std::optional<std::string>& of_statistic_id,
                                const std::optional<std::int64_t>& of_index, const std::string& measure, const RollUp& roll) {
  const Database& db = sub.db();
  MeasurementValues v = measurement_values(sub, selection_id, of_field_id, of_statistic_id, of_index, measure);
  if (!roll.shift && !roll.whole && !roll.factor) return v;
  const MergeRule rule = measure_merge(db, measure);   // the subject's declaration, else the registry's; neither: refused (never summed by default)
  if (rule == MergeRule::none) throw Error(code::unsupported, "measure " + measure + " does not roll up: its value is a member's own");
  std::int64_t groups = 0;
  if (rule == MergeRule::mean) {
    MeasurementValues w = measurement_values(sub, selection_id, of_field_id, of_statistic_id, of_index, "weight");
    if (w.rows != v.rows || w.members != v.members) refuse("the weights of '" + measure + "' differ in shape from its values");
    std::vector<double> num(v.values.size());
    for (std::size_t i = 0; i < num.size(); ++i) num[i] = v.values[i] * w.values[i];
    const std::vector<double> ns = roll_array(db, selection_id, num, v.rows, v.members, roll, rollup::Merge::Sum, &groups);
    const std::vector<double> ws = roll_array(db, selection_id, w.values, v.rows, v.members, roll, rollup::Merge::Sum, &groups);
    MeasurementValues out = v;
    out.members = groups;
    out.values.resize(ns.size());
    for (std::size_t i = 0; i < ns.size(); ++i) out.values[i] = ws[i] > 0 ? ns[i] / ws[i] : kNaN;
    return out;
  }
  MeasurementValues out = v;
  out.values = roll_array(db, selection_id, v.values, v.rows, v.members, roll, rollup::merge_of(rule, measure), &groups);
  out.members = groups;
  return out;
}
}  // namespace

MeasurementValues rolled_values(SubjectWriter& sub, const std::string& selection_id, const std::optional<std::string>& of_field_id, const std::optional<std::string>& of_statistic_id,
                                const std::optional<std::int64_t>& of_index, const std::string& measure, const RollUp& roll) {
  if (!measure_is_derived(measure)) return merged_stored(sub, selection_id, of_field_id, of_statistic_id, of_index, measure, roll);
  const std::vector<std::string> from = ingredients_of(measure);
  std::vector<MeasurementValues> parts;
  for (const auto& f : from) parts.push_back(merged_stored(sub, selection_id, of_field_id, of_statistic_id, of_index, f, roll));
  MeasurementValues out = parts[0];
  for (const auto& p : parts) if (p.rows != out.rows || p.members != out.members) refuse("the ingredients of '" + measure + "' differ in shape");
  out.unit = std::nullopt;
  out.computed_by.clear();
  for (const auto& p : parts) for (const auto& c : p.computed_by) out.computed_by.push_back(c);
  for (std::size_t i = 0; i < out.values.size(); ++i) {
    std::map<std::string, double> m;
    bool ok = true;
    for (std::size_t k = 0; k < from.size(); ++k) { const double v = parts[k].values[i]; if (!std::isfinite(v)) ok = false; m[from[k]] = v; }
    const double v = ok ? measure_formula(measure, m) : kNaN;
    out.values[i] = std::isfinite(v) ? v : kNaN;
  }
  return out;
}

MeasurementValues rolled_values_across(const std::vector<SubjectArray>& subjects, const std::string& measure, const RollUp& roll) {
  if (subjects.empty()) refuse("across subjects: at least one subject");
  const std::vector<std::string> from = ingredients_of(measure);
  RollUp within = roll;
  within.across_subjects = false;
  std::vector<std::vector<MeasurementValues>> per_ingredient(from.size());
  for (const SubjectArray& s : subjects) {
    if (!s.subject) refuse("across subjects: a subject is not open");
    for (std::size_t i = 0; i < from.size(); ++i) per_ingredient[i].push_back(merged_stored(*s.subject, s.selection_id, s.of_field_id, s.of_statistic_id, s.of_index, from[i], within));
  }
  std::vector<MeasurementValues> merged;
  for (std::size_t i = 0; i < from.size(); ++i) {
    // each subject's declaration, else the registry's: they must agree (one name, one meaning across the group)
    const MergeRule rule = measure_merge(subjects[0].subject->db(), from[i]);
    for (const SubjectArray& s : subjects)
      if (measure_merge(s.subject->db(), from[i]) != rule) refuse("across subjects: the subjects declare '" + from[i] + "' with different merge rules");
    if (rule == MergeRule::none) throw Error(code::unsupported, "measure " + from[i] + " does not roll up across subjects: its value is a member's own");
    MeasurementValues out = per_ingredient[i][0];
    for (const auto& p : per_ingredient[i]) if (p.rows != out.rows || p.members != out.members) refuse("across subjects: the same selection differs in shape between subjects (not on one group tree)");
    std::vector<const std::vector<double>*> stack;
    std::vector<std::vector<double>> weighted;
    if (rule == MergeRule::mean) {          // a mean across subjects through the weights too: sum(w v) / sum(w)
      std::vector<std::vector<double>> weights;
      for (std::size_t s = 0; s < subjects.size(); ++s) {
        const MeasurementValues wr = merged_stored(*subjects[s].subject, subjects[s].selection_id, subjects[s].of_field_id, subjects[s].of_statistic_id, subjects[s].of_index, "weight", within);
        weights.push_back(wr.values);
        std::vector<double> num(per_ingredient[i][s].values.size());
        for (std::size_t k = 0; k < num.size(); ++k) num[k] = per_ingredient[i][s].values[k] * wr.values[k];
        weighted.push_back(std::move(num));
      }
      for (const auto& x : weighted) stack.push_back(&x);
      const std::vector<double> num = rollup::across(stack, rollup::Merge::Sum);
      stack.clear();
      for (const auto& x : weights) stack.push_back(&x);
      const std::vector<double> den = rollup::across(stack, rollup::Merge::Sum);
      out.values.resize(num.size());
      for (std::size_t k = 0; k < num.size(); ++k) out.values[k] = den[k] > 0 ? num[k] / den[k] : kNaN;
    } else {
      for (const auto& p : per_ingredient[i]) stack.push_back(&p.values);
      out.values = rollup::across(stack, rollup::merge_of(rule, from[i]));
    }
    merged.push_back(std::move(out));
  }
  MeasurementValues out = merged[0];
  if (!measure_is_derived(measure)) return out;
  out.unit = std::nullopt;
  for (std::size_t i = 0; i < out.values.size(); ++i) {
    std::map<std::string, double> m;
    bool ok = true;
    for (std::size_t k = 0; k < from.size(); ++k) { const double v = merged[k].values[i]; if (!std::isfinite(v)) ok = false; m[from[k]] = v; }
    const double v = ok ? measure_formula(measure, m) : kNaN;
    out.values[i] = std::isfinite(v) ? v : kNaN;
  }
  return out;
}

}  // namespace nxr::io::query
