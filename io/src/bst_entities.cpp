// nxr::bst — nxr.convert.entities: labelling, label_set, flag_levels.
#include "nxr/bst/entities.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace nxr::bst {

using nxr::io::Cols;
using nxr::io::DictionaryResult;
using nxr::io::DictionarySpec;
using nxr::io::SelectionSpec;
using nxr::io::SubjectWriter;
namespace code = nxr::io::code;

std::string labelling(SubjectWriter& sub, const std::string& path, const std::vector<std::int64_t>& codes, const std::vector<std::string>& levels,
                      const LabellingOptions& opt) {
  DictionarySpec d;
  d.name = opt.dictionary;
  d.scope = opt.scope;
  d.names = levels;
  d.colors = opt.colors;
  const DictionaryResult dr = sub.dictionary(d);
  std::vector<std::int32_t> data(codes.size());
  for (std::size_t i = 0; i < codes.size(); ++i) {
    const auto lvl = static_cast<std::size_t>(codes[i]);
    data[i] = dr.codes.empty() ? static_cast<std::int32_t>(codes[i]) : static_cast<std::int32_t>(dr.codes.at(lvl));
  }
  SelectionSpec s;
  s.name = path.substr(path.rfind('/') + 1);
  s.path = path;
  s.type = "set";
  s.manifold_id = opt.manifold_id;
  s.data = nxr::io::view(data, {static_cast<std::int64_t>(data.size())});
  s.no_elements = !opt.element_rows;
  s.acquisition = opt.acquisition;
  if (opt.acquisition_id) s.cols.set("acquisition_id", *opt.acquisition_id);
  s.cols.set("dictionary_id", dr.id).set("n_elements", static_cast<std::int64_t>(data.size()));
  if (opt.description) s.cols.set("description", *opt.description);
  if (opt.of_field_id) s.cols.set("of_field_id", *opt.of_field_id);
  if (opt.member_of) s.cols.set("member_of_id", opt.member_of->first).set("member_ordinal", static_cast<std::int64_t>(opt.member_of->second));
  return sub.selection(std::move(s));
}

std::pair<std::string, std::vector<std::string>> label_set(SubjectWriter& sub, const std::string& set_path, const std::vector<LabelMember>& members,
                                                           const std::string& manifold_id, const std::string& description, const std::optional<std::string>& acquisition,
                                                           const std::function<std::pair<std::string, std::string>(const std::string&)>& dictionary) {
  SelectionSpec set;
  set.name = set_path.substr(set_path.rfind('/') + 1);
  set.path = set_path;
  set.type = "set";
  set.manifold_id = manifold_id;
  set.acquisition = acquisition;
  set.cols.set("n_members", static_cast<std::int64_t>(members.size())).set("description", description);
  const std::string set_id = sub.selection(std::move(set));
  std::vector<std::string> ids;
  int k = 0;
  for (const auto& m : members) {
    const auto [dname, scope] = dictionary(m.family);
    LabellingOptions o;
    o.manifold_id = manifold_id;
    o.dictionary = dname;
    o.scope = scope;
    o.colors = m.colors;
    o.acquisition = acquisition;
    o.description = m.family + " \xe2\x80\x94 one level per element";
    o.member_of = std::make_pair(set_id, k++);
    ids.push_back(labelling(sub, set_path + "/" + m.family, m.codes, m.levels, o));
  }
  return {set_id, ids};
}

FlagLevels flag_levels(const std::vector<std::int64_t>& flags, const std::string& where) {
  std::set<std::int64_t, std::greater<>> present(flags.begin(), flags.end());   // sorted, descending
  std::vector<std::int64_t> unknown;
  for (auto v : present) if (v != 1 && v != -1) unknown.push_back(v);
  if (!unknown.empty()) {
    std::string list;
    for (auto it = unknown.rbegin(); it != unknown.rend(); ++it) list += (list.empty() ? "" : ", ") + std::to_string(*it);
    throw nxr::io::Error(code::invalid_argument, where + "channel flags hold [" + list + "], which is neither 1 (good) nor -1 (bad)");
  }
  FlagLevels out;
  std::vector<std::int64_t> order(present.begin(), present.end());
  for (auto v : flags) out.codes.push_back(std::find(order.begin(), order.end(), v) - order.begin());
  for (auto v : order) {
    out.levels.push_back(v == 1 ? "good" : "bad");
    out.colors.push_back(v == 1 ? Rgb{90, 170, 110} : Rgb{228, 87, 86});
  }
  return out;
}

Rgb rgb255(const std::vector<double>& c_in) {
  std::vector<double> c(c_in.begin(), c_in.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(3, c_in.size())));
  if (c.size() < 3) c = {0.5, 0.5, 0.5};
  const double scale = *std::max_element(c.begin(), c.end()) <= 1.0 ? 255.0 : 1.0;
  Rgb out{};
  for (std::size_t i = 0; i < 3; ++i) out[i] = static_cast<int>(std::nearbyint(c[i] * scale));   // Python's round: half to even
  return out;
}

std::string time_spans(SubjectWriter& sub, const std::string& path, const std::vector<TimeSpan>& spans, const std::vector<std::string>& types,
                       const TimeSpansOptions& opt) {
  const std::string name = path.substr(path.rfind('/') + 1);
  DictionarySpec d;
  d.name = name;
  d.scope = "subject";
  d.names = types;
  if (!opt.colors.empty()) for (const auto& c : opt.colors) d.colors.push_back(rgb255(c));
  const DictionaryResult dr = sub.dictionary(d);
  SelectionSpec s;
  s.name = name;
  s.path = path;
  s.type = "spans";
  s.manifold_id = opt.line_id;
  std::vector<nxr::io::SpanSpec> rows;
  for (const auto& sp : spans) rows.push_back({sp.start, sp.stop, dr.codes.at(static_cast<std::size_t>(sp.type))});
  s.spans = std::move(rows);
  s.acquisition = opt.acquisition;
  if (opt.acquisition_id) s.cols.set("acquisition_id", *opt.acquisition_id);
  s.cols.set("cell", "1").set("dictionary_id", dr.id).set("unit", "s").set("of_field_id", opt.of_field_id).set("description", opt.description);
  if (opt.params) s.cols.json("params_json", *opt.params);
  return sub.selection(std::move(s));
}

}  // namespace nxr::bst
