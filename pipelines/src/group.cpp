// nxr-pipelines — the group scope of the measurement pyramid (group.h): the group tree per subject, and the group reduce.
#include "nxr/pipelines/group.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>

#include "nxr/io/layout.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/pyramid.h"

namespace nxr::pipelines {

using namespace nxr::io;
namespace lad = nxr::ladder;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(code::no_such_row, m); }

constexpr const char* kBy = "nxr-neuro pyramid reduce";

// an array's values (or the slice [b0, b0+1) of its first axis) widened to double, with the shape of what was read
std::vector<double> widen(const ArrayReader& r, const std::vector<std::uint8_t>& raw) {
  const std::size_t n = raw.size() / r.itemsize();
  std::vector<double> out(n);
  auto conv = [&](auto tag) { using T = decltype(tag); for (std::size_t i = 0; i < n; ++i) { T v; std::memcpy(&v, raw.data() + i * sizeof(T), sizeof(T)); out[i] = static_cast<double>(v); } };
  switch (r.dtype()) {
    case DType::Float64: std::memcpy(out.data(), raw.data(), raw.size()); break;
    case DType::Float32: conv(float{}); break;
    case DType::Int64: conv(std::int64_t{}); break;
    case DType::Int32: conv(std::int32_t{}); break;
    case DType::UInt32: conv(std::uint32_t{}); break;
    case DType::UInt8: conv(std::uint8_t{}); break;
    default: throw Error(code::unsupported, "the group reduce reads float and integer arrays; this one is " + std::string(dtype_to_string(r.dtype())));
  }
  return out;
}
std::vector<double> read_all(const Store& store, const std::string& key, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = ArrayReader::open(store, key);
  if (shape) *shape = r.shape();
  return widen(r, r.read());
}
std::vector<std::int32_t> read_i32(const Store& store, const std::string& key) {
  ArrayReader r = ArrayReader::open(store, key);
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<std::int32_t> out(raw.size() / 4);
  if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * 4);
  return out;
}
Json parse_obj(const Row& r, const char* col = "params_json") {
  const Json j = Json::parse(opt_text(r, col).value_or("{}"), nullptr, false);
  return j.is_object() ? j : Json::object();
}
std::optional<SurfaceInput> sphere_of(SubjectWriter& sub, const std::string& surface) {
  try { return read_surface(sub, surface + "_sphere", true); } catch (const Error& e) { if (e.code() == code::no_such_row) return std::nullopt; throw; }
}
std::string level_name(const std::string& surface, int l) { return surface + " group tiles L" + std::to_string(l); }
std::string safe(const std::string& s) {
  std::string o;
  for (char c : s) o += (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '-') ? c : '_';
  std::size_t a = o.find_first_not_of('_'), b = o.find_last_not_of('_');
  return a == std::string::npos ? std::string() : o.substr(a, b - a + 1);
}

// nxr's ATLAS_MEASURES (atlas_store.py): an array's name (relative to its statistics group) -> the measure it is; ARRAY_OF is the inverse (count is the map's `n`)
const std::vector<std::pair<std::string, std::string>>& atlas_measures() {
  static const std::vector<std::pair<std::string, std::string>> t = {
      {"w", "weight"}, {"s1", "sum"}, {"s2", "sum_square"}, {"n", "count"}, {"min", "min"}, {"max", "max"}, {"power", "power_sum"}, {"env", "envelope_sum"}, {"envmax", "envelope_max"}, {"samples", "count"},
      {"grad/w", "gradient_weight"}, {"grad/c1", "gradient_1"}, {"grad/c2", "gradient_2"}, {"grad/cn", "gradient_n"}, {"grad/c11", "gradient_11"}, {"grad/c22", "gradient_22"}, {"grad/c12", "gradient_12"},
      {"grad/mag", "gradient_magnitude"}, {"tensor/t11", "tensor_11"}, {"tensor/t22", "tensor_22"}, {"tensor/tnn", "tensor_nn"}, {"tensor/t12r", "tensor_12_re"}, {"tensor/t12i", "tensor_12_im"},
      {"tensor/t1nr", "tensor_1n_re"}, {"tensor/t1ni", "tensor_1n_im"}, {"tensor/t2nr", "tensor_2n_re"}, {"tensor/t2ni", "tensor_2n_im"}, {"tensor/total", "tensor_total"}};
  return t;
}
std::optional<std::string> measure_of(const std::string& rel) { for (const auto& [r, m] : atlas_measures()) if (r == rel) return m; return std::nullopt; }
std::optional<std::string> array_of(const std::string& measure) { for (const auto& [r, m] : atlas_measures()) if (m == measure && r != "samples") return r; return std::nullopt; }
// how a measure merges across members (nxr MERGE): extremes take the extreme, the rest add
enum class Op { Sum, Max, Min };
Op merge_of(const std::string& rel) {
  const auto m = measure_of(rel).value_or("");
  return (m == "max" || m == "envelope_max") ? Op::Max : m == "min" ? Op::Min : Op::Sum;
}
struct Acc {
  std::vector<std::int64_t> shape;
  std::vector<double> v;
};
void merge(std::map<std::string, Acc>& acc, const std::string& rel, const std::vector<double>& data, const std::vector<std::int64_t>& shape) {
  auto it = acc.find(rel);
  if (it == acc.end()) { acc[rel] = Acc{shape, data}; return; }
  if (it->second.shape != shape) invalid("the group reduce: " + rel + " has another shape in a member than in the others (the members do not share leaves or bands)");
  const Op op = merge_of(rel);
  for (std::size_t i = 0; i < data.size(); ++i) it->second.v[i] = op == Op::Sum ? it->second.v[i] + data[i] : op == Op::Max ? std::max(it->second.v[i], data[i]) : std::min(it->second.v[i], data[i]);
}

// the `--kernels` selector (nxr build.select_kernels): comma tokens, kinds AND, tokens of one kind OR
bool glob(const std::string& pat, const std::string& s, std::size_t p = 0, std::size_t i = 0) {
  for (; p < pat.size(); ++p) {
    if (pat[p] == '*') { for (std::size_t k = i; k <= s.size(); ++k) if (glob(pat, s, p + 1, k)) return true; return false; }
    if (i >= s.size()) return false;
    if (pat[p] == '?') { ++i; continue; }
    if (pat[p] == '[') {
      const std::size_t close = pat.find(']', p + 2);
      if (close == std::string::npos) { if (pat[p] != s[i]) return false; ++i; continue; }
      std::string set = pat.substr(p + 1, close - p - 1);
      bool neg = !set.empty() && (set[0] == '!' || set[0] == '^');
      if (neg) set.erase(0, 1);
      bool in = false;
      for (std::size_t k = 0; k < set.size(); ++k) in = in || (k + 2 < set.size() && set[k + 1] == '-' ? (s[i] >= set[k] && s[i] <= set[k + 2] ? (k += 2, true) : (k += 2, false)) : set[k] == s[i]);
      if (in == neg) return false;
      p = close; ++i; continue;
    }
    if (pat[p] != s[i]) return false;
    ++i;
  }
  return i == s.size();
}
struct KernelItem { std::string name, method, stamp, group; int n_orient; };
std::set<std::string> select_kernels(const std::vector<KernelItem>& items, const std::string& spec) {
  std::vector<std::string> toks;
  std::stringstream ss(spec);
  std::string t;
  while (std::getline(ss, t, ',')) {
    const auto a = t.find_first_not_of(" \t"), b = t.find_last_not_of(" \t");
    if (a == std::string::npos) continue;
    t = t.substr(a, b - a + 1);
    if (t != "all") toks.push_back(t);
  }
  auto has = [&](const char* x) { return std::find(toks.begin(), toks.end(), x) != toks.end(); };
  std::set<int> orients;
  if (has("constrained")) orients.insert(1);
  if (has("free")) orients.insert(3);
  std::vector<std::string> names;
  for (const auto& k : toks) if (k != "constrained" && k != "free" && k != "latest") names.push_back(k);
  std::vector<const KernelItem*> out;
  for (const KernelItem& k : items) {
    if (!orients.empty() && !orients.count(k.n_orient)) continue;
    if (!names.empty()) {
      bool any = false;
      for (const auto& n : names) any = any || n == k.method || n == k.stamp || n == k.name || glob(n, k.name);
      if (!any) continue;
    }
    out.push_back(&k);
  }
  if (has("latest")) {
    std::map<std::tuple<std::string, std::string, int>, const KernelItem*> newest;
    for (const KernelItem* k : out) {
      auto key = std::make_tuple(k->group, k->method, k->n_orient);
      auto it = newest.find(key);
      if (it == newest.end() || k->stamp > it->second->stamp) newest[key] = k;      // yymmdd_hhmm: the string order is the time order
    }
    std::vector<const KernelItem*> keep;
    for (const KernelItem* k : out) if (newest[std::make_tuple(k->group, k->method, k->n_orient)] == k) keep.push_back(k);
    out = keep;
  }
  std::set<std::string> names_out;
  for (const KernelItem* k : out) names_out.insert(k->name);
  return names_out;
}
}  // namespace

int group_default_depth(DatasetWriter& ds) {
  const auto tpl = ds.template_();
  if (!tpl) missing(ds.db().one("SELECT name FROM dataset WHERE id = ?", {V(ds.id())}) ? "the dataset has no default subject (`dataset create --templates`)" : "no default subject");
  const auto lad_row = ds.db().one("SELECT params_json FROM selection WHERE subject_id = ? AND type = 'ladder' AND json_extract(params_json, '$.atlas.default') = 1", {V(text(*tpl, "id"))});
  if (!lad_row) return 8;
  const Json p = parse_obj(*lad_row);
  return p.contains("atlas") && p["atlas"].is_object() ? p["atlas"].value("depth", 8) : 8;
}

GroupTilesResult group_tiles_compute(DatasetWriter& ds, SubjectWriter& sub, const GroupTilesOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (text(sub.row(), "kind") == "template") invalid(sub.name() + ": the default subject's group tree is its own ladder (`tiles compute`, made by `dataset create`)");
  const auto tpl = ds.template_();
  if (!tpl) missing("the dataset has no default subject: the group tree is its ladder");
  const SurfaceInput S = read_surface(sub, opt.surface, true);
  const int nV = static_cast<int>(S.n_vertices()), nF = static_cast<int>(S.n_faces());
  const std::string sname = S.name, mid = S.id, gname = sname + "_group_tiles";
  GroupTilesResult res;
  const auto have = sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND manifold_id = ? AND name = ? AND type = 'set' AND member_of_id IS NULL AND status = 'complete'", {V(sub.id()), V(mid), V(gname)});
  if (have) {
    if (!opt.replace) { res.existed = true; res.info = Json{{"surface", sname}, {"group_tiles", gname}, {"kept", true}}; return res; }
    std::set<std::string> nodes;
    const auto members = sub.db().all("SELECT * FROM selection WHERE member_of_id = ?", {V(text(*have, "id"))});
    for (const Row& m : members) { const Json p = parse_obj(m); if (p.contains("node")) nodes.insert(p["node"].get<std::string>()); sub.remove("selection", text(m, "id")); }
    sub.remove("selection", text(*have, "id"));
    for (const std::string& n : nodes) sub.store().erase_prefix(n);
  }
  // the template: its primary cortex, its registered sphere, its group ladder and the finest level's codes
  SubjectWriter tsub = SubjectWriter::open(ds, text(*tpl, "name"));
  const SurfaceInput T = read_surface(tsub, std::nullopt, true);
  const auto tsphere = sphere_of(tsub, T.name), ssphere = sphere_of(sub, sname);
  if (!tsphere) missing(text(*tpl, "name") + ": no registered sphere (" + T.name + "_sphere)");
  if (!ssphere) missing(sub.name() + ": no registered sphere (" + sname + "_sphere): the group tree is assigned through it");
  if (ssphere->n_vertices() != nV) invalid(sname + "_sphere has " + std::to_string(ssphere->n_vertices()) + " vertices, the surface " + std::to_string(nV));
  if (tsphere->n_vertices() != T.n_vertices()) invalid(T.name + "_sphere has another vertex count than " + T.name);
  const auto tl = tsub.db().one("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND type = 'ladder' AND status = 'complete'", {V(text(*tpl, "id")), V(T.name + " tiles")});
  if (!tl) missing(text(*tpl, "name") + ": no ladder `" + T.name + " tiles` (the group tree)");
  if (parse_obj(*tl).value("tree", std::string()) != "group") invalid(text(*tpl, "name") + ": `" + T.name + " tiles` is not a group tree (tree = " + parse_obj(*tl).value("tree", std::string("subject")) + ")");
  const std::vector<Row> tlevels = tsub.db().all("SELECT * FROM selection WHERE member_of_id = ? AND status = 'complete' ORDER BY member_ordinal", {V(text(*tl, "id"))});
  if (tlevels.empty()) invalid(text(*tpl, "name") + ": its group ladder has no levels");
  const int finest = static_cast<int>(tlevels.size()) - 1;
  const std::vector<std::int32_t> tcodes = read_i32(tsub.store(), text(tlevels.back(), "array_path"));
  if (static_cast<std::int64_t>(tcodes.size()) != T.n_vertices()) invalid(text(*tpl, "name") + ": its finest level does not fit its surface");
  std::vector<std::pair<int, int>> truns, sruns;
  try { truns = lad::pieceRuns(T.faces.data(), static_cast<int>(T.n_faces()), static_cast<int>(T.n_vertices())); sruns = lad::pieceRuns(S.faces.data(), nF, nV); }
  catch (const std::exception& e) { invalid(std::string("group tree: ") + e.what()); }
  say(Json{{"stage", "group tiles"}, {"surface", sname}, {"template", text(*tpl, "name")}, {"finest", finest}});
  double med = 0, mx = 0;
  std::vector<lad::Tree> trees;
  try {
    for (int l = 0; l <= finest; ++l) trees.push_back(lad::groupTreeFor(tsphere->vertices.data(), tcodes.data(), truns, finest, ssphere->vertices.data(), nV, sruns, l, l == finest ? &med : nullptr, l == finest ? &mx : nullptr));
  } catch (const std::invalid_argument& e) { invalid(sname + ": " + e.what()); }
  lad::Mesh mesh; mesh.n = nV; mesh.positions = S.vertices; mesh.faces = S.faces;
  const std::vector<double> area = lad::vertexAreas(mesh);
  const std::string by = "nxr-neuro pyramid build (group tiles)", producer = internal::producer_string("group tiles");
  sub.compose([&] {
    SelectionSpec top;
    top.name = gname; top.type = "set"; top.manifold_id = mid; top.no_elements = true;
    top.cols.set("n_members", static_cast<std::int64_t>(finest + 1)).set("n_elements", static_cast<std::int64_t>(nV))
        .set("description", "this subject's vertices on the dataset's GROUP tree (the default subject's tiles, through the registered sphere); member L<k> codes h 2^k + t")
        .json("params_json", Json{{"tree", "group"}, {"template", text(*tpl, "name")}, {"finest", finest}, {"sphere_match_median", med}, {"sphere_match_max", mx}, {"by", by}, {"producer", producer}});
    const std::string parent = sub.selection(std::move(top));
    for (int l = 0; l <= finest; ++l) {
      const std::vector<std::int32_t>& node = trees[static_cast<std::size_t>(l)].codes;
      const int nodes = trees[static_cast<std::size_t>(l)].nodes(l);
      const std::string name = level_name(sname, l), dir = atlas_node(sub.db(), mid, name);
      sub.write_array(dir + "/codes", view(node, {static_cast<std::int64_t>(nV)}));
      const std::vector<double> w = lad::tileAreas(node, nodes, area);
      sub.write_array(dir + "/w", view(w, {static_cast<std::int64_t>(nodes)}));
      SelectionSpec ls;
      ls.name = name; ls.type = "set"; ls.manifold_id = mid; ls.no_elements = true;
      ls.cols.set("member_of_id", parent).set("member_ordinal", static_cast<std::int64_t>(l)).set("code", static_cast<std::int64_t>(l)).set("n_elements", static_cast<std::int64_t>(nV)).set("n_members", static_cast<std::int64_t>(nodes))
          .set("array_path", dir + "/codes").set("dictionary_id", text(tlevels[static_cast<std::size_t>(l)], "dictionary_id"))
          .set("description", "level " + std::to_string(l) + " of the GROUP tree: the default subject's tile of the nearest vertex on the registered sphere, tile code h 2^" + std::to_string(l) + " + t")
          .json("params_json", Json{{"tree", "group"}, {"level", l}, {"tiles", nodes}, {"finest", finest}, {"node", dir}, {"areas", dir + "/w"}});
      sub.selection(std::move(ls));
    }
  });
  res.info = Json{{"surface", sname}, {"group_tiles", gname}, {"template", text(*tpl, "name")}, {"levels", finest + 1}, {"finest", finest}, {"sphere_match_median", med}, {"sphere_match_max", mx}, {"kept", false}};
  return res;
}

PyramidLeaves group_leaves(DatasetWriter& ds, SubjectWriter& sub, const SurfaceInput& S, std::optional<int> depth, const std::string& who, const std::function<void(const Json&)>& say) {
  const std::string gname = S.name + "_group_tiles";
  auto parent_of = [&]() { return sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND manifold_id = ? AND name = ? AND type = 'set' AND member_of_id IS NULL AND status = 'complete'", {V(sub.id()), V(S.id), V(gname)}); };
  if (!parent_of()) { GroupTilesOptions go; go.surface = S.name; if (say) say(Json{{"stage", "group tiles"}, {"surface", S.name}, {"made", true}}); group_tiles_compute(ds, sub, go, say); }
  const auto parent = parent_of();
  if (!parent) missing(sub.name() + ": the group tree of " + S.name + " was not stored");
  const Json pp = parse_obj(*parent);
  PyramidLeaves out;
  out.finest = pp.value("finest", 0);
  out.depth = depth ? *depth : group_default_depth(ds);
  if (out.depth < 0 || out.depth > out.finest) invalid(who + ": the group tree's depth " + std::to_string(out.depth) + " is outside the default subject's ladder (levels 0 \xe2\x80\xa6 " + std::to_string(out.finest) + ")");
  out.name = level_name(S.name, out.depth);
  out.part = sub.db().one("SELECT * FROM selection WHERE member_of_id = ? AND name = ? AND status = 'complete'", {V(text(*parent, "id")), V(out.name)});
  if (!out.part) invalid(sub.name() + ": the group tree of " + S.name + " has no level " + std::to_string(out.depth));
  out.record = parse_obj(*out.part);
  if (!out.record.contains("node") || !out.record.contains("areas")) invalid(out.name + ": its record lacks the node of its arrays");
  out.tree = "group";
  out.node = out.record["node"].get<std::string>();
  out.leaves = integer(*out.part, "n_members");
  out.codes = read_i32(sub.store(), text(*out.part, "array_path"));
  out.leaf_area = read_all(sub.store(), out.record["areas"].get<std::string>());
  if (static_cast<std::int64_t>(out.codes.size()) != S.n_vertices() || static_cast<std::int64_t>(out.leaf_area.size()) != out.leaves) invalid(out.name + ": its codes or areas do not fit the surface");
  return out;
}

namespace {
// THE REDUCTION, whatever the form of the dataset. `tds` is the writer the default subject can be read through (`tname`); `open_member(name)` gives a member's subject (and, in a
// package dataset, keeps that member's package open); `out` is where the group fields are written: in a one-database dataset the default subject itself (`own_target`), in a
// catalogue dataset the group package's subject, whose rows reach the default subject's through anchor rows (`anchor` carries each contribution's member rows into it, before
// anything is written). The maths, the checks and the stored rows are the same in both.
struct ReduceTarget {
  DatasetWriter* tds = nullptr;
  std::string tname;
  std::function<SubjectWriter(const std::string&)> open_member;
  std::function<Row(const std::string&)> member_row;
  SubjectWriter* out = nullptr;                      // null: the default subject itself
  std::function<void(const std::vector<std::pair<std::string, std::string>>&)> anchor;
};

PyramidReduceResult reduce_core(ReduceTarget& tg, const PyramidReduceOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  DatasetWriter& ds = *tg.tds;
  const std::string& tname = tg.tname;
  SubjectWriter tsub = SubjectWriter::open(ds, tname);
  SubjectWriter& dsub = tsub;                        // (the default subject: read here)
  SubjectWriter& wsub = tg.out ? *tg.out : tsub;     // where the group fields are written
  const bool own_target = tg.out == nullptr;
  const int D = opt.depth ? *opt.depth : group_default_depth(ds);

  // the definitions and the default subject's tile partition at depth D (the group's leaves)
  const SurfaceInput T = read_surface(dsub, std::nullopt, true);
  const auto dladder = dsub.db().one("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND type = 'ladder' AND status = 'complete'", {V(dsub.id()), V(T.name + " tiles")});
  if (!dladder) missing(tname + ": no ladder `" + T.name + " tiles`");
  const Json ladder_params = parse_obj(*dladder);
  const Json defs = ladder_params.contains("atlas") && ladder_params["atlas"].is_object() ? ladder_params["atlas"] : Json::object();
  const auto dpart = dsub.db().one("SELECT * FROM selection WHERE member_of_id = ? AND name = ? AND status = 'complete'", {V(text(*dladder, "id")), V(T.name + " tiles L" + std::to_string(D))});
  if (!dpart) missing(tname + ": no partition `" + T.name + " tiles L" + std::to_string(D) + "` (the default subject's tiles at depth " + std::to_string(D) + ")");
  const std::int64_t n_leaves = integer(*dpart, "n_members");
  const std::string part_manifold = text(*dpart, "manifold_id");
  std::vector<std::pair<double, double>> bands;
  if (defs.contains("bands_hz") && defs["bands_hz"].is_array()) for (const Json& b : defs["bands_hz"]) bands.emplace_back(b[0].get<double>(), b[1].get<double>());

  struct Member { Row subject; std::string name; SubjectWriter sub; Row part; std::vector<double> w; };
  std::vector<Member> members;
  for (const std::string& name : opt.subjects) {
    const Row row = tg.member_row(name);
    SubjectWriter sub = tg.open_member(name);
    // the member's group-tree partition at depth D, by its rows
    const auto part = sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND json_extract(params_json, '$.atlas.tree') = 'group' AND json_extract(params_json, '$.joint') IS NULL AND name LIKE '% group tiles L' || ? AND status = 'complete'", {V(sub.id()), V(std::to_string(D))});
    if (!part || !parse_obj(*part).contains("atlas") || !parse_obj(*part)["atlas"].contains("areas"))
      invalid("pyramid reduce: " + name + " has no group-tree atlas at depth " + std::to_string(D) + " (`nxr-neuro pyramid build --tree group` first)");
    if (integer(*part, "n_members") != n_leaves) invalid("pyramid reduce: " + name + "'s group tiles have " + std::to_string(integer(*part, "n_members")) + " leaves, the default subject's " + std::to_string(n_leaves) + " at depth " + std::to_string(D) + ": not the same tree");
    std::vector<double> w = read_all(sub.store(), parse_obj(*part)["atlas"]["areas"].get<std::string>());
    if (static_cast<std::int64_t>(w.size()) != n_leaves) invalid("pyramid reduce: " + name + "'s leaf areas do not fit its leaves");
    members.push_back(Member{row, name, std::move(sub), *part, std::move(w)});
  }

  std::map<std::string, std::map<std::string, Acc>> spatial, temporal;
  std::map<std::string, std::vector<std::pair<std::string, std::vector<double>>>> spatial_stack, temporal_stack;   // key -> (member, values)
  std::map<std::string, std::vector<std::int64_t>> temporal_stack_shape;
  std::map<std::string, double> samples_total;
  std::map<std::string, std::vector<Json>> sources;                         // "spatial|kind" / "time|key" -> contributions
  std::map<std::string, std::set<std::string>> kernels_of;
  std::map<std::string, std::string> units;
  std::set<std::string> left_out;
  const std::size_t nB = bands.size();

  for (Member& M : members) {
    SubjectWriter& sub = M.sub;
    Database& db = sub.db();
    const Json ref = Json{{"subject_id", text(M.subject, "id")}};
    // SPATIAL: the measurements on the partition, by the field's KIND
    const auto ms = db.all("SELECT m.*, f.kind AS fkind, f.unit AS funit FROM selection_measurement m JOIN field f ON f.id = m.of_field_id WHERE m.selection_id = ? AND m.array_path IS NOT NULL", {V(text(M.part, "id"))});
    std::map<std::pair<std::string, std::string>, std::vector<Row>> by_field;
    for (const Row& m : ms) by_field[{text(m, "of_field_id"), text(m, "fkind")}].push_back(m);
    for (const auto& [key, rows] : by_field) {
      const std::string kind = safe(key.second);
      std::map<std::string, std::pair<std::vector<double>, std::vector<std::int64_t>>> arrs;
      for (const Row& r : rows) {
        const auto rel = array_of(text(r, "measure"));
        if (!rel) { left_out.insert(text(r, "measure")); continue; }
        std::vector<std::int64_t> shape;
        arrs[*rel] = {read_all(sub.store(), text(r, "array_path"), &shape), shape};
      }
      if (!arrs.count("s1") || !arrs.count("w")) invalid("pyramid reduce: " + M.name + ": the map '" + key.second + "' has no weight and sum arrays on its group tiles");
      for (const auto& [rel, a] : arrs) merge(spatial[kind], rel, a.first, a.second);
      std::vector<double> mean(arrs.at("s1").first.size());
      for (std::size_t i = 0; i < mean.size(); ++i) { const double w = arrs.at("w").first[i]; mean[i] = w > 0 ? arrs.at("s1").first[i] / w : std::nan(""); }
      spatial_stack[kind].emplace_back(M.name, std::move(mean));
      Json c = ref; c["source_field_id"] = key.first;
      sources["spatial|" + kind].push_back(c);
      if (!rows.empty()) units[kind] = opt_text(rows.front(), "funit").value_or("");
    }
    // TIME: every source estimate's joint partition on the group tree
    struct JointItem { Row joint; Json prod; std::map<std::string, std::string> paths; std::string smp_path, rec, fid, fname; KernelItem k; };
    std::vector<JointItem> joints;
    for (const Row& j : db.all("SELECT x.*, f.producer_json AS fp, f.id AS fid, f.name AS fname, r.name AS rec FROM selection x JOIN field f ON f.id = x.of_field_id JOIN field r ON r.id = f.derived_from_id "
                               "WHERE x.subject_id = ? AND json_extract(x.params_json, '$.joint') IS NOT NULL AND json_extract(x.params_json, '$.atlas.tree') = 'group' AND x.status = 'complete'", {V(sub.id())})) {
      JointItem ji;
      ji.joint = j;
      ji.prod = Json::parse(opt_text(j, "fp").value_or("{}"), nullptr, false);
      if (!ji.prod.is_object()) ji.prod = Json::object();
      if (parse_obj(j)["joint"].value("space", std::string()) != text(M.part, "id")) continue;          // a joint on another depth of the group tree
      for (const Row& r : db.all("SELECT DISTINCT measure, array_path FROM selection_measurement WHERE selection_id = ? AND array_path IS NOT NULL", {V(text(j, "id"))})) ji.paths[text(r, "measure")] = text(r, "array_path");
      const auto smp = db.one("SELECT array_path FROM selection_measurement WHERE of_field_id = ? AND measure = 'count' AND array_path IS NOT NULL", {V(text(j, "fid"))});
      if (!smp) invalid("pyramid reduce: " + M.name + ": the estimate of " + text(j, "fname") + " has no per-frame sample count");
      ji.smp_path = text(*smp, "array_path");
      ji.rec = text(j, "rec"); ji.fid = text(j, "fid"); ji.fname = text(j, "fname");
      const std::string kn = ji.prod.contains("kernel") && ji.prod["kernel"].is_string() ? ji.prod["kernel"].get<std::string>() : ji.fname.substr(ji.fname.find(" \xc3\x97 ") == std::string::npos ? 0 : ji.fname.find(" \xc3\x97 ") + 3);
      const std::string orient = ji.prod.value("orientation", std::string());
      ji.k = KernelItem{kn, ji.prod.contains("method") && ji.prod["method"].is_string() ? ji.prod["method"].get<std::string>() : std::string(), ji.prod.contains("stamp") && ji.prod["stamp"].is_string() ? ji.prod["stamp"].get<std::string>() : std::string(), ji.rec, orient == "constrained" ? 1 : 3};
      joints.push_back(std::move(ji));
    }
    std::vector<KernelItem> items;
    for (const JointItem& ji : joints) items.push_back(ji.k);
    const std::set<std::string> chosen = select_kernels(items, opt.kernels);
    std::vector<JointItem> kept;
    for (JointItem& ji : joints) if (chosen.count(ji.k.name)) kept.push_back(std::move(ji));
    std::map<std::tuple<std::string, std::string, int>, std::string> seen;
    for (const JointItem& ji : kept) {
      auto [it, fresh] = seen.emplace(std::make_tuple(ji.rec, ji.k.method, ji.k.n_orient), ji.k.name);
      if (!fresh && it->second != ji.k.name)
        invalid("pyramid reduce: " + M.name + ": " + ji.rec + " has two " + ji.k.method + " (" + (ji.k.n_orient == 1 ? "constrained" : "free") + ") kernels, " + it->second + " and " + ji.k.name + " \xe2\x80\x94 they would sum into one group atlas; choose one with --kernels (a stamp, a name, or latest)");
    }
    std::map<std::string, std::pair<std::vector<double>, double>> per_sub;      // key -> (the total, the samples) summed over this member's joints
    std::map<std::string, std::vector<std::int64_t>> per_sub_shape;
    for (const JointItem& ji : kept) {
      // the member's bands must be the definitions' bank (a band's code is its index)
      const auto wr = db.one("SELECT DISTINCT within_selection_id FROM selection_measurement WHERE selection_id = ? AND within_selection_id IS NOT NULL", {V(text(ji.joint, "id"))});
      if (wr) {
        const auto spans = db.all("SELECT start, stop, code FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {V(text(*wr, "within_selection_id"))});
        bool same = spans.size() == nB;
        for (std::size_t b = 0; same && b < spans.size(); ++b) same = std::abs(std::stod(text(spans[b], "start")) - bands[b].first) < 1e-9 && std::abs(std::stod(text(spans[b], "stop")) - bands[b].second) < 1e-9 && integer(spans[b], "code") == static_cast<std::int64_t>(b);
        if (!same) invalid("pyramid reduce: " + M.name + ": " + ji.fname + " is measured in other frequency bands than the dataset's definitions (a band's code is its index in the bank)");
      }
      const std::string task_key = [&] {
        const std::size_t at = ji.rec.find("task-");
        std::string task = "rec";
        if (at != std::string::npos) { std::size_t e = at + 5; while (e < ji.rec.size() && std::isalnum(static_cast<unsigned char>(ji.rec[e]))) ++e; if (e > at + 5) task = ji.rec.substr(at + 5, e - at - 5); }
        return task + "__" + (ji.k.method.empty() ? safe(ji.k.name) : ji.k.method) + "__" + (ji.k.n_orient == 1 ? "constrained" : "free");
      }();
      kernels_of[task_key].insert(M.name + ": " + ji.k.name);
      std::map<std::string, std::pair<std::vector<double>, std::vector<std::int64_t>>> arrs;
      for (const auto& [measure, path] : ji.paths) {
        const auto rel = array_of(measure);
        if (!rel) { left_out.insert(measure); continue; }
        ArrayReader r = ArrayReader::open(sub.store(), path);
        const auto& sh = r.shape();
        if (sh.size() != 3 || sh[0] != static_cast<std::int64_t>(nB) || sh[1] != n_leaves) invalid("pyramid reduce: " + M.name + ": " + path + " is not [" + std::to_string(nB) + " bands, " + std::to_string(n_leaves) + " leaves, frames]");
        const bool maxop = merge_of(*rel) == Op::Max;
        std::vector<double> collapsed(static_cast<std::size_t>(sh[0] * sh[1]));
        for (std::int64_t b = 0; b < sh[0]; ++b) {                       // one band at a time: [leaves x frames], collapsed at once
          const std::int64_t b_next = b + 1;                              // a stored node's band slice, read by name (not a recording)
          const std::vector<double> a = widen(r, r.read({b, 0, 0}, {b_next, sh[1], sh[2]}));
          for (std::int64_t l = 0; l < sh[1]; ++l) {
            const double* row = a.data() + static_cast<std::size_t>(l * sh[2]);
            double acc = maxop ? -std::numeric_limits<double>::infinity() : 0.0;
            for (std::int64_t f = 0; f < sh[2]; ++f) acc = maxop ? std::max(acc, row[f]) : acc + row[f];
            collapsed[static_cast<std::size_t>(b * sh[1] + l)] = acc;
          }
        }
        arrs[*rel] = {std::move(collapsed), {sh[0], sh[1]}};
      }
      const std::string total_rel = arrs.count("power") ? "power" : "tensor/total";
      if (!arrs.count(total_rel)) invalid("pyramid reduce: " + M.name + ": " + ji.fname + " has neither a band power nor an orientation tensor's total to take a per-subject density from");
      for (const auto& [rel, a] : arrs) merge(temporal[task_key], rel, a.first, a.second);
      const std::vector<double> smpv = read_all(sub.store(), ji.smp_path);
      double smp = 0;
      for (double x : smpv) smp += x;
      samples_total[task_key] += smp;
      auto& cur = per_sub[task_key];
      if (cur.first.empty()) { cur.first = arrs.at(total_rel).first; cur.second = smp; }
      else { for (std::size_t i = 0; i < cur.first.size(); ++i) cur.first[i] += arrs.at(total_rel).first[i]; cur.second += smp; }
      Json c = ref; c["source_field_id"] = ji.fid;
      sources["time|" + task_key].push_back(c);
      temporal_stack_shape[task_key] = {static_cast<std::int64_t>(nB), n_leaves};
    }
    for (auto& [key, ts] : per_sub) {
      std::vector<double> dens(ts.first.size());
      for (std::size_t b = 0; b < nB; ++b)
        for (std::int64_t l = 0; l < n_leaves; ++l) {
          const double w = M.w[static_cast<std::size_t>(l)];
          dens[static_cast<std::size_t>(b * n_leaves + l)] = ts.first[static_cast<std::size_t>(b * n_leaves + l)] / ((w > 0 ? w : std::nan("")) * ts.second);
        }
      temporal_stack[key].emplace_back(M.name, std::move(dens));
    }
  }

  // ── store: the group fields' rows and arrays in the default subject, as nxr's reduce writes them ──
  const Json part_params = parse_obj(*dpart);
  const std::string node = part_params.contains("node") ? part_params["node"].get<std::string>() : atlas_node(dsub.db(), part_manifold, text(*dpart, "name"));
  const std::string part_id = text(*dpart, "id");
  const auto frames_row = dsub.db().one("SELECT id FROM field WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(dsub.id()), V(T.name + "_canonical_frames")});
  Json spatial_out = Json::array(), time_out = Json::array();
  if (tg.anchor) {   // a group package: the members' rows the contributions name are carried into it first
    std::set<std::pair<std::string, std::string>> pairs;
    for (const auto& [key, list] : sources) for (const Json& c : list) pairs.insert({c["subject_id"].get<std::string>(), c["source_field_id"].get<std::string>()});
    tg.anchor(std::vector<std::pair<std::string, std::string>>(pairs.begin(), pairs.end()));
  }
  wsub.compose([&] {
    // a re-run REPLACES: the group fields (their rows go with them), and the arrays
    for (const Row& r : wsub.db().all("SELECT id FROM field WHERE subject_id = ? AND path IS NULL AND json_extract(producer_json, '$.atlas') IN ('spatial', 'time')", {V(wsub.id())})) wsub.remove("field", text(r, "id"));
    wsub.store().erase_prefix(node + "/maps");
    wsub.store().erase_prefix(node + "/time");
    Json p = part_params;
    Json a = p.contains("atlas") && p["atlas"].is_object() ? p["atlas"] : Json::object();
    a["tree"] = "group"; a["level"] = D; a["node"] = node;
    if (!a.contains("finest")) a["finest"] = ladder_params.value("finest", D);
    if (!a.contains("areas")) a["areas"] = part_params.value("areas", node + "/w");
    a["frame"] = frames_row ? Json(text(*frames_row, "id")) : (a.contains("frame") ? a["frame"] : Json(nullptr));   // the canonical frames' field (G6: "frame")
    a.erase("gauge");
    Json names = Json::array();
    for (const Member& m : members) names.push_back(m.name);
    a["members"] = names;
    p["atlas"] = a;
    if (own_target) wsub.db().execute("UPDATE selection SET params_json = ? WHERE id = ?", {V(jdump(p)), V(part_id)});   // (in a group package the partition is the default subject's own row, held as an anchor: left as it is)
    const std::string bands_id = (nB > 0 && !temporal.empty()) ? pyramid_bands_selection(wsub, bands) : std::string();

    auto group_field = [&](const std::string& name, const std::string& kind, const std::optional<std::string>& unit, const Json& producer, const std::vector<Json>& contributions) {
      FieldSpec fs;
      fs.name = name; fs.kind = kind; fs.manifold_id = part_manifold; fs.value_type = "scalar";
      fs.cols.set("data_type", "float32").set("function", "sum").set("description", "the group sum over the dataset's members \xe2\x80\x94 its reductions only (D130)");
      if (unit && !unit->empty()) fs.cols.set("unit", *unit);
      Json prod = producer;
      Json foreign = Json::array();
      for (const Json& c : contributions) {
        if (c.contains("dataset")) { foreign.push_back(c); continue; }
        ContributionSpec cs; cs.subject_id = c["subject_id"].get<std::string>(); cs.source_field_id = c["source_field_id"].get<std::string>();
        fs.contributions.push_back(cs);
      }
      if (!foreign.empty()) prod["sources"] = foreign;
      fs.cols.json("producer_json", prod);
      return wsub.field(std::move(fs));
    };
    auto put = [&](const std::string& path, const std::vector<double>& v, const std::vector<std::int64_t>& shape) { wsub.write_array(path, view(v, shape)); };
    auto stack = [&](const std::vector<std::pair<std::string, std::vector<double>>>& st, std::vector<std::int64_t> inner) {
      std::vector<double> out;
      for (const auto& [n, v] : st) out.insert(out.end(), v.begin(), v.end());
      inner.insert(inner.begin(), static_cast<std::int64_t>(st.size()));
      return std::make_pair(out, inner);
    };
    for (auto& [kind, arrs] : spatial) {
      const std::string g = node + "/maps/" + kind;
      for (const auto& [rel, a] : arrs) put(g + "/" + rel, a.v, a.shape);
      Json subs = Json::array();
      for (const auto& s : spatial_stack[kind]) subs.push_back(s.first);
      const auto [sv, ss] = stack(spatial_stack[kind], {n_leaves});
      put(g + "/per_subject/mean", sv, ss);
      const std::string un = units.count(kind) ? units[kind] : std::string();
      const std::string fid = group_field("group " + kind, "group map", un, Json{{"atlas", "spatial"}, {"key", kind}, {"subjects", subs}}, sources["spatial|" + kind]);
      for (const auto& [rel, a] : arrs) {
        const auto m = measure_of(rel);
        if (!m) continue;
        Cols c;
        c.set("measure", *m).set("of_field_id", fid).set("array_path", g + "/" + rel).set("computed_by", kBy);
        const std::string unit = rel.rfind("grad/", 0) == 0 ? (un.empty() ? std::string("value") : un) + "/m" : un;
        if (!unit.empty()) c.set("unit", unit);
        wsub.measurement(part_id, c);
      }
      spatial_out.push_back(Json{{"kind", kind}, {"field", fid}, {"subjects", subs}});
    }
    for (auto& [key, arrs] : temporal) {
      const std::string g = node + "/time/" + key;
      for (const auto& [rel, a] : arrs) put(g + "/" + rel, a.v, a.shape);
      put(g + "/samples", {samples_total[key]}, {1});
      Json subs = Json::array();
      for (const auto& s : temporal_stack[key]) subs.push_back(s.first);
      const auto [dv, ds_] = stack(temporal_stack[key], temporal_stack_shape[key]);
      put(g + "/per_subject/density", dv, ds_);
      const std::size_t p1 = key.find("__"), p2 = key.find("__", p1 + 2);
      Json kern = Json::array();
      for (const std::string& k : kernels_of[key]) kern.push_back(k);
      const std::string fid = group_field("group " + key, "source estimate", std::nullopt,
                                          Json{{"atlas", "time"}, {"condition", key.substr(0, p1)}, {"method", key.substr(p1 + 2, p2 - p1 - 2)}, {"orientation", key.substr(p2 + 2)},
                                               {"collapsed", "frames and runs summed: subjects share no clock"}, {"subjects", subs}, {"kernels", kern}, {"kernel_selector", opt.kernels}},
                                          sources["time|" + key]);
      for (const auto& [rel, a] : arrs) {
        const auto m = measure_of(rel);
        if (!m) continue;
        for (std::size_t b = 0; b < nB; ++b) {
          Cols c;
          c.set("measure", *m).set("of_field_id", fid).set("array_path", g + "/" + rel).set("within_selection_id", bands_id).set("within_code", static_cast<std::int64_t>(b)).set("computed_by", kBy);
          wsub.measurement(part_id, c);
        }
      }
      time_out.push_back(Json{{"key", key}, {"field", fid}, {"subjects", subs}, {"samples", samples_total[key]}});
    }
  });
  Json lo = Json::array();
  for (const std::string& m : left_out) lo.push_back(m);
  PyramidReduceResult res;
  res.info = Json{{"members", members.size()}, {"subjects", opt.subjects}, {"depth", D}, {"leaves", n_leaves}, {"node", node}, {"spatial", spatial_out}, {"time", time_out}, {"left_out_measures", lo}, {"kernels", opt.kernels}};
  say(Json{{"stage", "reduced"}, {"members", members.size()}, {"spatial", spatial.size()}, {"time", temporal.size()}});
  return res;
}
}  // namespace

PyramidReduceResult pyramid_reduce(DatasetWriter& ds, const PyramidReduceOptions& opt, const std::function<void(const Json&)>& progress) {
  if (opt.subjects.empty()) invalid("pyramid reduce: --subjects names the members (a cohort; never all by default)");
  {
    std::set<std::string> uniq(opt.subjects.begin(), opt.subjects.end());
    if (uniq.size() != opt.subjects.size()) invalid("pyramid reduce: a subject is named twice");
  }
  const auto tpl = ds.template_();
  if (!tpl) missing("pyramid reduce: the dataset has no default subject (the group lives in its store)");
  const std::string tname = text(*tpl, "name");
  for (const std::string& n : opt.subjects) if (n == tname) invalid("pyramid reduce: the default subject is where the group is stored, not a member");
  ReduceTarget tg;
  tg.tds = &ds;
  tg.tname = tname;
  tg.member_row = [&](const std::string& name) {
    const auto row = ds.db().one("SELECT * FROM subject WHERE dataset_id = ? AND name = ? AND status = 'complete'", {V(ds.id()), V(name)});
    if (!row) missing("pyramid reduce: no subject '" + name + "' in the dataset");
    return *row;
  };
  tg.open_member = [&](const std::string& name) { return SubjectWriter::open(ds, name); };
  return reduce_core(tg, opt, progress);
}

PyramidReduceResult pyramid_reduce_group(const std::filesystem::path& dataset, const PyramidReduceOptions& opt, const std::string& group_name, const nxr::io::TemplateSearch& templates,
                                         const std::function<void(const Json&)>& progress) {
  if (opt.subjects.empty()) invalid("pyramid reduce: --subjects names the members (a cohort; never all by default)");
  {
    std::set<std::string> uniq(opt.subjects.begin(), opt.subjects.end());
    if (uniq.size() != opt.subjects.size()) invalid("pyramid reduce: a subject is named twice");
  }
  if (group_name.empty()) invalid("pyramid reduce: --into names the group package the result goes to (one per analysis; never a default one)");
  nxr::io::GroupAnalysis ga = nxr::io::GroupAnalysis::open(dataset, group_name, opt.subjects, templates, "nxr-neuro pyramid reduce");
  PyramidReduceResult res;
  {
    SubjectWriter gsub = SubjectWriter::open(ga.group(), group_name);
    ReduceTarget tg;
    tg.tds = &ga.group();                          // the default subject's rows are in the group package, as anchors; its arrays are read from its own package
    tg.tname = ga.template_name();
    tg.out = &gsub;
    tg.member_row = [&](const std::string& name) {
      DatasetWriter& w = ga.member(name);
      const auto row = w.db().one("SELECT * FROM subject WHERE name = ? AND status = 'complete' AND kind = 'subject'", {V(name)});
      if (!row) missing("pyramid reduce: no subject '" + name + "' in the dataset");
      return *row;
    };
    tg.open_member = [&](const std::string& name) { return SubjectWriter::open(ga.member(name), name); };
    tg.anchor = [&](const std::vector<std::pair<std::string, std::string>>& p) { ga.anchor_contributions(p); };
    res = reduce_core(tg, opt, progress);
    gsub.complete();
  }
  res.info["group"] = group_name;
  res.info["registered"] = ga.finish();
  return res;
}

}  // namespace nxr::pipelines
