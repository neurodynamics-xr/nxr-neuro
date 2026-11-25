// Export of BIDS-derivatives: see nxr/exporters/bids.h.
#include "nxr/exporters/bids.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <regex>
#include <set>

#include "cortex.h"
#include "nxr/exporters/fif.h"
#include "nxr/exporters/gifti.h"
#include "nxr/exporters/tsv.h"
#include "nxr/io/errors.h"

namespace nxr::exporters {

namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::Error;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }

// a BIDS label: letters and digits only
std::string label_of(const std::string& s) {
  std::string out;
  for (const char c : s) if (std::isalnum(static_cast<unsigned char>(c))) out += c;
  return out;
}

Json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) return Json();
  const Json j = Json::parse(f, nullptr, false);
  return j.is_discarded() ? Json() : j;
}

// the sources of an export as BIDS URIs: the inputs imported from a BIDS dataset (origin `bids:<path>`)
Json sources_of(const Json& prov) {
  std::set<std::string> uris;
  if (prov.contains("terms") && prov["terms"].contains("origins"))
    for (const auto& o : prov["terms"]["origins"])
      if (o.contains("source") && o["source"].is_string() && o["source"].get<std::string>().rfind("bids:", 0) == 0) {
        std::string path = o["source"].get<std::string>().substr(5);
        const auto hash = path.find('#');
        if (hash != std::string::npos) path = path.substr(0, hash);
        uris.insert("bids::" + path);
      }
  Json out = Json::array();
  for (const auto& u : uris) out.push_back(u);
  return out;
}

std::string strip(std::string name, const std::vector<std::string>& exts) {
  for (const auto& e : exts) if (name.size() > e.size() && name.compare(name.size() - e.size(), e.size(), e) == 0) return name.substr(0, name.size() - e.size());
  return name;
}

}  // namespace

Json export_bids_derivatives(const fs::path& package, const BidsDerivativesExport& o) {
  if (o.root.empty()) refuse("name the derivatives folder to write");
  std::regex pattern;
  try { pattern = std::regex(o.results); } catch (const std::regex_error& e) { refuse(std::string("the results pattern is not a regular expression: ") + e.what()); }
  if (o.results.empty() && o.measurements.empty()) refuse("name what to export: a pattern over the names of fields, labellings and statistics (--results) or a measurements table (--measurements)");
  const nxr::io::SubjectPackage pkg = nxr::io::SubjectPackage::open(package);
  const task::Subject subject = task::Subject::of(pkg);
  const nxr::io::PackageRecord rec = pkg.record();

  std::string space;
  if (o.space) space = label_of(*o.space);
  if (space.empty()) refuse("no space was named: name the `space-` entity (--space); a subject names no template (format 56)");
  const std::string sub = label_of(rec.name);
  if (sub.empty()) refuse("the subject's name '" + rec.name + "' holds no letter or digit for the `sub-` entity");

  std::string entities = "sub-" + sub;
  if (o.session) entities += "_ses-" + label_of(*o.session);
  if (o.task) entities += "_task-" + label_of(*o.task);
  if (o.run) entities += "_run-" + label_of(*o.run);
  entities += "_space-" + space;
  fs::path folder = fs::path(o.root) / ("sub-" + sub);
  if (o.session) folder /= "ses-" + label_of(*o.session);
  fs::create_directories(folder);

  // ---- what matches, by kind
  struct Item { task::Node node; std::string kind; };    // kind: map · series · dseg · stat
  std::vector<Item> items;
  Json skipped = Json::array();
  auto skip = [&](const std::string& name, const std::string& why) { skipped.push_back(Json{{"name", name}, {"reason", why}}); };
  for (const task::Node& f : subject.find_all({.cls = task::Cls::field})) {
    if (!std::regex_match(f.name, pattern)) continue;
    if (!f.path || !f.manifold_id) { skip(f.name, "stores no values"); continue; }
    const auto factors = subject.factors(*f.manifold_id);
    const auto first = factors.empty() ? std::nullopt : subject.by_id(task::Cls::manifold, factors[0]);
    if (!first || !first->type || *first->type != "surface" || factors.size() > 2) { skip(f.name, "is not a field on a cortical surface"); continue; }
    const std::string vt = f.value_type.value_or("scalar");
    if (vt != "scalar" && vt != "integer") { skip(f.name, "is " + vt + ": only scalar fields are written"); continue; }
    items.push_back({f, factors.size() == 2 ? "series" : "map"});
  }
  for (const task::Node& s : subject.find_all({.cls = task::Cls::selection})) {
    if (!std::regex_match(s.name, pattern)) continue;
    if (!s.type || *s.type != "set" || !s.array_path || !s.manifold_id) continue;     // spans and members of other kinds are not labellings
    const auto m = subject.by_id(task::Cls::manifold, *s.manifold_id);
    if (!m || !m->type || *m->type != "surface") continue;
    items.push_back({s, "dseg"});
  }
  for (const task::Node& st : subject.find_all({.cls = task::Cls::statistic})) if (std::regex_match(st.name, pattern) && st.path) items.push_back({st, "stat"});

  std::map<std::string, std::string> seen;                // label -> kind:name, to refuse two names that reduce to one label
  for (const auto& it : items) {
    const std::string key = label_of(it.node.name) + "/" + it.kind;
    if (label_of(it.node.name).empty()) refuse("'" + it.node.name + "' has no letter or digit for its `desc-` label");
    const auto [pos, fresh] = seen.insert({key, it.node.name});
    if (!fresh) refuse("'" + it.node.name + "' and '" + pos->second + "' reduce to the same `desc-` label '" + label_of(it.node.name) + "'; export them separately");
  }
  for (const auto& it : items) if (it.kind == "series" && !o.range) refuse("'" + it.node.name + "' is a series over time: state the time range (from and to in seconds, or all)");

  Json files = Json::array(), sidecars = Json::array(), all_sources = Json::array();
  std::set<std::string> source_set;
  std::vector<Json> terms_seen;
  auto note = [&](const Json& prov) {
    for (const auto& u : sources_of(prov)) source_set.insert(u.get<std::string>());
    terms_seen.push_back(prov["terms"]);
  };
  auto write_bids_sidecar = [&](const std::string& path, const Json& prov, const std::string& what) {
    Json side = Json{{"Description", what}, {"GeneratedBy", Json::array({prov})}};
    const Json src = sources_of(prov);
    if (!src.empty()) side["Sources"] = src;
    write_json(path, side);
    if (std::find(sidecars.begin(), sidecars.end(), Json(path)) == sidecars.end()) sidecars.push_back(path);
  };
  auto put_files = [&](const Json& r) { for (const auto& f : r["files"]) files.push_back(f); };

  for (const auto& it : items) {
    const std::string d = label_of(it.node.name);
    auto name_of = [&](const std::string& hemi, const std::string& suffix) { return entities + (hemi.empty() ? "" : "_hemi-" + hemi) + "_desc-" + d + "_" + suffix; };
    if (it.kind == "stat") {
      TsvExport t;
      t.statistic = it.node.name; t.out = (folder / name_of("", "stat")).string(); t.ctx = o.ctx;
      const Json r = export_tsv(package, t);
      put_files(r);
      Json side = read_json(r["sidecar"].get<std::string>());
      const Json prov = side["GeneratedBy"][0];
      note(prov);
      if (!sources_of(prov).empty()) { side["Sources"] = sources_of(prov); write_json(r["sidecar"].get<std::string>(), side); }
      sidecars.push_back(r["sidecar"]);
      continue;
    }
    // is the cortex split into hemispheres?
    const task::Node surf_node = [&] {
      const task::Node& n = it.node;
      return subject.require_id(task::Cls::manifold, it.kind == "dseg" ? *n.manifold_id : subject.factors(*n.manifold_id)[0]);
    }();
    bool split = false;
    {
      const auto whole = detail::load_surface(pkg, subject, surf_node);
      if (whole.hemi.empty()) {
        try { split = detail::hemispheres_of(pkg, subject, surf_node).size() == 2; } catch (const Error&) { split = false; }
      }
    }
    const std::string suffix = it.kind;
    GiftiExport g;
    if (it.kind == "dseg") g.labelling = it.node.name; else g.field = it.node.name;
    g.hemispheres = split;
    g.surface = o.with_surface && it.kind != "dseg";
    if (it.kind == "series") g.range = o.range;
    g.out = (folder / name_of("", suffix)).string();
    g.hemisphere_stem = [&, suffix](const std::string& h) { return (folder / name_of(h, suffix)).string(); };
    g.ctx = o.ctx;
    const Json r = export_gifti(package, g);
    put_files(r);
    const Json prov = read_json(r["sidecar"].get<std::string>());
    note(prov);
    const std::string what = std::string(it.kind == "dseg" ? "the labelling " : it.kind == "series" ? "the series " : "the map ") + it.node.name + " on " + surf_node.name;
    for (const auto& f : r["files"]) {
      const std::string path = f.get<std::string>();
      write_bids_sidecar(strip(path, {".func.gii", ".shape.gii", ".label.gii", ".surf.gii"}) + ".json", prov, what);
    }
    fs::remove(r["sidecar"].get<std::string>());
    if (o.stc && it.kind != "dseg") {
      if (!split) refuse("--stc needs the cortex's two hemispheres stored as components; '" + surf_node.name + "' has none");
      StcExport e;
      e.field = it.node.name; e.out = (folder / name_of("", "stc")).string(); e.ctx = o.ctx;
      if (it.kind == "series") e.range = o.range;
      const Json re = export_stc(package, e);
      put_files(re);
      const Json pe = read_json(re["sidecar"].get<std::string>());
      note(pe);
      write_bids_sidecar(e.out + ".json", pe, "the source estimate " + it.node.name + " (MNE -lh.stc and -rh.stc)");
      fs::remove(re["sidecar"].get<std::string>());
    }
  }
  if (!o.measurements.empty()) {
    std::vector<std::string> parts;
    std::string cur;
    for (const char c : o.measurements + ":") { if (c == ':') { parts.push_back(cur); cur.clear(); } else cur += c; }
    if (parts.size() != 3 || parts[0].empty() || parts[1].empty() || parts[2].empty()) refuse("--measurements is PARTITION:FIELD:measure,measure");
    TsvExport t;
    t.partition = parts[0]; t.field = parts[1];
    cur.clear();
    for (const char c : parts[2] + ",") { if (c == ',') { if (!cur.empty()) t.measures.push_back(cur); cur.clear(); } else cur += c; }
    t.out = (folder / (entities + "_desc-" + label_of(parts[0]) + "_measures")).string();
    t.ctx = o.ctx;
    const Json r = export_tsv(package, t);
    put_files(r);
    Json side = read_json(r["sidecar"].get<std::string>());
    const Json prov = side["GeneratedBy"][0];
    note(prov);
    if (!sources_of(prov).empty()) { side["Sources"] = sources_of(prov); write_json(r["sidecar"].get<std::string>(), side); }
    sidecars.push_back(r["sidecar"]);
  }
  if (files.empty()) refuse("nothing matched '" + o.results + "'" + (skipped.empty() ? std::string() : " that can be exported (" + std::to_string(skipped.size()) + " matched and were skipped, see `skipped`)"));

  // ---- dataset_description.json: merged with what a previous run left
  const std::string desc_path = (fs::path(o.root) / "dataset_description.json").string();
  Json d = read_json(desc_path);
  if (!d.is_object()) d = Json::object();
  d["Name"] = "nxr-neuro derivatives";
  d["BIDSVersion"] = "1.9.0";
  d["DatasetType"] = "derivative";
  Json gen = d.contains("GeneratedBy") && d["GeneratedBy"].is_array() ? d["GeneratedBy"] : Json::array();
  bool have = false;
  for (const auto& g : gen) if (g.contains("Name") && g["Name"] == o.ctx.tool && g.contains("Version") && g["Version"] == o.ctx.version) have = true;
  if (!have) gen.push_back(Json{{"Name", o.ctx.tool}, {"Version", o.ctx.version}, {"CodeURL", "https://github.com/neurodynamics-xr/nxr-neuro"}});
  d["GeneratedBy"] = gen;
  std::set<std::string> src = source_set;
  if (d.contains("SourceDatasets") && d["SourceDatasets"].is_array()) for (const auto& s : d["SourceDatasets"]) if (s.contains("URL") && s["URL"].is_string()) src.insert(s["URL"].get<std::string>());
  Json srcs = Json::array();
  for (const auto& s : src) srcs.push_back(Json{{"URL", s}});
  if (!srcs.empty()) d["SourceDatasets"] = srcs;
  // the licence: the ShareAlike one when an input carries it (two different ones cannot meet), else the common one, else the sidecars say
  std::set<std::string> lic;
  if (d.contains("License") && d["License"].is_string()) lic.insert(d["License"].get<std::string>());
  for (const auto& t : terms_seen) lic.insert(t["licence"].get<std::string>());
  std::set<std::string> sa;
  for (const auto& l : lic) if (is_share_alike(l)) sa.insert(l);
  if (sa.size() > 1) refuse("the derivatives folder would combine different ShareAlike licences; use another folder");
  d["License"] = !sa.empty() ? *sa.begin() : lic.size() == 1 ? *lic.begin() : std::string("see the sidecar of each file");
  d["NxrNeuro"] = Json{
      {"Suffixes", Json{{"map", "a scalar map on a cortical surface (.func.gii, one file per hemisphere)"}, {"series", "a scalar series on a cortical surface over a stated time range (.func.gii; one array per frame, time in `nxr-neuro:time`)"},
                        {"dseg", "a labelling of the cortex (.label.gii, with the dictionary's names and colours)"}, {"stc", "a source estimate as MNE reads it (-lh.stc and -rh.stc)"},
                        {"surf", "the cortical surface a map or series lies on (.surf.gii)"}, {"stat", "a stored statistic in long form (.tsv)"}, {"measures", "stored measurements of a partition's members (.tsv)"}}},
      {"DescLabel", "desc-<name> is the stored node's name reduced to letters and digits"},
      {"Note", "these suffixes are not in the BIDS standard; each file's sidecar carries its provenance under GeneratedBy"}};
  write_json(desc_path, d);
  return Json{{"root", o.root}, {"files", files}, {"sidecars", sidecars}, {"description", desc_path}, {"skipped", skipped}};
}

}  // namespace nxr::exporters
