// nxr-pipelines — the stored rows scale selection reads (scale_inputs.h). Reading only.
#include "nxr/io/subject_api.h"
#include "nxr/pipelines/scale_inputs.h"
#include "nxr/io/selection_reads.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/subject_reads.h"

#include <cstring>
#include <set>

#include "nxr/io/array_reader.h"
#include "nxr/io/statistic.h"

namespace nxr::pipelines {

using namespace nxr::io;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }

template <class T> std::vector<T> read_array(SubjectWriter& sub, const std::string& path, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = ArrayReader::open(sub.store(), path);
  if (shape) *shape = r.shape();
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<T> out(raw.size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}
// the cells' amplitude is stored in the precision of the input (float32 or float64): read as double
std::vector<double> read_doubles(SubjectWriter& sub, const std::string& path, std::vector<std::int64_t>* shape) {
  ArrayReader r = ArrayReader::open(sub.store(), path);
  if (shape) *shape = r.shape();
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<double> out;
  if (r.dtype() == DType::Float64) { out.resize(raw.size() / 8); if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * 8); }
  else if (r.dtype() == DType::Float32) { out.resize(raw.size() / 4); for (std::size_t i = 0; i < out.size(); ++i) { float x; std::memcpy(&x, raw.data() + 4 * i, 4); out[i] = x; } }
  else invalid("scaleselection: " + path + " holds neither float32 nor float64 values");
  return out;
}
std::string list_of(const std::vector<int>& v) { std::string s; for (const int x : v) s += (s.empty() ? "" : ",") + std::to_string(x); return s; }
}  // namespace

Json product_facts(const Json& params, const std::function<std::optional<Row>(const std::string&)>& selection, const std::function<std::optional<Row>(const std::string&)>& manifold, const std::string& who) {
  if (params.is_object() && params.contains("joint")) return params["joint"];   // a joint record made before the product selections
  auto bad = [&](const std::string& why) { invalid(who + " is not a product of a surface level and a time grid (`nxr-neuro run select_product` first): " + why); };
  if (!params.is_object() || !params.contains("product") || !params["product"].is_array() || params["product"].size() != 2) bad("it has no two factors");
  const Json& fs = params["product"][0];
  const Json& ft = params["product"][1];
  const auto space = fs.contains("selection") ? selection(fs["selection"].get<std::string>()) : std::nullopt;
  const auto time = ft.contains("selection") ? selection(ft["selection"].get<std::string>()) : std::nullopt;
  if (!space || !time || !fs.contains("level")) bad("its first factor is not a level of the surface's grid, or its second not a time grid");
  const Json tp = Json::parse(opt_text(*time, "params_json").value_or("{}"), nullptr, false);
  if (!tp.is_object() || !tp.contains("grid") || tp["grid"].value("kind", std::string()) != "time") bad("its second factor is not a time grid");
  const auto man = manifold(text(*space, "manifold_id"));
  if (!man) bad("its surface is not stored");
  Json out = Json{{"surface", text(*man, "name")}, {"space_level", fs["level"]}, {"tiles", fs.value("members", std::int64_t(0))}, {"frames", ft.value("members", tp["grid"].value("count", std::int64_t(0)))},
                  {"window_samples", tp["grid"].value("width", std::int64_t(0))}, {"time_level", tp["grid"].value("level", -1)}, {"time_manifold_id", text(*time, "manifold_id")}};
  if (tp.contains("band")) out["band"] = tp["band"];
  return out;
}

ScaleInputs read_scale_inputs(SubjectWriter& sub, const std::string& recording_id, const std::string& recording_name, const std::vector<int>& space_levels, const std::vector<int>& time_levels,
                              const std::string& source, const std::optional<std::string>& surface) {
  ScaleInputs out;
  std::string missing_joint, missing_run;
  std::vector<Row> joints, labellings;
  joints = nxr::io::selection::set_rows_with_param(sub, "product");
  labellings = nxr::io::selection::set_rows_with_param(sub, "cells_of");
  const auto selection_of = [&](const std::string& id) { return sub.db().read("selection", id); };
  const auto manifold_of = [&](const std::string& id) { return sub.db().read("manifold", id); };
  const auto stats = statistics_of(sub, recording_id, std::string("amplitude"));
  struct Found { ScalePairInput p; Row joint, labelling, baseline; std::string amp_path; std::int64_t amp_component = 0; };
  std::vector<Found> found;
  for (const int l : space_levels)
    for (const int m : time_levels) {
      Found f;
      f.p.space_level = l; f.p.time_level = m;
      const std::string pair = "space level " + std::to_string(l) + " x time level " + std::to_string(m);
      // the PRODUCT selections of this pair: the surface's level l x a time grid of 2^m samples (`select_product`); the one that holds a labelling made by `source` is the pair's
      std::vector<std::pair<Row, Json>> candidates;
      for (const Row& r : joints) {
        const Json jp = Json::parse(opt_text(r, "params_json").value_or("{}"), nullptr, false);
        Json j;
        try { j = product_facts(jp, selection_of, manifold_of, text(r, "name")); } catch (const Error&) { continue; }
        if (j.value("space_level", -1) == l && j.value("time_level", -1) == m && j.value("window_samples", std::int64_t(0)) > 0) candidates.emplace_back(r, std::move(j));
      }
      if (candidates.empty()) { missing_joint += (missing_joint.empty() ? "" : "; ") + pair + ": nxr-neuro run select_product --recording " + recording_name + " at the spatial scale and the band's time grid of that pair (quantities: a tile diameter or parts, the band)"; continue; }
      std::vector<Row> labs;
      for (auto& [jr, jj] : candidates) {
        for (const Row& r : labellings) {
          const Json lp = Json::parse(opt_text(r, "params_json").value_or("{}"), nullptr, false);
          if (!lp.is_object() || !lp.contains("cells_of") || lp["cells_of"].value("selection_id", std::string()) != text(jr, "id")) continue;
          if (!lp.contains("pipeline") || !lp["pipeline"].is_object() || lp["pipeline"].value("name", std::string()) != source) continue;
          labs.push_back(r);
        }
        if (!labs.empty()) { f.joint = jr; f.p.joint = jj; break; }
      }
      if (labs.empty()) { missing_run += (missing_run.empty() ? "" : "; ") + pair; continue; }
      if (labs.size() > 1) { std::string n; for (const Row& r : labs) n += (n.empty() ? "" : ", ") + text(r, "name"); invalid("scaleselection: " + std::to_string(labs.size()) + " labellings of " + text(f.joint, "name") + " were made by " + source + " (" + n + "): remove all but one"); }
      f.labelling = labs[0];
      f.p.tiles = f.p.joint["tiles"].get<std::int64_t>(); f.p.frames = f.p.joint["frames"].get<std::int64_t>(); f.p.window = f.p.joint.value("window_samples", std::int64_t(0));
      f.p.events = integer(f.labelling, "n_members");
      if (is_null(f.labelling, "array_path")) invalid(text(f.labelling, "name") + ": it stores no array of cells");
      const std::string jid = text(f.joint, "id");
      const auto am = nxr::io::selection::measurement_array(sub, jid, recording_id, "amplitude");
      if (!am) { missing_run += (missing_run.empty() ? "" : "; ") + pair + " (no cell amplitude)"; continue; }
      f.amp_path = am->array_path;
      f.amp_component = am->component;
      std::vector<Row> bs;
      for (const Row& r : stats) {   // the cells' baseline: a summary of the cell amplitude, a median and a mad per spatial tile, made by `source` over this product
        if (text(r, "method") != "summary") continue;
        const Json pj = Json::parse(opt_text(r, "producer_json").value_or("{}"), nullptr, false);
        if (!pj.is_object() || !pj.contains("pipeline") || !pj["pipeline"].is_object() || pj["pipeline"].value("name", std::string()) != source) continue;
        if (!pj.contains("inputs") || !pj["inputs"].contains("product") || pj["inputs"]["product"].value("id", std::string()) != jid) continue;
        const Json ax = Json::parse(opt_text(r, "axes_json").value_or("[]"), nullptr, false);
        bool median = false, mad = false;
        if (ax.is_array() && ax.size() == 2 && ax[1].contains("values")) for (const Json& v : ax[1]["values"]) { const std::string nm = v.is_string() ? v.get<std::string>() : v.value("name", std::string()); median = median || nm == "median"; mad = mad || nm == "mad"; }
        if (ax.is_array() && ax.size() == 2 && ax[0].value("size", std::int64_t(-1)) == f.p.tiles && median && mad) bs.push_back(r);
      }
      if (bs.empty()) { missing_run += (missing_run.empty() ? "" : "; ") + pair + " (no cell baseline)"; continue; }
      if (bs.size() > 1) invalid("scaleselection: " + std::to_string(bs.size()) + " baselines of " + text(f.joint, "name") + " were made by " + source + ": remove all but one");
      f.baseline = bs[0];
      found.push_back(std::move(f));
    }
  if (!missing_joint.empty()) invalid("scaleselection: " + recording_name + " has no product of the surface's level and the band's time grid for: " + missing_joint);
  if (!missing_run.empty())
    invalid("scaleselection: no stored cells, baseline and labelling made by " + source + " for: " + missing_run + " (nxr-neuro run " + source + " over each pair's product, at its spatial scale and band: --spatial-scale, --product, --band)");
  out.surface = found[0].p.joint["surface"].get<std::string>();
  out.manifold_id = text(found[0].joint, "manifold_id");
  for (const Found& f : found) {
    if (f.p.joint["surface"].get<std::string>() != out.surface) invalid("scaleselection: the joint Selections are on different surfaces (" + out.surface + ", " + f.p.joint["surface"].get<std::string>() + ")");
    if (text(f.joint, "manifold_id") != out.manifold_id) invalid("scaleselection: the joint Selections are on different manifolds");
  }
  if (surface && *surface != out.surface) invalid("scaleselection: surface " + *surface + " is not the joint Selections' (" + out.surface + ")");
  for (Found& f : found) {
    ScalePairInput& p = f.p;
    p.joint_id = text(f.joint, "id"); p.joint_name = text(f.joint, "name");
    p.labelling_id = text(f.labelling, "id"); p.labelling_name = text(f.labelling, "name"); p.baseline_name = text(f.baseline, "name");
    std::vector<std::int64_t> shape;
    // the labelling is an irregular family on the product of space tiles x time tiles: a member NUMBER per cell, the cells in the product's order (the space tile fastest: code = tile + tiles x time tile); the event's ordinal is the member's label
    p.labels = read_array<std::int32_t>(sub, text(f.labelling, "array_path"), &shape);
    if (shape != std::vector<std::int64_t>{p.tiles * p.frames})
      invalid("scaleselection: the labelling '" + p.labelling_name + "' is not " + std::to_string(p.tiles) + " tiles x " + std::to_string(p.frames) + " time tiles, flattened (the joint Selection's cells)");
    {
      const std::vector<std::int64_t> event_of_member = nxr::io::selection::member_labels(sub, p.labelling_id);
      for (std::int32_t& c : p.labels) c = c >= 0 && static_cast<std::size_t>(c) < event_of_member.size() ? static_cast<std::int32_t>(event_of_member[static_cast<std::size_t>(c)]) : -1;
      std::vector<std::int32_t> by_tile(p.labels.size());   // the step reads [tile x time tile], time fastest
      for (std::int64_t s = 0; s < p.tiles; ++s) for (std::int64_t t = 0; t < p.frames; ++t) by_tile[static_cast<std::size_t>(s * p.frames + t)] = p.labels[static_cast<std::size_t>(s + p.tiles * t)];
      p.labels = std::move(by_tile);
    }
    const std::vector<double> amp = read_doubles(sub, f.amp_path, &shape);
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != p.tiles * p.frames || f.amp_component < 0 || f.amp_component >= shape[2]) invalid("scaleselection: the cell amplitude of '" + p.joint_name + "' is not [1 x " + std::to_string(p.tiles * p.frames) + " cells x measures] (" + std::to_string(p.tiles) + " tiles x " + std::to_string(p.frames) + " time tiles)");
    p.amplitude.resize(static_cast<std::size_t>(p.tiles * p.frames));
    for (std::int64_t s = 0; s < p.tiles; ++s) for (std::int64_t t = 0; t < p.frames; ++t) p.amplitude[static_cast<std::size_t>(s * p.frames + t)] = amp[static_cast<std::size_t>((s + p.tiles * t) * shape[2] + f.amp_component)];
    const StoredStatistic bs = read_statistic(sub, p.baseline_name);
    int im = -1, is = -1;
    if (bs.axes.is_array() && bs.axes.size() == 2 && bs.axes[1].contains("values"))
      for (std::size_t k = 0; k < bs.axes[1]["values"].size(); ++k) {
        const Json& v = bs.axes[1]["values"][k];
        const std::string nm = v.is_string() ? v.get<std::string>() : v.value("name", std::string());
        if (nm == "median") im = static_cast<int>(k);
        if (nm == "mad") is = static_cast<int>(k);
      }
    if (im < 0 || is < 0 || bs.shape.size() != 2 || bs.shape[0] != p.tiles) invalid("scaleselection: the baseline '" + p.baseline_name + "' does not hold a median and a mad per spatial tile of '" + p.joint_name + "'");
    const std::int64_t K = bs.shape[1];
    p.median.resize(static_cast<std::size_t>(p.tiles)); p.sd.resize(static_cast<std::size_t>(p.tiles));
    for (std::int64_t s = 0; s < p.tiles; ++s) { p.median[static_cast<std::size_t>(s)] = bs.values[static_cast<std::size_t>(s * K + im)]; p.sd[static_cast<std::size_t>(s)] = bs.values[static_cast<std::size_t>(s * K + is)]; }
    for (const std::int32_t c : p.labels) if (c >= p.events) invalid("scaleselection: the labelling '" + p.labelling_name + "' labels an event beyond its " + std::to_string(p.events));
    out.pairs.push_back(std::move(p));
  }
  return out;
}

}  // namespace nxr::pipelines
