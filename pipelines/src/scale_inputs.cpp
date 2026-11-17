// nxr-pipelines — the stored rows scale selection reads (scale_inputs.h). Reading only.
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
std::string list_of(const std::vector<int>& v) { std::string s; for (const int x : v) s += (s.empty() ? "" : ",") + std::to_string(x); return s; }
}  // namespace

ScaleInputs read_scale_inputs(SubjectWriter& sub, const std::string& recording_id, const std::string& recording_name, const std::vector<int>& space_levels, const std::vector<int>& time_levels,
                              const std::string& source, const std::optional<std::string>& surface) {
  ScaleInputs out;
  std::string missing_joint, missing_run;
  std::vector<Row> joints, labellings;
  joints = nxr::io::selection::set_rows_with_param(sub.db(), sub.id(), "joint");
  labellings = nxr::io::selection::set_rows_with_param(sub.db(), sub.id(), "cells_of");
  const auto stats = statistics_of(sub.db(), sub.id(), recording_id, std::string("amplitude"));
  struct Found { ScalePairInput p; Row joint, labelling, baseline; std::string amp_path; std::int64_t amp_component = 0; };
  std::vector<Found> found;
  for (const int l : space_levels)
    for (const int m : time_levels) {
      Found f;
      f.p.space_level = l; f.p.time_level = m;
      bool have = false;
      for (const Row& r : joints) {
        const Json jp = Json::parse(opt_text(r, "params_json").value_or("{}"), nullptr, false);
        if (!jp.is_object() || !jp.contains("joint")) continue;
        const Json& j = jp["joint"];
        if (j.value("recording", std::string()) == recording_name && j.value("space_level", -1) == l && j.value("time_level", -1) == m) { f.joint = r; f.p.joint = j; have = true; break; }
      }
      const std::string pair = "space level " + std::to_string(l) + " x time level " + std::to_string(m);
      if (!have) { missing_joint += (missing_joint.empty() ? "" : "; ") + pair + ": nxr-neuro run joint --recording " + recording_name + " --space-level " + std::to_string(l) + " --time-level " + std::to_string(m); continue; }
      const std::string jid = text(f.joint, "id");
      std::vector<Row> labs;
      for (const Row& r : labellings) {
        const Json lp = Json::parse(opt_text(r, "params_json").value_or("{}"), nullptr, false);
        if (!lp.is_object() || !lp.contains("cells_of") || lp["cells_of"].value("selection_id", std::string()) != jid) continue;
        if (!lp.contains("pipeline") || !lp["pipeline"].is_object() || lp["pipeline"].value("name", std::string()) != source) continue;
        labs.push_back(r);
      }
      if (labs.empty()) { missing_run += (missing_run.empty() ? "" : "; ") + pair; continue; }
      if (labs.size() > 1) { std::string n; for (const Row& r : labs) n += (n.empty() ? "" : ", ") + text(r, "name"); invalid("scaleselection: " + std::to_string(labs.size()) + " labellings of " + text(f.joint, "name") + " were made by " + source + " (" + n + "): remove all but one"); }
      f.labelling = labs[0];
      f.p.tiles = f.p.joint["tiles"].get<std::int64_t>(); f.p.frames = f.p.joint["frames"].get<std::int64_t>(); f.p.window = f.p.joint.value("window_samples", std::int64_t(0));
      if (f.p.window < 1) invalid("scaleselection: the joint Selection '" + text(f.joint, "name") + "' records no time tile width (made before `window_samples`: `run joint --replace`)");
      f.p.events = integer(f.labelling, "n_members");
      if (is_null(f.labelling, "array_path")) invalid(text(f.labelling, "name") + ": it stores no array of cells");
      const auto am = nxr::io::selection::measurement_array(sub.db(), jid, recording_id, "amplitude");
      if (!am) { missing_run += (missing_run.empty() ? "" : "; ") + pair + " (no cell amplitude)"; continue; }
      f.amp_path = am->array_path;
      f.amp_component = am->component;
      std::vector<Row> bs;
      for (const Row& r : stats) {
        if (text(r, "method") != "summary" || is_null(r, "partition_selection_id") || text(r, "partition_selection_id") != jid) continue;
        const Json pj = Json::parse(opt_text(r, "producer_json").value_or("{}"), nullptr, false);
        if (pj.is_object() && pj.contains("pipeline") && pj["pipeline"].is_object() && pj["pipeline"].value("name", std::string()) == source) bs.push_back(r);
      }
      if (bs.empty()) { missing_run += (missing_run.empty() ? "" : "; ") + pair + " (no cell baseline)"; continue; }
      if (bs.size() > 1) invalid("scaleselection: " + std::to_string(bs.size()) + " baselines of " + text(f.joint, "name") + " were made by " + source + ": remove all but one");
      f.baseline = bs[0];
      found.push_back(std::move(f));
    }
  if (!missing_joint.empty()) invalid("scaleselection: " + recording_name + " has no joint Selection for: " + missing_joint);
  if (!missing_run.empty())
    invalid("scaleselection: no stored cells, baseline and labelling made by " + source + " for: " + missing_run + " (nxr-neuro run " + source + " --set 'space_levels=[" + list_of(space_levels) + "]' --set 'bands=[the bands of the time levels]' --chunks all)");
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
    // the labelling is an irregular family on the product of space tiles x time tiles: a member NUMBER per cell, the cells flattened row-major (time fastest); the event's ordinal is the member's label
    p.labels = read_array<std::int32_t>(sub, text(f.labelling, "array_path"), &shape);
    if (shape != std::vector<std::int64_t>{p.tiles * p.frames} && shape != std::vector<std::int64_t>{p.tiles, p.frames})   // [tiles x time tiles] is the shape an earlier version stored the cells in
      invalid("scaleselection: the labelling '" + p.labelling_name + "' is not " + std::to_string(p.tiles) + " tiles x " + std::to_string(p.frames) + " time tiles, flattened (the joint Selection's cells)");
    {
      const std::vector<std::int64_t> event_of_member = nxr::io::selection::member_labels(sub.db(), sub.store(), p.labelling_id);
      for (std::int32_t& c : p.labels) c = c >= 0 && static_cast<std::size_t>(c) < event_of_member.size() ? static_cast<std::int32_t>(event_of_member[static_cast<std::size_t>(c)]) : -1;
    }
    const std::vector<float> amp = read_array<float>(sub, f.amp_path, &shape);
    if (shape.size() != 3 || shape[0] != p.tiles || shape[1] != p.frames || f.amp_component < 0 || f.amp_component >= shape[2]) invalid("scaleselection: the cell amplitude of '" + p.joint_name + "' is not [" + std::to_string(p.tiles) + " x " + std::to_string(p.frames) + " x components]");
    p.amplitude.resize(static_cast<std::size_t>(p.tiles * p.frames));
    for (std::int64_t s = 0; s < p.tiles; ++s) for (std::int64_t t = 0; t < p.frames; ++t) p.amplitude[static_cast<std::size_t>(s * p.frames + t)] = static_cast<double>(amp[static_cast<std::size_t>((s * p.frames + t) * shape[2] + f.amp_component)]);
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
