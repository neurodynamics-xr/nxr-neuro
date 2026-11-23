// nxr::bst — a Brainstorm condition as a session: nxr.convert.condition (convert_condition, resolve_raw_binary), nxr.convert.subject.condition_plan,
// nxr.convert.naming (session-scoped names) and the imported-excerpt recording of nxr.convert.timeseries.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>

#include "np.h"
#include "nxr/bst/entities.h"
#include "nxr/bst/inverse.h"
#include "nxr/bst/timefreq.h"
#include "nxr/bst/naming.h"
#include "nxr/bst/pyrepr.h"
#include "nxr/bst/sensors.h"
#include "nxr/bst/projectors.h"
#include "nxr/bst/session.h"
#include "nxr/bst/subject.h"
#include "nxr/io/layout.h"

namespace nxr::bst {

namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::SubjectWriter;

namespace {

[[noreturn]] void value_error(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }
bool is_file(const fs::path& p) { std::error_code ec; return fs::is_regular_file(p, ec); }
bool is_dir(const fs::path& p) { std::error_code ec; return fs::is_directory(p, ec); }
bool dotted(const std::string& n) { return n.rfind("._", 0) == 0; }

std::string str_field(const Value* v) { return v && v->is_char() ? v->string() : ""; }

std::vector<std::string> sorted_glob(const fs::path& dir, const std::string& prefix, const std::string& suffix, const std::string& middle = "") {
  std::vector<std::string> out;
  if (!is_dir(dir)) return out;
  for (const auto& n : glob_names(dir, prefix, suffix, middle)) if (!dotted(n)) out.push_back(n);
  return out;
}

}  // namespace

fs::path resolve_raw_binary(const fs::path& raw_link, const Value& sfile, const Emit& progress) {
  const fs::path recorded = str_field(sfile.field("filename"));
  if (is_file(recorded)) return recorded;
  const fs::path beside = raw_link.parent_path() / recorded.filename();
  if (!recorded.filename().empty() && is_file(beside)) {
    if (progress) progress(Json{{"stage", "relinked"}, {"link", raw_link.filename().string()}, {"from", py_path(recorded.string())}, {"to", py_path(beside.string())}});
    return beside;
  }
  throw nxr::io::Error(code::not_found, raw_link.filename().string() + ": its binary is neither at the recorded path " + py_path(recorded.string()) + " nor beside the link (" +
                                            py_path(beside.string()) + ")");
}

// ── the condition ──

namespace {

std::int64_t acquisition_rows(SubjectWriter& sub, const std::string& acquisition) {
  std::int64_t n = 0;
  for (const char* t : {"manifold", "field", "operator", "selection"})
    n += nxr::io::integer(*sub.db().one(std::string("SELECT count(*) AS n FROM ") + t + " x WHERE x.subject_id = ? AND x.acquisition_id IN (SELECT id FROM acquisition WHERE subject_id = x.subject_id AND label = ?)",
                                        {sub.id(), acquisition}), "n");
  return n;
}

std::vector<std::string> channel_names_of(const Value* channel) {
  std::vector<std::string> out;
  if (channel && channel->is_struct()) for (std::size_t i = 0; i < channel->numel(); ++i) out.push_back(str_field(channel->field("Name", i)));
  return out;
}

}  // namespace

Json convert_condition(SubjectWriter& sub, const fs::path& bst_root, const std::string& condition, const ConditionPlan& plan, const ConvertOptions& opt, const Emit& progress) {
  auto emit = [&](const Json& j) { if (progress) progress(j); };
  if (plan.data_files.empty() && !plan.raw_data_file) value_error("nothing to convert: no data files and no raw link");
  if (acquisition_rows(sub, condition)) value_error("acquisition " + condition + " already exists in " + sub.name() + " \xe2\x80\x94 delete it first to re-import");
  const fs::path cond_dir = !plan.data_files.empty() ? plan.data_files[0].parent_path() : plan.channel_file.parent_path();
  const fs::path anat_dir_ = plan.surface_file.parent_path();
  const std::string acquisition = condition;

  // ---- the anatomy: THE ONE SURFACE the leadfield is defined on ----
  const std::string primary_surf = surface_node_name(plan.surface_file);
  auto present = [&](const fs::path& f) { const std::string n = surface_node_name(f); return sub.find("manifold", nxr::io::folder_for("surface", n) + "/" + n).has_value(); };
  std::vector<fs::path> surf_files;
  if (opt.all_surfaces) {
    for (const auto& n : sorted_glob(anat_dir_, "tess_", ".mat")) if (n.rfind("tess_fibers", 0) != 0) surf_files.push_back(anat_dir_ / n);
  } else {
    surf_files.push_back(plan.surface_file);
    for (const auto& e : plan.extra_surfaces) surf_files.push_back(e);
  }
  Json names = Json::array();
  for (const auto& sf : surf_files) {
    if (present(sf)) continue;
    if (!is_file(sf))
      value_error("subject has no anatomy (" + sf.filename().string() + " not found under anat/) \xe2\x80\x94 a sensor-only subject cannot import yet: the store is manifold-centric.");
    const bool primary = surface_node_name(sf) == primary_surf;
    const auto one = [&] {
      const auto info = export_surface(sub, sf, bst_root.string(), primary);
      names.push_back(info.name);
      emit(Json{{"stage", "surface"}, {"name", info.name}, {"folder", info.folder}, {"n_vertices", info.n_vertices}, {"hemispheres", info.hemispheres}});
    };
    if (primary) one();   // the surface the kernels index: the condition cannot go on without it
    else or_skip_unreadable_v73(sub, emit, one);
  }
  const auto surface = sub.find("manifold", nxr::io::folder_for("surface", primary_surf) + "/" + primary_surf);
  if (!surface) value_error("primary surface " + primary_surf + " not among exported " + names.dump());

  // SOURCE POWER BY BAND (D55/O31): the condition's timefreq_*.mat on the cortex the kernel indexes
  export_condition_timefreqs(sub, cond_dir, condition, acquisition, emit);

  // ---- the channels ----
  auto chan = read_all_variables(plan.channel_file);
  std::vector<std::int64_t> flags;
  std::optional<std::map<std::string, Value>> raw_mat;
  if (!plan.data_files.empty()) {
    const auto first = read_variable(plan.data_files[0], "ChannelFlag");
    if (!first) throw nxr::io::Error(code::no_such_row, "ChannelFlag");
    for (double x : first->doubles()) flags.push_back(static_cast<std::int64_t>(x));
  } else {
    raw_mat = read_all_variables(*plan.raw_data_file);
    const Value* cf = nullptr;
    if (auto it = raw_mat->find("ChannelFlag"); it != raw_mat->end()) cf = &it->second;
    if (!cf) if (auto it = raw_mat->find("F"); it != raw_mat->end() && it->second.is_struct()) cf = it->second.field("channelflag");
    if (cf && cf->is_numeric()) for (double x : cf->doubles()) flags.push_back(static_cast<std::int64_t>(x));
    else { const auto ch = chan.find("Channel"); for (std::size_t i = 0; ch != chan.end() && i < ch->second.numel(); ++i) flags.push_back(1); }
  }
  const auto entries = projector_entries(chan);
  Projection projection;                       // the record of the projectors: the samples are stored as recorded (the chunk service applies the active ones on load)
  // SESSION-SCOPED NAMES (D121): shared when the file and the flags are identical, qualified otherwise
  std::string flag_bytes;
  for (auto f : flags) flag_bytes.push_back(static_cast<char>(static_cast<std::int8_t>(f)));
  const std::string chan_fp = source_fingerprint({plan.channel_file}, flag_bytes);
  const auto [chan_name, chan_shared] = claim_node_name(sub, nxr::io::folder_for("points"), channel_node_name(plan.channel_file), acquisition, chan_fp);
  std::string channels_id;
  std::vector<std::string> ch_names;
  if (chan_shared) {
    const auto row = sub.find("manifold", nxr::io::folder_for("points") + "/" + chan_name);
    ch_names = channel_names_of(chan.count("Channel") ? &chan.at("Channel") : nullptr);
    channels_id = nxr::io::text(*row, "id");
    emit(Json{{"stage", "sensors-shared"}, {"name", chan_name}, {"n_channels", nxr::io::integer(*row, "n_vertices")}});
  } else {
    SensorOptions so;
    so.name = chan_name;
    so.acquisition = acquisition;
    so.source_sha1 = chan_fp;
    const SensorInfo si = export_sensors(sub, chan, flags, so);
    ch_names = si.names;
    channels_id = si.id;
    emit(Json{{"stage", "sensors"}, {"name", si.name}, {"n_channels", si.n_channels}});
    const auto pnames = export_projectors(sub, entries, chan_name, channels_id, acquisition);
    std::optional<CtfGrade> g0;
    if (raw_mat) if (auto itF = raw_mat->find("F"); itF != raw_mat->end()) g0 = ctf_grade(&itF->second);
    const auto cname = export_compensation(sub, chan, chan_name, channels_id, acquisition, g0);
    if (!pnames.empty() || cname) emit(Json{{"stage", "projectors"}, {"name", chan_name}, {"projectors", static_cast<std::int64_t>(pnames.size())}, {"ctf_compensation", cname.has_value()}});
  }

  // ---- the recording(s) ----
  for (const auto& df : plan.data_files)
    or_skip_unreadable_v73(sub, emit, [&] {   // a recording in MATLAB v7.3 this build cannot read: skipped, named
      const Json ti = export_timeseries(sub, recording_node_name(df), df, acquisition, channels_id, ch_names, opt.resample,
                                        [&] { Projection p = projection; p.record = recording_record(entries, chan, chan_name, std::nullopt); return p; }());
      emit(Json{{"stage", "timeseries"}, {"name", ti["name"]}, {"n_samples", ti["n_samples"]}, {"sfreq", ti["sfreq"]}, {"events", ti["events"]}});
    });
  Json truncated;   // the raw recording, when its binary was cut short (reported in the result)
  if (plan.raw_data_file) {
    if (!raw_mat) raw_mat = read_all_variables(*plan.raw_data_file);
    const auto itF = raw_mat->find("F");
    if (itF == raw_mat->end()) throw nxr::io::Error(code::no_such_row, "F");
    const Value& sfile = itF->second;
    if (!sfile.is_struct()) value_error(plan.raw_data_file->string() + " is not a raw link (F is a matrix)");
    const Value* cf = nullptr;
    if (auto it = raw_mat->find("ChannelFlag"); it != raw_mat->end()) cf = &it->second;
    const Value* comment = nullptr;
    if (auto it = raw_mat->find("Comment"); it != raw_mat->end()) comment = &it->second;
    const fs::path bst = resolve_raw_binary(*plan.raw_data_file, sfile, progress);
    const Json ri = export_raw_timeseries(sub, recording_node_name(*plan.raw_data_file), sfile, cf, acquisition, channels_id, comment, bst, ch_names, progress, opt.resample,
                                          [&] { Projection p = projection; p.record = recording_record(entries, chan, chan_name, ctf_grade(&sfile)); return p; }());
    Json line = Json{{"stage", "timeseries-raw"}};
    for (auto it = ri.begin(); it != ri.end(); ++it) if (it.key() != "id" && it.key() != "time_id") line[it.key()] = it.value();
    emit(line);
    if (ri.contains("truncated")) truncated = ri["truncated"];
  }

  // ---- kernels (+ head models, deduped per file) ----
  ConditionKernels ck;
  ck.cond_dir = cond_dir;
  ck.bst_root = bst_root;
  ck.acquisition = acquisition;
  ck.channels_id = channels_id;
  ck.surface_id = nxr::io::text(*surface, "id");
  ck.n_channels = static_cast<std::int64_t>(ch_names.size());
  ck.n_vertices = nxr::io::integer(*surface, "n_vertices");
  ck.channel_names = ch_names;
  ck.kernel_files = plan.kernel_files;
  ck.write_gain = opt.write_gain;
  const std::vector<std::string> inv_names = import_condition_kernels(sub, ck, emit);
  Json res = Json{{"subject", sub.name()}, {"surfaces", names}, {"inverses", inv_names}, {"acquisition", acquisition}};
  if (!truncated.is_null()) res["truncated"] = truncated;
  return res;
}

}  // namespace nxr::bst
