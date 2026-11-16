// nxr-pipelines — the joint partition of space × time (joint.h).
#include "nxr/pipelines/joint.h"

#include <cmath>

#include "nxr/io/finder.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/spatial_band.h"

namespace nxr::pipelines {

using namespace nxr::io;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(code::no_such_row, m); }
}  // namespace

int joint_bits(std::int64_t frames) {
  int b = 1;
  while ((std::int64_t(1) << b) < std::max<std::int64_t>(2, frames)) ++b;
  return b;
}

JointComputeResult joint_compute(DatasetWriter& ds, SubjectWriter& sub, const JointComputeOptions& opt, const std::function<void(const Json&)>& progress) {
  (void)ds;
  if (opt.recording.empty()) invalid("joint: --recording names the recording whose time line is the cells' time");
  if (opt.space_level < 0) invalid("joint: --space-level is the surface ladder's level (0, 1, 2, …)");
  if (opt.time_level < 0) invalid("joint: --time-level is the constant-Q ladder's level (the member ordinal of `constantq compute`'s ladder)");
  const auto rec = sub.db().one("SELECT * FROM field WHERE subject_id = ? AND name = ? AND kind = 'recording' AND status = 'complete'", {V(sub.id()), V(opt.recording)});
  if (!rec) missing(sub.name() + ": no recording '" + opt.recording + "'");
  const finder::Value rv = finder::value_of(sub.db(), "field", *rec);
  if (!rv.time) invalid(opt.recording + ": it has no time line");
  const std::string tid = *rv.time;
  const auto tline = sub.db().read("manifold", tid);
  // THE TIME LEVEL: a member of the constant-Q ladder on the recording's time line
  const auto tladder = sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND manifold_id = ? AND type = 'ladder' AND status = 'complete'", {V(sub.id()), V(tid)});
  if (!tladder) missing(sub.name() + ": " + text(*tline, "name") + " has no constant-Q ladder; run `nxr-neuro constantq compute` first");
  const auto tlevel = sub.db().one("SELECT * FROM selection WHERE member_of_id = ? AND member_ordinal = ? AND status = 'complete'", {V(text(*tladder, "id")), V(static_cast<std::int64_t>(opt.time_level))});
  if (!tlevel) invalid(sub.name() + ": the constant-Q ladder of " + text(*tline, "name") + " has no level " + std::to_string(opt.time_level) + " (its levels are 0 … " + std::to_string(integer(*tladder, "n_members") - 1) + ")");
  const Json tp = Json::parse(text(*tlevel, "params_json"), nullptr, false);
  if (!tp.is_object() || !tp.contains("tiling") || !tp.contains("resolves_hz")) invalid(text(*tlevel, "name") + ": not a constant-Q level (no tiling or band in its record)");
  const std::int64_t frames = tp["tiling"]["count"].get<std::int64_t>();
  // THE SPACE LEVEL: a member of the surface's tile ladder
  const SurfaceInput S = read_surface(sub, opt.surface, true);
  const SurfaceLevel SL = read_surface_level(sub, S, opt.space_level);
  const std::int64_t tiles = SL.nodes;
  const int bits = joint_bits(frames);
  if ((tiles << bits) >= (std::int64_t(1) << 31)) invalid("joint: " + std::to_string(tiles) + " tiles × 2^" + std::to_string(bits) + " time indices do not fit a 32-bit code");
  const std::string mid = sub.product({S.id, tid});
  const std::string name = SL.tiling + " x " + text(*tlevel, "name") + " (" + opt.recording + ")";
  const std::string path = node_path(sub.db(), mid, safe_name(name));
  JointComputeResult res;
  if (const auto have = sub.find("selection", path)) {
    if (!opt.replace) {
      res.existed = true;
      res.info = Json{{"selection", name}, {"id", text(*have, "id")}, {"kept", true}};
      return res;
    }
    sub.remove("selection", text(*have, "id"));
  }
  const Json joint = Json{{"space", text(SL.level, "id")}, {"time", text(*tlevel, "id")}, {"time_ladder", text(*tladder, "id")}, {"space_level", opt.space_level}, {"time_level", opt.time_level},
                          {"band", Json::array({tp["resolves_hz"]["lo"], tp["resolves_hz"]["hi"]})}, {"bits", bits}, {"tiles", tiles}, {"frames", frames}, {"layout", Json::array({"tile", "frame"})},
                          {"code", "(tile << bits) | frame"}, {"recording", opt.recording}, {"surface", S.name}, {"window_samples", tp.value("window_samples", std::int64_t(0))}};
  std::string id;
  sub.compose([&] {
    SelectionSpec ss;
    ss.name = name; ss.path = path; ss.type = "set"; ss.manifold_id = mid; ss.no_elements = true;
    ss.cols.set("cell", "0,1").set("n_members", tiles * frames).set("label", std::to_string(tiles) + " tiles x " + std::to_string(frames) + " time tiles")
        .set("description", "the joint cells of " + SL.tiling + " and " + text(*tlevel, "name") + ": code = (tile << " + std::to_string(bits) + ") | time tile (the Studio atlas's layout), closed form")
        .json("params_json", Json{{"joint", joint}, {"by", "nxr-neuro joint compute"}, {"producer", internal::producer_string("joint compute")}});
    id = sub.selection(std::move(ss));
  });
  res.info = Json{{"selection", name}, {"id", id}, {"tiles", tiles}, {"frames", frames}, {"bits", bits}, {"band", joint["band"]}, {"kept", false}};
  if (progress) progress(res.info);
  return res;
}

}  // namespace nxr::pipelines
