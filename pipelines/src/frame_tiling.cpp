// nxr-pipelines — the frame tiling of a recording (frame_tiling.h).
#include "nxr/pipelines/frame_tiling.h"

#include "nxr/pipelines/task.h"

#include <cmath>
#include <cstdio>
#include <sstream>

#include "nxr/tower.h"
#include "nxr/io/finder.h"
#include "nxr/io/selection_store.h"
#include "nxr/pipelines/meeg_internal.h"

namespace nxr::pipelines {

using namespace nxr::io;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(code::no_such_row, m); }
std::string g(double v) { std::ostringstream s; s << v; return s.str(); }
}  // namespace

std::string frame_tiling_name(const std::string& recording, int frame_level) {
  char b[64];
  std::snprintf(b, sizeof b, "%g", nxr::tower::periodS(frame_level));
  return recording + " frames " + b + " s";
}

FrameTilingResult frame_tiling_compute(DatasetWriter& ds, SubjectWriter& sub, const FrameTilingOptions& opt, const std::function<void(const Json&)>& progress) {
  (void)ds;
  if (opt.recording.empty()) invalid("frames: --recording names the recording whose time line the frames tile");
  const bool from_band = opt.centre_hz || opt.band;
  if (opt.frame_level && from_band) invalid("frames: --frame-level or the band (--centre / --band), not both");
  if (!opt.frame_level && !from_band) invalid("frames: name the tower level of a frame (--frame-level, e.g. -25) or derive it from the band (--centre HZ or --band LO,HI, with --frames-per-cycle)");
  if (opt.centre_hz && opt.band) invalid("frames: --centre or --band, not both");
  double centre = opt.centre_hz ? *opt.centre_hz : 0.0;
  int level = 0;
  if (from_band) {
    if (opt.band) {
      if (!(opt.band->second > opt.band->first && opt.band->first >= 0)) invalid("frames: --band LO,HI needs 0 <= LO < HI (Hz)");
      centre = (opt.band->first + opt.band->second) / 2;
    }
    if (!(centre > 0)) invalid("frames: the centre is above 0 Hz");
    if (opt.frames_per_cycle < 1) invalid("frames: --frames-per-cycle is at least 1");
    level = nxr::tower::frameLevelFor(centre, opt.frames_per_cycle);
  } else {
    level = *opt.frame_level;
  }
  if (level > 0) invalid("frames: --frame-level is 0 (the day) or finer (negative)");
  const task::Subject subj = task::Subject::of(sub);
  const auto rec = subj.find({.cls = task::Cls::field, .name = opt.recording, .kind = "recording"});
  if (!rec) missing(sub.name() + ": no recording '" + opt.recording + "'");
  const finder::Value rv = subj.describe(*rec);
  if (!rv.time || !(rv.rate > 0)) invalid(opt.recording + ": it has no time line with a rate");
  const std::string tid = *rv.time;
  const auto tline = subj.require_id(task::Cls::manifold, tid);
  const std::int64_t samples = *tline.n_vertices;
  nxr::tower::FrameTiling t;
  try { t = nxr::tower::makeFrameTiling(rv.rate, samples, level, opt.start_s); } catch (const std::invalid_argument& e) { invalid(e.what()); }
  const std::int64_t count = t.count;
  const std::string name = frame_tiling_name(opt.recording, level);
  FrameTilingResult res;
  if (const auto have = subj.find({.cls = task::Cls::selection, .name = name, .manifold_id = tid})) {
    if (!opt.replace) {
      res.existed = true;
      res.info = Json{{"selection", name}, {"id", have->id}, {"kept", true}};
      return res;
    }
    sub.remove("selection", have->id);
  }
  const double width = t.frame_s(), origin = rv.origin + t.origin_s();
  Json ft = Json{{"sfreq", rv.rate}, {"samples", samples}, {"fps", t.fps()}, {"recording", opt.recording}, {"frame_samples", t.tile_samples()},
                 {"count", "the tower tiles touched by at least one sample (the first and the last may be partial)"},
                 {"frame", "frame k is the tower tile frame_code0 + k; its time is the tile's START: t = origin + k * frame_s"}, {"reference", "nxr pipelines/atlas/tower.py (D135 TOWER)"}};
  if (from_band) { ft["centre_hz"] = centre; ft["frames_per_cycle"] = opt.frames_per_cycle; }
  if (opt.band) ft["band"] = Json::array({opt.band->first, opt.band->second});
  std::string id;
  sub.compose([&] {
    nxr::io::selection::Description tiles;
    tiles.type = nxr::io::selection::Type::set;
    tiles.tiling = nxr::io::selection::Tiling{origin, width, width, count};
    nxr::io::selection::Placement ss;
    ss.name = name; ss.manifold_id = tid; ss.cell = "1"; ss.of_field_id = rec->id;
    ss.cols.set("n_elements", samples);
    ss.description = ("the measurement pyramid's time tiles: " + std::to_string(count) + " cycles of tower level " + std::to_string(level) + " (" + g(width) + " s), frame k is code " + std::to_string(t.code0) + " + k, a level-m tile code >> m");
    ss.params = Json{{"end", origin + static_cast<double>(count) * width},
                                     {"atlas", Json{{"frame_s", width}, {"frame_level", level}, {"frame_code0", t.code0}, {"anchor", t.day_anchored ? "day" : "recording"}, {"start_s", opt.start_s ? Json(*opt.start_s) : Json(nullptr)}, {"made", true}}},
                                     {"frame_tiling", ft}, {"by", "nxr-neuro frametiling compute"}, {"producer", internal::producer_string("frametiling compute")}};
    id = nxr::io::selection::save(sub, tiles, ss).id;
  });
  res.info = Json{{"selection", name}, {"id", id}, {"frames", count}, {"fps", t.fps()}, {"frame_level", level}, {"frame_s", width}, {"frame_code0", t.code0}, {"kept", false}};
  if (from_band) { res.info["centre_hz"] = centre; res.info["frames_per_cycle"] = opt.frames_per_cycle; }
  if (progress) progress(res.info);
  return res;
}

}  // namespace nxr::pipelines
