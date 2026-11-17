// nxr-pipelines — the constant-Q ladder of a time Line, designed by compute and stored by io (constant_q.h).
#include "nxr/pipelines/constant_q.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "nxr/io/errors.h"
#include "nxr/io/selection_store.h"
#include "nxr/pipelines/selection_bridge.h"
#include "nxr/selection/partition.h"

namespace nxr::pipelines {

namespace {
std::string fixed(double x, int d) { char b[64]; std::snprintf(b, sizeof b, "%.*f", d, x); return b; }
std::string fmt_hz(double x) { return x >= 10 ? fixed(x, 1) : x >= 1 ? fixed(x, 2) : fixed(x, 3); }
}  // namespace

ConstantQWritten constant_q_write(nxr::io::SubjectWriter& sub, const std::string& manifold_id, const ConstantQOptions& o, const std::optional<std::string>& by) {
  namespace sel = nxr::io::selection;
  // READ the line (io): its sample rate and length are what the design is made for
  const sel::TimeLine line = sel::time_line(sub.db(), manifold_id);
  if (o.method && *o.method != "hertz" && *o.method != "cycles") throw nxr::io::Error(nxr::io::code::invalid_argument, "constantQ: method must be hertz or cycles");
  ConstantQWritten w;
  w.name = line.name + "_constantQ";
  // the ladder already on the line (a ladder whose params name `voices`)
  if (const auto have = sel::find_with_param(sub.db(), sub.id(), manifold_id, "ladder", "voices")) {
    w.existed = true;
    w.set_id = have->id;
    w.member_ids = have->member_ids;
    return w;
  }
  // DESIGN (compute)
  nxr::selection::Line time;
  time.unit = "s";
  time.count = line.count;
  time.origin = line.origin;
  time.spacing = line.spacing;
  nxr::selection::ConstantQ method;
  if (o.method) method.method = *o.method;
  if (o.voices) method.voices = *o.voices;
  method.cycles = o.cycles;
  method.f_min = o.f_min;
  method.f_max = o.f_max;
  const sel::Description design = describe(nxr::selection::constant_q(time, method));
  // STORE (io): the designed levels, where and under what name; what the design was asked for goes into the ladder's parameters
  sel::Placement at;
  at.name = w.name;
  at.manifold_id = manifold_id;
  at.acquisition_id = line.acquisition_id;
  at.created_by = by;
  at.cell = "1";
  double f_lo = 0, f_hi = 0;
  for (const auto& lv : design.members)
    if (lv.band) { f_lo = f_lo == 0 ? lv.band->first : std::min(f_lo, lv.band->first); f_hi = std::max(f_hi, lv.band->second); }
  at.params = nxr::io::Json{{"method", o.method.value_or("hertz")}, {"voices", o.voices.value_or(6)}};
  if (o.cycles) at.params["cycles"] = *o.cycles;
  if (f_hi > 0) { at.params["f_max"] = f_hi; at.params["f_min"] = f_lo; }
  at.label = "constant-Q, " + std::to_string(design.members.size()) + " levels";
  at.description = "the constant-Q tiling of the time line: " + std::to_string(design.members.size()) + " time resolutions, designed at " + std::to_string(o.voices.value_or(6)) +
                   " voices/octave over " + (f_hi > 0 ? fmt_hz(f_lo) + "\xe2\x80\x93" + fmt_hz(f_hi) + " Hz" : std::string("the method's range")) +
                   " \xe2\x80\x94 a tiling of TIME, on the time manifold and no other; the hertz describe what the windows resolve and reference nothing (D84)";
  const sel::Saved saved = sel::save(sub, design, at);
  w.set_id = saved.id;
  w.member_ids = saved.member_ids;
  const double sfreq = 1.0 / line.spacing;
  for (std::size_t k = 0; k < design.members.size(); ++k) {
    const auto& lv = design.members[k];
    ConstantQLevel l;
    l.level = static_cast<int>(k);
    l.window_samples = static_cast<std::int64_t>(std::llround(lv.tiling->width * sfreq));
    l.window_s = lv.tiling->width;
    l.tiles = lv.tiling->count;
    if (lv.band) { l.resolves_lo = lv.band->first; l.resolves_hi = lv.band->second; }
    w.levels.push_back(l);
  }
  return w;
}

}  // namespace nxr::pipelines
