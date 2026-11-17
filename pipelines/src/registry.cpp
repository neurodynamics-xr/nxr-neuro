// nxr-pipelines — the registered steps (registry.h): their ports, parameters, references and arrow checks. The steps' maths are pure functions of compute/
// and meeg/ (bound and run by the runner); this file only declares and checks, as nxr's nxr/pipelines/steps.py.
//
//   band_filter     chunk       a field on … × time, band-limited, zero phase (the Studio's taps of a Meyer band: gainTaps · convolveRows)
//   analytic        chunk       ℝ → ℂ, the analytic signal x + i H[x] (the Studio's odd taps, hilbertImag; the transform "complex")
//   envelope        chunk       ℝ → ℝ, |analytic| (NON-LINEAR: no linear map may follow it, D106)
//   frames          chunk       RETIMING: a field on the recording's samples sampled on the FRAME TILING (`frametiling compute`: the cycles of one TOWER level, D135; nxr band_at_frames resamples at them):
//                               band-limited (Kaiser-sinc) interpolation of a band's signal at the frames inside the chunk; the steps after it fold FRAMES; with `within` (a spans Selection) only the frames inside it are good
//   apply_kernel    chunk       a linear operator from the field's first factor (an imaging kernel: sensors → sources); ℝ or ℂ kept; with `tiles` and `level`, the level's
//                               spatial Weyl band folded into the kernel once (M = (F_l ⊗ I) K)
//   magnitude       chunk       |x| per vertex, the norm over its components (and the complex part): pointwise, non-linear
//   form            chunk       a form of a field of its type (the Studio's FORMS[T], ℂ-extended: linear forms keep the phase, abs / absq give the envelope)
//   power           chunk       |x|² per element per sample (over its components): POINTWISE
//   phase_current   chunk       j = Im(z̄ ∇z) on the tangent plane, per vertex per sample (nxr core_phase): POINTWISE
//   gradient_power  chunk       |∇z|² of the tangent gradient, per vertex per sample: POINTWISE
//   measure         accumulate  a field reduced over the run's span on its time line (the span as one window of the time partition; the measurement
//                               functions, the Studio's reduceTiles folded chunk by chunk), the samples in excluded spans left out; or, with `over` (a
//                               constant-Q ladder), per TILE of the level of the band (D84: the level IS the band): `amplitude` · `power` (a measurement); or over a
//                               JOINT partition of space × time (`joint compute`), per CELL (spatial tile × time tile): `amplitude` · `envelope_max` of |z| per vertex, `power` (the power step's |z|²)
//                               and `phase_current` (the phase_current step's vector: x y z, a [rows x cells x 3] measurement)
//   fit_gate        after       the bursts `risefall` fitted well: r² ≥ min_r2 (0.7) and the rise and fall times not at their bound → a spans Selection of the kept bursts (the unfiltered spans stay)
//   burst_stages    after       each burst's stages from its stored fit: onset (t_on ± 2 τ_rise), within (the plateau), offset (t_off ± 2 τ_fall) → three spans Selections (nxr::risefall::stagesOf)
//   detect_spans    after       a measurement's tiles labelled: per row (or the mean over rows) the median and robust SD over the good tiles, hysteresis
//                               crossings, spans merged and kept by length (nxr::intervals; Rheome detect.spindle's rule) → a span set (a spans Selection)
#include <sstream>

#include "nxr/pipelines/registry.h"

#include <algorithm>
#include <set>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>

#include "nxr/filterbank.h"
#include "nxr/forms.h"
#include "nxr/frame_tiling.h"
#include "nxr/tower.h"
#include "nxr/intervals.h"
#include "nxr/stats.h"
#include "nxr/partition.h"
#include "nxr/phase.h"
#include "nxr/risefall.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/input.h"
#include "nxr/monogenic.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/taps.h"

namespace nxr::pipelines::graph {

void add_extent_steps(std::vector<StepSpec>& r);   // extent_steps.cpp: cell_extent, extent_measure
void add_cores_steps(std::vector<StepSpec>& r);    // cores_steps.cpp: core_histogram, cores
void add_tracks_steps(std::vector<StepSpec>& r);   // tracks_steps.cpp: tracks
void add_crossings_steps(std::vector<StepSpec>& r);   // crossings_steps.cpp: tile_crossings
void add_surrogate_steps(std::vector<StepSpec>& r);   // surrogate_steps.cpp: power_series, lagfree_surrogate
void add_spatial_events_steps(std::vector<StepSpec>& r);   // spatial_events_steps.cpp: at_level, relative, …

void refuse(const std::string& message) { throw nxr::io::Error(nxr::io::code::invalid_argument, message); }

namespace {
void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }

// THE STORED EVENTS OF A ROLL-UP (maintainers spec docs/specs/2026-11-13-burst-phase-current.md): a spans Selection of this recording's time line (the global bursts `global_bursts` stored), read as the tile events
// [first, stop) of the tiles x was measured on. A span holds on EVERY row (a global event: it names no channel); one that names a channel is refused (a joint measurement's rows are spatial tiles).
// The run is refused, naming the span, when the set is of another recording or time line, or a span does not begin and end on a tile edge of x's time level.
std::vector<nxr::intervals::TileEvent> stored_event_tiles(const Bound& db, const Value& ev, const Value& x, const std::string& who) {
  const nxr::io::Row& sr = *ev.row;
  const std::string sname = nxr::io::text(sr, "name");
  need(ev.manifolds.size() == 1 && x.time && ev.manifolds[0] == *x.time, who + ": the spans " + sname + " are on another time line than x's (" + (x.time ? *x.time : std::string("none")) + "): the events must be of x's own recording");
  const Json& over = x.facts["over"];
  if (over.value("joint", false)) {
    const auto js = db.selection(over["selection_id"].get<std::string>());
    const Json jp = js ? Json::parse(nxr::io::opt_text(*js, "params_json").value_or("{}"), nullptr, false) : Json();
    const std::string rec = jp.is_object() && jp.contains("joint") ? jp["joint"].value("recording", std::string()) : std::string();
    if (const auto fid = nxr::io::opt_text(sr, "of_field_id")) {
      const auto f = db.field(*fid);
      need(!f || rec.empty() || nxr::io::text(*f, "name") == rec, who + ": the spans " + sname + " are of recording " + (f ? nxr::io::text(*f, "name") : *fid) + ", x's joint partition is of recording " + rec);
    }
  }
  const double w = over.value("width_s", 0.0), origin = over.value("origin", 0.0);
  const std::int64_t T = over.value("tiles", std::int64_t(0));
  need(w > 0 && T > 0, who + ": x carries no tile width");
  const Json params = Json::parse(nxr::io::opt_text(sr, "params_json").value_or("{}"), nullptr, false);
  const bool channels = params.is_object() && params.contains("channels") && params["channels"].is_object() && !params["channels"].empty();
  need(!channels, who + ": the spans " + sname + " name the channel that found each event (per_hemisphere); a measurement over tiles takes global events only");
  std::vector<nxr::intervals::TileEvent> out;
  std::size_t ordinal = 0;
  auto sec = [](double x) { std::ostringstream o; o << x; return o.str(); };   // seconds as written: 0.5, not 0.500000
  for (const auto& r : db.spans(nxr::io::text(sr, "id"))) {
    const double a = (r.first - origin) / w, b = (r.second - origin) / w;
    const std::string which = "span " + std::to_string(ordinal++) + " [" + sec(r.first) + ", " + sec(r.second) + ") s of " + sname;
    need(std::abs(a - std::round(a)) <= 1e-3 && std::abs(b - std::round(b)) <= 1e-3, who + ": " + which + " does not snap to the tiles of " + over.value("selection", std::string("x's time level")) + " (tile " + std::to_string(w) + " s from " + std::to_string(origin) + " s)");
    const std::int64_t f = static_cast<std::int64_t>(std::llround(a)), e = static_cast<std::int64_t>(std::llround(b));
    need(f >= 0 && e <= T && e > f, who + ": " + which + " lies outside the " + std::to_string(T) + " tiles of x or is empty");
    out.push_back({f, e, -1});
  }
  return out;
}
Value made(const Value& x) { Value v = x; v.row.reset(); return v; }
bool nonlinear(const Value& x) { return x.facts.value("nonlinear", false); }
std::string manifold_type(const PlanContext& c, const std::string& id) {
  return c.bound.manifold_type(id);
}
// `frames_ok`: a step that is pointwise in time (a linear map, a form, |x|², a series kept) may read a field on the FRAME TILING (the `frames` step's output); the steps that read the samples' time (a band's taps, a time
// partition's tiles) refuse it
void in_loop(const Value& x, const std::string& who, bool frames_ok = false) {
  need(x.cls == "field" && x.time && x.chunked, who + ": x must be a field on … × time, inside the chunk loop");
  need(frames_ok || x.facts.value("axis", std::string("samples")) == "samples", who + ": x is on the frame tiling (made by `frames`); this step reads the recording's own samples");
}
// the Studio's field type of a value (forms.h): ℝ a number, ℂ an embedded tangent vector, ℍ a free-orientation current (vector3 or quaternion)
nxr::forms::FieldType field_type(const Value& x) {
  if (x.facts.value("field_type", std::string()) == "complex") return nxr::forms::FieldType::Complex;
  if (x.value_type == "vector3" || x.value_type == "quaternion") return nxr::forms::FieldType::Quaternion;
  return nxr::forms::FieldType::Real;
}
const char* type_name(nxr::forms::FieldType t) { return t == nxr::forms::FieldType::Real ? "ℝ" : t == nxr::forms::FieldType::Complex ? "ℂ" : "ℍ"; }
// the phase's steps read a complex field whose phase is kept, on a surface, with 1 or 3 components a vertex
void phase_input(const Value& z, const PlanContext& c, const std::string& who) {
  in_loop(z, who, true);
  need(z.complex, who + ": x must be complex (the analytic signal, or a linear map or form of it): a real field has no phase");
  need(!nonlinear(z), who + ": x was made by a non-linear step (" + z.facts.value("made_by", std::string("?")) + "): its phase is gone");
  need(z.value_type == "scalar" || z.value_type == "vector3", who + ": x is " + z.value_type + "; one complex number a vertex, or three (a free-orientation current)");
  need(manifold_type(c, z.manifolds[0]) == "surface", who + ": x's first factor is not a surface (the phase's gradient is taken on the cortex)");
}

// THE LEVEL OF A BAND (D84, the Studio's measurements.ts: "the level IS the band"): the member of a constant-Q ladder whose `resolves_hz` is the band, its
// tiling in samples (contiguous from the line's start). A band that is no level's is refused, naming the levels' bands.
struct TimeLevel { std::string id, name; int level = 0; std::int64_t width = 0, count = 0; Json resolves; };
TimeLevel level_of_band(const Bound& db, const std::string& ladder_id, const Json& band, double rate, double origin) {
  need(band.is_array() && band.size() == 2, "measure: the band " + band.dump() + " is [lo, hi]");
  const double lo = band[0].get<double>(), hi = band[1].get<double>();
  std::string have;
  for (const auto& m : db.level_members(ladder_id)) {
    const Json pj = Json::parse(nxr::io::opt_text(m, "params_json").value_or("{}"), nullptr, false);
    if (!pj.is_object() || !pj.contains("resolves_hz") || !pj.contains("tiling")) continue;
    const double rl = pj["resolves_hz"]["lo"].get<double>(), rh = pj["resolves_hz"]["hi"].get<double>();
    char b[64];
    std::snprintf(b, sizeof b, "%s[%g, %g]", have.empty() ? "" : ", ", rl, rh);
    have += b;
    if (std::abs(rl - lo) > 1e-9 * std::max(1.0, lo) || std::abs(rh - hi) > 1e-9 * std::max(1.0, hi)) continue;
    TimeLevel t;
    t.id = nxr::io::text(m, "id");
    t.name = nxr::io::text(m, "name");
    t.level = static_cast<int>(nxr::io::integer(m, "member_ordinal"));
    t.width = pj.value("window_samples", std::int64_t(0));
    t.count = pj["tiling"]["count"].get<std::int64_t>();
    t.resolves = pj["resolves_hz"];
    need(t.width > 0 && std::llround(pj["tiling"]["width"].get<double>() * rate) == t.width && std::abs(pj["tiling"]["origin"].get<double>() - origin) < 0.5 / rate,
         "measure: " + t.name + " is not a contiguous tiling of x's time line from its first sample");
    return t;
  }
  refuse("measure: the band " + band.dump() + " Hz is no level's band of this constant-Q ladder (the level IS the band, D84); its levels' bands: " + have);
}

void bind_execution(std::vector<StepSpec>& r);
void bind_traits(std::vector<StepSpec>& r);
// a band's parameters checked (band_filter, analytic, envelope): [lo, hi] Hz below Nyquist; `optional`: null allowed (the plain Hilbert, all-pass)
void band_check(const Json& p, const Value& x, const std::string& who, bool optional) {
  const Json& band = p["band"];
  if (optional && band.is_null()) return;
  need(band.is_array() && band.size() == 2 && band[0].is_number() && band[1].is_number() && band[0].get<double>() >= 0 && band[0].get<double>() < band[1].get<double>(),
       who + ": band " + band.dump() + ": [lo, hi] in Hz, lo < hi" + (optional ? " (or null: the plain Hilbert, cut at 1024 taps)" : ""));
  need(p["shape"] == "morlet" || p["shape"] == "meyer", who + ": shape " + p["shape"].dump() + ": morlet (the default: a Gaussian centred on the band, its width the FWHM) or meyer (compact edges, half_width octaves)");
  need(p["half_width"].is_number() && p["half_width"].get<double>() > 0, who + ": half_width " + p["half_width"].dump() + ": a positive number of octaves (the Meyer edges' half width; unused by morlet)");
  need(band[1].get<double>() < x.rate / 2, who + ": band " + band.dump() + " reaches past Nyquist (" + std::to_string(x.rate / 2) + " Hz)");
}
std::vector<StepSpec> build() {
  std::vector<StepSpec> r;
  r.push_back({"band_filter", "chunk", {"x"}, {}, {"y"}, Json{{"band", nullptr}, {"shape", "morlet"}, {"half_width", 1.0 / 3.0}},
               "Studio engine/gpu/deviceChain.ts gainTaps · convolveRows (a wavelet band of the Signal chain: toolbox/wavelets/dyadicFrame.ts meyerBand; `shape` morlet, the default: kernels.ts morlet, nxr vortex.morlet_gain); nxr.filters.taps; compute nxr::taps",
               "a field on … × time filtered to a band, zero phase: the band's gain taken to taps once, each chunk convolved with them over its halo",
               [](const Values& ins, const Json& p, const PlanContext&) {
                 const Value& x = ins.at("x");
                 in_loop(x, "band_filter");
                 need(!x.complex, "band_filter: x is complex; the band is taken on the real recording, before the analytic signal");
                 band_check(p, x, "band_filter", false);
                 Value y = made(x);
                 y.facts["band"] = p["band"];
                 return Values{{"y", y}};
               }});
  r.push_back({"analytic", "chunk", {"x"}, {}, {"y"}, Json{{"band", nullptr}, {"shape", "morlet"}, {"half_width", 1.0 / 3.0}},
               "Studio engine/gpu/deviceChain.ts THE ANALYTIC SIGNAL (the band's even taps and its odd taps, hilbertImag, on the same raw rows); compute nxr::taps (gainTaps, convolveRows)",
               "the analytic signal of a band of a real field on … × time (the Signal chain's filter with the transform 'complex'): the band's even taps give the real part, "
               "its odd (Hilbert) taps the imaginary, on the same raw rows; band null: the plain Hilbert, cut at 1024 taps (its lowest frequencies lose their quadrature)",
               [](const Values& ins, const Json& p, const PlanContext&) {
                 const Value& x = ins.at("x");
                 in_loop(x, "analytic");
                 need(!x.complex, "analytic: x is already complex");
                 band_check(p, x, "analytic", true);
                 Value y = made(x);
                 y.complex = true;
                 if (!p["band"].is_null()) y.facts["band"] = p["band"];
                 y.facts["band_shape"] = p["shape"];
                 return Values{{"y", y}};
               }});
  r.push_back({"envelope", "chunk", {"x"}, {}, {"y"}, Json{{"band", nullptr}, {"shape", "morlet"}, {"half_width", 1.0 / 3.0}}, "Studio engine/gpu/deviceChain.ts THE ENVELOPE (√(re² + im²) of the even and the odd pass)",
               "the envelope of a band of a real field on … × time, |analytic| (non-linear: no linear map may follow it, D106)",
               [](const Values& ins, const Json& p, const PlanContext&) {
                 const Value& x = ins.at("x");
                 in_loop(x, "envelope");
                 need(!x.complex, "envelope: x is complex; the envelope is taken of the real field (|analytic|)");
                 band_check(p, x, "envelope", true);
                 Value y = made(x);
                 y.facts["nonlinear"] = true;
                 y.facts["made_by"] = "envelope";
                 if (!p["band"].is_null()) y.facts["band"] = p["band"];
                 return Values{{"y", y}};
               }});
  r.push_back({"frames", "chunk", {"x", "tiling", "within"}, {"within"}, {"y"}, Json{{"half_width", 32}},
               "nxr vortex.band_at_frames (the band's signal at the frames, by FFT resampling: exact for a band-limited signal); here its local form, a Kaiser-windowed sinc (nxr::frametiling), at the START of each tower frame (nxr::tower)",
               "a field sampled on the FRAME TILING (a closed-form spans Selection of TOWER frames, `frametiling compute`: the cycles of one tower level, D135, each a dyadic tile; a frame's time is its tile's start). A linear step: the band's signal x, real or complex "
               "(the analytic signal), is interpolated at each frame's (non-integer) sample position by a Kaiser-windowed sinc of `half_width` samples either side (the step's halo), exact for a signal band-limited well inside the Nyquist band. "
               "A frame belongs to the chunk holding the sample at or before it, so the result does not depend on the chunk size. The output's columns are FRAMES (axis `frames`): the steps after it are pointwise in time "
               "(a linear map, a form, |x|², a series kept) and fold frames; a step that reads the samples' own time refuses it. With `within` (a spans Selection of the recording's time line: the bursts, a stage) only the frames "
               "inside a span are GOOD (the steps after it leave the others out); without it every frame is. Beyond the recording the signal reads 0",
               [](const Values& ins, const Json& p, const PlanContext&) {
                 const Value &x = ins.at("x"), &t = ins.at("tiling");
                 in_loop(x, "frames");
                 need(!nonlinear(x), "frames: x was made by a non-linear step (" + x.facts.value("made_by", std::string("?")) + "): its band limit is gone, it cannot be interpolated");
                 need(t.cls == "selection" && t.row && nxr::io::text(*t.row, "type") == "spans", "frames: tiling must be the frame tiling (`frametiling compute`)");
                 const Json tp = Json::parse(nxr::io::opt_text(*t.row, "params_json").value_or("{}"), nullptr, false);
                 need(tp.is_object() && tp.contains("atlas") && tp["atlas"].contains("frame_level") && tp.contains("frame_tiling") && tp.contains("tiling"), "frames: " + nxr::io::text(*t.row, "name") + " is not a frame tiling (`frametiling compute`)");
                 need(t.manifolds.size() == 1 && x.time && t.manifolds[0] == *x.time, "frames: the frame tiling " + nxr::io::text(*t.row, "name") + " is of another time line than x's");
                 need(std::fabs(tp["frame_tiling"].value("sfreq", 0.0) - x.rate) < 1e-9 * x.rate, "frames: the tiling was made for " + std::to_string(tp["frame_tiling"].value("sfreq", 0.0)) + " samples a second, x has " + std::to_string(x.rate));
                 need(p["half_width"].is_number_integer() && p["half_width"].get<int>() >= 1, "frames: half_width is a whole number of samples, at least 1");
                 if (ins.count("within")) {
                   const Value& w = ins.at("within");
                   need(w.cls == "selection" && w.row && nxr::io::text(*w.row, "type") == "spans" && w.manifolds.size() == 1 && w.manifolds[0] == *x.time, "frames: `within` must be a spans Selection of x's time line");
                 }
                 Value y = made(x);
                 y.facts["axis"] = "frames";
                 y.facts["frames"] = Json{{"tiling", nxr::io::text(*t.row, "name")}, {"fps", tp["frame_tiling"]["fps"]}, {"frame_level", tp["atlas"]["frame_level"]}, {"frame_code0", tp["atlas"]["frame_code0"]}, {"anchor", tp["atlas"]["anchor"]}, {"half_width", p["half_width"]}, {"within", ins.count("within") ? Json(nxr::io::text(*ins.at("within").row, "name")) : Json(nullptr)}};
                 y.rate = tp["frame_tiling"]["fps"].get<double>();
                 y.origin = tp["tiling"].value("origin", 0.0);
                 return Values{{"y", y}};
               }});
  r.push_back({"apply_kernel", "chunk", {"x", "kernel", "tiles"}, {"tiles"}, {"y"}, Json{{"level", nullptr}, {"levels", nullptr}, {"glued", false}, {"transform", "auto"}, {"modes", 0}, {"order", 0}, {"tolerance", 0.01}, {"envelope", "temporal"}, {"spatial_shape", "mexhat"}, {"form", "none"}},
               "Studio adapters/nxr kernelOf (the kernel padded to every member, its from_selection's columns); nxr.io.adapter.kernel_from_rows; with `tiles`, the level's Weyl band folded into the kernel (nxr::filterbank, pipelines/spatial_band.h)",
               "a linear operator from the field's first factor applied to it: x [members × time] → K x [target × time] (an imaging kernel: sensors → sources). `levels` (; instead of `level`) is ONE operator a hat level, all bound in preparation: the output is [vertex × level] (the rows of a vertex's levels consecutive, then a vertex's components: row = (vertex · levels + level) · components + component; `facts.levels` names them), the scale-space the spatial events' scale selection reads. `glued` (default false; a `form` is always glued) evaluates the band on EVERY vertex of the glued surface for a field with no form too (a free current: each component through the band, the bridge filled). `form` folds a LINEAR form of the free current into the same operator, in preparation: kernel (x y z a vertex) → its tangent part → curl_tan (or div_tan; `normal`: n·x) → the spatial band, ONE linear map before any non-linear step (D106); the output is ONE scalar a vertex (complex when x is: the analytic signal's curl; nxr vortex.Cortex.readings: div_curl(tangent(q)), then the hat of vortex_cores.hat_operator). With `tiles` the output is on EVERY vertex of the GLUED surface. With `tiles` (the target surface's "
               "tile ladder) and `level`: M x with M = (F_l ⊗ I) K, F_l the level's Weyl band on each component of the kernel's rows, bound ONCE in preparation (a linear map reads the chain "
               "before any non-linear step, D106; no graph filtering per sample). `envelope` temporal (the default: M x) | spatiotemporal. `spatial_shape` mexhat | meyer (the level IS its Weyl band: compact, the levels add up). `transform` auto, eigen, chebyshev (order 0: chosen from `tolerance`) or exact (dense modes of the mesh, ≤ 4096 vertices)",
               [](const Values& ins, const Json& p, const PlanContext& c) {
                 const Value &x = ins.at("x"), &k = ins.at("kernel");
                 need(k.cls == "operator", "apply_kernel: kernel must be an operator");
                 in_loop(x, "apply_kernel", true);
                 need(!nonlinear(x), "apply_kernel: x was made by a non-linear step (" + x.facts.value("made_by", std::string("?")) + "); a linear map reads the chain before any non-linear transform (D106)");
                 need(x.manifolds[0] == k.manifolds[0], "apply_kernel: the kernel maps from manifold " + k.manifolds[0] + ", x's first factor is " + x.manifolds[0] + ": not the same manifold");
                 need(!k.manifolds[1].empty(), "apply_kernel: the kernel has no target manifold");
                 Value y = made(x);
                 y.manifolds[0] = k.manifolds[1];
                 y.value_type = k.value_type;
                 need(p["envelope"] == "temporal" || p["envelope"] == "spatiotemporal", "apply_kernel: envelope " + p["envelope"].dump() + ": temporal or spatiotemporal");
                 if (p["envelope"] == "spatiotemporal") {
                   need(ins.count("tiles"), "apply_kernel: envelope spatiotemporal needs `tiles` and `level` (the Riesz part is the spatial band's)");
                   need(k.value_type == "scalar" || k.value_type == "vector3", "apply_kernel: envelope spatiotemporal: the kernel's field type is " + k.value_type + "; one number or three components a vertex");
                   y.facts["monogenic"] = Json{{"components", k.value_type == "scalar" ? 1 : 3}};
                   y.facts["envelope"] = "spatiotemporal";
                   y.value_type = "monogenic";
                 }
                 y.complex = x.complex || k.complex;
                 y.unit.reset();
                 need(p["form"].is_string() && (p["form"] == "none" || p["form"] == "curl_tan" || p["form"] == "div_tan" || p["form"] == "normal"), "apply_kernel: form " + p["form"].dump() + ": none (the default), curl_tan, div_tan or normal");
                 if (p["form"] != "none") {
                   need(k.value_type == "vector3", "apply_kernel: form " + p["form"].get<std::string>() + " reads a free current (x y z a vertex); the kernel's field type is " + k.value_type);
                   need(p["envelope"] == "temporal", "apply_kernel: a form gives one number a vertex; envelope spatiotemporal needs the current itself");
                   need(manifold_type(c, k.manifolds[1]) == "surface", "apply_kernel: form: the kernel's target is not a surface (a form is taken on the cortex)");
                   y.value_type = "scalar";
                   y.facts["form"] = p["form"];
                   if (ins.count("tiles")) {
                     // (one cortex): with the spatial band the form is on EVERY vertex of the GLUED surface (the bridge, the corpus callosum, filled by the band), so the output's surface is the glued one:
                     // the stored `<surface>_glued` of a cortex of two hemispheres; the surface itself for one closed piece. Preparation checks it is the package's.
                     if (const auto gid = c.bound.glued_surface_id(k.manifolds[1])) y.manifolds[0] = *gid;
                     y.facts["glued"] = true;
                   }
                 }
                 need(p["level"].is_null() || p["levels"].is_null(), "apply_kernel: `level` (one hat) or `levels` (a list of hats), not both");
                 need(ins.count("tiles") == ((p["level"].is_null() && p["levels"].is_null()) ? 0u : 1u), "apply_kernel: `tiles` (the target surface's tile ladder) and `level` (or `levels`) come together: the level's band is the ladder's");
                 need(p["glued"].is_boolean(), "apply_kernel: glued is true or false");
                 need(!p["glued"].get<bool>() || ins.count("tiles"), "apply_kernel: glued evaluates a spatial band on the glued surface: it needs `tiles` and a `level`");
                 if (ins.count("tiles")) {
                   const Value& t = ins.at("tiles");
                   need(t.cls == "selection" && t.row && nxr::io::text(*t.row, "type") == "ladder" && t.manifolds[0] == k.manifolds[1],
                        "apply_kernel: tiles must be the tile ladder (`tiles compute`) of the kernel's target surface");
                   need(manifold_type(c, k.manifolds[1]) == "surface", "apply_kernel: the kernel's target is not a surface");
                   std::vector<int> lvs;
                   if (!p["levels"].is_null()) {
                     need(p["levels"].is_array() && !p["levels"].empty(), "apply_kernel: levels is a list of the ladder's levels, e.g. [4, 5, 6, 7, 8]");
                     for (const auto& e : p["levels"]) { need(e.is_number_integer() && e.get<int>() >= 0, "apply_kernel: levels " + p["levels"].dump() + ": whole numbers, the ladder's levels (0, 1, …)"); lvs.push_back(e.get<int>()); }
                     need(std::set<int>(lvs.begin(), lvs.end()).size() == lvs.size(), "apply_kernel: levels " + p["levels"].dump() + " repeats a level");
                     need(p["envelope"] == "temporal", "apply_kernel: several levels need the temporal envelope (the monogenic field is of one level)");
                   } else {
                     need(p["level"].is_number_integer() && p["level"].get<int>() >= 0, "apply_kernel: level " + p["level"].dump() + ": a whole number, the ladder's level (0, 1, …)");
                     lvs.push_back(p["level"].get<int>());
                   }
                   for (const int l : lvs) need(l < nxr::io::integer(*t.row, "n_members"), "apply_kernel: the tile ladder has no level " + std::to_string(l) + " (its levels are 0 … " + std::to_string(nxr::io::integer(*t.row, "n_members") - 1) + ")");
                   const int l = lvs[0];
                   need(p["transform"] == "auto" || p["transform"] == "eigen" || p["transform"] == "chebyshev" || p["transform"] == "exact", "apply_kernel: transform " + p["transform"].dump() + ": auto, eigen, chebyshev or exact");
                   need(p["spatial_shape"] == "mexhat" || p["spatial_shape"] == "meyer", "apply_kernel: spatial_shape " + p["spatial_shape"].dump() + ": mexhat (the default: the level's Mexican hat) or meyer (the level's Weyl band)");
                   need(p["modes"].is_number_integer() && p["modes"].get<int>() >= 0, "apply_kernel: modes is at least 0 (0: the standard package size, or what the band needs)");
                   need(p["order"].is_number_integer() && p["order"].get<int>() >= 0 && p["tolerance"].is_number() && p["tolerance"].get<double>() > 0, "apply_kernel: order is at least 0 (0: chosen), tolerance above 0");
                   y.facts["spatial_band"] = Json{{"level", p["levels"].is_null() ? Json(l) : Json(nullptr)}, {"ladder", nxr::io::text(*t.row, "id")}, {"transform", p["transform"]}, {"spatial_shape", p["spatial_shape"]}};
                   if (!p["levels"].is_null()) { y.facts["spatial_band"]["levels"] = p["levels"]; y.facts["levels"] = p["levels"]; }
                   if (p["glued"].get<bool>() && p["form"] == "none") {
                     //: a free current (no form) evaluated on EVERY vertex of the glued surface too (each component through the band; the bridge carries 0 before it)
                     if (const auto gid = c.bound.glued_surface_id(k.manifolds[1])) y.manifolds[0] = *gid;
                     y.facts["glued"] = true;
                   }
                 }
                 return Values{{"y", y}};
               }});
  r.push_back({"magnitude", "chunk", {"x"}, {}, {"y"}, Json::object(), "compute: |x| per element, the norm over its components (the Studio's abs of a ℍ / ℂ field; nxr::phase::power's square root)",
               "|x| per vertex per sample, the norm over the vertex's components and the complex part (a free current: no orientation assumed): pointwise, non-linear (nothing linear may follow, D106)",
               [](const Values& ins, const Json&, const PlanContext&) {
                 const Value& x = ins.at("x");
                 in_loop(x, "magnitude", true);
                 need(x.value_type == "scalar" || x.value_type == "vector3" || x.value_type == "monogenic", "magnitude: x is " + x.value_type + "; one number, three components a vertex, or a monogenic field (`apply_kernel` with envelope spatiotemporal)");
                 Value y = made(x);
                 y.value_type = "scalar";
                 y.complex = false;
                 y.facts["nonlinear"] = true;
                 y.facts["made_by"] = "magnitude";
                 y.facts.erase("field_type");
                 y.facts.erase("monogenic");
                 if (x.value_type == "monogenic") y.facts["envelope"] = "spatiotemporal";
                 return Values{{"y", y}};
               }});
  r.push_back({"tile_max", "chunk", {"x", "tiles"}, {}, {"y"}, Json{{"level", nullptr}, {"groups", "whole_brain"}},
               "docs/specs/2026-11-13-cortical-events.md, Global bursts; compute nxr::partition::meanOverMembers (the area-weighted mean per member, a column at a time)",
               "per SAMPLE: the area-weighted mean of |z| per spatial tile of the level (a spatial reduction, time left unreduced), then the MAX over the tiles of each group (`whole_brain`: one group; "
               "`per_hemisphere`: one per connected piece of the surface): [groups × samples]. The [tiles × samples] intermediate lives inside one chunk and is never stored",
               [](const Values& ins, const Json& p, const PlanContext& c) {
                 const Value &x = ins.at("x"), &t = ins.at("tiles");
                 in_loop(x, "tile_max");
                 need(x.facts.value("made_by", std::string()) == "magnitude", "tile_max: x is |z| per vertex (`magnitude`); x was made by " + x.facts.value("made_by", std::string("?")));
                 need(manifold_type(c, x.manifolds[0]) == "surface", "tile_max: x's first factor is not a surface");
                 need(t.cls == "selection" && t.row && nxr::io::text(*t.row, "type") == "ladder" && t.manifolds[0] == x.manifolds[0], "tile_max: tiles must be the tile ladder of x's surface");
                 need(p["level"].is_number_integer() && p["level"].get<int>() >= 0 && p["level"].get<int>() < nxr::io::integer(*t.row, "n_members"), "tile_max: level " + p["level"].dump() + " is not a level of the tile ladder");
                 need(p["groups"] == "whole_brain" || p["groups"] == "per_hemisphere", "tile_max: groups " + p["groups"].dump() + ": whole_brain or per_hemisphere");
                 need(x.facts.contains("spatial_band") && x.facts["spatial_band"]["level"] == p["level"], "tile_max: the tiles' level is the spatial band's (x was filtered to level " + (x.facts.contains("spatial_band") ? x.facts["spatial_band"]["level"].dump() : std::string("none")) + ")");
                 Value y = made(x);
                 y.manifolds = {*x.time};
                 y.value_type = "scalar";
                 y.facts["made_by"] = "tile_max";
                 y.facts["groups"] = p["groups"];
                 y.facts["space_level"] = p["level"];
                 return Values{{"y", y}};
               }});
  r.push_back({"collect", "accumulate", {"x"}, {}, {"y"}, Json::object(), "the samples of the run kept in order (the chunk service's own columns): the trace at the recording's rate",
               "the own columns of every chunk appended: a [rows × samples] series of the span (MEMORY: rows × samples doubles, held until the run ends). A SERIES, not a reduction: what the sample-level fit (`risefall`) reads.",
               [](const Values& ins, const Json&, const PlanContext&) {
                 const Value& x = ins.at("x");
                 in_loop(x, "collect");
                 need(!x.complex, "collect: x is complex");
                 Value y = made(x);
                 y.cls = "series";
                 y.chunked = false;
                 return Values{{"y", y}};
               }});
  r.push_back({"reduce_cells", "after", {"x"}, {}, {"y"}, Json{{"groups", "whole_brain"}},
               "docs/specs/2026-11-13-cortical-events.md, Global bursts; the measure `max` over the groups' partition of the spatial tiles",
               "a measurement over a JOINT partition [spatial tiles × time tiles] reduced over the spatial partition to [groups × time tiles] by the MAX over each group's tiles (a tile NaN is left out; a time tile "
               "NaN in every tile stays NaN): the trace the time events are found on. Its tiles are the TIME level's (so `detect_spans` and `rollup_events` take it unchanged)",
               [](const Values& ins, const Json& p, const PlanContext& c) {
                 const Value& x = ins.at("x");
                 need(x.cls == "measurement" && x.facts.contains("over") && x.facts["over"].value("joint", false), "reduce_cells: x must be a measurement over a joint partition (`measure` with a joint `over`)");
                 need(x.facts.value("measure", std::string()) != "phase_current", "reduce_cells: x is a vector measurement (the phase current); the max over tiles is of a scalar");
                 need(p["groups"] == "whole_brain" || p["groups"] == "per_hemisphere", "reduce_cells: groups " + p["groups"].dump() + ": whole_brain or per_hemisphere");
                 const std::int64_t S = x.facts["over"]["space_tiles"].get<std::int64_t>(), l = x.facts["over"]["space_level"].get<std::int64_t>();
                 need(S % (std::int64_t(1) << l) == 0, "reduce_cells: " + std::to_string(S) + " tiles are not a whole number of pieces of 2^" + std::to_string(l));
                 if (p["groups"] == "per_hemisphere") need(S >> l == 2, "reduce_cells: per_hemisphere needs a surface of two pieces (the ladder's level has " + std::to_string(S >> l) + ")");
                 const auto tsel = c.bound.selection(x.facts["over"]["selection_id"].get<std::string>());
                 need(tsel.has_value(), "reduce_cells: the joint partition is not among the subject's rows");
                 const Json j = Json::parse(nxr::io::text(*tsel, "params_json"))["joint"];
                 const auto tl = c.bound.selection(j["time"].get<std::string>());
                 need(tl.has_value(), "reduce_cells: the joint partition's time level is not among the subject's rows");
                 Value y = made(x);
                 Json over = x.facts["over"];
                 over.erase("joint"); over.erase("space_level"); over.erase("space_tiles");
                 over["selection_id"] = nxr::io::text(*tl, "id"); over["selection"] = nxr::io::text(*tl, "name");
                 y.facts["over"] = over;
                 y.facts["reduced"] = Json{{"groups", p["groups"]}, {"space_level", l}, {"by", "max over the spatial tiles"}};
                 return Values{{"y", y}};
               }});
  r.push_back({"risefall", "after", {"x", "events"}, {}, {"t_on", "t_off", "tau_rise", "tau_fall", "fit_baseline", "fit_height", "fit_r2", "plateau_duration"},
               Json{{"model", "linear"}, {"margin", 1.0}, {"smooth", 0.1}, {"tau_max", 1.0}, {"min_r2", nullptr}, {"durations", "seconds"}, {"margin_cycles", 10.0}, {"smooth_cycles", 1.0}, {"tau_max_cycles", 10.0}},
               "Rheome +rheome/+detect/risefall.m (the double-sigmoid fit; linear and quadrature models; the posterior model is not built); compute nxr::risefall",
               "each event (the spans `detect_spans` made) fitted on the SAMPLE series (`collect`) over [start − margin, stop + margin]: the half-rise and half-fall times, the rise and fall times, the baseline and height, r², "
               "the plateau's duration; per-event measurements [rows × events] on the row that found the event, NaN elsewhere and for a fit that is not ok; `min_r2` (default none) sets the fit of an event below it NaN. "
               "`durations` seconds (the default: margin, smooth, tau_max) | cycles (nxr `burst_stages`' own: margin_cycles 10, smooth_cycles 1, tau_max_cycles 10 of the band's centre; the seconds parameters are then not used)",
               [](const Values& ins, const Json& p, const PlanContext&) {
                 const Value &x = ins.at("x"), &ev = ins.at("events");
                 need(x.cls == "series" && x.time, "risefall: x must be the sample series of the run (`collect`)");
                 need(ev.cls == "spans" && ev.facts.contains("detect") && ev.facts.contains("over"), "risefall: events must be the spans `detect_spans` made");
                 try { nxr::risefall::parseModel(p["model"].is_string() ? p["model"].get<std::string>() : std::string("?")); } catch (const std::invalid_argument& e) { refuse(std::string("risefall: ") + e.what()); }
                 need(p["durations"] == "seconds" || p["durations"] == "cycles", "risefall: durations " + p["durations"].dump() + ": seconds (margin, smooth, tau_max) or cycles (margin_cycles, smooth_cycles, tau_max_cycles of the band's centre)");
                 for (const char* k : {"margin", "smooth", "tau_max", "margin_cycles", "smooth_cycles", "tau_max_cycles"}) need(p[k].is_number() && p[k].get<double>() >= 0, std::string("risefall: ") + k + " is a number of at least 0");
                 need(p["tau_max"].get<double>() > 0 && p["tau_max_cycles"].get<double>() > 0, "risefall: tau_max is above 0");
                 need(p["durations"] == "seconds" || ev.facts["over"].contains("resolves_hz"), "risefall: durations cycles need the band the tiles resolve (the events' `over` names none)");
                 need(p["min_r2"].is_null() || (p["min_r2"].is_number() && p["min_r2"].get<double>() <= 1), "risefall: min_r2 is null (every event kept) or a number up to 1 (labelalpha's is 0.7)");
                 Values out;
                 for (const char* m : {"t_on", "t_off", "tau_rise", "tau_fall", "fit_baseline", "fit_height", "fit_r2", "plateau_duration"}) {
                   Value y = made(ev);
                   y.cls = "measurement";
                   y.facts["measure"] = m;
                   y.facts["over"] = Json{{"events", true}, {"selection", "the events of this run"}, {"tile_level", ev.facts["over"]}};
                   if (std::string(m) == "t_on" || std::string(m) == "t_off" || std::string(m).rfind("tau", 0) == 0 || std::string(m) == "plateau_duration") y.unit = "s";
                   out[m] = y;
                 }
                 return out;
               }});
  r.push_back({"form", "chunk", {"x"}, {}, {"y"}, Json{{"form", nullptr}},
               "Studio differential/fieldForms.ts FORMS, formValues (ℂ-extended: compute nxr::forms::formComplex)",
               "a form of a field of its type (ℝ f · ℂ abs div curl … v Jv · ℍ absq nx w xt x …): linear forms keep a complex field's phase, abs / absq give its envelope",
               [](const Values& ins, const Json& p, const PlanContext& c) {
                 const Value& x = ins.at("x");
                 need(x.cls == "field", "form: x must be a field");
                 need(p["form"].is_string(), "form: name the form (with: {form: nx})");
                 const auto id = nxr::forms::formOf(p["form"].get<std::string>());
                 const auto t = field_type(x);
                 std::string have;
                 for (const auto& f : nxr::forms::formsOf(t)) have += (have.empty() ? "" : " ") + std::string(f.key);
                 const auto info = id ? nxr::forms::formInfo(t, *id) : std::nullopt;
                 need(info.has_value(), "form: '" + p["form"].get<std::string>() + "' is not a form of a " + type_name(t) + " field (its forms: " + have + ")");
                 need(t == nxr::forms::FieldType::Real || manifold_type(c, x.manifolds[0]) == "surface", "form: x's first factor is not a surface (a vector field's forms are taken on the cortex)");
                 Value y = made(x);
                 const bool linear = nxr::forms::isLinear(*id);
                 y.value_type = info->out == nxr::forms::FieldType::Real ? "scalar" : info->out == nxr::forms::FieldType::Quaternion ? "quaternion" : "vector3";
                 if (info->out == nxr::forms::FieldType::Complex) y.facts["field_type"] = "complex";
                 else y.facts.erase("field_type");
                 y.facts["form"] = p["form"];
                 if (!linear) { y.complex = false; y.facts["nonlinear"] = true; y.facts["made_by"] = "form " + p["form"].get<std::string>(); }
                 if (!linear || nxr::forms::isDivCurl(*id)) y.unit.reset();
                 return Values{{"y", y}};
               }});
  r.push_back({"power", "chunk", {"x"}, {}, {"y"}, Json::object(), "compute nxr::phase::power (|z|², summed over the components); the Studio's absq",
               "|x|² per element per sample, summed over its components (a real or complex field): pointwise; a reduction over time is a `measure` after it",
               [](const Values& ins, const Json&, const PlanContext&) {
                 const Value& x = ins.at("x");
                 in_loop(x, "power", true);
                 Value y = made(x);
                 y.value_type = "scalar";
                 y.complex = false;
                 y.facts["nonlinear"] = true;
                 y.facts["made_by"] = "power";
                 y.facts["of_complex"] = x.complex;
                 y.facts.erase("field_type");
                 if (x.unit) y.unit = "(" + *x.unit + ")^2";
                 return Values{{"y", y}};
               }});
  r.push_back({"phase_current", "chunk", {"x"}, {}, {"y"}, Json::object(), "nxr detect/core_phase.py PhaseReader (Im(conj(z) grad z), the GV gradient); compute nxr::phase::phaseCurrent",
               "the phase current j = Im(z̄ ∇z) = |z|² ∇φ on the tangent plane, per vertex per sample (three numbers a vertex: x y z)",
               [](const Values& ins, const Json&, const PlanContext& c) {
                 const Value& z = ins.at("x");
                 phase_input(z, c, "phase_current");
                 Value y = made(z);
                 y.value_type = "vector3";
                 y.complex = false;
                 y.facts = Json{{"nonlinear", true}, {"made_by", "phase_current"}};
                 for (const char* k : {"band", "spatial_band"}) if (z.facts.contains(k)) y.facts[k] = z.facts[k];   // the pair of bands a joint partition names (the level IS the band)
                 y.unit.reset();
                 return Values{{"y", y}};
               }});
  r.push_back({"gradient_power", "chunk", {"x"}, {}, {"y"}, Json::object(), "nxr detect/core_phase.py (|grad z|²); compute nxr::phase::gradientPower",
               "|∇z|² of the tangent gradient of a complex field, per vertex per sample",
               [](const Values& ins, const Json&, const PlanContext& c) {
                 const Value& z = ins.at("x");
                 phase_input(z, c, "gradient_power");
                 Value y = made(z);
                 y.value_type = "scalar";
                 y.complex = false;
                 y.facts = Json{{"nonlinear", true}, {"made_by", "gradient_power"}};
                 y.unit.reset();
                 return Values{{"y", y}};
               }});
  r.push_back({"measure", "accumulate", {"x", "exclude", "over"}, {"exclude", "over"}, {"y"}, Json{{"measure", "mean"}, {"bad_segments", "exclude"}},
               "Studio backend/src/db/measurements.ts reduceTiles (compute nxr::partition::reduceTiles), folded chunk by chunk; with `over`, per tile of the constant-Q level of the "
               "band (the registry's amplitude / power, D84: the level IS the band); nxr.pipelines.parameters.bad_mask (the bad segments' rule)",
               "a field reduced over the run's span on its time line, per element: mean · mean_square · max · min · abs_max; or, with `over` (a constant-Q ladder on its "
               "time line), per element per TILE of the level whose band is the input's: amplitude (mean |z| of a band's envelope) · power (mean |z|² of a band's "
               "analytic signal); the samples in excluded spans left out",
               [](const Values& ins, const Json& p, const PlanContext& c) {
                 const Value& x = ins.at("x");
                 in_loop(x, "measure", ins.count("over") == 0);
                 need(!x.complex, "measure: x is complex; measure a real quantity of it (power, a form, the phase current)");
                 need(p["bad_segments"] == "exclude" || p["bad_segments"] == "ignore", "measure: bad_segments " + p["bad_segments"].dump() + ": exclude or ignore");
                 if (ins.count("exclude")) {
                   const Value& ex = ins.at("exclude");
                   need(ex.cls == "selection" && ex.manifolds[0] == *x.time, "measure: exclude must be a selection on x's time line (" + *x.time + "), not on " + ex.manifolds[0]);
                 }
                 if (ins.count("over") && ins.at("over").row && nxr::io::text(*ins.at("over").row, "type") == "set") {
                   // OVER A JOINT PARTITION (space × time, `joint compute`): per cell, a spatial tile of level l × a time tile of the band's level (the pair names both bands)
                   const Value& ov = ins.at("over");
                   const Json pj = Json::parse(nxr::io::opt_text(*ov.row, "params_json").value_or("{}"), nullptr, false);
                   need(pj.is_object() && pj.contains("joint"), "measure: over a `set` Selection must be a joint partition of space × time (`joint compute`): " + nxr::io::text(*ov.row, "name") + " records none");
                   const Json& j = pj["joint"];
                   const std::string jm = p["measure"].is_string() ? p["measure"].get<std::string>() : std::string();
                   need(jm == "amplitude" || jm == "envelope_max" || jm == "power" || jm == "phase_current",
                        "measure: over a joint cell the measures are amplitude (the area-weighted mean |z|), envelope_max (the largest |z|), power (the area-weighted mean of the `power` step's |z|^2) and phase_current (the area-weighted mean of the `phase_current` step's vector)");
                   const std::string mk = x.facts.value("made_by", std::string("?"));
                   if (jm == "power") need(mk == "power" && x.facts.value("of_complex", false), "measure: power over a joint cell is taken of the `power` step's |z|^2 of a complex field; x was made by " + mk);
                   else if (jm == "phase_current") need(mk == "phase_current" && x.value_type == "vector3", "measure: phase_current over a joint cell is taken of the `phase_current` step's vector (three rows a vertex); x was made by " + mk);
                   else need(mk == "magnitude", "measure: " + jm + " over a joint cell is taken of |z| per vertex (`magnitude`); x was made by " + mk);
                   need(manifold_type(c, x.manifolds[0]) == "surface", "measure: over a joint cell x's first factor must be a surface");
                   need(x.facts.contains("band") && x.facts.contains("spatial_band"), "measure: x carries a time band and a spatial band (`analytic` with a band, `apply_kernel` with `tiles` and `level`): the pair of levels the joint partition names");
                   const auto tl = c.bound.selection(j["time"].get<std::string>());
                   const auto sl = c.bound.selection(j["space"].get<std::string>());
                   need(tl && sl, "measure: the joint partition's levels are not among the subject's rows");
                   need(nxr::io::text(*sl, "manifold_id") == x.manifolds[0], "measure: the joint partition is over another surface than x's");
                   need(j["space_level"].get<int>() == x.facts["spatial_band"]["level"].get<int>(),
                        "measure: the joint partition's spatial level is " + std::to_string(j["space_level"].get<int>()) + ", x was filtered to level " + std::to_string(x.facts["spatial_band"]["level"].get<int>()) + " (the pair names both bands)");
                   const Json tp = Json::parse(nxr::io::opt_text(*tl, "params_json").value_or("{}"), nullptr, false);
                   const double lo = x.facts["band"][0].get<double>(), hi = x.facts["band"][1].get<double>();
                   need(tp.is_object() && tp.contains("resolves_hz") && std::abs(tp["resolves_hz"]["lo"].get<double>() - lo) <= 1e-9 * std::max(1.0, lo) && std::abs(tp["resolves_hz"]["hi"].get<double>() - hi) <= 1e-9 * std::max(1.0, hi),
                        "measure: the joint partition's time level resolves " + (tp.contains("resolves_hz") ? tp["resolves_hz"].dump() : std::string("?")) + " Hz, x's band is [" + std::to_string(lo) + ", " + std::to_string(hi) + "] (the level IS the band, D84)");
                   const std::int64_t width = tp.value("window_samples", std::int64_t(0));
                   need(width > 0 && std::llround(tp["tiling"]["width"].get<double>() * x.rate) == width && std::abs(tp["tiling"]["origin"].get<double>() - x.origin) < 0.5 / x.rate,
                        "measure: " + nxr::io::text(*tl, "name") + " is not a contiguous tiling of x's time line from its first sample");
                   Value y = made(x);
                   y.cls = "measurement";
                   y.manifolds = {*x.time};
                   y.chunked = false;
                   y.facts["measure"] = p["measure"];
                   y.facts["over"] = Json{{"selection_id", nxr::io::text(*ov.row, "id")}, {"selection", nxr::io::text(*ov.row, "name")}, {"level", j["time_level"]}, {"window_samples", width}, {"tiles", j["frames"]},
                                         {"width_s", static_cast<double>(width) / x.rate}, {"origin", x.origin}, {"resolves_hz", tp["resolves_hz"]}, {"joint", true}, {"space_level", j["space_level"]}, {"space_tiles", j["tiles"]}};
                   return Values{{"y", y}};
                 }
                 if (ins.count("over")) {
                   const Value& ov = ins.at("over");
                   need(ov.cls == "selection" && ov.row && nxr::io::text(*ov.row, "type") == "ladder" && ov.manifolds[0] == *x.time,
                        "measure: over must be a constant-Q ladder on x's time line (`constantq compute`)");
                   need(p["measure"] == "amplitude" || p["measure"] == "power" || p["measure"] == "median" || p["measure"] == "envelope_max", "measure: over a constant-Q level the measures are amplitude, power, median and envelope_max (the level IS the band, D84), not " + p["measure"].dump());
                   if (p["measure"] == "amplitude" || p["measure"] == "median" || p["measure"] == "envelope_max")
                     need(x.facts.value("made_by", std::string()) == "envelope", "measure: " + p["measure"].get<std::string>() + " is taken of a band's envelope (the median of |z| per tile; amplitude is its mean, envelope_max its largest value); x was made by " + x.facts.value("made_by", std::string("no step")));
                   else
                     need(x.facts.value("made_by", std::string()) == "power" && x.facts.value("of_complex", false),
                          "measure: power is the mean |z|² of a band's analytic signal (`analytic` then `power`)");
                   need(x.facts.contains("band"), "measure: x carries no band (its band_filter, analytic or envelope named none)");
                   const TimeLevel lv = level_of_band(c.bound, nxr::io::text(*ov.row, "id"), x.facts["band"], x.rate, x.origin);
                   Value y = made(x);
                   y.cls = "measurement";
                   y.manifolds = {*x.time};
                   y.chunked = false;
                   y.facts["measure"] = p["measure"];
                   y.facts["over"] = Json{{"selection_id", lv.id}, {"selection", lv.name}, {"level", lv.level}, {"window_samples", lv.width}, {"tiles", lv.count},
                                         {"width_s", static_cast<double>(lv.width) / x.rate}, {"origin", x.origin}, {"resolves_hz", lv.resolves}};
                   if (p["measure"] == "power" && x.unit) y.unit = x.unit;
                   return Values{{"y", y}};
                 }
                 static const std::vector<std::string> kMeasures = {"mean", "mean_square", "max", "min", "abs_max", "median"};
                 need(p["measure"].is_string() && std::find(kMeasures.begin(), kMeasures.end(), p["measure"].get<std::string>()) != kMeasures.end(),
                      "measure: measure " + p["measure"].dump() + ": one of mean, mean_square, max, min, abs_max, median (amplitude, power and median with `over`)");
                 Value y = made(x);
                 y.manifolds.erase(std::remove(y.manifolds.begin(), y.manifolds.end(), *x.time), y.manifolds.end());
                 y.time.reset();
                 y.rate = 0;
                 y.chunked = false;
                 y.facts["measure"] = p["measure"];
                 if (p["measure"] == "mean_square" && x.unit) y.unit = "(" + *x.unit + ")^2";
                 return Values{{"y", y}};
               }});
  r.push_back({"summarize", "after", {"x"}, {}, {"y"}, Json{{"summaries", Json::array({"median", "mad"})}, {"combine", "each"}, {"ddof", 1}},
               "a statistic across a partition's members; the summaries of compute nxr::stats (numpy median and linear percentile, Rheome's robust SD, MATLAB's sd, kurtosis, skewness)",
               "a measurement of the run summarised over its partition's members into a STATISTIC (field -> selection/measurement -> statistic): per row (each), pooled, or of the mean over rows",
               [](const Values& ins, const Json& p, const PlanContext&) {
                 const Value& x = ins.at("x");
                 need(x.cls == "measurement" && x.facts.contains("over"), "summarize: x must be a measurement over a partition (`measure` with `over`)");
                 need(x.facts.value("measure", std::string()) != "phase_current", "summarize: x is a vector measurement (the phase current: x y z); the statistics across the bursts are not built (a null is the maintainers' choice)");
                 need(!(x.facts["over"].value("stored_events", false) && x.facts["over"].contains("space_tiles")), "summarize: x is a measurement of the cells over stored events (rows are spatial tiles); its statistics are not built");
                 need(p["summaries"].is_array() && !p["summaries"].empty(), "summarize: summaries is a list (median, mad, percentile:75, …)");
                 std::vector<std::string> names;
                 for (const auto& n : p["summaries"]) { need(n.is_string(), "summarize: a summary is a name"); names.push_back(n.get<std::string>()); }
                 try { nxr::stats::parseSummaries(names); } catch (const std::invalid_argument& e) { refuse(std::string("summarize: ") + e.what()); }
                 need(p["combine"] == "each" || p["combine"] == "pool" || p["combine"] == "mean", "summarize: combine " + p["combine"].dump() + ": each, pool or mean");
                 Value y = made(x);
                 y.cls = "statistic";
                 y.facts["summaries"] = p["summaries"];
                 y.facts["combine"] = p["combine"];
                 return Values{{"y", y}};
               }});
  r.push_back({"fit_gate", "after", {"events", "fit_r2"}, {}, {"spans"}, Json{{"min_r2", 0.7}},
               "nxr vortex.burst_stages (a fit is kept when risefall is ok, not at its bound, and r² ≥ min_r2 = labelalpha's 0.7); compute nxr::risefall::passesGate",
               "the events `detect_spans` made, kept when their rise-fall fit is good: ok, r² ≥ min_r2 and neither time at its bound (within 1% of tau_max). Gives a spans Selection of the KEPT events (the fits and the unfiltered spans stay "
               "as they are; each kept event's ordinal among the detected ones is recorded)",
               [](const Values& ins, const Json& p, const PlanContext&) {
                 const Value &ev = ins.at("events"), &r2 = ins.at("fit_r2");
                 need(ev.cls == "spans" && ev.facts.contains("detect") && ev.facts.contains("over"), "fit_gate: events must be the spans `detect_spans` made");
                 need(r2.cls == "measurement" && r2.facts.value("measure", std::string()) == "fit_r2" && r2.facts.contains("over") && r2.facts["over"].value("events", false), "fit_gate: fit_r2 must be `risefall`'s fit_r2 of these events");
                 need(p["min_r2"].is_number() && p["min_r2"].get<double>() <= 1, "fit_gate: min_r2 is a number up to 1 (labelalpha's, nxr's: 0.7)");
                 Value y = made(ev);
                 y.facts["gate"] = Json{{"min_r2", p["min_r2"]}, {"tau", "not at its bound"}};
                 return Values{{"spans", y}};
               }});
  r.push_back({"burst_stages", "after", {"events", "t_on", "t_off", "tau_rise", "tau_fall", "kept"}, {"kept"}, {"onset", "within", "offset"}, Json::object(),
               "nxr vortex.burst_stages (the build-up tOn ± 2 tauR, the plateau, the decay tOff ± 2 tauF of rheome.flow.labelalpha's episodes); compute nxr::risefall::stagesOf",
               "each event's STAGES from its stored rise-fall measurements: ONSET [t_on − 2 τ_rise, t_on + 2 τ_rise], WITHIN the plateau [t_on + 2 τ_rise, t_off − 2 τ_fall] (none when they meet), OFFSET [t_off − 2 τ_fall, t_off + 2 τ_fall], "
               "clipped to the run; three spans Selections, one span a stage of an event whose fit is finite (an event with no fit has none). With `kept` (`fit_gate`) only the kept events have stages. The edges are seconds, not tile edges",
               [](const Values& ins, const Json&, const PlanContext&) {
                 const Value& ev = ins.at("events");
                 need(ev.cls == "spans" && ev.facts.contains("detect") && ev.facts.contains("over"), "burst_stages: events must be the spans `detect_spans` made");
                 for (const char* m : {"t_on", "t_off", "tau_rise", "tau_fall"}) {
                   const Value& v = ins.at(m);
                   need(v.cls == "measurement" && v.facts.value("measure", std::string()) == m && v.facts.contains("over") && v.facts["over"].value("events", false), std::string("burst_stages: ") + m + " must be `risefall`'s " + m + " of these events");
                 }
                 if (ins.count("kept")) need(ins.at("kept").cls == "spans" && ins.at("kept").facts.contains("gate"), "burst_stages: kept must be the spans `fit_gate` made");
                 Values out;
                 for (const char* n : {"onset", "within", "offset"}) { Value y = made(ev); y.facts["stage"] = n; y.facts.erase("gate"); out[n] = y; }
                 return out;
               }});
  r.push_back({"detect_spans", "after", {"x", "baseline"}, {"baseline"}, {"spans"},
               Json{{"combine", "each"}, {"enter_sd", 1.5}, {"exit_sd", 1.0}, {"merge_s", 0.1}, {"min_s", 0.4}, {"max_s", nullptr}, {"durations", "seconds"}, {"merge_cycles", 1.0}, {"min_cycles", 3.0}, {"max_cycles", nullptr}},
               "Rheome +detect/spindle.m (its rule: per channel the median and robust SD 1.4826·MAD over the baseline, enter at median + hiSD·SD, extend while ≥ median + loSD·SD, "
               "merge gaps ≤ mergeGap, keep minDur ≤ length ≤ maxDur), as +flow/labelalpha.m calls it for alpha episodes (1.5 / 1.0 SD, 0.1 s, 0.4 s, no max); compute nxr::intervals",
               "a measurement's tiles labelled with events: per row (combine each) or on the mean over rows (combine mean), the median and robust SD over the good "
               "tiles of the run's span (statistics computed here, recorded with the result, never stored); spans of tiles ≥ median + enter_sd·SD extended while ≥ "
               "median + exit_sd·SD, joined across gaps ≤ merge_s, kept when min_s ≤ length ≤ max_s (seconds compared in whole tiles). `durations` seconds (the default: merge_s, min_s, max_s) | cycles (the project's nxr "
               "`spindle_envelope`, constant-Q: merge_cycles 1, min_cycles 3, max_cycles none, of the CENTRE of the band the tiles resolve: a cycle is 1/centre s; the seconds parameters are then not used); either way compared in whole tiles",
               [](const Values& ins, const Json& p, const PlanContext&) {
                 const Value& x = ins.at("x");
                 need(x.cls == "measurement" && x.facts.contains("over"), "detect_spans: x must be a measurement over a time partition (`measure` with `over`)");
                 need(x.facts.value("measure", std::string()) != "phase_current", "detect_spans: x is a vector measurement (the phase current); events are found on a scalar trace");
                 need(p["combine"] == "each" || p["combine"] == "mean", "detect_spans: combine " + p["combine"].dump() + ": each (per row) or mean (over rows)");
                 for (const char* k : {"enter_sd", "exit_sd", "merge_s", "min_s"}) need(p[k].is_number(), std::string("detect_spans: ") + k + " is a number");
                 need(p["enter_sd"].get<double>() >= p["exit_sd"].get<double>(), "detect_spans: enter_sd is at least exit_sd (enter high, extend low)");
                 need(p["durations"] == "seconds" || p["durations"] == "cycles", "detect_spans: durations " + p["durations"].dump() + ": seconds (merge_s, min_s, max_s) or cycles (merge_cycles, min_cycles, max_cycles of the band's centre)");
                 if (p["durations"] == "cycles") {
                   need(x.facts["over"].contains("resolves_hz"), "detect_spans: durations cycles need the band the tiles resolve (x's `over` names none)");
                   need(p["merge_cycles"].is_number() && p["min_cycles"].is_number() && p["merge_cycles"].get<double>() >= 0 && p["min_cycles"].get<double>() >= 0, "detect_spans: merge_cycles and min_cycles are numbers of at least 0");
                   need(p["max_cycles"].is_null() || (p["max_cycles"].is_number() && p["max_cycles"].get<double>() >= p["min_cycles"].get<double>()), "detect_spans: max_cycles is null (no limit) or at least min_cycles");
                 }
                 need(p["merge_s"].get<double>() >= 0 && p["min_s"].get<double>() >= 0, "detect_spans: merge_s and min_s are at least 0");
                 need(p["max_s"].is_null() || (p["max_s"].is_number() && p["max_s"].get<double>() >= p["min_s"].get<double>()), "detect_spans: max_s is null (no limit) or at least min_s");
                 need(!(x.facts.contains("reduced") && x.facts["reduced"].value("groups", std::string()) == "per_hemisphere" && p["combine"] == "mean"),
                      "detect_spans: x is a trace per hemisphere (per_hemisphere): combine each finds the events of each hemisphere; the mean over the hemispheres is not a per-hemisphere detection");
                 if (ins.count("baseline")) {   // the thresholds' ingredients from a stored statistic (summarize: median and mad, on the same rows)
                   const Value& b = ins.at("baseline");
                   need(b.cls == "statistic" && b.facts.contains("summaries"), "detect_spans: baseline must be a statistic made by `summarize`");
                   bool med = false, mad = false;
                   for (const auto& n : b.facts["summaries"]) { med = med || n == "median"; mad = mad || n == "mad"; }
                   need(med && mad, "detect_spans: the baseline must hold median and mad");
                   need(b.facts.value("combine", std::string()) == p["combine"].get<std::string>(), "detect_spans: the baseline is combined '" + b.facts.value("combine", std::string()) + "', the detection '" + p["combine"].get<std::string>() + "'");
                 }
                 Value y = made(x);
                 y.cls = "spans";
                 y.facts["detect"] = p;
                 return Values{{"spans", y}};
               }});
  r.push_back({"rollup_events", "after", {"x", "events"}, {}, {"y"}, Json{{"measure", nullptr}},
               "docs/specs/2026-11-13-burst-statistics.md (a burst measured from the tile measurements it covers, by their exact roll-up: the registry's reductions, as the lookup merges members); "
               "compute nxr::intervals::rollupEvents, eventDurations, eventIntervalNext",
               "a measurement of a tile level's tiles (a constant-Q level's, or a joint partition's cells) rolled up over the tiles of each event: the spans `detect_spans` made in this run, or a STORED spans Selection "
               "(an input `selection: spans`, e.g. the bursts of `global_bursts`): amplitude (the count-weighted mean of the tile means), envelope_max (the max of the tile maxima), power and phase_current (the "
               "count-weighted mean, per component for the vector), or the event's own geometry, duration and interval_next (seconds); [rows x events], an event's value on the row that found it (every row for the "
               "mean trace and for a global stored event), NaN elsewhere; an event holding a tile not whole and good is NaN for the sampled measures",
               [](const Values& ins, const Json& p, const PlanContext& c) {
                 const Value& x = ins.at("x");
                 const Value& ev = ins.at("events");
                 need(p["measure"].is_string(), "rollup_events: measure is one of amplitude, envelope_max, power, phase_current, duration, interval_next");
                 const std::string m = p["measure"].get<std::string>();
                 need(m == "amplitude" || m == "envelope_max" || m == "power" || m == "phase_current" || m == "duration" || m == "interval_next",
                      "rollup_events: measure " + p["measure"].dump() + ": amplitude, envelope_max, power, phase_current, duration or interval_next");
                 need(x.cls == "measurement" && x.facts.contains("over") && !x.facts["over"].value("events", false), "rollup_events: x must be a measurement over a tile level's tiles (`measure` with `over`)");
                 const bool stored = ev.cls == "selection";
                 if (stored) {
                   need(ev.row && nxr::io::text(*ev.row, "type") == "spans", "rollup_events: events must be the spans `detect_spans` made or a stored spans Selection (an input `selection: spans`)");
                   (void)stored_event_tiles(c.bound, ev, x, "rollup_events");   // refused here, naming the span, when another recording's or not on the tiles
                 } else {
                   need(ev.cls == "spans" && ev.facts.contains("detect") && ev.facts.contains("over"), "rollup_events: events must be the spans `detect_spans` made or a stored spans Selection (an input `selection: spans`)");
                   need(ev.facts["over"].value("selection_id", std::string("a")) == x.facts["over"].value("selection_id", std::string("b")),
                        "rollup_events: the events were found on the tiles of " + ev.facts["over"].value("selection", std::string("another level")) + ", x is over " + x.facts["over"].value("selection", std::string("?")));
                 }
                 if (m != "duration" && m != "interval_next") need(x.facts.value("measure", std::string()) == m, "rollup_events: x is the " + x.facts.value("measure", std::string("?")) + " of the tiles, not their " + m);
                 Value y = made(x);
                 y.cls = "measurement";
                 y.facts["measure"] = m;
                 if (stored) {
                   y.facts["over"] = Json{{"selection_id", nxr::io::text(*ev.row, "id")}, {"selection", nxr::io::text(*ev.row, "name")}, {"stored_events", true}, {"tile_level", x.facts["over"]}};
                   if (x.facts["over"].value("joint", false)) y.facts["over"]["space_tiles"] = x.facts["over"]["space_tiles"];
                 } else {
                   y.facts["over"] = Json{{"events", true}, {"selection", "the " + std::string("events of this run")}, {"tile_level", x.facts["over"]}};
                 }
                 if (m == "duration" || m == "interval_next") y.unit = "s";
                 return Values{{"y", y}};
               }});
  bind_execution(r);
  bind_traits(r);
  add_extent_steps(r);
  add_cores_steps(r);
  add_tracks_steps(r);
  add_crossings_steps(r);
  add_spatial_events_steps(r);
  add_surrogate_steps(r);
  return r;
}

// ─────────────────────────────────────────── execution: each step's state and its kind's methods ───────────────────────────────────────────

// the block cut to its own columns and `keep` of its halo on each side
Block crop(const Block& b, std::int64_t keep) {
  const std::int64_t kl = std::min(keep, b.left), kr = std::min(keep, b.right), w = b.own() + kl + kr, c0 = b.left - kl;
  Block o{b.rows, w, kl, kr, std::vector<float>(static_cast<std::size_t>(b.rows * w)), {}};
  if (b.complex()) o.im.resize(o.re.size());
  for (std::int64_t r = 0; r < b.rows; ++r) {
    std::copy_n(b.re.begin() + r * b.width + c0, w, o.re.begin() + r * w);
    if (b.complex()) std::copy_n(b.im.begin() + r * b.width + c0, w, o.im.begin() + r * w);
  }
  return o;
}
// rows convolved with taps (compute nxr::taps::convolveRows) over the own columns and `keep` of the halo
std::vector<float> convolve(const Block& b, const std::vector<float>& taps, std::int64_t keep, bool odd, std::int64_t& kl, std::int64_t& kr) {
  kl = std::min(keep, b.left);
  kr = std::min(keep, b.right);
  return nxr::taps::convolveRows(b.re.data(), static_cast<std::size_t>(b.rows), static_cast<std::size_t>(b.width), taps, static_cast<std::size_t>(b.left - kl),
                                 static_cast<std::size_t>(b.own() + kl + kr), odd);
}
// a band's gain: shape morlet or meyer (the Studio's
// meyerBand, half_width octaves); band null: all-pass, the plain Hilbert's)
std::function<double(double)> band_gain(const Json& p) {
  if (p["band"].is_null()) return [](double) { return 1.0; };
  const double lo = p["band"][0].get<double>(), hi = p["band"][1].get<double>(), hw = p["half_width"].get<double>();
  if (p["shape"] == "morlet") return [lo, hi](double f) { return nxr::filterbank::morletBand(f, lo, hi); };
  return [lo, hi, hw](double f) { return nxr::filterbank::meyerBand(f, lo, hi, hw); };
}
struct AnalyticTaps { std::vector<float> even, odd; };
struct KernelState { std::shared_ptr<const Eigen::MatrixXd> K; Json resolved; };
// the Studio's field type and the number of rows a vertex holds
int per_vertex(const Value& v) { return v.value_type == "scalar" ? 1 : v.value_type == "quaternion" ? 4 : v.value_type == "monogenic" ? 4 * v.facts["monogenic"]["components"].get<int>() : 3; }
// the surface a field lies on (its first factor), as the Studio's face-gradient operator (positions float32, as the GPU receives them)
std::shared_ptr<nxr::forms::FaceGradientOperator> surface_operator(ExecContext& c, const std::string& manifold_id) {
  const auto m = c.bound.manifold(manifold_id);
  const auto S = c.bound.surface(m ? std::optional<std::string>(nxr::io::text(*m, "name")) : std::nullopt);
  std::vector<float> P(S.vertices.begin(), S.vertices.end());
  return std::make_shared<nxr::forms::FaceGradientOperator>(nxr::forms::assembleFaceGradient(P.data(), S.faces.data(), static_cast<int>(S.n_vertices()), static_cast<int>(S.n_faces())));
}
struct PhaseState { nxr::phase::Gradient G; Eigen::MatrixXd N; int components = 1; };
std::shared_ptr<PhaseState> phase_state(const Value& z, ExecContext& c) {
  const auto op = surface_operator(c, z.manifolds[0]);
  const int nV = op->nV;
  Eigen::MatrixXd P(nV, 3), N(nV, 3);
  Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor> F(op->nF, 3);
  for (int v = 0; v < nV; ++v) for (int d = 0; d < 3; ++d) { P(v, d) = op->position[3 * v + d]; N(v, d) = op->vertexNormals[3 * v + d]; }   // area-weighted unit normals (nxr core_phase)
  for (int f = 0; f < op->nF; ++f) for (int d = 0; d < 3; ++d) F(f, d) = op->faces[3 * f + d];
  return std::make_shared<PhaseState>(PhaseState{nxr::phase::vertexGradient(P, F), N, z.value_type == "vector3" ? 3 : 1});
}
// a complex block's own (+ keep) columns as Z [rows x m]
Eigen::MatrixXcd complex_of(const Block& b) {
  Eigen::MatrixXcd Z(b.rows, b.width);
  for (std::int64_t r = 0; r < b.rows; ++r)
    for (std::int64_t t = 0; t < b.width; ++t) Z(r, t) = {b.re[static_cast<std::size_t>(r * b.width + t)], b.im[static_cast<std::size_t>(r * b.width + t)]};
  return Z;
}
Block block_of(const Eigen::MatrixXd& Y, std::int64_t left, std::int64_t right) {
  Block o{Y.rows(), Y.cols(), left, right, std::vector<float>(static_cast<std::size_t>(Y.size())), {}};
  for (Eigen::Index r = 0; r < Y.rows(); ++r) for (Eigen::Index t = 0; t < Y.cols(); ++t) o.re[static_cast<std::size_t>(r * Y.cols() + t)] = static_cast<float>(Y(r, t));
  return o;
}
// The MEDIAN of a member (numpy's, exact;) cannot be folded from sums: each member's good samples are HELD per row until the member closes
// (the whole span, or a tile once a later tile begins), then reduced and the samples freed. MEMORY: the longest open member's samples x rows (4 bytes each),
// at most kMedianCap samples a row; a longer member is REFUSED, never truncated.
constexpr std::int64_t kMedianCap = std::int64_t(1) << 20;
struct MeasureAcc { std::vector<double> sum, sumSq, count; std::vector<float> min, max; std::vector<std::vector<float>> held; };
// a joint partition's cells: every vertex's spatial tile, its barycentric area, and each tile's area: the accumulators are 2 doubles a cell
struct JointState { std::int64_t nV = 0, S = 0; std::vector<std::int32_t> tile; std::vector<double> area, tile_area; };
struct MeasureState { std::string measure; std::optional<TimeLevel> level; double rate = 0, origin = 0; std::shared_ptr<JointState> joint; int ncomp = 1; };   // ncomp: the rows a vertex of x holds (3 for the phase current)
struct TileSums { std::vector<double> sum, mx; std::int64_t count = 0; std::vector<std::vector<float>> held; std::vector<double> med; };
// the median of each row's held samples (NaN for none), the samples freed
std::vector<double> close_medians(std::vector<std::vector<float>>& held) {
  std::vector<double> out(held.size(), std::numeric_limits<double>::quiet_NaN());
  for (std::size_t r = 0; r < held.size(); ++r) {
    if (held[r].empty()) continue;
    std::vector<double> v(held[r].begin(), held[r].end());
    out[r] = nxr::stats::median(std::move(v));
  }
  held.clear();
  return out;
}
struct TileAcc { std::map<std::int64_t, TileSums> tiles; std::int64_t rows = 0; };

// every node declares its time behaviour and numeric precision (NodeTraits); the nodes of the other files declare theirs where they are registered
void bind_traits(std::vector<StepSpec>& r) {
  auto declare = [&](const char* n, NodeTraits t) { for (auto& s : r) if (s.name == n) { s.traits = std::move(t); return; } refuse(std::string("no step ") + n); };
  declare("band_filter", node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::follows_input, "a zero-phase FIR band filter; float32 taps, each output sample a float64 sum over the window, stored as float32", "the taps' reach: taps.size() - 1 samples each side (set by the band's width; a recording's end is reflected)"));
  declare("analytic", node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::follows_input, "the band's even and odd FIR passes (the analytic signal); float64 sums, float32 result", "the longer of the even and odd taps' reach: taps.size() - 1 samples each side"));
  declare("envelope", node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::follows_input, "the analytic signal's modulus: the same FIR passes, then |z| in double, stored as float32", "the longer of the even and odd taps' reach: taps.size() - 1 samples each side"));
  declare("frames", node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::follows_input, "samples the band's signal at the frame tiling by a Kaiser-windowed sinc; float64 weights and sum, float32 result", "half_width samples each side of a frame's position (the sinc's reach; default 32)"));
  declare("apply_kernel", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "a linear map of the rows (kernel, spatial band) applied to each column independently; the product is formed in double and stored as float32", {}));
  declare("magnitude", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "the norm over an element's components, per sample; sums in double, float32 result", {}));
  declare("tile_max", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "per sample, the area-weighted mean per spatial tile (double) and the maximum over a group's tiles", {}));
  declare("collect", node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64, "appends each block's own columns in time order: the order of the blocks fixes the result", {}));
  declare("reduce_cells", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "the maximum over a group of spatial tiles, separately for each time tile; time tiles do not interact", {}));
  declare("risefall", node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::float64, "fits each event on the series over [start - margin, stop + margin]; events are independent of each other", "the event's span widened by `margin` (seconds, or cycles of the band) on each side"));
  declare("form", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "a differential form of the field (curl, divergence, normal part) per sample: the surface operator applied to each column; double sums, float32 result", {}));
  declare("power", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "|z|^2 summed over an element's components, per sample; double sums, float32 result", {}));
  declare("phase_current", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "Im(conj(z) grad z) per sample on the surface; double, float32 result", {}));
  declare("gradient_power", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "|grad z|^2 per sample on the surface; double, float32 result", {}));
  declare("measure", node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64, "a reduction over the run's tiles; sums, sums of squares, counts, minima and maxima do not depend on block order, but the median holds each tile's samples in time order, so the node is declared stateful for every measure", {}));
  declare("summarize", node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64, "a statistic over all the members of a partition, computed once on the finished series; it is not defined per timestep and must see every member", {}, true));
  declare("fit_gate", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "keeps or drops each event on its own fit; events do not interact (the unit is the event, not a timestep)", {}, true));
  declare("burst_stages", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "each event's stage spans from its own fit; events do not interact (the unit is the event, not a timestep)", {}, true));
  declare("detect_spans", node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64, "a hysteresis scan along each series: the in-span state is carried from tile to tile, then gaps are merged and short spans dropped", {}));
  declare("rollup_events", node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::float64, "rolls the tile measurements up over the tiles each event covers; the unit is the event", "the event's own span of tiles (variable length)", true));
}

void bind_execution(std::vector<StepSpec>& r) {
  auto at = [&](const char* n) -> StepSpec& { for (auto& s : r) if (s.name == n) return s; refuse(std::string("no step ") + n); };
  auto taps_halo = [](const std::any& st) { return static_cast<std::int64_t>(std::any_cast<const std::vector<float>&>(st).size()) - 1; };

  StepSpec& bf = at("band_filter");
  bf.prepare = [](const Values& ins, const Json& p, ExecContext&) { return std::any(nxr::taps::gainTaps(band_gain(p), ins.at("x").rate)); };
  bf.halo = taps_halo;
  bf.chunk = [](const std::any& st, const Blocks& ins, std::int64_t keep) {
    const Block& b = ins.at("x");
    std::int64_t kl, kr;
    auto y = convolve(b, std::any_cast<const std::vector<float>&>(st), keep, false, kl, kr);
    return Blocks{{"y", Block{b.rows, b.own() + kl + kr, kl, kr, std::move(y), {}}}};
  };

  // the analytic signal and the envelope of a band: the band's even and odd taps on the same raw rows (the Studio's THE ANALYTIC SIGNAL)
  auto analytic_prepare = [](const Values& ins, const Json& p, ExecContext&) {
    const auto g = band_gain(p);
    return std::any(AnalyticTaps{nxr::taps::gainTaps(g, ins.at("x").rate, false), nxr::taps::gainTaps(g, ins.at("x").rate, true)});
  };
  auto analytic_halo = [](const std::any& st) {
    const auto& t = std::any_cast<const AnalyticTaps&>(st);
    return static_cast<std::int64_t>(std::max(t.even.size(), t.odd.size())) - 1;
  };
  for (const char* n : {"analytic", "envelope"}) {
    StepSpec& s = at(n);
    const bool env = std::string(n) == "envelope";
    s.prepare = analytic_prepare;
    s.halo = analytic_halo;
    s.chunk = [env](const std::any& st, const Blocks& ins, std::int64_t keep) {
      const auto& t = std::any_cast<const AnalyticTaps&>(st);
      const Block& b = ins.at("x");
      std::int64_t kl, kr;
      Block y{b.rows, 0, 0, 0, convolve(b, t.even, keep, false, kl, kr), convolve(b, t.odd, keep, true, kl, kr)};
      y.width = b.own() + kl + kr; y.left = kl; y.right = kr;
      if (env) {
        for (std::size_t i = 0; i < y.re.size(); ++i) y.re[i] = static_cast<float>(std::hypot(static_cast<double>(y.re[i]), static_cast<double>(y.im[i])));
        y.im.clear();
      }
      return Blocks{{"y", std::move(y)}};
    };
  }

  // the frame tiling: the chunk's analytic field sampled at the frames inside its own samples
  struct FramesState { nxr::tower::FrameTiling t; double origin = 0; int H = 32; std::vector<std::pair<double, double>> within; bool has_within = false; };
  StepSpec& fr = at("frames");
  fr.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
    const Value& x = ins.at("x");
    const Json tp = Json::parse(nxr::io::opt_text(*ins.at("tiling").row, "params_json").value_or("{}"), nullptr, false);
    const auto line = c.bound.manifold(*x.time);
    FramesState s;
    try { s.t = nxr::tower::makeFrameTiling(x.rate, nxr::io::integer(*line, "n_vertices"), tp["atlas"]["frame_level"].get<int>(), tp["atlas"]["start_s"].is_null() ? std::optional<double>() : std::optional<double>(tp["atlas"]["start_s"].get<double>())); } catch (const std::invalid_argument& e) { refuse(std::string("frames: ") + e.what()); }
    if (s.t.count != tp["tiling"]["count"].get<std::int64_t>() || s.t.code0 != tp["atlas"]["frame_code0"].get<std::int64_t>()) refuse("frames: the stored tiling's " + tp["tiling"]["count"].dump() + " frames from code " + tp["atlas"]["frame_code0"].dump() + " are not the " + std::to_string(s.t.count) + " from code " + std::to_string(s.t.code0) + " of its recording (a stored tiling of another run): `frames compute --replace`");
    s.origin = x.origin;
    s.H = p["half_width"].get<int>();
    if (ins.count("within")) {
      s.has_within = true;
      for (const auto& r : c.bound.spans(nxr::io::text(*ins.at("within").row, "id"))) s.within.push_back({r.first, r.second});
    }
    return std::any(std::make_shared<FramesState>(std::move(s)));
  };
  fr.halo = [](const std::any& st) { return static_cast<std::int64_t>(std::any_cast<const std::shared_ptr<FramesState>&>(st)->H); };
  fr.chunk_at = [](const std::any& st_, const Blocks& ins, std::int64_t keep, const Place& at) {
    const FramesState& st = *std::any_cast<const std::shared_ptr<FramesState>&>(st_);
    if (keep != 0) refuse("frames: a step after it needs a halo; the steps after the frame tiling are pointwise in time");
    const Block& b = ins.at("x");
    const auto [j0, j1] = nxr::tower::frameSpan(st.t, at.first, at.n);
    std::vector<double> pos;
    for (std::int64_t j = j0; j < j1; ++j) pos.push_back(st.t.position(j) - static_cast<double>(at.first - b.left));          // from the block's column 0
    Block y{b.rows, j1 - j0, 0, 0, nxr::frametiling::sincInterpolate(b.re.data(), b.rows, b.width, pos, st.H), {}};
    if (b.complex()) y.im = nxr::frametiling::sincInterpolate(b.im.data(), b.rows, b.width, pos, st.H);
    return Blocks{{"y", std::move(y)}};
  };
  fr.frame_place = [](const std::any& st_, const Place& at) {
    const FramesState& st = *std::any_cast<const std::shared_ptr<FramesState>&>(st_);
    const auto [j0, j1] = nxr::tower::frameSpan(st.t, at.first, at.n);
    FramePlace f{j0, j1 - j0, {}};
    const bool bad = at.good && !at.good->empty();
    const bool early = j1 > j0 && st.t.position(j0) < 0;      // a partial first frame (a day anchor) starts before the recording: nothing is sampled there, it is not good
    if (bad || st.has_within || early) {
      f.good.assign(static_cast<std::size_t>(j1 - j0), 1);
      for (std::int64_t j = j0; j < j1; ++j) {
        const double ps = st.t.position(j);
        if (ps < 0) f.good[static_cast<std::size_t>(j - j0)] = 0;
        if (bad) {
          const std::int64_t u = std::min<std::int64_t>(at.n - 1, std::max<std::int64_t>(0, static_cast<std::int64_t>(std::floor(ps)) - at.first));      // the sample at or before the frame
          if (!(*at.good)[static_cast<std::size_t>(u)]) f.good[static_cast<std::size_t>(j - j0)] = 0;
        }
        if (st.has_within) {
          const double t = st.origin + ps / st.t.sfreq;
          bool in = false;
          for (const auto& [a, e] : st.within) if (t >= a && t < e) { in = true; break; }
          if (!in) f.good[static_cast<std::size_t>(j - j0)] = 0;
        }
      }
    }
    return f;
  };

  // the kernel padded to every member of the field's first factor (nxr's kernel_from_rows: its from_selection's columns)
  StepSpec& ak = at("apply_kernel");
  ak.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
    const nxr::io::Row& k = *ins.at("kernel").row;
    // the bound map is a constant of (kernel, the field's first factor, level, transform request): every step that asks for the same one (the same level read by several time bands) shares the first one's
    const std::string shared_key = "apply_kernel|" + nxr::io::text(k, "id") + "|" + ins.at("x").manifolds[0] + "|" + (ins.count("tiles") ? p.dump() : std::string("none")) + "|" + p["form"].get<std::string>();
    if (const auto it = c.shared.find(shared_key); it != c.shared.end()) return it->second;
    const std::string kname = nxr::io::text(k, "name");
    const auto m = c.bound.manifold(ins.at("x").manifolds[0]);
    const std::int64_t C = m ? nxr::io::integer(*m, "n_vertices") : 0;
    std::vector<std::int32_t> chans;
    if (!nxr::io::is_null(k, "from_selection_id")) {
      const auto sel = c.bound.selection(nxr::io::text(k, "from_selection_id"));
      if (!sel || nxr::io::is_null(*sel, "array_path")) refuse("apply_kernel: " + kname + ": its channels (from_selection) are not stored");
      chans = c.bound.array_int32(nxr::io::text(*sel, "array_path"));
    } else {
      for (std::int64_t i = 0; i < C; ++i) chans.push_back(static_cast<std::int32_t>(i));
    }
    std::vector<std::int64_t> shape;
    const auto kf = c.bound.array_float(nxr::io::text(k, "path"), &shape);
    if (shape.size() != 2 || shape[1] != static_cast<std::int64_t>(chans.size())) refuse("apply_kernel: " + kname + ": its array is not [rows x " + std::to_string(chans.size()) + " channels]");
    auto K = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Zero(shape[0], C));
    for (std::int64_t i = 0; i < shape[0]; ++i)
      for (std::size_t j = 0; j < chans.size(); ++j) {
        if (chans[j] < 0 || chans[j] >= C) refuse("apply_kernel: " + kname + ": a channel index is outside the field's channels");
        (*K)(i, chans[j]) = kf[static_cast<std::size_t>(i * shape[1]) + j];
      }
    Json resolved = Json::object();
    if (p["form"] != "none") {
      // the form folded into the operator, BEFORE the band: each channel's free current (x y z a vertex) → its tangent part → curl_tan / div_tan / n·x: [3 nV × C] → [nV × C] (nxr forms::formOfFreeColumns)
      const auto op = surface_operator(c, ins.at("kernel").manifolds[1]);
      if (shape[0] != 3 * static_cast<std::int64_t>(op->nV)) refuse("apply_kernel: form: " + kname + ": its " + std::to_string(shape[0]) + " rows are not 3 x the " + std::to_string(op->nV) + " vertices of its target surface");
      const std::string f = p["form"].get<std::string>();
      K = std::make_shared<Eigen::MatrixXd>(nxr::forms::formOfFreeColumns(f == "curl_tan" ? nxr::forms::Form::curl_tan : f == "div_tan" ? nxr::forms::Form::div_tan : nxr::forms::Form::nx, *K, *op));
      shape[0] = op->nV;
      resolved = Json{{"form", f}, {"form_order", "kernel, tangent part, form, spatial band: one linear operator"}};
    }
    if (ins.count("tiles")) {
      // M = (F_l ⊗ I) K: the level's band applied ONCE to every component of the kernel's rows, on the channels the kernel uses (the rest of K is zero); with `levels`, one such operator a level, stacked
      // (row = (vertex · levels + level) · components + component). MEMORY: the bound K [vertices · levels · components rows × channels] in double, held for the run; one level's [vertices × used channels] block at a time while it is built
      const auto man = c.bound.manifold(ins.at("kernel").manifolds[1]);
      const nxr::pipelines::SurfaceInput S = c.bound.surface(nxr::io::text(*man, "name"), true);
      const std::int64_t nV = S.n_vertices();
      if (shape[0] % nV != 0) refuse("apply_kernel: " + kname + ": its " + std::to_string(shape[0]) + " rows are not a whole number of components for the " + std::to_string(nV) + " vertices of " + nxr::io::text(*man, "name"));
      const std::int64_t comps = shape[0] / nV;
      std::vector<int> lvs;
      if (!p["levels"].is_null()) for (const auto& e : p["levels"]) lvs.push_back(e.get<int>());
      else lvs.push_back(p["level"].get<int>());
      const std::int64_t NL = static_cast<std::int64_t>(lvs.size());
      const bool form = p["form"] != "none";
      const bool glued = form || p["glued"].get<bool>();         //: a form is evaluated on every glued vertex (the bridge too); a free current may be, by request
      const bool mono = p["envelope"] == "spatiotemporal";
      const std::shared_ptr<Eigen::MatrixXd> Kraw_p = K;
      const Eigen::MatrixXd& Kraw = *Kraw_p;
      if (form && comps != 1) refuse("apply_kernel: form: the form of " + kname + " has " + std::to_string(comps) + " numbers a vertex; one");
      if (mono && comps != 1 && comps != 3) refuse("apply_kernel: envelope spatiotemporal: " + kname + " has " + std::to_string(comps) + " components a vertex; one or three");
      std::shared_ptr<Eigen::MatrixXd> Kout;
      Json bands = Json::array();
      for (std::int64_t li = 0; li < NL; ++li) {
        const nxr::pipelines::SurfaceLevel L = c.bound.surface_level(S, lvs[static_cast<std::size_t>(li)]);
        nxr::pipelines::SpatialBandOptions bo;
        bo.transform = p["transform"].get<std::string>(); bo.modes = p["modes"].get<int>(); bo.order = p["order"].get<int>(); bo.tolerance = p["tolerance"].get<double>();
        bo.spatial_shape = p["spatial_shape"].get<std::string>();
        bo.glued = glued;
        bo.riesz = mono;
        const nxr::pipelines::SpatialBand band = c.bound.make_spatial_band(S, L, bo);
        bands.push_back(band.info);
        std::int64_t rows_v = nV, out_comps = comps;                // the rows a vertex's level holds: its components
        if (glued) {
          const auto gm = c.bound.manifold_named(band.glued_surface);
          if (!gm) refuse("apply_kernel: glued: the glued surface " + band.glued_surface + " is not stored (`nxr-neuro eigen solve` stores it)");
          if (nxr::io::integer(*gm, "n_vertices") != band.glued_vertices) refuse("apply_kernel: glued: the glued surface " + band.glued_surface + " holds " + std::to_string(nxr::io::integer(*gm, "n_vertices")) + " vertices, the band " + std::to_string(band.glued_vertices));
          rows_v = band.glued_vertices;
          if (li == 0) resolved["glued_surface"] = Json{{"surface", band.glued_surface}, {"id", nxr::io::text(*gm, "id")}, {"vertices", band.glued_vertices}};
        }
        if (!Kout) Kout = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Zero(mono ? nV * 4 * comps : rows_v * NL * out_comps, C));
        Eigen::MatrixXd& Ko = *Kout;
        for (std::int64_t comp = 0; comp < comps; ++comp) {
          Eigen::MatrixXd Kc(nV, chans.size());
          for (std::int64_t v = 0; v < nV; ++v) for (std::size_t j = 0; j < chans.size(); ++j) Kc(v, static_cast<Eigen::Index>(j)) = Kraw(v * comps + comp, chans[j]);
          if (glued) {
            // the columns [nV × C], zero on the bridge, through the band on the GLUED surface: [Vg × C], the bridge filled by the band. ONE linear operator
            const Eigen::MatrixXd Mc = band.apply_glued(Kc);
            for (std::int64_t v = 0; v < rows_v; ++v) for (std::size_t j = 0; j < chans.size(); ++j) Ko((v * NL + li) * out_comps + comp, chans[j]) = Mc(v, static_cast<Eigen::Index>(j));
          } else if (!mono) {
            const Eigen::MatrixXd Mc = band.apply(Kc);
            for (std::int64_t v = 0; v < nV; ++v) for (std::size_t j = 0; j < chans.size(); ++j) Ko((v * NL + li) * comps + comp, chans[j]) = Mc(v, static_cast<Eigen::Index>(j));
          } else {
            // spatiotemporal: M_l and R_l STACKED per vertex, 4c rows [M (c) ; R (3c: component a, direction d)]: MEMORY 4x the rows of K, float64, held for the run
            const Eigen::MatrixXd Mc = band.apply(Kc), Rc = band.apply_riesz(Kc);          // [nV × chans], [3 nV × chans]
            for (std::int64_t v = 0; v < nV; ++v)
              for (std::size_t j = 0; j < chans.size(); ++j) {
                Ko(v * 4 * comps + comp, chans[j]) = Mc(v, static_cast<Eigen::Index>(j));
                for (int d = 0; d < 3; ++d) Ko(v * 4 * comps + comps + comp * 3 + d, chans[j]) = Rc(3 * v + d, static_cast<Eigen::Index>(j));
              }
          }
        }
      }
      K = Kout;
      resolved["spatial_band"] = NL == 1 ? bands[0] : bands;
      if (NL > 1) resolved["levels"] = p["levels"];
      resolved["envelope"] = p["envelope"];
    }
    std::any bound = KernelState{std::shared_ptr<const Eigen::MatrixXd>(K), resolved};
    c.shared[shared_key] = bound;
    return bound;
  };
  ak.record = [](const std::any& st) { return std::any_cast<const KernelState&>(st).resolved; };
  // the spatial band's choice (eigenmodes stored or computable, else the series) enters the run's cache key, read before anything is computed or stored
  ak.resolve = [](const Values& ins, const Json& p, const Bound& bound) -> Json {
    if (!ins.count("tiles")) return nullptr;
    const auto man = bound.manifold(ins.at("kernel").manifolds[1]);
    const nxr::pipelines::SurfaceInput S = bound.surface(nxr::io::text(*man, "name"), true);
    nxr::pipelines::SpatialBandOptions bo;
    bo.transform = p["transform"].get<std::string>(); bo.modes = p["modes"].get<int>(); bo.order = p["order"].get<int>(); bo.tolerance = p["tolerance"].get<double>();
    bo.spatial_shape = p["spatial_shape"].get<std::string>();
    bo.glued = p["form"] != "none" || p["glued"].get<bool>();
    if (p["levels"].is_null()) return bound.plan_spatial_band(S, bound.surface_level(S, p["level"].get<int>()), bo).key;
    Json keys = Json::array();                                                                  // one choice a level: each level's transform is in the run's key
    for (const auto& e : p["levels"]) keys.push_back(bound.plan_spatial_band(S, bound.surface_level(S, e.get<int>()), bo).key);
    return keys;
  };
  ak.chunk = [](const std::any& st, const Blocks& ins, std::int64_t keep) {
    const Eigen::MatrixXd& K = *std::any_cast<const KernelState&>(st).K;
    const Block b = crop(ins.at("x"), keep);
    using RM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    auto mul = [&](const std::vector<float>& x) {
      const Eigen::MatrixXd X = Eigen::Map<const RM>(x.data(), b.rows, b.width).cast<double>();
      const RM Y = (K * X).cast<float>();
      return std::vector<float>(Y.data(), Y.data() + Y.size());
    };
    Block y{K.rows(), b.width, b.left, b.right, mul(b.re), b.complex() ? mul(b.im) : std::vector<float>()};
    return Blocks{{"y", std::move(y)}};
  };

  // |x| per vertex: the norm over the vertex's components (consecutive rows) and the complex part, per sample
  StepSpec& mg = at("magnitude");
  struct MagState { int per = 1, zc = 0; };               // rows a vertex; zc > 0: a monogenic field, [z (zc) ; R z (3 zc)] a vertex
  mg.prepare = [](const Values& ins, const Json&, ExecContext&) {
    const Value& x = ins.at("x");
    return std::any(MagState{per_vertex(x), x.value_type == "monogenic" ? x.facts["monogenic"]["components"].get<int>() : 0});
  };
  mg.chunk = [](const std::any& st, const Blocks& ins, std::int64_t keep) {
    const int per = std::any_cast<const MagState&>(st).per, zc = std::any_cast<const MagState&>(st).zc;
    const Block b = crop(ins.at("x"), keep);
    if (b.rows % per != 0) refuse("magnitude: " + std::to_string(b.rows) + " rows are not a whole number of " + std::to_string(per) + "-component elements");
    const std::int64_t V = b.rows / per;
    Block y{V, b.width, b.left, b.right, std::vector<float>(static_cast<std::size_t>(V * b.width)), {}};
    if (zc > 0) {
      // the spatial amplitude A = sqrt(Σ|z|² + Σ|R z|²) (nxr::monogenic::amplitude), per vertex and sample: the stacked rows split into z and its Riesz part
      const int rc = 3 * zc;
      Eigen::MatrixXcd Z(V * zc, b.width), R(V * rc, b.width);
      for (std::int64_t v = 0; v < V; ++v)
        for (int r = 0; r < per; ++r)
          for (std::int64_t t = 0; t < b.width; ++t) {
            const std::size_t i = static_cast<std::size_t>((v * per + r) * b.width + t);
            const std::complex<double> w(b.re[i], b.complex() ? b.im[i] : 0.0f);
            if (r < zc) Z(v * zc + r, t) = w; else R(v * rc + (r - zc), t) = w;
          }
      const Eigen::MatrixXd A = nxr::monogenic::amplitude(Z, zc, R, rc);
      for (std::int64_t v = 0; v < V; ++v) for (std::int64_t t = 0; t < b.width; ++t) y.re[static_cast<std::size_t>(v * b.width + t)] = static_cast<float>(A(v, t));
      return Blocks{{"y", std::move(y)}};
    }
    for (std::int64_t v = 0; v < V; ++v)
      for (std::int64_t t = 0; t < b.width; ++t) {
        double a = 0;
        for (int c = 0; c < per; ++c) {
          const double re = b.re[static_cast<std::size_t>((v * per + c) * b.width + t)], im = b.complex() ? b.im[static_cast<std::size_t>((v * per + c) * b.width + t)] : 0.0;
          a += re * re + im * im;
        }
        y.re[static_cast<std::size_t>(v * b.width + t)] = static_cast<float>(std::sqrt(a));
      }
    return Blocks{{"y", std::move(y)}};
  };

  // per sample: the area-weighted mean of |z| per spatial tile (nxr::partition::meanOverMembers, time unreduced), then the MAX over the tiles of each group
  // MEMORY: one chunk's [tiles x (chunk + halo)] means in double, inside this step, never stored
  struct TileMaxState { std::vector<std::int32_t> tile; std::vector<double> area; std::int64_t S = 0, G = 1; std::vector<int> group; };
  StepSpec& tm = at("tile_max");
  tm.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
    const Value& x = ins.at("x");
    const auto man = c.bound.manifold(x.manifolds[0]);
    const nxr::pipelines::SurfaceInput S = c.bound.surface(nxr::io::text(*man, "name"), true);
    const int l = p["level"].get<int>();
    const nxr::pipelines::SurfaceLevel L = c.bound.surface_level(S, l);
    nxr::ladder::Mesh mesh; mesh.n = static_cast<int>(S.n_vertices()); mesh.positions = S.vertices; mesh.faces = S.faces;
    TileMaxState st;
    st.tile = L.tile; st.area = nxr::ladder::vertexAreas(mesh); st.S = L.nodes;
    const bool pieces = p["groups"] == "per_hemisphere";
    if (pieces && (L.nodes >> l) != 2) refuse("tile_max: per_hemisphere needs a surface of two pieces (the ladder's level has " + std::to_string(L.nodes >> l) + ")");
    st.G = pieces ? 2 : 1;
    for (std::int64_t t = 0; t < L.nodes; ++t) st.group.push_back(pieces ? static_cast<int>(t >> l) : 0);
    return std::any(std::make_shared<TileMaxState>(std::move(st)));
  };
  tm.chunk = [](const std::any& st_, const Blocks& ins, std::int64_t keep) {
    const TileMaxState& st = *std::any_cast<const std::shared_ptr<TileMaxState>&>(st_);
    const Block b = crop(ins.at("x"), keep);
    if (b.rows != static_cast<std::int64_t>(st.tile.size())) refuse("tile_max: x has " + std::to_string(b.rows) + " rows, the surface " + std::to_string(st.tile.size()) + " vertices");
    const Eigen::MatrixXd M = nxr::partition::meanOverMembers(b.re.data(), static_cast<std::size_t>(b.rows), static_cast<std::size_t>(b.width), static_cast<std::size_t>(b.width), st.tile.data(), static_cast<int>(st.S), st.area.data());
    Block y{st.G, b.width, b.left, b.right, std::vector<float>(static_cast<std::size_t>(st.G * b.width), -std::numeric_limits<float>::infinity()), {}};
    for (std::int64_t t = 0; t < st.S; ++t)
      for (std::int64_t u = 0; u < b.width; ++u) {
        const double v = M(t, u);
        if (std::isnan(v)) continue;                      // a tile of no vertex
        float& o = y.re[static_cast<std::size_t>(st.group[static_cast<std::size_t>(t)] * b.width + u)];
        o = std::max(o, static_cast<float>(v));
      }
    return Blocks{{"y", std::move(y)}};
  };
  // the run's samples in order: the own columns of every chunk appended
  struct CollectAcc { std::vector<std::vector<double>> rows; std::int64_t first = -1; };
  StepSpec& cl = at("collect");
  cl.prepare = [](const Values& ins, const Json&, ExecContext&) { return std::any(std::make_pair(ins.at("x").rate, ins.at("x").origin)); };
  cl.start = [](const std::any&) { return std::any(CollectAcc{}); };
  cl.fold = [](const std::any&, std::any& acc_, const Blocks& ins, const Place& place) {
    CollectAcc& acc = std::any_cast<CollectAcc&>(acc_);
    const Block& b = ins.at("x");
    if (acc.first < 0) { acc.first = place.first; acc.rows.assign(static_cast<std::size_t>(b.rows), {}); }
    for (std::int64_t r = 0; r < b.rows; ++r) acc.rows[static_cast<std::size_t>(r)].insert(acc.rows[static_cast<std::size_t>(r)].end(), b.re.begin() + r * b.width + b.left, b.re.begin() + r * b.width + b.left + b.own());
  };
  cl.finish = [](const std::any& st, std::any& acc_) {
    CollectAcc& acc = std::any_cast<CollectAcc&>(acc_);
    const auto [rate, origin] = std::any_cast<const std::pair<double, double>&>(st);
    if (acc.first < 0) refuse("collect: no sample in the span");
    const std::int64_t R = static_cast<std::int64_t>(acc.rows.size()), n = static_cast<std::int64_t>(acc.rows[0].size());
    Array y{R, n, std::vector<double>(static_cast<std::size_t>(R * n))};
    for (std::int64_t r = 0; r < R; ++r) std::copy(acc.rows[static_cast<std::size_t>(r)].begin(), acc.rows[static_cast<std::size_t>(r)].end(), y.data.begin() + r * n);
    y.meta = Json{{"rate", rate}, {"origin", origin}, {"first_sample", acc.first}, {"samples", n}};
    return Arrays{{"y", std::move(y)}};
  };

  // a form: the block's columns as the Studio's frames (sample-major), compute nxr::forms::formComplex, back to rows
  StepSpec& fm = at("form");
  struct FormState { std::shared_ptr<nxr::forms::FaceGradientOperator> op; nxr::forms::FieldType t; nxr::forms::Form id; int in_rows, out_rows; };
  fm.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
    const Value& x = ins.at("x");
    FormState s;
    s.t = field_type(x);
    s.id = *nxr::forms::formOf(p["form"].get<std::string>());
    s.op = surface_operator(c, x.manifolds[0]);
    s.in_rows = static_cast<int>(nxr::forms::widthOf(s.t, static_cast<std::size_t>(s.op->nV)));
    const auto out = nxr::forms::formInfo(s.t, s.id)->out;
    s.out_rows = static_cast<int>(nxr::forms::widthOf(out, static_cast<std::size_t>(s.op->nV)));
    return std::any(std::make_shared<FormState>(s));
  };
  fm.chunk = [](const std::any& st, const Blocks& ins, std::int64_t keep) {
    const FormState& s = *std::any_cast<const std::shared_ptr<FormState>&>(st);
    const Block b = crop(ins.at("x"), keep);
    // the field's rows as the type's frame: a free current's x y z a vertex is ℍ with w = 0
    const int per = s.t == nxr::forms::FieldType::Quaternion ? 4 : s.t == nxr::forms::FieldType::Complex ? 3 : 1;
    const int have = static_cast<int>(b.rows / s.op->nV);
    if (b.rows % s.op->nV != 0 || (have != per && !(per == 4 && have == 3))) refuse("form: the field's rows do not fit its type on the surface");
    std::vector<float> re(static_cast<std::size_t>(s.in_rows * b.width), 0.0f), im;
    if (b.complex()) im.assign(re.size(), 0.0f);
    const int off = per == 4 && have == 3 ? 1 : 0;
    for (std::int64_t t = 0; t < b.width; ++t)
      for (std::int64_t r = 0; r < b.rows; ++r) {
        const std::int64_t v = r / have, a = r % have, dst = t * s.in_rows + v * per + a + off;
        re[static_cast<std::size_t>(dst)] = b.re[static_cast<std::size_t>(r * b.width + t)];
        if (b.complex()) im[static_cast<std::size_t>(dst)] = b.im[static_cast<std::size_t>(r * b.width + t)];
      }
    const auto f = nxr::forms::formComplex(s.t, s.id, re.data(), b.complex() ? im.data() : nullptr, re.size(), *s.op);
    Block y{s.out_rows, b.width, b.left, b.right, std::vector<float>(static_cast<std::size_t>(s.out_rows * b.width)), {}};
    if (!f.im.empty()) y.im.resize(y.re.size());
    for (std::int64_t t = 0; t < b.width; ++t)
      for (int r = 0; r < s.out_rows; ++r) {
        y.re[static_cast<std::size_t>(r * b.width + t)] = f.re[static_cast<std::size_t>(t * s.out_rows + r)];
        if (!f.im.empty()) y.im[static_cast<std::size_t>(r * b.width + t)] = f.im[static_cast<std::size_t>(t * s.out_rows + r)];
      }
    return Blocks{{"y", std::move(y)}};
  };

  // |x|² summed over a vertex's components
  StepSpec& pw = at("power");
  pw.prepare = [](const Values& ins, const Json&, ExecContext&) { return std::any(per_vertex(ins.at("x"))); };
  pw.chunk = [](const std::any& st, const Blocks& ins, std::int64_t keep) {
    const int per = std::any_cast<int>(st);
    const Block b = crop(ins.at("x"), keep);
    Block y{b.rows / per, b.width, b.left, b.right, std::vector<float>(static_cast<std::size_t>(b.rows / per * b.width), 0.0f), {}};
    for (std::int64_t r = 0; r < b.rows; ++r)
      for (std::int64_t t = 0; t < b.width; ++t) {
        const double re = b.re[static_cast<std::size_t>(r * b.width + t)], im = b.complex() ? b.im[static_cast<std::size_t>(r * b.width + t)] : 0.0;
        float& o = y.re[static_cast<std::size_t>((r / per) * b.width + t)];
        o = static_cast<float>(static_cast<double>(o) + re * re + im * im);
      }
    return Blocks{{"y", std::move(y)}};
  };

  for (const char* n : {"phase_current", "gradient_power"}) {
    StepSpec& s = at(n);
    const bool current = std::string(n) == "phase_current";
    s.prepare = [](const Values& ins, const Json&, ExecContext& c) { return std::any(phase_state(ins.at("x"), c)); };
    s.chunk = [current](const std::any& st, const Blocks& ins, std::int64_t keep) {
      const PhaseState& ps = *std::any_cast<const std::shared_ptr<PhaseState>&>(st);
      const Block b = crop(ins.at("x"), keep);
      const Eigen::MatrixXcd Z = complex_of(b);
      const Eigen::MatrixXd Y = current ? nxr::phase::phaseCurrent(Z, ps.components, ps.G, ps.N) : nxr::phase::gradientPower(Z, ps.components, ps.G, ps.N);
      return Blocks{{"y", block_of(Y, b.left, b.right)}};
    };
  }

  // a field reduced over the run's span, per row: the Studio's reduceTiles on each run of good samples, folded (sums in double)
  StepSpec& ms = at("measure");
  ms.prepare = [](const Values& ins, const Json& p, ExecContext& ctx) {
    MeasureState st{p["measure"].get<std::string>(), std::nullopt, ins.at("x").rate, ins.at("x").origin, nullptr, 1};
    if (st.measure == "phase_current") st.ncomp = 3;
    if (ins.count("over") && nxr::io::text(*ins.at("over").row, "type") == "set") {
      const Value& x = ins.at("x");
      const Json j = Json::parse(nxr::io::text(*ins.at("over").row, "params_json"))["joint"];
      const auto man = ctx.bound.manifold(x.manifolds[0]);
      const nxr::pipelines::SurfaceInput S = ctx.bound.surface(nxr::io::text(*man, "name"), true);
      const nxr::pipelines::SurfaceLevel L = ctx.bound.surface_level(S, j["space_level"].get<int>());
      if (L.nodes != j["tiles"].get<std::int64_t>()) refuse("measure: the joint partition records " + j["tiles"].dump() + " tiles, level " + std::to_string(L.l) + " of " + S.name + " has " + std::to_string(L.nodes));
      nxr::ladder::Mesh mesh; mesh.n = static_cast<int>(S.n_vertices()); mesh.positions = S.vertices; mesh.faces = S.faces;
      auto js = std::make_shared<JointState>();
      js->nV = S.n_vertices(); js->S = L.nodes; js->tile = L.tile; js->area = nxr::ladder::vertexAreas(mesh);
      js->tile_area.assign(static_cast<std::size_t>(L.nodes), 0.0);
      for (std::int64_t v = 0; v < js->nV; ++v) if (js->tile[static_cast<std::size_t>(v)] >= 0) js->tile_area[static_cast<std::size_t>(js->tile[static_cast<std::size_t>(v)])] += js->area[static_cast<std::size_t>(v)];
      st.joint = js;
      TimeLevel tl;
      tl.id = nxr::io::text(*ins.at("over").row, "id"); tl.name = nxr::io::text(*ins.at("over").row, "name"); tl.level = j["time_level"].get<int>();
      tl.width = j["window_samples"].get<std::int64_t>(); tl.count = j["frames"].get<std::int64_t>();
      const auto tsel = ctx.bound.selection(j["time"].get<std::string>());
      tl.resolves = Json::parse(nxr::io::text(*tsel, "params_json"))["resolves_hz"];
      st.level = tl;
      return std::any(st);
    }
    if (ins.count("over")) {
      const Value& x = ins.at("x");
      st.level = level_of_band(ctx.bound, nxr::io::text(*ins.at("over").row, "id"), x.facts["band"], x.rate, x.origin);
    }
    if (st.measure == "median" && st.level && st.level->width > kMedianCap)
      refuse("measure: median: a tile of " + st.level->name + " holds " + std::to_string(st.level->width) + " samples a row, over the " + std::to_string(kMedianCap) + " that may be held (the median is exact: it holds a member's samples)");
    return std::any(st);
  };
  ms.start = [](const std::any& st) { return std::any_cast<const MeasureState&>(st).level ? std::any(TileAcc{}) : std::any(MeasureAcc{}); };
  ms.fold = [](const std::any& st_, std::any& acc_, const Blocks& ins, const Place& place) {
    const MeasureState& st = std::any_cast<const MeasureState&>(st_);
    if (st.level) {                                       // PER TILE of the level: each run of good samples inside one tile reduced (reduceTiles), summed into its tile
      TileAcc& acc = std::any_cast<TileAcc&>(acc_);
      const Block& b = ins.at("x");
      const std::int64_t R = st.joint ? st.joint->S : b.rows, n = b.own(), W = st.level->width;
      if (st.joint && b.rows != st.joint->nV * st.ncomp) refuse("measure: over a joint partition x has " + std::to_string(b.rows) + " rows, the surface " + std::to_string(st.joint->nV) + " vertices of " + std::to_string(st.ncomp) + " rows");
      auto good = [&](std::int64_t j) { return !(place.good && !place.good->empty() && !(*place.good)[static_cast<std::size_t>(j)]); };
      for (std::int64_t a = 0; a < n;) {
        if (!good(a)) { ++a; continue; }
        const std::int64_t tile = (place.first + a) / W;
        std::int64_t e = a;
        while (e < n && good(e) && (place.first + e) / W == tile) ++e;
        if (st.joint) {                                   // PER CELL: each vertex's value over the run, area-weighted into its spatial tile (Σ a_v x, and for |z| the largest); a vector (the phase current) per component
          const JointState& J = *st.joint;
          const std::int64_t C = st.ncomp;
          auto& t = acc.tiles[tile];
          if (t.sum.empty()) { t.sum.assign(static_cast<std::size_t>(R * C), 0.0); t.mx.assign(static_cast<std::size_t>(R), -std::numeric_limits<double>::infinity()); }
          for (std::int64_t v = 0; v < J.nV; ++v) {
            const std::int32_t s = J.tile[static_cast<std::size_t>(v)];
            if (s < 0) continue;
            for (std::int64_t c = 0; c < C; ++c) {
              double sum = 0, mx = -std::numeric_limits<double>::infinity();
              const float* z = b.re.data() + (v * C + c) * b.width + b.left;
              for (std::int64_t u = a; u < e; ++u) { sum += z[u]; mx = std::max(mx, static_cast<double>(z[u])); }
              t.sum[static_cast<std::size_t>(s * C + c)] += J.area[static_cast<std::size_t>(v)] * sum;
              if (C == 1) t.mx[static_cast<std::size_t>(s)] = std::max(t.mx[static_cast<std::size_t>(s)], mx);
            }
          }
          t.count += e - a;
          acc.rows = R;
          a = e;
          continue;
        }
        if (st.measure == "median") {                     // hold the tile's samples; the tiles before this one are closed (chunks arrive in time order)
          for (auto& [k, tp] : acc.tiles) if (k < tile && !tp.held.empty()) tp.med = close_medians(tp.held);
          auto& t = acc.tiles[tile];
          if (t.held.empty()) t.held.assign(static_cast<std::size_t>(R), {});
          if (t.count + (e - a) > kMedianCap) refuse("measure: median: a tile holds more than " + std::to_string(kMedianCap) + " samples a row (the median is exact: it holds a member's samples)");
          for (std::int64_t r = 0; r < R; ++r) t.held[static_cast<std::size_t>(r)].insert(t.held[static_cast<std::size_t>(r)].end(), b.re.begin() + r * b.width + b.left + a, b.re.begin() + r * b.width + b.left + e);
          t.count += e - a;
          acc.rows = R;
          a = e;
          continue;
        }
        auto& t = acc.tiles[tile];
        if (t.sum.empty()) { t.sum.assign(static_cast<std::size_t>(R), 0.0); t.mx.assign(static_cast<std::size_t>(R), -std::numeric_limits<double>::infinity()); }
        for (std::int64_t r = 0; r < R; ++r) {
          const auto red = nxr::partition::reduceTiles(b.re.data() + r * b.width + b.left + a, 1, static_cast<std::size_t>(e - a), static_cast<std::size_t>(e - a), 1);
          t.mx[static_cast<std::size_t>(r)] = std::max(t.mx[static_cast<std::size_t>(r)], static_cast<double>(red.max[0]));   // envelope_max: the largest |z| of the tile
          t.sum[static_cast<std::size_t>(r)] += red.sum[0];   // amplitude: Σ|z| of the envelope; power: Σ|z|² (the power step's values)
        }
        t.count += e - a;
        acc.rows = R;
        a = e;
      }
      return;
    }
    MeasureAcc& acc = std::any_cast<MeasureAcc&>(acc_);
    const Block& b = ins.at("x");
    const std::size_t R = static_cast<std::size_t>(b.rows);
    if (acc.sum.empty()) {
      acc.sum.assign(R, 0.0); acc.sumSq.assign(R, 0.0); acc.count.assign(R, 0.0);
      acc.min.assign(R, std::numeric_limits<float>::infinity()); acc.max.assign(R, -std::numeric_limits<float>::infinity());
    }
    const std::int64_t n = b.own();
    std::int64_t a = 0;
    if (st.measure == "median" && acc.held.empty()) acc.held.assign(R, {});
    while (a < n) {
      if (place.good && !place.good->empty() && !(*place.good)[static_cast<std::size_t>(a)]) { ++a; continue; }
      std::int64_t e = a;
      while (e < n && !(place.good && !place.good->empty() && !(*place.good)[static_cast<std::size_t>(e)])) ++e;
      if (st.measure == "median") {                       // the span is one member: its good samples held per row until the span ends
        if (static_cast<std::int64_t>(acc.held[0].size()) + (e - a) > kMedianCap) refuse("measure: median: the span holds more than " + std::to_string(kMedianCap) + " good samples a row (the median is exact: it holds a member's samples); read a shorter span");
        for (std::size_t r = 0; r < R; ++r) acc.held[r].insert(acc.held[r].end(), b.re.begin() + static_cast<std::int64_t>(r) * b.width + b.left + a, b.re.begin() + static_cast<std::int64_t>(r) * b.width + b.left + e);
        acc.count[0] += static_cast<double>(e - a);
        a = e;
        continue;
      }
      for (std::size_t r = 0; r < R; ++r) {
        const auto t = nxr::partition::reduceTiles(b.re.data() + static_cast<std::int64_t>(r) * b.width + b.left + a, 1, static_cast<std::size_t>(e - a), static_cast<std::size_t>(e - a), 1);
        acc.sum[r] += t.sum[0]; acc.sumSq[r] += t.sumSq[0]; acc.count[r] += t.count[0];
        acc.min[r] = std::min(acc.min[r], t.min[0]); acc.max[r] = std::max(acc.max[r], t.max[0]);
      }
      a = e;
    }
  };
  ms.finish = [](const std::any& st_, std::any& acc_) {
    const MeasureState& st = std::any_cast<const MeasureState&>(st_);
    if (st.level) {                                       // [rows x tiles of the level]: the mean over a tile WHOLE in the span and free of bad samples, else NaN
      TileAcc& acc = std::any_cast<TileAcc&>(acc_);
      if (acc.tiles.empty()) refuse("measure: no good sample in the span");
      const std::int64_t T = st.level->count, W = st.level->width, R = acc.rows;
      const std::int64_t C = st.ncomp;
      Array y{R, T, std::vector<double>(static_cast<std::size_t>(R * T * C), std::numeric_limits<double>::quiet_NaN())};
      y.comps = C;
      Json counts = Json::array();
      std::int64_t whole = 0, first = -1, last = -1;
      for (auto& [tile, t] : acc.tiles) {
        if (tile < 0 || tile >= T) continue;
        counts.push_back(Json::array({tile, t.count}));
        if (t.count != W) continue;                       // cut by the span's edge, or holding a bad sample: not this tiling's window
        ++whole;
        if (first < 0) first = tile;
        last = tile;
        if (st.measure == "median") {
          if (!t.held.empty()) t.med = close_medians(t.held);
          for (std::int64_t r = 0; r < R; ++r) y.data[static_cast<std::size_t>(r * T + tile)] = t.med[static_cast<std::size_t>(r)];
          continue;
        }
        for (std::int64_t r = 0; r < R; ++r)
          for (std::int64_t c = 0; c < C; ++c) {
            // a joint cell's value is the AREA-WEIGHTED mean: Σ a_v x over the tile's area and the window's samples (a tile of no vertex: NaN)
            const double denom = st.joint ? static_cast<double>(W) * st.joint->tile_area[static_cast<std::size_t>(r)] : static_cast<double>(W);
            y.data[static_cast<std::size_t>((r * T + tile) * C + c)] = !(denom > 0) ? std::numeric_limits<double>::quiet_NaN() : st.measure == "envelope_max" ? t.mx[static_cast<std::size_t>(r)] : t.sum[static_cast<std::size_t>(r * C + c)] / denom;
          }
      }
      if (!whole) refuse("measure: no tile of " + st.level->name + " (" + std::to_string(W) + " samples) lies whole and good in the span");
      y.meta = Json{{"measure", st.measure}, {"selection_id", st.level->id}, {"selection", st.level->name}, {"window_samples", W}, {"width_s", static_cast<double>(W) / st.rate},
                    {"origin", st.origin}, {"resolves_hz", st.level->resolves}, {"tiles", T}, {"whole_tiles", whole},
                    {"tiles_measured", Json::array({first, last})}, {"counts", counts}};
      if (st.joint) { y.meta["joint"] = true; y.meta["space_tiles"] = R; }
      return Arrays{{"y", std::move(y)}};
    }
    const std::string& m = st.measure;
    MeasureAcc& acc = std::any_cast<MeasureAcc&>(acc_);
    if (acc.count.empty() || acc.count[0] == 0) refuse("measure: no good sample in the span");
    if (m == "median") {
      const std::vector<double> med = close_medians(acc.held);
      return Arrays{{"y", Array{static_cast<std::int64_t>(med.size()), 1, med}}};
    }
    Array y{static_cast<std::int64_t>(acc.sum.size()), 1, std::vector<double>(acc.sum.size())};
    for (std::size_t r = 0; r < acc.sum.size(); ++r)
      y.data[r] = m == "mean" ? acc.sum[r] / acc.count[r] : m == "mean_square" ? acc.sumSq[r] / acc.count[r] : m == "max" ? acc.max[r] : m == "min" ? acc.min[r]
                : std::max(std::fabs(acc.min[r]), std::fabs(acc.max[r]));
    return Arrays{{"y", std::move(y)}};
  };

  // a measurement's tiles labelled (Rheome detect.spindle's rule, in whole tiles): out [spans x 4] = (first tile, stop tile, row or −1, peak value), the
  // thresholds per row in meta
  // a measurement of the run summarised (nxr::stats::summarizeRows): [rows' x summaries], the summaries' names and the members used in meta
  StepSpec& sc = at("summarize");
  sc.prepare = [](const Values&, const Json& p, ExecContext&) { return std::any(p); };
  sc.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
    const Json& p = std::any_cast<const Json&>(st);
    const Array& x = ins.at("x");
    std::vector<std::string> names;
    for (const auto& n : p["summaries"]) names.push_back(n.get<std::string>());
    const auto sums = nxr::stats::parseSummaries(names);
    std::vector<std::int64_t> used;
    // no members at all (a run that found no event): one NaN member, so the counts and sums read 0 and the rest NaN
    const std::vector<double> none(static_cast<std::size_t>(x.rows), std::numeric_limits<double>::quiet_NaN());
    const auto v = nxr::stats::summarizeRows(x.cols ? x.data : none, x.rows, std::max<std::int64_t>(1, x.cols), sums, p["combine"].get<std::string>(), p["ddof"].get<int>(), &used);
    Array y{p["combine"] == "each" ? x.rows : 1, static_cast<std::int64_t>(sums.size()), v};
    Json sj = Json::array();
    for (const auto& sm : sums) sj.push_back(sm.name == "percentile" ? Json{{"name", "percentile"}, {"p", sm.p}} : Json{{"name", sm.name}});
    y.meta = Json{{"summaries", sj}, {"combine", p["combine"]}, {"ddof", p["ddof"]}, {"members", x.cols}, {"members_used", used}, {"of", x.meta}};
    return Arrays{{"y", std::move(y)}};
  };

  // the events' values from the tile measurement: [rows x events], an event on its row (every row for -1), counts the samples over its tiles. The events are those of this run's `detect_spans`
  // (an array) or a STORED spans Selection (read in preparation, snapped to the tiles: global, so on every row). A vector measurement (comps 3) is rolled up per component: the count-weighted
  // mean of each, never of a ratio (phase_gradient is derived at read from the stored sums).
  struct RollupState { std::string m; std::optional<std::vector<nxr::intervals::TileEvent>> stored; Json selection; };
  StepSpec& re = at("rollup_events");
  re.prepare = [](const Values& ins, const Json& p, ExecContext& ctx) {
    RollupState st{p["measure"].get<std::string>(), std::nullopt, nullptr};
    const Value& ev = ins.at("events");
    if (ev.cls == "selection") {
      st.stored = stored_event_tiles(ctx.bound, ev, ins.at("x"), "rollup_events");
      st.selection = Json{{"id", nxr::io::text(*ev.row, "id")}, {"name", nxr::io::text(*ev.row, "name")}};
    }
    return std::any(st);
  };
  re.run = [](const std::any& st_, const Arrays& ins, ExecContext&) {
    const RollupState& st = std::any_cast<const RollupState&>(st_);
    const std::string& m = st.m;
    const Array& x = ins.at("x");
    const std::int64_t R = x.rows, T = x.cols, C = x.comps;
    const double width_s = x.meta.value("width_s", 0.0);
    need(width_s > 0, "rollup_events: x carries no tile width");
    Eigen::VectorXd counts = Eigen::VectorXd::Zero(T);
    for (const auto& c : x.meta["counts"]) counts(c[0].get<std::int64_t>()) = c[1].get<double>();
    std::vector<nxr::intervals::TileEvent> evs;
    if (st.stored) evs = *st.stored;
    else for (std::int64_t k = 0; k < ins.at("events").rows; ++k) { const double* r = ins.at("events").data.data() + k * 4; evs.push_back({static_cast<std::int64_t>(r[0]), static_cast<std::int64_t>(r[1]), static_cast<std::int64_t>(r[2])}); }
    const std::int64_t E = static_cast<std::int64_t>(evs.size());
    Array y{R, E, std::vector<double>(static_cast<std::size_t>(R * E * C), std::numeric_limits<double>::quiet_NaN())};
    y.comps = C;
    Eigen::VectorXd ecount = Eigen::VectorXd::Zero(E);
    if (E > 0) {
      if (m == "duration" || m == "interval_next") {
        const auto g = m == "duration" ? nxr::intervals::eventDurations(evs, width_s) : nxr::intervals::eventIntervalNext(evs, width_s);
        for (std::int64_t e = 0; e < E; ++e) {
          for (std::int64_t t = evs[static_cast<std::size_t>(e)].first; t < evs[static_cast<std::size_t>(e)].stop; ++t) ecount(e) += counts(t);
          for (std::int64_t r = 0; r < R; ++r) if (evs[static_cast<std::size_t>(e)].row < 0 || evs[static_cast<std::size_t>(e)].row == r) y.data[static_cast<std::size_t>(r * E + e)] = g[static_cast<std::size_t>(e)];
        }
      } else {
        for (std::int64_t c = 0; c < C; ++c) {                // each component: the count-weighted mean (max for envelope_max) of the tiles over each event
          Eigen::MatrixXd M(R, T);
          for (std::int64_t r = 0; r < R; ++r) for (std::int64_t t = 0; t < T; ++t) M(r, t) = x.data[static_cast<std::size_t>((r * T + t) * C + c)];
          const auto v = nxr::intervals::rollupEvents(M, counts, evs, m == "envelope_max" ? "max" : "mean");
          for (std::int64_t r = 0; r < R; ++r) for (std::int64_t e = 0; e < E; ++e) y.data[static_cast<std::size_t>((r * E + e) * C + c)] = v.value(r, e);
          ecount = v.count;
        }
      }
    }
    Json tl = x.meta;
    tl.erase("counts");
    Json counts_j = Json::array();
    for (std::int64_t e = 0; e < E; ++e) counts_j.push_back(Json::array({e, ecount(e)}));
    std::int64_t finite = 0;
    for (std::int64_t e = 0; e < E; ++e) { bool any = false; for (std::int64_t r = 0; r < R; ++r) any = any || std::isfinite(y.data[static_cast<std::size_t>((r * E + e) * C)]); finite += any; }
    y.meta = Json{{"measure", m}, {"tiles", E}, {"counts", counts_j}, {"whole_tiles", finite}, {"tiles_measured", nullptr},
                  {"window_samples", x.meta.value("window_samples", std::int64_t(0))}, {"width_s", width_s}, {"origin", x.meta.value("origin", 0.0)}, {"tile_level", tl}};
    if (st.stored) { y.meta["selection_id"] = st.selection["id"]; y.meta["selection"] = st.selection["name"]; y.meta["stored_events"] = true; if (x.meta.value("joint", false)) y.meta["space_tiles"] = R; }
    else { y.meta["events"] = true; y.meta["selection"] = "the events of this run"; }
    return Arrays{{"y", std::move(y)}};
  };

  // the max over each group's spatial tiles of a joint measurement: [S x T] -> [groups x T]; its meta is the TIME level's (the cells' time tiles)
  StepSpec& rc = at("reduce_cells");
  rc.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
    const Json& over = ins.at("x").facts["over"];
    const Json j = Json::parse(nxr::io::text(*c.bound.selection(over["selection_id"].get<std::string>()), "params_json"))["joint"];
    const auto tl = c.bound.selection(j["time"].get<std::string>());
    return std::any(Json{{"groups", p["groups"]}, {"space_level", over["space_level"]}, {"time_id", nxr::io::text(*tl, "id")}, {"time_name", nxr::io::text(*tl, "name")}});
  };
  rc.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
    const Json& p = std::any_cast<const Json&>(st);
    const Array& x = ins.at("x");
    const std::int64_t S = x.rows, T = x.cols, l = p["space_level"].get<std::int64_t>(), G = p["groups"] == "per_hemisphere" ? 2 : 1;
    Array y{G, T, std::vector<double>(static_cast<std::size_t>(G * T), std::numeric_limits<double>::quiet_NaN())};
    for (std::int64_t s = 0; s < S; ++s) {
      const std::int64_t g = G == 2 ? (s >> l) : 0;
      for (std::int64_t t = 0; t < T; ++t) {
        const double v = x.data[static_cast<std::size_t>(s * T + t)];
        double& o = y.data[static_cast<std::size_t>(g * T + t)];
        if (std::isfinite(v) && (std::isnan(o) || v > o)) o = v;
      }
    }
    y.meta = x.meta;
    y.meta["selection_id"] = p["time_id"]; y.meta["selection"] = p["time_name"];
    y.meta.erase("joint"); y.meta.erase("space_tiles");
    y.meta["reduced"] = Json{{"groups", p["groups"]}, {"spatial_tiles", S}, {"by", "max over the group's spatial tiles"}};
    return Arrays{{"y", std::move(y)}};
  };

  // each event fitted on the sample series (nxr::risefall: Rheome's detect.risefall); the events' tiles give [t0, t1]
  StepSpec& rf = at("risefall");
  rf.prepare = [](const Values&, const Json& p, ExecContext&) { return std::any(p); };
  rf.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
    const Json& p = std::any_cast<const Json&>(st);
    const Array &x = ins.at("x"), &ev = ins.at("events");
    const std::int64_t R = x.rows, E = ev.rows;
    const double fs = x.meta["rate"].get<double>(), origin = x.meta["origin"].get<double>(), first = static_cast<double>(x.meta["first_sample"].get<std::int64_t>());
    const double tw = ev.meta["tiles_of"]["width_s"].get<double>(), t_origin = ev.meta["tiles_of"]["origin"].get<double>();
    const double offset = origin + first / fs;                 // the series' first sample's time on the recording's clock
    nxr::risefall::Options o;
    o.model = nxr::risefall::parseModel(p["model"].get<std::string>()); o.margin = p["margin"].get<double>(); o.smooth = p["smooth"].get<double>(); o.tauMax = p["tau_max"].get<double>();
    // DURATIONS IN CYCLES: a cycle is 1/centre of the band the tiles resolve (nxr burst_stages: margin 10 cycles, smoothing 1 cycle, tau <= 10 cycles)
    Json durations = Json{{"mode", "seconds"}};
    if (p["durations"] == "cycles") {
      need(ev.meta["tiles_of"].contains("resolves_hz"), "risefall: durations cycles: the events' tiles carry no band");
      const double centre = (ev.meta["tiles_of"]["resolves_hz"]["lo"].get<double>() + ev.meta["tiles_of"]["resolves_hz"]["hi"].get<double>()) / 2, cyc = 1.0 / centre;
      o.margin = p["margin_cycles"].get<double>() * cyc; o.smooth = p["smooth_cycles"].get<double>() * cyc; o.tauMax = p["tau_max_cycles"].get<double>() * cyc;
      durations = Json{{"mode", "cycles"}, {"centre_hz", centre}, {"cycle_s", cyc}, {"margin", o.margin}, {"smooth", o.smooth}, {"tau_max", o.tauMax}};
    }
    static const char* names[] = {"t_on", "t_off", "tau_rise", "tau_fall", "fit_baseline", "fit_height", "fit_r2", "plateau_duration"};
    std::map<std::string, Array> out;
    for (const char* n : names) out[n] = Array{R, E, std::vector<double>(static_cast<std::size_t>(R * E), std::numeric_limits<double>::quiet_NaN())};
    Json fits = Json::array();
    std::int64_t ok = 0, bound = 0, below = 0;
    for (std::int64_t e = 0; e < E; ++e) {
      const double* r = ev.data.data() + e * 4;
      const std::int64_t row = static_cast<std::int64_t>(r[2]);
      if (row >= R) refuse("risefall: an event on row " + std::to_string(row) + " but the series has " + std::to_string(R) + " rows (the events must come from a trace of these rows)");
      const double t0 = t_origin + r[0] * tw - offset, t1 = t_origin + r[1] * tw - offset;
      // an event of row -1 (a global event, `combine mean`) is fitted on the MEAN over the series' rows and holds on every row
      Eigen::VectorXd series = Eigen::Map<const Eigen::VectorXd>(x.data.data() + std::max<std::int64_t>(row, 0) * x.cols, x.cols);
      if (row < 0) { series.setZero(); for (std::int64_t q = 0; q < R; ++q) series += Eigen::Map<const Eigen::VectorXd>(x.data.data() + q * x.cols, x.cols); series /= static_cast<double>(R); }
      const auto F = nxr::risefall::fit(series, fs, t0, t1, o);
      const bool keep = F.ok && !(p["min_r2"].is_number() && F.r2 < p["min_r2"].get<double>());
      ok += F.ok; bound += F.ok && F.atBound; below += F.ok && !keep;
      fits.push_back(Json{{"event", e}, {"row", row}, {"ok", F.ok}, {"kept", keep}, {"at_bound", F.atBound}, {"r2", F.ok ? Json(F.r2) : Json()}});
      if (!keep) continue;
      const double vals[] = {F.tOn + offset, F.tOff + offset, F.tauR, F.tauF, F.b, F.h, F.r2, F.hasPlateau ? F.plateau[1] - F.plateau[0] : 0.0};
      for (int k = 0; k < 8; ++k)
        for (std::int64_t q = row < 0 ? 0 : row; q < (row < 0 ? R : row + 1); ++q) out[names[k]].data[static_cast<std::size_t>(q * E + e)] = vals[k];
    }
    // the count row an event's measures share: the samples over the event's tiles (as rollup_events', one row a Selection and recording)
    std::map<std::int64_t, double> tile_count;
    for (const auto& c : ev.meta["tiles_of"]["counts"]) tile_count[c[0].get<std::int64_t>()] = c[1].get<double>();
    Json counts = Json::array();
    for (std::int64_t e = 0; e < E; ++e) {
      double n = 0;
      for (std::int64_t t = static_cast<std::int64_t>(ev.data[static_cast<std::size_t>(e * 4)]); t < static_cast<std::int64_t>(ev.data[static_cast<std::size_t>(e * 4 + 1)]); ++t) n += tile_count[t];
      counts.push_back(Json::array({e, n}));
    }
    Arrays res;
    for (const char* n : names) {
      Array& y = out[n];
      y.meta = Json{{"measure", n}, {"events", true}, {"tiles", E}, {"counts", counts}, {"whole_tiles", ok - below}, {"tiles_measured", nullptr}, {"selection", "the events of this run"},
                    {"window_samples", ev.meta["tiles_of"].value("window_samples", std::int64_t(0))}, {"width_s", tw}, {"origin", t_origin}, {"tile_level", ev.meta["tiles_of"]},
                    {"risefall", Json{{"model", p["model"]}, {"margin", o.margin}, {"smooth", o.smooth}, {"tau_max", o.tauMax}, {"durations", durations}, {"min_r2", p["min_r2"]}, {"ok", ok}, {"at_bound", bound}, {"below_min_r2", below},
                                      {"series_rate", fs}, {"fits", fits}, {"reference", "Rheome +detect/risefall.m"}}}};
      res[n] = std::move(y);
    }
    return res;
  };

  // THE FIT GATE and THE STAGES: from the stored fits (risefall's measurements), per event; spans in the detection's own tile units (fractional for a stage: seconds, not tile edges)
  auto event_fit = [](const Arrays& ins, std::int64_t e) {
    const Array& r2 = ins.at("fit_r2");
    nxr::risefall::Fit F;
    const Json& f = r2.meta["risefall"]["fits"][static_cast<std::size_t>(e)];
    F.ok = f.value("ok", false); F.atBound = f.value("at_bound", false); F.r2 = f["r2"].is_number() ? f["r2"].get<double>() : std::numeric_limits<double>::quiet_NaN();
    return F;
  };
  StepSpec& fg = at("fit_gate");
  fg.prepare = [](const Values&, const Json& p, ExecContext&) { return std::any(p); };
  fg.run = [event_fit](const std::any& st, const Arrays& ins, ExecContext&) {
    const Json& p = std::any_cast<const Json&>(st);
    const Array& ev = ins.at("events");
    const double min_r2 = p["min_r2"].get<double>();
    Array out{0, 4, {}};
    Json ordinals = Json::array();
    std::int64_t refused_bound = 0, refused_r2 = 0, refused_fit = 0;
    for (std::int64_t e = 0; e < ev.rows; ++e) {
      const nxr::risefall::Fit F = event_fit(ins, e);
      if (nxr::risefall::passesGate(F, min_r2)) {
        out.data.insert(out.data.end(), ev.data.begin() + e * 4, ev.data.begin() + e * 4 + 4);
        ++out.rows;
        ordinals.push_back(e);
      } else if (!F.ok) ++refused_fit;
      else if (F.atBound) ++refused_bound;
      else ++refused_r2;
    }
    out.meta = ev.meta;
    out.meta["gate"] = Json{{"min_r2", min_r2}, {"tau", "not within 1% of its bound"}, {"events", ev.rows}, {"kept", out.rows}, {"ordinals", ordinals}, {"refused", Json{{"no_fit", refused_fit}, {"tau_at_bound", refused_bound}, {"r2_below", refused_r2}}},
                            {"reference", "nxr vortex.burst_stages"}};
    return Arrays{{"spans", std::move(out)}};
  };
  StepSpec& bs = at("burst_stages");
  bs.prepare = [](const Values& ins, const Json&, ExecContext&) { return std::any(ins.count("kept") > 0); };
  bs.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
    const bool gated = std::any_cast<bool>(st);
    const Array &ev = ins.at("events"), &on = ins.at("t_on"), &off = ins.at("t_off"), &tr = ins.at("tau_rise"), &tf = ins.at("tau_fall");
    const std::int64_t R = on.rows, E = ev.rows;
    const Json& tiles = ev.meta["tiles_of"];
    const double w = tiles["width_s"].get<double>(), t_origin = tiles["origin"].get<double>(), T = tiles.value("tiles", 0) > 0 ? static_cast<double>(tiles["tiles"].get<std::int64_t>()) : std::numeric_limits<double>::infinity();
    std::vector<bool> keep(static_cast<std::size_t>(E), !gated);
    if (gated) for (const auto& o : ins.at("kept").meta["gate"]["ordinals"]) keep[static_cast<std::size_t>(o.get<std::int64_t>())] = true;
    Arrays res;
    for (const char* n : {"onset", "within", "offset"}) res[n] = Array{0, 4, {}};
    std::int64_t with_stages = 0;
    for (std::int64_t e = 0; e < E; ++e) {
      if (!keep[static_cast<std::size_t>(e)]) continue;
      const std::int64_t row = static_cast<std::int64_t>(ev.data[static_cast<std::size_t>(e * 4 + 2)]), q = std::max<std::int64_t>(row, 0);
      if (q >= R) refuse("burst_stages: an event on row " + std::to_string(row) + " but the fits have " + std::to_string(R) + " rows");
      const auto at = [&](const Array& a) { return a.data[static_cast<std::size_t>(q * a.cols + e)]; };
      const nxr::risefall::Stages S = nxr::risefall::stagesOf(at(on), at(off), at(tr), at(tf));
      if (!S.valid) continue;
      ++with_stages;
      auto put = [&](const char* n, const double* iv) {
        const double a = std::max(0.0, (iv[0] - t_origin) / w), b = std::min(T, (iv[1] - t_origin) / w);       // seconds on the recording's clock → tile units; clipped to the run
        if (!(b > a)) return;
        Array& o = res[n];
        o.data.insert(o.data.end(), {a, b, static_cast<double>(row), static_cast<double>(e)});                 // the value: the event's ordinal among the detected ones
        ++o.rows;
      };
      put("onset", S.onset);
      if (S.hasWithin) put("within", S.within);
      put("offset", S.offset);
    }
    for (auto& [n, o] : res) {
      o.meta = ev.meta;
      o.meta["stages"] = Json{{"stage", n}, {"rule", std::string(n) == "onset" ? "t_on ± 2 tau_rise" : std::string(n) == "within" ? "[t_on + 2 tau_rise, t_off − 2 tau_fall], none when they meet" : "t_off ± 2 tau_fall"},
                              {"events", E}, {"with_stages", with_stages}, {"gated", gated}, {"edges", "seconds (fractions of a tile), clipped to the run"}, {"reference", "nxr vortex.burst_stages"}};
    }
    return res;
  };

  StepSpec& ds = at("detect_spans");
  ds.prepare = [](const Values&, const Json& p, ExecContext&) { return std::any(p); };
  ds.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
    Json p = std::any_cast<const Json&>(st);
    const Array& x = ins.at("x");
    // DURATIONS IN CYCLES: a cycle is 1/centre of the band the tiles resolve; the three durations become seconds here, then are compared in whole tiles as ever
    Json durations = Json{{"mode", "seconds"}};
    if (p["durations"] == "cycles") {
      need(x.meta.contains("resolves_hz"), "detect_spans: durations cycles: x's tiles carry no band");
      const double centre = (x.meta["resolves_hz"]["lo"].get<double>() + x.meta["resolves_hz"]["hi"].get<double>()) / 2;
      need(centre > 0, "detect_spans: durations cycles: the band's centre is above 0 Hz");
      const double cyc = 1.0 / centre;
      p["merge_s"] = p["merge_cycles"].get<double>() * cyc;
      p["min_s"] = p["min_cycles"].get<double>() * cyc;
      p["max_s"] = p["max_cycles"].is_null() ? Json() : Json(p["max_cycles"].get<double>() * cyc);
      durations = Json{{"mode", "cycles"}, {"centre_hz", centre}, {"cycle_s", cyc}, {"merge_s", p["merge_s"]}, {"min_s", p["min_s"]}, {"max_s", p["max_s"]}};
    }
    const std::int64_t R = x.rows, T = x.cols, W = x.meta.value("window_samples", std::int64_t(0));
    const double width_s = x.meta.value("width_s", 0.0);
    need(W > 0 && width_s > 0, "detect_spans: x carries no tile width");
    // seconds in whole tiles: joined when the tiles between them span ≤ merge_s; kept when min_s ≤ tiles × width ≤ max_s
    const std::int64_t gap = static_cast<std::int64_t>(std::floor(p["merge_s"].get<double>() / width_s + 1e-9)) + 1;
    const std::int64_t minL = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::ceil(p["min_s"].get<double>() / width_s - 1e-9)));
    const std::int64_t maxL = p["max_s"].is_null() ? std::numeric_limits<std::int64_t>::max() : static_cast<std::int64_t>(std::floor(p["max_s"].get<double>() / width_s + 1e-9));
    std::vector<std::pair<std::int64_t, Eigen::VectorXd>> series;    // (row, values)
    if (p["combine"] == "mean") {
      Eigen::VectorXd v(T);
      for (std::int64_t t = 0; t < T; ++t) { double s = 0; for (std::int64_t r = 0; r < R; ++r) s += x.data[static_cast<std::size_t>(r * T + t)]; v(t) = s / static_cast<double>(R); }
      series.emplace_back(-1, v);
    } else {
      for (std::int64_t r = 0; r < R; ++r) series.emplace_back(r, Eigen::Map<const Eigen::VectorXd>(x.data.data() + r * T, T));
    }
    Array out{0, 4, {}};
    Json thresholds = Json::array();
    // the median and robust SD per series: from the BASELINE statistic when given (summarize, same rows), else computed here
    const Array* base = ins.count("baseline") ? &ins.at("baseline") : nullptr;
    std::int64_t bmed = -1, bmad = -1;
    if (base) for (std::size_t k = 0; k < base->meta["summaries"].size(); ++k) { const std::string n = base->meta["summaries"][k]["name"]; if (n == "median") bmed = static_cast<std::int64_t>(k); if (n == "mad") bmad = static_cast<std::int64_t>(k); }
    std::size_t si = 0;
    for (const auto& [row, v] : series) {
      const std::size_t this_series = si++;
      std::vector<double> good;
      for (Eigen::Index t = 0; t < v.size(); ++t) if (std::isfinite(v(t))) good.push_back(v(t));
      if (good.size() < 2) { thresholds.push_back(Json{{"row", row}, {"skipped", "fewer than 2 good tiles"}}); continue; }
      const Eigen::Map<const Eigen::VectorXd> g(good.data(), static_cast<Eigen::Index>(good.size()));
      const double med = base ? base->data[this_series * static_cast<std::size_t>(base->cols) + static_cast<std::size_t>(bmed)] : nxr::intervals::median(g);
      const double sd = base ? base->data[this_series * static_cast<std::size_t>(base->cols) + static_cast<std::size_t>(bmad)] : nxr::intervals::mad(g);
      if (!(sd > 0)) { thresholds.push_back(Json{{"row", row}, {"median", med}, {"robust_sd", sd}, {"skipped", "no spread (robust SD 0)"}}); continue; }
      const double hi = med + p["enter_sd"].get<double>() * sd, lo = med + p["exit_sd"].get<double>() * sd;
      Eigen::VectorXd z = v;
      for (Eigen::Index t = 0; t < z.size(); ++t) if (!std::isfinite(z(t))) z(t) = -std::numeric_limits<double>::infinity();   // a bad tile ends a span
      auto spans = nxr::intervals::keepSpans(nxr::intervals::mergeSpans(nxr::intervals::crossings(z, hi, lo), gap), minL, maxL);
      thresholds.push_back(Json{{"row", row}, {"median", med}, {"robust_sd", sd}, {"enter", hi}, {"exit", lo}, {"good_tiles", static_cast<std::int64_t>(good.size())}, {"spans", static_cast<std::int64_t>(spans.size())}});
      for (const auto& sp : spans) {
        out.data.insert(out.data.end(), {static_cast<double>(sp.start), static_cast<double>(sp.stop), static_cast<double>(row), sp.value});
        ++out.rows;
      }
    }
    out.meta = Json{{"tiles_of", x.meta}, {"durations", durations}, {"thresholds", thresholds}, {"tiles_compared", Json{{"merge_gap_tiles_below", gap}, {"min_tiles", minL},
                    {"max_tiles", p["max_s"].is_null() ? Json() : Json(maxL)}, {"width_s", width_s}}},
                    {"statistic", ins.count("baseline") ? "per series: the median and the robust SD (1.4826 x MAD) of the baseline statistic (summarize)" :
                                                         "per series: the median and the robust SD (1.4826 x MAD) over its good tiles in the run's span (nxr::intervals::median, mad), computed here"}};
    return Arrays{{"spans", std::move(out)}};
  };
}
}  // namespace

const std::vector<StepSpec>& registry() {
  static const std::vector<StepSpec> r = build();
  return r;
}

const StepSpec& step(const std::string& name) {
  for (const auto& s : registry()) if (s.name == name) return s;
  std::vector<std::string> n;
  for (const auto& s : registry()) n.push_back(s.name);
  std::sort(n.begin(), n.end());
  std::string list;
  for (const auto& x : n) list += (list.empty() ? "" : ", ") + x;
  refuse("no pipeline step '" + name + "' (registered: " + list + ")");
}

std::vector<std::string> undeclared_traits() {
  std::vector<std::string> out;
  for (const auto& s : registry()) if (!s.traits.declared()) out.push_back(s.name);
  return out;
}

void check_traits(const StepSpec& node) {
  if (!node.traits.declared())
    refuse(node.name + ": the node declares no time behaviour or numeric precision (NodeTraits, registry.h): a node is registered with its traits (windowed: with its window rule)");
}

Json catalogue() {
  Json out = Json::array();
  std::vector<const StepSpec*> s;
  for (const auto& x : registry()) s.push_back(&x);
  std::sort(s.begin(), s.end(), [](auto a, auto b) { return a->name < b->name; });
  for (const auto* x : s)
    out.push_back(Json{{"name", x->name}, {"kind", x->kind}, {"inputs", x->inputs}, {"optional", x->optional}, {"outputs", x->outputs}, {"parameters", x->parameters},
                       {"matches", x->matches}, {"description", x->description}});
  return out;
}

}  // namespace nxr::pipelines::graph
