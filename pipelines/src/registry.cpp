// nxr-pipelines — the registered steps (registry.h): their ports, parameters, references and arrow checks. The steps' maths are pure functions of compute/
// and meeg/ (bound and run by the runner); this file only declares and checks, as nxr's nxr/pipelines/steps.py.
//
//   band_filter     chunk       a field on … × time, band-limited, zero phase (the Studio's taps of a Meyer band: gainTaps · convolveRows)
//   analytic        chunk       ℝ → ℂ, the analytic signal x + i H[x] (the Studio's odd taps, hilbertImag; the transform "complex")
//   envelope        chunk       ℝ → ℝ, |analytic| (NON-LINEAR: no linear map may follow it, D106)
//   frames          chunk       RETIMING: a field on the recording's samples sampled on the FRAME TILING (`run frametiling`: the cycles of one TOWER level, D135; nxr band_at_frames resamples at them):
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
#include "nxr/pipelines/welch_steps.h"
#include "nxr/pipelines/meeg_steps.h"
#include "nxr/pipelines/partition_steps.h"
#include "nxr/pipelines/measurement_nodes.h"
#include "nxr/pipelines/select_nodes.h"
#include "nxr/quantity.h"

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
#include "nxr/intervals.h"
#include "nxr/stats.h"
#include "nxr/phase.h"
#include "nxr/pyramid.h"
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
void add_surrogate_steps(std::vector<StepSpec>& r);   // surrogate_steps.cpp: power_series, surrogate.lagfree_mixing, surrogate.lagfree
void add_connectome_steps(std::vector<StepSpec>& r);   // connectome_steps.cpp: fiber_ends, nearest_vertex, end_gate, registration_match, carry_ends, smooth_memberships, partition_counts, end_connectome, connection_weights, prune_below, largest_component
void add_tile_steps(std::vector<StepSpec>& r);          // tiles_steps.cpp: tile_grid, group_tile_grid
void add_eigen_steps(std::vector<StepSpec>& r);         // eigen_steps.cpp: hodge_dirac
void add_montage_steps(std::vector<StepSpec>& r);       // montage_steps.cpp: eeg_montage_1005
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
// THE SPATIAL SCALES of apply_kernel (`spatial_scale`: one, `spatial_scales`: a list) as the ladder's levels: each a QUANTITY (a tile diameter, an area, a wavelength, parts per hemisphere), resolved on the
// ladder's MEASURED level diameters by the one resolver (select_nodes.h level_of_quantity). A bare level index is never a parameter (`index:N` only, for diagnostics). `snapped` is what each level is.
std::vector<int> scale_levels(const Value& tiles, const Json& p, const Bound& bound, const std::string& who, Json* snapped = nullptr, Json* dictionary = nullptr) {
  Json list = Json::array();
  need(p["spatial_scale"].is_null() || p["spatial_scales"].is_null(), who + ": `spatial_scale` (one hat) or `spatial_scales` (a list of hats), not both");
  if (!p["spatial_scales"].is_null()) { need(p["spatial_scales"].is_array() && !p["spatial_scales"].empty(), who + ": spatial_scales is a list of quantities, e.g. [\"40 mm\", \"20 mm\"]"); list = p["spatial_scales"]; }
  else list.push_back(p["spatial_scale"]);
  std::string unit;
  const nxr::selection::Selection grid = ladder_grid(bound, tiles, &unit);   // the grid's levels with their measured diameters and bands, from the ladder's level records
  if (dictionary) *dictionary = ladder_dictionary(grid);                    // recorded in the field's facts: a later node resolves its quantity without the ladder (at_level)
  std::vector<int> out;
  Json snaps = Json::array();
  for (const auto& q : list) {
    Json sn = Json::object();
    const int l = level_of_quantity(grid, q, unit, who, sn);
    need(l >= 0 && l < static_cast<int>(grid.members.size()), who + ": " + q.dump() + " is level " + std::to_string(l) + "; the tile ladder has levels 0 .. " + std::to_string(static_cast<int>(grid.members.size()) - 1));
    need(std::find(out.begin(), out.end(), l) == out.end(), who + ": " + q.dump() + " repeats a level (" + std::to_string(l) + ")");
    out.push_back(l);
    snaps.push_back(Json{{"stated", q}, {"level", l}, {"snapped", sn}});
  }
  if (snapped) *snapped = snaps;
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
// A BAND SET as [lo, hi) Hz pairs in member order: a stored selection of spans on the Frequency line (read through the binding), or one made earlier in the run (`select.bands`: its set in the facts)
std::vector<std::pair<double, double>> band_set_of(const Value& b, const Bound& bound, const std::string& who) {
  std::vector<std::pair<double, double>> out;
  if (b.row) { for (const auto& sp : bound.spans(nxr::io::text(*b.row, "id"))) out.push_back({sp.first, sp.second}); }
  else if (b.facts.contains("set") && b.facts["set"].contains("members")) { for (const auto& m : b.facts["set"]["members"]) out.push_back({m["lo"].get<double>(), m["hi"].get<double>()}); }
  need(!out.empty(), who + ": `bands` is a band set (a selection of spans on the Frequency line, `select.bands`) with at least one member");
  for (const auto& e : out) need(e.first >= 0 && e.first < e.second, who + ": a band [" + std::to_string(e.first) + ", " + std::to_string(e.second) + ") needs 0 <= lo < hi");
  return out;
}
// the recording's rows an analytic band set keeps: an operator's from_selection (the kernel's channels), a stored cross-spectrum's channels, or every row
std::vector<std::int32_t> channels_for(const Values& ins, ExecContext& c) {
  std::vector<std::int32_t> ch;
  if (ins.count("channels_of")) {
    const Value& o = ins.at("channels_of");
    need(o.row.has_value(), "analytic: channels_of is not stored");
    if (o.cls == "statistic") ch = c.bound.cross_spectrum(nxr::io::text(*o.row, "name")).channels;
    else if (!nxr::io::is_null(*o.row, "from_selection_id")) {
      const auto sel = c.bound.selection(nxr::io::text(*o.row, "from_selection_id"));
      need(sel.has_value() && !nxr::io::is_null(*sel, "array_path"), "analytic: channels_of " + nxr::io::text(*o.row, "name") + ": its channels (from_selection) are not stored");
      ch = c.bound.array_int32(nxr::io::text(*sel, "array_path"));
    }
    need(!ch.empty(), "analytic: channels_of " + nxr::io::text(*o.row, "name") + " names no channel");
  } else {
    const auto m = c.bound.manifold(ins.at("x").manifolds[0]);
    const std::int64_t n = m ? nxr::io::integer(*m, "n_vertices") : 0;
    for (std::int64_t i = 0; i < n; ++i) ch.push_back(static_cast<std::int32_t>(i));
  }
  return ch;
}
struct BandAnalytic { std::vector<std::int32_t> channels; double sfreq = 0; nxr::pyramid::Shape shape = nxr::pyramid::Shape::Morlet; nxr::pyramid::Bands bank; std::int64_t halo = 0; };
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
  r.push_back({"analytic", "chunk", {"x", "bands", "channels_of"}, {"bands", "channels_of"}, {"y"}, Json{{"band", nullptr}, {"shape", "morlet"}, {"half_width", 1.0 / 3.0}, {"halo_seconds", 8.0}},
               "Studio engine/gpu/deviceChain.ts THE ANALYTIC SIGNAL (the band's even taps and its odd taps, hilbertImag, on the same raw rows); compute nxr::taps (gainTaps, convolveRows)",
               "the analytic signal of a band of a real field on … × time (the Signal chain's filter with the transform 'complex'): the band's even taps give the real part, "
               "its odd (Hilbert) taps the imaginary, on the same raw rows; band null: the plain Hilbert, cut at 1024 taps (its lowest frequencies lose their quadrature)",
               [](const Values& ins, const Json& p, const PlanContext& c) {
                 const Value& x = ins.at("x");
                 in_loop(x, "analytic");
                 need(!x.complex, "analytic: x is already complex");
                 if (ins.count("bands")) {
                   // A BAND SET (rule 23: ONE transform a block, shared by all bands, then the bands selected): the field `channels x members(bands) x T`, complex, band-major rows (row = band x channels + channel)
                   need(p["band"].is_null(), "analytic: `band` (one band) or `bands` (a band set), not both");
                   const std::vector<std::pair<double, double>> bank = band_set_of(ins.at("bands"), c.bound, "analytic");
                   need(p["shape"] == "morlet" || p["shape"] == "power_partition", "analytic: bands: shape " + p["shape"].dump() + ": morlet (a Gaussian a band, neighbours overlap) or power_partition (nxr's bank: the band powers add to broadband)");
                   need(p["halo_seconds"].is_number() && p["halo_seconds"].get<double>() >= 0, "analytic: halo_seconds is 0 or more (the filter is circular over the block read: its edge effect lies inside the halo)");
                   for (const auto& b : bank) need(b.second < x.rate / 2, "analytic: band [" + std::to_string(b.first) + ", " + std::to_string(b.second) + ") Hz reaches past Nyquist (" + std::to_string(x.rate / 2) + " Hz)");
                   try { nxr::pyramid::bandGains(Eigen::VectorXd::Ones(1), bank, nxr::pyramid::shapeOf(p["shape"].get<std::string>())); } catch (const std::invalid_argument& e) { refuse(std::string("analytic: ") + e.what()); }
                   Value y = made(x);
                   y.complex = true;
                   Json names = Json::array();
                   for (const auto& b : bank) names.push_back(Json::array({b.first, b.second}));
                   y.facts["members"] = Json{{"axis", "bands"}, {"count", static_cast<std::int64_t>(bank.size())}, {"layout", "band-major: row = band x channels + channel"}, {"bands_hz", names}};
                   y.facts["band_shape"] = p["shape"];
                   return Values{{"y", y}};
                 }
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
  r.push_back({"frames", "chunk", {"x", "tiling", "within"}, {"within"}, {"y"}, Json{{"half_width", 32}, {"band", nullptr}, {"frames_per_cycle", nullptr}, {"frames_per_tile", nullptr}, {"origin", nullptr}},
               "nxr vortex.band_at_frames (the band's signal at the frames, by FFT resampling: exact for a band-limited signal); here its local form, a Kaiser-windowed sinc (nxr::frametiling), at the START of each tower frame (nxr::tower)",
               "a field sampled on the FRAME TILING (a closed-form spans Selection of TOWER frames, `run frametiling`: the cycles of one tower level, D135, each a dyadic tile; a frame's time is its tile's start). A linear step: the band's signal x, real or complex "
               "(the analytic signal), is interpolated at each frame's (non-integer) sample position by a Kaiser-windowed sinc of `half_width` samples either side (the step's halo), exact for a signal band-limited well inside the Nyquist band. "
               "A frame belongs to the chunk holding the sample at or before it, so the result does not depend on the chunk size. The output's columns are FRAMES (axis `frames`): the steps after it are pointwise in time "
               "(a linear map, a form, |x|², a series kept) and fold frames; a step that reads the samples' own time refuses it. With `within` (a spans Selection of the recording's time line: the bursts, a stage) only the frames "
               "inside a span are GOOD (the steps after it leave the others out); without it every frame is. Beyond the recording the signal reads 0",
               [](const Values& ins, const Json& p, const PlanContext&) {
                 const Value &x = ins.at("x"), &t = ins.at("tiling");
                 in_loop(x, "frames");
                 need(!nonlinear(x), "frames: x was made by a non-linear step (" + x.facts.value("made_by", std::string("?")) + "): its band limit is gone, it cannot be interpolated");
                 need(t.cls == "selection", "frames: tiling must be the frame grid (`select.time_grid` rule frames)");
                 Json tp = Json::object();
                 if (t.row) tp = Json::parse(nxr::io::opt_text(*t.row, "params_json").value_or("{}"), nullptr, false);
                 else if (t.facts.contains("rule")) tp = Json{{"rule", t.facts["rule"]}};
                 need(t.row ? (tp.is_object() && tp.contains("grid") && tp["grid"].value("kind", std::string()) == "time" && tp["grid"].value("rule", std::string()) == "frames") : t.facts.value("rule", std::string()) == "frames",
                      "frames: " + (t.row ? nxr::io::text(*t.row, "name") : std::string("tiling")) + " is not a frame grid (`select.time_grid` rule frames)");
                 need(t.manifolds.size() == 1 && x.time && t.manifolds[0] == *x.time, "frames: the frame grid is of another time line than x's");
                 need(p["half_width"].is_number_integer() && p["half_width"].get<int>() >= 1, "frames: half_width is a whole number of samples, at least 1");
                 // `band`, `frames_per_cycle`, `frames_per_tile` and `origin` are NOT how the grid is made (rule 15: the frame grid is found by name, made by `select.time_grid` rule frames): a definition may state them
                 // to say what it expects, and the stored grid must be that (refused with the reason otherwise)
                 if (t.row && tp.contains("quantity") && tp["quantity"].contains("stated")) {
                   const Json& st = tp["quantity"]["stated"];
                   auto band_of_text = [](const Json& j) -> std::pair<double, double> {
                     if (j.is_array() && j.size() == 2) return {j[0].get<double>(), j[1].get<double>()};
                     const auto q = nxr::quantity::parse(j.get<std::string>());
                     return q.chart == "log10" ? std::pair<double, double>{std::pow(10.0, q.lo), std::pow(10.0, q.hi)} : std::pair<double, double>{q.lo, q.hi};
                   };
                   if (!p["band"].is_null() && st.contains("band") && !st["band"].is_null()) {
                     const auto a = band_of_text(p["band"]), b = band_of_text(st["band"]);
                     need(std::abs(a.first - b.first) < 1e-9 * std::max(1.0, a.first) && std::abs(a.second - b.second) < 1e-9 * std::max(1.0, a.second), "frames: the run's band " + p["band"].dump() + " is not the frame grid's (" + st["band"].dump() + "): `select.time_grid` rule frames first, with this band");
                   }
                   const bool grid_tile = st.contains("frames_per_tile") && st["frames_per_tile"].is_number(), run_tile = p["frames_per_tile"].is_number();
                   need(grid_tile == run_tile, std::string("frames: the run states frames ") + (run_tile ? "per tile" : "per cycle") + ", the frame grid was made " + (grid_tile ? "per tile" : "per cycle"));
                   const char* key = run_tile ? "frames_per_tile" : "frames_per_cycle";
                   need(p[key].is_null() || !st.contains(key) || st[key].is_null() || std::abs(p[key].get<double>() - st[key].get<double>()) < 1e-9, std::string("frames: the run states ") + key + " " + p[key].dump() + ", the frame grid has " + st[key].dump());
                 }
                 if (ins.count("within")) {
                   const Value& w = ins.at("within");
                   need(w.cls == "selection" && w.row && nxr::io::text(*w.row, "type") == "spans" && w.manifolds.size() == 1 && w.manifolds[0] == *x.time, "frames: `within` must be a spans Selection of x's time line");
                 }
                 Value y = made(x);
                 y.facts["axis"] = "frames";
                 const Json gd = t.row ? tp["grid"] : Json::object();
                 const double hop_s = t.row ? (gd.contains("hop_exact") ? gd["hop_exact"].get<double>() : static_cast<double>(gd["hop"].get<std::int64_t>())) / x.rate : t.facts.value("hop_s", 0.0);   // frames per cycle are not whole samples (exact, rule 36)
                 const double origin_s = t.row ? x.origin + static_cast<double>(gd["origin"].get<std::int64_t>()) / x.rate : t.facts.value("origin_s", x.origin);
                 need(hop_s > 0, "frames: the frame grid has no hop");
                 y.facts["frames"] = Json{{"tiling", t.row ? Json(nxr::io::text(*t.row, "name")) : Json(t.facts.value("node", std::string("select.time_grid")))}, {"level", t.facts.value("level", t.row ? gd.value("level", -1) : -1)}, {"half_width", p["half_width"]},
                                          {"within", ins.count("within") ? Json(nxr::io::text(*ins.at("within").row, "name")) : Json(nullptr)}};
                 y.rate = 1.0 / hop_s;
                 y.origin = origin_s;
                 return Values{{"y", y}};
               }});
  r.push_back({"apply_kernel", "chunk", {"x", "kernel", "tiles", "pre_operator", "glued"}, {"tiles", "pre_operator", "glued"}, {"y"}, Json{{"spatial_scale", nullptr}, {"spatial_scales", nullptr}, {"transform", "auto"}, {"modes", 0}, {"order", 0}, {"tolerance", 0.01}, {"envelope", "temporal"}, {"spatial_shape", "mexhat"}, {"form", "none"}},
               "Studio adapters/nxr kernelOf (the kernel padded to every member, its from_selection's columns); nxr.io.adapter.kernel_from_rows; with `tiles`, the level's Weyl band folded into the kernel (nxr::filterbank, pipelines/spatial_band.h)",
               "a linear operator from the field's first factor applied to it: x [members × time] → K x [target × time] (an imaging kernel: sensors → sources). `levels` (; instead of `level`) is ONE operator a hat level, all bound in preparation: the output is [vertex × level] (the rows of a vertex's levels consecutive, then a vertex's components: row = (vertex · levels + level) · components + component; `facts.levels` names them), the scale-space the spatial events' scale selection reads. `glued` (default false; a `form` is always glued) evaluates the band on EVERY vertex of the glued surface for a field with no form too (a free current: each component through the band, the bridge filled). `form` folds a LINEAR form of the free current into the same operator, in preparation: kernel (x y z a vertex) → its tangent part → curl_tan (or div_tan; `normal`: n·x) → the spatial band, ONE linear map before any non-linear step (D106); the output is ONE scalar a vertex (complex when x is: the analytic signal's curl; nxr vortex.Cortex.readings: div_curl(tangent(q)), then the hat of vortex_cores.hat_operator). With `tiles` the output is on EVERY vertex of the GLUED surface. With `tiles` (the target surface's "
               "tile ladder) and `level`: M x with M = (F_l ⊗ I) K, F_l the level's Weyl band on each component of the kernel's rows, bound ONCE in preparation (a linear map reads the chain "
               "before any non-linear step, D106; no graph filtering per sample). `envelope` temporal (the default: M x) | spatiotemporal. `spatial_shape` mexhat | meyer (the level IS its Weyl band: compact, the levels add up). `transform` auto, eigen, chebyshev (order 0: chosen from `tolerance`) or exact (dense modes of the mesh, ≤ 4096 vertices). "
               "`pre_operator` (optional): a linear operator on x's rows applied BEFORE the kernel (an array [M × M] over `meta.channels`, made by an earlier node: a surrogate's zero-lag mixing), COMPOSED once into the same operator (M · P, the type system's compose move; "
               "spec docs/specs/2026-11-17-surrogates.md, section 5); x then awaits it (`awaits_operator` names what the operator is of) and the mark is cleared. The bound operator is cached across runs by its inputs' provenance",
               [](const Values& ins, const Json& p, const PlanContext& c) {
                 const Value &x = ins.at("x"), &k = ins.at("kernel");
                 need(k.cls == "operator", "apply_kernel: kernel must be an operator");
                 in_loop(x, "apply_kernel", true);
                 need(!nonlinear(x), "apply_kernel: x was made by a non-linear step (" + x.facts.value("made_by", std::string("?")) + "); a linear map reads the chain before any non-linear transform (D106)");
                 need(x.manifolds[0] == k.manifolds[0], "apply_kernel: the kernel maps from manifold " + k.manifolds[0] + ", x's first factor is " + x.manifolds[0] + ": not the same manifold");
                 need(!k.manifolds[1].empty(), "apply_kernel: the kernel has no target manifold");
                 // a PRE-OPERATOR (spec 2026-11-17-surrogates.md, section 5): x awaits it exactly when it is given, and both name what the operator is of
                 const bool awaits = x.facts.contains("awaits_operator");
                 if (ins.count("pre_operator")) {
                   const Value& po = ins.at("pre_operator");
                   need(po.cls == "operator" && po.facts.contains("operator_of"), "apply_kernel: pre_operator must be a linear operator made by an earlier node (an array over x's rows, e.g. surrogate.lagfree_mixing's)");
                   need(awaits, "apply_kernel: pre_operator " + po.facts["operator_of"].dump() + " is given, but x awaits no operator: it would be applied twice (or to a value it is not made for)");
                   need(x.facts["awaits_operator"].value("of", Json()) == po.facts["operator_of"], "apply_kernel: x awaits the operator of " + x.facts["awaits_operator"].value("of", Json()).dump() + "; pre_operator is of " + po.facts["operator_of"].dump());
                 } else {
                   need(!awaits, "apply_kernel: x awaits its linear operator (" + (awaits ? x.facts["awaits_operator"].dump() : std::string()) + ": an unmixed surrogate); give it as `pre_operator` (the mixing node's output)");
                 }
                 Value y = made(x);
                 if (ins.count("pre_operator")) { y.facts["pre_operator"] = ins.at("pre_operator").facts; y.facts.erase("awaits_operator"); }
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
                 need(p["spatial_scale"].is_null() || p["spatial_scales"].is_null(), "apply_kernel: `spatial_scale` (one hat) or `spatial_scales` (a list of hats), not both");
                 need(ins.count("tiles") == ((p["spatial_scale"].is_null() && p["spatial_scales"].is_null()) ? 0u : 1u), "apply_kernel: `tiles` (the target surface's tile grid) and `spatial_scale` (or `spatial_scales`) come together: the scale's band is the grid's");
                 if (ins.count("tiles")) {
                   const Value& t = ins.at("tiles");
                   need(t.cls == "selection" && t.row && nxr::io::text(*t.row, "type") == "ladder" && t.manifolds[0] == k.manifolds[1],
                        "apply_kernel: tiles must be the tile ladder (`tiles compute`) of the kernel's target surface");
                   need(manifold_type(c, k.manifolds[1]) == "surface", "apply_kernel: the kernel's target is not a surface");
                   const std::vector<int> lvs = scale_levels(t, p, c.bound, "apply_kernel");
                   need(p["spatial_scales"].is_null() || p["envelope"] == "temporal", "apply_kernel: several scales need the temporal envelope (the monogenic field is of one scale)");
                   const int l = lvs[0];
                   need(p["transform"] == "auto" || p["transform"] == "eigen" || p["transform"] == "chebyshev" || p["transform"] == "exact", "apply_kernel: transform " + p["transform"].dump() + ": auto, eigen, chebyshev or exact");
                   need(p["spatial_shape"] == "mexhat" || p["spatial_shape"] == "meyer", "apply_kernel: spatial_shape " + p["spatial_shape"].dump() + ": mexhat (the default: the level's Mexican hat) or meyer (the level's Weyl band)");
                   need(p["modes"].is_number_integer() && p["modes"].get<int>() >= 0, "apply_kernel: modes is at least 0 (0: the standard package size, or what the band needs)");
                   need(p["order"].is_number_integer() && p["order"].get<int>() >= 0 && p["tolerance"].is_number() && p["tolerance"].get<double>() > 0, "apply_kernel: order is at least 0 (0: chosen), tolerance above 0");
                   if (ins.count("glued")) {
                     // THE GLUED SURFACE is the caller's choice, by name (rule 33): the derived manifold `glue` made from the kernel's target; the field is on it, restricted back to the original for storage. A flag that only
                     // chooses the input is never a parameter (rule 28).
                     const Value& g = ins.at("glued");
                     const std::string gid = g.row ? nxr::io::text(*g.row, "id") : (g.manifolds.empty() ? std::string() : g.manifolds[0]);
                     need(!gid.empty() && manifold_type(c, gid) == "surface", "apply_kernel: glued must be a surface (the glued cortex made by `glue`)");
                     const auto want = c.bound.glued_surface_id(k.manifolds[1]);
                     need(want.has_value() && *want == gid, "apply_kernel: glued is not the glued surface of the kernel's target (`glue` the target first, and name its result)");
                     y.manifolds[0] = gid;
                     y.facts["glued"] = true;
                   }
                   Json snapped, dictionary;
                   scale_levels(t, p, c.bound, "apply_kernel", &snapped, &dictionary);   // the stated scales, their levels and the snapped quantities, in the run's record
                   const std::string lunit = nxr::io::opt_text(*t.row, "unit").value_or(std::string()).empty() ? std::string("m") : nxr::io::text(*t.row, "unit");
                   y.facts["spatial_band"] = Json{{"level", p["spatial_scales"].is_null() ? Json(l) : Json(nullptr)}, {"ladder", nxr::io::text(*t.row, "id")}, {"transform", p["transform"]}, {"spatial_shape", p["spatial_shape"]}, {"scales", snapped},
                                                  {"ladder_levels", dictionary}, {"ladder_unit", lunit}};
                   if (!p["spatial_scales"].is_null()) { y.facts["spatial_band"]["levels"] = lvs; y.facts["levels"] = lvs; }
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
  r.push_back({"summarize", "after", {"x"}, {}, {"y"}, Json{{"summaries", Json::array({"median", "mad"})}, {"combine", "each"}, {"ddof", 1}},
               "a statistic across a partition's members; the summaries of compute nxr::stats (numpy median and linear percentile, Rheome's robust SD, MATLAB's sd, kurtosis, skewness)",
               "a measurement of the run summarised over its partition's members into a STATISTIC (field -> selection/measurement -> statistic): per row (each), pooled, or of the mean over rows",
               [](const Values& ins, const Json& p, const PlanContext&) {
                 const Value& x = ins.at("x");
                 need(x.cls == "measurement", "summarize: x must be a measurement over a selection's members (`reduce`, or a stored measurement found by name); it is never of a field");
                 need(!x.facts.contains("over") || !x.facts["over"].value("irregular", false) || x.facts.value("measure", std::string()) != "", "summarize: a measurement over an irregular selection names its measure");
                 need(x.facts.value("measure", std::string()) != "phase_current", "summarize: x is a vector measurement (the phase current: x y z); the statistics across the bursts are not built (a null is the maintainers' choice)");
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
  add_extent_steps(r);
  add_cores_steps(r);
  add_tracks_steps(r);
  add_crossings_steps(r);
  add_spatial_events_steps(r);
  add_surrogate_steps(r);
  add_connectome_steps(r);
  add_montage_steps(r);
  add_tile_steps(r);
  add_eigen_steps(r);
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
// the spatial band's choice for apply_kernel's request (eigenmodes stored or computable, else the series): null without `tiles`; one choice a level with `levels`
Json spatial_band_choice(const Values& ins, const Json& p, const Bound& bound) {
  if (!ins.count("tiles")) return nullptr;
  const auto man = bound.manifold(ins.at("kernel").manifolds[1]);
  const nxr::pipelines::SurfaceInput S = bound.surface(nxr::io::text(*man, "name"), true);
  nxr::pipelines::SpatialBandOptions bo;
  bo.transform = p["transform"].get<std::string>(); bo.modes = p["modes"].get<int>(); bo.order = p["order"].get<int>(); bo.tolerance = p["tolerance"].get<double>();
  bo.spatial_shape = p["spatial_shape"].get<std::string>();
  bo.glued = p["form"] != "none" || ins.count("glued") != 0;                                                             // a form is taken on the glued surface (the glued surface is a manifold of its own, named by the kernel's target)
  const std::vector<int> lvs = scale_levels(ins.at("tiles"), p, bound, "apply_kernel");
  if (p["spatial_scales"].is_null()) return bound.plan_spatial_band(S, bound.surface_level(S, lvs[0]), bo).key;
  Json keys = Json::array();                                                                  // one choice a level: each level's transform is in the run's key
  for (const int l : lvs) keys.push_back(bound.plan_spatial_band(S, bound.surface_level(S, l), bo).key);
  return keys;
}
// THE BOUND OPERATOR is STORED THROUGH IO with its provenance, found by its key (rule 32; spec section 8): a kernel with its form and spatial band, and with a pre-operator composed, under the key of its inputs'
// provenance (the rows' ids, which are unique, the request and the band's resolved choice). A constant of its inputs, so a hit is the same matrix. The node only LOOKS UP (`Bound::stored_operator`, the runner's
// side: B4) and records the key in its resolved options (`operator_key`) so the runner stores a made operator under it; a static cache in this process is gone.
std::optional<std::any> cached_operator(const Bound& bound, const std::string& key) {
  const auto stored = bound.stored_operator(Json(key));            // io: the operator found by the key its producer record carries (matrix and provenance)
  if (!stored) return std::nullopt;
  Json resolved = stored->provenance.is_object() && stored->provenance.contains("resolved") ? stored->provenance["resolved"] : (stored->provenance.is_object() ? stored->provenance : Json::object());
  resolved["operator_key"] = key;
  return std::any(KernelState{std::make_shared<const Eigen::MatrixXd>(stored->matrix), resolved});
}
// THE FUSED OPERATOR: M' = M . P, P a pre-operator [m x m] over the channels `meta.channels` (in order) of x's rows: column ch_j of M' = sum_i column ch_i of M times P(i, j); the other columns 0 (the rows of
// x outside P's channels carry nothing). ONE linear operator, built once (spec section 5: the mixing commutes with the time steps before it, so M' x_unmixed = M x_mixed).
std::any compose_pre_operator(const KernelState& base, const nxr::graph::Array& P, const Json& provenance) {
  const std::vector<std::int32_t> ch = P.meta.value("channels", std::vector<std::int32_t>());
  const Eigen::Index m = static_cast<Eigen::Index>(ch.size());
  if (m == 0 || P.rows != m || P.cols != m || static_cast<Eigen::Index>(P.data.size()) != m * m) refuse("apply_kernel: pre_operator is not [m x m] over its " + std::to_string(m) + " channels");
  const Eigen::MatrixXd& M = *base.K;
  for (const std::int32_t c : ch) if (c < 0 || c >= M.cols()) refuse("apply_kernel: pre_operator acts on channel " + std::to_string(c) + ", outside x's " + std::to_string(M.cols()) + " rows");
  const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Pm(P.data.data(), m, m);
  Eigen::MatrixXd Msub(M.rows(), m);
  for (Eigen::Index i = 0; i < m; ++i) Msub.col(i) = M.col(ch[static_cast<std::size_t>(i)]);
  const Eigen::MatrixXd F = Msub * Pm;
  auto out = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Zero(M.rows(), M.cols()));
  for (Eigen::Index j = 0; j < m; ++j) out->col(ch[static_cast<std::size_t>(j)]) = F.col(j);
  Json resolved = base.resolved;
  resolved["pre_operator"] = Json{{"provenance", provenance}, {"channels", static_cast<std::int64_t>(m)},
                                  {"fused", "M . P: the pre-operator composed into the kernel, form and spatial band, one linear operator (spec docs/specs/2026-11-17-surrogates.md, section 4)"}};
  return std::any(KernelState{std::shared_ptr<const Eigen::MatrixXd>(out), resolved});
}
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
// every node declares its time behaviour and numeric precision (NodeTraits); the nodes of the other files declare theirs where they are registered
void bind_traits(std::vector<StepSpec>& r) {
  auto declare = [&](const char* n, NodeTraits t) { for (auto& s : r) if (s.name == n) { s.traits = std::move(t); return; } refuse(std::string("no step ") + n); };
  declare("band_filter", node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::follows_input, "a zero-phase FIR band filter; float32 taps, each output sample a float64 sum over the window, stored as float32", "the taps' reach: taps.size() - 1 samples each side (set by the band's width; a recording's end is reflected)"));
  declare("analytic", node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::follows_input, "the band's even and odd FIR passes (the analytic signal); float64 sums, float32 result", "the longer of the even and odd taps' reach: taps.size() - 1 samples each side"));
  declare("envelope", node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::follows_input, "the analytic signal's modulus: the same FIR passes, then |z| in double, stored as float32", "the longer of the even and odd taps' reach: taps.size() - 1 samples each side"));
  declare("frames", node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::follows_input, "samples the band's signal at the frame tiling by a Kaiser-windowed sinc; float64 weights and sum, float32 result", "half_width samples each side of a frame's position (the sinc's reach; default 32)"));
  declare("apply_kernel", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "a linear map of the rows (kernel, spatial band) applied to each column independently; the product is formed in double and stored as float32", {}));
  declare("magnitude", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "the norm over an element's components, per sample; sums in double, float32 result", {}));
  declare("collect", node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64, "appends each block's own columns in time order: the order of the blocks fixes the result", {}));
  declare("risefall", node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::float64, "fits each event on the series over [start - margin, stop + margin]; events are independent of each other", "the event's span widened by `margin` (seconds, or cycles of the band) on each side"));
  declare("form", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "a differential form of the field (curl, divergence, normal part) per sample: the surface operator applied to each column; double sums, float32 result", {}));
  declare("power", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "|z|^2 summed over an element's components, per sample; double sums, float32 result", {}));
  declare("phase_current", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "Im(conj(z) grad z) per sample on the surface; double, float32 result", {}));
  declare("gradient_power", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "|grad z|^2 per sample on the surface; double, float32 result", {}));
  declare("summarize", node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64, "a statistic over all the members of a partition, computed once on the finished series; it is not defined per timestep and must see every member: stateful, confirmed", {}));
  declare("fit_gate", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "keeps or drops each event on its own fit; events do not interact (the unit is the event, not a timestep): pointwise over events, confirmed", {}));
  declare("burst_stages", node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "each event's stage spans from its own fit; events do not interact (the unit is the event, not a timestep): pointwise over events, confirmed", {}));
  declare("detect_spans", node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64, "a hysteresis scan along each series: the in-span state is carried from tile to tile, then gaps are merged and short spans dropped", {}));
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
  auto analytic_prepare = [](const Values& ins, const Json& p, ExecContext& c) -> std::any {
    if (ins.count("bands")) {
      BandAnalytic a;
      a.sfreq = ins.at("x").rate;
      a.bank = band_set_of(ins.at("bands"), c.bound, "analytic");
      a.shape = nxr::pyramid::shapeOf(p["shape"].get<std::string>());
      a.channels = channels_for(ins, c);
      a.halo = static_cast<std::int64_t>(std::ceil(p["halo_seconds"].get<double>() * a.sfreq));
      return std::any(std::move(a));
    }
    const auto g = band_gain(p);
    return std::any(AnalyticTaps{nxr::taps::gainTaps(g, ins.at("x").rate, false), nxr::taps::gainTaps(g, ins.at("x").rate, true)});
  };
  auto analytic_halo = [](const std::any& st) -> std::int64_t {
    if (st.type() == typeid(BandAnalytic)) return std::any_cast<const BandAnalytic&>(st).halo;
    const auto& t = std::any_cast<const AnalyticTaps&>(st);
    return static_cast<std::int64_t>(std::max(t.even.size(), t.odd.size())) - 1;
  };
  for (const char* n : {"analytic", "envelope"}) {
    StepSpec& s = at(n);
    const bool env = std::string(n) == "envelope";
    s.prepare = analytic_prepare;
    s.halo = analytic_halo;
    s.chunk = [env](const std::any& st, const Blocks& ins, std::int64_t keep) {
      const Block& b = ins.at("x");
      if (st.type() == typeid(BandAnalytic)) {
        // ONE spectrum of the block (the chunk with its halo; circular, the halo holds the edge effect) shared by every band: each band's gain on it, positive frequencies doubled, back (rule 23)
        const BandAnalytic& a = std::any_cast<const BandAnalytic&>(st);
        const std::int64_t M = static_cast<std::int64_t>(a.channels.size()), NB = static_cast<std::int64_t>(a.bank.size());
        Eigen::MatrixXd block(b.width, M);                                   // [samples x channels]
        for (std::int64_t i = 0; i < M; ++i) {
          const std::int32_t ch = a.channels[static_cast<std::size_t>(i)];
          if (ch < 0 || ch >= b.rows) refuse("analytic: a channel (" + std::to_string(ch) + ") is outside the recording's " + std::to_string(b.rows) + " rows");
          for (std::int64_t t = 0; t < b.width; ++t) block(t, i) = b.re[static_cast<std::size_t>(ch * b.width + t)];
        }
        const nxr::pyramid::Spectrum spec = nxr::pyramid::spectrumOf(block, a.sfreq);
        const Eigen::MatrixXd gains = nxr::pyramid::bandGains(spec.freq, a.bank, a.shape);   // [bands x bins]
        const std::int64_t kl = std::min(keep, b.left), kr = std::min(keep, b.right), w = b.own() + kl + kr, c0 = b.left - kl;
        Block y{NB * M, w, kl, kr, std::vector<float>(static_cast<std::size_t>(NB * M * w)), std::vector<float>(static_cast<std::size_t>(NB * M * w))};
        for (std::int64_t bi = 0; bi < NB; ++bi) {
          const Eigen::MatrixXcd A = nxr::pyramid::analyticBand(spec, gains.row(static_cast<Eigen::Index>(bi)).transpose());   // [samples x channels]
          for (std::int64_t i = 0; i < M; ++i)
            for (std::int64_t t = 0; t < w; ++t) {
              const std::complex<double> v = A(c0 + t, i);
              y.re[static_cast<std::size_t>((bi * M + i) * w + t)] = static_cast<float>(v.real());
              y.im[static_cast<std::size_t>((bi * M + i) * w + t)] = static_cast<float>(v.imag());
            }
        }
        return Blocks{{"y", std::move(y)}};
      }
      const auto& t = std::any_cast<const AnalyticTaps&>(st);
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

  // the analytic signal's halo from its request alone (a band set reads stored channels in preparation: the plan must not prepare it): the closed form of the circular block's context, or the taps' reach
  at("analytic").halo_at_plan = [](const Values& ins, const Json& p) -> std::int64_t {
    if (ins.count("bands")) return static_cast<std::int64_t>(std::ceil(p["halo_seconds"].get<double>() * ins.at("x").rate));
    const auto g = band_gain(p);
    return static_cast<std::int64_t>(std::max(nxr::taps::gainTaps(g, ins.at("x").rate, false).size(), nxr::taps::gainTaps(g, ins.at("x").rate, true).size())) - 1;
  };
  // the frame tiling: the chunk's analytic field sampled at the frames inside its own samples
  // THE FRAME GRID is the time grid the maker `select.time_grid` (rule frames) made: frame j starts at sample origin + j hop (a closed-form Rule on the line, no tower, no day anchor); its time is the frame's START
  struct FramesState { std::int64_t origin = 0, count = 0; double hop = 1, rate = 0, origin_s = 0; int H = 32; std::vector<std::pair<double, double>> within; bool has_within = false;
    double position(std::int64_t j) const { return static_cast<double>(origin) + static_cast<double>(j) * hop; }          // the frame's start, in samples from the line's first
    std::pair<std::int64_t, std::int64_t> span(std::int64_t first, std::int64_t n) const {            // the frames whose start lies in [first, first + n): a frame belongs to the chunk holding it, so the result does not depend on the chunk size
      auto up = [&](std::int64_t sample) { return std::min(count, std::max<std::int64_t>(0, static_cast<std::int64_t>(std::ceil((static_cast<double>(sample - origin)) / hop - 1e-9)))); };   // first frame at or after the sample (hop in samples, exact)
      return {up(first), up(first + n)};
    } };
  fr.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
    const Value& x = ins.at("x");
    const Json tp = ins.at("tiling").row ? Json::parse(nxr::io::opt_text(*ins.at("tiling").row, "params_json").value_or("{}"), nullptr, false) : (c.made && c.made->count("tiling") ? c.made->at("tiling").meta.value("describe", Json::object()).value("params", Json::object()) : Json::object());
    need(tp.is_object() && tp.contains("grid") && tp["grid"].value("kind", std::string()) == "time", "frames: the frame grid records no time grid (`select.time_grid` rule frames)");
    const Json& gd = tp["grid"];
    FramesState s;
    s.origin = gd["origin"].get<std::int64_t>(); s.hop = gd.contains("hop_exact") ? gd["hop_exact"].get<double>() : static_cast<double>(gd["hop"].get<std::int64_t>()); s.count = gd["count"].get<std::int64_t>(); s.rate = x.rate;
    s.origin_s = x.origin;
    if (std::fabs(gd["rate"].get<double>() - x.rate) > 1e-9 * x.rate) refuse("frames: the frame grid was made for " + gd["rate"].dump() + " samples a second, x has " + std::to_string(x.rate));
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
    if (keep != 0) refuse("frames: a step after it needs a halo; the steps after the frame grid are pointwise in time");
    const Block& b = ins.at("x");
    const auto [j0, j1] = st.span(at.first, at.n);
    std::vector<double> pos;
    for (std::int64_t j = j0; j < j1; ++j) pos.push_back(st.position(j) - static_cast<double>(at.first - b.left));          // from the block's column 0
    Block y{b.rows, j1 - j0, 0, 0, nxr::frametiling::sincInterpolate(b.re.data(), b.rows, b.width, pos, st.H), {}};
    if (b.complex()) y.im = nxr::frametiling::sincInterpolate(b.im.data(), b.rows, b.width, pos, st.H);
    return Blocks{{"y", std::move(y)}};
  };
  fr.frame_place = [](const std::any& st_, const Place& at) {
    const FramesState& st = *std::any_cast<const std::shared_ptr<FramesState>&>(st_);
    const auto [j0, j1] = st.span(at.first, at.n);
    FramePlace f{j0, j1 - j0, {}};
    const bool bad = at.good && !at.good->empty();
    if (bad || st.has_within) {
      f.good.assign(static_cast<std::size_t>(j1 - j0), 1);
      for (std::int64_t j = j0; j < j1; ++j) {
        const double ps = st.position(j);
        if (bad) {
          const std::int64_t u = std::min<std::int64_t>(at.n - 1, std::max<std::int64_t>(0, static_cast<std::int64_t>(std::floor(ps)) - at.first));      // the sample at or before the frame
          if (!(*at.good)[static_cast<std::size_t>(u)]) f.good[static_cast<std::size_t>(j - j0)] = 0;
        }
        if (st.has_within) {
          const double t = st.origin_s + ps / st.rate;
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
    // the bound map is a constant of (kernel, the field's first factor, level, transform request): every step that asks for the same one (the same level read by several time bands) shares the first one's.
    // ACROSS RUNS (spec 2026-11-17-surrogates.md, section 5) it is kept in the process's operator cache under the same key plus the spatial band's resolved choice: the draws of a surrogate comparison share
    // every input of it, so it is built once.
    const std::string base_key = "apply_kernel|" + nxr::io::text(k, "id") + "|" + ins.at("x").manifolds[0] + "|" + (ins.count("tiles") ? p.dump() : std::string("none")) + "|" + p["form"].get<std::string>() + (ins.count("glued") ? "|glued" : "");
    const std::string band_choice = spatial_band_choice(ins, p, c.bound).dump();
    std::optional<Json> pre_provenance;
    const nxr::graph::Array* pre = nullptr;
    if (ins.count("pre_operator")) {
      if (!c.made || !c.made->count("pre_operator")) refuse("apply_kernel: pre_operator was not made before the loop (an earlier node's array)");
      pre = &c.made->at("pre_operator");
      pre_provenance = pre->meta.value("provenance", Json());
      if (pre_provenance->is_null()) refuse("apply_kernel: pre_operator records no provenance (what it is made from): it cannot be cached or recorded");
    }
    const std::string shared_key = base_key + "|" + band_choice + (pre ? "|pre|" + pre_provenance->dump() : std::string());
    if (const auto it = c.shared.find(shared_key); it != c.shared.end()) return it->second;
    if (auto hit = cached_operator(c.bound, shared_key)) { c.shared[shared_key] = *hit; return *hit; }
    if (pre)
      if (auto base = cached_operator(c.bound, base_key + "|" + band_choice)) {
        std::any fused = compose_pre_operator(std::any_cast<const KernelState&>(*base), *pre, *pre_provenance);
        std::any_cast<KernelState&>(fused).resolved["operator_key"] = shared_key;
        c.shared[shared_key] = fused;
        return fused;
      }
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
      const std::vector<int> lvs = scale_levels(ins.at("tiles"), p, c.bound, "apply_kernel");
      const std::int64_t NL = static_cast<std::int64_t>(lvs.size());
      const bool form = p["form"] != "none";
      const bool glued = form || ins.count("glued") != 0;        // a form, or the glued surface named as the input, is evaluated on every glued vertex (the bridge too)
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
      if (NL > 1) resolved["levels"] = lvs;
      resolved["envelope"] = p["envelope"];
    }
    std::any bound = KernelState{std::shared_ptr<const Eigen::MatrixXd>(K), resolved};
    std::any_cast<KernelState&>(bound).resolved["operator_key"] = base_key + "|" + band_choice;
    if (pre) {
      bound = compose_pre_operator(std::any_cast<const KernelState&>(bound), *pre, *pre_provenance);
      std::any_cast<KernelState&>(bound).resolved["operator_key"] = shared_key;
    }
    c.shared[shared_key] = bound;
    {
      // rule 32: the operator built is OFFERED to the runner, which stores it through io under its key with its provenance (found next time by `stored_operator`); no process-wide cache
      const KernelState& ks = std::any_cast<const KernelState&>(bound);
      nxr::pipelines::graph::MadeOperator op;
      op.key = Json(shared_key);
      op.matrix = ks.K;
      op.provenance = Json{{"resolved", ks.resolved}, {"request", p}, {"kernel", nxr::io::text(k, "id")}};
      op.name = "fused " + nxr::io::text(k, "name") + (pre ? " (pre-operator composed)" : "");
      op.from_manifold_id = ins.at("x").manifolds[0];
      op.to_manifold_id = ins.at("kernel").manifolds[1];
      op.description = "the kernel with its form, spatial band and pre-operator folded into one linear operator";
      c.bound.offer_operator(std::move(op));
    }
    return bound;
  };
  ak.record = [](const std::any& st) { return std::any_cast<const KernelState&>(st).resolved; };
  // the spatial band's choice (eigenmodes stored or computable, else the series) enters the run's cache key, read before anything is computed or stored
  ak.resolve = [](const Values& ins, const Json& p, const Bound& bound) -> Json { return spatial_band_choice(ins, p, bound); };
  ak.chunk = [](const std::any& st, const Blocks& ins, std::int64_t keep) {
    const Eigen::MatrixXd& K = *std::any_cast<const KernelState&>(st).K;
    const Block b = crop(ins.at("x"), keep);
    using RM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    auto mul = [&](const std::vector<float>& x) {
      const Eigen::MatrixXd X = Eigen::Map<const RM>(x.data(), b.rows, b.width).cast<double>();
      const RM Y = (K * X).cast<float>();
      return std::vector<float>(Y.data(), Y.data() + Y.size());
    };
    if (b.rows == K.cols()) {
      Block y{K.rows(), b.width, b.left, b.right, mul(b.re), b.complex() ? mul(b.im) : std::vector<float>()};
      return Blocks{{"y", std::move(y)}};
    }
    // A BAND SET on x (`analytic` with `bands`: band-major rows, row = band x channels + channel): the SAME fused operator on each band's channels; the output is band-major too (row = band x K.rows + target row)
    if (K.cols() == 0 || b.rows % K.cols() != 0) refuse("apply_kernel: x has " + std::to_string(b.rows) + " rows, not a whole number of bands x the kernel's " + std::to_string(K.cols()) + " channels");
    const std::int64_t NB = b.rows / K.cols(), C = K.cols(), R = K.rows();
    Block y{R * NB, b.width, b.left, b.right, std::vector<float>(static_cast<std::size_t>(R * NB * b.width)), b.complex() ? std::vector<float>(static_cast<std::size_t>(R * NB * b.width)) : std::vector<float>()};
    auto mul_band = [&](const std::vector<float>& x, std::vector<float>& out, std::int64_t bi) {
      const Eigen::MatrixXd X = Eigen::Map<const RM>(x.data() + bi * C * b.width, C, b.width).cast<double>();
      const RM Y = (K * X).cast<float>();
      std::copy(Y.data(), Y.data() + Y.size(), out.begin() + bi * R * b.width);
    };
    for (std::int64_t bi = 0; bi < NB; ++bi) { mul_band(b.re, y.re, bi); if (b.complex()) mul_band(b.im, y.im, bi); }
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

  // a measurement of the run summarised (nxr::stats::summarizeRows): [rows' x summaries], the summaries' names and the members used in meta
  StepSpec& sc = at("summarize");
  sc.prepare = [](const Values&, const Json& p, ExecContext&) { return std::any(p); };   // its run is nxr::measurement::summarize (measurement_nodes.cpp)

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
    // the count row an event's measures share: the samples over the event's tiles (as measure_events', one row a Selection and recording)
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

namespace {
// a node of nxr::graph::registry() as the definition reader sees it: its ports, parameters and traits (the executor runs the node itself, `native`)
StepSpec describe_native(const nxr::graph::NodeSpec& n) {
  StepSpec s;
  s.name = n.name;
  s.native = true;
  s.kind = n.kind == nxr::graph::Kind::chunk ? "chunk" : n.kind == nxr::graph::Kind::accumulate ? "accumulate" : "array";
  for (const auto& p : n.inputs) { s.inputs.push_back(p.name); if (p.optional) s.optional.push_back(p.name); }
  for (const auto& p : n.outputs) s.outputs.push_back(p.name);
  for (const auto& p : n.parameters) s.parameters[p.name] = p.def;
  s.matches = n.matches;
  s.description = n.description;
  s.traits = n.traits;
  return s;
}
}  // namespace

const std::vector<StepSpec>& registry() {
  static const std::vector<StepSpec> r = [] {
    nxr::pipelines::graph::register_welch_nodes();
    nxr::pipelines::graph::register_meeg_nodes();
    nxr::pipelines::graph::register_partition_nodes();
    nxr::pipelines::graph::register_select_nodes();
    std::vector<StepSpec> all = build();
    append_verb_steps(all);
    append_geometry_steps(all);                                           // the geometry makers (steps_geometry.cpp)
    append_patches_steps(all);
    append_measure_steps(all);                                            // the measurement verbs (steps_measure.cpp)
    add_select_nodes(all);                                                // the five makers (select_nodes.cpp); takes the steps they replace out of `all`
    add_measurement_nodes(all);                                           // field.form, reduce, roll_up, summarize, estimate (measurement_nodes.cpp)
    for (const auto& name : nxr::graph::registry().names()) all.push_back(describe_native(nxr::graph::registry().at(name)));   // the nodes registered before the first use
    std::set<std::string> seen;                                             // one step a name: a second one would be found in place of the first
    for (const auto& st : all) if (!seen.insert(st.name).second) throw std::logic_error("registry: two steps are named '" + st.name + "'");
    return all;
  }();
  return r;
}


namespace {
// A RETIRED STEP NAME is refused with the name of what replaces it (spec 2026-11-17-consolidation.md, sections 1, 2 and 9.2): no aliases are kept.
const std::map<std::string, std::string>& retired_steps() {
  static const std::map<std::string, std::string> m = {
      {"tile_grid", "select.dyadic_surface (rule subject or group)"}, {"group_tile_grid", "select.dyadic_surface (rule transfer)"},
      {"frame_tiling", "select.time_grid (rule frames)"}, {"constantq", "select.time_grid (rule constant_q)"},
      {"pyramid.bands", "select.bands (rule frequency)"}, {"pyramid.spatial_bands", "select.bands (rule levels)"}, {"joint.cells", "select.product"},
      {"measure.band_amplitude", "reduce (preset sensor_band_amplitude of the definition measure)"},
      {"detect_spans", "select.events.spans (rule detect)"}, {"fit_gate", "select.events.spans (rule keep_where)"}, {"burst_stages", "select.events.spans (rule stages)"},
      {"cell_extent", "select.events.regions"}, {"extent_measure", "reduce.members (a merge of the events' cells)"}, {"patches", "select.events.patches"}, {"cores", "select.events.cores"}, {"tracks", "select.events.tracks"},
      {"measure", "reduce"}, {"partition_measures", "reduce"}, {"window_measures", "reduce"}, {"measure_events", "reduce.members"}, {"tile_max", "roll_up (along the space factor, merge max)"},
      {"reduce_cells", "roll_up (along the space factor, merge max)"}, {"summarize_measurements", "summarize"},
      {"magnitude", "field.form (form magnitude)"}, {"power", "field.form (form power)"}, {"phase_current", "field.form (form phase_current)"},
      {"pyramid.band_sums", "reduce over select.product (layout joint)"}, {"pyramid.frame_samples", "reduce over select.product (layout joint)"},
      {"core_histogram", "estimate (method histogram)"}, {"relative_histogram", "estimate (method histogram)"}, {"baseline", "estimate (method sample)"},
      {"power_series", "reduce (preset surrogate_lagfree_envelope of the definition measure; surrogate.lagfree reads it by name)"}};
  return m;
}
}  // namespace

const StepSpec& step(const std::string& name) {
  for (const auto& s : registry()) if (s.name == name) return s;
  if (const auto it = retired_steps().find(name); it != retired_steps().end()) refuse("the step '" + name + "' is retired: use " + it->second);
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
