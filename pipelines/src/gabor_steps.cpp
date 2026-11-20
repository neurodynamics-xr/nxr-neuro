// nxr-pipelines — the registered steps of THE GABOR BURSTS (spec docs/specs/2026-11-20-gabor-bursts.md): burst detection in time by matched Gabor atoms of free width.
//
//   signal.gabor             accumulate  |W| on the Gabor atoms (the time filter bank's Gabor family, one member an atom), EACH WIDTH AT ITS OWN RATE, one width at a time; a series per width, rows atom-major
//   select.events.peaks      after   the candidates: local maxima of |W| per channel and width, in (frequency, frame), above the rows' levels read from a statistic
//   select.gabor_fit         after   each candidate fitted on the samples around it (as select.risefall): t0, f, σ, FWHM, cycles, amplitude, phase, |W|, the width test, R² against the band's signal
//   select.events.suppress   after   non-maximum suppression of the events within each event's own time and frequency resolution (σ, 1/(2πσ)), per row
//   event_edges              after   (select.events.spans rule `edges`) each event's onset and offset: the tangent on its envelope (default) or the atom's half maximum, and the events' measures
//   event_union              after   (select.events.spans rule `union`) the global windows: the events of all rows joined where they overlap in time (and, by default, in frequency)
//
// Every step is a node of the graph: a pure function of matrices (compute/: nxr::signal::timefilterbank, nxr::signal gaborFit, nxr::select localMaxima, suppress, tangentEdges, overlapGroups) with typed
// ports. The atom's level (median and robust spread of |W| over time) is never computed here: it is a summary over each width's frames (`measurement.summarize`), handed in.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <tuple>

#include "nxr/pipelines/measurement_nodes.h"
#include "nxr/pipelines/registry.h"
#include "nxr/quantity/quantity.h"
#include "nxr/select/edges.h"
#include "nxr/select/events.h"
#include "nxr/select/peaks.h"
#include "nxr/signal/fourier.h"
#include "nxr/signal/gabor.h"
#include "nxr/signal/timefilterbank.h"
#include "nxr/signal/wavelet.h"

namespace nxr::pipelines::graph {

namespace {
namespace tb = nxr::signal::timefilterbank;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;
void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
Value made(const Value& x) { Value v = x; v.row.reset(); return v; }

// a band as [lo, hi] Hz: a pair, or a quantity in a frequency chart ("8-35 Hz")
std::pair<double, double> band_of(const Json& j, const std::string& who) {
  if (j.is_array() && j.size() == 2 && j[0].is_number() && j[1].is_number()) return {j[0].get<double>(), j[1].get<double>()};
  need(j.is_string(), who + ": band is [lo, hi] Hz or a quantity (\"8-35 Hz\"), not " + j.dump());
  nxr::quantity::Quantity q;
  try { q = nxr::quantity::parse(j.get<std::string>()); } catch (const std::invalid_argument& e) { refuse(who + ": band " + j.dump() + ": " + e.what()); }
  need(q.unit == "Hz", who + ": band " + j.dump() + " is not in Hz");
  return {q.lo, q.hi};
}

// THE ATOMS a request names (the Gabor grid, compute's gaborGrid), checked
struct GaborRequest {
  double fmin = 0, fmax = 0, sigma_min = 0, sigma_max = 0, min_f_sigma = 0, tolerance = 1e-6, budget = 1e9;
  int per_octave = 0;
  std::string decimation = "own";
  tb::GaborGrid grid;
};
GaborRequest gabor_request(const Json& p, const std::string& who) {
  GaborRequest r;
  need(!p["band"].is_null(), who + ": band is required (the frequencies searched, e.g. \"8-35 Hz\")");
  std::tie(r.fmin, r.fmax) = band_of(p["band"], who);
  for (const char* k : {"sigma_min", "sigma_max", "min_f_sigma", "tolerance", "budget"}) need(p[k].is_number(), std::string(who) + ": " + k + " is a number");
  need(p["per_octave"].is_number_integer() && p["per_octave"].get<int>() >= 1, who + ": per_octave is a whole number of widths an octave, at least 1");
  r.sigma_min = p["sigma_min"].get<double>(); r.sigma_max = p["sigma_max"].get<double>(); r.min_f_sigma = p["min_f_sigma"].get<double>(); r.per_octave = p["per_octave"].get<int>();
  r.tolerance = p["tolerance"].get<double>(); r.budget = p["budget"].get<double>();
  need(r.tolerance > 0 && r.tolerance < 1, who + ": tolerance is in (0, 1) (the share of an atom's response left outside the halo)");
  need(r.budget > 0, who + ": budget is the bytes one block may hold, above 0");
  need(p["decimation"] == "own" || p["decimation"] == "none", who + ": decimation " + p["decimation"].dump() + ": own (each width at its own rate, the default) or none (every width at the recording's rate: for checks)");
  r.decimation = p["decimation"].get<std::string>();
  try { r.grid = tb::gaborGrid(r.sigma_min, r.sigma_max, r.per_octave, r.fmin, r.fmax, r.min_f_sigma); } catch (const std::invalid_argument& e) { refuse(who + ": " + e.what()); }
  need(!r.grid.atoms.empty(), who + ": no atom: every width's lowest frequency (min_f_sigma / σ) is above the band");
  return r;
}
Json gabor_facts(const GaborRequest& r) {
  Json atoms = Json::array(), width_of = Json::array();
  for (std::size_t a = 0; a < r.grid.atoms.size(); ++a) { atoms.push_back(Json::array({r.grid.atoms[a].first, r.grid.atoms[a].second})); width_of.push_back(r.grid.widthOf[a]); }
  return Json{{"band", Json::array({r.fmin, r.fmax})}, {"sigma_min", r.sigma_min}, {"sigma_max", r.sigma_max}, {"per_octave", r.per_octave}, {"min_f_sigma", r.min_f_sigma},
              {"widths", r.grid.widths}, {"atoms", atoms}, {"width_of", width_of}, {"decimation", r.decimation}, {"normalisation", "unit energy (the matched filter): A = 2|W| / (pi^(1/4) sqrt(sigma))"}};
}
tb::GaborGrid grid_of_facts(const Json& g) {
  tb::GaborGrid out;
  out.widths = g["widths"].get<std::vector<double>>();
  for (const auto& a : g["atoms"]) out.atoms.push_back({a[0].get<double>(), a[1].get<double>()});
  out.widthOf = g["width_of"].get<std::vector<int>>();
  return out;
}
// THE HALO: the reach of the widest atom's response down to `tolerance` of its L1 norm (compute's timefilterbank::haloOf on a one-member bank of the widest atom), samples on each side
std::int64_t gabor_halo(const GaborRequest& r, double fs) {
  std::size_t widest = 0;
  for (std::size_t a = 0; a < r.grid.atoms.size(); ++a) if (r.grid.atoms[a].first > r.grid.atoms[widest].first) widest = a;
  tb::Design d(tb::Family::Gabor, 1.0, tb::Sides::Analytic);
  d.atoms = {r.grid.atoms[widest]};
  d.edgeBins = tb::EdgeBins::Drop;
  const std::size_t n = static_cast<std::size_t>(std::ceil(12 * r.grid.atoms[widest].first * fs)) + 16;
  return static_cast<std::int64_t>(tb::haloOf(tb::timeBank(d, tb::gridLength(n, tb::Padding::FastLen), fs), 0, r.tolerance));
}
// EACH WIDTH'S OWN RATE: the coarsest decimation (a power of two, so every width's frames lie on the finest's) that keeps at least 3 samples a σ; 1 with decimation none
std::int64_t decimation_of(double sigma, double fs, const std::string& mode) {
  if (mode == "none") return 1;
  std::int64_t D = 1;
  while (static_cast<double>(2 * D) <= sigma * fs / 3) D *= 2;
  return D;
}
struct GaborWidth {
  tb::Design design{tb::Family::Gabor, 1.0, tb::Sides::Analytic};
  double sigma = 0;
  std::int64_t D = 1, atom0 = 0, atoms = 0;
};
struct GaborState {
  std::vector<GaborWidth> widths;
  std::int64_t atoms = 0, halo = 0, Dmax = 1;
  double fs = 0, origin = 0, budget = 1e9;
};
GaborState gabor_state(const GaborRequest& q, double fs, double origin) {
  GaborState st;
  st.fs = fs; st.origin = origin; st.budget = q.budget;
  st.atoms = static_cast<std::int64_t>(q.grid.atoms.size());
  for (std::size_t a = 0; a < q.grid.atoms.size(); ++a) {
    if (a == 0 || q.grid.widthOf[a] != q.grid.widthOf[a - 1]) {
      GaborWidth w;
      w.design.edgeBins = tb::EdgeBins::Drop;                                 // the atoms' gain at DC is not taken (the reference's positive frequencies only)
      w.sigma = q.grid.atoms[a].first;
      w.D = decimation_of(w.sigma, fs, q.decimation);
      w.atom0 = static_cast<std::int64_t>(a);
      st.widths.push_back(w);
      st.Dmax = std::max(st.Dmax, w.D);
    }
    st.widths.back().design.atoms.push_back(q.grid.atoms[a]);
    ++st.widths.back().atoms;
  }
  st.halo = gabor_halo(q, fs);
  return st;
}
// THE GRID A BLOCK IS TRANSFORMED ON: `width` samples, aligned on the coarsest decimation and zero-padded to twice (no wrap), a multiple of every width's decimation
std::size_t block_grid(const GaborState& st, std::int64_t width) {
  return static_cast<std::size_t>(st.Dmax) * nxr::signal::next_fast_len(static_cast<std::size_t>((2 * (width + st.Dmax) + st.Dmax - 1) / st.Dmax), true);
}
// THE BYTES ONE BLOCK HOLDS (stated so a run is sized before it allocates): the padded block and its spectrum, then, one width at a time, one atom's decimated coefficients and the width's |W| on the block's own frames
double block_bytes(const GaborState& st, std::int64_t channels, std::int64_t width, std::int64_t own) {
  const double C = static_cast<double>(channels), N = static_cast<double>(block_grid(st, width));
  double per_width = 0;
  for (const auto& w : st.widths) per_width = std::max(per_width, C * (N / static_cast<double>(w.D)) * 16 * 2 + static_cast<double>(w.atoms) * C * std::ceil(static_cast<double>(own) / static_cast<double>(w.D)) * 4);
  return C * N * 8 + C * (N / 2 + 1) * 16 + per_width;
}
std::string gigabytes(double b) { char t[32]; std::snprintf(t, sizeof t, "%.3g GB", b / 1e9); return t; }
void size_check(const GaborState& st, std::int64_t channels, std::int64_t own, const std::string& who) {
  const double b = block_bytes(st, channels, own + 2 * st.halo, own);
  need(b <= st.budget, who + ": one block would hold " + gigabytes(b) + " (" + std::to_string(channels) + " channels, a block of " + std::to_string(own) + " samples and a halo of " + std::to_string(st.halo) +
                           " each side), over the budget of " + gigabytes(st.budget) + ": run fewer channels, a narrower band (fewer atoms a width), a smaller sigma_max (a shorter halo), or a shorter chunk; or raise `budget`");
}
// THE SERIES THE RUN HOLDS, bytes a second of span: Σ over widths of atoms x channels x (the rate / the width's decimation) x 4 (float32): each width's |W| over the span, kept for its level
double series_bytes_per_second(const GaborState& st, std::int64_t channels) {
  double b = 0;
  for (const auto& w : st.widths) b += static_cast<double>(w.atoms) * static_cast<double>(channels) * st.fs / static_cast<double>(w.D) * 4;
  return b;
}
// THE RUN'S SIZE CHECK, on the span the caller asked (known before the first block is folded, nothing accumulated yet): the block (the largest chunk's) and the series over the span, against the budget
void span_check(const GaborState& st, std::int64_t channels, std::int64_t span_samples, const std::string& who) {
  const double block = block_bytes(st, channels, (std::int64_t(1) << 15) + 2 * st.halo, std::int64_t(1) << 15), per_s = series_bytes_per_second(st, channels);
  const double span_s = static_cast<double>(span_samples) / st.fs, total = block + per_s * span_s;
  if (total <= st.budget) return;
  char t[64];
  std::snprintf(t, sizeof t, "%.1f s", std::max(0.0, (st.budget - block) / per_s));
  char u[64];
  std::snprintf(u, sizeof u, "%.1f s", span_s);
  refuse(who + ": the run would hold " + gigabytes(total) + ": one block " + gigabytes(block) + " and each width's |W| over the span, " + gigabytes(per_s) + " a second (" + std::to_string(channels) +
         " channels) x " + u + "; over the budget of " + gigabytes(st.budget) + ". The longest span that fits is " + t +
         ": run a shorter span (--from/--to), fewer channels, a narrower band (fewer atoms a width), a smaller sigma_max, or raise `budget`");
}
struct GaborAcc {
  std::vector<std::vector<std::vector<float>>> rows;   // per width, per row (atom x channels + channel): |W| at the width's frames, in order
  std::vector<std::int64_t> first;                     // per width: the global index of its first frame
  std::int64_t channels = -1;
};

// the column a measurement over an event set holds for row i of `ev` (a subset of that set keeps `measured_ordinals`); the same row when the measurement is over `ev` itself
std::int64_t column_of(const Array& ev, const Array& x, std::int64_t i) {
  if (x.cols == ev.rows || !ev.meta.contains("measured_ordinals")) return i;
  return ev.meta["measured_ordinals"][static_cast<std::size_t>(i)].get<std::int64_t>();
}
// a measurement's columns by measure name ([rows x members x comps], the components of a member consecutive)
std::map<std::string, std::int64_t> measure_columns(const Array& x) {
  std::map<std::string, std::int64_t> c;
  std::int64_t k = 0;
  for (const auto& m : x.meta.value("measures", Json::array())) { c[m.value("measure", std::string())] = k; k += m.value("components", 1); }
  return c;
}
double value_at(const Array& x, std::int64_t row, std::int64_t col, std::int64_t comp) {
  const std::int64_t C = std::max<std::int64_t>(1, x.comps);
  return x.at(static_cast<std::size_t>((std::max<std::int64_t>(row, 0) * x.cols + col) * C + comp));
}
bool measures_has(const Value& x, const std::string& m) {
  if (x.facts.value("measure", std::string()) == m) return true;
  for (const auto& e : x.facts.value("measures", Json::array())) if (e == m || (e.is_object() && e.value("measure", std::string()) == m)) return true;
  return false;
}
void over_events(const Value& x, const std::string& who, const std::string& port, const std::vector<std::string>& measures) {
  need(x.cls == "measurement" && x.facts.contains("over") && x.facts["over"].value("events", false), who + ": " + port + " must be a measurement over the events (`select.gabor_fit`, or the `edges` rule's values)");
  for (const auto& m : measures) need(measures_has(x, m), who + ": " + port + " holds no measure " + m + " (it holds " + x.facts.value("measures", Json::array()).dump() + ")");
}
void series_input(const Value& s, const std::string& who, const std::string& port) {
  need(s.cls == "series" && s.time, who + ": " + port + " must be a sample series of the run (`measurement.collect`)");
}

// THE MEASURES OF AN ATOM (one per column), their units; stored as a member's own measures (`member`, span events, in the measures registry)
const std::vector<std::pair<std::string, std::string>>& atom_measures() {
  static const std::vector<std::pair<std::string, std::string>> m = {
      {"atom_time", "s"}, {"atom_frequency", "Hz"}, {"atom_width", "s"}, {"atom_fwhm", "s"}, {"atom_cycles", "cycles"}, {"atom_amplitude", ""}, {"atom_phase", "rad"},
      {"atom_coefficient", ""}, {"atom_width_peak", "-"}, {"atom_r2", "-"}};
  return m;
}
Json measures_json(const std::vector<std::pair<std::string, std::string>>& ms) {
  Json a = Json::array();
  for (const auto& [n, u] : ms) a.push_back(Json{{"measure", n}, {"components", 1}, {"unit", u}});
  return a;
}
// a measurement over events, [rows x events x measures], NaN off each event's row; `describe` for the declared store
Array event_table(std::int64_t R, std::int64_t E, const std::vector<std::pair<std::string, std::string>>& ms) {
  const std::int64_t C = static_cast<std::int64_t>(ms.size());
  Array y{R, E, std::vector<double>(static_cast<std::size_t>(R * E * C), kNaN)};
  y.comps = C;
  Json cols = Json::array();
  for (const auto& [n, u] : ms) cols.push_back(Json{{"measure", n}, {"unit", u}});
  y.meta = Json{{"members", E}, {"events", true}, {"measures", measures_json(ms)}, {"describe", Json{{"measures", cols}, {"shape", Json::array({R, E, C})}, {"values_f32", false}}}};
  return y;
}
Value event_measurement(const Value& ev, const std::vector<std::pair<std::string, std::string>>& ms, const std::string& kind) {
  Value y = made(ev);
  y.cls = "measurement";
  y.chunked = false;
  y.complex = false;
  Json names = Json::array();
  for (const auto& m : ms) names.push_back(m.first);
  y.facts = Json{{"measure", ms.front().first}, {"measures", names}, {"over", Json{{"events", true}, {"events_kind", kind}, {"selection", "the " + kind + " of this run"}}}};
  if (ev.facts.contains("gabor")) y.facts["gabor"] = ev.facts["gabor"];
  return y;
}
}  // namespace

void add_gabor_steps(std::vector<StepSpec>& r) {
  // ───────────────────────── signal.gabor ─────────────────────────
  {
    StepSpec s;
    s.name = "signal.gabor"; s.kind = "accumulate";
    s.inputs = {"x", "exclude"}; s.optional = {"exclude"}; s.outputs = {"y"};
    s.parameters = Json{{"band", nullptr}, {"sigma_min", 0.025}, {"sigma_max", 0.64}, {"per_octave", 4}, {"min_f_sigma", 0.5}, {"tolerance", 1e-6}, {"decimation", "own"}, {"budget", 1e9}, {"bad_segments", "exclude"}};
    s.matches = "the reference detector's channel_atoms (unit-energy atoms of free width; its grid; each width on its own band, at its own rate); compute nxr::signal::timefilterbank (Family::Gabor, gaborGrid; spectrum), nxr::signal::wtDecimated";
    s.description = "|W| = |∫ x ψ*| of a real field on … × time on every Gabor atom of the grid (unit-energy atoms ψ = (πσ²)^(−1/4) e^(−t²/2σ²) e^(i2πft): widths sigma_min · 2^(k/per_octave) up to sigma_max, at each width the "
                    "frequencies of `band` widened by one σ_f = 1/(2πσ), in steps of σ_f/2, f·σ at least min_f_sigma), EACH WIDTH AT ITS OWN RATE (decimation own: every D-th sample, D the largest power of two keeping at least 3 samples a σ; "
                    "none: every sample, for checks): computed one width at a time from one spectrum of the block, on the width's own bins (exactly the full-rate values at its frames), and kept as a series on the width's frames "
                    "(a line tiled from the recording's samples). Rows atom-major (row = atom x channels + channel), one group of rows a width over its own frames. A frame belongs to the chunk holding its sample; the block "
                    "carries the halo the widest atom needs (its response down to `tolerance` of its L1 norm), so the series does not depend on the chunk size. Frames on a bad sample are NaN. The magnitude is taken here, as the "
                    "series leaves the loop (holding it complex would double it; nothing after reads its phase: the fit reads the samples). MEMORY of one block, checked against `budget` (bytes) before any data flows and "
                    "refused with the remedy: channels x (the padded block x 8 + its spectrum x 16) + the largest width's atoms x channels x the block's frames x 4. THE RUN holds each width's |W| over the span: "
                    "Σ over widths of atoms x channels x (rate / decimation) x 4 bytes a second; that times the span asked, plus the block, is checked against `budget` on the first block (before anything is held) and refused "
                    "with both figures, the longest span that fits and the remedies.";
    s.traits = node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::follows_input, "the block is transformed in double; |W| is kept in float32 (follows the float32 blocks)",
                           "the widest atom's reach: timefilterbank::haloOf of the Gabor member of sigma_max down to `tolerance`, samples on each side");
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      const Value& x = ins.at("x");
      need(x.cls == "field" && x.time && x.chunked && x.facts.value("axis", std::string("samples")) == "samples", "signal.gabor: x must be a field on … × time, inside the chunk loop, on the recording's own samples");
      need(!x.complex, "signal.gabor: x is complex; the atoms are analytic (one-sided) and are taken of a real field");
      need(x.value_type == "scalar", "signal.gabor: x is " + x.value_type + "; one number a channel");
      need(p["bad_segments"] == "exclude" || p["bad_segments"] == "ignore", "signal.gabor: bad_segments " + p["bad_segments"].dump() + ": exclude or ignore");
      const GaborRequest q = gabor_request(p, "signal.gabor");
      double top = 0;
      for (const auto& a : q.grid.atoms) top = std::max(top, a.second + 1 / (2 * kPi * a.first));
      need(top < x.rate / 2, "signal.gabor: the atoms reach " + std::to_string(top) + " Hz, past Nyquist (" + std::to_string(x.rate / 2) + " Hz)");
      // THE SIZE CHECK, before any data flows: a chunk is at most 2^15 samples, with the halo each side
      if (!x.manifolds.empty() && x.manifold_facts.contains(x.manifolds[0]) && x.manifold_facts[x.manifolds[0]].contains("n"))
        size_check(gabor_state(q, x.rate, x.origin), x.manifold_facts[x.manifolds[0]]["n"].get<std::int64_t>(), std::int64_t(1) << 15, "signal.gabor");
      Value y = made(x);
      y.cls = "measurement";
      y.chunked = false;
      y.complex = false;
      y.manifolds = {*x.time};
      y.facts = Json{{"measure", "gabor_magnitude"}, {"gabor", gabor_facts(q)}, {"made_by", "gabor"}, {"nonlinear", true},
                     {"over", Json{{"frames", "each width's own frames: every D-th sample of the recording's line (tiled from it)"}, {"made_from", "tiled"}, {"selection", "the widths' frames of this run"}}}};
      return Values{{"y", y}};
    };
    s.prepare = [](const Values& ins, const Json& p, ExecContext&) { return std::any(gabor_state(gabor_request(p, "signal.gabor"), ins.at("x").rate, ins.at("x").origin)); };
    s.halo = [](const std::any& st) { return std::any_cast<const GaborState&>(st).halo; };
    s.halo_at_plan = [](const Values& ins, const Json& p) { return gabor_halo(gabor_request(p, "signal.gabor"), ins.at("x").rate); };
    s.record = [](const std::any& st_) {
      const GaborState& st = std::any_cast<const GaborState&>(st_);
      Json w = Json::array();
      for (const auto& g : st.widths) w.push_back(Json{{"sigma", g.sigma}, {"decimation", g.D}, {"atoms", g.atoms}});
      return Json{{"halo_samples", st.halo}, {"atoms", st.atoms}, {"widths", w}, {"budget", st.budget}};
    };
    s.start = [](const std::any& st_) {
      const GaborState& st = std::any_cast<const GaborState&>(st_);
      GaborAcc a;
      a.rows.resize(st.widths.size());
      a.first.assign(st.widths.size(), -1);
      return std::any(std::move(a));
    };
    s.fold = [](const std::any& st_, std::any& acc_, const Blocks& ins, const Place& at) {
      const GaborState& st = std::any_cast<const GaborState&>(st_);
      GaborAcc& acc = std::any_cast<GaborAcc&>(acc_);
      const Block& b = ins.at("x");
      const std::int64_t C = b.rows;
      need(b.width >= 2, "signal.gabor: a block holds at least 2 samples");
      size_check(st, C, at.n, "signal.gabor");
      if (acc.channels < 0 && at.end != std::numeric_limits<std::int64_t>::max()) span_check(st, C, at.end - at.origin, "signal.gabor");   // the first block: nothing held yet
      if (acc.channels < 0) { acc.channels = C; for (std::size_t k = 0; k < st.widths.size(); ++k) acc.rows[k].assign(static_cast<std::size_t>(st.widths[k].atoms * C), {}); }
      // the block on the line: column 0 is sample s0; padded on the left to a0 (a multiple of every width's decimation), so a width's frames fall on its decimated grid
      const std::int64_t s0 = at.first - b.left, a0 = (s0 >= 0 ? s0 / st.Dmax : (s0 - st.Dmax + 1) / st.Dmax) * st.Dmax, pad = s0 - a0;
      const std::size_t nfft = block_grid(st, pad + b.width);
      Eigen::MatrixXd xs = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nfft), C);   // [samples x channels], zero outside the block (no wrap)
      for (std::int64_t c = 0; c < C; ++c) for (std::int64_t t = 0; t < b.width; ++t) xs(pad + t, c) = b.re[static_cast<std::size_t>(c * b.width + t)];
      Eigen::MatrixXcd X;
      for (std::size_t k = 0; k < st.widths.size(); ++k) {
        const GaborWidth& w = st.widths[k];
        const tb::TimeFilterbank bank = tb::timeBank(w.design, nfft, st.fs);
        if (X.size() == 0) X = tb::spectrum(bank, xs);                            // one spectrum of the block, every width read from it
        // this block's own frames of the width: frame j at sample j·D, j·D in [first, first + n)
        const std::int64_t j0 = (at.first + w.D - 1) / w.D, j1 = (at.first + at.n + w.D - 1) / w.D;
        if (acc.first[k] < 0) acc.first[k] = j0;
        for (std::int64_t m = 0; m < w.atoms; ++m) {
          const Eigen::MatrixXcd W = nxr::signal::wtDecimated(bank, static_cast<int>(m), X, static_cast<int>(w.D));   // [channels x nfft/D]: the block's samples a0 + i·D
          for (std::int64_t c = 0; c < C; ++c) {
            std::vector<float>& row = acc.rows[k][static_cast<std::size_t>(m * C + c)];
            for (std::int64_t j = j0; j < j1; ++j) {
              const std::int64_t i = (j * w.D - a0) / w.D, u = j * w.D - at.first;
              const bool good = !(at.good && !at.good->empty() && !(*at.good)[static_cast<std::size_t>(u)]);
              row.push_back(good ? static_cast<float>(std::abs(W(c, i))) : std::numeric_limits<float>::quiet_NaN());
            }
          }
        }
      }
    };
    s.finish = [](const std::any& st_, std::any& acc_) {
      const GaborState& st = std::any_cast<const GaborState&>(st_);
      GaborAcc& acc = std::any_cast<GaborAcc&>(acc_);
      need(acc.channels > 0, "signal.gabor: no sample in the span");
      Array y;
      Json groups = Json::array(), widths = Json::array();
      std::int64_t off = 0, row0 = 0;
      for (std::size_t k = 0; k < st.widths.size(); ++k) {
        const GaborWidth& w = st.widths[k];
        const std::int64_t R = w.atoms * acc.channels, n = R ? static_cast<std::int64_t>(acc.rows[k][0].size()) : 0;
        for (const auto& r : acc.rows[k]) { y.single.insert(y.single.end(), r.begin(), r.end()); }
        acc.rows[k].clear();
        groups.push_back(Json{{"first_row", row0}, {"rows", R}, {"members", n}, {"offset", off}});
        widths.push_back(Json{{"sigma", w.sigma}, {"decimation", w.D}, {"hop_s", static_cast<double>(w.D) / st.fs}, {"first_frame", acc.first[k]}, {"frames", n}, {"atom0", w.atom0}, {"atoms", w.atoms}});
        off += R * n; row0 += R;
      }
      y.rows = row0; y.cols = 1;
      y.meta = Json{{"measure", "gabor_magnitude"}, {"row_groups", groups}, {"widths", widths}, {"channels", acc.channels}, {"rate", st.fs}, {"origin", st.origin},
                    {"layout", "per width (a row group), rows atom x channels + channel, each over the width's own frames (frame j at sample j x decimation)"}};
      return Arrays{{"y", std::move(y)}};
    };
    r.push_back(std::move(s));
  }

  // ───────────────────────── select.events.peaks ─────────────────────────
  {
    StepSpec s;
    s.name = "select.events.peaks"; s.kind = "after";
    s.inputs = {"x", "baseline"}; s.outputs = {"spans"};
    s.parameters = Json{{"k", 4.0}, {"edge", 0.5}};
    s.matches = "the reference detector's peaks (scipy maximum_filter 3 x 3, mode nearest; median + k · 1.4826 MAD of |W| over time, per frequency row); compute nxr::select::localMaxima";
    s.description = "POINT EVENTS at the local maxima of `signal.gabor`'s |W| (each width over its own frames): per channel and width, the cells of the (frequency x frame) grid that are at least every neighbour in their 3 x 3 box and "
                    "at least their row's level, median + k x robust SD, read from `baseline` (`measurement.summarize`: median and mad of the same rows, each over its width's frames); frames within max(edge, 3σ) of the run's "
                    "ends are left out. Each event is its frame's instant on its channel (spans in samples of the line, one decimation long), and records its atom (σ, f), its value and its score (value − median) / robust SD.";
    s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the series and the statistic: double; a table of events");
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      const Value &x = ins.at("x"), &b = ins.at("baseline");
      need(x.cls == "measurement" && x.facts.contains("gabor") && x.facts.value("measure", std::string()) == "gabor_magnitude", "select.events.peaks: x must be `signal.gabor`'s |W| (each width over its own frames)");
      need(b.cls == "statistic" && b.facts.contains("summaries"), "select.events.peaks: baseline must be a statistic made by `summarize`");
      bool med = false, mad = false;
      for (const auto& n : b.facts["summaries"]) { med = med || n == "median"; mad = mad || n == "mad"; }
      need(med && mad, "select.events.peaks: the baseline must hold median and mad");
      need(b.facts.value("combine", std::string()) == "each", "select.events.peaks: the baseline is per row (combine each): one level an atom and channel");
      need(p["k"].is_number() && p["edge"].is_number() && p["edge"].get<double>() >= 0, "select.events.peaks: k (robust SDs above the median) and edge (s, at least 0) are numbers");
      Value y = made(x);
      y.cls = "spans";
      y.facts = Json{{"events_kind", "atoms"}, {"gabor", x.facts["gabor"]}, {"peaks", p}};
      return Values{{"spans", y}};
    };
    s.prepare = [](const Values& ins, const Json& p, ExecContext&) { Json q = p; q["gabor"] = ins.at("x").facts["gabor"]; return std::any(q); };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const Json& p = std::any_cast<const Json&>(st);
      const Array &x = ins.at("x"), &base = ins.at("baseline");
      const tb::GaborGrid grid = grid_of_facts(p["gabor"]);
      const std::int64_t C = x.meta["channels"].get<std::int64_t>();
      const double fs = x.meta["rate"].get<double>(), origin = x.meta["origin"].get<double>();
      std::int64_t bmed = -1, bmad = -1;
      for (std::size_t k = 0; k < base.meta["summaries"].size(); ++k) { const std::string n = base.meta["summaries"][k]["name"]; if (n == "median") bmed = static_cast<std::int64_t>(k); if (n == "mad") bmad = static_cast<std::int64_t>(k); }
      need(base.rows == x.rows, "select.events.peaks: the baseline has " + std::to_string(base.rows) + " rows, x " + std::to_string(x.rows));
      const double k = p["k"].get<double>(), edge = p["edge"].get<double>();
      Array out{0, 4, {}};
      Json events = Json::array();
      std::int64_t below_spread = 0, last_sample = 0;
      for (std::size_t g = 0; g < x.meta["widths"].size(); ++g) {
        const Json& wd = x.meta["widths"][g];
        const Json& gr = x.meta["row_groups"][g];
        const std::int64_t D = wd["decimation"].get<std::int64_t>(), j0 = wd["first_frame"].get<std::int64_t>(), T = wd["frames"].get<std::int64_t>(), a0 = wd["atom0"].get<std::int64_t>(), F = wd["atoms"].get<std::int64_t>();
        const std::int64_t r0 = gr["first_row"].get<std::int64_t>(), off = gr["offset"].get<std::int64_t>();
        const double sigma = wd["sigma"].get<double>();
        last_sample = std::max(last_sample, (j0 + T) * D);
        for (std::int64_t c = 0; c < C; ++c) {
          Eigen::MatrixXd v(F, T);
          Eigen::VectorXd level(F), med(F), sd(F);
          std::int64_t first = T, last = -1;
          for (std::int64_t i = 0; i < F; ++i) {
            const std::int64_t local = i * C + c, row = r0 + local;
            for (std::int64_t j = 0; j < T; ++j) { v(i, j) = x.at(static_cast<std::size_t>(off + local * T + j)); if (std::isfinite(v(i, j))) { first = std::min(first, j); last = std::max(last, j); } }
            med(i) = base.at(static_cast<std::size_t>(row * base.cols + bmed));
            sd(i) = base.at(static_cast<std::size_t>(row * base.cols + bmad));
            level(i) = med(i) + k * sd(i);
            if (!(sd(i) > 0)) ++below_spread;
          }
          auto time = [&](std::int64_t j) { return origin + static_cast<double>((j0 + j) * D) / fs; };
          const double t_lo = time(first) + std::max(edge, 3 * sigma), t_hi = time(last) - std::max(edge, 3 * sigma);
          for (const auto& [i, j] : nxr::select::localMaxima(v, level)) {
            const double t = time(j);
            if (!(t >= t_lo && t <= t_hi)) continue;
            const std::int64_t atom = a0 + i, sample = (j0 + j) * D;
            out.data.insert(out.data.end(), {static_cast<double>(sample), static_cast<double>(sample + D), static_cast<double>(c), static_cast<double>(atom)});
            ++out.rows;
            events.push_back(Json{{"atom", atom}, {"sigma", sigma}, {"f", grid.atoms[static_cast<std::size_t>(atom)].second}, {"value", v(i, j)},
                                  {"score", sd(i) > 0 ? Json((v(i, j) - med(i)) / sd(i)) : Json()}, {"f_at_edge", i == 0 || i == F - 1}});
          }
        }
      }
      out.meta = Json{{"tiles_of", Json{{"width_s", 1 / fs}, {"origin", origin}, {"unit", "sample"}, {"instant", true}, {"tiles", last_sample}, {"rate", fs}}}, {"candidates", events},
                      {"thresholds", Json{{"k", k}, {"edge_s", edge}, {"rows_without_spread", below_spread},
                                          {"statistic", "per atom and channel: median + k x the robust SD (1.4826 x MAD) of the baseline statistic (summarize) over the width's own frames"}}},
                      {"reference", "the reference detector's peaks"}};
      return Arrays{{"spans", std::move(out)}};
    };
    r.push_back(std::move(s));
  }

  // ───────────────────────── select.gabor_fit ─────────────────────────
  {
    StepSpec s;
    s.name = "select.gabor_fit"; s.kind = "after";
    s.inputs = {"x", "events", "band"}; s.optional = {"band"}; s.outputs = {"atoms"};
    s.parameters = Json{{"steps", 2}};
    s.matches = "the reference detector's refine, is_burst and fit_r2; compute nxr::signal::gaborFit, gaborWidthPeak, gaborR2";
    s.description = "each event of `select.events.peaks` fitted on the SAMPLE series of its channel (`collect`, as `select.risefall` fits on it): a parabola through log|W| in frequency (± σ_f/2) and in log width (× 2^(±1/8)), "
                    "`steps` rounds, by the inner product itself; then the atom's time (the tile's centre), frequency, width σ, FWHM 2.355 σ, cycles at half maximum, amplitude 2|W| / (π^(1/4) √σ), phase, |W|, "
                    "the width test (1: |W| peaks along width too, inside the grid's widths; 0: not), and R² of the atom against `band` (a band-passed series of the run, `collect`; NaN without it) in its half-maximum "
                    "window. A measurement over the events [channels x events x measures], NaN off each event's channel. MEMORY: the series held by `collect`.";
    s.traits = node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::float64, "fits each event on the series around it (±5σ); events are independent of each other; a table of events",
                           "the event's time ± 5σ of its widest trial width");
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      const Value &x = ins.at("x"), &ev = ins.at("events");
      series_input(x, "select.gabor_fit", "x");
      need(ev.cls == "spans" && ev.facts.contains("gabor"), "select.gabor_fit: events must be the candidates `select.events.peaks` found on a `signal.gabor` field");
      if (ins.count("band")) series_input(ins.at("band"), "select.gabor_fit", "band");
      need(p["steps"].is_number_integer() && p["steps"].get<int>() >= 0, "select.gabor_fit: steps is a whole number of refinement rounds, at least 0");
      return Values{{"atoms", event_measurement(ev, atom_measures(), "atoms")}};
    };
    s.prepare = [](const Values& ins, const Json& p, ExecContext&) { Json q = p; q["gabor"] = ins.at("events").facts["gabor"]; q["has_band"] = ins.count("band") > 0; return std::any(q); };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const Json& p = std::any_cast<const Json&>(st);
      const Array &x = ins.at("x"), &ev = ins.at("events");
      const Array* band = p["has_band"].get<bool>() ? &ins.at("band") : nullptr;
      const tb::GaborGrid grid = grid_of_facts(p["gabor"]);
      const double fs = x.meta["rate"].get<double>(), offset = x.meta["origin"].get<double>() + static_cast<double>(x.meta.value("first_sample", std::int64_t(0))) / fs;
      const double w = ev.meta["tiles_of"]["width_s"].get<double>(), origin = ev.meta["tiles_of"]["origin"].get<double>();
      const double step = std::pow(2.0, 1.0 / p["gabor"]["per_octave"].get<double>());
      const std::int64_t R = x.rows, E = ev.rows, steps = p["steps"].get<std::int64_t>();
      if (band) need(band->rows == R && band->cols == x.cols && band->meta.value("first_sample", std::int64_t(0)) == x.meta.value("first_sample", std::int64_t(0)), "select.gabor_fit: band is not a series of the same rows and samples as x");
      Array y = event_table(R, E, atom_measures());
      const std::int64_t M = static_cast<std::int64_t>(atom_measures().size());
      std::int64_t width_peaks = 0;
      for (std::int64_t e = 0; e < E; ++e) {
        const double* sp = ev.data.data() + e * 4;
        const std::int64_t row = static_cast<std::int64_t>(sp[2]), atom = static_cast<std::int64_t>(sp[3]);
        need(row >= 0 && row < R, "select.gabor_fit: an event on row " + std::to_string(row) + " but the series has " + std::to_string(R) + " rows");
        need(atom >= 0 && atom < static_cast<std::int64_t>(grid.atoms.size()), "select.gabor_fit: an event names atom " + std::to_string(atom) + ", not one of the grid's");
        const Eigen::Map<const Eigen::VectorXd> series(x.data.data() + row * x.cols, x.cols);
        const double t = origin + (sp[0] + (ev.meta["tiles_of"].value("instant", false) ? 0.0 : 0.5)) * w - offset;   // the frame's instant (a tile's centre) on the series' clock
        const auto [sigma, f] = grid.atoms[static_cast<std::size_t>(atom)];
        const nxr::signal::GaborAtom a = nxr::signal::gaborFit(series, fs, t, f, sigma, static_cast<int>(steps));
        const bool peak = nxr::signal::gaborWidthPeak(series, fs, a, grid.widths.front(), grid.widths.back(), step);
        width_peaks += peak;
        const double r2 = band ? nxr::signal::gaborR2(Eigen::Map<const Eigen::VectorXd>(band->data.data() + row * band->cols, band->cols), fs, a) : kNaN;
        const double vals[] = {a.t0 + offset, a.f, a.sigma, a.fwhm, a.cycles, a.amplitude, a.phase, a.coefficient, peak ? 1.0 : 0.0, r2};
        for (std::int64_t m = 0; m < M; ++m) y.data[static_cast<std::size_t>((row * E + e) * M + m)] = vals[m];
      }
      y.meta["fit"] = Json{{"steps", steps}, {"width_step", step}, {"widths", Json::array({grid.widths.front(), grid.widths.back()})}, {"width_peaks", width_peaks}, {"r2_of", band ? "the band series given" : "none (NaN)"},
                           {"reference", "the reference detector's refine, is_burst and fit_r2"}};
      return Arrays{{"atoms", std::move(y)}};
    };
    r.push_back(std::move(s));
  }

  // ───────────────────────── select.events.suppress ─────────────────────────
  {
    StepSpec s;
    s.name = "select.events.suppress"; s.kind = "after";
    s.inputs = {"events", "x"}; s.outputs = {"spans"};
    s.parameters = Json{{"strength", "atom_coefficient"}, {"time", "atom_time"}, {"frequency", "atom_frequency"}, {"width", "atom_width"}};
    s.matches = "the reference detector's merge (strongest |W| first, the matched-filter choice); compute nxr::select::suppress";
    s.description = "NON-MAXIMUM SUPPRESSION of the events of each row (a channel): the strongest first (`strength`); an event is dropped when a kept one lies within the larger of their time resolutions and the larger of their "
                    "frequency resolutions, the resolutions those of a Gabor atom of the event's width σ (`width`): σ in time, 1/(2πσ) in frequency. The events' places are measures of x (`time`, `frequency`). "
                    "Gives the kept events (a subset, in their order), each kept event's column in x recorded.";
    s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the events' measures: double");
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      const Value &ev = ins.at("events"), &x = ins.at("x");
      need(ev.cls == "spans", "select.events.suppress: events must be a spans selection of the run");
      for (const char* k : {"strength", "time", "frequency", "width"}) need(p[k].is_string(), std::string("select.events.suppress: ") + k + " names a measure of x");
      over_events(x, "select.events.suppress", "x", {p["strength"].get<std::string>(), p["time"].get<std::string>(), p["frequency"].get<std::string>(), p["width"].get<std::string>()});
      Value y = made(ev);
      y.facts["gate"] = Json{{"suppress", p}};
      return Values{{"spans", y}};
    };
    s.prepare = [](const Values&, const Json& p, ExecContext&) { return std::any(p); };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const Json& p = std::any_cast<const Json&>(st);
      const Array &ev = ins.at("events"), &x = ins.at("x");
      const auto col = measure_columns(x);
      const std::int64_t cs = col.at(p["strength"]), ct = col.at(p["time"]), cf = col.at(p["frequency"]), cw = col.at(p["width"]);
      std::vector<nxr::select::PointEvent> pe;
      for (std::int64_t i = 0; i < ev.rows; ++i) {
        const std::int64_t row = static_cast<std::int64_t>(ev.data[static_cast<std::size_t>(i * 4 + 2)]), c = column_of(ev, x, i);
        nxr::select::PointEvent e;
        e.t = value_at(x, row, c, ct); e.f = value_at(x, row, c, cf); e.tResolution = value_at(x, row, c, cw); e.fResolution = 1 / (2 * kPi * e.tResolution);
        e.strength = value_at(x, row, c, cs); e.group = row;
        pe.push_back(e);
      }
      std::vector<std::size_t> kept = nxr::select::suppress(pe);
      std::sort(kept.begin(), kept.end());
      Array out{0, 4, {}};
      Json ordinals = Json::array(), measured = Json::array();
      for (const std::size_t i : kept) {
        out.data.insert(out.data.end(), ev.data.begin() + static_cast<std::ptrdiff_t>(i * 4), ev.data.begin() + static_cast<std::ptrdiff_t>(i * 4 + 4));
        ++out.rows;
        ordinals.push_back(i);
        measured.push_back(column_of(ev, x, static_cast<std::int64_t>(i)));
      }
      out.meta = ev.meta;
      out.meta["measured_ordinals"] = measured;
      out.meta["gate"] = Json{{"suppress", Json{{"strength", p["strength"]}, {"resolution", "a Gabor atom's: sigma in time, 1/(2 pi sigma) in frequency"}}}, {"events", ev.rows}, {"kept", out.rows}, {"ordinals", ordinals},
                              {"reference", "the reference detector's merge"}};
      return Arrays{{"spans", std::move(out)}};
    };
    r.push_back(std::move(s));
  }

  // ───────────────────────── event_edges (select.events.spans rule edges) ─────────────────────────
  {
    StepSpec s;
    s.name = "event_edges"; s.kind = "after";
    s.inputs = {"events", "x", "series"}; s.optional = {"series"}; s.outputs = {"spans", "values"};
    s.parameters = Json{{"edges", "tangent"}, {"envelope_window", 3.0}, {"trough_reach", 1.0}};
    s.matches = "the reference detector's tangent_edges (the tangent edges, the default) or the atom's half maximum (centre ∓ 1.18σ); compute nxr::select::tangentEdges, nxr::signal::wt (a Morlet member)";
    s.description = "each event's ONSET and OFFSET, as spans on its row: `edges` tangent (the default): the envelope of its row's samples (`series`, `collect`) read by an analytic Morlet member at the event's frequency, of "
                    "width σ_r = max(σ/2, 3/(2πf)) (gain 2: the amplitude), over the event's time ± max(envelope_window, 6σ); its peak within the half-maximum window; the troughs either side (the first local minimum "
                    "below half the peak, else the lowest point within trough_reach); onset where the tangent at the steepest rise meets the trough's level, offset the same on the fall (an event with no such rise "
                    "and fall has no edges, and is counted); `half_max`: the atom's half maximum, time ∓ FWHM/2. Also the events' measures (x's, for the events kept) with their onset and offset.";
    s.traits = node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::float64, "each event read on the series around it; events are independent of each other; a table of events",
                           "the event's time ± max(envelope_window, 6σ)");
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      const Value &ev = ins.at("events"), &x = ins.at("x");
      need(ev.cls == "spans", "select.events.spans edges: events must be a spans selection of the run");
      need(p["edges"] == "tangent" || p["edges"] == "half_max", "select.events.spans edges: edges " + p["edges"].dump() + ": tangent (on the envelope, the default) or half_max (the atom's)");
      over_events(x, "select.events.spans edges", "x", {"atom_time", "atom_frequency", "atom_width", "atom_fwhm"});
      if (p["edges"] == "tangent") {
        need(ins.count("series") != 0, "select.events.spans edges: tangent edges are read on the envelope of the events' rows: `series` (the run's samples, `collect`) is needed");
        series_input(ins.at("series"), "select.events.spans edges", "series");
        need(p["envelope_window"].is_number() && p["envelope_window"].get<double>() > 0 && p["trough_reach"].is_number() && p["trough_reach"].get<double>() > 0,
             "select.events.spans edges: envelope_window and trough_reach are durations above 0 s");
      }
      Value y = made(ev);
      y.facts.erase("gate");
      y.facts.erase("detect");
      y.facts["events_kind"] = "bursts";
      y.facts["edges"] = p["edges"];
      std::vector<std::pair<std::string, std::string>> ms;
      for (const auto& m : x.facts.value("measures", Json::array())) ms.push_back({m.get<std::string>(), ""});
      ms.push_back({"onset", "s"}); ms.push_back({"offset", "s"});
      return Values{{"spans", y}, {"values", event_measurement(y, ms, "bursts")}};
    };
    s.prepare = [](const Values&, const Json& p, ExecContext&) { return std::any(p); };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const Json& p = std::any_cast<const Json&>(st);
      const Array &ev = ins.at("events"), &x = ins.at("x");
      const bool tangent = p["edges"] == "tangent";
      const Array* series = ins.count("series") ? &ins.at("series") : nullptr;
      const auto col = measure_columns(x);
      const double w = ev.meta["tiles_of"]["width_s"].get<double>(), origin = ev.meta["tiles_of"]["origin"].get<double>();
      const double tiles = ev.meta["tiles_of"].value("tiles", 0) > 0 ? static_cast<double>(ev.meta["tiles_of"]["tiles"].get<std::int64_t>()) : std::numeric_limits<double>::infinity();
      std::vector<std::pair<std::string, std::string>> ms;
      for (const auto& m : x.meta.value("measures", Json::array())) ms.push_back({m.value("measure", std::string()), m.value("unit", std::string())});
      const std::int64_t MX = static_cast<std::int64_t>(ms.size());
      ms.push_back({"onset", "s"}); ms.push_back({"offset", "s"});
      const std::int64_t M = static_cast<std::int64_t>(ms.size()), R = x.rows;
      Array out{0, 4, {}};
      Json measured = Json::array(), record = Json::array();
      std::vector<std::vector<double>> rows_values;
      std::vector<std::int64_t> rows_of;
      std::int64_t no_edges = 0;
      for (std::int64_t i = 0; i < ev.rows; ++i) {
        const std::int64_t row = static_cast<std::int64_t>(ev.data[static_cast<std::size_t>(i * 4 + 2)]), c = column_of(ev, x, i);
        const double t0 = value_at(x, row, c, col.at("atom_time")), f = value_at(x, row, c, col.at("atom_frequency")), sigma = value_at(x, row, c, col.at("atom_width")), fwhm = value_at(x, row, c, col.at("atom_fwhm"));
        double on = t0 - fwhm / 2, off = t0 + fwhm / 2;
        if (tangent) {
          const double fs = series->meta["rate"].get<double>(), offset = series->meta["origin"].get<double>() + static_cast<double>(series->meta.value("first_sample", std::int64_t(0))) / fs;
          const std::int64_t T = series->cols, ic = static_cast<std::int64_t>(std::llround((t0 - offset) * fs));
          const std::int64_t h = static_cast<std::int64_t>(std::max(p["envelope_window"].get<double>(), 6 * sigma) * fs);
          const std::int64_t i0 = std::max<std::int64_t>(0, ic - h), i1 = std::min<std::int64_t>(T, ic + h);
          nxr::select::TangentEdges te;
          if (i1 - i0 >= 3 && ic >= i0 && ic < i1) {
            // the envelope at the event's frequency: an analytic Morlet member of width 1/(2π σ_r) in frequency, gain 2 (|c| = A), DC and Nyquist dropped
            const double sr = std::max(sigma / 2, 3 / (2 * kPi * f));
            tb::Design d(tb::Family::Morlet, 2.0, tb::Sides::Analytic);
            d.centres = {f}; d.cycles = 2 * kPi * f * sr; d.edgeBins = tb::EdgeBins::Drop;
            const Eigen::MatrixXd win = Eigen::Map<const Eigen::VectorXd>(series->data.data() + row * T + i0, i1 - i0);
            const tb::TimeFilterbank bank = tb::timeBank(d, tb::gridLength(static_cast<std::size_t>(i1 - i0), tb::Padding::FastLen), fs);
            const Eigen::VectorXd env = nxr::signal::wt(bank, win)[0].col(0).head(i1 - i0).cwiseAbs();
            te = nxr::select::tangentEdges(env, ic - i0, std::max<std::int64_t>(1, static_cast<std::int64_t>(fwhm / 2 * fs)), static_cast<std::int64_t>(p["trough_reach"].get<double>() * fs));
            if (te.ok) { on = (te.onset + static_cast<double>(i0)) / fs + offset; off = (te.offset + static_cast<double>(i0)) / fs + offset; }
          }
          if (!te.ok) { ++no_edges; record.push_back(Json{{"event", i}, {"edges", nullptr}}); continue; }
          record.push_back(Json{{"event", i}, {"peak_s", (static_cast<double>(te.peak + i0)) / fs + offset}, {"trough_before_s", (static_cast<double>(te.troughBefore + i0)) / fs + offset},
                                {"trough_after_s", (static_cast<double>(te.troughAfter + i0)) / fs + offset}, {"sigma_read", std::max(sigma / 2, 3 / (2 * kPi * f))}});
        }
        const double a = std::max(0.0, (on - origin) / w), b = std::min(tiles, (off - origin) / w);
        if (!(b > a)) { ++no_edges; continue; }
        out.data.insert(out.data.end(), {a, b, static_cast<double>(row), static_cast<double>(c)});
        ++out.rows;
        measured.push_back(c);
        std::vector<double> v;
        for (std::int64_t m = 0; m < MX; ++m) v.push_back(value_at(x, row, c, m));
        v.push_back(on); v.push_back(off);
        rows_values.push_back(std::move(v));
        rows_of.push_back(row);
      }
      out.meta = Json{{"tiles_of", ev.meta["tiles_of"]}, {"measured_ordinals", measured},
                      {"edges", Json{{"rule", p["edges"]}, {"events", ev.rows}, {"with_edges", out.rows}, {"without_edges", no_edges}, {"envelope_window_s", p["envelope_window"]}, {"trough_reach_s", p["trough_reach"]},
                                     {"events_read", record}, {"edges_in", "seconds (fractions of a tile), clipped to the run"},
                                     {"reference", tangent ? "the reference detector's tangent_edges" : "the atom's half maximum, time -/+ FWHM/2"}}}};
      Array y = event_table(R, out.rows, ms);
      for (std::int64_t e = 0; e < out.rows; ++e)
        for (std::int64_t m = 0; m < M; ++m) y.data[static_cast<std::size_t>((rows_of[static_cast<std::size_t>(e)] * out.rows + e) * M + m)] = rows_values[static_cast<std::size_t>(e)][static_cast<std::size_t>(m)];
      return Arrays{{"spans", std::move(out)}, {"values", std::move(y)}};
    };
    r.push_back(std::move(s));
  }

  // ───────────────────────── event_union (select.events.spans rule union) ─────────────────────────
  {
    StepSpec s;
    s.name = "event_union"; s.kind = "after";
    s.inputs = {"events", "x"}; s.optional = {"x"}; s.outputs = {"spans"};
    s.parameters = Json{{"overlap", "time_frequency"}};
    s.matches = "global windows first (a window where at least one channel bursts, the channels' events kept inside it); compute nxr::select::overlapGroups";
    s.description = "THE GLOBAL WINDOWS of the events of every row: events joined where their spans overlap in time and (overlap time_frequency, the default) their frequency boxes overlap too (f ± 1/(2πσ), from x's "
                    "atom_frequency and atom_width), joined events chained; each window spans its events, on no row (a global event), and records its events' ordinals (the rows' events kept inside it). "
                    "`overlap: time` joins by time alone.";
    s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the events: double");
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      const Value& ev = ins.at("events");
      need(ev.cls == "spans", "select.events.spans union: events must be a spans selection of the run");
      need(p["overlap"] == "time" || p["overlap"] == "time_frequency", "select.events.spans union: overlap " + p["overlap"].dump() + ": time_frequency (the default) or time");
      if (p["overlap"] == "time_frequency") {
        need(ins.count("x") != 0, "select.events.spans union: overlap time_frequency reads each event's frequency box: x (a measurement over the events with atom_frequency and atom_width) is needed");
        over_events(ins.at("x"), "select.events.spans union", "x", {"atom_frequency", "atom_width"});
      }
      Value y = made(ev);
      y.facts.erase("gate");
      y.facts.erase("detect");
      y.facts["events_kind"] = "windows";
      y.facts["union"] = p["overlap"];
      return Values{{"spans", y}};
    };
    s.prepare = [](const Values&, const Json& p, ExecContext&) { return std::any(p); };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const Json& p = std::any_cast<const Json&>(st);
      const Array& ev = ins.at("events");
      const bool freq = p["overlap"] == "time_frequency";
      std::vector<double> a, b, lo, hi;
      for (std::int64_t i = 0; i < ev.rows; ++i) { a.push_back(ev.data[static_cast<std::size_t>(i * 4)]); b.push_back(ev.data[static_cast<std::size_t>(i * 4 + 1)]); }
      if (freq) {
        const Array& x = ins.at("x");
        const auto col = measure_columns(x);
        for (std::int64_t i = 0; i < ev.rows; ++i) {
          const std::int64_t row = static_cast<std::int64_t>(ev.data[static_cast<std::size_t>(i * 4 + 2)]), c = column_of(ev, x, i);
          const double f = value_at(x, row, c, col.at("atom_frequency")), sf = 1 / (2 * kPi * value_at(x, row, c, col.at("atom_width")));
          lo.push_back(f - sf); hi.push_back(f + sf);
        }
      }
      const std::vector<std::int64_t> g = nxr::select::overlapGroups(a, b, lo, hi);
      const std::int64_t G = g.empty() ? 0 : *std::max_element(g.begin(), g.end()) + 1;
      std::vector<double> start(static_cast<std::size_t>(G), std::numeric_limits<double>::infinity()), stop(static_cast<std::size_t>(G), -std::numeric_limits<double>::infinity());
      Json members = Json::array();
      for (std::int64_t k = 0; k < G; ++k) members.push_back(Json::array());
      for (std::size_t i = 0; i < g.size(); ++i) {
        const std::size_t k = static_cast<std::size_t>(g[i]);
        start[k] = std::min(start[k], a[i]); stop[k] = std::max(stop[k], b[i]);
        members[k].push_back(static_cast<std::int64_t>(i));
      }
      Array out{G, 4, {}};
      for (std::int64_t k = 0; k < G; ++k) out.data.insert(out.data.end(), {start[static_cast<std::size_t>(k)], stop[static_cast<std::size_t>(k)], -1.0, static_cast<double>(members[static_cast<std::size_t>(k)].size())});
      out.meta = Json{{"tiles_of", ev.meta["tiles_of"]}, {"union", Json{{"overlap", p["overlap"]}, {"events", ev.rows}, {"windows", G}, {"members", members},
                      {"rule", freq ? "events joined where they overlap in time and in frequency (f -/+ 1/(2 pi sigma)), chained" : "events joined where they overlap in time, chained"}}}};
      return Arrays{{"spans", std::move(out)}};
    };
    r.push_back(std::move(s));
  }
}

}  // namespace nxr::pipelines::graph
