// nxr::selection (compute) — the partition methods (partition.h).
#include "nxr/selection/partition.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "nxr/selection/constant_q.h"
#include "nxr/selection/dyadic.h"

namespace nxr::selection {

namespace {
int k_of(double samples) { return static_cast<int>(std::lround(std::ceil(std::log2(samples)))); }   // the power of two a window is rounded up to
constexpr double kFourPi = 4 * M_PI;
std::string number(double x) { char b[32]; std::snprintf(b, sizeof b, "%.4g", x); return b; }

void need_surface(const Surface& s, const char* what) {
  if (!s.positions || !s.faces || s.vertices < 1 || s.face_count < 1) throw std::invalid_argument(std::string(what) + ": a surface needs positions and faces");
  if (s.unit.empty()) throw std::invalid_argument(std::string(what) + ": a surface needs its positions' unit (m, mm, …)");
}

// the constant-Q band design of the Studio's constant-q.ts (constantQTiling, anchorsFor): the bands, highest first, each with its level k
struct Band { double lo, hi; int k; };
struct Designed { std::vector<Band> bands; double cycles = 0, f_min = 0, f_max = 0, sfreq = 0; };
Designed design(const Line& line, const ConstantQ& o) {
  const std::int64_t samples = line.count;
  const double sfreq = line.spacing > 0 ? 1.0 / line.spacing : 0;
  if (samples < 1 || !(sfreq > 0)) throw std::invalid_argument("constant-Q: a line of " + std::to_string(samples) + " points " + std::to_string(line.spacing) + " apart");
  if (o.method != "hertz" && o.method != "cycles") throw std::invalid_argument("constant-Q: the method is hertz or cycles (got '" + o.method + "')");
  if (o.voices < 1) throw std::invalid_argument("constant-Q: voices is a whole number >= 1");
  const double nyquist = sfreq / 2;
  // the method's anchors: hertz takes the cycles of a power-of-two window at 8 samples a cycle and caps f_max at 64 Hz; cycles takes 8 cycles to Nyquist
  const bool hertz = o.method == "hertz";
  const auto [anchor_cycles, anchor_fmax] = constantq::anchorsFor(hertz ? constantq::Method::Hertz : constantq::Method::Cycles, sfreq);
  const double cycles = o.cycles.value_or(anchor_cycles);
  if (!(cycles > 0)) throw std::invalid_argument("constant-Q: cycles is > 0");
  const double f_max = std::min(o.f_max.value_or(anchor_fmax), nyquist);
  if (!(f_max > 0)) throw std::invalid_argument("constant-Q: f_max is > 0");
  const double f_min = std::max(o.f_min.value_or(hertz ? constantq::kFloorHz : 0.0), 0.0);
  std::vector<Band> bands;
  for (int b = 0;; ++b) {                                              // down from f_max by 1/voices of an octave, while the window fits the line
    const double hi = f_max * std::pow(2.0, -static_cast<double>(b) / o.voices), lo = f_max * std::pow(2.0, -static_cast<double>(b + 1) / o.voices);
    if (lo < f_min) break;
    const int k = k_of(cycles * sfreq / lo);
    if ((std::int64_t(1) << k) > samples) break;
    bands.push_back(Band{lo, hi, k});
  }
  Designed d;
  d.cycles = cycles;
  d.f_max = f_max;
  d.f_min = bands.empty() ? f_max : bands.back().lo;
  d.sfreq = sfreq;
  d.bands = std::move(bands);
  return d;
}

// a set of spans on a spectrum line, one member a band [lo, hi), labelled by its range; the bands must tile the line without gap or overlap
Selection band_spans(std::vector<std::pair<double, double>> b, const std::string& unit) {
  if (b.empty()) throw std::invalid_argument("bands: there are none");
  std::sort(b.begin(), b.end());
  std::vector<SpanRow> rows;
  Lookup lookup;
  for (std::size_t i = 0; i < b.size(); ++i) {
    if (i + 1 < b.size() && std::abs(b[i].second - b[i + 1].first) > 1e-9 * std::max(1.0, std::abs(b[i].second)))
      throw std::invalid_argument("bands: a gap or an overlap at " + number(b[i].second) + " " + unit);
    rows.push_back(SpanRow{b[i].first, b[i].second, static_cast<std::int64_t>(i)});
    lookup.push_back(Entry{static_cast<std::int64_t>(i), number(b[i].first) + "-" + number(b[i].second) + " " + unit, std::nullopt});
  }
  return set_of_spans(std::move(rows), unit, std::move(lookup));
}
}  // namespace

// ── a time line, by constant-Q ───────────────────────────────────────────────────────────
ConstantQDesign constant_q_design(const Line& line, const ConstantQ& o) {
  const Designed d = design(line, o);
  std::map<int, std::pair<double, double>> by_k;                       // level k: the union of its bands
  std::map<int, int> shared;                                           // level k: how many bands share it
  for (const Band& b : d.bands) {
    auto [it, fresh] = by_k.try_emplace(b.k, b.lo, b.hi);
    if (!fresh) { it->second.first = std::min(it->second.first, b.lo); it->second.second = std::max(it->second.second, b.hi); }
    ++shared[b.k];
  }
  ConstantQDesign out;
  out.method = o.method;
  out.voices = o.voices;
  out.sfreq = d.sfreq;
  out.cycles = d.cycles;
  out.f_min = d.f_min;
  out.f_max = d.f_max;
  std::vector<Selection> levels;
  for (const auto& [k, band] : by_k) {                                 // ascending k: finest first; a level's identity is its position, its label k
    const std::int64_t width = std::int64_t(1) << k;
    Selection level = time_grid(line, k);                              // a level is a tiling rule: nothing per member
    level.band = band;
    level.band_unit = "Hz";
    levels.push_back(std::move(level));
    ConstantQLevel info;
    info.power = k;
    info.window_samples = width;
    info.bands = shared[k];
    info.lo = band.first;
    info.hi = band.second;
    info.stride = constantq::strideFor(d.sfreq, info.lo);
    info.rate_hz = d.sfreq / static_cast<double>(info.stride);
    info.points = width / info.stride;
    info.samples_per_cycle_lo = info.rate_hz / info.lo;
    info.samples_per_cycle_hi = info.rate_hz / info.hi;
    info.nyquist_bound = info.stride == 1 && info.samples_per_cycle_lo < constantq::kMultirateMin;
    out.levels.push_back(info);
  }
  out.ladder = set(std::move(levels));
  return out;
}

Selection constant_q(const Line& line, const ConstantQ& o) { return constant_q_design(line, o).ladder; }

Selection constant_q_bands(const Line& line, const ConstantQ& o) {
  std::vector<std::pair<double, double>> b;
  for (const Band& x : design(line, o).bands) b.emplace_back(x.lo, x.hi);
  return band_spans(std::move(b), "Hz");
}

// ── a time line, by a dyadic tiling: tiles of 2^k samples from the first sample ─────────────
Selection time_grid(const Line& line, int k) {
  if (k < 0 || k > 62) throw std::invalid_argument("time_grid: k (tiles of 2^k samples) is 0..62");
  if (line.count < 0 || !(line.spacing > 0)) throw std::invalid_argument("time_grid: a line of count >= 0 points, spacing > 0 apart");
  const std::int64_t width = std::int64_t(1) << k;
  Selection g = tiled(Tiling{line.origin, static_cast<double>(width) * line.spacing, static_cast<double>(width) * line.spacing, (line.count + width - 1) / width}, line.unit);
  g.level = k;
  g.code = k;
  return g;
}

// ── a surface, dyadic: the Studio's ladder per connected piece; level l pairs with lambda = 4 pi m / A (m = the level's tiles, A = the surface's area) ──
// A GRID: its member numbers are the finest tile codes (h 2^D + t), level l's the code >> (D - l) (closed form); a level holds only its label and band.
Selection dyadic(const Surface& s, const Dyadic& o) {
  need_surface(s, "dyadic");
  const bisection::Tree t = bisection::subjectTree(s.positions, s.vertices, s.faces, s.face_count, o.depth.value_or(-1));
  double area = 0;
  for (double a : t.pieceArea) area += a;
  const auto pieces = static_cast<std::int64_t>(t.runs.size());
  std::vector<Selection> levels;
  for (int l = 0; l <= t.depth; ++l) {
    const std::int64_t tiles = pieces << l;
    Selection level;
    level.type = Type::set;
    level.level = l;
    level.band = std::make_pair(l == 0 ? 0.0 : kFourPi * static_cast<double>(tiles / 2) / area, kFourPi * static_cast<double>(tiles) / area);
    level.band_unit = "1/" + s.unit + "^2";                            // area in unit^2, so lambda = 4 pi m / A in 1/unit^2
    level.diameter = t.diameter[static_cast<std::size_t>(l)];          // measured, as the ladder's
    levels.push_back(std::move(level));
  }
  Selection p = dyadic_grid(t.codes, t.depth, t.slots, std::move(levels));
  p.grid->finest = t.finest;                                           // the deepest level every piece reaches
  p.unit = s.unit;
  return p;
}

Selection eigenvalue_bands(const Selection& levels) {
  std::vector<std::pair<double, double>> b;
  std::string unit;
  for (const Selection& l : levels.members) if (l.band) { b.push_back(*l.band); unit = l.band_unit; }
  if (b.empty()) throw std::invalid_argument("eigenvalue_bands: the selection has no levels with bands");
  return band_spans(std::move(b), unit);
}

// ── a surface, by transfer: the nearest vertex on the registered spheres, piece by piece (nxr geom.trees Template.tree_for) ─────────────
Selection transfer(const Surface& s, const Transfer& o) {
  need_surface(s, "transfer");
  if (!o.from || !o.from_sphere || !o.from_faces || !o.sphere || o.from_vertices < 1)
    throw std::invalid_argument("transfer: the source grid, both spheres and the source's faces are needed");
  const Selection& from = *o.from;
  if (from.type != Type::set || !from.grid || from.grid->kind != Grid::Kind::dyadic ||
      static_cast<std::int64_t>(o.from_vertices) != static_cast<std::int64_t>(from.member_of.size()))
    throw std::invalid_argument("transfer: the source is a dyadic grid on a surface of " + std::to_string(o.from_vertices) + " vertices");
  const int depth = from.grid->depth;
  const auto from_runs = bisection::pieceRuns(o.from_faces, o.from_face_count, o.from_vertices);
  const auto runs = bisection::pieceRuns(s.faces, s.face_count, s.vertices);
  double median = 0, max_chord = 0;
  const bisection::Tree t = bisection::groupTreeFor(o.from_sphere, from.member_of.data(), from_runs, depth, o.sphere, s.vertices, runs, depth, &median, &max_chord);
  Selection p = dyadic_grid(t.codes, depth, from.grid->slots, from.members);   // the same codes: a tile no vertex reaches is an empty member
  p.grid->finest = from.grid->finest >= 0 ? from.grid->finest : depth;
  p.grid->match_median = median;
  p.grid->match_max = max_chord;
  p.lookup = from.lookup;
  p.unit = s.unit;
  return p;
}

}  // namespace nxr::selection
