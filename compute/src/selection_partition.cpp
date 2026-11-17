// nxr::selection (compute) — the partition methods (partition.h).
#include "nxr/selection/partition.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "nxr/selection/dyadic.h"

namespace nxr::selection {

namespace {
double js_round(double x) { return std::floor(x + 0.5); }            // the Studio's Math.round: halves toward +infinity
int k_of(double samples) { return static_cast<int>(std::lround(std::ceil(std::log2(samples)))); }   // the power of two a window is rounded up to
constexpr double kFourPi = 4 * M_PI;
std::string number(double x) { char b[32]; std::snprintf(b, sizeof b, "%.4g", x); return b; }

void need_surface(const Surface& s, const char* what) {
  if (!s.positions || !s.faces || s.vertices < 1 || s.face_count < 1) throw std::invalid_argument(std::string(what) + ": a surface needs positions and faces");
  if (s.unit.empty()) throw std::invalid_argument(std::string(what) + ": a surface needs its positions' unit (m, mm, …)");
}

// the constant-Q band design of the Studio's constant-q.ts (constantQTiling, anchorsFor): the bands, highest first, each with its level k
struct Band { double lo, hi; int k; };
std::vector<Band> design(const Line& line, const ConstantQ& o) {
  const std::int64_t samples = line.count;
  const double sfreq = line.spacing > 0 ? 1.0 / line.spacing : 0;
  if (samples < 1 || !(sfreq > 0)) throw std::invalid_argument("constant-Q: a line of " + std::to_string(samples) + " points " + std::to_string(line.spacing) + " apart");
  if (o.method != "hertz" && o.method != "cycles") throw std::invalid_argument("constant-Q: the method is hertz or cycles (got '" + o.method + "')");
  if (o.voices < 1) throw std::invalid_argument("constant-Q: voices is a whole number >= 1");
  const double nyquist = sfreq / 2;
  // the method's anchors: hertz takes the cycles of a power-of-two window at 8 samples a cycle and caps f_max at 64 Hz; cycles takes 8 cycles to Nyquist
  const bool hertz = o.method == "hertz";
  const double anchor_cycles = hertz ? std::pow(2.0, js_round(std::log2(8.0 * sfreq))) / sfreq : 8.0;
  const double anchor_fmax = hertz ? std::min(std::pow(2.0, std::floor(std::log2(nyquist))), 64.0) : nyquist;
  const double cycles = o.cycles.value_or(anchor_cycles);
  if (!(cycles > 0)) throw std::invalid_argument("constant-Q: cycles is > 0");
  const double f_max = std::min(o.f_max.value_or(anchor_fmax), nyquist);
  if (!(f_max > 0)) throw std::invalid_argument("constant-Q: f_max is > 0");
  const double f_min = std::max(o.f_min.value_or(hertz ? 0.5 : 0.0), 0.0);
  std::vector<Band> bands;
  for (int b = 0;; ++b) {                                              // down from f_max by 1/voices of an octave, while the window fits the line
    const double hi = f_max * std::pow(2.0, -static_cast<double>(b) / o.voices), lo = f_max * std::pow(2.0, -static_cast<double>(b + 1) / o.voices);
    if (lo < f_min) break;
    const int k = k_of(cycles * sfreq / lo);
    if ((std::int64_t(1) << k) > samples) break;
    bands.push_back(Band{lo, hi, k});
  }
  return bands;
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
Selection constant_q(const Line& line, const ConstantQ& o) {
  std::map<int, std::pair<double, double>> by_k;                       // level k: the union of its bands
  for (const Band& b : design(line, o)) {
    auto [it, fresh] = by_k.try_emplace(b.k, b.lo, b.hi);
    if (!fresh) { it->second.first = std::min(it->second.first, b.lo); it->second.second = std::max(it->second.second, b.hi); }
  }
  std::vector<Selection> levels;
  for (const auto& [k, band] : by_k) {
    const std::int64_t width = std::int64_t(1) << k;
    Selection level = tiled(Tiling{line.origin, static_cast<double>(width) * line.spacing, static_cast<double>(width) * line.spacing, (line.count + width - 1) / width}, line.unit);
    level.level = k;
    level.code = k;
    level.band = band;
    level.band_unit = "Hz";
    levels.push_back(std::move(level));
  }
  return set(std::move(levels));
}

Selection constant_q_bands(const Line& line, const ConstantQ& o) {
  std::vector<std::pair<double, double>> b;
  for (const Band& x : design(line, o)) b.emplace_back(x.lo, x.hi);
  return band_spans(std::move(b), "Hz");
}

// ── a surface, dyadic: the Studio's ladder per connected piece; level l pairs with lambda = 4 pi m / A (m = the level's tiles, A = the surface's area) ──
Selection dyadic(const Surface& s, const Dyadic& o) {
  need_surface(s, "dyadic");
  const bisection::Tree t = bisection::subjectTree(s.positions, s.vertices, s.faces, s.face_count, o.depth.value_or(-1));
  Selection p = labels(t.codes);
  p.unit = s.unit;
  p.depth = t.depth;
  double area = 0;
  for (double a : t.pieceArea) area += a;
  const auto pieces = static_cast<std::int64_t>(t.runs.size());
  for (int l = 0; l <= t.depth; ++l) {
    const std::int64_t tiles = pieces << l;
    Selection level;
    level.type = Type::set;
    level.level = l;
    level.code = l;
    level.band = std::make_pair(l == 0 ? 0.0 : kFourPi * static_cast<double>(tiles / 2) / area, kFourPi * static_cast<double>(tiles) / area);
    level.band_unit = "1/" + s.unit + "^2";                            // area in unit^2, so lambda = 4 pi m / A in 1/unit^2
    p.members.push_back(std::move(level));
  }
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
    throw std::invalid_argument("transfer: the source selection, both spheres and the source's faces are needed");
  if (o.from->type != Type::set || o.from->codes.empty() || static_cast<std::int64_t>(o.from_vertices) != static_cast<std::int64_t>(o.from->codes.size()))
    throw std::invalid_argument("transfer: the source is a set by codes on a surface of " + std::to_string(o.from_vertices) + " vertices");
  const auto from_runs = bisection::pieceRuns(o.from_faces, o.from_face_count, o.from_vertices);
  const auto runs = bisection::pieceRuns(s.faces, s.face_count, s.vertices);
  const bisection::Tree t = bisection::groupTreeFor(o.from_sphere, o.from->codes.data(), from_runs, o.from->depth, o.sphere, s.vertices, runs, o.from->depth);
  Selection p = *o.from;                                               // its levels (bands in the source's units) and lookup
  p.codes = t.codes;
  p.unit = s.unit;
  return p;
}

}  // namespace nxr::selection
