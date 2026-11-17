// nxr::selection (compute) — a partition of a manifold by a method (partition.h).
#include "nxr/selection/partition.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include "nxr/selection/dyadic.h"

namespace nxr::selection {

namespace {
double js_round(double x) { return std::floor(x + 0.5); }            // the Studio's Math.round: halves toward +infinity
int k_of(double samples) { return static_cast<int>(std::lround(std::ceil(std::log2(samples)))); }   // the power of two a window is rounded up to
constexpr double kFourPi = 4 * M_PI;

void need_surface(const Surface& s, const char* what) {
  if (!s.positions || !s.faces || s.vertices < 1 || s.face_count < 1) throw std::invalid_argument(std::string(what) + ": a surface needs positions and faces");
  if (s.unit.empty()) throw std::invalid_argument(std::string(what) + ": a surface needs its positions' unit (m, mm, …)");
}
const Level* level_of(const Partition& p, int level) {
  for (const Level& l : p.levels) if (l.level == level) return &l;
  return nullptr;
}
std::string number(double x) { char b[32]; std::snprintf(b, sizeof b, "%.4g", x); return b; }
}  // namespace

// ── members ──────────────────────────────────────────────────────────────────────────────
std::int64_t member(const Partition& p, int level, std::int64_t element) {
  if (p.elements == 0) throw std::invalid_argument("member: this partition divides a line's values, not elements (member_at)");
  if (element < 0 || element >= p.elements) throw std::invalid_argument("member: element " + std::to_string(element) + " outside 0.." + std::to_string(p.elements - 1));
  if (!p.codes.empty()) {
    if (level < 0 || level > p.depth) throw std::invalid_argument("member: level " + std::to_string(level) + " outside 0.." + std::to_string(p.depth));
    const std::int32_t code = p.codes[static_cast<std::size_t>(element)];
    return code < 0 ? -1 : code >> (p.depth - level);
  }
  if (level < 0 || level > 62) throw std::invalid_argument("member: level " + std::to_string(level) + " outside 0..62");
  return element >> level;                                            // a line's tiles of 2^level points
}

std::int64_t member_at(const Partition& p, int level, double value) {
  if (const Level* l = level_of(p, level); l && !l->edges.empty()) {
    if (!(value >= l->edges.front()) || !(value < l->edges.back())) return -1;
    return static_cast<std::int64_t>(std::upper_bound(l->edges.begin(), l->edges.end(), value) - l->edges.begin()) - 1;
  }
  if (!p.codes.empty() || !(p.spacing > 0)) throw std::invalid_argument("member_at: level " + std::to_string(level) + " is not a partition of a line");
  const double i = std::floor((value - p.origin) / p.spacing);
  if (!(i >= 0) || i >= static_cast<double>(p.elements)) return -1;
  return member(p, level, static_cast<std::int64_t>(i));
}

// ── a time line, by constant-Q: the band design of the Studio's constant-q.ts (constantQTiling, anchorsFor) ─────────────
Partition partition(const Line& line, const ConstantQ& o) {
  if (line.quantity != "time") throw std::invalid_argument("partition (constant-Q): a time line, not " + line.quantity);
  const std::int64_t samples = line.count;
  const double sfreq = line.spacing > 0 ? 1.0 / line.spacing : 0;
  if (samples < 1 || !(sfreq > 0)) throw std::invalid_argument("partition (constant-Q): a line of " + std::to_string(samples) + " points " + std::to_string(line.spacing) + " apart");
  if (o.method != "hertz" && o.method != "cycles") throw std::invalid_argument("partition (constant-Q): the method is hertz or cycles (got '" + o.method + "')");
  if (o.voices < 1) throw std::invalid_argument("partition (constant-Q): voices is a whole number >= 1");
  const double nyquist = sfreq / 2;
  // the method's anchors: hertz takes the cycles of a power-of-two window at 8 samples a cycle and caps f_max at 64 Hz; cycles takes 8 cycles to Nyquist
  const bool hertz = o.method == "hertz";
  const double anchor_cycles = hertz ? std::pow(2.0, js_round(std::log2(8.0 * sfreq))) / sfreq : 8.0;
  const double anchor_fmax = hertz ? std::min(std::pow(2.0, std::floor(std::log2(nyquist))), 64.0) : nyquist;
  const double cycles = o.cycles.value_or(anchor_cycles);
  if (!(cycles > 0)) throw std::invalid_argument("partition (constant-Q): cycles is > 0");
  const double f_max = std::min(o.f_max.value_or(anchor_fmax), nyquist);
  if (!(f_max > 0)) throw std::invalid_argument("partition (constant-Q): f_max is > 0");
  const double f_min = std::max(o.f_min.value_or(hertz ? 0.5 : 0.0), 0.0);
  Partition p;
  p.on = "time"; p.unit = line.unit; p.method = "constant-q";
  p.elements = samples; p.origin = line.origin; p.spacing = line.spacing;
  p.spectrum = "frequency"; p.spectrum_unit = "Hz";
  // the bands, down from f_max by 1/voices of an octave, while their window fits the line
  for (int b = 0;; ++b) {
    const double hi = f_max * std::pow(2.0, -static_cast<double>(b) / o.voices), lo = f_max * std::pow(2.0, -static_cast<double>(b + 1) / o.voices);
    if (lo < f_min) break;
    const int k = k_of(cycles * sfreq / lo);
    if ((std::int64_t(1) << k) > samples) break;
    p.bands.push_back(Band{lo, hi, k});
  }
  // the levels: one a distinct k, resolving the union of its bands
  std::map<int, Level> by_k;
  for (const Band& band : p.bands) {
    auto [it, fresh] = by_k.try_emplace(band.level, Level{band.level, (samples + (std::int64_t(1) << band.level) - 1) >> band.level, band.lo, band.hi, {}});
    if (!fresh) { it->second.band_lo = std::min(it->second.band_lo, band.lo); it->second.band_hi = std::max(it->second.band_hi, band.hi); }
  }
  for (const auto& [k, level] : by_k) p.levels.push_back(level);
  return p;
}

// ── a surface, dyadic: the Studio's ladder per connected piece; level l pairs with lambda = 4 pi m / A (m = the level's tiles, A = the surface's area) ──
Partition partition(const Surface& s, const Dyadic& o) {
  need_surface(s, "partition (dyadic)");
  const dyadic::Tree t = dyadic::subjectTree(s.positions, s.vertices, s.faces, s.face_count, o.depth.value_or(-1));
  Partition p;
  p.on = "surface"; p.unit = s.unit; p.method = "dyadic";
  p.elements = s.vertices;
  p.depth = t.depth;
  p.codes = t.codes;
  p.spectrum = "eigenvalue";
  p.spectrum_unit = "1/" + s.unit + "^2";                              // area in unit^2, so lambda = 4 pi m / A in 1/unit^2
  double area = 0;
  for (double a : t.pieceArea) area += a;
  const auto pieces = static_cast<std::int64_t>(t.runs.size());
  for (int l = 0; l <= t.depth; ++l) {
    const std::int64_t count = pieces << l;
    const double lo = l == 0 ? 0.0 : kFourPi * static_cast<double>(count / 2) / area, hi = kFourPi * static_cast<double>(count) / area;
    p.levels.push_back(Level{l, count, lo, hi, {}});
  }
  for (auto it = p.levels.rbegin(); it != p.levels.rend(); ++it) p.bands.push_back(Band{it->band_lo, it->band_hi, it->level});
  return p;
}

// ── a surface, by transfer: the nearest vertex on the registered spheres, piece by piece (nxr geom.trees Template.tree_for) ─────────────
Partition partition(const Surface& s, const Transfer& o) {
  need_surface(s, "partition (transfer)");
  if (!o.from || !o.from_sphere || !o.from_faces || !o.sphere || o.from_vertices < 1)
    throw std::invalid_argument("partition (transfer): the source partition, both spheres and the source's faces are needed");
  if (o.from->codes.empty() || static_cast<std::int64_t>(o.from_vertices) != o.from->elements)
    throw std::invalid_argument("partition (transfer): the source is a partition by codes of a surface of " + std::to_string(o.from_vertices) + " vertices");
  const auto from_runs = dyadic::pieceRuns(o.from_faces, o.from_face_count, o.from_vertices);
  const auto runs = dyadic::pieceRuns(s.faces, s.face_count, s.vertices);
  const dyadic::Tree t = dyadic::groupTreeFor(o.from_sphere, o.from->codes.data(), from_runs, o.from->depth, o.sphere, s.vertices, runs, o.from->depth);
  Partition p = *o.from;                                               // the source's members: its levels, bands (in the source's units) and lookup
  p.method = "transfer";
  p.unit = s.unit;
  p.elements = s.vertices;
  p.codes = t.codes;
  return p;
}

// ── labels: a code per element and its lookup ────────────────────────────────────────────
Partition labels(std::string on, std::vector<std::int32_t> codes, Lookup lookup) {
  check(lookup);
  std::set<std::int32_t> seen;
  for (std::int32_t c : codes) {
    if (c < 0) continue;
    if (seen.insert(c).second && !lookup.empty()) entry(lookup, c);   // throws for a code the lookup lacks
  }
  Partition p;
  p.on = std::move(on); p.method = "labels";
  p.elements = static_cast<std::int64_t>(codes.size());
  p.codes = std::move(codes);
  p.lookup = std::move(lookup);
  p.levels.push_back(Level{0, static_cast<std::int64_t>(seen.size()), 0, 0, {}});
  return p;
}

// ── the spectrum's partition: the bands' line divided at their edges ──────────────────────
Partition spectrum_partition(const Partition& src) {
  if (src.bands.empty()) throw std::invalid_argument("spectrum_partition: the " + src.method + " partition has no bands");
  std::vector<Band> b = src.bands;
  std::sort(b.begin(), b.end(), [](const Band& x, const Band& y) { return x.lo < y.lo; });
  Level level{0, static_cast<std::int64_t>(b.size()), b.front().lo, b.back().hi, {b.front().lo}};
  for (std::size_t i = 0; i < b.size(); ++i) {
    if (i + 1 < b.size() && std::abs(b[i].hi - b[i + 1].lo) > 1e-9 * std::max(1.0, std::abs(b[i].hi)))
      throw std::invalid_argument("spectrum_partition: the bands leave a gap or overlap at " + number(b[i].hi) + " " + src.spectrum_unit);
    level.edges.push_back(b[i].hi);
  }
  Partition p;
  p.on = src.spectrum; p.unit = src.spectrum_unit; p.method = "spectrum";
  p.levels.push_back(level);
  for (std::size_t i = 0; i < b.size(); ++i)
    p.lookup.push_back(Entry{static_cast<std::int64_t>(i), number(b[i].lo) + "-" + number(b[i].hi) + (src.spectrum_unit.empty() ? "" : " " + src.spectrum_unit), std::nullopt});
  return p;
}

}  // namespace nxr::selection
