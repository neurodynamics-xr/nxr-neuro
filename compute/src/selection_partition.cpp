// nxr::selection — a partition of a manifold by a method (partition.h).
#include "nxr/selection/partition.h"

#include <algorithm>
#include <cmath>
#include <map>
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
}
}  // namespace

std::int64_t member(const Partition& p, int level, std::int64_t element) {
  if (element < 0 || element >= p.elements) throw std::invalid_argument("member: element " + std::to_string(element) + " outside 0.." + std::to_string(p.elements - 1));
  if (p.on == "line") {
    if (level < 0 || level > 62) throw std::invalid_argument("member: level " + std::to_string(level) + " outside 0..62");
    return element >> level;
  }
  if (level < 0 || level > p.depth) throw std::invalid_argument("member: level " + std::to_string(level) + " outside 0.." + std::to_string(p.depth));
  const std::int32_t code = p.codes[static_cast<std::size_t>(element)];
  return code < 0 ? -1 : code >> (p.depth - level);
}

// ── a line, by constant-Q: the band design of the Studio's constant-q.ts (constantQTiling, anchorsFor) ─────────────
Partition partition(const Line& line, const ConstantQ& o) {
  const std::int64_t samples = line.samples;
  const double sfreq = line.sfreq;
  if (samples < 1 || !(sfreq > 0)) throw std::invalid_argument("partition: a line of " + std::to_string(samples) + " samples at " + std::to_string(sfreq) + " Hz");
  if (o.method != "hertz" && o.method != "cycles") throw std::invalid_argument("partition: constant-Q's method is hertz or cycles (got '" + o.method + "')");
  if (o.voices < 1) throw std::invalid_argument("partition: voices is a whole number >= 1");
  const double nyquist = sfreq / 2;
  // the method's anchors: hertz takes the cycles of a power-of-two window at 8 samples a cycle and caps f_max at 64 Hz; cycles takes 8 cycles to Nyquist
  const bool hertz = o.method == "hertz";
  const double anchor_cycles = hertz ? std::pow(2.0, js_round(std::log2(8.0 * sfreq))) / sfreq : 8.0;
  const double anchor_fmax = hertz ? std::min(std::pow(2.0, std::floor(std::log2(nyquist))), 64.0) : nyquist;
  const double cycles = o.cycles.value_or(anchor_cycles);
  if (!(cycles > 0)) throw std::invalid_argument("partition: cycles is > 0");
  const double f_max = std::min(o.f_max.value_or(anchor_fmax), nyquist);
  if (!(f_max > 0)) throw std::invalid_argument("partition: f_max is > 0");
  const double f_min = std::max(o.f_min.value_or(hertz ? 0.5 : 0.0), 0.0);
  Partition p;
  p.on = "line";
  p.method = "constant-q";
  p.elements = samples;
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
    auto [it, fresh] = by_k.try_emplace(band.level, Level{band.level, (samples + (std::int64_t(1) << band.level) - 1) >> band.level, band.lo, band.hi});
    if (!fresh) { it->second.band_lo = std::min(it->second.band_lo, band.lo); it->second.band_hi = std::max(it->second.band_hi, band.hi); }
  }
  for (const auto& [k, level] : by_k) p.levels.push_back(level);
  return p;
}

// ── a surface, dyadic: the Studio's ladder per connected piece; level l pairs with lambda = 4 pi m / A (m = the level's tiles, A = the surface's area) ──
namespace {
std::vector<Level> weyl_levels(int depth, std::int64_t pieces, double area) {
  std::vector<Level> levels;
  for (int l = 0; l <= depth; ++l) {
    const std::int64_t count = pieces << l;
    levels.push_back(Level{l, count, l == 0 ? 0.0 : kFourPi * static_cast<double>(count / 2) / area, kFourPi * static_cast<double>(count) / area});
  }
  return levels;
}
}  // namespace

Partition partition(const Surface& s, const Dyadic& o) {
  need_surface(s, "partition (dyadic)");
  const dyadic::Tree t = dyadic::subjectTree(s.positions, s.vertices, s.faces, s.face_count, o.depth.value_or(-1));
  Partition p;
  p.on = "surface";
  p.method = "dyadic";
  p.elements = s.vertices;
  p.depth = t.depth;
  p.codes = t.codes;
  double area = 0;
  for (double a : t.pieceArea) area += a;
  p.levels = weyl_levels(t.depth, static_cast<std::int64_t>(t.runs.size()), area);
  return p;
}

// ── a surface, by transfer: the nearest vertex on the registered spheres, piece by piece (nxr geom.trees Template.tree_for) ─────────────
Partition partition(const Surface& s, const Transfer& o) {
  need_surface(s, "partition (transfer)");
  if (!o.from || !o.from_sphere || !o.from_faces || !o.sphere || o.from_vertices < 1) throw std::invalid_argument("partition (transfer): the source partition, both spheres and the source's faces are needed");
  if (o.from->on != "surface" || static_cast<std::int64_t>(o.from_vertices) != o.from->elements)
    throw std::invalid_argument("partition (transfer): the source is a partition of a surface of " + std::to_string(o.from_vertices) + " vertices");
  const auto from_runs = dyadic::pieceRuns(o.from_faces, o.from_face_count, o.from_vertices);
  const auto runs = dyadic::pieceRuns(s.faces, s.face_count, s.vertices);
  const dyadic::Tree t = dyadic::groupTreeFor(o.from_sphere, o.from->codes.data(), from_runs, o.from->depth, o.sphere, s.vertices, runs, o.from->depth);
  Partition p;
  p.on = "surface";
  p.method = "transfer";
  p.elements = s.vertices;
  p.depth = o.from->depth;
  p.codes = t.codes;
  p.levels = o.from->levels;   // the source's members, so its levels (and their bands, the source's)
  return p;
}

}  // namespace nxr::selection
