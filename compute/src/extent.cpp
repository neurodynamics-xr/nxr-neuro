// nxr::extent — the space × time extent of an event on the joint cells (extent.h).
#include "nxr/extent.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>

#include "nxr/ladder.h"

namespace nxr::extent {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}

int bitsFor(std::int64_t frames) {
  if (frames < 1) throw std::invalid_argument("extent: " + std::to_string(frames) + " time tiles: at least one");
  int b = 1;
  while ((std::int64_t(1) << b) < std::max<std::int64_t>(2, frames)) ++b;
  return b;
}

std::int32_t cellCode(std::int64_t tile, std::int64_t frame, int bits, std::int64_t frames) {
  if (tile < 0) throw std::invalid_argument("extent: a cell's tile is " + std::to_string(tile) + ", at least 0");
  if (frame < 0 || frame >= frames) throw std::invalid_argument("extent: a cell's time tile is " + std::to_string(frame) + ", the layout has " + std::to_string(frames) + " (0 … " + std::to_string(frames - 1) + ")");
  if (bits < 1 || (std::int64_t(1) << bits) < frames) throw std::invalid_argument("extent: " + std::to_string(bits) + " bits cannot hold " + std::to_string(frames) + " time tiles");
  const std::int64_t code = (tile << bits) | frame;
  if (code >= (std::int64_t(1) << 31)) throw std::invalid_argument("extent: tile " + std::to_string(tile) + " with " + std::to_string(bits) + " time bits does not fit a 32-bit code");
  return static_cast<std::int32_t>(code);
}

Cell cellOf(std::int32_t code, int bits, std::int64_t frames) {
  if (code < 0) throw std::invalid_argument("extent: a cell's code is " + std::to_string(code) + ", at least 0 (a negative code is in no event)");
  if (bits < 1 || bits > 30) throw std::invalid_argument("extent: " + std::to_string(bits) + " time bits");
  const std::int32_t frame = code & ((std::int32_t(1) << bits) - 1);
  if (frame >= frames) throw std::invalid_argument("extent: code " + std::to_string(code) + " holds time tile " + std::to_string(frame) + ", the layout has " + std::to_string(frames) + " (a gap of the layout)");
  return Cell{code >> bits, frame};
}

Pairs spaceTimePairs(const Pairs& tilePairs, int tiles, int frames) {
  for (const auto& [a, b] : tilePairs)
    if (a < 0 || b < 0 || a >= tiles || b >= tiles) throw std::invalid_argument("extent: a tile pair (" + std::to_string(a) + ", " + std::to_string(b) + ") outside 0 … " + std::to_string(tiles - 1));
  Pairs out;
  out.reserve(static_cast<std::size_t>(frames) * tilePairs.size() + static_cast<std::size_t>(std::max(0, frames - 1)) * static_cast<std::size_t>(tiles));
  for (int f = 0; f < frames; ++f) {
    const std::int32_t o = f * tiles;
    for (const auto& [a, b] : tilePairs) out.emplace_back(o + a, o + b);
    if (f + 1 < frames) for (int s = 0; s < tiles; ++s) out.emplace_back(o + s, o + tiles + s);
  }
  return out;
}

Eigen::MatrixXd standardize(const Eigen::MatrixXd& amplitude, const Eigen::VectorXd& median, const Eigen::VectorXd& sd, int* skipped) {
  if (median.size() != amplitude.rows() || sd.size() != amplitude.rows()) throw std::invalid_argument("extent: the baseline has " + std::to_string(median.size()) + " tiles, the cells " + std::to_string(amplitude.rows()));
  Eigen::MatrixXd z(amplitude.rows(), amplitude.cols());
  int none = 0;
  for (Eigen::Index s = 0; s < amplitude.rows(); ++s) {
    const bool ok = std::isfinite(sd(s)) && sd(s) > 0 && std::isfinite(median(s));
    if (!ok) ++none;
    for (Eigen::Index f = 0; f < amplitude.cols(); ++f) z(s, f) = ok && std::isfinite(amplitude(s, f)) ? (amplitude(s, f) - median(s)) / sd(s) : kNaN;
  }
  if (skipped) *skipped = none;
  return z;
}

WindowRegions windowRegions(const Eigen::MatrixXd& z, const Pairs& tilePairs, const Eigen::VectorXd& tileArea, double enter, double exit, int minCells, double minArea) {
  const int S = static_cast<int>(z.rows()), F = static_cast<int>(z.cols());
  if (tileArea.size() != S) throw std::invalid_argument("extent: " + std::to_string(tileArea.size()) + " tile areas for " + std::to_string(S) + " tiles");
  if (minCells < 1) throw std::invalid_argument("extent: minCells is at least 1");
  WindowRegions out;
  out.label.assign(static_cast<std::size_t>(S) * static_cast<std::size_t>(F), -1);
  if (S == 0 || F == 0) return out;
  const Eigen::VectorXd values = Eigen::Map<const Eigen::VectorXd>(z.data(), z.size());
  Eigen::VectorXd w(z.size());
  for (int f = 0; f < F; ++f) w.segment(static_cast<Eigen::Index>(f) * S, S) = tileArea;
  const regions::Regions R = regions::detectRegions(values, spaceTimePairs(tilePairs, S, F), w, enter, exit, minArea);
  for (const auto& g : R.regions) {
    if (static_cast<int>(g.nodes.size()) < minCells) continue;
    const std::int32_t k = static_cast<std::int32_t>(out.regions.size());
    out.regions.push_back(g);
    for (const auto n : g.nodes) out.label[static_cast<std::size_t>(n)] = k;
  }
  return out;
}

Rollup rollup(const Labels& label, int events, const Eigen::VectorXd& tileArea, const Eigen::VectorXd& samples, const Eigen::MatrixXd* amplitude, const Eigen::MatrixXd* peak) {
  const Eigen::Index S = label.rows(), T = label.cols();
  if (tileArea.size() != S || samples.size() != T) throw std::invalid_argument("extent: the areas or the sample counts do not match the labelling");
  if ((amplitude && (amplitude->rows() != S || amplitude->cols() != T)) || (peak && (peak->rows() != S || peak->cols() != T))) throw std::invalid_argument("extent: a cell measurement does not match the labelling");
  Rollup r;
  const Eigen::Index E = events;
  r.cells = Eigen::VectorXi::Zero(E); r.tiles = Eigen::VectorXi::Zero(E); r.first = Eigen::VectorXi::Constant(E, -1); r.last = Eigen::VectorXi::Constant(E, -1);
  r.area = Eigen::VectorXd::Zero(E); r.weight = Eigen::VectorXd::Zero(E);
  r.amplitude = Eigen::VectorXd::Constant(E, amplitude ? 0.0 : kNaN); r.peak = Eigen::VectorXd::Constant(E, kNaN);
  std::vector<std::vector<char>> seen(static_cast<std::size_t>(E), std::vector<char>(static_cast<std::size_t>(S), 0));
  for (Eigen::Index f = 0; f < T; ++f)
    for (Eigen::Index s = 0; s < S; ++s) {
      const std::int32_t e = label(s, f);
      if (e < 0) continue;
      if (e >= events) throw std::invalid_argument("extent: a cell holds event " + std::to_string(e) + " of " + std::to_string(events));
      ++r.cells(e);
      if (r.first(e) < 0) r.first(e) = static_cast<int>(f);
      r.last(e) = static_cast<int>(f);
      if (!seen[static_cast<std::size_t>(e)][static_cast<std::size_t>(s)]) { seen[static_cast<std::size_t>(e)][static_cast<std::size_t>(s)] = 1; ++r.tiles(e); r.area(e) += tileArea(s); }
      const double w = tileArea(s) * samples(f);
      r.weight(e) += w;
      if (amplitude) r.amplitude(e) += w * (*amplitude)(s, f);
      if (peak) r.peak(e) = std::isnan(r.peak(e)) ? (*peak)(s, f) : std::max(r.peak(e), (*peak)(s, f));
    }
  if (amplitude) for (Eigen::Index e = 0; e < E; ++e) r.amplitude(e) = r.weight(e) > 0 ? r.amplitude(e) / r.weight(e) : kNaN;
  return r;
}

Eigen::MatrixXd areaByFrame(const Labels& label, int events, const Eigen::VectorXd& tileArea) {
  Eigen::MatrixXd a = Eigen::MatrixXd::Constant(events, label.cols(), kNaN);
  for (Eigen::Index f = 0; f < label.cols(); ++f)
    for (Eigen::Index s = 0; s < label.rows(); ++s) {
      const std::int32_t e = label(s, f);
      if (e < 0) continue;
      a(e, f) = std::isnan(a(e, f)) ? tileArea(s) : a(e, f) + tileArea(s);
    }
  return a;
}

Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic> centres(const Labels& label, int events, const std::vector<std::int32_t>& tileOfVertex, const std::vector<double>& positions,
                                                                   const std::vector<double>& vertexArea, const std::vector<int>& frames) {
  const Eigen::Index T = label.cols();
  Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic> c = Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>::Constant(events, T, -1);
  std::vector<int> todo = frames;
  if (todo.empty()) for (Eigen::Index f = 0; f < T; ++f) todo.push_back(static_cast<int>(f));
  const std::size_t V = tileOfVertex.size();
  std::vector<std::int32_t> node(V);
  for (const int f : todo) {
    if (f < 0 || f >= T) throw std::invalid_argument("extent: time tile " + std::to_string(f) + " outside the labelling");
    bool any = false;
    for (Eigen::Index s = 0; s < label.rows(); ++s) any = any || label(s, f) >= 0;
    if (!any) continue;
    for (std::size_t v = 0; v < V; ++v) node[v] = tileOfVertex[v] >= 0 ? label(tileOfVertex[v], f) : -1;
    const std::vector<int> ce = ladder::tileCentres(positions.data(), node, events, vertexArea);
    for (int e = 0; e < events; ++e) c(e, f) = ce[static_cast<std::size_t>(e)];
  }
  return c;
}

Eigen::VectorXd pathLength(const Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>& centres, const std::vector<double>& positions) {
  Eigen::VectorXd len = Eigen::VectorXd::Zero(centres.rows());
  for (Eigen::Index e = 0; e < centres.rows(); ++e) {
    Eigen::Index first = -1, last = -1;
    for (Eigen::Index f = 0; f < centres.cols(); ++f) if (centres(e, f) >= 0) { if (first < 0) first = f; last = f; }
    if (first < 0) { len(e) = kNaN; continue; }
    for (Eigen::Index f = first; f < last; ++f) {
      const std::int32_t a = centres(e, f), b = centres(e, f + 1);
      if (a < 0 || b < 0) { len(e) = kNaN; break; }
      double d2 = 0;
      for (int k = 0; k < 3; ++k) { const double d = positions[static_cast<std::size_t>(a) * 3 + k] - positions[static_cast<std::size_t>(b) * 3 + k]; d2 += d * d; }
      len(e) += std::sqrt(d2);
    }
  }
  return len;
}

}  // namespace nxr::extent
