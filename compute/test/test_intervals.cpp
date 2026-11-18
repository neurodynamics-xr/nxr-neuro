// nxr::intervals on synthetic series only: hysteresis crossings (enter high, extend low; a plain threshold), merging by gap, keeping by length,
// negative half-waves with their troughs, the median and robust SD; a planted burst found from its envelope; refusals.
#include <cmath>
#include <iostream>
#include <random>

#include "nxr/select/events.h"
#include "nxr/measurement/roll_up.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)
using namespace nxr::select;

int main() {
  std::cout << "test_intervals\n";
  Eigen::VectorXd x(12);
  x << 0, 1, 3, 1.5, 1, 0, 1.5, 1.8, 0, 2.5, 2.5, 0;
  const auto h = crossings(x, 2.0, 1.0);                     // enter at 2, extend while >= 1: [1, 5) (peak 3 at 2) and [9, 11); [6, 8) never reaches 2
  EXPECT(h.size() == 2 && h[0].start == 1 && h[0].stop == 5 && h[0].peak == 2 && h[0].value == 3 && h[1].start == 9 && h[1].stop == 11,
         "crossings with hysteresis: enter high, extend while above the low level, a span never reaching the high level dropped");
  EXPECT(crossings(x, 1.5, 1.5).size() == 3, "entry = exit: a plain threshold ([2, 4), [6, 8), [9, 11))");
  const auto m = mergeSpans({{0, 5, 2, 3}, {7, 9, 8, 4}, {20, 22, 21, 1}}, 3);
  EXPECT(m.size() == 2 && m[0].stop == 9 && m[0].peak == 8 && m[0].value == 4 && m[1].start == 20, "mergeSpans joins spans closer than the gap, keeping the larger peak");
  const auto k = keepSpans(m, 3, 10);
  EXPECT(k.size() == 1 && k[0].start == 0, "keepSpans by length");
  Eigen::VectorXd w(10);
  w << 1, -1, -3, -2, 1, 2, -1, 1, -5, -4;
  const auto hw = halfWaves(w, 2, 5);                         // [1, 4) trough −3 at 2; [6, 7) too short; [8, 10) runs off the end
  EXPECT(hw.size() == 1 && hw[0].start == 1 && hw[0].stop == 4 && hw[0].peak == 2 && hw[0].value == -3, "halfWaves: from a downward to an upward zero crossing, the trough, by length");
  Eigen::VectorXd r(7);
  r << 1, 2, 3, 4, 100, 2, 3;
  EXPECT(median(r) == 3 && std::abs(mad(r) - 1.4826) < 1e-12, "median and the robust SD (1.4826 MAD), unmoved by the outlier");
  bool refused = false;
  try { crossings(x, 1.0, 2.0); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "an exit level above the entry level is refused");
  // a planted 10 Hz burst (2–4 s) in noise at 200 Hz: its amplitude, smoothed over a cycle, crosses median + 1.5 robust SD and is found to within a cycle
  const double fs = 200; const int n = 1200;
  std::mt19937 rng(2); std::normal_distribution<double> nd(0, 0.3);
  Eigen::VectorXd a(n);
  for (int t = 0; t < n; ++t) a(t) = std::abs((t >= 400 && t < 800 ? 2.0 : 0.0) + nd(rng));
  Eigen::VectorXd s = Eigen::VectorXd::Zero(n);
  const int c = static_cast<int>(fs / 10);
  for (int t = 0; t < n; ++t) { int lo = std::max(0, t - c / 2), hi = std::min(n, t + c / 2 + 1); s(t) = a.segment(lo, hi - lo).mean(); }
  const double md = median(s), sd = mad(s);
  const auto b = keepSpans(mergeSpans(crossings(s, md + 1.5 * sd, md + 1.0 * sd), c), 3 * c, n);
  std::printf("  planted burst [400, 800): found %zu span(s)%s\n", b.size(), b.empty() ? "" : (" [" + std::to_string(b[0].start) + ", " + std::to_string(b[0].stop) + ")").c_str());
  EXPECT(b.size() == 1 && std::abs(b[0].start - 400) <= c && std::abs(b[0].stop - 800) <= c, "a planted burst found from its smoothed amplitude, to within a cycle");
  {   // THE ROLL-UP OF TILES OVER EVENTS: a measurement of tiles over the tiles of each event, by the measure's merge rule (nxr::measurement::roll_values: the count-weighted mean, max, sum), NaN for a bad tile
    Eigen::MatrixXd t(2, 6);
    t << 1, 2, 3, 4, 5, 6,
         10, std::nan(""), 30, 40, 50, 60;
    Eigen::MatrixXd n(1, 6);
    n << 100, 100, 50, 100, 100, 100;
    nxr::selection::Elements ev;                       // the events' tiles: {1, 2, 3}, {1, 2, 3} (on row 1), {3, 4, 5}
    ev.size = 6;
    ev.elements = {1, 2, 3, 1, 2, 3, 3, 4, 5};
    ev.offsets = {0, 3, 6, 9};
    const nxr::measurement::Registry& reg = nxr::measurement::Registry::standard();
    const Eigen::MatrixXd m = nxr::measurement::roll_values(reg, "mean", t, &n, ev);
    EXPECT(m.rows() == 2 && m.cols() == 3, "[rows x events]");
    EXPECT(std::abs(m(0, 0) - (2.0 * 100 + 3 * 50 + 4 * 100) / 250) < 1e-12, "an event's value: the count-weighted mean of its tiles");
    EXPECT(std::isnan(m(1, 1)), "an event holding a NaN tile is NaN on its row");
    EXPECT(std::abs(m(0, 2) - (4.0 + 5 + 6) / 3) < 1e-12 && std::abs(m(1, 2) - 50.0) < 1e-12, "an event found on the mean over the rows holds on every row");
    const Eigen::MatrixXd cnt = nxr::measurement::roll_values(reg, "count", n, nullptr, ev);
    EXPECT(cnt(0, 0) == 250 && cnt(0, 2) == 300, "the samples over the tiles: the count adds");
    const Eigen::MatrixXd mx = nxr::measurement::roll_values(reg, "max", t, nullptr, ev);
    EXPECT(mx(0, 0) == 4 && mx(1, 2) == 60 && mx(0, 2) == 6, "max: the max of the tile maxima");
    bool bad = false;
    try { nxr::measurement::roll_values(reg, "median", t, nullptr, ev); } catch (const std::invalid_argument&) { bad = true; }
    EXPECT(bad, "a measure that never rolls up (a median) is refused");
    const std::vector<TileEvent> g = {{0, 2, 0}, {5, 6, 0}, {3, 4, 1}, {9, 12, 0}};
    const auto du = eventDurations(g, 0.5);
    EXPECT(du == (std::vector<double>{1.0, 0.5, 0.5, 1.5}), "duration = tiles x width");
    const auto in = eventIntervalNext(g, 0.5);
    EXPECT(std::abs(in[0] - 1.5) < 1e-12 && std::abs(in[1] - 1.5) < 1e-12 && std::isnan(in[2]) && std::isnan(in[3]), "the next event on the same row minus this stop; NaN for the last on a row");
    const auto any = eventIntervalNext({{0, 2, -1}, {3, 4, -1}}, 1.0);
    EXPECT(any[0] == 1.0 && std::isnan(any[1]), "row -1: the next event of any row");
  }
  std::cout << (failures ? "FAIL" : "PASS") << " test_intervals (" << failures << " failures)\n";
  return failures ? 1 : 0;
}
