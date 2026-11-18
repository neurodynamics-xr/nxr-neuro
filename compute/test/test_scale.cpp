// nxr::select::scale_selection on hand-made pairs only: the ladders' nesting (a tile's ancestor by a shift, equal levels, the hemisphere bit, a sibling does not nest), the time overlap of windows of different sizes and origins, the facts of an event
// (cells, peak, mean, first and last sample), the links across pairs (nesting AND overlap needed; sibling tiles, disjoint times, the same pair never link directly; min_link), the components numbered by their earliest sample,
// the footprint (the largest of an event's members at a pair, NaN where none), the best pair (the largest response; a tie to the first), the mean response, and the refusals.
#include <cmath>
#include <iostream>

#include "nxr/select/scale_selection.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)
using namespace nxr::select::scale_selection;

template <class F> bool throws(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

// a pair of S tiles of `level` and T time tiles, all cells NaN and unlabelled
static Pair make(int level, std::int64_t window, int S, int T, std::int64_t origin = 0) {
  Pair p;
  p.level = level; p.window = window; p.origin = origin;
  p.z = Eigen::MatrixXd::Constant(S, T, std::nan(""));
  p.label = Labels::Constant(S, T, -1);
  return p;
}
static void put(Pair& p, int event, int tile, int frame, double z) { p.label(tile, frame) = event; p.z(tile, frame) = z; p.events = std::max(p.events, event + 1); }

int main() {
  std::cout << "test_scale\n";
  {   // NESTING: the ancestor of tile s of level l at level l' is s >> (l - l'); the hemisphere bit (code h 2^l + t) goes with it
    EXPECT(nests(5, 3, 5, 3) && !nests(5, 3, 4, 3), "the same level: the same tile");
    EXPECT(nests(13, 4, 3, 2) && nests(3, 2, 13, 4) && !nests(12 + 4, 4, 3, 2), "tile 13 of level 4 lies in tile 3 of level 2 (both orders); tile 16 of level 4 is in tile 4 of level 2, not 3");
    EXPECT(nests(13, 3, 6, 2) && !nests(13, 3, 5, 2) && nests(13, 3, 3, 1) && nests(13, 3, 1, 0), "13 = hemisphere 1 tile 5 of level 3: its parents are 6 (level 2), 3 (level 1), 1 (level 0)");
    EXPECT(throws([] { nests(-1, 2, 0, 2); }) && throws([] { nests(0, -1, 0, 2); }), "a negative tile or level is refused");
  }
  {   // OVERLAP of time windows
    EXPECT(overlaps(0, 4, 0, 0, 4, 0) && !overlaps(0, 4, 0, 1, 4, 0), "equal windows overlap when equal");
    EXPECT(overlaps(3, 4, 0, 1, 8, 0) && overlaps(2, 4, 0, 1, 8, 0) && !overlaps(1, 4, 0, 1, 8, 0) && !overlaps(4, 4, 0, 1, 8, 0), "windows of 4 and 8: window 1 of 8 is [8, 16): windows 2 and 3 of 4 are in it, 1 and 4 are not");
    EXPECT(overlaps(0, 4, 2, 0, 4, 0) && !overlaps(0, 4, 4, 0, 4, 0), "an origin shifts the window");
    EXPECT(throws([] { overlaps(0, 0, 0, 0, 4, 0); }), "a window of no sample is refused");
  }
  // THREE PAIRS. A: level 2 (4 tiles a piece, 8 in all), windows of 4 samples, 6 time tiles. B: level 3 (16 tiles), windows of 4. C: level 2, windows of 8, 3 time tiles.
  Pair A = make(2, 4, 8, 6), B = make(3, 4, 16, 6), C = make(2, 8, 8, 3);
  // event 0 of A: tile 1 at time tiles 1, 2 (samples 4 … 12); an unrelated event 1: tile 6 at time tile 5 (samples 20 … 24)
  put(A, 0, 1, 1, 3.0); put(A, 0, 1, 2, 5.0); put(A, 1, 6, 5, 4.0);
  // B: an event 0 on tiles 2 and 3 (children of tile 1) at time tiles 2, 3 (samples 8 … 16): overlaps A's event 0 at sample 8-12 ; an event 1 on tile 13 (child of tile 6 of A: 13 >> 1 = 6) at time tile 0 (samples 0 … 4): NOT in time with A's event 1
  put(B, 0, 2, 2, 7.0); put(B, 0, 3, 3, 6.0); put(B, 1, 13, 0, 9.0);
  // C: tile 1 at time tile 1 (samples 8 … 16): overlaps A's event 0 in time (A's cells at 4 … 12) and nests (same tile); also B's event 0; and an event 1 on tile 6, time tile 2 (samples 16 … 24): overlaps A's event 1 (20 … 24)
  put(C, 0, 1, 1, 2.0); put(C, 1, 6, 2, 1.0);
  {
    const auto f = eventFacts(A);
    EXPECT(f.size() == 2 && f[0].cells == 2 && f[0].peak == 5.0 && f[0].mean == 4.0 && f[0].first == 4 && f[0].stop == 12 && f[1].cells == 1 && f[1].first == 20 && f[1].stop == 24, "the facts of A's events: cells, peak, mean, first and last sample");
    Pair O = A; O.origin = 100;
    EXPECT(eventFacts(O)[0].first == 104 && eventFacts(O)[0].stop == 112, "the origin shifts the samples");
  }
  const auto s = selectScale({A, B, C});
  {
    // expected cross events: X = A0 + B0 + C0 (nest + overlap); Y = A1 + C1 (tile 6 level 2, A's time 20-24 overlaps C's 16-24); B1 is alone (time 0-4 vs A1 20-24: none; vs C: C0 samples 8-16, C1 16-24: none)
    EXPECT(s.events == 3, "three cross-pair events (A0 B0 C0), (A1 C1), (B1)");
    EXPECT(s.cross[0][0] == s.cross[1][0] && s.cross[0][0] == s.cross[2][0] && s.cross[0][1] == s.cross[2][1] && s.cross[0][0] != s.cross[0][1] && s.cross[1][1] != s.cross[0][0] && s.cross[1][1] != s.cross[0][1], "the members are linked as planted");
    EXPECT(s.cross[1][1] == 0 && s.cross[0][0] == 1 && s.cross[0][1] == 2, "numbered by the earliest sample: B1 (sample 0), then A0 (4), then A1 (20)");
    const int x = s.cross[0][0], y = s.cross[0][1], w = s.cross[1][1];
    EXPECT(s.response(0, x) == 5.0 && s.response(1, x) == 7.0 && s.response(2, x) == 2.0, "the footprint of X: its peak at A, B, C");
    EXPECT(s.best(x) == 1 && s.peak(x) == 7.0 && s.pairs(x) == 3, "X's best pair is B (7.0), three pairs hold a member");
    EXPECT(s.response(0, y) == 4.0 && std::isnan(s.response(1, y)) && s.response(2, y) == 1.0 && s.best(y) == 0 && s.pairs(y) == 2, "Y: A and C, none at B (NaN), best A");
    EXPECT(s.best(w) == 1 && s.peak(w) == 9.0 && s.pairs(w) == 1 && std::isnan(s.response(0, w)), "the lone event is its own cross-pair event at its own pair");
    EXPECT(s.start(x) == 4 && s.stop(x) == 16 && s.start(y) == 16 && s.stop(y) == 24, "the span in samples: the union of the members' time tiles");
    EXPECT(s.members[static_cast<std::size_t>(x)].size() == 3 && s.members[static_cast<std::size_t>(x)][0] == std::make_pair(0, 0), "the members, ascending");
  }
  {   // the mean response
    const auto m = selectScale({A, B, C}, Response::Mean);
    EXPECT(m.response(0, m.cross[0][0]) == 4.0 && m.best(m.cross[0][0]) == 1 && std::abs(m.peak(m.cross[0][0]) - 6.5) < 1e-12, "mean response: A's event 0 reads 4.0 and B's 6.5");
  }
  {   // SIBLINGS DO NOT NEST: a child of the neighbouring tile at the same time never links
    Pair A2 = make(2, 4, 8, 6), B2 = make(3, 4, 16, 6);
    put(A2, 0, 1, 1, 3.0);
    put(B2, 0, 4, 1, 8.0);          // tile 4 of level 3 is a child of tile 2, not of tile 1
    const auto t = selectScale({A2, B2});
    EXPECT(t.events == 2 && t.cross[0][0] != t.cross[1][0], "a child of the neighbouring tile is not linked");
    Pair B3 = make(3, 4, 16, 6);
    put(B3, 0, 3, 4, 8.0);          // a child of tile 1, but samples 16 … 20: disjoint from A2's 4 … 8
    const auto u = selectScale({A2, B3});
    EXPECT(u.events == 2, "a nested tile at a time that does not overlap is not linked");
  }
  {   // min_link: one overlapping pair of cells links; two are asked
    Pair A2 = make(2, 4, 8, 6), B2 = make(3, 4, 16, 6);
    put(A2, 0, 1, 1, 3.0); put(A2, 0, 1, 2, 3.0);
    put(B2, 0, 2, 2, 8.0);         // one cell of B nests in one of A's (time tile 2 of 4 samples: 8 … 12)
    EXPECT(selectScale({A2, B2}, Response::Peak, 1).events == 1 && selectScale({A2, B2}, Response::Peak, 2).events == 2, "min_link 1 links, 2 does not (one cell pair overlaps)");
    put(B2, 0, 3, 2, 8.0);         // a second cell of B (tile 3, also a child of tile 1) over the same time tile
    EXPECT(selectScale({A2, B2}, Response::Peak, 2).events == 1, "two overlapping cell pairs link at min_link 2");
  }
  {   // TIES go to the first pair; NaN-only events have no best pair; pairs of events in one pair do not link directly
    Pair A2 = make(2, 4, 8, 4), B2 = make(2, 8, 8, 2);
    put(A2, 0, 1, 1, 5.0); put(A2, 1, 1, 3, 5.0);
    put(B2, 0, 1, 0, 5.0);           // samples 0 … 8: overlaps A's event 0 (4 … 8) only
    const auto t = selectScale({A2, B2});
    EXPECT(t.events == 2 && t.best(t.cross[0][0]) == 0 && t.pairs(t.cross[0][0]) == 2, "a tie (5.0 and 5.0) goes to the first pair");
    EXPECT(t.cross[0][1] != t.cross[0][0], "two events of one pair are two events");
    Pair N = make(2, 4, 8, 4);
    N.label(1, 1) = 0; N.events = 1;     // labelled, no value
    const auto v = selectScale({N});
    EXPECT(v.events == 1 && v.best(0) == -1 && v.pairs(0) == 0 && std::isnan(v.peak(0)), "an event with no finite z has no best pair");
  }
  {   // REFUSALS
    Pair A2 = make(2, 4, 8, 4);
    EXPECT(throws([&] { selectScale({}); }), "no pair is refused");
    EXPECT(throws([&] { selectScale({A2}, Response::Peak, 0); }), "min_link 0 is refused");
    Pair bad = A2; bad.z = Eigen::MatrixXd::Zero(8, 3);
    EXPECT(throws([&] { eventFacts(bad); }), "z and labelling of different shapes are refused");
    Pair over = A2; over.label(0, 0) = 3; over.events = 1;
    EXPECT(throws([&] { eventFacts(over); }), "a label beyond the events is refused");
    Pair w = A2; w.window = 0;
    EXPECT(throws([&] { eventFacts(w); }), "a window of no sample is refused");
    EXPECT(throws([] { parseResponse("max"); }) && parseResponse("peak") == Response::Peak && parseResponse("mean") == Response::Mean, "response: peak or mean");
  }
  std::cout << (failures ? "FAILED\n" : "all passed\n");
  return failures ? 1 : 0;
}
