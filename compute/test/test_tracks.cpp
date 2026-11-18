// nxr::tracks::Linker against nxr's `_track_stats` linking (golden: test/golden/tracks_golden.txt, gen_tracks_golden.py; planted moving vortices of both senses on a sphere, two same-sense cores passing within the link
// distance, lone cores, frames with no core and a gap): the same links (sense, from, displacement, track number, frame) in the same order, and the same (frames, path, displacement) of the tracks of 3 frames or more.
// The tracks do not depend on how the cores are cut into blocks: any cut (also one that splits a frame) gives the same labels. Hand-made cases: a gap breaks every track, a track takes one core a frame (the first in the
// order takes the nearest), the sense must agree, the distance is inclusive, the stats; refusals.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/select/tracks.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

namespace {
std::vector<double> nums(std::istringstream& s) { std::vector<double> v; double x; while (s >> x) v.push_back(x); return v; }
template <class F> bool throws(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
Eigen::MatrixXd rows(const std::vector<std::array<double, 3>>& c) { Eigen::MatrixXd m(static_cast<Eigen::Index>(c.size()), 3); for (std::size_t i = 0; i < c.size(); ++i) for (int k = 0; k < 3; ++k) m(static_cast<Eigen::Index>(i), k) = c[i][static_cast<std::size_t>(k)]; return m; }
}  // namespace

int main() {
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/tracks_golden.txt");
  if (!in) { std::cout << "no golden\n"; return 1; }
  std::vector<double> centre, steps, tracks3;
  std::vector<std::int64_t> frame;
  std::vector<std::int8_t> sense;
  double link = 0;
  int n = 0;
  std::string ln;
  while (std::getline(in, ln)) {
    std::istringstream s(ln);
    std::string tag;
    s >> tag;
    if (tag == "cores") { std::string a, b, c; int ns, nt; s >> n >> a >> ns >> b >> nt >> c >> link; }
    else if (tag == "frame") { for (double f : nums(s)) frame.push_back(static_cast<std::int64_t>(f)); }
    else if (tag == "sense") { for (double f : nums(s)) sense.push_back(static_cast<std::int8_t>(f)); }
    else if (tag == "centre") centre = nums(s);
    else if (tag == "steps") steps = nums(s);
    else if (tag == "tracks3") tracks3 = nums(s);
  }
  const Eigen::MatrixXd C = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(centre.data(), n, 3);
  const nxr::tracks::Linked L = nxr::tracks::link(frame, sense, C, link);
  // the steps: every core with a predecessor, in order
  std::size_t k = 0;
  bool ok = true;
  double dmax = 0;
  for (int i = 0; i < n; ++i) {
    const int p = L.previous[static_cast<std::size_t>(i)];
    if (p < 0) continue;
    if (9 * (k + 1) > steps.size()) { ok = false; break; }
    const double* s = steps.data() + 9 * k;
    const Eigen::Vector3d from = C.row(p).transpose(), disp = C.row(i).transpose() - from;
    ok = ok && static_cast<int>(s[0]) == sense[static_cast<std::size_t>(i)] && static_cast<int>(s[7]) == L.track[static_cast<std::size_t>(i)] + 1 && static_cast<std::int64_t>(s[8]) == frame[static_cast<std::size_t>(i)];
    for (int d = 0; d < 3; ++d) dmax = std::max({dmax, std::abs(s[1 + d] - from[d]), std::abs(s[4 + d] - disp[d])});
    ++k;
  }
  std::cout << "  " << n << " cores, " << L.tracks << " tracks, " << k << " links (nxr " << steps.size() / 9 << "), steps differ by " << dmax << " mm\n";
  EXPECT(ok && k * 9 == steps.size() && dmax < 1e-9, "the links are nxr's: sense, from, displacement, track, frame");
  // the tracks of >= 3 frames
  const nxr::tracks::TrackStats S = nxr::tracks::trackStats(frame, sense, C, L.track, L.previous);
  std::vector<std::array<double, 3>> mine;
  for (std::int64_t t = 0; t < S.size(); ++t) if (S.n_frames[static_cast<std::size_t>(t)] >= 3) mine.push_back({static_cast<double>(S.n_frames[static_cast<std::size_t>(t)]), S.path[static_cast<std::size_t>(t)], S.displacement[static_cast<std::size_t>(t)]});
  std::sort(mine.begin(), mine.end());
  bool same3 = mine.size() * 3 == tracks3.size();
  double d3 = 0;
  for (std::size_t t = 0; same3 && t < mine.size(); ++t) for (int j = 0; j < 3; ++j) d3 = std::max(d3, std::abs(mine[t][static_cast<std::size_t>(j)] - tracks3[3 * t + static_cast<std::size_t>(j)]));
  std::cout << "  " << mine.size() << " tracks of 3 frames or more (nxr " << tracks3.size() / 3 << "), (frames, path, displacement) differ by " << d3 << "\n";
  EXPECT(same3 && d3 < 1e-9, "the tracks' frames, path and displacement are nxr's");
  // any cut into blocks, also through a frame
  for (const std::size_t cut : {std::size_t(1), std::size_t(7), std::size_t(50), std::size_t(133)}) {
    nxr::tracks::Linker l(link);
    for (std::size_t a = 0; a < static_cast<std::size_t>(n); a += cut) {
      const std::size_t b = std::min<std::size_t>(a + cut, static_cast<std::size_t>(n));
      l.feed(std::vector<std::int64_t>(frame.begin() + a, frame.begin() + b), std::vector<std::int8_t>(sense.begin() + a, sense.begin() + b), C.middleRows(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b - a)));
    }
    l.finish();
    EXPECT(l.track() == L.track && l.previous() == L.previous && l.tracks() == L.tracks, "blocks of " << cut << " cores give the same tracks");
  }
  // hand-made
  { const auto x = nxr::tracks::link({0, 1, 2, 4, 5}, {1, 1, 1, 1, 1}, rows({{0, 0, 0}, {5, 0, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}}), 15.0);
    EXPECT(x.track == std::vector<std::int32_t>({0, 0, 0, 1, 1}) && x.previous == std::vector<std::int32_t>({-1, 0, 1, -1, 3}) && x.tracks == 2, "a frame without a core (frame 3) breaks the track"); }
  { const auto x = nxr::tracks::link({0, 1}, {1, -1}, rows({{0, 0, 0}, {1, 0, 0}}), 15.0);
    EXPECT(x.track == std::vector<std::int32_t>({0, 1}), "the sense must agree"); }
  { const auto x = nxr::tracks::link({0, 1}, {1, 1}, rows({{0, 0, 0}, {15, 0, 0}}), 15.0), y = nxr::tracks::link({0, 1}, {1, 1}, rows({{0, 0, 0}, {15.001, 0, 0}}), 15.0);
    EXPECT(x.tracks == 1 && y.tracks == 2, "the link distance is inclusive"); }
  { // greedy: the first core of the frame takes the nearest track even if the second is nearer to it than to its own; each track takes one core
    const auto x = nxr::tracks::link({0, 0, 1, 1}, {1, 1, 1, 1}, rows({{0, 0, 0}, {20, 0, 0}, {9, 0, 0}, {10, 0, 0}}), 15.0);
    EXPECT(x.track[2] == 0 && x.track[3] == 1 && x.previous[2] == 0 && x.previous[3] == 1 && x.tracks == 2, "the first core in the order takes the nearest track; the second takes the one left"); }
  { // one core a frame per track: two cores near one track: the second starts a new one
    const auto x = nxr::tracks::link({0, 1, 1}, {1, 1, 1}, rows({{0, 0, 0}, {3, 0, 0}, {4, 0, 0}}), 15.0);
    EXPECT(x.track == std::vector<std::int32_t>({0, 0, 1}), "a track takes one core a frame"); }
  { const auto x = nxr::tracks::link({3, 4, 5}, {-1, -1, -1}, rows({{0, 0, 0}, {3, 4, 0}, {3, 4, 12}}), 15.0);
    const auto s = nxr::tracks::trackStats({3, 4, 5}, {-1, -1, -1}, rows({{0, 0, 0}, {3, 4, 0}, {3, 4, 12}}), x.track, x.previous);
    EXPECT(s.size() == 1 && s.first_frame[0] == 3 && s.last_frame[0] == 5 && s.n_frames[0] == 3 && s.sense[0] == -1 && std::abs(s.path[0] - 17.0) < 1e-12 && std::abs(s.displacement[0] - 13.0) < 1e-12, "a track's stats"); }
  { const auto x = nxr::tracks::link({}, {}, Eigen::MatrixXd(0, 3), 15.0);
    EXPECT(x.tracks == 0 && x.track.empty(), "no cores, no tracks"); }
  // refusals
  EXPECT(throws([] { nxr::tracks::Linker l(0.0); }), "a link distance of 0 is refused");
  EXPECT(throws([&] { nxr::tracks::link({1, 0}, {1, 1}, rows({{0, 0, 0}, {0, 0, 0}}), 15.0); }), "frames that go back are refused");
  EXPECT(throws([&] { nxr::tracks::link({0}, {0}, rows({{0, 0, 0}}), 15.0); }), "a sense that is not +-1 is refused");
  EXPECT(throws([&] { nxr::tracks::link({0, 1}, {1}, rows({{0, 0, 0}}), 15.0); }), "sizes that disagree are refused");
  EXPECT(throws([&] { nxr::tracks::link({0}, {1}, rows({{std::nan(""), 0, 0}}), 15.0); }), "a non-finite centre is refused");
  EXPECT(throws([&] { nxr::tracks::Linker l(5.0); l.finish(); l.feed({0}, {1}, rows({{0, 0, 0}})); }), "a feed after finish is refused");
  std::cout << (fails ? "FAILED\n" : "test_tracks: all passed\n");
  return fails ? 1 : 0;
}
