// nxr::tower against nxr's D135 tower (golden: test/golden/tower_golden.txt, gen_tower_golden.py; synthetic placements and arrays): the code of every sample, the count and counts of the frames,
// the time roll-up on the tower (sum, max, min; every code0 phase); and what is the tower's own: the frame level of a band, the frame tiling's origin / starts / span partition, refusals.
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/tower.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)
using namespace nxr::tower;

int main() {
    std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/tower_golden.txt");
    EXPECT(in.good(), "the golden is there");
    std::string line;
    Placement pl;
    std::string pname;
    int level = 0;
    std::vector<std::int64_t> codes;
    int ncodes = 0, nround = 0, nlev = 0;
    std::int64_t c0 = 0, cN = 0, cnt = 0;
    while (std::getline(in, line)) {
        std::istringstream s(line);
        std::string tag;
        s >> tag;
        if (tag == "placement") {
            std::string anchor, top, sts; double fs; std::int64_t n, day;
            s >> pname >> fs >> n >> sts >> day >> anchor >> top;
            const double st = sts == "nan" ? std::nan("") : std::stod(sts);
            pl = Placement{fs, n, std::isnan(st) ? std::nullopt : std::optional<double>(st), day < 0 ? std::nullopt : std::optional<std::int64_t>(day)};
            EXPECT(pl.anchor() == anchor, pname << " anchor");
            EXPECT((top == "none") == !pl.coarsest().has_value() && (top == "none" || pl.coarsest() == std::stoi(top)), pname << " coarsest level " << top);
        } else if (tag == "refused") {
            int l; s >> l;
            bool threw = false;
            try { pl.sampleCodes(l); } catch (const std::invalid_argument&) { threw = true; }
            EXPECT(threw, pname << " level " << l << " refused (above what is known)");
        } else if (tag == "codes") {
            double per; s >> level >> per >> c0 >> cN >> cnt;
            EXPECT(periodS(level) == per, "period of level " << level << " exact");
            codes = pl.sampleCodes(level);
            EXPECT(codes.front() == c0 && codes.back() == cN && cN - c0 + 1 == cnt, pname << " L" << level << " first, last, count of the tiles");
            ++ncodes;
        } else if (tag == "samples") {
            std::int64_t v; std::size_t i = 0; bool ok = true;
            while (s >> v) { ok = ok && codes[i * 7] == v; ++i; }
            EXPECT(ok && i == (codes.size() + 6) / 7, pname << " L" << level << " the code of every 7th sample, exact");
        } else if (tag == "counts") {
            std::vector<std::int64_t> want; std::int64_t v;
            while (s >> v) want.push_back(v);
            std::vector<std::int64_t> got(static_cast<std::size_t>(cnt), 0);
            for (auto c : codes) ++got[static_cast<std::size_t>(c - c0)];
            bool ok = true;
            for (std::size_t i = 0; i < want.size(); ++i) ok = ok && got[i] == want[i];
            EXPECT(ok, pname << " L" << level << " the samples of each frame");
            if (level <= 0 && !pl.start_s.has_value() ? true : (level <= 0 && !pl.day)) {   // the frame tiling: recording and day anchors
                const FrameTiling t = makeFrameTiling(pl.sfreq, pl.n_samples, level, pl.start_s);
                const auto sc = sampleCounts(t);
                bool same = t.code0 == c0 && t.count == cnt && static_cast<std::int64_t>(sc.size()) == cnt;
                for (std::size_t i = 0; same && i < sc.size(); ++i) same = sc[i] == got[i];
                EXPECT(same, pname << " L" << level << " FrameTiling: code0, count and per-frame samples as nxr's");
                bool fr = true;
                for (std::int64_t i = 0; i < pl.n_samples; i += 13) fr = fr && t.frameOfSample(i) == codes[static_cast<std::size_t>(i)] - c0;
                EXPECT(fr, pname << " L" << level << " frameOfSample");
            }
        } else if (tag == "level_for_seconds") {
            double sec; int l; s >> sec >> l;
            EXPECT(levelForSeconds(sec) == l, "level_for_seconds " << sec);
            ++nlev;
        } else if (tag == "rollup") {
            std::string op; int m; std::int64_t code0, rows, frames, ntl;
            s >> op >> m >> code0 >> rows >> frames >> ntl;
            std::string xl, yl;
            std::getline(in, xl); std::getline(in, yl);
            std::istringstream xs(xl.substr(2)), ys(yl.substr(2));
            std::vector<double> x, y; double v;
            while (xs >> v) x.push_back(v);
            while (ys >> v) y.push_back(v);
            std::int64_t nt = 0;
            const auto got = rollTime(x, rows, frames, m, op == "sum" ? Merge::Sum : op == "max" ? Merge::Max : Merge::Min, code0, &nt);
            double e = 0;
            for (std::size_t i = 0; i < y.size() && got.size() == y.size(); ++i) e = std::max(e, std::fabs(got[i] - y[i]));
            EXPECT(nt == ntl && got.size() == y.size() && e < 1e-12, "rollup " << op << " m=" << m << " code0=" << code0 << " err " << e);
            ++nround;
        }
    }
    EXPECT(ncodes == 12 && nlev == 8 && nround == 18, "the golden's cases were all read");

    // the frame level of a band: the coarsest tower level whose frame is at or under 1/(centre x frames a cycle)
    EXPECT(frameLevelFor(10, 36) == -25 && std::fabs(periodS(-25) - 0.002574920654296875) < 1e-18, "10 Hz x 36 -> level -25 (2.57 ms; target 2.78 ms)");
    EXPECT(frameLevelFor(12, 36) == -26, "12 Hz x 36 -> level -26 (1.29 ms; level -25 is 2.57 ms, over the 2.31 ms target)");
    EXPECT(frameLevelFor(1.0 / periodS(-20) / 4, 4) == -20, "a target exactly on a level keeps it");
    EXPECT(levelAtOrUnder(86400.0) == 0 && levelAtOrUnder(86399.0) == -1 && levelAtOrUnder(43200.0) == -1, "levelAtOrUnder at the day");
    bool threw = false;
    try { frameLevelFor(0, 36); } catch (const std::invalid_argument&) { threw = true; }
    EXPECT(threw, "a centre of 0 is refused");
    EXPECT(parent(13, 2) == 3 && parent(13, 0) == 13, "parent = code >> up");

    // the frame tiling of a recording (anchor recording): frame k starts at k * frame, 395.5 samples at 1200 Hz
    {
        const FrameTiling t = makeFrameTiling(1200.0, 4000, -18);
        EXPECT(t.code0 == 0 && t.count == 11 && std::fabs(t.tile_samples() - 395.5078125) < 1e-12 && t.origin_s() == 0, "recording anchor: frame 0 at sample 0, 11 frames");
        EXPECT(t.position(0) == 0 && std::fabs(t.position(2) - 2 * 395.5078125) < 1e-9, "frame starts in samples");
        for (std::int64_t chunk : {std::int64_t(1), std::int64_t(7), std::int64_t(1000), std::int64_t(4000)}) {
            std::int64_t next = 0; bool ok = true;
            for (std::int64_t s0 = 0; s0 < t.samples; s0 += chunk) {
                const auto [a, b] = frameSpan(t, s0, std::min(chunk, t.samples - s0));
                ok = ok && a == next && b >= a;
                for (std::int64_t k = a; k < b; ++k) ok = ok && t.position(k) >= s0 - 1e-6 && t.position(k) < s0 + std::min(chunk, t.samples - s0);
                next = b;
            }
            EXPECT(ok && next == t.count, "every frame once, in order, in the chunk holding its start (chunk " << chunk << ")");
        }
    }
    // a day anchor: the first frame is partial, its start before the first sample, and it belongs to the first span
    {
        const FrameTiling t = makeFrameTiling(250.0, 3000, -18, 43200.123456);
        EXPECT(t.position(0) < 0 && t.position(1) >= 0 && frameSpan(t, 0, 10).first == 0, "day anchor: a partial first frame starts before the recording");
        EXPECT(std::fabs(t.origin_s() - (static_cast<double>(t.code0) * periodS(-18) - 43200.123456)) < 1e-9 && t.origin_s() < 0, "origin = code0 x frame - start_s");
        std::int64_t sum = 0;
        for (const auto c : sampleCounts(t)) sum += c;
        EXPECT(sum == 3000, "every sample is in a frame");
        std::vector<std::uint8_t> good(3000, 1);
        for (int i = 0; i < 100; ++i) good[static_cast<std::size_t>(i)] = 0;
        std::int64_t g = 0;
        for (const auto c : sampleCounts(t, good)) g += c;
        EXPECT(g == 2900, "bad samples carry nothing");
        EXPECT(t.tileOfFrame(5, 3) == (t.code0 + 5) >> 3, "a frame's tile m levels up");
    }
    threw = false;
    try { makeFrameTiling(250.0, 100, 1); } catch (const std::invalid_argument&) { threw = true; }
    EXPECT(threw, "a frame level above the day is refused");
    threw = false;
    try { makeFrameTiling(250.0, 100, -5, 90000.0); } catch (const std::invalid_argument&) { threw = true; }
    EXPECT(threw, "a start outside the day is refused");
    // roll-up of a partial front: codes 5.. at m=2 -> tiles of codes 4-7 hold frames 0..2
    {
        std::int64_t nt = 0;
        const auto y = rollTime({1, 2, 3, 4, 5}, 1, 5, 2, Merge::Sum, 5, &nt);
        EXPECT(nt == 2 && y[0] == 6 && y[1] == 9, "neutral front pad: frames of codes 5,6,7 | 8,9");
    }
    std::cout << (fails ? "test_tower FAILED\n" : "test_tower ok\n");
    return fails ? 1 : 0;
}
