// The chunk service (chunk_loop.h) on a synthetic recording [3 x 75000] float32 stored in 2^15-sample chunks (and the same samples in 700-sample chunks):
// the span rules (chunk 0 by default; "k", "a-b", "all"; [start, stop) s; the refusals), each chunk read WHOLE and only once, the rows asked for in their order,
// the values exact, the bad-sample mask, the loop's result the same whatever the chunk width; the HALO (read from the neighbouring chunks, clipped at the
// recording's ends); the JOB: progress after each chunk, a cancel stops it (code::cancelled), two jobs on one queue never overlap; the producer record.
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <set>
#include <tuple>
#include <thread>

#include "check.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/chunk_loop.h"
#include "nxr/io/errors.h"

using namespace nxr::io;
namespace fs = std::filesystem;

static fs::path store(const fs::path& at, const std::vector<float>& x, std::int64_t C, std::int64_t T, std::int64_t chunk) {
  ArrayOptions o;
  o.chunks = Shape{C, chunk};
  o.compress = false;
  FolderStore fst(at.parent_path());
  layout_array(fst, at.filename().string(), array_meta(DType::Float32, {C, T}, o));
  populate_array(fst, at.filename().string(), view(x, {C, T}));
  return at;
}
template <class F> static std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_chunk_loop_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const std::int64_t C = 3, T = 75000, W = kChunk;
  const double sfreq = 1000.0, t0 = 0.0;
  std::vector<float> x(static_cast<std::size_t>(C * T));
  for (std::int64_t c = 0; c < C; ++c) for (std::int64_t t = 0; t < T; ++t) x[static_cast<std::size_t>(c * T + t)] = static_cast<float>(c * 1e5 + t);
  const fs::path big = store(root / "big", x, C, T, W), small = store(root / "small", x, C, T, 700);
  const FolderStore rootstore(root);
  // THE SPAN
  CHECK_EQ(code_of([&] { resolve_span({}, T, sfreq, t0, W); }), std::string(code::invalid_argument));   // no default span: the caller states its range
  const Span d = resolve_span({std::string("0"), {}, {}}, T, sfreq, t0, W);   // the first chunk's samples, stated
  CHECK(d.s0 == 0 && d.s1 == W && d.what == "chunk 0");
  const Span k2 = resolve_span({std::string("2"), {}, {}}, T, sfreq, t0, W);
  CHECK(k2.s0 == 2 * W && k2.s1 == T);                                             // the last chunk is partial
  const Span r = resolve_span({std::string("0-1"), {}, {}}, T, sfreq, t0, W);
  CHECK(r.s0 == 0 && r.s1 == 2 * W);
  const Span all = resolve_span({std::string("all"), {}, {}}, T, sfreq, t0, W);
  CHECK(all.s0 == 0 && all.s1 == T && all.what == "all 3 chunks");
  const Span sec = resolve_span({std::nullopt, 10.0, 40.0}, T, sfreq, t0, W);
  CHECK(sec.s0 == 10000 && sec.s1 == 40000);
  // THE SNAP to tiles: the nearest multiple of 2^level on the line's grid; an end at the recording's end stays; a span with no whole tile is refused
  const Span t12 = resolve_span({std::nullopt, 10.0, 40.0, 12}, T, sfreq, t0, W);   // 10000 -> 8192 (2.4 tiles), 40000 -> 40960 (9.77 tiles)
  CHECK(t12.s0 == 8192 && t12.s1 == 40960 && t12.level == 12 && t12.asked && t12.asked->first == 10.0);
  CHECK(span_record(t12, sfreq, t0)["tile_level"] == 12);
  const Span tend = resolve_span({std::nullopt, 60.0, 80.0, 12}, T, sfreq, t0, W);   // stop past the end: the end (75000), not a tile boundary
  CHECK(tend.s0 == 61440 && tend.s1 == T);
  CHECK(resolve_span({std::string("1"), {}, {}, 12}, T, sfreq, t0, W).s0 == W);   // a chunk span is level-15 tiles already
  CHECK_EQ(code_of([&] { resolve_span({std::string("1"), {}, {}, 16}, T, sfreq, t0, W); }), std::string(code::invalid_argument));   // [32768, 65536) -> [65536, 65536)
  CHECK_EQ(code_of([&] { resolve_span({std::nullopt, 10.0, 10.1, 12}, T, sfreq, t0, W); }), std::string(code::invalid_argument));    // 10000, 10100 -> 8192, 8192
  for (const SpanRequest& bad : {SpanRequest{std::string("3"), {}, {}}, SpanRequest{std::string("x"), {}, {}}, SpanRequest{std::string("2-1"), {}, {}},
                                 SpanRequest{std::nullopt, 1.0, std::nullopt}, SpanRequest{std::string("0"), 1.0, 2.0}})
    CHECK_EQ(code_of([&] { resolve_span(bad, T, sfreq, t0, W); }), std::string(code::invalid_argument));
  // THE LOOP over [10 s, 40 s): chunks 0 and 1 (the span's parts of them), rows {2, 0}, a bad span [12 s, 12.5 s)
  const ChunkSource src(rootstore, "big", W, sfreq, t0, {2, 0}, std::vector<std::pair<double, double>>{{12.0, 12.5}});
  CHECK(src.count() == 3 && src.rows() == 2 && src.n_samples() == T);
  std::int64_t worst = 0, next = sec.s0;
  const ChunkRun run = chunk_loop(src, sec, [&](const Chunk& c) {
    CHECK_EQ(c.first, next);
    next = c.first + c.n;
    for (std::int64_t j = 0; j < c.n; ++j) {
      worst = std::max<std::int64_t>(worst, std::llabs(static_cast<std::int64_t>(c.data[static_cast<std::size_t>(j)]) - (2 * 100000 + c.first + j)));
      worst = std::max<std::int64_t>(worst, std::llabs(static_cast<std::int64_t>(c.data[static_cast<std::size_t>(c.n + j)]) - (c.first + j)));
    }
  });
  CHECK_EQ(worst, std::int64_t(0));
  CHECK_EQ(next, sec.s1);
  CHECK(run.visited == (std::vector<std::int64_t>{0, 1}));
  CHECK_EQ(run.samples, std::int64_t(30000));
  CHECK_EQ(run.bad_samples, std::int64_t(500));
  // the same span on 700-sample chunks: the same samples handed over, in more pieces
  const ChunkSource s7(rootstore, "small", 700, sfreq, t0, {2, 0}, std::vector<std::pair<double, double>>{{12.0, 12.5}});
  double sum_big = 0, sum_small = 0;
  chunk_loop(src, sec, [&](const Chunk& c) { for (std::size_t i = 0; i < c.data.size(); ++i) sum_big += c.data[i] * (c.good.empty() ? 1 : c.good[i % static_cast<std::size_t>(c.n)]); });
  const ChunkRun r7 = chunk_loop(s7, sec, [&](const Chunk& c) { for (std::size_t i = 0; i < c.data.size(); ++i) sum_small += c.data[i] * (c.good.empty() ? 1 : c.good[i % static_cast<std::size_t>(c.n)]); });
  CHECK(sum_big == sum_small && r7.bad_samples == 500 && r7.samples == 30000 && r7.visited.size() == 44);
  // WHOLE chunks: two chunks touched, two read (each once), the span's part handed over
  CHECK_EQ(src.reads(), std::int64_t(4));                                          // two loops above, two chunks each
  // THE HALO: 500 samples each side of [10 s, 40 s), read from the same chunks; at the recording's start it is clipped
  const Span h = with_halo(sec, 500, 500, T, W);
  std::int64_t hfirst = -1, hlast = -1;
  chunk_loop(src, h, [&](const Chunk& c) { if (hfirst < 0) hfirst = c.first; hlast = c.first + c.n; });
  CHECK(hfirst == 9500 && hlast == 40500 && h.before == 500 && h.after == 500);
  const Span h0 = with_halo(d, 1000, 1000, T, W);
  CHECK(h0.before == 0 && h0.after == 1000 && h0.first_chunk == 0 && h0.last_chunk == 1);  // chunk 0's halo reaches into chunk 1
  const Span hend = with_halo(k2, 0, 5000, T, W);
  CHECK(hend.after == 0);                                                          // the recording ends
  const Json rec = span_record(h, sfreq, t0);
  CHECK(rec["chunks"] == "[10, 40) s" && rec["chunks_read"][0] == 0 && rec["chunks_read"][1] == 1 && rec["halo_samples"][0] == 500);
  // THE JOB: progress after each chunk, the last "done"
  std::vector<ChunkJob> seen;
  ChunkLoopOptions po;
  po.progress = [&](const ChunkJob& j) { seen.push_back(j); };
  chunk_loop(src, all, [](const Chunk&) {}, &po);
  CHECK(seen.size() == 3 && seen.back().done == 3 && seen.back().count == 3 && seen.back().state == "done" && seen.back().visited == (std::vector<std::int64_t>{0, 1, 2}));
  // a CANCEL after the first chunk: code::cancelled, the job's state "cancelled", one chunk folded
  int folded = 0;
  ChunkLoopOptions co;
  std::string last_state;
  co.cancelled = [&] { return folded >= 1; };
  co.progress = [&](const ChunkJob& j) { last_state = j.state; };
  CHECK_EQ(code_of([&] { chunk_loop(src, all, [&](const Chunk&) { ++folded; }, &co); }), std::string(code::cancelled));
  CHECK(folded == 1 && last_state == "cancelled");
  // THE QUEUE: two jobs on one queue file never overlap
  ChunkLoopOptions qo;
  qo.queue = root / "queue.lock";
  std::vector<std::pair<double, double>> spans(2);
  const auto start = std::chrono::steady_clock::now();
  auto now = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
  auto job = [&](int i) {
    bool firstc = true;
    chunk_loop(src, all, [&](const Chunk&) {
      if (firstc) { spans[static_cast<std::size_t>(i)].first = now(); firstc = false; }
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      spans[static_cast<std::size_t>(i)].second = now();
    }, &qo);
  };
  std::thread a(job, 0), b(job, 1);
  a.join(); b.join();
  const bool apart = spans[0].second <= spans[1].first || spans[1].second <= spans[0].first;
  std::printf("  queue: job A [%.3f, %.3f] s, job B [%.3f, %.3f] s\n", spans[0].first, spans[0].second, spans[1].first, spans[1].second);
  CHECK(apart);
  // THE HALOED WALK: each chunk of the span with its halo from the neighbouring chunks, clipped at the recording's ends; a halo wider than a chunk; a span
  // cut mid-chunk; every chunk still read once
  for (const auto& [sp, H, w, what] : {std::tuple<Span, std::int64_t, std::int64_t, const char*>{all, 300, 700, "all, halo 300, 700-sample chunks"},
                                       {sec, 1500, 700, "[10, 40) s, halo 1500 (> a chunk)"}, {sec, 0, 700, "[10, 40) s, no halo"}, {all, 2000, W, "all, halo 2000, 2^15 chunks"}}) {
    const ChunkSource hsrc(rootstore, w == W ? "big" : "small", w, sfreq, t0, {2, 0});
    std::int64_t bad = 0, next_own = sp.s0, blocks = 0;
    const ChunkRun hr = haloed_loop(hsrc, sp, H, [&](const HaloedChunk& h) {
      ++blocks;
      if (h.first != next_own) ++bad;
      next_own = h.first + h.n;
      if (h.left != std::min(H, h.first) || h.right != std::min(H, T - (h.first + h.n))) ++bad;
      const std::int64_t wd = h.left + h.n + h.right;
      for (std::int64_t j = 0; j < wd; ++j) {
        const std::int64_t t = h.first - h.left + j;
        if (static_cast<std::int64_t>(h.data[static_cast<std::size_t>(j)]) != 2 * 100000 + t || static_cast<std::int64_t>(h.data[static_cast<std::size_t>(wd + j)]) != t) ++bad;
      }
    });
    std::set<std::int64_t> seen(hr.visited.begin(), hr.visited.end());
    std::printf("  haloed: %s: %lld blocks, %lld chunks read\n", what, static_cast<long long>(blocks), static_cast<long long>(hr.visited.size()));
    CHECK(bad == 0 && next_own == sp.s1 && seen.size() == hr.visited.size() && hsrc.reads() == static_cast<std::int64_t>(hr.visited.size()));
  }
  fs::remove_all(root);
  return nxrtest::finish("test_chunk_loop");
}
