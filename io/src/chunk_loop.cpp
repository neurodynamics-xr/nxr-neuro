// nxr-io — the chunk service (chunk_loop.h).
#include "nxr/io/chunk_loop.h"

#include <fcntl.h>
#include <unistd.h>
#ifndef __EMSCRIPTEN__
#include <sys/file.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "nxr/io/array_reader.h"
#include "nxr/io/errors.h"
#include "nxr/io/json_text.h"

namespace nxr::io {

std::int64_t chunk_width(const Row& recording) {
  const auto js = opt_text(recording, "chunk_shape_json");
  if (!js) return kChunk;
  const Json shape = Json::parse(*js, nullptr, false);
  if (!shape.is_array() || shape.size() < 2 || !shape[1].is_number_integer()) return kChunk;
  return std::max<std::int64_t>(1, std::min(kChunk, shape[1].get<std::int64_t>()));
}

std::pair<std::int64_t, std::int64_t> chunk_span(std::int64_t k, std::int64_t n_samples, std::int64_t width) {
  const std::int64_t n = (n_samples + width - 1) / width;
  if (k < 0 || k >= n) throw Error(code::invalid_argument, "chunk " + std::to_string(k) + ": the recording has " + std::to_string(n) + " chunks (0 to " + std::to_string(n - 1) + ")");
  return {k * width, std::min((k + 1) * width, n_samples)};
}

namespace {
Span ranged(Span s, std::int64_t width) {
  const std::int64_t r0 = s.s0 - s.before, r1 = s.s1 + s.after;
  s.first_chunk = r1 > r0 ? r0 / width : 0;
  s.last_chunk = r1 > r0 ? (r1 - 1) / width : -1;
  return s;
}
}  // namespace

namespace {
// THE SNAP: an end moves to the nearest multiple of 2^level (ties up), clamped to [0, n]; an end at n stays n (the line's end)
std::int64_t snapped(std::int64_t x, int level, std::int64_t n) {
  if (level <= 0 || x >= n) return std::min(std::max<std::int64_t>(0, x), n);
  const std::int64_t t = std::int64_t(1) << level;
  return std::min(((x + t / 2) / t) * t, n);
}
Span snap(Span s, int level, std::int64_t n_samples, double sfreq, std::int64_t width) {
  if (level < 0 || level > 40) throw Error(code::invalid_argument, "a tile level is 0 to 40 (got " + std::to_string(level) + ")");
  s.level = level;
  const std::int64_t a = snapped(s.s0, level, n_samples), b = snapped(s.s1, level, n_samples);
  if (b <= a) {
    char m[200];
    std::snprintf(m, sizeof m, "the span %s snaps to no whole tile of level %d (%lld samples, %g s)", s.what.c_str(), level, static_cast<long long>(std::int64_t(1) << level),
                  static_cast<double>(std::int64_t(1) << level) / sfreq);
    throw Error(code::invalid_argument, m);
  }
  if (a != s.s0 || b != s.s1) {
    char m[64];
    std::snprintf(m, sizeof m, " snapped to level %d", level);
    s.what += m;
  }
  s.s0 = a; s.s1 = b;
  return ranged(s, width);
}
}  // namespace

Span resolve_span(const SpanRequest& r, std::int64_t n_samples, double sfreq, double t0, std::int64_t width) {
  const std::int64_t n = (n_samples + width - 1) / width;
  if (r.chunks && (r.start || r.stop)) throw Error(code::invalid_argument, "the span is chunks or [start, stop) seconds, not both");
  if (r.start || r.stop) {
    if (!r.start || !r.stop) throw Error(code::invalid_argument, "an explicit span gives both start and stop (seconds); the default is chunk 0, the whole recording is chunks \"all\"");
    const std::int64_t s0 = std::max<std::int64_t>(0, static_cast<std::int64_t>(std::llround((*r.start - t0) * sfreq)));
    const std::int64_t s1 = std::min<std::int64_t>(n_samples, static_cast<std::int64_t>(std::llround((*r.stop - t0) * sfreq)));
    char b[96];
    std::snprintf(b, sizeof b, "[%g, %g) s", *r.start, *r.stop);
    Span s{s0, std::max(s0, s1), b};
    s.asked = std::make_pair(*r.start, *r.stop);
    if (r.level > 0) return snap(s, r.level, n_samples, sfreq, width);
    return ranged(s, width);
  }
  const std::string c = r.chunks.value_or("0");
  if (c == "all") return snap(Span{0, n_samples, "all " + std::to_string(n) + " chunks"}, r.level, n_samples, sfreq, width);
  const auto bad = [&] { return Error(code::invalid_argument, "chunks must be k, a-b or all (got '" + c + "')"); };
  const auto dash = c.find('-');
  std::int64_t a = 0, z = 0;
  try {
    std::size_t used = 0;
    if (dash == std::string::npos) {
      a = z = std::stoll(c, &used);
      if (used != c.size()) throw bad();
    } else {
      a = std::stoll(c.substr(0, dash), &used);
      if (used != dash) throw bad();
      z = std::stoll(c.substr(dash + 1), &used);
      if (used != c.size() - dash - 1) throw bad();
    }
  } catch (const std::logic_error&) {
    throw bad();
  }
  if (z < a) throw bad();
  return snap(Span{chunk_span(a, n_samples, width).first, chunk_span(z, n_samples, width).second, (a == z ? "chunk " : "chunks ") + c}, r.level, n_samples, sfreq, width);
}

Span with_halo(const Span& span, std::int64_t before, std::int64_t after, std::int64_t n_samples, std::int64_t width) {
  if (before < 0 || after < 0) throw Error(code::invalid_argument, "a halo is a number of samples, at least 0");
  Span h = span;
  h.before = std::min(before, span.s0);
  h.after = std::min(after, std::max<std::int64_t>(0, n_samples - span.s1));
  return ranged(h, width);
}

Json span_record(const Span& span, double sfreq, double t0) {
  return Json{{"start_sample", span.s0}, {"stop_sample", span.s1}, {"start_s", t0 + static_cast<double>(span.s0) / sfreq}, {"stop_s", t0 + static_cast<double>(span.s1) / sfreq},
              {"chunks", span.what}, {"chunks_read", Json::array({span.first_chunk, span.last_chunk})}, {"halo_samples", Json::array({span.before, span.after})},
              {"tile_level", span.level}, {"asked_s", span.asked ? Json::array({span.asked->first, span.asked->second}) : Json()}};
}

ChunkSource::ChunkSource(const Store& store, std::string array, std::int64_t width, double sfreq, double t0, std::vector<std::int32_t> rows,
                         std::optional<std::vector<std::pair<double, double>>> bad)
    : store_(&store), array_(std::move(array)), width_(std::max<std::int64_t>(1, std::min(kChunk, width))), sfreq_(sfreq), t0_(t0), rows_(std::move(rows)), bad_(std::move(bad)) {
  const ArrayReader ar = ArrayReader::open(*store_, array_);
  if (ar.shape().size() != 2) throw Error(code::invalid_argument, "a recording is [channels x samples]: " + array_);
  n_channels_ = ar.shape()[0];
  n_samples_ = ar.shape()[1];
  if (rows_.empty()) for (std::int64_t c = 0; c < n_channels_; ++c) rows_.push_back(static_cast<std::int32_t>(c));
  for (std::int32_t c : rows_) if (c < 0 || c >= n_channels_) throw Error(code::invalid_argument, "a channel index is outside the recording's channels");
}

namespace {
template <class S> void widen(const std::uint8_t* src, std::int64_t row, std::int64_t n, double* dst) {
  for (std::int64_t j = 0; j < n; ++j) { S v; std::memcpy(&v, src + static_cast<std::size_t>((row * n + j)) * sizeof(S), sizeof(S)); dst[j] = static_cast<double>(v); }
}
}  // namespace

void ChunkSource::read(std::int64_t k, Chunk& out) const {
  const auto [first, last] = chunk_span(k, n_samples_, width_);              // the WHOLE chunk (the Studio's residency: never a part)
  const std::int64_t n = last - first, R = rows();
  const ArrayReader ar = ArrayReader::open(*store_, array_);
  const std::vector<std::uint8_t> raw = ar.read({0, first}, {n_channels_, last});    // [channels x n], row after row
  out.k = k;
  out.first = first;
  out.n = n;
  out.data.resize(static_cast<std::size_t>(R * n));
  for (std::int64_t i = 0; i < R; ++i) {
    double* dst = out.data.data() + static_cast<std::size_t>(i * n);
    const std::int64_t row = rows_[static_cast<std::size_t>(i)];
    switch (ar.dtype()) {
      case DType::Float32: widen<float>(raw.data(), row, n, dst); break;
      case DType::Float64: widen<double>(raw.data(), row, n, dst); break;
      case DType::Int16: widen<std::int16_t>(raw.data(), row, n, dst); break;
      case DType::Int32: widen<std::int32_t>(raw.data(), row, n, dst); break;
      default: throw Error(code::unsupported, std::string("a recording of type ") + dtype_to_string(ar.dtype()) + " is not read here");
    }
  }
  out.good.clear();
  if (bad_) {
    out.good.assign(static_cast<std::size_t>(n), 1);
    for (const auto& [ta, tb] : *bad_) {
      const std::int64_t lo = std::max<std::int64_t>(first, static_cast<std::int64_t>(std::floor((ta - t0_) * sfreq_)) - 1);
      const std::int64_t hi = std::min<std::int64_t>(last, static_cast<std::int64_t>(std::ceil((tb - t0_) * sfreq_)) + 1);
      for (std::int64_t i = lo; i < hi; ++i) {
        const double t = t0_ + static_cast<double>(i) / sfreq_;
        if (t >= ta && t < tb) out.good[static_cast<std::size_t>(i - first)] = 0;
      }
    }
  }
  ++*reads_;
}

std::int64_t ChunkSource::reads() const { return reads_->load(); }

namespace {
std::mutex defaults_mutex;
ChunkLoopOptions& defaults_ref() { static ChunkLoopOptions d; return d; }

// the part [a, b) of a whole chunk, in place
void cut(Chunk& c, std::int64_t a, std::int64_t b, std::int64_t rows) {
  if (a == c.first && b == c.first + c.n) return;
  const std::int64_t n = b - a, off = a - c.first;
  std::vector<double> d(static_cast<std::size_t>(rows * n));
  for (std::int64_t i = 0; i < rows; ++i)
    std::copy_n(c.data.begin() + static_cast<std::ptrdiff_t>(i * c.n + off), n, d.begin() + static_cast<std::ptrdiff_t>(i * n));
  c.data = std::move(d);
  if (!c.good.empty()) c.good = std::vector<std::uint8_t>(c.good.begin() + off, c.good.begin() + off + n);
  c.first = a;
  c.n = n;
}

// THE QUEUE: an exclusive lock on a file, held for the job (flock: released by the system if the process dies)
struct QueueLock {
  int fd = -1;
  explicit QueueLock(const std::optional<std::filesystem::path>& p) {
    if (!p) return;
#ifdef __EMSCRIPTEN__
    return;                                                         // the browser's module runs one job at a time already (the Studio's queue)
#endif
    fd = ::open(p->c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) throw Error(code::io, "the chunk loop's queue file cannot be opened: " + p->string());
#ifndef __EMSCRIPTEN__
    while (::flock(fd, LOCK_EX) != 0) {}
#endif
  }
  ~QueueLock() {
#ifndef __EMSCRIPTEN__
    if (fd >= 0) { ::flock(fd, LOCK_UN); ::close(fd); }
#endif
  }
  QueueLock(const QueueLock&) = delete;
  QueueLock& operator=(const QueueLock&) = delete;
};
}  // namespace

void set_chunk_loop_defaults(ChunkLoopOptions o) { std::lock_guard<std::mutex> g(defaults_mutex); defaults_ref() = std::move(o); }
const ChunkLoopOptions& chunk_loop_defaults() { return defaults_ref(); }

ChunkRun chunk_loop(const ChunkSource& src, const Span& span, const std::function<void(const Chunk&)>& visit, const ChunkLoopOptions* opt) {
  const ChunkLoopOptions& o = opt ? *opt : chunk_loop_defaults();
  ChunkRun run;
  const std::int64_t r0 = span.s0 - span.before, r1 = span.s1 + span.after;
  if (r1 <= r0) return run;
  const std::int64_t w = src.width(), k0 = r0 / w, k1 = (r1 - 1) / w;
  const QueueLock queue(o.queue);                                   // waits for the job before it
  const auto t0 = std::chrono::steady_clock::now();
  ChunkJob job;
  job.count = k1 - k0 + 1;
  auto read = [&src](std::int64_t k) { Chunk c; src.read(k, c); return c; };
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
  const auto ahead = std::launch::deferred;                         // no threads: chunk k + 1 is read when it is needed
#else
  const auto ahead = std::launch::async;                            // chunk k + 1 is read on its own thread while chunk k is folded
#endif
  std::future<Chunk> next = std::async(ahead, read, k0);
  for (std::int64_t k = k0; k <= k1; ++k) {
    if (o.cancelled && o.cancelled()) {
      next.wait();                                                  // never leave a read running behind the job
      job.state = "cancelled";
      if (o.progress) o.progress(job);
      throw Error(code::cancelled, "the job was cancelled after " + std::to_string(job.done) + " of " + std::to_string(job.count) + " chunks; nothing of it is stored");
    }
    Chunk c = next.get();
    if (k < k1) next = std::async(ahead, read, k + 1);              // chunk k + 1 is read while chunk k is folded
    cut(c, std::max(c.first, r0), std::min(c.first + c.n, r1), src.rows());
    try {
      visit(c);
    } catch (...) {
      if (next.valid()) next.wait();
      throw;
    }
    run.visited.push_back(k);
    run.samples += c.n;
    for (auto g : c.good) run.bad_samples += g == 0;
    job.done += 1;
    job.visited.push_back(k);
    job.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (k == k1) job.state = "done";
    if (o.progress) o.progress(job);
  }
  return run;
}

ChunkRun haloed_loop(const ChunkSource& src, const Span& span, std::int64_t halo, const std::function<void(const HaloedChunk&)>& visit, const ChunkLoopOptions* opt) {
  if (halo < 0) throw Error(code::invalid_argument, "haloed_loop: a negative halo");
  const std::int64_t W = src.width(), N = src.n_samples(), R = src.rows();
  const Span hs = with_halo(span, halo, halo, N, W);
  const std::int64_t a = span.s0 - hs.before, b = span.s1 + hs.after;   // what the job reads: the span and its halo, clipped at the recording's ends
  // the buffer: the columns [lo, hi) read and not yet dropped, per row
  std::vector<std::vector<double>> buf(static_cast<std::size_t>(R));
  std::vector<std::uint8_t> gbuf;
  bool masked = false;
  std::int64_t lo = a, hi = a;
  std::int64_t k = span.s0 / W;                                          // the next chunk of the span to hand over
  const std::int64_t k_last = span.s1 > span.s0 ? (span.s1 - 1) / W : k - 1;
  auto emit_ready = [&](bool end) {
    while (k <= k_last) {
      const std::int64_t own0 = std::max(span.s0, k * W), own1 = std::min(span.s1, (k + 1) * W);
      const std::int64_t left = std::min(halo, own0 - a), right = std::min(halo, b - own1);
      if (!end && hi < own1 + right) return;                              // its right-hand context has not arrived yet
      HaloedChunk h;
      h.k = k; h.first = own0; h.n = own1 - own0; h.left = left; h.right = right;
      const std::int64_t c0 = own0 - left - lo, w = left + h.n + right;
      h.data.resize(static_cast<std::size_t>(R * w));
      for (std::int64_t r = 0; r < R; ++r)
        std::copy_n(buf[static_cast<std::size_t>(r)].begin() + c0, w, h.data.begin() + r * w);
      if (masked) h.good.assign(gbuf.begin() + c0, gbuf.begin() + c0 + w);
      visit(h);
      ++k;
      // drop what no later chunk needs: everything before the next chunk's left context
      const std::int64_t keep_from = std::min(hi, std::max(lo, std::max(span.s0, k * W) - halo));
      const std::int64_t drop = keep_from - lo;
      if (drop > 0) {
        for (auto& row : buf) row.erase(row.begin(), row.begin() + drop);
        if (masked) gbuf.erase(gbuf.begin(), gbuf.begin() + drop);
        lo = keep_from;
      }
    }
  };
  const ChunkRun run = chunk_loop(src, hs, [&](const Chunk& c) {
    if (c.first != hi) throw Error(code::invalid_argument, "haloed_loop: the chunk service handed over samples out of order");
    for (std::int64_t r = 0; r < R; ++r) {
      auto& row = buf[static_cast<std::size_t>(r)];
      row.insert(row.end(), c.data.begin() + r * c.n, c.data.begin() + (r + 1) * c.n);
    }
    if (!c.good.empty()) {
      if (!masked) { masked = true; gbuf.assign(static_cast<std::size_t>(hi - lo), 1); }
      gbuf.insert(gbuf.end(), c.good.begin(), c.good.end());
    } else if (masked) {
      gbuf.insert(gbuf.end(), static_cast<std::size_t>(c.n), 1);
    }
    hi += c.n;
    emit_ready(false);
  }, opt);
  emit_ready(true);
  return run;
}

}  // namespace nxr::io
