// nxr-io — THE CHUNK SERVICE: the one place a recording is loaded and looped over. The Studio's design, ported (`backend/src/db/chunk.ts`,
// `studio/src/studio/engine/chunkLoop.ts`, `ports.ts` ChunkSource; nxr's `nxr/io/chunks.py`), so the toolkit makes the same architectural decision once:
//
//   THE CHUNK  2^15 samples of a recording's time axis, every channel: what is LOADED and computed on. Chunk k is samples
//                                    [k·2^15, min((k+1)·2^15, n)); the last is partial. A shard (2^17, four chunks) is storage, never a chunk. A recording's chunk
//                                    width is its row's `chunk_shape_json[1]`, never more than 2^15 (as nxr's chunk_width).
//   THE SPAN of a run  which samples a run loads: `chunks` "k", "a-b" (inclusive) or "all" (the whole recording, as a loop over its chunks), or an
//                                    explicit [start, stop) in seconds (both given). Neither: an Error — the span is always stated by the caller. NOT the FFT window:
//                                    a computation's own windows (a Welch segment's length, taper, overlap) are its choice and are carried across chunk edges.
//   THE HALO (the Studio's `haloOf`, `HALO`)  samples of CONTEXT a computation needs on each side of its span so its edges are computed with the samples beyond
//                                    them (a wavelet's or a filter's margin), never with a reflection: read from the neighbouring chunks, clipped where the recording
//                                    ends (what to do at a recording's end is the computation's). Its size is the computation's (`with_halo`); a Welch spectrum needs none.
//   ChunkSource                      a recording read one WHOLE chunk at a time (the Studio's residency: never a part), only the rows (channels) asked for, as float64
//                                    [rows x n] row-major, with the chunk's good-sample mask when bad spans are given; `reads` counts what was read.
//   chunk_loop                       THE JOB (the Studio's `chunkLoop`): walks the chunks of the span and its halo in order, each read once, chunk k + 1 read WHILE
//                                    chunk k is folded (read-ahead), each handed to `visit` cut to what was asked; progress after each chunk; cancellable (a cancelled job
//                                    throws `code::cancelled`, so nothing of it is stored); one job at a time (a QUEUE: a lock file, so two jobs never share the disk).
//                                    Never a whole-recording buffer: the memory held is two chunks (the one folded, the one read ahead).
//   defaults                         the process's progress, cancel and queue (`set_chunk_loop_defaults`), set once by the command line; pipelines pass nothing.
//
// COMPUTE NEVER SEES THIS: a pipeline declares the recording, its rows and its span and passes `visit`; the maths (nxr_compute, nxr_meeg) receives a matrix and
// does not know how big it is or where it came from. No pipeline opens a recording's array or loops over its samples itself (ARCHITECTURE.md, the layout rule;
// enforced by the test `layering_recordings_read_by_chunk_service`).
#pragma once
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/json_text.h"
#include "nxr/io/store.h"

namespace nxr::io {

constexpr std::int64_t kChunk = std::int64_t(1) << 15;

/** a recording's chunk width from its Field row (nxr's chunk_width) */
std::int64_t chunk_width(const Row& recording);
/** the same from the Field's chunk shape (its `chunk_shape_json` as JSON; null: the default) */
std::int64_t chunk_width(const Json& chunk_shape);
/** chunk k's samples [a, b); a chunk outside the recording is an Error (invalid_argument) */
std::pair<std::int64_t, std::int64_t> chunk_span(std::int64_t k, std::int64_t n_samples, std::int64_t width);

/** what a run asks for: chunks ("k", "a-b", "all") or [start, stop) seconds; REQUIRED: the caller always states its range (nothing given is an Error). `level`: the TILE LEVEL the run measures on: both ends of the span snap to the NEAREST multiple of
 *  2^level, clamped to the recording (an end at the recording's end stays there: the last tile may be partial). 0 (a sample) is the old rounding. A
 *  chunk span is level-15 tiles already, so it moves only when level > 15. The level is the pipeline's (its window, its segment), never compute's. */
struct SpanRequest {
  std::optional<std::string> chunks;
  std::optional<double> start, stop;
  int level = 0;
};
struct Span {
  std::int64_t s0 = 0, s1 = 0;      // samples [s0, s1): the span proper
  std::string what;                 // "chunk 0" · "chunks 2-5" · "all 7 chunks" · "[2, 25) s": for the producer record
  std::int64_t before = 0, after = 0;   // the halo read on each side (clipped at the recording's ends): samples [s0 - before, s1 + after) are read
  std::int64_t first_chunk = 0, last_chunk = -1;   // the chunks read (the span with its halo): every one of them is read when a result is stored
  int level = 0;                    // the tile level the span was snapped to (2^level samples)
  std::optional<std::pair<double, double>> asked;   // the [start, stop) seconds asked, when the span was given in seconds (the snapped extent is s0, s1)
};
/** the span of a run (the rule above); an invalid request is an Error (invalid_argument) naming the valid forms */
Span resolve_span(const SpanRequest& r, std::int64_t n_samples, double sfreq, double t0, std::int64_t width);
/** the span with a halo of `before` and `after` samples, clipped at the recording's ends (`span.before`/`after` say how much was there) */
Span with_halo(const Span& span, std::int64_t before, std::int64_t after, std::int64_t n_samples, std::int64_t width);
/** the span as a producer record states it: samples, seconds, the chunks read, the halo, the tile level and the seconds asked */
Json span_record(const Span& span, double sfreq, double t0);

/** one chunk's part of a span, as handed to `visit` */
struct Chunk {
  std::int64_t k = 0;                         // the chunk's index
  std::int64_t first = 0, n = 0;              // its samples [first, first + n) (the span's part of the chunk)
  std::vector<double> data;                   // [rows x n] row-major, the rows in the order asked for
  std::vector<std::uint8_t> good;             // [n] 1 = a good sample; empty when no bad spans were given
};

/** a recording that is NOT in the store: read in place from the file it was imported from (`import recording --in-place`). `read` fills `out` with samples [s0, s0 + n) of ALL channels,
 *  float32, row after row ([channels x n]); the same samples a copy would hold. The chunk service reads it exactly as it reads a stored array. */
struct LinkedBlocks {
  std::int64_t n_channels = 0, n_samples = 0;
  std::function<void(std::int64_t s0, std::int64_t n, float* out)> read;
};

class ChunkSource {
 public:
  /** `array`: the recording's key in `store` ([channels x samples]; unused when `linked` is given: a recording read in place from its file); `rows`: the channels handed over, in order (empty: all); `bad`: spans [a, b) in seconds on
   *  the recording's clock (a sample at t0 + i / sfreq is bad when a <= t < b, nxr.pipelines.parameters.bad_mask). */
  ChunkSource(const Store& store, std::string array, std::int64_t width, double sfreq, double t0, std::vector<std::int32_t> rows = {},
              std::optional<std::vector<std::pair<double, double>>> bad = std::nullopt, std::shared_ptr<const LinkedBlocks> linked = nullptr);
  std::int64_t width() const { return width_; }
  std::int64_t n_channels() const { return n_channels_; }
  std::int64_t n_samples() const { return n_samples_; }
  std::int64_t rows() const { return static_cast<std::int64_t>(rows_.size()); }
  std::int64_t count() const { return (n_samples_ + width_ - 1) / width_; }
  /** chunk k, WHOLE (its first sample in out.first, its length in out.n) */
  void read(std::int64_t k, Chunk& out) const;
  /** the chunks read so far (each read once per job: what a test counts) */
  std::int64_t reads() const;

 private:
  const Store* store_ = nullptr;   // the store must outlive the source (null: a linked recording)
  std::shared_ptr<const LinkedBlocks> linked_;
  std::string array_;    // the recording's key in it
  std::int64_t width_ = kChunk, n_channels_ = 0, n_samples_ = 0;
  double sfreq_ = 0, t0_ = 0;
  std::vector<std::int32_t> rows_;
  std::optional<std::vector<std::pair<double, double>>> bad_;
  std::shared_ptr<std::atomic<std::int64_t>> reads_ = std::make_shared<std::atomic<std::int64_t>>(0);
};

/** a job's state, as `progress` sees it after each chunk (the Studio's ChunkJob) */
struct ChunkJob {
  std::int64_t done = 0, count = 0;           // chunks folded, of count
  double seconds = 0;                         // since the job started (waiting in the queue excluded)
  std::vector<std::int64_t> visited;          // the chunks, in order, each once
  std::string state = "running";              // running · done · cancelled
};
struct ChunkLoopOptions {
  std::function<void(const ChunkJob&)> progress;   // after each chunk
  std::function<bool()> cancelled;                 // asked before each chunk: true stops the job (code::cancelled)
  std::optional<std::filesystem::path> queue;      // a lock file: one job at a time across processes (waits for the running job)
};
/** the process's options, used by a loop given none (the command line sets them once) */
void set_chunk_loop_defaults(ChunkLoopOptions o);
const ChunkLoopOptions& chunk_loop_defaults();

struct ChunkRun {
  std::vector<std::int64_t> visited;          // the chunks, in order, each once
  std::int64_t samples = 0, bad_samples = 0;  // samples handed over (span and halo); of them, bad
};
/** THE JOB: walk the chunks of the span with its halo in order; `visit` each once, cut to [s0 - before, s1 + after); `opt` (null: the process's defaults) */
ChunkRun chunk_loop(const ChunkSource& src, const Span& span, const std::function<void(const Chunk&)>& visit, const ChunkLoopOptions* opt = nullptr);

/** one chunk of a span WITH ITS HALO, as `haloed_loop` hands it over: the chunk's own samples [first, first + n) (the span's part of chunk k) and `left` /
 *  `right` samples of context before and after them, read from the neighbouring chunks (fewer only where the recording ends: what to do there is the
 *  computation's, as the Studio's convolveRows reflects) */
struct HaloedChunk {
  std::int64_t k = 0, first = 0, n = 0, left = 0, right = 0;
  std::vector<double> data;                   // [rows x (left + n + right)] row-major
  std::vector<std::uint8_t> good;             // [left + n + right]; empty when no bad spans were given
};
/** THE JOB WITH A HALO (the Studio's `#haloed`: a window read with H samples of context on each side from the neighbouring chunks, "read apart, kept
 *  nowhere"; nxr's runner, chunk k read as [s0 − H, s1 + H)): each chunk of the span handed to `visit` once, in order, with `halo` samples on each side.
 *  Every chunk is still read once (`chunk_loop` underneath: read-ahead, progress, cancel, the queue); the memory held is one chunk and its two halos. */
ChunkRun haloed_loop(const ChunkSource& src, const Span& span, std::int64_t halo, const std::function<void(const HaloedChunk&)>& visit, const ChunkLoopOptions* opt = nullptr);

}  // namespace nxr::io
