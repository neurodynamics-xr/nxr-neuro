// nxr::parallel: threads for the maths, with results BIT-IDENTICAL to the serial path (std::thread only; no new dependency).
// A function of matrices may split its work into independent pieces (blocks of samples, ranges of rows); the pieces are computed on `threads` workers, and what is folded across pieces (sums) is folded on the
// CALLING thread in the pieces' own order, so the sums are added in exactly the order the serial loop adds them. The thread count is therefore never part of a result or of a cache key.
#pragma once

#include <cstdint>
#include <functional>

namespace nxr::core {

/** The machine's performance cores (Apple: `hw.perflevel0.physicalcpu`; elsewhere the hardware's threads); at least 1. */
int performanceCores();
/** The CPUs this process may use, and where that number came from:
 *  `SLURM_CPUS_PER_TASK` when set; else the process's CPU affinity and the cgroup quota (Linux: the smaller); else the machine's performance cores. `source` says which
 *  ("SLURM_CPUS_PER_TASK", "affinity", "cgroup", "performance_cores"); `limited` is true when an allocation or a limit set it (not the machine's cores). */
struct Allocation { int cpus = 1; const char* source = "performance_cores"; bool limited = false; };
Allocation allocation();
/** The same from given inputs, for tests: the environment variable's text (null: unset), the affinity count (0: unknown), the cgroup quota in CPUs (0: none). */
Allocation allocationFrom(const char* slurm_cpus_per_task, int affinity_cpus, double cgroup_cpus);
/** A requested thread count: 0 = allocation().cpus; a positive request is capped at the allocation when one limits the process; negative or 1 = 1 (serial). WebAssembly has no threads: always 1. */
int resolveThreads(int requested);

/** Pieces 0 .. n-1 in order: `produce(i, slot)` runs on up to `threads` workers (slot = the buffer piece i writes into, 0 <= slot < slots(threads)); `consume(i, slot)` runs on the calling thread, once each, in the
 *  order i = 0, 1, 2, ... after produce(i, slot) is done (and before produce of piece i + slots(threads) may start on that slot). threads <= 1: a plain loop (slot 0). An exception in either is rethrown here. */
int slots(int threads);
void orderedPieces(std::int64_t n, int threads, const std::function<void(std::int64_t i, int slot)>& produce, const std::function<void(std::int64_t i, int slot)>& consume);

/** f(begin, end) over [0, n) cut into contiguous ranges, one per worker: for work whose pieces are independent (each writes its own output). */
void forRanges(std::int64_t n, int threads, const std::function<void(std::int64_t begin, std::int64_t end)>& f);

}  // namespace nxr::core
