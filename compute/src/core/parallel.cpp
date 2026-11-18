// nxr::core (parallel.h).
#include "nxr/core/parallel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <string>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
#include <sched.h>
#endif

namespace nxr::core {

int performanceCores() {
#ifdef __EMSCRIPTEN__
  return 1;
#else
  int n = 0;
#ifdef __APPLE__
  std::size_t len = sizeof(n);
  if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, nullptr, 0) != 0) n = 0;
#endif
  if (n < 1) n = static_cast<int>(std::thread::hardware_concurrency());
  return std::max(1, n);
#endif
}

Allocation allocationFrom(const char* slurm, int affinity, double cgroup) {
  Allocation a;
  a.cpus = performanceCores();
  if (slurm && *slurm) {
    char* end = nullptr;
    const long v = std::strtol(slurm, &end, 10);
    if (end != slurm && v >= 1) { a.cpus = static_cast<int>(v); a.source = "SLURM_CPUS_PER_TASK"; a.limited = true; return a; }
  }
  const int hw = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  int n = a.cpus;
  const char* src = "performance_cores";
  if (affinity >= 1 && affinity < hw) { n = affinity; src = "affinity"; }
  if (cgroup > 0) { const int q = std::max(1, static_cast<int>(cgroup)); if (q < n || std::string(src) == "performance_cores") if (q < hw) { n = std::min(n, q); src = "cgroup"; } }
  a.cpus = n; a.source = src; a.limited = std::string(src) != "performance_cores";
  return a;
}

Allocation allocation() {
  int affinity = 0;
  double cgroup = 0;
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) affinity = CPU_COUNT(&set);
  {   // cgroup v2: "max 100000" or "<quota> <period>"; v1: cfs_quota_us / cfs_period_us
    std::ifstream f("/sys/fs/cgroup/cpu.max");
    std::string q; double period = 0;
    if (f >> q >> period && q != "max" && period > 0) cgroup = std::atof(q.c_str()) / period;
    else {
      std::ifstream fq("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), fp("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
      double qq = -1, pp = 0;
      if (fq >> qq && fp >> pp && qq > 0 && pp > 0) cgroup = qq / pp;
    }
  }
#endif
  return allocationFrom(std::getenv("SLURM_CPUS_PER_TASK"), affinity, cgroup);
}

int resolveThreads(int requested) {
#ifdef __EMSCRIPTEN__
  (void)requested;
  return 1;
#else
  const Allocation a = allocation();
  if (requested == 0) return a.cpus;
  const int n = std::max(1, requested);
  return a.limited ? std::min(n, a.cpus) : n;                       // never more than the allocation
#endif
}

int slots(int threads) { return threads <= 1 ? 1 : threads + 1; }

void orderedPieces(std::int64_t n, int threads, const std::function<void(std::int64_t, int)>& produce, const std::function<void(std::int64_t, int)>& consume) {
  if (n <= 0) return;
  if (threads <= 1 || n == 1) {
    for (std::int64_t i = 0; i < n; ++i) { produce(i, 0); consume(i, 0); }
    return;
  }
  const int nslots = slots(threads);
  const int workers = static_cast<int>(std::min<std::int64_t>(threads, n));
  std::mutex m;
  std::condition_variable cv;
  std::vector<char> ready(static_cast<std::size_t>(n), 0);
  std::int64_t consumed = 0;
  bool abort = false;
  std::exception_ptr err;
  std::atomic<std::int64_t> next{0};
  auto fail = [&](std::exception_ptr e) { std::lock_guard<std::mutex> l(m); if (!err) err = e; abort = true; cv.notify_all(); };
  auto work = [&] {
    for (;;) {
      const std::int64_t i = next.fetch_add(1);
      if (i >= n) return;
      {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return abort || i < consumed + nslots; });   // the slot of piece i - nslots is free once that piece is consumed
        if (abort) return;
      }
      try { produce(i, static_cast<int>(i % nslots)); } catch (...) { fail(std::current_exception()); return; }
      { std::lock_guard<std::mutex> l(m); ready[static_cast<std::size_t>(i)] = 1; }
      cv.notify_all();
    }
  };
  std::vector<std::thread> pool;
  try { for (int t = 0; t < workers; ++t) pool.emplace_back(work); } catch (...) { fail(std::current_exception()); }
  for (std::int64_t i = 0; i < n; ++i) {
    {
      std::unique_lock<std::mutex> l(m);
      cv.wait(l, [&] { return abort || ready[static_cast<std::size_t>(i)]; });
      if (abort) break;
    }
    try { consume(i, static_cast<int>(i % nslots)); } catch (...) { fail(std::current_exception()); break; }
    { std::lock_guard<std::mutex> l(m); consumed = i + 1; }
    cv.notify_all();
  }
  for (auto& t : pool) t.join();
  if (err) std::rethrow_exception(err);
}

void forRanges(std::int64_t n, int threads, const std::function<void(std::int64_t, std::int64_t)>& f) {
  if (n <= 0) return;
  const int w = static_cast<int>(std::min<std::int64_t>(std::max(1, threads), n));
  if (w <= 1) { f(0, n); return; }
  std::vector<std::thread> pool;
  std::mutex m;
  std::exception_ptr err;
  for (int t = 0; t < w; ++t) {
    const std::int64_t b = n * t / w, e = n * (t + 1) / w;
    pool.emplace_back([&, b, e] { try { f(b, e); } catch (...) { std::lock_guard<std::mutex> l(m); if (!err) err = std::current_exception(); } });
  }
  for (auto& t : pool) t.join();
  if (err) std::rethrow_exception(err);
}

}  // namespace nxr::core
