// `--threads 0` means the CPUs this process may use (nxr::core::allocation): SLURM_CPUS_PER_TASK, else the affinity / cgroup quota, else the performance cores;
// never more than the allocation. Synthetic inputs and the environment variable only.
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "nxr/core/parallel.h"

static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::cerr << "FAIL line " << __LINE__ << ": " #x "\n"; ++failures; } } while (0)

int main() {
  using nxr::core::allocationFrom;
  // the variable wins over everything
  { const auto a = allocationFrom("4", 64, 0); CHECK(a.cpus == 4 && a.limited && std::strcmp(a.source, "SLURM_CPUS_PER_TASK") == 0); }
  { const auto a = allocationFrom("1", 0, 0); CHECK(a.cpus == 1 && a.limited); }
  // an unusable variable is ignored
  { const auto a = allocationFrom("zero", 0, 0); CHECK(!a.limited && a.cpus >= 1 && std::strcmp(a.source, "performance_cores") == 0); }
  { const auto a = allocationFrom("", 0, 0); CHECK(!a.limited); }
  // affinity and cgroup, when smaller than the machine
  const int hw = nxr::core::performanceCores();
  { const auto a = allocationFrom(nullptr, 2, 0); CHECK(a.cpus == (hw < 2 ? hw : 2) || a.cpus == 2); }
  { const auto a = allocationFrom(nullptr, 0, 0); CHECK(!a.limited && a.cpus == hw); }
  { const auto a = allocationFrom(nullptr, 0, 1.5); CHECK(a.cpus <= 1 || !a.limited); }
  // the real environment: set the variable, then `0` and a larger request resolve to it
  setenv("SLURM_CPUS_PER_TASK", "3", 1);
  CHECK(nxr::core::allocation().cpus == 3);
  CHECK(nxr::core::resolveThreads(0) == 3);
  CHECK(nxr::core::resolveThreads(8) == 3);                     // never more than allocated
  CHECK(nxr::core::resolveThreads(2) == 2);
  CHECK(nxr::core::resolveThreads(1) == 1);
  setenv("SLURM_CPUS_PER_TASK", "1", 1);
  CHECK(nxr::core::resolveThreads(0) == 1);
  unsetenv("SLURM_CPUS_PER_TASK");
  CHECK(nxr::core::resolveThreads(0) >= 1);
  if (failures) return 1;
  std::cout << "parallel allocation: ok\n";
  return 0;
}
