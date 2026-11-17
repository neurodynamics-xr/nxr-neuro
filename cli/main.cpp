// nxr-neuro: the command line's entry. The commands, their grammar and their help are declared in cli/command_table.cpp and built by cli/registry.cpp.
#include <cstdlib>

#include "cli.h"
#include "nxr/io/deterministic.h"
#include "registry.h"

int main(int argc, char** argv) {
  // for the parity tests only: ids a counter (from the number given) and a fixed clock, as nxr's side is run with
  if (const char* det = std::getenv("NXR_IO_DETERMINISTIC")) nxr::io::testing::deterministic(true, std::strtoull(det, nullptr, 10));
  return cli::run_command_line(argc, argv);
}
