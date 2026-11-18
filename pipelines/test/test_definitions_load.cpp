// Every shipped definition LOADS (parses, resolves its steps and ports, passes the definition checker): each file is loaded on its own and every failure is printed, not only the first. Synthetic only: no dataset.
#include <cstdio>
#include <filesystem>
#include <string>

#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/registry.h"

namespace fs = std::filesystem;

int main() {
  int failures = 0, loaded = 0;
  for (const auto& e : fs::directory_iterator(NXR_TEST_DEFINITIONS_DIR)) {
    if (e.path().extension() != ".yaml") continue;
    const std::string name = e.path().stem().string();
    try {
      (void)nxr::pipelines::graph::load(name);
      ++loaded;
    } catch (const std::exception& ex) {
      std::printf("  %s: %s\n", name.c_str(), ex.what());
      ++failures;
    }
  }
  std::printf("  %d definitions load, %d do not\n", loaded, failures);
  std::printf("%s test_definitions_load\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
