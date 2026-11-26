// A synthetic two-hemisphere cortex's HEMISPHERE LABELLING, stored as Brainstorm's scout atlas `Structures` is (the `glue` definition's default input): the connected piece holding vertex 0 is
// `Cortex L`, the other `Cortex R`. For tests only: the glue reads the hemispheres from this labelling, never from the mesh.
#pragma once
#include <cstdint>
#include <numeric>
#include <vector>

#include "nxr/io/surface.h"

namespace nxrtest {

inline nxr::io::AtlasMember structures_of(const std::vector<double>& P, const std::vector<std::int32_t>& F) {
  const std::size_t n = P.size() / 3;
  std::vector<std::size_t> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](std::size_t x) { while (parent[x] != x) x = parent[x] = parent[parent[x]]; return x; };
  for (std::size_t f = 0; f + 2 < F.size(); f += 3) { parent[find(static_cast<std::size_t>(F[f]))] = find(static_cast<std::size_t>(F[f + 1])); parent[find(static_cast<std::size_t>(F[f + 1]))] = find(static_cast<std::size_t>(F[f + 2])); }
  std::vector<std::int32_t> codes(n);
  const std::size_t left = find(0);
  for (std::size_t v = 0; v < n; ++v) codes[v] = find(v) == left ? 0 : 1;
  return nxr::io::AtlasMember{"Structures", codes, {"Cortex L", "Cortex R"}, {{{200, 80, 80}}, {{80, 80, 200}}}};
}

}  // namespace nxrtest
