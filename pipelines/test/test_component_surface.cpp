// A HEMISPHERE READ AS A SURFACE: io stores a cortex's hemispheres as components over the union's arrays (the vertices [from_index, from_index + n_vertices) and the
// faces wholly inside them); `read_surface` hands out that view, re-indexed from 0, never the union under the hemisphere's name (it did, so a hemisphere was "2 pieces").
//   - two separate icospheres stored as one cortex with two hemispheres: each hemisphere reads back as its own sphere (its positions, its faces, indices from 0)
//   - the union reads back unchanged
#include <unistd.h>

#include <filesystem>

#include "check.h"
#include "mesh_helpers.h"
#include "nxr/io.h"
#include "nxr/pipelines/input.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_component_surface_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};

  std::vector<double> U;
  std::vector<std::int32_t> F0;
  icosphere(2, U, F0);
  const std::int64_t nv = static_cast<std::int64_t>(U.size() / 3), nf = static_cast<std::int64_t>(F0.size() / 3);
  std::vector<double> pos = U;                                              // the right hemisphere: the same sphere moved 3 units along x
  for (std::int64_t i = 0; i < nv; ++i) { pos.push_back(U[3 * i] + 3.0); pos.push_back(U[3 * i + 1]); pos.push_back(U[3 * i + 2]); }
  std::vector<std::int32_t> faces = F0;
  for (std::int32_t f : F0) faces.push_back(f + static_cast<std::int32_t>(nv));

  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter a = SubjectWriter::create(ds, "S1");
    SurfaceOptions so{.name = "cortex"};
    so.hemispheres = {Hemisphere{"left", 0, nv}, Hemisphere{"right", nv, nv}};
    a.compose([&] { add_surface(a, so, pos, faces); });
  }
  SubjectWriter s = SubjectWriter::open(ds, "S1");

  const SurfaceInput whole = read_surface(s, std::string("cortex"), true);
  CHECK(whole.n_vertices() == 2 * nv && static_cast<std::int64_t>(whole.faces.size()) == 6 * nf);

  for (int k = 0; k < 2; ++k) {
    const SurfaceInput h = read_surface(s, std::string("cortex_") + std::to_string(k + 1), true);
    CHECK(h.n_vertices() == nv);
    CHECK(static_cast<std::int64_t>(h.faces.size()) == 3 * nf);
    CHECK(h.faces == F0);                                                   // re-indexed from 0: exactly the sphere's own faces
    for (std::int64_t i = 0; i < 3 * nv; ++i) CHECK(h.vertices[i] == pos[3 * k * nv + i]);
  }
  std::printf("component surfaces: each hemisphere reads back as its own closed piece (%lld vertices, %lld faces)\n", static_cast<long long>(nv), static_cast<long long>(nf));
  return nxrtest::finish("test_component_surface");
}
