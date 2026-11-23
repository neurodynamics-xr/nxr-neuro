// THE ORIGIN OF A DERIVED ITEM: a run on inputs that carry ORIGIN records (a template's surfaces) gives every item it makes an origin record (its source naming the pipeline and
// its inputs, the inputs' licence and attribution, redistributable only when every input is), so a template package with derived items (its glued surface, its eigenmodes)
// is released with every item's terms. A run on inputs with no record writes none. Synthetic two-hemisphere brain, never a real dataset.
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/io/subject_package.h"
#include "run_definition.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
bool has(const std::string& s, const std::string& x) { return s.find(x) != std::string::npos; }

// a template package holding the two-hemisphere cortex with its Desikan-Killiany medial wall (the glue's input); with `terms`, every item given that origin
DatasetWriter make_package(const fs::path& folder, const Brain& b, const std::optional<OriginRecord>& terms) {
  CreatePackageOptions po;
  po.home_dataset = "templates";
  po.subject.kind = "template";
  DatasetWriter w = create_package(folder, "T", po);
  {
    SubjectWriter sub = SubjectWriter::open(w, "T");
    std::vector<std::int32_t> dk(static_cast<std::size_t>(b.n), 1);
    for (int i = 0; i < b.n; ++i) if (std::abs(b.P[static_cast<std::size_t>(3 * i + 1)]) < 0.6) dk[static_cast<std::size_t>(i)] = 0;
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      so.atlases.push_back(AtlasMember{"Desikan-Killiany", dk, {"unassigned", "cortex"}, {{{0, 0, 0}}, {{200, 200, 200}}}});
      add_surface(sub, so, b.P, b.F);
    });
  }
  if (terms) for (const auto& it : package_items(w.db())) set_origin(w.db(), it.table, it.id, *terms);
  return w;
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_derived_origin_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(3, false);

  // inputs WITH origin records: every item the run makes carries the inputs' terms
  {
    const OriginRecord terms{"a synthetic brain", "Licence A", "A. Author", true};
    DatasetWriter w = make_package(root / "with.nxr.zarr", b, terms);
    CHECK(items_without_origin(w.db()).empty());
    SubjectWriter sub = SubjectWriter::open(w, "T");
    nxrtest::run_definition(w, sub, "glue", Json{{"surface", "cortex"}});
    CHECK(items_without_origin(w.db()).empty());                              // the glued surface, its map and its bridge: none left without
    const auto glued = w.db().one("SELECT id FROM manifold WHERE name = 'cortex_glued'");
    CHECK(glued.has_value());
    const auto o = origin_of(w.db(), "manifold", text(*glued, "id"));
    CHECK(o && has(o->source, "pipeline glue") && has(o->source, "cortex") && o->licence == "Licence A" && o->attribution == "A. Author" && o->redistributable);
  }

  // an input that is NOT redistributable makes the derived items not redistributable
  {
    const OriginRecord terms{"a synthetic brain", "Licence B", "B. Author", false};
    DatasetWriter w = make_package(root / "closed.nxr.zarr", b, terms);
    SubjectWriter sub = SubjectWriter::open(w, "T");
    nxrtest::run_definition(w, sub, "glue", Json{{"surface", "cortex"}});
    const auto glued = w.db().one("SELECT id FROM manifold WHERE name = 'cortex_glued'");
    const auto o = origin_of(w.db(), "manifold", text(*glued, "id"));
    CHECK(o && !o->redistributable && o->licence == "Licence B");
  }

  // inputs with NO record (a subject's own data): nothing is written
  {
    DatasetWriter w = make_package(root / "without.nxr.zarr", b, std::nullopt);
    SubjectWriter sub = SubjectWriter::open(w, "T");
    nxrtest::run_definition(w, sub, "glue", Json{{"surface", "cortex"}});
    CHECK(w.db().all("SELECT 1 FROM origin").empty());
  }
  return nxrtest::finish("test_derived_origin");
}
