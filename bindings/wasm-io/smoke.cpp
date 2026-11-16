// nxr-io in WebAssembly — the build's SMOKE TEST (not the binding: its JavaScript interface is not designed yet).
// Compiled with NODERAWFS, so in Node the WASM sees the real disk. Two modes:
//   write <datastore> <name>   create a dataset, a subject `sub-wasm` and an octahedron surface (as `surface add` does), then read it back
//   read  <dataset>            open a dataset (no lock), list its subjects and each one's surfaces with their vertex counts read from the arrays
// Prints one JSON line. The test (test/wasm_io_smoke.sh) checks the native CLI reads what this wrote, and this reads what the CLI wrote.
#include <iostream>

#include "nxr/io/array_reader.h"
#include "nxr/io/store.h"
#include "nxr/io/surface.h"
#include "nxr/io/writer.h"

using namespace nxr::io;

static Json read_dataset(const std::filesystem::path& folder) {
  OpenOptions oo;
  oo.lock = false;
  DatasetWriter ds = open_writer(folder, oo);
  Json subs = Json::array();
  const FolderStore store(folder);
  for (const Row& s : ds.db().all("SELECT id, name, kind, status FROM subject WHERE dataset_id = ? ORDER BY name", {ds.id()})) {
    Json surfs = Json::array();
    for (const Row& m : ds.db().all("SELECT m.name, g.positions_path, m.path AS mpath, sub.path AS spath FROM manifold m JOIN geometry g ON g.id = m.geometry_id JOIN subject sub ON sub.id = m.subject_id "
                                    "WHERE m.subject_id = ? AND m.type = 'surface' AND m.component_of_id IS NULL ORDER BY m.name", {V(text(s, "id"))})) {
      ArrayReader r = ArrayReader::open(store, join_key(text(m, "spath"), text(m, "positions_path")));
      surfs.push_back(Json{{"name", text(m, "name")}, {"values", r.read().size() / r.itemsize()}});
    }
    subs.push_back(Json{{"name", text(s, "name")}, {"kind", text(s, "kind")}, {"status", text(s, "status")}, {"surfaces", surfs}});
  }
  return Json{{"dataset", text(ds.row(), "name")}, {"subjects", subs}};
}

int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "write" && argc == 4) {
      {
        CreateOptions co;
        co.created_by = "nxr-io wasm smoke";
        DatasetWriter ds = create_dataset(argv[2], argv[3], co);
        SubjectWriter sub = SubjectWriter::create(ds, "sub-wasm");
        const std::vector<double> v = {0.01, 0, 0, -0.01, 0, 0, 0, 0.01, 0, 0, -0.01, 0, 0, 0, 0.01, 0, 0, -0.01};
        const std::vector<std::int32_t> f = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
        SurfaceOptions so;
        so.name = "octahedron";
        sub.compose([&] { add_surface(sub, so, v, f); });
      }
      std::cout << read_dataset(std::filesystem::path(argv[2]) / argv[3]).dump() << std::endl;
    } else if (mode == "read" && argc == 3) {
      std::cout << read_dataset(argv[2]).dump() << std::endl;
    } else {
      std::cerr << "usage: smoke write <datastore> <name> | read <dataset>\n";
      return 2;
    }
  } catch (const std::exception& e) {
    std::cout << Json{{"error", e.what()}}.dump() << std::endl;
    return 1;
  }
  return 0;
}
