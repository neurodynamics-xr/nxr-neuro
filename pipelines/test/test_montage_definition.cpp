// `run montage` (pipelines/definitions/montage.yaml) on a synthetic template package: a sphere as the scalp `head` and the landmarks as the points manifold `head_landmarks` (as the template importers
// store them). The montage is stored as the points manifold `eeg_1005` with its names labelling, the `system` partition, aliases and origin; its positions equal nxr::meeg::tenFive with the
// inion the convention gives; the refusals (a second montage, a missing landmarks manifold, landmarks in another system).
#include "nxr/io/subject_api.h"
#include <unistd.h>

#include <cmath>
#include <filesystem>

#include "check.h"
#include "nxr/bst/entities.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/surface.h"
#include "nxr/io/task/task.h"
#include "nxr/io/writer.h"
#include "nxr/meeg/montage.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/runner.h"
#include "synthetic_meshes.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception&) { return "other"; }
  return "";
}
void add_landmarks(SubjectWriter& sub, const std::string& cs_key, const std::vector<float>& pos) {
  const std::string name = "head_landmarks", path = folder_for("points") + "/" + name, mid = uuid7();
  ManifoldSpec m;
  m.id = mid; m.name = name; m.path = path; m.type = "points"; m.n_vertices = 3;
  m.topology = {"none", 0, 1, std::nullopt, {}};
  GeometrySpec g;
  g.form = "stored"; g.cs_key = cs_key; g.positions_path = path;
  m.geometry = g;
  m.populate = [&sub, path, pos, mid](Store&, const std::string&) { sub.write_array(path, view(pos, {3, 3}), {}, std::nullopt, "manifold", mid); };
  sub.manifold(std::move(m));
  nxr::bst::LabellingOptions lo;
  lo.manifold_id = mid; lo.dictionary = name + "_names"; lo.scope = "subject";
  nxr::bst::labelling(sub, path + "_names", {0, 1, 2}, {"NAS", "LPA", "RPA"}, lo);
}
void make(const fs::path& pkg, const std::string& lm_cs) {
  ::Mesh m = ::icosphere(4);
  const double R = 0.095;
  for (auto& x : m.V) x *= R;
  CreatePackageOptions po;
  po.home_dataset = "demo";
  DatasetWriter w = create_package(pkg, "S", po);
  SubjectWriter sub = SubjectWriter::open(w, "S");
  SurfaceOptions so;
  so.name = "head"; so.primary = false; so.cs_key = "scs";
  const SurfaceInfo si = add_surface(sub, so, std::vector<double>(m.V), std::vector<std::int32_t>(m.F));
  add_landmarks(sub, lm_cs, {0, float(R), 0, -float(R), 0, 0, float(R), 0, 0});   // NAS, LPA, RPA
  set_origin(w, "manifold", si.id, OriginRecord{"fixture:scalp", "CC-BY-4.0", "A. Author", true});
  sub.complete();                                                    // (a manifold is found by name once it is complete)
  if (const auto r = nxr::io::find_manifold(sub, "head_landmarks")) set_origin(w, "manifold", text(*r, "id"), OriginRecord{"fixture:landmarks", "CC-BY-4.0", "A. Author", true});
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_pipelines_test_montage_definition_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  make(root / "A.nxr.zarr", "scs");
  const auto def = nxr::pipelines::graph::load("montage");
  nxr::pipelines::graph::RunOptions opt;
  {
    DatasetWriter w = open_package_writer(root / "A.nxr.zarr");
    SubjectWriter sub = SubjectWriter::open(w, "S");
    const auto rep = nxr::pipelines::graph::run(w, sub, def, Json::object(), opt).at(0);
    std::map<std::string, Json> stored;
    for (const auto& e : rep["stored"]) stored[e["name"].get<std::string>()] = e;
    CHECK(stored.count("eeg_1005") && stored.count("eeg_1005_names") && stored.count("eeg_1005_system"));
    const auto row = nxr::io::manifold_row_named(sub, "eeg_1005");
    CHECK(row.has_value());
    ArrayReader a = ArrayReader::open(sub.store(), text(*row, "pp"));
    CHECK_EQ(a.shape()[0], std::int64_t(345));
    CHECK_EQ(a.shape()[1], std::int64_t(3));
    const auto raw = a.read();
    std::vector<double> got(raw.size() / 8);
    std::memcpy(got.data(), raw.data(), raw.size());
    // the positions equal the compute function's with the convention's inion
    ::Mesh m = ::icosphere(4);
    const double R = 0.095;
    for (auto& x : m.V) x *= R;
    nxr::meeg::Landmarks L;
    L.nasion = {0, double(float(R)), 0}; L.lpa = {-double(float(R)), 0, 0}; L.rpa = {double(float(R)), 0, 0};
    L.inion = nxr::meeg::inionFromLandmarks(m.V, m.F, L.nasion, L.lpa, L.rpa);
    const auto want = nxr::meeg::tenFive(m.V, m.F, L);
    CHECK_EQ(want.labels.size(), static_cast<std::size_t>(345));
    for (std::size_t i = 0; i < got.size(); ++i) CHECK(std::abs(got[i] - want.positions[i]) < 1e-5);
    const Json pj = Json::parse(text(*row, "pj"));
    CHECK_EQ(pj["aliases"]["T3"].get<std::string>(), std::string("T7"));
    CHECK_EQ(pj["systems"]["10-20"].get<int>(), 21);
    CHECK_EQ(pj["inion"]["point"].size(), static_cast<std::size_t>(3));
  }
  // refusals: the landmarks in another coordinate system, and landmarks that do not exist
  make(root / "B.nxr.zarr", "mni");
  {
    DatasetWriter w = open_package_writer(root / "B.nxr.zarr");
    SubjectWriter sub = SubjectWriter::open(w, "S");
    CHECK(!code_of([&] { nxr::pipelines::graph::run(w, sub, def, Json::object(), opt); }).empty());                    // the scalp in scs, the landmarks in mni
    CHECK(!code_of([&] { nxr::pipelines::graph::run(w, sub, def, Json{{"landmarks", "nope"}}, opt); }).empty());      // no such landmarks
  }
  fs::remove_all(root);
  return nxrtest::finish("montage_definition");
}
