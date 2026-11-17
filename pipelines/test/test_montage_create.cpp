// `montage create` (pipelines/src/montage_create.cpp) on a synthetic template package: a sphere as the scalp `head` and the landmarks as the points manifold `head_landmarks` (as the template importers
// store them). The montage is stored as the points manifold `eeg_1005` with its names labelling, the `system` partition, aliases and origin; its positions equal nxr::montage::tenFive with the
// inion the convention gives; the refusals (a second montage, a missing landmarks manifold, landmarks in another system).
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
#include "nxr/montage.h"
#include "nxr/pipelines/montage_create.h"
#include "synthetic_meshes.h"

using namespace nxr::io;
namespace fs = std::filesystem;
using nxr::pipelines::MontageCreateOptions;

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
  set_origin(w.db(), "manifold", si.id, OriginRecord{"fixture:scalp", "CC-BY-4.0", "A. Author", true});
  for (const auto& r : sub.db().all("SELECT id FROM manifold WHERE name = 'head_landmarks'")) set_origin(w.db(), "manifold", text(r, "id"), OriginRecord{"fixture:landmarks", "CC-BY-4.0", "A. Author", true});
  sub.complete();
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_pipelines_test_montage_create_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  make(root / "A.nxr.zarr", "scs");
  MontageCreateOptions o;
  o.package = root / "A.nxr.zarr";
  const auto r = nxr::pipelines::montage_create(o);
  CHECK_EQ(r.info["points"].get<int>(), 345);
  CHECK_EQ(r.info["systems"]["10-20"].get<int>(), 21);
  CHECK_EQ(r.info["systems"]["10-10"].get<int>(), 56);      // 77 names of the 10-10 system, 21 of them the 10-20
  CHECK_EQ(r.info["systems"]["10-05"].get<int>(), 268);
  CHECK_EQ(r.info["cs_key"].get<std::string>(), std::string("scs"));
  CHECK_EQ(r.info["origin"]["licence"].get<std::string>(), std::string("CC-BY-4.0"));

  // stored: the manifold, its positions equal the compute function's with the convention's inion, the names, the system partition, the aliases, the origin
  {
    DatasetWriter w = open_package_writer(o.package);
    SubjectWriter sub = SubjectWriter::open(w, "S");
    const auto row = sub.db().one("SELECT m.id AS id, m.producer_json AS pj, g.positions_path AS pp FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.name = 'eeg_1005'");
    CHECK(row.has_value());
    const Json pj = Json::parse(text(*row, "pj"));
    CHECK_EQ(pj["aliases"]["T3"].get<std::string>(), std::string("T7"));
    CHECK_EQ(pj["inion"]["point"].size(), static_cast<std::size_t>(3));
    ArrayReader a = ArrayReader::open(sub.store(), text(*row, "pp"));
    CHECK_EQ(a.shape()[0], std::int64_t(345));
    const auto raw = a.read();
    std::vector<float> got(raw.size() / 4);
    std::memcpy(got.data(), raw.data(), raw.size());
    ::Mesh m = ::icosphere(4);
    const double R = 0.095;
    for (auto& x : m.V) x *= R;
    nxr::montage::Landmarks L;
    L.nasion = {0, double(float(R)), 0}; L.lpa = {-double(float(R)), 0, 0}; L.rpa = {double(float(R)), 0, 0};
    L.inion = nxr::montage::inionFromLandmarks(m.V, m.F, L.nasion, L.lpa, L.rpa);
    const auto want = nxr::montage::tenFive(m.V, m.F, L);
    CHECK_EQ(want.labels.size(), static_cast<std::size_t>(345));
    for (std::size_t i = 0; i < got.size(); ++i) CHECK(std::abs(got[i] - want.positions[i]) < 1e-6);
    const auto sels = sub.db().all("SELECT name FROM selection WHERE manifold_id = ?", {V(text(*row, "id"))});
    int found = 0;
    for (const auto& s : sels) { const std::string n = text(s, "name"); found += (n.find("eeg_1005_names") != std::string::npos) + (n.find("eeg_1005_system") != std::string::npos); }
    CHECK_EQ(found, 2);
    const auto org = origin_of(w.db(), "manifold", text(*row, "id"));
    CHECK(org.has_value() && org->redistributable && org->licence == "CC-BY-4.0");
  }
  // refusals
  CHECK_EQ(code_of([&] { nxr::pipelines::montage_create(o); }), std::string(code::invalid_argument));     // the montage exists
  make(root / "B.nxr.zarr", "mni");
  MontageCreateOptions b;
  b.package = root / "B.nxr.zarr";
  CHECK_EQ(code_of([&] { nxr::pipelines::montage_create(b); }), std::string(code::invalid_argument));     // landmarks in another system
  b.landmarks = "nope";
  CHECK_EQ(code_of([&] { nxr::pipelines::montage_create(b); }), std::string(code::invalid_argument));     // no such landmarks
  fs::remove_all(root);
  return nxrtest::finish("montage_create");
}
