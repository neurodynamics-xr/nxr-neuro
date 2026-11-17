// The CIFTI importer (nxr/convert/cifti_import.h) on the readers' synthetic dense maps: cortical brain models on named surfaces through an indices Selection, maps as Fields
// (NaN where the file has no vertex), a dlabel only when named, voxel models reported as left out, refusals.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>

#include "check.h"
#include "nxr/convert/cifti_import.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/surface.h"

using namespace nxr::io;
namespace fs = std::filesystem;
namespace cv = nxr::convert;

static const std::string data = std::string(NXR_IO_DATA_DIR) + "/../../formats/test/data/cifti/";
template <class F> static std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception&) { return "other"; } return ""; }

// a closed cone-fan surface of n vertices (a ring of n - 2 and two poles)
static void add_cone(SubjectWriter& sub, const std::string& name, int n) {
  const int k = n - 2;
  std::vector<double> v;
  std::vector<std::int32_t> f;
  for (int i = 0; i < k; ++i) { v.push_back(std::cos(6.283185307 * i / k)); v.push_back(std::sin(6.283185307 * i / k)); v.push_back(0); }
  v.insert(v.end(), {0, 0, 1, 0, 0, -1});
  for (int i = 0; i < k; ++i) { const int j = (i + 1) % k; f.insert(f.end(), {i, j, k}); f.insert(f.end(), {j, i, k + 1}); }
  SurfaceOptions o;
  o.name = name;
  o.primary = false;
  add_surface(sub, o, v, f);
}

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_import_cifti_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  const fs::path pkg = root / "c.nxr.zarr";
  CreatePackageOptions po;
  po.home_dataset = "demo";
  {
    DatasetWriter w = create_package(pkg, "S", po);
    SubjectWriter sub = SubjectWriter::open(w, "S");
    sub.compose([&] { add_cone(sub, "lh", 20); add_cone(sub, "rh", 18); });
  }
  auto run = [&](const std::string& file, cv::CiftiImport o) {
    DatasetWriter w = open_package_writer(pkg);
    SubjectWriter sub = SubjectWriter::open(w, SubjectPackage::open(pkg).record().name);
    o.source = file;
    o.source_file = data + file;
    return cv::import_cifti(w, sub, o.source_file, o);
  };
  cv::CiftiImport o;
  o.name = "ds";
  o.left = "lh";
  o.right = "rh";
  const Json out = run("dscalar.nii", o);
  CHECK_EQ(out["structures"].size(), static_cast<std::size_t>(2));
  CHECK_EQ(out["skipped"][0].get<std::string>(), std::string("CIFTI_STRUCTURE_THALAMUS_LEFT"));
  SubjectPackage p = SubjectPackage::open(pkg);
  CHECK(p.verify().ok);
  CHECK(items_without_origin(p.db()).empty());
  const auto sel = p.db().one("SELECT * FROM selection WHERE name = 'ds_left_vertices'");
  CHECK(sel.has_value());
  CHECK_EQ(integer(*sel, "n_elements"), 5);
  const auto f0 = p.db().one("SELECT * FROM field WHERE name = 'ds_left_0'");
  CHECK(f0.has_value());
  const auto raw = ArrayReader::open(p.store(), *opt_text(*f0, "path")).read({0}, {20});
  std::vector<float> v(20);
  std::memcpy(v.data(), raw.data(), raw.size());
  CHECK(std::isnan(v[1]) && !std::isnan(v[0]) && !std::isnan(v[11]));   // the file covers vertices 0, 2, 3, 7, 11 of the left cortex
  // a dlabel only when its parcellation is named; a wrong vertex count refused; parcellated and series files refused
  CHECK_EQ(code_of([&] { run("dlabel.nii", o); }), std::string(code::invalid_argument));
  cv::CiftiImport l = o;
  l.name = "dl";
  l.parcellation = "toy";
  CHECK(run("dlabel.nii", l).contains("structures"));
  CHECK_EQ(code_of([&] { run("dtseries.nii", o); }), std::string(code::invalid_argument));
  CHECK_EQ(code_of([&] { run("pscalar.nii", o); }), std::string(code::invalid_argument));
  cv::CiftiImport bad = o;
  bad.name = "bad";
  bad.left = "rh";
  CHECK_EQ(code_of([&] { run("dscalar.nii", bad); }), std::string(code::invalid_argument));
  fs::remove_all(root);
  return nxrtest::finish("import_cifti");
}
