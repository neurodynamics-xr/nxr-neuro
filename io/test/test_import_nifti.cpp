// The NIfTI importer (nxr/convert/nifti_import.h) on the readers' goldens (made with nibabel): the voxel chart, the transition to the file's world system, the intensities,
// label volumes (only when named), the refusals.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "check.h"
#include "nxr/convert/nifti_import.h"
#include "nxr/formats/nifti.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/subject_package.h"

using namespace nxr::io;
namespace fs = std::filesystem;
namespace cv = nxr::convert;

static const std::string data = std::string(NXR_IO_DATA_DIR) + "/../../formats/test/data/";

template <class F> static std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception&) { return "other"; } return ""; }

static Json import_file(const fs::path& pkg, const std::string& file, const cv::NiftiImport& opt) {
  const PackageRecord rec = SubjectPackage::open(pkg).record();
  DatasetWriter w = open_package_writer(pkg);
  SubjectWriter sub = SubjectWriter::open(w, rec.name);
  cv::NiftiImport o = opt;
  o.source = file;
  o.source_file = file.find('/') == std::string::npos ? data + file : file;
  return cv::import_nifti(w, sub, o.source_file, o);
}

static fs::path make_package(const fs::path& pkg) {
  CreatePackageOptions po;
  po.home_dataset = "demo";
  DatasetWriter w = create_package(pkg, "S", po);
  return pkg;
}

static std::optional<Row> row(const SubjectPackage& p, const char* table, const std::string& name) {
  return p.db().one(std::string("SELECT * FROM ") + table + " WHERE name = ?", {V(name)});
}

static std::vector<double> read_numbers(const SubjectPackage& p, const std::string& path, std::vector<std::int64_t> shape) {
  std::size_t n = 1;
  for (auto s : shape) n *= static_cast<std::size_t>(s);
  const auto raw = ArrayReader::open(p.store(), path).read(std::vector<std::int64_t>(shape.size(), 0), shape);
  std::vector<double> v(n);
  if (raw.size() == n * 4) { std::vector<float> f(n); std::memcpy(f.data(), raw.data(), raw.size()); v.assign(f.begin(), f.end()); }
  else if (raw.size() == n * 8) std::memcpy(v.data(), raw.data(), raw.size());
  else CHECK(false);
  return v;
}

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_import_nifti_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  std::ifstream gf(std::string(NXR_IO_DATA_DIR) + "/../../formats/test/golden/nifti_golden.json");
  const Json g = Json::parse(gf)["cases"];

  // each golden volume: the chart's spacing, the transition (mm, the file's best affine), the values in C order
  struct Case { const char* key; const char* system; const char* unit; };
  for (const Case& c : {Case{"f32_sform", "nifti-aligned", "mm"}, Case{"i16_qform_scaled", "scanner-ras", "m"}, Case{"nifti2_f64", "nifti-aligned", "mm"}, Case{"f32_be", "nifti-aligned", "mm"}}) {
    const Json& gc = g[c.key];
    const fs::path pkg = make_package(root / (std::string(c.key) + ".nxr.zarr"));
    cv::NiftiImport o;
    o.name = "vol";
    const Json out = import_file(pkg, gc["file"].get<std::string>(), o);
    CHECK_EQ(out["transition"].get<std::string>(), std::string("vol_volume_") + c.system);
    SubjectPackage p = SubjectPackage::open(pkg);
    const auto vol = row(p, "manifold", "vol_volume");
    CHECK(vol.has_value());
    const auto dims = p.db().all("SELECT unit, spacing, n_vertices FROM geometry_dimension WHERE geometry_id = ? ORDER BY ordinal", {V(text(*vol, "geometry_id"))});
    CHECK_EQ(dims.size(), static_cast<std::size_t>(3));
    for (int d = 0; d < 3; ++d) {
      CHECK_EQ(text(dims[d], "unit"), std::string("mm"));
      CHECK(std::fabs(std::get<double>(dims[d].at("spacing")) - gc["zooms"][d].get<double>()) < 1e-6);
      CHECK_EQ(integer(dims[d], "n_vertices"), gc["shape"][d].get<std::int64_t>());
    }
    const auto cs = row(p, "coordinate_system", c.system);
    CHECK(cs.has_value());
    if (cs) CHECK_EQ(text(*cs, "unit"), std::string(c.unit));
    const auto tr = row(p, "operator", out["transition"].get<std::string>());
    CHECK(tr.has_value());
    CHECK_EQ(text(*tr, "kind"), std::string("transition"));
    const auto M = read_numbers(p, text(*tr, "path"), {3, 4});
    for (int i = 0; i < 12; ++i) CHECK(std::fabs(M[i] - gc["best_affine"][i].get<double>()) < 1e-5 * (1.0 + std::fabs(gc["best_affine"][i].get<double>())));
    // a voxel's position in the world, by the transition, is nibabel's
    // the values: nibabel's are x fastest, the field's are C order
    const std::int64_t ni = gc["shape"][0], nj = gc["shape"][1], nk = gc["shape"][2];
    const auto f = row(p, "field", "vol");
    CHECK(f.has_value());
    const auto v = read_numbers(p, text(*f, "path"), {ni, nj, nk});
    bool all = true;
    for (std::int64_t i = 0; i < ni; ++i)
      for (std::int64_t j = 0; j < nj; ++j)
        for (std::int64_t k = 0; k < nk; ++k) {
          const double want = gc["data"][static_cast<std::size_t>(i + ni * (j + nj * k))].get<double>();
          all = all && std::fabs(v[static_cast<std::size_t>((i * nj + j) * nk + k)] - want) < 2e-6 * std::fabs(want) + 1e-9;
        }
    CHECK(all);
    CHECK(items_without_origin(p.db()).empty());
    CHECK(p.verify().ok);
  }

  // a file with no transform code states no world frame: no transition, no guess
  {
    const fs::path pkg = make_package(root / "noform.nxr.zarr");
    cv::NiftiImport o;
    o.name = "n";
    const Json out = import_file(pkg, "nifti_noform.nii", o);
    CHECK(out["transition"].is_null());
    SubjectPackage p = SubjectPackage::open(pkg);
    CHECK(!p.db().one("SELECT id FROM operator WHERE kind = 'transition'").has_value());
    const auto vol = row(p, "manifold", "n_volume");
    const auto dims = p.db().all("SELECT spacing FROM geometry_dimension WHERE geometry_id = ? ORDER BY ordinal", {V(text(*vol, "geometry_id"))});
    CHECK(std::fabs(std::get<double>(dims[2].at("spacing")) - 2.5) < 1e-6);
  }

  // a MNI code is the `mni` system (mm); a label volume only when its parcellation is named
  {
    nxr::formats::nifti::Header h = nxr::formats::nifti::make_header(1, {4, 3, 2}, {2.0, 2.0, 2.0}, {2, 0, 0, -4, 0, 2, 0, -3, 0, 0, 2, -2});
    h.sform_code = 4;
    h.qform_code = 0;
    std::vector<double> vals(24);
    for (std::size_t i = 0; i < 24; ++i) vals[i] = static_cast<double>(i % 3 == 0 ? 0 : (i % 3) + 6);
    const std::string file = (root / "labels.nii").string();
    nxr::formats::nifti::write_file(file, h, vals);
    const fs::path pkg = make_package(root / "labels.nxr.zarr");
    cv::NiftiImport o;
    o.name = "lab";
    o.kind = "labels";
    CHECK_EQ(code_of([&] { import_file(pkg, file, o); }), std::string(code::invalid_argument));   // not named
    o.parcellation = "Demo";
    {
      std::ofstream(root / "names.txt") << "# code name\n7 left thing\n8 right thing\n";
      o.label_names = (root / "names.txt").string();
    }
    const Json out = import_file(pkg, file, o);
    CHECK_EQ(out["n_labels"].get<int>(), 3);
    SubjectPackage p = SubjectPackage::open(pkg);
    CHECK(row(p, "coordinate_system", "mni").has_value());
    CHECK(row(p, "operator", "lab_volume_mni").has_value());
    const auto sel = row(p, "selection", "lab");
    CHECK(sel.has_value());
    CHECK(p.db().one("SELECT name FROM dictionary_entry WHERE name = 'left thing'").has_value());
    CHECK(p.db().one("SELECT name FROM dictionary_entry WHERE name = 'unlabelled'").has_value());
    // the same volume as intensities, not named: refused as labels, taken as a volume
    cv::NiftiImport v;
    v.name = "lab2";
    v.kind = "labels";
    CHECK_EQ(code_of([&] { import_file(pkg, file, v); }), std::string(code::invalid_argument));
    // a name twice is refused
    cv::NiftiImport again;
    again.name = "lab";
    CHECK_EQ(code_of([&] { import_file(pkg, file, again); }), std::string(code::invalid_argument));
    CHECK(p.verify().ok);
  }

  // a series of volumes is refused
  {
    const fs::path pkg = make_package(root / "series.nxr.zarr");
    CHECK_EQ(code_of([&] { import_file(pkg, "nifti_u8_4d.nii.gz", {}); }), std::string(code::invalid_argument));
  }

  fs::remove_all(root);
  return nxrtest::finish("import_nifti");
}
