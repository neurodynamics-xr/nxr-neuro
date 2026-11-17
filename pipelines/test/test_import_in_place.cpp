// `import recording --in-place` (nxr/convert/recording.h): a recording linked to its file, read by the chunk service from the original, gives the same results as a copied
// one; the package records the file's path and SHA-256 and the chunk service refuses a file that changed. Synthetic files only (the readers' goldens).
#include <unistd.h>

#include <filesystem>
#include <fstream>

#include "psd_run.h"
#include "check.h"
#include "nxr/convert/recording.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/task/task.h"
#include "nxr/pipelines/meeg.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
namespace cv = nxr::convert;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception&) { return "other"; }
  return "";
}

void import_into(const fs::path& pkg, const std::string& file, bool in_place) {
  CreatePackageOptions po;
  po.home_dataset = "demo";
  DatasetWriter w = create_package(pkg, "S", po);
  SubjectWriter sub = SubjectWriter::open(w, "S");
  const std::string fmt = cv::format_of(file);
  auto reader = cv::open_recording(fmt, file);
  cv::RecordingImport opt;
  opt.name = "rec";
  opt.format = fmt;
  opt.source = fs::path(file).filename().string();
  opt.source_file = file;
  opt.in_place = in_place;
  cv::import_recording(w, sub, *reader, opt);
}

// `name`: a new field each time, so a run is never a cache hit (which reads nothing)
Eigen::MatrixXd psd_of(const fs::path& pkg, const std::string& name = "psd_welch") {
  DatasetWriter w = open_package_writer(pkg);
  SubjectWriter sub = SubjectWriter::open(w, "S");
  const nxrtest::PsdRun r = nxrtest::run_psd(w, sub, "rec", Json{{"name", name}});
  std::vector<std::int64_t> shape;
  const std::vector<float> v = nxrtest::psd_values(sub, r, &shape);
  Eigen::MatrixXd m(shape[0], shape[1]);
  for (std::int64_t i = 0; i < shape[0]; ++i) for (std::int64_t k = 0; k < shape[1]; ++k) m(i, k) = v[static_cast<std::size_t>(i * shape[1] + k)];
  return m;
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_test_import_in_place_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  const std::string data = NXR_FORMATS_DATA_DIR;
  // a private copy of the file, so the test can change it
  const std::string file = (root / "edf_plain.edf").string();
  fs::copy_file(data + "/edf_plain.edf", file);

  import_into(root / "copy.nxr.zarr", file, false);
  import_into(root / "link.nxr.zarr", file, true);

  // the package records the link (path, size, SHA-256), the copy does not; a linked recording stores no samples
  {
    SubjectPackage p = SubjectPackage::open(root / "link.nxr.zarr");
    CHECK(p.verify().ok);
    SubjectPackage c = SubjectPackage::open(root / "copy.nxr.zarr");
    const auto link_row = nxr::io::task::Subject::of(p).require({.cls = nxr::io::task::Cls::field, .name = "rec"}, "rec");
    const Json link = cv::link_of(link_row.producer);
    CHECK(link.is_object());
    CHECK_EQ(link["format"].get<std::string>(), std::string("edf"));
    CHECK_EQ(link["sha256"].get<std::string>().size(), static_cast<std::size_t>(64));
    CHECK(cv::link_of(nxr::io::task::Subject::of(c).require({.cls = nxr::io::task::Cls::field, .name = "rec"}, "rec").producer).is_null());
  }

  // the same computation on both gives the same numbers
  const auto a = psd_of(root / "copy.nxr.zarr");
  const auto b = psd_of(root / "link.nxr.zarr");
  CHECK(a.size() > 0);
  CHECK(a.rows() == b.rows() && a.cols() == b.cols());
  CHECK((a - b).cwiseAbs().maxCoeff() == 0.0);

  // a file that changed is refused (the same size, other bytes), by the chunk service
  {
    std::fstream f(file, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(-2, std::ios::end);
    f.put('\x7f');
  }
  CHECK_EQ(code_of([&] { psd_of(root / "link.nxr.zarr", "again"); }), std::string(code::hash_mismatch));
  CHECK(code_of([&] { psd_of(root / "copy.nxr.zarr", "again"); }).empty());   // the copy does not need the file
  fs::remove(file);
  CHECK_EQ(code_of([&] { psd_of(root / "link.nxr.zarr", "third"); }), std::string(code::not_found));

  // a folder (CTF) cannot be linked
  CHECK_EQ(code_of([&] { cv::link_record("ctf", root.string()); }), std::string(code::invalid_argument));
  fs::remove_all(root);
  return nxrtest::finish("import_in_place");
}
