// The importers of recordings (nxr/convert/recording.h) on the readers' synthetic files: the copy equals what the reader reads; --anonymize drops identifying fields;
// the origin record is on every item; refusals. Nothing here is a real recording.
#include <unistd.h>

#include <cmath>
#include <filesystem>

#include "check.h"
#include "nxr/convert/recording.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/task/task.h"

using namespace nxr::io;
namespace fs = std::filesystem;
namespace cv = nxr::convert;

template <class F> static std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception&) { return "other"; } return ""; }

static const std::string data = std::string(NXR_IO_DATA_DIR) + "/../../formats/test/data/";

static Json import_into(const fs::path& pkg, const std::string& file, bool anonymize, const std::string& name = "rec") {
  CreatePackageOptions po;
  po.home_dataset = "demo";
  Json out;
  {
    DatasetWriter w = create_package(pkg, "S", po);
    SubjectWriter sub = SubjectWriter::open(w, "S");
    const std::string fmt = cv::format_of(file);
    auto reader = cv::open_recording(fmt, data + file);
    cv::RecordingImport opt;
    opt.name = name;
    opt.format = fmt;
    opt.source = file;
    opt.source_file = data + file;
    opt.anonymize = anonymize;
    out = cv::import_recording(w, sub, *reader, opt);
  }
  return out;
}

// the stored recording equals the reader's samples, block after block
static void check_copy(const fs::path& pkg, const std::string& file, const std::string& name = "rec") {
  SubjectPackage p = SubjectPackage::open(pkg);
  const task::Subject s = task::Subject::of(p);
  const task::Node rec = s.require({.cls = task::Cls::field, .name = name}, "the imported recording");
  CHECK(rec.path.has_value());
  const ArrayReader ar = ArrayReader::open(p.store(), *rec.path);
  auto reader = cv::open_recording(cv::format_of(file), data + file);
  const auto& h = reader->header();
  CHECK_EQ(static_cast<std::uint64_t>(ar.shape()[0]), static_cast<std::uint64_t>(h.channels.size()));
  CHECK_EQ(static_cast<std::uint64_t>(ar.shape()[1]), h.n_samples);
  const std::int64_t C = ar.shape()[0], T = ar.shape()[1];
  const std::vector<std::uint8_t> raw = ar.read({0, 0}, {C, T});
  std::vector<float> want(static_cast<std::size_t>(C * T));
  reader->read(0, static_cast<std::uint64_t>(T), 0, static_cast<std::size_t>(C), want.data());
  CHECK_EQ(raw.size(), want.size() * sizeof(float));
  CHECK(std::memcmp(raw.data(), want.data(), raw.size()) == 0);
}

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_import_recording_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);

  // every format's synthetic file is copied as the reader reads it
  int n = 0;
  for (const char* f : {"edf_plain.edf", "bdf_plain.bdf", "bv_mux_i16.vhdr", "eeglab_fdt.set", "fif_basic_raw.fif", "fif_skip_raw.fif"}) {
    const fs::path pkg = root / ("p" + std::to_string(n++) + ".nxr.zarr");
    const Json out = import_into(pkg, f, false);
    CHECK(out.contains("origin"));
    CHECK_EQ(out["origin"]["schema"].get<std::string>(), std::string("nxr-origin/1"));
    check_copy(pkg, f);
    SubjectPackage p = SubjectPackage::open(pkg);
    CHECK(items_without_origin(p.db()).empty());                                  // an origin record on every item
    CHECK(p.verify().ok);
  }

  // identifying header fields: kept as the file has them, dropped with --anonymize
  {
    const Json kept = import_into(root / "k.nxr.zarr", "edf_plain.edf", false);
    const Json anon = import_into(root / "a.nxr.zarr", "edf_plain.edf", true);
    SubjectPackage pk = SubjectPackage::open(root / "k.nxr.zarr"), pa = SubjectPackage::open(root / "a.nxr.zarr");
    const auto rec_k = task::Subject::of(pk).require({.cls = task::Cls::field, .name = "rec"}, "rec");
    const auto rec_a = task::Subject::of(pa).require({.cls = task::Cls::field, .name = "rec"}, "rec");
    CHECK(rec_k.producer["anonymized"] == false);
    CHECK(rec_a.producer["anonymized"] == true);
    bool any_k = false;
    for (auto it = rec_k.producer["header"].begin(); it != rec_k.producer["header"].end(); ++it) {
      const std::string k = it.key();
      if (k.find("patient") != std::string::npos || k.find("recording") != std::string::npos) any_k = true;
    }
    for (auto it = rec_a.producer["header"].begin(); it != rec_a.producer["header"].end(); ++it) {
      const std::string k = it.key();
      CHECK(k.find("patient") == std::string::npos && k.find("startdate") == std::string::npos);
    }
    CHECK(any_k);
    CHECK(rec_k.producer["header"].size() > rec_a.producer["header"].size());                 // something identifying was dropped
    check_copy(root / "a.nxr.zarr", "edf_plain.edf");                              // the samples are the same
  }

  // refusals
  CHECK_EQ(code_of([&] { cv::format_of("x.txt"); }), std::string(code::invalid_argument));
  CHECK(!code_of([&] { import_into(root / "e.nxr.zarr", "eeglab_epoched.set", false); }).empty());   // epoched: refused
  CHECK(!code_of([&] { import_into(root / "t.nxr.zarr", "edf_truncated.edf", false); }).empty());
  fs::remove_all(root);
  return nxrtest::finish("import_recording");
}
