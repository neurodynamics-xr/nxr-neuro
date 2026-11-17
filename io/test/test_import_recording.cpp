// The importers of recordings (nxr/convert/recording.h) on the readers' synthetic files: the copy equals what the reader reads; --anonymize drops identifying fields;
// the origin record is on every item; refusals. Nothing here is a real recording.
#include <unistd.h>

#include <cmath>
#include <filesystem>

#include "check.h"
#include "nxr/convert/recording.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/chunk_loop.h"
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

  // in place: the package links the file, holds no samples, and the chunk service reads the file's chunks equal to a copy's
  {
    const std::string f = "fif_basic_raw.fif";
    const fs::path pc = root / "ipc.nxr.zarr", pl = root / "ipl.nxr.zarr";
    for (const bool in_place : {false, true}) {
      CreatePackageOptions po;
      po.home_dataset = "demo";
      DatasetWriter w = create_package(in_place ? pl : pc, "S", po);
      SubjectWriter sub = SubjectWriter::open(w, "S");
      auto reader = cv::open_recording("fif", data + f);
      cv::RecordingImport opt;
      opt.name = "rec";
      opt.format = "fif";
      opt.source = f;
      opt.source_file = data + f;
      opt.in_place = in_place;
      cv::import_recording(w, sub, *reader, opt);
    }
    SubjectPackage a = SubjectPackage::open(pc), b = SubjectPackage::open(pl);
    CHECK(b.verify().ok);
    const task::Node ra = task::Subject::of(a).require({.cls = task::Cls::field, .name = "rec"}, "rec");
    const task::Node rb = task::Subject::of(b).require({.cls = task::Cls::field, .name = "rec"}, "rec");
    const Json link = cv::link_of(rb.producer);
    CHECK(link.is_object());
    CHECK_EQ(link["sha256"].get<std::string>().size(), static_cast<std::size_t>(64));
    CHECK(cv::link_of(ra.producer).is_null());
    const std::int64_t width = chunk_width(ra.fact("chunk_shape_json"));
    const ArrayReader arr = ArrayReader::open(b.store(), *rb.path);                // laid out (shape, type), holding no samples
    CHECK(arr.shape() == ArrayReader::open(a.store(), *ra.path).shape());
    const ChunkSource sa(a.store(), *ra.path, width, 1.0, 0.0), sl(b.store(), *rb.path, width, 1.0, 0.0, {}, std::nullopt, cv::open_linked(link));
    CHECK_EQ(sa.count(), sl.count());
    for (std::int64_t k = 0; k < sa.count(); ++k) {
      Chunk ca, cl;
      sa.read(k, ca);
      sl.read(k, cl);
      CHECK(ca.data == cl.data);
    }
  }

  // refusals
  CHECK_EQ(code_of([&] { cv::format_of("x.txt"); }), std::string(code::invalid_argument));
  CHECK(!code_of([&] { import_into(root / "e.nxr.zarr", "eeglab_epoched.set", false); }).empty());   // epoched: refused
  CHECK(!code_of([&] { import_into(root / "t.nxr.zarr", "edf_truncated.edf", false); }).empty());
  fs::remove_all(root);
  return nxrtest::finish("import_recording");
}
