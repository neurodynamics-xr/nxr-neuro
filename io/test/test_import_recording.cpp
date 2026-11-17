// The importers of recordings (nxr/convert/recording.h) on the readers' synthetic files: the copy equals what the reader reads; --anonymize drops identifying fields;
// the origin record is on every item; refusals. Nothing here is a real recording.
#include <unistd.h>

#include <cmath>
#include <algorithm>
#include <cstring>
#include <set>
#include <filesystem>
#include <fstream>

#include "check.h"
#include "nxr/convert/recording.h"
#include "nxr/convert/sensor_frames.h"
#include "nxr/formats/byte_source.h"
#include "nxr/formats/fif_objects.h"
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

  // positions are in a TRUE coordinate system: a FIF file's MEG device positions and EEG head positions are carried into SCS by the file's dev_head_t and the
  // head -> SCS transform the digitised landmarks define, checked against MNE's own values (the golden: its dev_head_t and channel locations)
  {
    const fs::path pkg = root / "frames.nxr.zarr";
    import_into(pkg, "fif_basic_raw.fif", false);
    SubjectPackage p = SubjectPackage::open(pkg);
    const task::Subject s = task::Subject::of(p);
    const task::Node man = s.require({.cls = task::Cls::manifold, .name = "rec_channels"}, "the sensors");
    CHECK_EQ(man.fact("coordinate_system")["name"].get<std::string>(), std::string("scs"));
    const task::Geometry g = s.geometry(man.id);
    CHECK(g.coordinate_unit.has_value() && *g.coordinate_unit == "m");
    const ArrayReader ar = ArrayReader::open(p.store(), *g.positions_path);
    const std::int64_t N = ar.shape()[0];
    const std::vector<std::uint8_t> raw = ar.read({0, 0}, {N, 3});
    std::vector<float> pos(static_cast<std::size_t>(N * 3));
    std::memcpy(pos.data(), raw.data(), raw.size());
    std::ifstream gf(std::string(NXR_IO_DATA_DIR) + "/../../formats/test/golden/fif_golden.json");
    const Json gold = Json::parse(gf)["cases"]["basic"]["info"];
    const Json& T = gold["dev_head_t"]["trans"];
    CHECK_EQ(static_cast<std::size_t>(N), gold["chs"].size());
    for (std::int64_t i = 0; i < N; ++i) {
      const Json& c = gold["chs"][static_cast<std::size_t>(i)];
      double v[3] = {c["loc"][0].get<double>(), c["loc"][1].get<double>(), c["loc"][2].get<double>()}, h[3];
      if (c["coord_frame"].get<int>() == 1)
        for (int r = 0; r < 3; ++r) h[r] = T[r][0].get<double>() * v[0] + T[r][1].get<double>() * v[1] + T[r][2].get<double>() * v[2] + T[r][3].get<double>();
      else
        for (int r = 0; r < 3; ++r) h[r] = v[r];
      // the golden's landmarks are LPA (-0.08, 0, 0), nasion (0, 0.1, 0), RPA (0.08, 0, 0): SCS = (head y, -head x, head z)
      const double want[3] = {h[1], -h[0], h[2]};
      for (int r = 0; r < 3; ++r) CHECK(std::fabs(pos[static_cast<std::size_t>(i * 3 + r)] - want[r]) < 1e-6);
    }
    // the systems and their transitions: device -> head -> SCS, each a transition operator
    std::set<std::string> names;
    for (const auto& o : s.find_all({.cls = task::Cls::operator_, .kind = "transition"})) names.insert(o.name);
    CHECK(names.count("rec_channels_device_to_head") == 1);
    CHECK(names.count("rec_channels_head_to_scs") == 1);
    // the digitisation: a points manifold in the head frame it was digitised in (LPA first, then the nasion, RPA, an HPI coil, an EEG point, an extra point)
    const task::Node hp = s.require({.cls = task::Cls::manifold, .name = "rec_channels_headpoints"}, "the head points");
    CHECK_EQ(*hp.n_vertices, std::int64_t(6));
    CHECK_EQ(hp.fact("coordinate_system")["name"].get<std::string>(), std::string("head"));
    const ArrayReader hr = ArrayReader::open(p.store(), *s.geometry(hp.id).positions_path);
    const std::vector<std::uint8_t> hraw = hr.read({0, 0}, {6, 3});
    std::vector<float> hpos(18);
    std::memcpy(hpos.data(), hraw.data(), hraw.size());
    CHECK(std::fabs(hpos[0] + 0.08f) < 1e-6f && std::fabs(hpos[4] - 0.1f) < 1e-6f);
    CHECK(p.verify().ok);
  }
  // an ACTIVE SSP projector is an operator and is applied to the samples as they are copied; an inactive one is only an operator
  {
    const fs::path pkg = root / "proj.nxr.zarr";
    import_into(pkg, "fif_nodate_activeproj_raw.fif", false);
    SubjectPackage p = SubjectPackage::open(pkg);
    const task::Subject s = task::Subject::of(p);
    const task::Node op = s.require({.cls = task::Cls::operator_, .name = "rec_channels_projector_1"}, "the projector");
    CHECK_EQ(*op.kind, std::string("projector"));
    const task::Node rec = s.require({.cls = task::Cls::field, .name = "rec"}, "the recording");
    CHECK(rec.producer["projectors"][0]["active"].get<bool>());
    const ArrayReader ar = ArrayReader::open(p.store(), *rec.path);
    const std::int64_t C = ar.shape()[0], T = ar.shape()[1];
    std::vector<float> got(static_cast<std::size_t>(C * T)), raw(static_cast<std::size_t>(C * T));
    const std::vector<std::uint8_t> bytes = ar.read({0, 0}, {C, T});
    std::memcpy(got.data(), bytes.data(), bytes.size());
    auto reader = cv::open_recording("fif", data + "fif_nodate_activeproj_raw.fif");
    reader->read(0, static_cast<std::uint64_t>(T), 0, static_cast<std::size_t>(C), raw.data());
    const double v[3] = {0.618984043598175, -0.7750996947288513, 0.1268039047718048};   // MNE's projector vector over EEG000..2 (the golden)
    double worst = 0, scale = 0;
    for (std::int64_t t = 0; t < T; ++t) {
      double d = 0;
      for (int c = 0; c < 3; ++c) d += v[c] * raw[static_cast<std::size_t>(c * T + t)];
      for (int c = 0; c < 3; ++c) {
        const double want = raw[static_cast<std::size_t>(c * T + t)] - v[c] * d;
        worst = std::max(worst, std::fabs(got[static_cast<std::size_t>(c * T + t)] - want));
        scale = std::max(scale, std::fabs(want));
      }
      CHECK(got[static_cast<std::size_t>(3 * T + t)] == raw[static_cast<std::size_t>(3 * T + t)]);   // the other channel is untouched
    }
    CHECK(worst < 1e-5 * scale);
  }
  // the CTF compensation of a FIF file: an operator on the channels (MEG rows, reference columns), at the grade the data are at
  {
    const fs::path pkg = root / "comp.nxr.zarr";
    import_into(pkg, "fif_comp_raw.fif", false);
    SubjectPackage p = SubjectPackage::open(pkg);
    const task::Subject s = task::Subject::of(p);
    const task::Node op = s.require({.cls = task::Cls::operator_, .name = "rec_channels_ctf_compensation"}, "the compensation");
    CHECK_EQ(*op.kind, std::string("ctf compensation"));
    CHECK_EQ(op.fact("nnz").get<std::int64_t>(), std::int64_t(4));
    const task::Node rec = s.require({.cls = task::Cls::field, .name = "rec"}, "the recording");
    CHECK_EQ(rec.producer["ctf_compensation"]["grade"]["current"].get<int>(), 3);
    // the coil definitions the channels refer to (the forward model's integration points are looked up by these names)
    const task::Node man = s.require({.cls = task::Cls::manifold, .name = "rec_channels"}, "the sensors");
    CHECK(man.producer["coil_definitions"]["coil_types"].size() == 2);
    CHECK(man.producer["coil_definitions"]["source"].get<std::string>().find("coil_def.dat") != std::string::npos);
    CHECK(p.verify().ok);
  }
  // an MNE head -> MRI transform becomes a transition operator from the head frame to freesurfer-tkr
  {
    const fs::path pkg = root / "trans.nxr.zarr";
    import_into(pkg, "fif_basic_raw.fif", false);
    {
      DatasetWriter w = open_package_writer(pkg);
      SubjectWriter sub = SubjectWriter::open(w, "S");
      const task::Node man = task::Subject::of(sub).require({.cls = task::Cls::manifold, .name = "rec_channels"}, "the sensors");
      nxr::formats::fif::File f(std::make_shared<nxr::formats::FileSource>(data + "fif_obj-trans.fif"));
      cv::import_head_mri_transition(sub, nxr::formats::fif::read_transform(f), "rec_head_to_mri", man.id);
    }
    SubjectPackage p = SubjectPackage::open(pkg);
    const task::Subject s = task::Subject::of(p);
    const task::Node t = s.require({.cls = task::Cls::operator_, .name = "rec_head_to_mri"}, "the head to MRI transition");
    CHECK_EQ(t.fact("from_coordinate_system")["name"].get<std::string>(), std::string("head"));
    CHECK_EQ(t.fact("to_coordinate_system")["name"].get<std::string>(), std::string("freesurfer-tkr"));
    CHECK(p.verify().ok);
  }
  // EEGLAB's and BrainVision's frames are their own, with no transition; an unstated unit is not metres
  for (const auto& [file, key] : {std::pair<const char*, const char*>{"eeglab_fdt.set", "eeglab"}, {"bv_mux_i16.vhdr", "brainvision-sphere"}}) {
    const fs::path pkg = root / (std::string("own_") + key + ".nxr.zarr");
    import_into(pkg, file, false);
    SubjectPackage p = SubjectPackage::open(pkg);
    const task::Subject s = task::Subject::of(p);
    const task::Node man = s.require({.cls = task::Cls::manifold, .name = "rec_channels"}, "the sensors");
    if (man.fact("coordinate_system").is_object()) {
      CHECK_EQ(man.fact("coordinate_system")["name"].get<std::string>(), std::string(key));
      CHECK(man.fact("coordinate_system")["unit"].is_null());
    }
    CHECK(s.find_all({.cls = task::Cls::operator_, .kind = "transition"}).empty());
  }

  // refusals
  CHECK_EQ(code_of([&] { cv::format_of("x.txt"); }), std::string(code::invalid_argument));
  CHECK(!code_of([&] { import_into(root / "e.nxr.zarr", "eeglab_epoched.set", false); }).empty());   // epoched: refused
  CHECK(!code_of([&] { import_into(root / "t.nxr.zarr", "edf_truncated.edf", false); }).empty());
  fs::remove_all(root);
  return nxrtest::finish("import_recording");
}
