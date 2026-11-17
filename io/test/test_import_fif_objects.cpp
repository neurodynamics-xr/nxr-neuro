// The importers of MNE's FIF object files (nxr/convert/fif_objects.h) on the readers' goldens (made with MNE-Python): what each file becomes in a package.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "check.h"
#include "nxr/bst/sensors.h"
#include "nxr/convert/fif_objects.h"
#include "nxr/convert/synthetic.h"
#include "nxr/formats/byte_source.h"
#include "nxr/formats/fif_objects.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/subject_package.h"

using namespace nxr::io;
namespace fs = std::filesystem;
namespace cv = nxr::convert;
namespace fif = nxr::formats::fif;

static const std::string data = std::string(NXR_IO_DATA_DIR) + "/../../formats/test/data/";

template <class F> static std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception&) { return "other"; } return ""; }

static nxr::bst::Value text_value(const std::string& s) {
  nxr::bst::Value v;
  v.cls = nxr::bst::Class::Char;
  v.dims = {1, s.size()};
  v.text = {s};
  return v;
}

// a package whose sensor manifold has these channels (no positions): what a recording import would have made
static void make_package(const fs::path& pkg, const std::vector<std::string>& names, bool with_recording) {
  CreatePackageOptions po;
  po.home_dataset = "demo";
  DatasetWriter w = create_package(pkg, "S", po);
  SubjectWriter sub = SubjectWriter::open(w, "S");
  if (with_recording) {
    cv::SyntheticRecording s;
    s.n_channels = 4;
    s.duration = 20;
    cv::create_synthetic(w, sub, s);
    return;
  }
  sub.compose([&] {
    nxr::bst::Value ch;
    ch.cls = nxr::bst::Class::Struct;
    ch.dims = {1, names.size()};
    ch.fields = {"Name", "Type", "Loc", "Orient"};
    for (const auto& n : names) {
      ch.elems.push_back(text_value(n));
      ch.elems.push_back(text_value("MEG"));
      ch.elems.emplace_back();
      ch.elems.emplace_back();
    }
    nxr::bst::SensorOptions so;
    so.name = "sensors";
    so.acquisition = "rest";
    so.placed = false;
    nxr::bst::export_sensors(sub, {{"Channel", std::move(ch)}}, {}, so);
  });
}

static Json import_file(const fs::path& pkg, const std::string& file, cv::FifObjectImport opt) {
  const PackageRecord rec = SubjectPackage::open(pkg).record();
  DatasetWriter w = open_package_writer(pkg);
  SubjectWriter sub = SubjectWriter::open(w, rec.name);
  opt.source = file;
  opt.source_file = data + file;
  return cv::import_fif_object(w, sub, data + file, opt);
}

static std::vector<std::string> names_of(const std::string& file) {
  auto f = fif::File(std::make_shared<nxr::formats::FileSource>(data + file));
  std::vector<std::string> n;
  if (file.find("-fwd") != std::string::npos) for (const auto& c : fif::read_forward(f).parent.channels) n.push_back(c.name);
  else if (file.find("-inv") != std::string::npos) for (const auto& c : fif::read_inverse(f).parent.channels) n.push_back(c.name);
  else n = fif::read_covariance(f).names;
  return n;
}

static std::vector<float> read_f32(const SubjectPackage& p, const std::string& path, std::int64_t r, std::int64_t c) {
  const ArrayReader ar = ArrayReader::open(p.store(), path);
  const auto raw = ar.read({0, 0}, {r, c});
  std::vector<float> v(raw.size() / 4);
  std::memcpy(v.data(), raw.data(), raw.size());
  return v;
}

static std::optional<Row> row(const SubjectPackage& p, const char* table, const std::string& name) {
  return p.db().one(std::string("SELECT * FROM ") + table + " WHERE name = ?", {V(name)});
}

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_import_fif_objects_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  const std::vector<std::string> none;

  // detection
  CHECK_EQ(cv::fif_object_kind(data + "fif_obj_surf-src.fif"), std::string("src"));
  CHECK_EQ(cv::fif_object_kind(data + "fif_obj_free-fwd.fif"), std::string("fwd"));
  CHECK_EQ(cv::fif_object_kind(data + "fif_obj-inv.fif"), std::string("inv"));
  CHECK_EQ(cv::fif_object_kind(data + "fif_obj_full-cov.fif"), std::string("cov"));
  CHECK_EQ(cv::fif_object_kind(data + "fif_obj-eve.fif"), std::string("eve"));
  CHECK_EQ(cv::fif_object_kind(data + "fif_obj-annot.fif"), std::string("annot"));
  CHECK_EQ(cv::fif_object_kind(data + "fif_obj-trans.fif"), std::string("trans"));

  // a surface source space: the full surface in the MRI frame (metres) and a Selection of the used vertices
  {
    const fs::path pkg = root / "src.nxr.zarr";
    make_package(pkg, none, false);
    cv::FifObjectImport opt;
    opt.name = "ss";
    const Json out = import_file(pkg, "fif_obj_surf-src.fif", opt);
    SubjectPackage p = SubjectPackage::open(pkg);
    auto f = fif::File(std::make_shared<nxr::formats::FileSource>(data + "fif_obj_surf-src.fif"));
    const fif::SourceSpaces src = fif::read_source_spaces(f);
    std::int64_t np = 0, nuse = 0;
    for (const auto& s : src.spaces) { np += s.np; nuse += s.nuse; }
    CHECK_EQ(out["n_vertices"].get<std::int64_t>(), np);
    CHECK_EQ(out["n_used"].get<std::int64_t>(), nuse);
    const auto surf = row(p, "manifold", "ss_surface");
    CHECK(surf.has_value());
    CHECK_EQ(integer(*surf, "n_vertices"), np);
    const auto cs = p.db().one("SELECT cs.name AS name, cs.unit AS unit FROM manifold m JOIN geometry g ON g.id = m.geometry_id JOIN coordinate_system cs ON cs.id = g.coordinate_system_id WHERE m.id = ?", {V(text(*surf, "id"))});
    CHECK_EQ(text(*cs, "name"), std::string("freesurfer-tkr"));
    CHECK_EQ(text(*cs, "unit"), std::string("m"));
    const auto sel = row(p, "selection", "ss_source_space");
    CHECK(sel.has_value());
    CHECK_EQ(text(*sel, "type"), std::string("indices"));
    CHECK_EQ(text(*sel, "manifold_id"), text(*surf, "id"));
    const auto raw = ArrayReader::open(p.store(), text(*sel, "path")).read({0}, {nuse});
    std::vector<std::int32_t> got(static_cast<std::size_t>(nuse));
    std::memcpy(got.data(), raw.data(), got.size() * 4);
    std::vector<std::int32_t> want;
    std::int64_t off = 0;
    for (const auto& s : src.spaces) { for (auto v : s.vertno) want.push_back(static_cast<std::int32_t>(v + off)); off += s.np; }
    CHECK(got == want);
    CHECK(items_without_origin(p.db()).empty());
    CHECK(p.verify().ok);
    // a second import onto the first's surface
    cv::FifObjectImport on;
    on.name = "again";
    on.surface = "ss_surface";
    import_file(pkg, "fif_obj_surf-src.fif", on);
    SubjectPackage q = SubjectPackage::open(pkg);
    CHECK(row(q, "selection", "again_source_space").has_value());
    CHECK(!row(q, "manifold", "again_surface").has_value());
    cv::FifObjectImport bad;
    bad.name = "bad";
    bad.surface = "nothing";
    CHECK_EQ(code_of([&] { import_file(pkg, "fif_obj_surf-src.fif", bad); }), std::string(code::invalid_argument));
  }

  // a volume source space is a lattice in its voxel chart with a transition in millimetres; a discrete one a points manifold
  {
    const fs::path pkg = root / "vol.nxr.zarr";
    make_package(pkg, none, false);
    // MNE-Python's golden writes its "volume" as a discrete space, so a true volume source space is made here: a 4 x 3 x 2 grid of 5 mm voxels, F-ordered vertices
    fif::SourceSpace vs;
    vs.type = "vol";
    vs.coord_frame = 5;
    vs.np = 24;
    vs.rr.assign(72, 0.0);
    vs.nn.assign(72, 0.0);
    vs.shape = {4, 3, 2};
    fif::CoordTrans vt;
    vt.from = 2001;
    vt.to = 5;
    vt.trans = {0.005, 0, 0, -0.01, 0, 0.005, 0, -0.0075, 0, 0, 0.005, 0.0, 0, 0, 0, 1};
    vs.vox_mri_t = vt;
    vs.inuse.assign(24, 0);
    for (int v : {0, 1, 5, 13, 23}) { vs.inuse[v] = 1; vs.vertno.push_back(v); }
    vs.nuse = 5;
    fif::SourceSpaces vss;
    vss.spaces.push_back(vs);
    nxr::formats::MemorySink sink;
    fif::write_source_spaces_file(sink, vss, fif::Id{});
    const fs::path vfile = root / "vol-src.fif";
    { std::ofstream o(vfile, std::ios::binary); o.write(reinterpret_cast<const char*>(sink.data().data()), static_cast<std::streamsize>(sink.data().size())); }
    cv::FifObjectImport opt;
    opt.name = "v";
    {
      const PackageRecord rec = SubjectPackage::open(pkg).record();
      DatasetWriter w = open_package_writer(pkg);
      SubjectWriter sub = SubjectWriter::open(w, rec.name);
      cv::import_fif_object(w, sub, vfile.string(), opt);
    }
    cv::FifObjectImport d;
    d.name = "d";
    import_file(pkg, "fif_obj_discrete-src.fif", d);
    SubjectPackage p = SubjectPackage::open(pkg);
    const auto vol = row(p, "manifold", "v_volume");
    CHECK(vol.has_value());
    CHECK_EQ(text(*vol, "type"), std::string("volume"));
    const auto dims = p.db().all("SELECT unit, spacing FROM geometry_dimension WHERE geometry_id = ? ORDER BY ordinal", {V(text(*vol, "geometry_id"))});
    CHECK_EQ(dims.size(), static_cast<std::size_t>(3));
    CHECK_EQ(text(dims[0], "unit"), std::string("mm"));
    CHECK(std::fabs(std::get<double>(dims[0].at("spacing")) - 5.0) < 1e-4);
    const auto tr = row(p, "operator", "v_volume_tkr");
    CHECK(tr.has_value());
    CHECK_EQ(text(*tr, "kind"), std::string("transition"));
    CHECK(row(p, "selection", "v_source_space").has_value());
    CHECK(row(p, "manifold", "d_points").has_value());
    CHECK(row(p, "selection", "d_source_space").has_value());
    CHECK(p.verify().ok);
  }

  // covariances: a statistic on the channels' Selection, detached, with nfree as its members
  for (const char* file : {"fif_obj_full-cov.fif", "fif_obj_diag-cov.fif"}) {
    const fs::path pkg = root / (std::string(file) + ".nxr.zarr");
    const auto names = names_of(file);
    make_package(pkg, names, false);
    cv::FifObjectImport opt;
    opt.name = "c";
    import_file(pkg, file, opt);
    SubjectPackage p = SubjectPackage::open(pkg);
    const auto st = row(p, "statistic", "c_noise_cov");
    CHECK(st.has_value());
    CHECK_EQ(text(*st, "measure"), std::string("covariance"));
    CHECK(items_without_origin(p.db()).empty());
    // a channel the package lacks is refused, by name
    const fs::path pk2 = root / (std::string(file) + "-missing.nxr.zarr");
    make_package(pk2, std::vector<std::string>(names.begin() + 1, names.end()), false);
    CHECK_EQ(code_of([&] { import_file(pk2, file, opt); }), std::string(code::invalid_argument));
  }

  // a forward solution is an operator [channels x components] with both Selections; the head-to-MRI transform a transition
  {
    const std::string file = "fif_obj_free-fwd.fif";
    const fs::path pkg = root / "fwd.nxr.zarr";
    make_package(pkg, names_of(file), false);
    cv::FifObjectImport opt;
    opt.name = "fw";
    const Json out = import_file(pkg, file, opt);
    SubjectPackage p = SubjectPackage::open(pkg);
    auto f = fif::File(std::make_shared<nxr::formats::FileSource>(data + file));
    const fif::ForwardFile fw = fif::read_forward(f);
    const auto op = row(p, "operator", "fw");
    CHECK(op.has_value());
    CHECK_EQ(text(*op, "kind"), std::string("forward"));
    CHECK_EQ(integer(*op, "n_rows"), out["n_channels"].get<std::int64_t>());
    CHECK_EQ(integer(*op, "n_cols"), 3 * out["n_sources"].get<std::int64_t>());
    CHECK_EQ(text(*op, "from_value_type"), std::string("vector3"));
    CHECK(!is_null(*op, "from_selection_id") && !is_null(*op, "to_selection_id"));
    const auto& s = fw.solutions[0];
    const std::vector<float> g = read_f32(p, text(*op, "path"), integer(*op, "n_rows"), integer(*op, "n_cols"));
    // the gain's entry (channel c, component k) is the file's (k, c)
    for (std::int32_t c = 0; c < s.nchan; ++c)
      for (std::int32_t k = 0; k < s.sol.nrow; k += 5)
        CHECK(std::fabs(g[static_cast<std::size_t>(c) * s.sol.nrow + k] - static_cast<float>(s.sol.data[static_cast<std::size_t>(k) * s.sol.ncol + c])) < 1e-12);
    CHECK(row(p, "operator", "fw_head_to_mri").has_value());
    CHECK(row(p, "selection", "fw_source_space").has_value());
    CHECK(p.verify().ok);
  }

  // an inverse operator: its parts, never a kernel
  {
    const std::string file = "fif_obj-inv.fif";
    const fs::path pkg = root / "inv.nxr.zarr";
    make_package(pkg, names_of(file), false);
    cv::FifObjectImport opt;
    opt.name = "iv";
    const Json out = import_file(pkg, file, opt);
    SubjectPackage p = SubjectPackage::open(pkg);
    CHECK(row(p, "operator", "iv_eigen_leads").has_value());
    CHECK(row(p, "operator", "iv_eigen_fields").has_value());
    CHECK(!p.db().one("SELECT id FROM operator WHERE kind = 'inverse kernel'").has_value());
    CHECK(row(p, "field", "iv_singular_values").has_value());
    CHECK(row(p, "statistic", "iv_noise_cov").has_value());
    CHECK_EQ(out["rank"].get<std::int64_t>(), static_cast<std::int64_t>(17));
    CHECK(p.verify().ok);
  }

  // events and annotations on an existing recording; an annotation relative to a date needs the recording's date
  {
    const fs::path pkg = root / "eve.nxr.zarr";
    make_package(pkg, none, true);
    cv::FifObjectImport opt;
    opt.recording = "synthetic";
    const Json out = import_file(pkg, "fif_obj-eve.fif", opt);
    CHECK_EQ(out["n_events"].get<std::int64_t>(), static_cast<std::int64_t>(5));
    CHECK_EQ(code_of([&] { import_file(pkg, "fif_obj-eve.fif", opt); }), std::string(code::invalid_argument));   // never overwrites
    const fs::path pk2 = root / "annot.nxr.zarr";
    make_package(pk2, none, true);
    CHECK_EQ(code_of([&] { import_file(pk2, "fif_obj-annot.fif", opt); }), std::string(code::invalid_argument));   // orig_time, and no date
    CHECK_EQ(code_of([&] { import_file(pk2, "fif_obj-eve.fif", cv::FifObjectImport{}); }), std::string(code::invalid_argument));   // no recording named
  }

  // a head-to-MRI transform: a transition operator head -> freesurfer-tkr
  {
    const fs::path pkg = root / "trans.nxr.zarr";
    make_package(pkg, none, true);
    cv::FifObjectImport opt;
    opt.name = "h";
    import_file(pkg, "fif_obj-trans.fif", opt);
    SubjectPackage p = SubjectPackage::open(pkg);
    const auto t = row(p, "operator", "h_head_to_mri");
    CHECK(!t.has_value());
    const auto t2 = row(p, "operator", "h");
    CHECK(t2.has_value());
    CHECK_EQ(text(*t2, "kind"), std::string("transition"));
    // the head origin lands where MNE's transform puts it (the golden: a translation of (1, -2, 3) mm, head -> MRI)
    const auto raw = ArrayReader::open(p.store(), text(*t2, "path")).read({0, 0}, {3, 4});
    std::vector<double> m(12);
    std::memcpy(m.data(), raw.data(), raw.size());
    CHECK(std::fabs(m[3] - 0.001) < 1e-7 && std::fabs(m[7] + 0.002) < 1e-7 && std::fabs(m[11] - 0.003) < 1e-7 && m[0] == 1.0 && m[5] == 1.0 && m[10] == 1.0);
  }

  fs::remove_all(root);
  return nxrtest::finish("import_fif_objects");
}
