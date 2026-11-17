// The FreeSurfer and GIFTI importers (nxr/convert/freesurfer_import.h, gifti_import.h) on synthetic files: a subject folder written here, the readers' GIFTI goldens.
// The surfaces' system and unit, c_ras as the transition to scanner RAS, annotations only when named, overlays, the origin on every item, the refusals.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "check.h"
#include "nxr/convert/freesurfer_import.h"
#include "nxr/convert/gifti_import.h"
#include "nxr/formats/deflate.h"
#include "nxr/formats/gifti.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/subject_package.h"

using namespace nxr::io;
namespace fs = std::filesystem;
namespace cv = nxr::convert;
namespace gii = nxr::formats::gifti;

static const std::string data = std::string(NXR_IO_DATA_DIR) + "/../../formats/test/data/";

template <class F> static std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception&) { return "other"; } return ""; }

static void put32(std::string& s, std::uint32_t u) { for (int k = 3; k >= 0; --k) s.push_back(static_cast<char>((u >> (8 * k)) & 0xff)); }
static void putf(std::string& s, float x) { std::uint32_t u; std::memcpy(&u, &x, 4); put32(s, u); }
static void write_file(const fs::path& p, const std::string& s) { fs::create_directories(p.parent_path()); std::ofstream(p, std::ios::binary) << s; }

// an octahedron (6 vertices, 8 faces) of radius 30 mm shifted by dx; the volume-geometry tail names c_ras
static std::string surface_bytes(double dx, const std::array<double, 3>& cras) {
  std::string s = "\xff\xff\xfe";
  s += "created by someone on somehost at Mon Jan  1 00:00:00 2026\n\n";
  put32(s, 6); put32(s, 8);
  const double v[6][3] = {{30, 0, 0}, {-30, 0, 0}, {0, 30, 0}, {0, -30, 0}, {0, 0, 30}, {0, 0, -30}};
  for (auto& p : v) { putf(s, static_cast<float>(p[0] + dx)); putf(s, static_cast<float>(p[1])); putf(s, static_cast<float>(p[2])); }
  const int f[8][3] = {{0, 2, 4}, {2, 1, 4}, {1, 3, 4}, {3, 0, 4}, {2, 0, 5}, {1, 2, 5}, {3, 1, 5}, {0, 3, 5}};
  for (auto& t : f) { put32(s, t[0]); put32(s, t[1]); put32(s, t[2]); }
  put32(s, 20);
  char buf[512];
  std::snprintf(buf, sizeof buf, "valid = 1  # volume info valid\nfilename = /home/someone/orig.mgz\nvolume = 256 256 256\nvoxelsize = 1.000000 1.000000 1.000000\nxras = -1 0 0\nyras = 0 0 -1\nzras = 0 1 0\ncras = %.6f %.6f %.6f\n",
                cras[0], cras[1], cras[2]);
  return s + buf;
}
static std::string curv_bytes(float base) {
  std::string s = "\xff\xff\xff";
  put32(s, 6); put32(s, 8); put32(s, 1);
  for (int i = 0; i < 6; ++i) putf(s, base + static_cast<float>(i) / 4.0f);
  return s;
}
// a new-format annotation: label values pack the colour (r + g*256 + b*65536); vertex 5 has no label
static std::string annot_bytes(bool right) {
  std::string s;
  put32(s, 6);
  const std::uint32_t packed[3] = {0x0000FFu, 0x00FF00u, 0xFF0000u};   // red, green, blue (r + g*256 + b*65536)
  for (int v = 0; v < 6; ++v) { put32(s, v); put32(s, v == 5 ? 0u : packed[v % 2 + (right ? 1 : 0)]); }
  put32(s, 1);
  put32(s, static_cast<std::uint32_t>(-2));
  put32(s, 3);
  put32(s, 4); s += std::string("ctb\0", 4);
  put32(s, 3);
  const char* names[3] = {"bankssts", "caudalmiddlefrontal", "cuneus"};
  const int rgb[3][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};
  for (int k = 0; k < 3; ++k) {
    put32(s, k);
    put32(s, static_cast<std::uint32_t>(std::strlen(names[k]) + 1));
    s += names[k]; s.push_back('\0');
    for (int c = 0; c < 3; ++c) put32(s, rgb[k][c]);
    put32(s, 0);
  }
  return s;
}
// an MGH volume header (version 1, 256^3, one frame, type 0) with c_ras, gzipped as an .mgz
static std::string mgz_bytes(const std::array<double, 3>& cras) {
  std::string s;
  put32(s, 1); put32(s, 256); put32(s, 256); put32(s, 256); put32(s, 1); put32(s, 0); put32(s, 0);
  s.push_back('\0'); s.push_back('\1');
  for (float x : {1.f, 1.f, 1.f, -1.f, 0.f, 0.f, 0.f, 0.f, -1.f, 0.f, 1.f, 0.f}) putf(s, x);
  for (double c : cras) putf(s, static_cast<float>(c));
  s.append(284 - s.size(), '\0');   // the header is 284 bytes
  const auto z = nxr::formats::deflate_bytes(reinterpret_cast<const unsigned char*>(s.data()), s.size(), nxr::formats::DeflateWrapper::gzip);
  return std::string(z.begin(), z.end());
}

static void make_subject(const fs::path& d, const std::array<double, 3>& cras_orig, const std::array<double, 3>& cras_tail, bool with_rh = true) {
  write_file(d / "surf/lh.white", surface_bytes(-50, cras_tail));
  if (with_rh) write_file(d / "surf/rh.white", surface_bytes(50, cras_tail));
  write_file(d / "surf/lh.thickness", curv_bytes(1.0f));
  write_file(d / "surf/rh.thickness", curv_bytes(2.0f));
  write_file(d / "label/lh.aparc.annot", annot_bytes(false));
  write_file(d / "label/rh.aparc.annot", annot_bytes(true));
  write_file(d / "mri/orig.mgz", mgz_bytes(cras_orig));
}

static fs::path make_package(const fs::path& pkg) {
  CreatePackageOptions po;
  po.home_dataset = "demo";
  DatasetWriter w = create_package(pkg, "S", po);
  return pkg;
}
static std::optional<Row> row(const SubjectPackage& p, const char* table, const std::string& name) { return p.db().one(std::string("SELECT * FROM ") + table + " WHERE name = ?", {V(name)}); }
static std::optional<Row> cs_of(const SubjectPackage& p, const std::string& manifold) {
  return p.db().one("SELECT cs.name AS name, cs.unit AS unit FROM manifold m JOIN geometry g ON g.id = m.geometry_id JOIN coordinate_system cs ON cs.id = g.coordinate_system_id WHERE m.name = ?", {V(manifold)});
}
static std::vector<float> read_f32(const SubjectPackage& p, const std::string& path, std::int64_t n) {
  const auto raw = ArrayReader::open(p.store(), path).read({0}, {n});
  std::vector<float> v(static_cast<std::size_t>(n));
  CHECK_EQ(raw.size(), v.size() * 4);
  std::memcpy(v.data(), raw.data(), raw.size());
  return v;
}
static std::vector<double> operator_matrix(const SubjectPackage& p, const std::string& name) {
  const auto o = row(p, "operator", name);
  CHECK(o.has_value());
  const auto raw = ArrayReader::open(p.store(), *opt_text(*o, "path")).read({0, 0}, {3, 4});
  std::vector<double> v(12);
  CHECK_EQ(raw.size(), v.size() * 8);
  std::memcpy(v.data(), raw.data(), raw.size());
  return v;
}

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_import_surfaces_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);

  // ── FreeSurfer ──
  {
    const fs::path sd = root / "subj";
    make_subject(sd, {5.0, -10.0, 15.0}, {5.0, -10.0, 15.0});
    const fs::path pkg = make_package(root / "fs.nxr.zarr");
    {
      const PackageRecord rec = SubjectPackage::open(pkg).record();
      DatasetWriter w = open_package_writer(pkg);
      SubjectWriter sub = SubjectWriter::open(w, rec.name);
      cv::FreeSurferImport o;
      o.source = "subj";
      o.annotations = {"aparc"};
      o.overlays = {"thickness"};
      const Json out = cv::import_freesurfer_subject(w, sub, sd.string(), o);
      CHECK_EQ(out["surfaces"][0]["n_vertices"].get<int>(), 12);
      CHECK_EQ(out["transition"].get<std::string>(), std::string("cortex_freesurfer-tkr_scanner-ras"));
    }
    SubjectPackage p = SubjectPackage::open(pkg);
    CHECK(p.verify().ok);
    CHECK(items_without_origin(p.db()).empty());
    const auto cs = cs_of(p, "cortex_white");
    CHECK(cs.has_value());
    CHECK_EQ(text(*cs, "name"), std::string("freesurfer-tkr"));
    CHECK_EQ(text(*cs, "unit"), std::string("m"));
    // c_ras as the transition to scanner RAS, in metres
    const std::vector<double> m = operator_matrix(p, "cortex_freesurfer-tkr_scanner-ras");
    const std::vector<double> want = {1, 0, 0, 0.005, 0, 1, 0, -0.010, 0, 0, 1, 0.015};
    for (int i = 0; i < 12; ++i) CHECK(std::abs(m[i] - want[i]) < 1e-12);
    const auto op = row(p, "operator", "cortex_freesurfer-tkr_scanner-ras");
    const auto ends = p.db().one("SELECT a.name AS a, b.name AS b, a.unit AS au FROM operator o JOIN coordinate_system a ON a.id = o.from_coordinate_system_id JOIN coordinate_system b ON b.id = o.to_coordinate_system_id WHERE o.name = ?",
                                 {V(std::string("cortex_freesurfer-tkr_scanner-ras"))});
    CHECK(ends.has_value());
    CHECK_EQ(text(*ends, "a"), std::string("freesurfer-tkr"));
    CHECK_EQ(text(*ends, "b"), std::string("scanner-ras"));
    // the annotation: a codes Selection, entries named with the hemisphere, vertex 5 of each hemisphere unlabelled
    const auto sel = row(p, "selection", "cortex_aparc");
    CHECK(sel.has_value());
    CHECK_EQ(integer(*sel, "n_elements"), 12);
    const auto entries = p.db().all("SELECT name FROM dictionary_entry WHERE dictionary_id = ? ORDER BY ordinal", {V(text(*sel, "dictionary_id"))});
    bool l = false, r = false;
    for (const auto& e : entries) { const std::string n = text(e, "name"); l = l || n == "bankssts L"; r = r || n == "caudalmiddlefrontal R"; }
    CHECK(l);
    CHECK(r);
    // the overlay: lh values then rh values
    const auto f = row(p, "field", "cortex_thickness");
    CHECK(f.has_value());
    const std::vector<float> v = read_f32(p, *opt_text(*f, "path"), 12);
    CHECK(std::abs(v[0] - 1.0f) < 1e-6 && std::abs(v[6] - 2.0f) < 1e-6 && std::abs(v[11] - 3.25f) < 1e-6);
    // the same file's c_ras as the volume's and the surface's own tail
    CHECK(std::abs((*cv::freesurfer_c_ras((sd / "surf/lh.white").string()))[2] - 15.0) < 1e-4);
    CHECK(std::abs((*cv::freesurfer_c_ras((sd / "mri/orig.mgz").string()))[1] + 10.0) < 1e-4);
  }
  // an annotation is imported only when it is named
  {
    const fs::path pkg = make_package(root / "fs2.nxr.zarr");
    const PackageRecord rec = SubjectPackage::open(pkg).record();
    DatasetWriter w = open_package_writer(pkg);
    SubjectWriter sub = SubjectWriter::open(w, rec.name);
    cv::FreeSurferImport o;
    cv::import_freesurfer_subject(w, sub, (root / "subj").string(), o);
    CHECK(!row(SubjectPackage::open(pkg), "selection", "cortex_aparc").has_value());
    // twice: refused; a name given: fine
    CHECK_EQ(code_of([&] { cv::import_freesurfer_subject(w, sub, (root / "subj").string(), o); }), std::string(code::invalid_argument));
  }
  // no orig.mgz: c_ras from the surface's own volume geometry; disagreeing ones refused; no rh refused
  {
    make_subject(root / "s2", {0, 0, 0}, {1.0, 2.0, 3.0});
    fs::remove(root / "s2/mri/orig.mgz");
    const fs::path pkg = make_package(root / "fs3.nxr.zarr");
    const PackageRecord rec = SubjectPackage::open(pkg).record();
    DatasetWriter w = open_package_writer(pkg);
    SubjectWriter sub = SubjectWriter::open(w, rec.name);
    cv::FreeSurferImport o;
    const Json out = cv::import_freesurfer_subject(w, sub, (root / "s2").string(), o);
    CHECK_EQ(out["transition"].get<std::string>(), std::string("cortex_freesurfer-tkr_scanner-ras"));
    CHECK(std::abs(operator_matrix(SubjectPackage::open(pkg), "cortex_freesurfer-tkr_scanner-ras")[11] - 0.003) < 1e-12);
    make_subject(root / "s3", {9, 9, 9}, {1, 2, 3});
    const fs::path pkg2 = make_package(root / "fs4.nxr.zarr");
    DatasetWriter w2 = open_package_writer(pkg2);
    SubjectWriter sub2 = SubjectWriter::open(w2, SubjectPackage::open(pkg2).record().name);
    CHECK_EQ(code_of([&] { cv::import_freesurfer_subject(w2, sub2, (root / "s3").string(), o); }), std::string(code::invalid_argument));
    make_subject(root / "s4", {0, 0, 0}, {0, 0, 0}, false);
    const fs::path pkg3 = make_package(root / "fs5.nxr.zarr");
    DatasetWriter w3 = open_package_writer(pkg3);
    SubjectWriter sub3 = SubjectWriter::open(w3, SubjectPackage::open(pkg3).record().name);
    CHECK(!code_of([&] { cv::import_freesurfer_subject(w3, sub3, (root / "s4").string(), o); }).empty());
  }

  // ── GIFTI ──
  auto gifti = [&](const fs::path& pkg, const std::string& file, const cv::GiftiImport& opt) {
    DatasetWriter w = open_package_writer(pkg);
    SubjectWriter sub = SubjectWriter::open(w, SubjectPackage::open(pkg).record().name);
    cv::GiftiImport o = opt;
    o.source = fs::path(file).filename().string();
    o.source_file = file.find('/') == std::string::npos ? data + file : file;
    return cv::import_gifti(w, sub, o.source_file, o);
  };
  {
    const fs::path pkg = make_package(root / "g.nxr.zarr");
    cv::GiftiImport o;
    o.name = "srf";
    const Json out = gifti(pkg, "gifti_surf_b64gz.surf.gii", o);
    CHECK_EQ(out["system"].get<std::string>(), std::string("talairach"));
    CHECK(out["transition"].is_null());   // the data space and the transformed space are the same
    // maps and labels on that surface (6 vertices)
    cv::GiftiImport f;
    f.name = "map";
    f.surface = "srf";
    f.unit = "mm";
    const Json fo = gifti(pkg, "gifti_func.func.gii", f);
    CHECK_EQ(fo["fields"].size(), static_cast<std::size_t>(1));
    cv::GiftiImport l;
    l.name = "lab";
    l.surface = "srf";
    CHECK_EQ(code_of([&] { gifti(pkg, "gifti_label.label.gii", l); }), std::string(code::invalid_argument));   // a label file needs its parcellation named
    l.parcellation = "toy";
    const Json lo = gifti(pkg, "gifti_label.label.gii", l);
    CHECK_EQ(lo["selections"].size(), static_cast<std::size_t>(1));
    cv::GiftiImport bad;
    bad.surface = "nonexistent";
    CHECK_EQ(code_of([&] { gifti(pkg, "gifti_func.func.gii", bad); }), std::string(code::not_found));
    SubjectPackage p = SubjectPackage::open(pkg);
    CHECK(p.verify().ok);
    CHECK(items_without_origin(p.db()).empty());
    const auto cs = cs_of(p, "srf");
    CHECK_EQ(text(*cs, "name"), std::string("talairach"));
    CHECK_EQ(text(*cs, "unit"), std::string("mm"));
    const auto fld = row(p, "field", "map");
    CHECK(fld.has_value());
    CHECK_EQ(text(*fld, "unit"), std::string("mm"));
  }
  // data space unknown: a system of its own, metres, no transition; with FreeSurfer's volume geometry: tkr with c_ras; MNI -> a different system: its transition in mm
  {
    gii::Image img = gii::read_file(data + "gifti_surf_ascii.surf.gii");
    auto variant = [&](const std::string& name, const std::string& data_space, const std::string& to_space, bool volgeom) {
      gii::Image v = img;
      for (auto& a : v.arrays) if (a.intent == "NIFTI_INTENT_POINTSET") {
        a.transforms[0].dataspace = data_space;
        a.transforms[0].xformspace = to_space;
        if (volgeom) { a.metadata.push_back({"VolGeomC_R", "5"}); a.metadata.push_back({"VolGeomC_A", "-10"}); a.metadata.push_back({"VolGeomC_S", "15"}); }
      }
      gii::write_file((root / (name + ".surf.gii")).string(), v);
      return (root / (name + ".surf.gii")).string();
    };
    const fs::path pkg = make_package(root / "g2.nxr.zarr");
    cv::GiftiImport o;
    const Json unk = gifti(pkg, variant("unk", "NIFTI_XFORM_UNKNOWN", "NIFTI_XFORM_UNKNOWN", false), o);
    CHECK_EQ(unk["system"].get<std::string>(), std::string("gifti:unk"));
    CHECK(unk.contains("note"));
    const Json tkr = gifti(pkg, variant("tkrsurf", "NIFTI_XFORM_UNKNOWN", "NIFTI_XFORM_UNKNOWN", true), o);
    CHECK_EQ(tkr["system"].get<std::string>(), std::string("freesurfer-tkr"));
    CHECK_EQ(tkr["transition"].get<std::string>(), std::string("tkrsurf_freesurfer-tkr_scanner-ras"));
    const Json mni = gifti(pkg, variant("mnisurf", "NIFTI_XFORM_SCANNER_ANAT", "NIFTI_XFORM_MNI_152", false), o);
    CHECK_EQ(mni["system"].get<std::string>(), std::string("scanner-ras"));
    CHECK_EQ(mni["transition"].get<std::string>(), std::string("mnisurf_scanner-ras_mni"));
    SubjectPackage p = SubjectPackage::open(pkg);
    CHECK(p.verify().ok);
    CHECK_EQ(text(*cs_of(p, "unk"), "unit"), std::string("m"));
    CHECK_EQ(text(*cs_of(p, "mnisurf"), "unit"), std::string("m"));
    // scanner (m) -> MNI (mm): the file's matrix (translation 1.5, -2.5, 3.5 mm) is in mm; the rotation part scales 1000 (m to mm)
    const std::vector<double> m = operator_matrix(p, "mnisurf_scanner-ras_mni");
    CHECK(std::abs(m[0] - 1000.0) < 1e-9 && std::abs(m[3] - 1.5) < 1e-9 && std::abs(m[7] + 2.5) < 1e-9 && std::abs(m[11] - 3.5) < 1e-9);
    const std::vector<double> t = operator_matrix(p, "tkrsurf_freesurfer-tkr_scanner-ras");
    CHECK(std::abs(t[3] - 0.005) < 1e-12);
    // the positions are the file's millimetres in metres: vertex 0 of the octahedron is at x = 30.011406 mm
    const auto mf = row(p, "manifold", "unk");
    const auto geo = p.db().one("SELECT positions_path FROM geometry WHERE id = ?", {V(text(*mf, "geometry_id"))});
    CHECK(geo.has_value());
    const auto raw = ArrayReader::open(p.store(), *opt_text(*geo, "positions_path")).read({0, 0}, {6, 3});
    CHECK_EQ(raw.size(), static_cast<std::size_t>(6 * 3 * 8));
    double x0;
    std::memcpy(&x0, raw.data(), 8);
    CHECK(std::abs(x0 - 0.030011406) < 1e-7);
  }
  fs::remove_all(root);
  return nxrtest::finish("import_surfaces");
}
