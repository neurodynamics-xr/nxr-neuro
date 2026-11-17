// The template importers (io/convert) on SYNTHETIC data only: small octahedra written as a FreeSurfer subject and as a Brainstorm default anatomy, a tiny TrackVis file.
// Each importer, the origin record on every item, parcellations only when named, the source manifest (a listed file with another checksum, an unlisted file), the
// redistributable edition leaving out what must never be redistributed, and the distance gate of the streamline ends. The .trk reader is compared with numbers made by
// nibabel from the same bytes (header orders LPS, RAS, a mismatch of header and matrix, and a permuted order).
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

extern "C" {
#include "matio.h"
}
#include "check.h"
#include "nxr/convert/templates.h"
#include "nxr/formats/fif_write.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/templates.h"

using namespace nxr::io;
using namespace nxr::convert;
namespace fs = std::filesystem;

namespace {

std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); }
  return "";
}
std::string what_of(const std::function<void()>& f) {
  try { f(); } catch (const std::exception& e) { return e.what(); }
  return "";
}
bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

void be32(std::ofstream& o, std::uint32_t x) { const char b[4] = {char(x >> 24), char(x >> 16), char(x >> 8), char(x)}; o.write(b, 4); }
void write_fs(const fs::path& p, const std::vector<double>& v, const std::vector<std::int32_t>& f) {
  std::ofstream o(p, std::ios::binary);
  o.write("\xff\xff\xfe", 3);
  o << "created by test_template_import\n\n";
  be32(o, static_cast<std::uint32_t>(v.size() / 3));
  be32(o, static_cast<std::uint32_t>(f.size() / 3));
  for (double x : v) { const float y = static_cast<float>(x); std::uint32_t u; std::memcpy(&u, &y, 4); be32(o, u); }
  for (auto i : f) be32(o, static_cast<std::uint32_t>(i));
}
// an old-format annotation: vertex labels packed as r + 256 g + 65536 b, and the colour table of (name, r, g, b)
void write_annot(const fs::path& p, const std::vector<int>& label_of_vertex, const std::vector<std::array<int, 3>>& colours, const std::vector<std::string>& names) {
  std::ofstream o(p, std::ios::binary);
  be32(o, static_cast<std::uint32_t>(label_of_vertex.size()));
  for (std::size_t v = 0; v < label_of_vertex.size(); ++v) {
    be32(o, static_cast<std::uint32_t>(v));
    const auto& c = colours[static_cast<std::size_t>(label_of_vertex[v])];
    be32(o, static_cast<std::uint32_t>(c[0] + 256 * c[1] + 65536 * c[2]));
  }
  be32(o, 1);
  be32(o, static_cast<std::uint32_t>(names.size()));
  be32(o, 4); o.write("test", 4);
  for (std::size_t k = 0; k < names.size(); ++k) {
    be32(o, static_cast<std::uint32_t>(names[k].size())); o.write(names[k].data(), static_cast<std::streamsize>(names[k].size()));
    for (int c : {colours[k][0], colours[k][1], colours[k][2], 0}) be32(o, static_cast<std::uint32_t>(c));
  }
}
void octahedron(double cx, double r, std::vector<double>& v, std::vector<std::int32_t>& f) {
  v = {cx + r, 0, 0, cx - r, 0, 0, cx, r, 0, cx, -r, 0, cx, 0, r, cx, 0, -r};
  f = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
}

// a synthetic FreeSurfer subject: two octahedra (left at x = -30 mm, right at +30), pial, white, inflated, sphere.reg; aparc with two regions; the cortex labels
void make_freesurfer(const fs::path& dir) {
  fs::create_directories(dir / "surf");
  fs::create_directories(dir / "label");
  std::vector<double> lh, rh, sph;
  std::vector<std::int32_t> f;
  octahedron(-30, 10, lh, f);
  octahedron(30, 10, rh, f);
  auto scaled = [](std::vector<double> v, double cx, double s) { for (std::size_t i = 0; i < v.size(); ++i) v[i] = (i % 3 == 0 ? cx + (v[i] - cx) * s : v[i] * s); return v; };
  for (const auto& [name, s] : std::vector<std::pair<std::string, double>>{{"pial", 1.0}, {"white", 0.9}, {"inflated", 1.2}}) {
    write_fs(dir / "surf" / ("lh." + name), scaled(lh, -30, s), f);
    write_fs(dir / "surf" / ("rh." + name), scaled(rh, 30, s), f);
  }
  octahedron(0, 100, sph, f);
  write_fs(dir / "surf" / "lh.sphere.reg", sph, f);
  write_fs(dir / "surf" / "rh.sphere.reg", sph, f);
  const std::vector<std::array<int, 3>> colours = {{25, 5, 25}, {220, 20, 20}, {20, 220, 20}};
  for (const char* h : {"lh", "rh"}) {
    write_annot(dir / "label" / (std::string(h) + ".aparc.annot"), {1, 1, 1, 2, 2, 2}, colours, {"unknown", "region_a", "region_b"});
    std::ofstream l(dir / "label" / (std::string(h) + ".cortex.label"));
    l << "#!ascii label  , from subject fsaverage vox2ras=TkReg\n5\n0 0 0 0 0\n1 0 0 0 0\n2 0 0 0 0\n3 0 0 0 0\n4 0 0 0 0\n";   // vertex 5 is the medial wall
  }
}

// ── matio helpers ──
matvar_t* num(const char* name, matio_classes cls, matio_types type, std::vector<size_t> dims, void* data) { return Mat_VarCreate(name, cls, type, static_cast<int>(dims.size()), dims.data(), data, 0); }
matvar_t* chars(const char* name, const std::string& s) {
  size_t d[2] = {1, s.size()};
  return Mat_VarCreate(name, MAT_C_CHAR, MAT_T_UINT8, 2, d, const_cast<char*>(s.data()), 0);
}
matvar_t* numeric(const char* name, const std::vector<double>& row_major, size_t rows, size_t cols) {
  std::vector<double> col(row_major.size());
  for (size_t r = 0; r < rows; ++r) for (size_t c = 0; c < cols; ++c) col[c * rows + r] = row_major[r * cols + c];
  return num(name, MAT_C_DOUBLE, MAT_T_DOUBLE, {rows, cols}, col.data());
}
struct Scout { std::string label; std::vector<double> vertices; };
// an atlas: Name and a struct array of scouts (Label, Vertices (1-based), Color)
matvar_t* atlas_struct(const std::vector<std::pair<std::string, std::vector<Scout>>>& atlases) {
  const char* fn[] = {"Name", "Scouts"};
  size_t d[2] = {1, atlases.size()};
  matvar_t* all = Mat_VarCreateStruct("Atlas", 2, d, fn, 2);
  for (size_t i = 0; i < atlases.size(); ++i) {
    Mat_VarSetStructFieldByName(all, "Name", i, chars(nullptr, atlases[i].first));
    const char* sf[] = {"Label", "Vertices", "Color"};
    size_t sd[2] = {1, atlases[i].second.size()};
    matvar_t* scouts = Mat_VarCreateStruct(nullptr, 2, sd, sf, 3);
    for (size_t k = 0; k < atlases[i].second.size(); ++k) {
      Mat_VarSetStructFieldByName(scouts, "Label", k, chars(nullptr, atlases[i].second[k].label));
      Mat_VarSetStructFieldByName(scouts, "Vertices", k, numeric(nullptr, atlases[i].second[k].vertices, 1, atlases[i].second[k].vertices.size()));
      Mat_VarSetStructFieldByName(scouts, "Color", k, numeric(nullptr, {0.2, 0.4, 0.6}, 1, 3));
    }
    Mat_VarSetStructFieldByName(all, "Scouts", i, scouts);
  }
  return all;
}
// a Brainstorm cortex: two octahedra (12 vertices, metres), its registered sphere, and the named atlases
void write_tess(const fs::path& p, const std::vector<std::pair<std::string, std::vector<Scout>>>& atlases) {
  std::vector<double> a, b, sa, sb;
  std::vector<std::int32_t> f;
  octahedron(-0.03, 0.01, a, f);
  octahedron(0.03, 0.01, b, f);
  std::vector<double> v = a, faces;
  v.insert(v.end(), b.begin(), b.end());
  for (int h = 0; h < 2; ++h) for (std::size_t i = 0; i < f.size(); ++i) faces.push_back(f[i] + 1 + 6 * h);
  octahedron(0, 1, sa, f);
  sa.insert(sa.end(), sa.begin(), sa.end());
  mat_t* m = Mat_CreateVer(p.string().c_str(), nullptr, MAT_FT_MAT5);
  Mat_VarWrite(m, numeric("Vertices", v, 12, 3), MAT_COMPRESSION_NONE);
  Mat_VarWrite(m, numeric("Faces", faces, 16, 3), MAT_COMPRESSION_NONE);
  Mat_VarWrite(m, chars("Comment", "cortex_synthetic"), MAT_COMPRESSION_NONE);
  Mat_VarWrite(m, atlas_struct(atlases), MAT_COMPRESSION_NONE);
  const char* rf[] = {"Sphere"};
  size_t one[2] = {1, 1};
  matvar_t* reg = Mat_VarCreateStruct("Reg", 2, one, rf, 1);
  const char* sf[] = {"Vertices"};
  matvar_t* sph = Mat_VarCreateStruct(nullptr, 2, one, sf, 1);
  Mat_VarSetStructFieldByName(sph, "Vertices", 0, numeric(nullptr, sa, 12, 3));
  Mat_VarSetStructFieldByName(reg, "Sphere", 0, sph);
  Mat_VarWrite(m, reg, MAT_COMPRESSION_NONE);
  Mat_Close(m);
}
// a subjectimage with the SCS and NCS frames: SCS.R turns the MRI axes by 90 degrees about z (x -> y, y -> -x), SCS.T = (10, 20, 30) mm; NCS.R = identity, NCS.T = (-100, -120, -80)
void write_subjectimage(const fs::path& p) {
  mat_t* m = Mat_CreateVer(p.string().c_str(), nullptr, MAT_FT_MAT5);
  auto frame = [&](const char* name, const std::vector<double>& R, const std::vector<double>& T) {
    const char* fn[] = {"R", "T"};
    size_t one[2] = {1, 1};
    matvar_t* s = Mat_VarCreateStruct(name, 2, one, fn, 2);
    Mat_VarSetStructFieldByName(s, "R", 0, numeric(nullptr, R, 3, 3));
    Mat_VarSetStructFieldByName(s, "T", 0, numeric(nullptr, T, 3, 1));
    Mat_VarWrite(m, s, MAT_COMPRESSION_NONE);
  };
  {
    // the SCS frame again, with the landmarks (in voxel units: NAS, LPA, RPA) and Voxsize, as Brainstorm's anatomy carries them
    const char* fn[] = {"R", "T", "NAS", "LPA", "RPA"};
    size_t one[2] = {1, 1};
    matvar_t* sc = Mat_VarCreateStruct("SCS", 2, one, fn, 5);
    Mat_VarSetStructFieldByName(sc, "R", 0, numeric(nullptr, {0, -1, 0, 1, 0, 0, 0, 0, 1}, 3, 3));
    Mat_VarSetStructFieldByName(sc, "T", 0, numeric(nullptr, {10, 20, 30}, 3, 1));
    Mat_VarSetStructFieldByName(sc, "NAS", 0, numeric(nullptr, {5, 1, 2}, 1, 3));
    Mat_VarSetStructFieldByName(sc, "LPA", 0, numeric(nullptr, {1, 2, 3}, 1, 3));
    Mat_VarSetStructFieldByName(sc, "RPA", 0, numeric(nullptr, {4, 5, 6}, 1, 3));
    Mat_VarWrite(m, sc, MAT_COMPRESSION_NONE);
    Mat_VarWrite(m, numeric("Voxsize", {2, 3, 1}, 1, 3), MAT_COMPRESSION_NONE);
  }
  frame("NCS", {1, 0, 0, 0, 1, 0, 0, 0, 1}, {-100, -120, -80});
  Mat_Close(m);
}

// a TrackVis v2 file: the header's order and voxel-to-RAS matrix, 10 x 12 x 14 voxels of 2 x 2.5 x 3 mm
void write_trk(const fs::path& p, const std::string& order, const float aff[4][4], const std::vector<std::vector<std::array<float, 3>>>& streamlines) {
  std::vector<char> h(1000, 0);
  std::memcpy(h.data(), "TRACK", 5);
  const std::int16_t dim[3] = {10, 12, 14};
  const float vs[3] = {2.0f, 2.5f, 3.0f};
  std::memcpy(h.data() + 6, dim, 6);
  std::memcpy(h.data() + 12, vs, 12);
  std::memcpy(h.data() + 440, aff, 64);
  std::memcpy(h.data() + 948, order.data(), 3);
  const std::int32_t version = 2, size = 1000;
  std::memcpy(h.data() + 992, &version, 4);
  std::memcpy(h.data() + 996, &size, 4);
  std::ofstream o(p, std::ios::binary);
  o.write(h.data(), 1000);
  for (const auto& s : streamlines) {
    const std::int32_t n = static_cast<std::int32_t>(s.size());
    o.write(reinterpret_cast<const char*>(&n), 4);
    for (const auto& q : s) o.write(reinterpret_cast<const char*>(q.data()), 12);
  }
}

bool near(double a, double b, double tol = 1e-4) { return std::abs(a - b) <= tol; }

std::vector<std::string> item_names(const Database& db, const std::string& table) {
  std::vector<std::string> out;
  for (const auto& it : package_items(db)) if (it.table == table) out.push_back(it.name);
  return out;
}

}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_template_import_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "work");
  const fs::path shelf = root / "shelf";

  // ── the manifest ──
  {
    const Json& r = source_manifest()["recipes"];
    CHECK(r.contains("fsaverage") && r.contains("icbm152") && r.contains("hcp1065"));
    for (auto it = r.begin(); it != r.end(); ++it)
      for (const auto& f : it.value()["files"]) CHECK(f["sha256"].get<std::string>().size() == 64);
    CHECK(has(r["hcp1065"]["licence"].get<std::string>(), "CC BY-SA 4.0") && has(r["hcp1065"]["licence"].get<std::string>(), "WU-Minn"));
    CHECK(has(r["hcp1065"]["citations"][0].get<std::string>(), "Nat Commun 13:4933"));
  }

  // ── FreeSurfer (fsaverage) ──
  make_freesurfer(root / "fs");
  {
    BuildRequest rq;
    rq.recipe = "fsaverage";
    rq.sources = root / "fs";
    rq.package = root / "work" / "fsaverage.nxr.zarr";
    rq.created_by = "test";
    // a file the manifest does not list is refused, and nothing is left behind
    CHECK(has(what_of([&] { build_template(rq); }), "--accept-unlisted-sources"));   // the synthetic files are not the manifest's
    CHECK(!fs::exists(rq.package));
    rq.accept_unlisted_sources = true;
    const Json made = build_template(rq);
    SubjectPackage p = SubjectPackage::open(rq.package);
    CHECK_EQ(p.record().kind, std::string("template"));
    CHECK(p.db().all("SELECT 1 FROM manifold WHERE name = 'cortex_pial_sphere'").size() == 1);               // the registered sphere
    CHECK(p.db().all("SELECT 1 FROM manifold WHERE name = 'cortex_pial' AND n_vertices = 12").size() == 1);
    CHECK(item_names(p.db(), "selection").empty());                                                           // no parcellation, none named
    CHECK(items_without_origin(p.db()).empty());
    const auto o = origin_of(p.db(), "manifold", text(*p.db().one("SELECT id FROM manifold WHERE name = 'cortex_pial'"), "id"));
    CHECK(o && o->redistributable && has(o->licence, "FreeSurfer Software License") && has(o->attribution, "MODIFIED") && has(o->source, "lh.pial") && has(o->source, "sha256"));
    CHECK(made["sources"]["files"].size() == 8);                                                              // the eight surface files read, and nothing else
    CHECK(p.verify().ok);
    // named parcellations and the cortex mask
    BuildRequest rq2 = rq;
    rq2.package = root / "work" / "fsaverage-parc.nxr.zarr";
    rq2.annotations = {"aparc"};
    rq2.cortex_mask = true;
    build_template(rq2);
    SubjectPackage p2 = SubjectPackage::open(rq2.package);
    const auto sel = item_names(p2.db(), "selection");
    CHECK(std::find(sel.begin(), sel.end(), "Desikan-Killiany") != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), "cortex_pial_cortex") != sel.end());
    CHECK(items_without_origin(p2.db()).empty());
    CHECK_EQ(code_of([&] { BuildRequest r3 = rq2; r3.package = root / "work" / "x.nxr.zarr"; r3.annotations = {"no_such_annot"}; build_template(r3); }), std::string(code::not_found));
    CHECK(!fs::exists(root / "work" / "x.nxr.zarr"));
    // a listed file whose checksum differs is refused: a file named like a listed one, with other bytes
    fs::create_directories(root / "fs-other" / "surf");
    fs::copy_file(root / "fs" / "surf" / "rh.pial", root / "fs-other" / "surf" / "lh.sphere.reg");   // the first file read: listed, with another checksum
    BuildRequest rq4 = rq;
    rq4.accept_unlisted_sources = false;
    rq4.sources = root / "fs-other";
    rq4.package = root / "work" / "y.nxr.zarr";
    CHECK(has(what_of([&] { build_template(rq4); }), "the manifest lists"));
    // the released template and its redistributable edition (everything here is FreeSurfer's: all kept)
    const ReleaseReport full = release_package(rq2.package, ReleaseOptions{"fsaverage", "1.0", "test", shelf});
    const ReleaseReport red = release_redistributable(full.folder, shelf, "test");
    CHECK(red.left_out.empty());
  }

  // ── the scalp and the landmarks: fsaverage (MNE's bem/outer_skin.surf and fiducials file, the surfaces in FreeSurfer's surface RAS in metres) ──
  {
    fs::create_directories(root / "fs" / "bem");
    std::vector<double> skin;
    std::vector<std::int32_t> sf;
    octahedron(0, 90, skin, sf);
    write_fs(root / "fs" / "bem" / "outer_skin.surf", skin, sf);
    {
      nxr::formats::FileSink sink((root / "fs" / "bem" / "fsaverage-fiducials.fif").string());
      nxr::formats::fif::TagWriter w(sink);
      w.start_file(nxr::formats::fif::Id{});
      w.start_block(nxr::formats::fif::block::kIsotrak);
      w.write_int(nxr::formats::fif::tag::kMneCoordFrame, 5);
      for (const auto& [ident, x, y, z] : std::vector<std::array<double, 4>>{{2, 0.0, 0.08, -0.04}, {1, -0.07, 0.0, -0.04}, {3, 0.07, 0.0, -0.04}}) {
        nxr::formats::fif::DigPoint d;
        d.kind = 1; d.ident = static_cast<std::int32_t>(ident); d.r = {x, y, z};
        w.write_dig_point(d);
      }
      w.end_block(nxr::formats::fif::block::kIsotrak);
      w.end_file();
    }
    const HeadLandmarks lm = landmarks_from_fif(root / "fs" / "bem" / "fsaverage-fiducials.fif");
    CHECK(std::fabs(lm.nasion[1] - 0.08) < 1e-6 && std::fabs(lm.lpa[0] + 0.07) < 1e-6 && std::fabs(lm.rpa[0] - 0.07) < 1e-6 && std::fabs(lm.rpa[2] + 0.04) < 1e-6);
    BuildRequest rq;
    rq.recipe = "fsaverage";
    rq.sources = root / "fs";
    rq.package = root / "work" / "fs-scalp.nxr.zarr";
    rq.created_by = "test";
    rq.accept_unlisted_sources = true;
    rq.scalp = true;
    const Json info = build_template(rq);
    SubjectPackage p = SubjectPackage::open(rq.package);
    CHECK(info["scalp"]["n_vertices"] == 6 && p.db().all("SELECT 1 FROM manifold WHERE name = 'head'").size() == 1 && p.db().all("SELECT 1 FROM manifold WHERE name = 'head_landmarks' AND n_vertices = 3").size() == 1);
    CHECK(items_without_origin(p.db()).empty());
    // every position (the cortex, the scalp, the landmarks) is in FreeSurfer's surface RAS, which has a unit: metres
    CHECK(p.db().all("SELECT 1 FROM coordinate_system WHERE name = 'freesurfer-tkr' AND unit = 'm'").size() == 1);
    CHECK(p.db().all("SELECT 1 FROM coordinate_system WHERE name = 'fsaverage'").empty());
    const auto o = origin_of(p.db(), "manifold", text(*p.db().one("SELECT id FROM manifold WHERE name = 'head'"), "id"));
    CHECK(o && has(o->source, "outer_skin.surf") && has(o->licence, "FreeSurfer"));
    // a fiducials file in another frame is refused
    {
      nxr::formats::FileSink sink2((root / "fs" / "bem" / "device.fif").string());
      nxr::formats::fif::TagWriter w2(sink2);
      w2.start_file(nxr::formats::fif::Id{});
      w2.start_block(nxr::formats::fif::block::kIsotrak);
      w2.write_int(nxr::formats::fif::tag::kMneCoordFrame, 4);
      w2.end_block(nxr::formats::fif::block::kIsotrak);
      w2.end_file();
    }
    { const std::string why = what_of([&] { landmarks_from_fif(root / "fs" / "bem" / "device.fif"); }); CHECK(has(why, "surface RAS")); if (!has(why, "surface RAS")) std::printf("  got: %s\n", why.c_str()); }
  }

  // ── Brainstorm's default anatomy (ICBM152) ──
  {
    fs::create_directories(root / "bst");
    write_subjectimage(root / "bst" / "subjectimage_T1.mat");
    const std::vector<std::pair<std::string, std::vector<Scout>>> atlases = {
        {"Structures", {{"Cortex L", {1, 2, 3, 4, 5, 6}}, {"Cortex R", {7, 8, 9, 10, 11, 12}}}},
        {"Desikan-Killiany", {{"a L", {1, 2, 3, 4}}, {"b L", {5, 6}}, {"a R", {7, 8, 9, 10}}, {"b R", {11, 12}}}},
        {"Hammers", {{"h1", {1, 2, 3, 4, 5, 6}}, {"h2", {7, 8, 9, 10, 11, 12}}}},
        {"LPBA40", {{"l1", {1, 2, 3, 4, 5, 6}}, {"l2", {7, 8, 9, 10, 11, 12}}}},
        {"Mystery", {{"m1", {1, 2, 3, 4, 5, 6}}, {"m2", {7, 8, 9, 10, 11, 12}}}}};
    write_tess(root / "bst" / "tess_cortex_pial_high.mat", atlases);
    write_tess(root / "bst" / "tess_head.mat", atlases);
    {
      // the landmarks through the SCS frame: scs_mm = R (voxel * Voxsize) + T, metres
      const HeadLandmarks lm = landmarks_from_subjectimage(root / "bst" / "subjectimage_T1.mat");
      CHECK(std::fabs(lm.nasion[0] - 0.007) < 1e-9 && std::fabs(lm.nasion[1] - 0.030) < 1e-9 && std::fabs(lm.nasion[2] - 0.032) < 1e-9);   // voxel (5,1,2) x (2,3,1) = (10,3,2); R = (-3,10,2); + T = (7,30,32) mm
      BuildRequest r;
      r.recipe = "icbm152";
      r.sources = root / "bst";
      r.package = root / "work" / "icbm-scalp.nxr.zarr";
      r.created_by = "test";
      r.accept_unlisted_sources = true;
      r.scalp = true;
      const Json info = build_template(r);
      SubjectPackage p = SubjectPackage::open(r.package);
      CHECK(info["scalp"]["name"] == "head" && p.db().all("SELECT 1 FROM manifold WHERE name = 'head'").size() == 1 && p.db().all("SELECT 1 FROM manifold WHERE name = 'head_landmarks' AND n_vertices = 3").size() == 1);
      CHECK(items_without_origin(p.db()).empty());
      // without --scalp nothing of it is written
      r.scalp = false;
      r.package = root / "work" / "icbm-noscalp.nxr.zarr";
      build_template(r);
      CHECK(SubjectPackage::open(r.package).db().all("SELECT 1 FROM manifold WHERE name = 'head'").empty());
    }
    BuildRequest rq;
    rq.recipe = "icbm152";
    rq.sources = root / "bst";
    rq.package = root / "work" / "icbm-none.nxr.zarr";
    rq.created_by = "test";
    rq.accept_unlisted_sources = true;
    build_template(rq);
    {
      SubjectPackage p = SubjectPackage::open(rq.package);
      CHECK(p.db().all("SELECT 1 FROM manifold WHERE name = 'cortex_pial_high_sphere'").size() == 1);
      const auto sel = item_names(p.db(), "selection");
      for (const char* n : {"Desikan-Killiany", "Hammers", "LPBA40", "Mystery"}) CHECK(std::find(sel.begin(), sel.end(), n) == sel.end());   // none named: none written
      CHECK(items_without_origin(p.db()).empty());
    }
    rq.package = root / "work" / "icbm.nxr.zarr";
    rq.annotations = {"Desikan-Killiany", "Hammers", "LPBA40", "Mystery"};
    build_template(rq);
    SubjectPackage p = SubjectPackage::open(rq.package);
    CHECK(items_without_origin(p.db()).empty());
    auto origin_named = [&](const std::string& n) { return *origin_of(p.db(), "selection", text(*p.db().one("SELECT id FROM selection WHERE name = ?", {n}), "id")); };
    CHECK(origin_named("Desikan-Killiany").redistributable && !origin_named("Hammers").redistributable && !origin_named("LPBA40").redistributable);
    CHECK(!origin_named("Mystery").redistributable && origin_named("Mystery").licence == "not confirmed");
    CHECK(has(origin_named("Hammers").licence, "never redistributed"));
    CHECK_EQ(code_of([&] { BuildRequest r = rq; r.package = root / "work" / "z.nxr.zarr"; r.annotations = {"Absent"}; build_template(r); }), std::string(code::invalid_argument));
    // two editions: the redistributable one leaves out Hammers, LPBA40 and the atlas whose terms are not confirmed, and keeps the rest byte for byte
    const ReleaseReport full = release_package(rq.package, ReleaseOptions{"icbm152", "2026", "test", shelf});
    const ReleaseReport red = release_redistributable(full.folder, shelf, "test");
    std::string left;
    for (const auto& l : red.left_out) left += l.get<std::string>() + "; ";
    CHECK(has(left, "Hammers") && has(left, "LPBA40") && has(left, "Mystery") && !has(left, "Desikan-Killiany"));
    SubjectPackage rp = SubjectPackage::open(red.folder);
    const auto kept = item_names(rp.db(), "selection");
    CHECK(std::find(kept.begin(), kept.end(), "Desikan-Killiany") != kept.end());
    for (const char* n : {"Hammers", "LPBA40", "Mystery"}) CHECK(std::find(kept.begin(), kept.end(), n) == kept.end());
    CHECK(rp.verify().ok && rp.content_hash() != full.content_hash);
    CHECK_EQ(rp.record().release["edition"].get<std::string>(), std::string("redistributable"));
  }

  // ── the MNI frame of the anatomy ──
  {
    const auto a = mni_to_scs_from_subjectimage(root / "bst" / "subjectimage_T1.mat");
    // mni (-100, -120, -80) is the MRI origin: scs_mm = T = (10, 20, 30) mm; mni (-99, -120, -80) one mm along MRI x, which SCS.R sends to +y
    auto apply = [&](double x, double y, double z, int j) { return x * a[static_cast<std::size_t>(j)] + y * a[static_cast<std::size_t>(3 + j)] + z * a[static_cast<std::size_t>(6 + j)] + a[static_cast<std::size_t>(9 + j)]; };
    CHECK(near(apply(-100, -120, -80, 0), 0.010, 1e-12) && near(apply(-100, -120, -80, 1), 0.020, 1e-12) && near(apply(-100, -120, -80, 2), 0.030, 1e-12));
    CHECK(near(apply(-99, -120, -80, 0), 0.010, 1e-12) && near(apply(-99, -120, -80, 1), 0.021, 1e-12));
    CHECK(near(apply(-100, -119, -80, 0), 0.009, 1e-12) && near(apply(-100, -119, -80, 1), 0.020, 1e-12));
  }

  // ── the .trk reader against nibabel's numbers (made from these same bytes) ──
  const std::vector<std::vector<std::array<float, 3>>> lines = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9.5f}}, {{10, 11, 12}, {13, 14.5f, 15}}};
  {
    struct Case { std::string order; float aff[4][4]; std::vector<std::vector<std::array<double, 3>>> want; };
    const Case cases[] = {
        {"LPS", {{-2, 0, 0, 79.5f}, {0, -2.5f, 0, 81.5f}, {0, 0, 3, -72}, {0, 0, 0, 1}}, {{{79.5, 80.75, -70.5}, {76.5, 77.75, -67.5}, {73.5, 74.75, -64.0}}, {{70.5, 71.75, -61.5}, {67.5, 68.25, -58.5}}}},
        {"RAS", {{2, 0, 0, -10}, {0, 2.5f, 0, -20}, {0, 0, 3, -30}, {0, 0, 0, 1}}, {{{-10, -19.25, -28.5}, {-7, -16.25, -25.5}, {-4, -13.25, -22.0}}, {{-1, -10.25, -19.5}, {2, -6.75, -16.5}}}},
        {"LPS", {{2, 0, 0, -10}, {0, 2.5f, 0, -20}, {0, 0, 3, -30}, {0, 0, 0, 1}}, {{{8, 6.75, -28.5}, {5, 3.75, -25.5}, {2, 0.75, -22.0}}, {{-1, -2.25, -19.5}, {-4, -5.75, -16.5}}}},
        {"PSR", {{2, 0, 0, -10}, {0, 2.5f, 0, -20}, {0, 0, 3, -30}, {0, 0, 0, 1}}, {{{7.4, -18.75, -30.0}, {5.0, -16.25, -25.5}, {2.6, -13.33333, -21.0}}, {{0.2, -11.25, -16.5}, {-2.6, -8.75, -12.0}}}}};
    int k = 0;
    for (const Case& c : cases) {
      const fs::path f = root / ("case" + std::to_string(k++) + ".trk");
      write_trk(f, c.order, c.aff, lines);
      const Tractogram t = read_trk(f);
      CHECK_EQ(t.n_streamlines(), std::int64_t(2));
      CHECK(t.offsets == (std::vector<std::int64_t>{0, 3, 5}));
      bool ok = true;
      for (std::size_t s = 0; s < 2; ++s)
        for (std::size_t i = 0; i < c.want[s].size(); ++i)
          for (int d = 0; d < 3; ++d) if (!near(t.points[static_cast<std::size_t>(3 * (t.offsets[s] + static_cast<std::int64_t>(i)) + d)], c.want[s][i][static_cast<std::size_t>(d)], 2e-5)) ok = false;
      CHECK(ok);
    }
    // refusals: a truncated record, another version, an all-zero matrix, not a TrackVis file
    const float aff[4][4] = {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 1}}, zero[4][4] = {};
    write_trk(root / "ok.trk", "RAS", aff, lines);
    { std::ifstream i(root / "ok.trk", std::ios::binary); std::string all((std::istreambuf_iterator<char>(i)), {}); std::ofstream o(root / "cut.trk", std::ios::binary); o << all.substr(0, all.size() - 5); }
    CHECK(has(what_of([&] { read_trk(root / "cut.trk"); }), "truncated"));
    write_trk(root / "zero.trk", "RAS", zero, lines);
    CHECK(has(what_of([&] { read_trk(root / "zero.trk"); }), "all zero"));
    { std::ofstream o(root / "junk.trk", std::ios::binary); o << std::string(1000, 'x'); }
    CHECK(has(what_of([&] { read_trk(root / "junk.trk"); }), "not a TrackVis file"));
    // the resampler keeps both ends exactly and spaces the points equally along the line; a streamline of one point is dropped
    Tractogram t;
    t.points = {0, 0, 0, 1, 0, 0, 3, 0, 0, 5, 5, 5};
    t.offsets = {0, 3, 4};
    std::vector<std::int64_t> kept;
    const std::vector<float> r = resample_streamlines(t, 4, &kept);
    CHECK(kept == (std::vector<std::int64_t>{0}) && r.size() == 12);
    CHECK(near(r[0], 0) && near(r[3], 1) && near(r[6], 2) && near(r[9], 3) && near(r[10], 0));
  }

  // ── HCP-1065: the streamlines in the anatomy, their ends, the gate, the origin ──
  {
    const float aff[4][4] = {{-2, 0, 0, 0}, {0, -2, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 1}};
    // three streamlines in voxel-mm; with the header above (order LPS, 10 x 12 x 14 voxels of 2 x 2.5 x 3 mm) a point (x, y, z) is at RAS (-x + 1, -y*? ...): the ends are chosen
    // through the transform the reader implements, checked above; here the geometry is set by taking the anatomy to be the identity in metres of MNI mm / 1000 (no shift)
    std::vector<std::vector<std::array<float, 3>>> s = {{{20, 0, 0}, {20, 0, 3}, {20, 0, 6}},          // MNI (-20, 0, 0) is a vertex; the other end 4 mm from it
                                                        {{20, 0, 0}, {20, 0, 15}, {20, 0, 30}},         // one end on the surface, one far from it
                                                        {{60, 40, 40}, {60, 40, 41}}};                  // both far
    write_trk(root / "hcp1065.trk", "LPS", aff, s);
    const Tractogram t = read_trk(root / "hcp1065.trk");
    BuildRequest rq;
    rq.recipe = "icbm152";
    rq.sources = root / "bst";
    rq.package = root / "work" / "icbm-hcp.nxr.zarr";
    rq.created_by = "test";
    rq.accept_unlisted_sources = true;
    rq.hcp_trk = root / "hcp1065.trk";
    CHECK(has(what_of([&] { build_template(rq); }), "hcp-end-surface"));
    rq.hcp_end_surface = "no_such_surface";
    CHECK(has(what_of([&] { build_template(rq); }), "no surface 'no_such_surface'"));
    CHECK(!fs::exists(rq.package));
    rq.hcp_end_surface = "cortex_pial_high";
    rq.hcp_points = 5;
    // MNI mm (the reader's RAS mm shifted by (-1, -1, 1)) to metres, as a pure scaling: the anatomy is MNI/1000
    rq.hcp_affine = std::array<double, 12>{0.001, 0, 0, 0, 0.001, 0, 0, 0, 0.001, 0, 0, 0};
    // the surface's vertices are at x = -30 +- 10 mm etc.; find the streamline ends' distances the importer will see
    rq.hcp_max_end_distance_mm = 5;
    const Json made = build_template(rq);
    const Json h = made["hcp1065"];
    CHECK_EQ(h["streamlines"].get<int>(), 3);
    CHECK_EQ(h["points"].get<int>(), 5);
    CHECK_EQ(h["ends"].get<int>(), 6);
    SubjectPackage p = SubjectPackage::open(rq.package);
    auto dr = ArrayReader::open(p.store(), "fibre/hcp1065/end_distance_mm");
    const auto raw = dr.read();
    CHECK_EQ(raw.size(), std::size_t(6 * 4));
    std::vector<float> dist(6);
    std::memcpy(dist.data(), raw.data(), raw.size());
    int within = 0;
    for (float d : dist) if (d <= 5.0f) ++within;
    CHECK_EQ(h["ends_within_gate"].get<int>(), within);                        // the operator holds exactly the ends within the gate
    CHECK(within > 0 && within < 6);                                           // and the gate separates some from the others
    const auto op = *p.db().one("SELECT * FROM operator WHERE name = 'hcp1065_ends'");
    CHECK_EQ(integer(op, "nnz"), static_cast<std::int64_t>(within));
    // a gate between two distances separates exactly those ends; a wide gate takes all
    {
      std::vector<float> sorted(dist);
      std::sort(sorted.begin(), sorted.end());
      BuildRequest wide = rq;
      wide.package = root / "work" / "icbm-hcp-wide.nxr.zarr";
      wide.hcp_max_end_distance_mm = 1e6;
      CHECK_EQ(build_template(wide)["hcp1065"]["ends_within_gate"].get<int>(), 6);
      BuildRequest mid = rq;
      mid.package = root / "work" / "icbm-hcp-mid.nxr.zarr";
      mid.hcp_max_end_distance_mm = 0.5 * (sorted[2] + sorted[3]);
      CHECK_EQ(build_template(mid)["hcp1065"]["ends_within_gate"].get<int>(), 3);
      BuildRequest none = rq;
      none.package = root / "work" / "icbm-hcp-none.nxr.zarr";
      none.hcp_max_end_distance_mm = 0;
      CHECK_EQ(build_template(none)["hcp1065"]["ends_within_gate"].get<int>(), 0);
    }
    // the origin: source, licence CC BY-SA 4.0 with the HCP terms, attribution naming Yeh 2022, redistributable, and the nxr-origin/1 document
    CHECK(items_without_origin(p.db()).empty());
    const auto fid = text(*p.db().one("SELECT id FROM manifold WHERE name = 'hcp1065'"), "id");
    const auto o = origin_of(p.db(), "manifold", fid);
    CHECK(o && o->redistributable && has(o->licence, "CC BY-SA 4.0") && has(o->licence, "WU-Minn HCP Open Access Data Use Terms") && has(o->attribution, "Nat Commun 13:4933") &&
          has(o->attribution, "Human Connectome Project") && has(o->source, "hcp1065.trk"));
    const Json prod = Json::parse(text(*p.db().one("SELECT producer_json FROM manifold WHERE name = 'hcp1065'"), "producer_json"));
    CHECK_EQ(prod["origin"]["schema"].get<std::string>(), std::string("nxr-origin/1"));
    CHECK(prod["origin"]["licence"]["share_alike"] == true && has(prod["origin"]["terms"].get<std::string>(), "term 4"));
    CHECK(prod["origin"]["source"]["files"][0]["sha256"].get<std::string>().size() == 64);
    const auto oo = origin_of(p.db(), "operator", text(op, "id"));
    CHECK(oo && has(oo->licence, "CC BY-SA 4.0"));
    // the fibres' own checksum is checked against the manifest: this file is not the listed one
    BuildRequest strict = rq;
    strict.package = root / "work" / "icbm-hcp-strict.nxr.zarr";
    strict.accept_unlisted_sources = false;
    CHECK(has(what_of([&] { build_template(strict); }), "--accept-unlisted-sources"));
    // a template whose full edition holds the streamlines keeps them in the redistributable edition (CC BY-SA allows it with attribution)
    const ReleaseReport full = release_package(rq.package, ReleaseOptions{"icbm152-hcp", "1.0", "test", shelf});
    const ReleaseReport red = release_redistributable(full.folder, shelf, "test");
    SubjectPackage rp = SubjectPackage::open(red.folder);
    CHECK(rp.db().all("SELECT 1 FROM manifold WHERE name = 'hcp1065'").size() == 1);
    (void)t;
  }

  fs::remove_all(root);
  return nxrtest::finish("test_template_import");
}
