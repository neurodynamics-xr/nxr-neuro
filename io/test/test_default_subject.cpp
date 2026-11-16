// The default subject on synthetic data: the FreeSurfer reader, the .npy reader, the surfaces written as any subject's, and the refusals.
// (Parity with nxr: test/parity/test_default_subject_parity.py. The Hodge–Dirac basis is solved by the pipelines: pipelines/test/test_pipelines.cpp.)
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>

#include "check.h"
#include "store_at.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/default_subject.h"
#include "nxr/io/npy.h"
#include "nxr/io/surface.h"
#include "nxr/io/writer.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {

std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); }
  return "";
}

void be32(std::ofstream& o, std::uint32_t x) { const char b[4] = {char(x >> 24), char(x >> 16), char(x >> 8), char(x)}; o.write(b, 4); }

// a FreeSurfer triangle file
void write_fs(const fs::path& p, const std::vector<double>& v, const std::vector<std::int32_t>& f) {
  std::ofstream o(p, std::ios::binary);
  o.write("\xff\xff\xfe", 3);
  o << "created by test_default_subject on today\n\n";
  be32(o, static_cast<std::uint32_t>(v.size() / 3));
  be32(o, static_cast<std::uint32_t>(f.size() / 3));
  for (double x : v) { const float y = static_cast<float>(x); std::uint32_t u; std::memcpy(&u, &y, 4); be32(o, u); }
  for (auto i : f) be32(o, static_cast<std::uint32_t>(i));
}

// a .npy file (format 1.0, C order)
template <class T>
void write_npy(const fs::path& p, const std::vector<T>& a, const std::vector<std::int64_t>& shape, const std::string& descr) {
  std::string sh = "(";
  for (std::size_t i = 0; i < shape.size(); ++i) sh += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
  sh += ")";
  std::string h = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + sh + ", }";
  while ((10 + h.size() + 1) % 64) h += ' ';
  h += '\n';
  std::ofstream o(p, std::ios::binary);
  o.write("\x93NUMPY\x01\x00", 8);
  const char len[2] = {char(h.size() & 0xff), char(h.size() >> 8)};
  o.write(len, 2);
  o << h;
  o.write(reinterpret_cast<const char*>(a.data()), static_cast<std::streamsize>(a.size() * sizeof(T)));
}

// an octahedron (6 vertices, 8 faces, ccw outward) of radius r (mm) round c
void octahedron(double cx, double r, std::vector<double>& v, std::vector<std::int32_t>& f) {
  v = {cx + r, 0, 0, cx - r, 0, 0, cx, r, 0, cx, -r, 0, cx, 0, r, cx, 0, -r};
  f = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
}

template <class T>
std::vector<T> stored(const Store& st, const std::string& key) {   // through the dataset's store: a file or a row of the subject's small-file store
  ArrayReader r = ArrayReader::open(st, key);
  const auto raw = r.read();
  std::vector<T> v(raw.size() / sizeof(T));
  std::memcpy(v.data(), raw.data(), raw.size());
  return v;
}

}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_default_subject_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const fs::path fsdir = root / "templates" / "fsaverage5";
  fs::create_directories(fsdir / "surf");

  // ── a FreeSurfer subject: two octahedra (left at x = −30 mm, right at +30), pial · white (×0.9) · inflated (×1.2) · sphere.reg (radius 100) ──
  std::vector<double> lh, rh;
  std::vector<std::int32_t> f;
  octahedron(-30, 10, lh, f);
  octahedron(30, 10, rh, f);
  auto scaled = [](std::vector<double> v, double cx, double s) { for (std::size_t i = 0; i < v.size(); ++i) v[i] = (i % 3 == 0 ? cx + (v[i] - cx) * s : v[i] * s); return v; };
  for (const auto& [name, s] : std::vector<std::pair<std::string, double>>{{"pial", 1.0}, {"white", 0.9}, {"inflated", 1.2}}) {
    write_fs(fsdir / "surf" / ("lh." + name), scaled(lh, -30, s), f);
    write_fs(fsdir / "surf" / ("rh." + name), scaled(rh, 30, s), f);
  }
  std::vector<double> sph;
  octahedron(0, 100, sph, f);
  write_fs(fsdir / "surf" / "lh.sphere.reg", sph, f);
  write_fs(fsdir / "surf" / "rh.sphere.reg", sph, f);

  // the reader: float32 positions as stored, faces as stored; a quadrangle file refused
  {
    const Mesh m = read_freesurfer_surface(fsdir / "surf" / "lh.pial");
    CHECK_EQ(m.n_vertices(), std::int64_t(6));
    CHECK_EQ(m.n_faces(), std::int64_t(8));
    CHECK(m.vertices[0] == -20.0 && m.vertices[3] == -40.0);
    CHECK(m.faces == f);
    CHECK_EQ(read_mesh(fsdir / "surf" / "rh.pial").n_vertices(), std::int64_t(6));   // read_mesh reads lh.* / rh.* with it
    std::ofstream q(root / "lh.quad", std::ios::binary);
    q.write("\xff\xff\xff\x00\x00\x01", 6);
    q.close();
    CHECK_EQ(code_of([&] { read_freesurfer_surface(root / "lh.quad"); }), std::string(code::unsupported));
  }

  // the .npy reader: shape, dtype, rows
  {
    fs::create_directories(root / "npy");
    const std::vector<double> a = {1, 2, 3, 4, 5, 6};
    write_npy(root / "npy" / "a.npy", a, {2, 3}, "<f8");
    const NpyFile n = NpyFile::open(root / "npy" / "a.npy");
    CHECK(n.shape() == (std::vector<std::int64_t>{2, 3}));
    CHECK_EQ(n.descr(), std::string("<f8"));
    CHECK(n.rows<double>(1, 2) == (std::vector<double>{4, 5, 6}));
  }

  // ── the dataset and its default subject ──
  {
    DatasetWriter ds = create_dataset(root / "store", "demo");
    DefaultSubjectOptions o;
    o.templates = root / "templates";
    const Json r = create_default_subject(ds, o, "test");
    CHECK_EQ(r["surfaces"].size(), std::size_t(3));

    const auto subj = ds.subject(kDefaultSubject);
    CHECK(subj.has_value());
    CHECK_EQ(text(*subj, "kind"), std::string("template"));
    CHECK_EQ(text(*subj, "path"), std::string("@default_subject.nxr.zarr"));
    const std::string store = join_key(text(ds.row(), "path"), "@default_subject.nxr.zarr");

    // the surfaces, as any subject's: the pial primary in metres with its hemispheres and sphere; white and inflated
    auto m = [&](const std::string& name) { return ds.db().one("SELECT * FROM manifold WHERE subject_id = ? AND name = ?", {V(text(*subj, "id")), V(name)}); };
    for (const char* n : {"cortex_pial", "cortex_pial_1", "cortex_pial_2", "cortex_pial_sphere", "cortex_white", "cortex_inflated"})
      CHECK(m(n).has_value());
    CHECK_EQ(integer(*m("cortex_pial"), "is_primary"), std::int64_t(1));
    CHECK_EQ(integer(*m("cortex_white"), "is_primary"), std::int64_t(0));
    CHECK_EQ(text(*m("cortex_pial_1"), "label"), std::string("left"));
    CHECK_EQ(integer(*m("cortex_pial_2"), "from_index"), std::int64_t(6));
    // the parcellations: no label/ folder here, so Structures alone (1 left, 2 right); the annotation families are left out, not half-written
    const auto set = ds.db().one("SELECT * FROM selection WHERE path = 'source/cortex_pial_atlases'");
    CHECK(set.has_value() && integer(*set, "n_members") == 1);
    CHECK(!ds.db().one("SELECT 1 FROM selection WHERE path = 'source/cortex_pial_atlases/Desikan-Killiany'").has_value());
    const auto structures = ds.db().one("SELECT * FROM selection WHERE path = 'source/cortex_pial_atlases/Structures'");
    CHECK(structures.has_value());
    if (structures) {
      const auto codes = stored<std::int32_t>(ds.store(), join_key(store, text(*structures, "array_path")));
      const auto e1 = ds.db().one("SELECT name FROM dictionary_entry WHERE dictionary_id = ? AND code = ?", {(*structures).at("dictionary_id"), V(static_cast<std::int64_t>(codes[0]))});
      const auto e2 = ds.db().one("SELECT name FROM dictionary_entry WHERE dictionary_id = ? AND code = ?", {(*structures).at("dictionary_id"), V(static_cast<std::int64_t>(codes[11]))});
      CHECK(codes.size() == 12 && e1 && text(*e1, "name") == "Cortex L" && e2 && text(*e2, "name") == "Cortex R");
    }
    const auto pos = stored<double>(ds.store(), join_key(store, "source/cortex_pial/vertices"));
    CHECK(pos.size() == 36 && std::abs(pos[0] + 0.020) < 1e-12 && std::abs(pos[18] - 0.040) < 1e-12);
    const auto sp = stored<double>(ds.store(), join_key(store, "source/cortex_pial_sphere/vertices"));
    CHECK(sp.size() == 36 && sp[0] == 100.0);                // the sphere as stored, no unit change (nxr's default subject)

    // a second default subject is refused; a subject created now points at this one
    CHECK_EQ(code_of([&] { create_default_subject(ds, o, "test"); }), std::string(code::already_exists));
    SubjectWriter s1 = SubjectWriter::create(ds, "S1");
    s1.complete();
    CHECK_EQ(text(*ds.subject("S1"), "template_id"), text(*subj, "id"));
  }

  // ── a refusal that leaves NO default subject (one composition): a templates folder with no surf/ ──
  {
    DatasetWriter ds = create_dataset(root / "store", "moved");
    DefaultSubjectOptions o;
    o.templates = root / "none";
    CHECK_EQ(code_of([&] { create_default_subject(ds, o, "test"); }), std::string(code::not_found));
  }

  // ── A MANIFOLD'S DELETE takes the geometries and topologies no manifold stands on any more, never one another manifold shares ──
  {
    DatasetWriter ds = create_dataset(root / "store-del", "d", {});
    DefaultSubjectOptions o;
    o.templates = root / "templates";
    create_default_subject(ds, o, "test");
    Database& db = ds.db();
    auto id_of = [&](const std::string& name) { return text(*db.one("SELECT id FROM manifold WHERE name = ?", {name}), "id"); };
    auto count = [&](const std::string& sql) { return std::get<std::int64_t>(db.one(sql)->at("n")); };
    const std::string orphans_g = "SELECT count(*) AS n FROM geometry g WHERE NOT EXISTS (SELECT 1 FROM manifold m WHERE m.geometry_id = g.id)";
    const std::string orphans_t = "SELECT count(*) AS n FROM topology t WHERE NOT EXISTS (SELECT 1 FROM manifold m WHERE m.topology_id = t.id)";
    const std::string store = join_key(text(ds.row(), "path"), "@default_subject.nxr.zarr");
    // the sphere SHARES the pial's topology: its own geometry goes, the shared topology and the pial's faces stay
    const std::string pial_top = text(*db.one("SELECT topology_id FROM manifold WHERE name = 'cortex_pial'"), "topology_id");
    remove_node(db, ds.store(), "manifold", id_of("cortex_pial_sphere"));
    CHECK(db.one("SELECT 1 FROM topology WHERE id = ?", {pial_top}).has_value());
    CHECK(ds.store().exists_prefix(join_key(store, "source/cortex_pial/faces/zarr.json")));
    CHECK(!ds.store().exists_prefix(join_key(store, "source/cortex_pial_sphere")));
    CHECK_EQ(count(orphans_g), std::int64_t(0));
    CHECK_EQ(count(orphans_t), std::int64_t(0));
    // the white surface: its components cascade, and every geometry and topology of the three goes with them, and its folder
    remove_node(db, ds.store(), "manifold", id_of("cortex_white"));
    CHECK(!db.one("SELECT 1 FROM manifold WHERE name LIKE 'cortex_white%'").has_value());
    CHECK_EQ(count(orphans_g), std::int64_t(0));
    CHECK_EQ(count(orphans_t), std::int64_t(0));
    CHECK(!ds.store().exists_prefix(join_key(store, "source/cortex_white")));
    CHECK(ds.store().exists_prefix(join_key(store, "source/cortex_inflated/faces/zarr.json")));
  }

  fs::remove_all(root);
  return nxrtest::finish("test_default_subject");
}
