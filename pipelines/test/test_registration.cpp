// `registration create` on synthetic data: two templates whose registered spheres are subdivided icosahedra (two pieces each, left then right), released; the
// registration built through the pipeline, its operators read back and checked against the compute node, its origin derived from the spheres', its refusals,
// and a field carried through it released as a registration package.
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <map>
#include <tuple>

#include "check.h"
#include "nxr/io/templates.h"
#include "nxr/pipelines/registration.h"
#include "nxr/sphere_registration.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
struct TestMesh { std::vector<double> p; std::vector<std::int32_t> f; };

TestMesh icosphere(int level) {
  const double t = (1 + std::sqrt(5.0)) / 2;
  TestMesh m;
  const double base[12][3] = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  for (auto& b : base) { const double n = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]); for (int k = 0; k < 3; ++k) m.p.push_back(b[k] / n); }
  m.f = {0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1};
  for (int l = 0; l < level; ++l) {
    std::map<std::pair<int, int>, int> mid;
    auto get = [&](int a, int b) {
      const auto key = std::make_pair(std::min(a, b), std::max(a, b));
      if (const auto it = mid.find(key); it != mid.end()) return it->second;
      double q[3], n = 0;
      for (int k = 0; k < 3; ++k) { q[k] = m.p[static_cast<std::size_t>(3 * a + k)] + m.p[static_cast<std::size_t>(3 * b + k)]; n += q[k] * q[k]; }
      n = std::sqrt(n);
      for (int k = 0; k < 3; ++k) m.p.push_back(q[k] / n);
      return mid[key] = static_cast<int>(m.p.size() / 3 - 1);
    };
    std::vector<std::int32_t> nf;
    for (std::size_t i = 0; i < m.f.size(); i += 3) {
      const int a = m.f[i], b = m.f[i + 1], c = m.f[i + 2], ab = get(a, b), bc = get(b, c), ca = get(c, a);
      for (int v : {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca}) nf.push_back(v);
    }
    m.f = nf;
  }
  return m;
}
TestMesh two_pieces(const TestMesh& a) {
  TestMesh m = a;
  const std::int32_t n = static_cast<std::int32_t>(a.p.size() / 3);
  m.p.insert(m.p.end(), a.p.begin(), a.p.end());
  for (auto f : a.f) m.f.push_back(f + n);
  return m;
}

fs::path make_template(const fs::path& at, const std::string& name, const TestMesh& sphere, bool free_to_share, const std::string& licence) {
  CreatePackageOptions po;
  po.home_dataset = "templates-work";
  po.subject.kind = "template";
  DatasetWriter w = create_package(at, name, po);
  {
    SubjectWriter sub = SubjectWriter::open(w, name);
    const std::int64_t n = static_cast<std::int64_t>(sphere.p.size() / 3), nf = static_cast<std::int64_t>(sphere.f.size() / 3);
    ManifoldSpec m;
    m.name = "cortex_pial_sphere"; m.path = "source/cortex_pial_sphere"; m.type = "surface"; m.n_vertices = n;
    m.topology = {"simplicial", 2, 1, std::string("ccw-outward"), {{2, "source/cortex_pial_sphere/faces"}}};
    GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "source/cortex_pial_sphere/vertices";
    m.geometry = g;
    m.populate = [&](Store&, const std::string&) {
      sub.write_array("source/cortex_pial_sphere/vertices", view(sphere.p, {n, 3}));
      sub.write_array("source/cortex_pial_sphere/faces", view(sphere.f, {nf, 3}));
    };
    sub.manifold(m);
    sub.complete();
  }
  for (const auto& it : package_items(w.db())) set_origin(w.db(), it.table, it.id, OriginRecord{"a synthetic sphere (" + name + ")", licence, "the test authors of " + name, free_to_share});
  return at;
}
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); }
  return "";
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_registration_test_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  ::unsetenv(kTemplatePathEnv);
  const fs::path shelf = root / "shelf";
  const TestMesh coarse = two_pieces(icosphere(2)), fine = two_pieces(icosphere(3));   // 2 x 162 and 2 x 642 vertices

  for (const auto& [name, mesh, free_to_share, licence] : {std::tuple<std::string, const TestMesh*, bool, std::string>{"coarse", &coarse, true, "CC0"}, {"fine", &fine, false, "a restricted licence"}, {"third", &coarse, true, "MIT"}}) {
    const fs::path w = make_template(root / "work" / (name + ".nxr.zarr"), name, *mesh, free_to_share, licence);
    release_package(w, ReleaseOptions{name, "1.0", "test", shelf});
  }
  nxr::pipelines::RegistrationCreateOptions o;
  o.package = root / "work" / "coarse-to-fine.nxr.zarr";
  o.name = "coarse-to-fine";
  o.from = "coarse@1.0";
  o.to = "fine@1.0";
  o.from_sphere = o.to_sphere = "cortex_pial_sphere";
  o.templates = {shelf};
  o.created_by = "test";

  // refusals
  {
    auto bad = [&](auto edit) { auto b = o; edit(b); b.package = root / "work" / "bad.nxr.zarr"; return code_of([&] { nxr::pipelines::registration_create(b); }); };
    CHECK_EQ(bad([](auto& b) { b.method = "linear"; }), std::string(code::invalid_argument));
    CHECK_EQ(bad([](auto& b) { b.to = b.from; }), std::string(code::invalid_argument));
    CHECK_EQ(bad([](auto& b) { b.from_sphere.clear(); }), std::string(code::invalid_argument));
    CHECK_EQ(bad([](auto& b) { b.neighbours = 0; }), std::string(code::invalid_argument));
    CHECK(!bad([](auto& b) { b.from_sphere = "no_such_surface"; }).empty());
    CHECK(!bad([](auto& b) { b.to = "absent@9.9"; }).empty());
    CHECK(!fs::exists(root / "work" / "bad.nxr.zarr"));   // a refusal leaves nothing behind
  }

  for (const char* method : {"shepard", "nearest", "barycentric"}) {
    auto b = o;
    b.method = method;
    b.package = root / "work" / (std::string("coarse-to-fine-") + method + ".nxr.zarr");
    const auto res = nxr::pipelines::registration_create(b);
    SubjectPackage p = SubjectPackage::open(res.package);
    CHECK_EQ(p.record().kind, std::string("registration"));
    CHECK(p.record().template_reference["from"]["name"] == "coarse" && p.record().template_reference["to"]["name"] == "fine");
    CHECK(p.record().description["method"].get<std::string>().size() > 5 && p.record().description["warnings"].size() == 3);
    CHECK(p.db().all("PRAGMA foreign_key_check").empty());
    CHECK(items_without_origin(p.db()).empty());
    CHECK_EQ(res.info["operators"].size(), std::size_t(2));
    CHECK(res.info["redistributable"] == false);                                   // one sphere is not redistributable: neither is the registration
    CHECK(res.info["licence"].get<std::string>().find("CC0") != std::string::npos && res.info["licence"].get<std::string>().find("restricted") != std::string::npos);
    const auto ops = p.db().all("SELECT * FROM operator WHERE kind = 'resampling' ORDER BY name");
    CHECK_EQ(ops.size(), std::size_t(2));
    // the stored matrix is the compute node's: coarse to fine is [1284 x 324]
    const auto fwd = *p.db().one("SELECT * FROM operator WHERE name = 'coarse_to_fine'");
    CHECK(integer(fwd, "n_rows") == 1284 && integer(fwd, "n_cols") == 324 && text(fwd, "layout") == "csr");
    const auto prod = Json::parse(text(fwd, "producer_json"));
    CHECK(prod["method"] == method);
    ArrayReader ptr = ArrayReader::open(p.store(), text(fwd, "path") + "/indptr");
    CHECK_EQ(ptr.shape()[0], std::int64_t(1285));
    nxr::registration::Options ro;
    ro.method = nxr::registration::parse_method(method);
    const auto want = nxr::registration::resample_operator<double>(coarse.p, coarse.f, {{0, 162}, {162, 162}}, fine.p, {{0, 642}, {642, 642}}, ro);
    CHECK_EQ(integer(fwd, "nnz"), want.nnz());
    ArrayReader dat = ArrayReader::open(p.store(), text(fwd, "path") + "/data");
    const auto raw = dat.read();
    CHECK_EQ(raw.size(), static_cast<std::size_t>(want.nnz()) * sizeof(double));
    if (raw.size() == static_cast<std::size_t>(want.nnz()) * sizeof(double)) {
      std::vector<double> got(static_cast<std::size_t>(want.nnz()));
      std::memcpy(got.data(), raw.data(), raw.size());
      CHECK(got == want.data);
    }
  }

  // one direction only; then the released registration is found by name and version, is frozen, and its redistributable edition is refused (its origin says no)
  {
    auto b = o;
    b.both_directions = false;
    b.package = root / "work" / "one-way.nxr.zarr";
    const auto res = nxr::pipelines::registration_create(b);
    CHECK_EQ(res.info["operators"].size(), std::size_t(1));
    CHECK_EQ(SubjectPackage::open(res.package).db().all("SELECT 1 FROM operator").size(), std::size_t(1));
    const ReleaseReport rr = release_package(res.package, ReleaseOptions{"coarse-to-fine", "1.0", "test", shelf});
    TemplateSearch here;
    here.command_paths = {shelf};
    here.use_environment = false;
    here.settings_file = root / "no-settings.json";
    TemplateRef ref = parse_template_ref("coarse-to-fine@1.0");
    ref.content_hash = rr.content_hash;
    const ResolvedTemplate reg = resolve_template(ref, here);
    CHECK_EQ(reg.record.kind, std::string("registration"));
    CHECK_EQ(code_of([&] { open_package_writer(reg.folder); }), std::string(code::not_writable));
    // a registration BETWEEN two redistributable templates is redistributable, and its licences are both named
    auto c = o;
    c.to = "third@1.0";
    c.name = "coarse-to-third";
    c.package = root / "work" / "coarse-to-third.nxr.zarr";
    const auto r2 = nxr::pipelines::registration_create(c);
    CHECK(r2.info["redistributable"] == true && r2.info["licence"] == "CC0; MIT");
    // a caller's origin replaces the derived one
    c.package = root / "work" / "coarse-to-third-own.nxr.zarr";
    c.licence = "CC-BY-4.0";
    c.redistributable = true;
    CHECK(nxr::pipelines::registration_create(c).info["licence"] == "CC-BY-4.0");
  }
  return nxrtest::finish("test_registration");
}
