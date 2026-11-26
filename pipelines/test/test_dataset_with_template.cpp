// `create_dataset_with_template` on a synthetic template (two icospheres as the hemispheres, a Medial_wall cap on each facing the other): the dataset, its default subject
// with the frames and the group tree, and, with `modes`, the pial's Hodge–Dirac basis SOLVED in the stored layout; a failure (no surf/ folder) leaves no dataset.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

#include "check.h"
#include "nxr/pipelines/dataset.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

namespace {
void icosphere(int levels, double radius, std::vector<double>& V, std::vector<std::int32_t>& F) {
  const double t = (1 + std::sqrt(5.0)) / 2;
  const double base[12][3] = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  const int tri[20][3] = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                          {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
  V.clear(); F.clear();
  for (auto& b : base) V.insert(V.end(), b, b + 3);
  for (auto& f : tri) F.insert(F.end(), f, f + 3);
  for (int l = 0; l < levels; ++l) {
    std::map<std::pair<int, int>, int> mid;
    auto midpoint = [&](int a, int b) {
      auto k = std::make_pair(std::min(a, b), std::max(a, b));
      auto it = mid.find(k);
      if (it != mid.end()) return it->second;
      const int id = static_cast<int>(V.size() / 3);
      for (int c = 0; c < 3; ++c) V.push_back((V[3 * a + c] + V[3 * b + c]) / 2);
      return mid[k] = id;
    };
    std::vector<std::int32_t> G;
    for (std::size_t f = 0; f < F.size(); f += 3) {
      const int a = F[f], b = F[f + 1], c = F[f + 2], ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
      for (int q : {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca}) G.push_back(q);
    }
    F = G;
  }
  for (std::size_t i = 0; i < V.size(); i += 3) {
    const double n = std::sqrt(V[i] * V[i] + V[i + 1] * V[i + 1] + V[i + 2] * V[i + 2]);
    for (int c = 0; c < 3; ++c) V[i + static_cast<std::size_t>(c)] *= radius / n;
  }
}
}  // namespace

namespace {
void be32(std::ofstream& o, std::uint32_t x) { const char b[4] = {char(x >> 24), char(x >> 16), char(x >> 8), char(x)}; o.write(b, 4); }
void write_fs(const fs::path& p, const std::vector<double>& v, const std::vector<std::int32_t>& f) {   // a FreeSurfer triangle file
  std::ofstream o(p, std::ios::binary);
  o.write("\xff\xff\xfe", 3);
  o << "created by test on today\n\n";
  be32(o, static_cast<std::uint32_t>(v.size() / 3));
  be32(o, static_cast<std::uint32_t>(f.size() / 3));
  for (double x : v) { const float y = static_cast<float>(x); std::uint32_t u; std::memcpy(&u, &y, 4); be32(o, u); }
  for (auto i : f) be32(o, static_cast<std::uint32_t>(i));
}
// a FreeSurfer annotation (colour-table version 2): label 0 `unknown`, 1 `Medial_wall`, 2 `frontal`
void write_annot(const fs::path& p, const std::vector<int>& label) {
  std::ofstream o(p, std::ios::binary);
  const int rgb[3][3] = {{25, 5, 25}, {80, 80, 80}, {220, 20, 10}};
  const char* names[3] = {"unknown", "Medial_wall", "frontal"};
  be32(o, static_cast<std::uint32_t>(label.size()));
  for (std::size_t v = 0; v < label.size(); ++v) { be32(o, static_cast<std::uint32_t>(v)); be32(o, static_cast<std::uint32_t>(rgb[label[v]][0] + rgb[label[v]][1] * 256 + rgb[label[v]][2] * 65536)); }
  be32(o, 1); be32(o, static_cast<std::uint32_t>(-2)); be32(o, 3);
  be32(o, 1); o.write("\0", 1);
  be32(o, 3);
  for (int k = 0; k < 3; ++k) {
    be32(o, static_cast<std::uint32_t>(k));
    const std::string n = std::string(names[k]) + '\0';
    be32(o, static_cast<std::uint32_t>(n.size()));
    o.write(n.data(), static_cast<std::streamsize>(n.size()));
    for (int c = 0; c < 3; ++c) be32(o, static_cast<std::uint32_t>(rgb[k][c]));
    be32(o, 0);
  }
}
std::string code_of(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_dataset_template_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const fs::path fsdir = root / "templates" / "fsaverage5";
  fs::create_directories(fsdir / "surf");
  fs::create_directories(fsdir / "label");
  // the template: two icospheres of radius 40 mm at x = -/+60 mm (white and inflated the same shape); sphere.reg an icosphere of radius 100
  std::vector<double> V, S;
  std::vector<std::int32_t> F, F2;
  icosphere(2, 40.0, V, F);
  icosphere(2, 100.0, S, F2);
  std::vector<double> L = V, R = V;
  for (std::size_t i = 0; i < V.size(); i += 3) { L[i] -= 60; R[i] += 60; }
  std::vector<int> ll, rl;
  for (std::size_t i = 0; i < V.size(); i += 3) { ll.push_back(V[i] > 25 ? 1 : 2); rl.push_back(V[i] < -25 ? 1 : 2); }
  for (const char* n : {"pial", "white", "inflated"}) { write_fs(fsdir / "surf" / (std::string("lh.") + n), L, F); write_fs(fsdir / "surf" / (std::string("rh.") + n), R, F); }
  write_fs(fsdir / "surf" / "lh.sphere.reg", S, F2);
  write_fs(fsdir / "surf" / "rh.sphere.reg", S, F2);
  write_annot(fsdir / "label" / "lh.aparc.annot", ll);
  write_annot(fsdir / "label" / "rh.aparc.annot", rl);

  DatasetWithTemplateOptions o;
  o.subject.templates = root / "templates";
  o.atlas_finest = "parts 2";                    // (was depth 1: parts per hemisphere 2^1)
  o.lc_scales = {"parts 2"};                     // (was lc_levels {1})
  o.modes = 20;
  const Json r = create_dataset_with_template(root / "store", "demo", o, "test");
  const Json& d = r["default_subject"];
  CHECK_EQ(d["surfaces"].size(), std::size_t(3));
  CHECK(d["frames"].is_object() && d["group_tree"].is_object());
  CHECK(d["modes"].is_object() && d["modes"]["glued"].get<bool>());

  DatasetWriter ds = open_writer(root / "store" / "demo");
  SubjectWriter sub = SubjectWriter::open(ds, kDefaultSubject);
  CHECK(sub.find("operator", "hodgeDirac_eigenvectors").has_value());      // the basis, in the layout `eigen solve` stores for any subject
  CHECK(!sub.find("selection", "hodgeDirac_support").has_value());     // no support: the glue's correspondence is derived from the glued cortex's labelling and the medial wall
  CHECK(sub.find("manifold", "source/cortex_pial_glued").has_value());
  ds.close();

  // without modes: the default subject, frames and group tree, no basis
  o.modes.reset();
  const Json r2 = create_dataset_with_template(root / "store", "plain", o, "test");
  CHECK(r2["default_subject"]["modes"].is_null());

  // a failure leaves no dataset
  o.subject.templates = root / "nowhere";
  CHECK(!code_of([&] { create_dataset_with_template(root / "store", "bad", o, "test"); }).empty());
  CHECK(!fs::exists(root / "store" / "bad"));

  fs::remove_all(root);
  std::cout << "test_dataset_with_template: ok\n";
  return 0;
}
