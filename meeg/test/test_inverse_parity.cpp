// Parity of nxr::meeg::unconstrainedKernel with Brainstorm's imaging kernel and with MNE-Python's, on the synthetic cases of golden/gen_kernel_golden.py
// (a sphere forward model, a full-rank covariance; the rank-deficient cases have an average reference and two SSP vectors). Error = max|a-b| / max|b|.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "check.h"
#include "nxr/meeg/inverse.h"

using namespace nxr::meeg;
using Eigen::MatrixXd;

namespace {
std::map<std::string, MatrixXd> readGolden(const std::string& path) {
  std::ifstream f(path);
  if (!f) { std::printf("  cannot open %s\n", path.c_str()); std::exit(1); }
  std::map<std::string, MatrixXd> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream h(line);
    std::string name;
    int r, c;
    h >> name >> r >> c;
    MatrixXd m(r, c);
    std::getline(f, line);
    std::istringstream v(line);
    for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) v >> m(i, j);
    out[name] = m;
  }
  return out;
}
double rel(const MatrixXd& a, const MatrixXd& b) { return a.rows() != b.rows() || a.cols() != b.cols() ? 1e9 : (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff(); }
}  // namespace

int main() {
  const auto g = readGolden(std::string(NXR_TEST_GOLDEN_DIR) + "/kernel_golden.txt");
  const char* kind[] = {"MEG MAG", "MEG GRAD", "EEG"};
  struct Variant { const char* name; KernelOptions o; };
  std::vector<Variant> vs(4);
  vs[0] = {"default", {}};
  vs[1] = {"nodepth", {}}; vs[1].o.depth.enabled = false;
  vs[2] = {"diag", {}};    vs[2].o.whitener.regularisation = Regularisation::Diagonal;
  vs[3] = {"none", {}};    vs[3].o.whitener.regularisation = Regularisation::None;
  double worstBstK = 0, worstBstW = 0, worstMne = 0;
  for (const char* cn : {"full_megeeg", "full_meg", "deficient_megeeg", "deficient_meg"}) {
    const std::string c = cn;
    const MatrixXd G = g.at(c + ".G"), C = g.at(c + ".C"), T = g.at(c + ".types");
    std::vector<std::string> types;
    for (int i = 0; i < T.rows(); ++i) types.push_back(kind[static_cast<int>(T(i, 0))]);
    for (const Variant& v : vs) {
      const std::string key = c + ".bst_" + v.name;
      if (!g.count(key + ".K")) continue;
      const Kernel k = unconstrainedKernel(G, C, types, v.o);
      const double ek = rel(k.K, g.at(key + ".K")), ew = rel(k.W, g.at(key + ".W"));
      std::printf("  Brainstorm %-17s %-8s K %.2e  W %.2e  rank %d\n", cn, v.name, ek, ew, k.whitenerRank);
      worstBstK = std::max(worstBstK, ek);
      worstBstW = std::max(worstBstW, ew);
    }
    const Kernel km = unconstrainedKernel(G, C, types, mneKernelOptions());
    const double em = rel(km.K, g.at(c + ".K_mne"));
    std::printf("  MNE-Python %-17s K %.2e  rank %d\n", cn, em, km.whitenerRank);
    worstMne = std::max(worstMne, em);
    if (g.count(c + ".K_mne_nodepth")) {
      KernelOptions o = mneKernelOptions();
      o.depth.enabled = false;
      std::printf("  MNE-Python %-17s no depth: K %.2e\n", cn, rel(unconstrainedKernel(G, C, types, o).K, g.at(c + ".K_mne_nodepth")));
    }
  }
  std::printf("  worst: Brainstorm K %.2e W %.2e; MNE-Python K %.2e\n", worstBstK, worstBstW, worstMne);
  // the inputs are the golden's 12-digit numbers; the references and this code differ by their rounding and eigensolvers only
  CHECK(worstBstK < 1e-10);
  CHECK(worstBstW < 1e-10);
  CHECK(worstMne < 1e-9);
  return nxrtest::finish("test_inverse_parity");
}
