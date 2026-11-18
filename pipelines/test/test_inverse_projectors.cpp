// The minimum-norm kernel WITH THE RECORDING'S PROJECTORS (the node meeg.minimum_norm: minimum_norm_inputs, then nxr::meeg::unconstrainedKernel) against the kernels
// Brainstorm and MNE-Python compute from the same unprojected gain and noise covariance with one active SSP projector, each applying it its own way
// (meeg/test/golden/gen_kernel_projectors_golden.py and .m; synthetic data only). The covariance is given as recorded and as of projected data. Error = max|a-b| / max|b|.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

#include "check.h"
#include "nxr/meeg/inverse.h"
#include "nxr/pipelines/meeg_steps.h"

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
  namespace g = nxr::pipelines::graph;
  const auto gold = readGolden(std::string(NXR_TEST_GOLDEN_DIR) + "/kernel_projectors_golden.txt");
  const MatrixXd G = gold.at("G"), u = gold.at("u"), T = gold.at("types");
  const int M = static_cast<int>(G.rows());
  std::vector<std::string> types;
  std::vector<std::int32_t> pick;
  for (int i = 0; i < M; ++i) { types.push_back(T(i, 0) == 0 ? "MEG MAG" : "MEG GRAD"); pick.push_back(i); }
  nxr::bst::StoredProjector p;
  p.name = "channels_projector_1"; p.kind = "ssp"; p.method = "SSP"; p.order = 1; p.status = 1; p.m = static_cast<std::size_t>(M);
  p.vectors.push_back(std::vector<double>(u.data(), u.data() + M));
  double worst_bst = 0, worst_mne = 0;
  for (const char* k : {"raw", "proj"}) {
    const MatrixXd C = gold.at(std::string("C_") + k);
    const std::vector<std::string> cov_projs = std::string(k) == "proj" ? std::vector<std::string>{p.name} : std::vector<std::string>{};
    const g::MinNormInputs b = g::minimum_norm_inputs(G, C, pick, types, {p}, cov_projs, "brainstorm", true);
    const double eb = rel(nxr::meeg::unconstrainedKernel(b.G, b.C, types).K, gold.at(std::string("K_bst_") + k));
    const g::MinNormInputs m = g::minimum_norm_inputs(G, C, pick, types, {p}, cov_projs, "mne", true);
    MatrixXd Km = nxr::meeg::unconstrainedKernel(m.G, m.C, types, nxr::meeg::mneKernelOptions()).K;
    CHECK(m.right.has_value());
    if (m.right) Km = Km * *m.right;
    const double em = rel(Km, gold.at(std::string("K_mne_") + k));
    std::printf("  covariance %-4s  Brainstorm K %.2e   MNE-Python K %.2e\n", k, eb, em);
    worst_bst = std::max(worst_bst, eb);
    worst_mne = std::max(worst_mne, em);
  }
  // the references' own precision: the golden holds 17 digits; the references and Eigen differ by their eigensolvers' rounding only (measured: 4e-15 and 6e-14)
  CHECK(worst_bst < 1e-12);
  CHECK(worst_mne < 1e-12);
  return nxrtest::finish("test_inverse_projectors");
}
