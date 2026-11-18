// nxr::scalespace::selectLevel: the level of the largest normalised power (power over the squared norm of the atom), the lowest on a tie, the response returned, columns independent (pointwise in the frames),
// elements independent; white noise (power proportional to the norm) gives no preferred level: every level ties; refusals.
#include <iostream>
#include <random>

#include "nxr/select/scale_selection.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_scalespace\n";
  const int L = 4;
  const Eigen::Index E = 5, F = 6;
  std::mt19937 rng(8);
  std::uniform_real_distribution<double> u(0.1, 2.0);
  Eigen::VectorXd n2(E * L);
  for (Eigen::Index i = 0; i < n2.size(); ++i) n2(i) = u(rng);
  Eigen::MatrixXd P(E * L, F);
  for (Eigen::Index i = 0; i < P.rows(); ++i) for (Eigen::Index j = 0; j < F; ++j) P(i, j) = u(rng);
  const auto c = nxr::scalespace::selectLevel(P, n2, L);
  bool same = c.index.rows() == E && c.index.cols() == F;
  for (Eigen::Index e = 0; e < E && same; ++e)
    for (Eigen::Index j = 0; j < F; ++j) {
      int at = 0;
      double best = -1;
      for (int l = 0; l < L; ++l) if (P(e * L + l, j) / n2(e * L + l) > best) { best = P(e * L + l, j) / n2(e * L + l); at = l; }
      same = same && c.index(e, j) == at && c.response(e, j) == best;
    }
  EXPECT(same, "the level of the largest power / norm, and that value");
  const auto a = nxr::scalespace::selectLevel(P.leftCols(2), n2, L), b = nxr::scalespace::selectLevel(P.rightCols(4), n2, L);
  EXPECT((a.index - c.index.leftCols(2)).cwiseAbs().maxCoeff() == 0 && (b.index - c.index.rightCols(4)).cwiseAbs().maxCoeff() == 0, "columns are independent (a block of frames gives the same levels)");
  Eigen::MatrixXd W(E * L, 1);
  for (Eigen::Index i = 0; i < W.rows(); ++i) W(i, 0) = 3.0 * (0.5 * (1 << (i % 5)));        // white noise: the power is the norm times a constant
  Eigen::VectorXd n2b(E * L);
  for (Eigen::Index i = 0; i < n2b.size(); ++i) n2b(i) = 0.5 * (1 << (i % 5));       // powers of two: the division is exact
  const auto t = nxr::scalespace::selectLevel(W, n2b, L);
  EXPECT(t.index.cwiseAbs().maxCoeff() == 0 && (t.response.array() - 3.0).abs().maxCoeff() < 1e-14, "power proportional to the norm ties at every level: the lowest is taken, the response is the constant");
  Eigen::MatrixXd Q = Eigen::MatrixXd::Constant(E * L, 1, 1.0);
  for (Eigen::Index e = 0; e < E; ++e) Q(e * L + (e % L), 0) = 1000.0;      // a peak at level e % L
  const auto q = nxr::scalespace::selectLevel(Q, Eigen::VectorXd::Ones(E * L), L);
  bool peaks = true;
  for (Eigen::Index e = 0; e < E; ++e) peaks = peaks && q.index(e, 0) == e % L;
  EXPECT(peaks, "a planted peak at a level is the chosen level");
  auto throws = [&](const Eigen::MatrixXd& p, const Eigen::VectorXd& n, int l, const char* part) { try { nxr::scalespace::selectLevel(p, n, l); } catch (const std::invalid_argument& ex) { return std::string(ex.what()).find(part) != std::string::npos; } return false; };
  EXPECT(throws(P, n2, 3, "whole number"), "rows that are not a whole number of levels are refused");
  EXPECT(throws(P, n2.head(3), L, "norms"), "a wrong number of norms is refused");
  Eigen::VectorXd z = n2; z(7) = 0;
  EXPECT(throws(P, z, L, "row 7"), "a zero norm is refused");
  Eigen::MatrixXd neg = P; neg(2, 1) = -1;
  EXPECT(throws(neg, n2, L, "row 2"), "a negative power is refused");
  return failures ? 1 : 0;
}
