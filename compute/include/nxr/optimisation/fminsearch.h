// nxr::meeg — the Nelder-Mead simplex minimiser, written from Lagarias, Reeds, Wright & Wright, "Convergence properties of the Nelder-Mead simplex method in
// low dimensions", SIAM J. Optim. 9:112-147, 1998 (the algorithm of their section 2, with the ordered simplex and the strict-inequality tests).
// Pure and deterministic: a function and a start in, the minimiser out. Depends on Eigen only.
#pragma once
#include <Eigen/Dense>
#include <functional>

namespace nxr::optimisation {

struct NelderMeadOptions {
  double reflection = 1;           // rho of the paper
  double expansion = 2;            // chi
  double contraction = 0.5;        // gamma
  double shrink = 0.5;             // sigma
  double relativeStep = 0.05;      // the start simplex is x0 and x0 + step_i e_i: step_i = relativeStep x x0_i ...
  double zeroStep = 0.00025;       // ... or zeroStep where x0_i = 0
  double xTolerance = 1e-10;       // stop when the simplex's vertices lie within this (max-norm) of the best one ...
  double fTolerance = 1e-14;       // ... and their values within this of the best value
  int maxEvaluations = 20000;
};
struct MinimiseResult {
  Eigen::VectorXd x;               // the best vertex
  double value = 0;                // f(x)
  int evaluations = 0;
  bool converged = false;          // false: maxEvaluations was reached first
};

// MATLAB's fminsearch as documented (Lagarias et al. 1998; start simplex 5% of each non-zero coordinate, 0.00025 for zeros; TolX = TolFun = 1e-4; MaxFunEvals = MaxIter = 200 n).
NelderMeadOptions fminsearchOptions(int n);

// Minimise f from x0. std::invalid_argument for an empty x0 or a non-positive maxEvaluations.
MinimiseResult fminsearch(const std::function<double(const Eigen::VectorXd&)>& f, const Eigen::VectorXd& x0, const NelderMeadOptions& opt = {});

}  // namespace nxr::optimisation
