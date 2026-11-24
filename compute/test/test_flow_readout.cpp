// nxr::signal flow_readout on synthetic data (an icosphere's complex and its Hodge-Dirac modes, random 1-forms):
//   flowSplit: the coefficients on the stored edge rows, the potential and the stream equal projectOneForm / exactPotential / coexactStream column by column;
//     with the remainder split, the remainder's parts equal OneFormRemainderSplit on w minus its synthesis, and the energies are the ⋆1 norms;
//   splitPhases: a batch of 1-forms equals splitPhase on each;
//   OneFormRemainderSplit: an exact 1-form d0 phi gives back phi and no stream; a coexact 1-form gives back its stream and no potential.
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <random>

#include "mesh_helpers.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/eigen.h"
#include "nxr/signal/flow_readout.h"
#include "nxr/signal/intrinsic_forms.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_flow_readout\n";
  namespace ns = nxr::signal;
  std::mt19937 rng(11);
  std::normal_distribution<double> nd(0, 1);
  auto randn = [&](Eigen::Index r, Eigen::Index c) { Eigen::MatrixXd m(r, c); for (Eigen::Index i = 0; i < r; ++i) for (Eigen::Index j = 0; j < c; ++j) m(i, j) = nd(rng); return m; };
  auto relative = [](const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) { return (a - b).cwiseAbs().maxCoeff() / std::max(1e-300, b.cwiseAbs().maxCoeff()); };

  // ── an icosphere, its complex and its modes ──
  std::vector<double> vertexPositions; std::vector<int32_t> faces;
  icosphere(2, vertexPositions, faces, 1.0);
  const int vertexCount = static_cast<int>(vertexPositions.size() / 3), faceCount = static_cast<int>(faces.size() / 3);
  const auto surface = nxr::geometry::surfaceArrays(vertexPositions.data(), vertexCount, faces.data(), faceCount);
  const ns::IntrinsicComplex complex = ns::intrinsicComplex(surface);
  const auto hodgeDirac = nxr::operators::hodgedirac::assemble(surface);
  nxr::operators::HodgeSolveOptions solveOptions; solveOptions.limit = 60;
  const ns::Modes modes = ns::modesOf(nxr::operators::hodgeDiracModes(hodgeDirac, 120, false, solveOptions), hodgeDirac.nV, hodgeDirac.nE, hodgeDirac.nF);
  const Eigen::SparseMatrix<double> sharpWeights = ns::sharpOperator(complex);

  // ── flowSplit ──
  {
    const Eigen::MatrixXd edgeCurrent = randn(complex.nE, 3);
    const ns::FlowSplit plain = ns::flowSplit(complex, modes, edgeCurrent, nullptr);
    const Eigen::MatrixXd coefficients = ns::projectOneForm(complex, modes, edgeCurrent);
    EXPECT(relative(plain.coefficients, coefficients) < 1e-12 && relative(plain.potential, ns::exactPotential(modes, coefficients)) < 1e-12
               && relative(plain.stream, ns::coexactStream(complex, modes, coefficients)) < 1e-12,
           "flowSplit: coefficients on the stored edge rows, potential sum (c / mu) exact, stream likewise");
    EXPECT(plain.remainderPotential.size() == 0 && plain.remainderStream.size() == 0 && plain.remainderEnergy.size() == 0, "flowSplit: no remainder parts unless a remainder split is given");
    const Eigen::VectorXd energy = edgeCurrent.cwiseAbs2().transpose() * complex.star1;
    EXPECT(relative(plain.flowEnergy, energy) < 1e-12, "flowSplit: the flow's ⋆1 energy per column");
    const ns::OneFormRemainderSplit remainderSplit(complex);
    const ns::FlowSplit withRemainder = ns::flowSplit(complex, modes, edgeCurrent, &remainderSplit);
    const Eigen::MatrixXd remainder = edgeCurrent - ns::synthesizeOneForm(complex, modes, coefficients);
    double worst = 0;
    for (Eigen::Index column = 0; column < edgeCurrent.cols(); ++column) {
      const ns::OneFormRemainderSplit::Parts parts = remainderSplit.split(remainder.col(column));
      worst = std::max({worst, relative(withRemainder.remainderPotential.col(column), parts.potential), relative(withRemainder.remainderStream.col(column), parts.stream)});
    }
    const Eigen::VectorXd remainderEnergy = remainder.cwiseAbs2().transpose() * complex.star1;
    EXPECT(worst < 1e-12 && relative(withRemainder.remainderEnergy, remainderEnergy) < 1e-12, "flowSplit: the remainder beyond the kept modes split by the sparse solves, its energy");
  }

  // ── splitPhases: a batch equals splitPhase on each ──
  {
    const Eigen::MatrixXd oneForms = randn(complex.nE, 4);
    const ns::PhaseSplits batch = ns::splitPhases(complex, modes, oneForms, sharpWeights);
    double worst = 0;
    for (Eigen::Index column = 0; column < oneForms.cols(); ++column) {
      const ns::PhaseSplit one = ns::splitPhase(complex, modes, oneForms.col(column));
      worst = std::max({worst, relative(batch.coefficients.col(column), one.coef), relative(batch.exact.col(column), one.exact), relative(batch.coexact.col(column), one.coexact),
                        relative(batch.potential.col(column), one.potential), relative(batch.stream.col(column), one.stream), relative(batch.propagation.col(column), one.propagation.col(0))});
    }
    std::printf("  splitPhases vs splitPhase: %.2e\n", worst);
    EXPECT(worst < 1e-12, "splitPhases: a batch of 1-forms equals splitPhase on each");
  }

  // ── OneFormRemainderSplit ──
  {
    const ns::OneFormRemainderSplit splitter(complex);
    Eigen::VectorXd potential = randn(vertexCount, 1);
    potential = potential.array() - potential.dot(complex.star0) / complex.star0.sum();
    const ns::OneFormRemainderSplit::Parts exactParts = splitter.split(complex.d0 * potential);
    EXPECT(relative(exactParts.potential, potential) < 1e-6 && exactParts.stream.cwiseAbs().maxCoeff() < 1e-6 * potential.cwiseAbs().maxCoeff(), "an exact 1-form: its potential back, no stream");
    Eigen::VectorXd starredStream = randn(complex.nF, 1);
    starredStream = starredStream.array() - starredStream.mean();                // orthogonal to the face Laplacian's constant
    const Eigen::VectorXd twoFormPotential = starredStream.cwiseQuotient(complex.star2);
    const Eigen::VectorXd coexact = complex.star1.cwiseInverse().cwiseProduct(complex.d1.transpose() * starredStream);
    const ns::OneFormRemainderSplit::Parts coexactParts = splitter.split(coexact);
    Eigen::VectorXd streamWant = ns::vertexDensity(complex, twoFormPotential);
    streamWant = streamWant.array() - streamWant.dot(complex.star0) / complex.star0.sum();
    EXPECT(relative(coexactParts.stream, streamWant) < 1e-6 && coexactParts.potential.cwiseAbs().maxCoeff() < 1e-6 * coexact.cwiseAbs().maxCoeff(), "a coexact 1-form: its stream back, no potential");
  }

  std::cout << (failures ? "FAILED " : "OK ") << failures << "\n";
  return failures ? 1 : 0;
}
