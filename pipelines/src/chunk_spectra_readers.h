// nxr-pipelines (internal) — what the readers of the chunk spectra share (sensor_flow_steps.cpp defines them; the sensor-flow and the source-flow nodes read them):
// the chunk spectra [chunks x bins x coordinates] as signal.chunk_spectra stores them, a kernel set's members and their gains on a line, the frequency weighting's members,
// and one frequency member's weighted columns Y (the cross-spectrum it weights is Y Yᴴ).
#pragma once

#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/signal/intrinsic_forms.h"

namespace nxr::pipelines::graph::spectra_readers {

struct Spectra {
  std::int64_t chunks = 0, bins = 0, rows = 0, first = 0, transform = 0, overlap = 0;
  double sfreq = 0;
  std::vector<std::int64_t> modes, owns;
  bool unit_mass = true;
  const nxr::graph::Array* array = nullptr;
  std::complex<double> at(std::int64_t t, std::int64_t k, std::int64_t r) const {
    const std::size_t i = static_cast<std::size_t>(((t * bins + k) * rows + r) * 2);
    return std::complex<double>(array->at(i), array->at(i + 1));
  }
  double frequency(std::int64_t k) const { return static_cast<double>(first + k) * sfreq / static_cast<double>(transform); }
};

Spectra spectra_of(const nxr::graph::Array& spectra, const char* who);
/** the field coordinates' synthesis [S x K] (the constant at unit star0 norm; each mode at unit star0 norm under unit_mass) and their eigenvalues */
std::pair<Eigen::MatrixXd, Eigen::VectorXd> synthesis_of(const Spectra& s, const nxr::signal::Modes& m, const Eigen::VectorXd& mass, const char* who);
std::vector<nxr::graph::Json> members_of(const nxr::graph::Array& selection, const char* who, const char* port);
/** a kernel set's gains [members x points] on points of its line's coordinate */
Eigen::MatrixXd gains_of(const std::vector<nxr::graph::Json>& members, const Eigen::VectorXd& x, const char* who);
/** the frequency weighting's members [members x bins] (parameters frequency_weighting, frequency_normalisation, support_floor; the port `frequency`) */
Eigen::MatrixXd frequency_gains(const Spectra& s, const nxr::graph::Arrays& c, const nxr::graph::Json& p, const char* who, std::vector<double>* standardDeviations = nullptr);
Eigen::MatrixXcd weighted_columns(const Spectra& s, const Eigen::RowVectorXd& gain, double floorFraction = 1e-12);

}  // namespace nxr::pipelines::graph::spectra_readers
