#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/bst/projectors.h"
#include "nxr/io/json_text.h"

namespace nxr::pipelines::graph {

/** THE MINIMUM-NORM KERNEL'S INPUTS WITH THE RECORDING'S PROJECTORS, as each reference prepares them (the node `meeg.minimum_norm`):
 *  F [channels x 3N] the gain on all the channels; C [M x M] the noise covariance on the pick (channel indices into F's rows) with the pick's types; `projectors` the projector
 *  Operators on those channels; `covariance_projectors` the names of those the covariance's recording was read with.
 *  brainstorm: P (status 1 and 2, all channels) on the gain and both sides of C, then the average reference per family (process_inverse_2018).
 *  mne: P on the pick from every projector (the average reference one of them), C := P C P^T, the gain as it is, the kernel then multiplied by P on the right (make_inverse_operator). */
struct MinNormInputs {
  Eigen::MatrixXd G, C;
  std::optional<Eigen::MatrixXd> right;
  bool averaged = false;
  nxr::io::Json record;
};
MinNormInputs minimum_norm_inputs(const Eigen::MatrixXd& F, const Eigen::MatrixXd& C, const std::vector<std::int32_t>& pick, const std::vector<std::string>& types,
                                  const std::vector<nxr::bst::StoredProjector>& projectors, const std::vector<std::string>& covariance_projectors, const std::string& conventions,
                                  bool average_reference);

/** registers the M/EEG nodes (grouped names `meeg.*`) in nxr::graph::registry() when they are not there yet; called before the registry of steps is first read */
void register_meeg_nodes();
/** registers the quadrature node (`quadrature.source_pairs`); called by register_meeg_nodes */
void register_quadrature_nodes();

}  // namespace nxr::pipelines::graph
