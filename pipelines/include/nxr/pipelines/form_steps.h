// nxr-pipelines — the nodes of the forms on a surface's intrinsic complex (signal.phase_split) registered in nxr::graph::registry(); idempotent. See src/form_steps.cpp and
// docs/specs/2026-11-19-intrinsic-form-nodes.md.
#pragma once
#include <memory>
#include <string>

#include "nxr/graph/types.h"
#include "nxr/signal/intrinsic_forms.h"

namespace nxr::pipelines::graph {

/** registers the intrinsic-form nodes when they are not there yet; called by `register_partition_nodes` with the other grouped nodes */
void register_form_nodes();

/** A HODGE-DIRAC PACKAGE AS BOUND for a node (its eigenvectors operator as the runner hands it over, with its 2-form parts and its complex's options: `reads_package_forms`): the intrinsic
 *  complex rebuilt from its surface, the modes' full forms (the stored 1-form parts when the package has them), and the boundary condition its surface implies (none: closed;
 *  neumann_potential_dirichlet_stream: open). Refuses (naming `who`) a value that is not such a package. */
struct BoundPackage {
  std::shared_ptr<const nxr::signal::IntrinsicComplex> complex;
  std::shared_ptr<const nxr::signal::Modes> modes;
  std::string boundary;
};
BoundPackage bind_hodge_package(const nxr::graph::Array& eigenvectors, const char* who);

}  // namespace nxr::pipelines::graph
