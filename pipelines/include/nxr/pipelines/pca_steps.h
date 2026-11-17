#pragma once

namespace nxr::pipelines::graph {

/** registers the principal-components node (`statistic.pca`) in nxr::graph::registry() when it is not there yet; called with the other native nodes */
void register_pca_nodes();

}  // namespace nxr::pipelines::graph
