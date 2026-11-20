#pragma once

namespace nxr::pipelines::graph {

/** registers the identification nodes of brain fingerprinting (`stats.featureTable`, `stats.identification`, `stats.identificationCurve`) in nxr::graph::registry() when they are not
 *  there yet; called with the other native nodes */
void register_identification_nodes();

}  // namespace nxr::pipelines::graph
