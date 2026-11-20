#pragma once

namespace nxr::pipelines::graph {

/** registers the per-subject feature nodes of MEG fingerprinting (`signal.hilbert`, `stats.cov`, `stats.correlation`, `stats.spectralFeature`, `stats.connectomeFeature`) in
 *  nxr::graph::registry() when they are not there yet; called with the other native nodes */
void register_fingerprint_feature_nodes();

}  // namespace nxr::pipelines::graph
