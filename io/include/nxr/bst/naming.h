// nxr-io — the names of nodes a Brainstorm import writes (nxr.convert.naming), shared by the sessions and the source port: one definition of each.
#pragma once
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "nxr/io/writer.h"

namespace nxr::bst {

std::string recording_node_name(const std::filesystem::path& data_file);   // data_0raw_<x>.mat / data_<x>.mat -> <x>
std::string channel_node_name(const std::filesystem::path& channel_file);  // the stem
std::string kernel_node_name(const std::filesystem::path& results_file);   // results_<x>.mat -> <x>
std::string headmodel_node_name(const std::filesystem::path& headmodel_file);   // the stem
// The short, human part of a condition that tells it from its siblings: `sub-X_ses-02_task-rest_run-02_meg` -> `ses-02_task-rest_run-02`.
std::string acquisition_label(const std::string& condition);
std::string sha1_hex(const std::string& bytes);
// sha1 over the files' bytes and `extra` - what makes two sessions' nodes the SAME node
std::string source_fingerprint(const std::vector<std::filesystem::path>& files, const std::string& extra = "");
// The name a session-scoped node gets in `sub`, and whether it already EXISTS as-is (its row holds byte-identical source: share it, write nothing).
std::pair<std::string, bool> claim_node_name(nxr::io::SubjectWriter& sub, const std::string& folder, const std::string& stem, const std::string& acquisition,
                                             const std::string& fingerprint);

}  // namespace nxr::bst
