// nxr-io — a Brainstorm study's own covariance, imported as a stored statistic: `nxr-neuro import covariance`.
//
// Brainstorm keeps a study's noise covariance (`noisecov_*.mat`) and data covariance (`ndatacov_*.mat`) as the study's `NoiseCov` records (first and second).
// Each file holds `NoiseCov` [nch x nch] on the study's channel file (zero rows for the channels it does not cover), `nSamples`, `FourthMoment`, `Comment`, `History`.
//
//     <condition>_noise_covariance   Statistic measure covariance, method empirical (computed elsewhere), [M, M] on the channels with non-zero rows
//     <condition>_data_covariance    (the same, for the data covariance)
//     <name>_channels                Selection indices on the condition's channel manifold: the M channels, in the channel file's order
//
// The statistic is the shape `run inverse --noise-cov` and `run lcmv --noise-cov/--data-cov` read (its axes are the channels Selection, twice). Not imported: `FourthMoment`, the
// empty-room recording the covariance was computed from (it is named in the options, not imported).
#pragma once
#include <filesystem>
#include <optional>
#include <string>

#include "nxr/bst/subject.h"
#include "nxr/io/writer.h"

namespace nxr::bst {

struct CovarianceOptions {
  std::string kind = "noise";                  // noise | data
  std::string bst_root;                        // the protocol's folder (the recorded source is relative to it)
  std::optional<std::string> name;             // the stored statistic's name; default <condition>_<kind>_covariance
  bool replace = false;                        // a stored statistic of that name (and its channels) is moved out first; else it is refused
};
// `condition_dir`: the study's folder (data/<subject>/<condition>); `condition`: the acquisition its channels were imported under.
// ValueError when the study has no such covariance, its size disagrees with the channel file, or no channel has a non-zero row.
// Announces `{"stage": "covariance", ...}`; returns that line's body: name, id, kind, channels, n_channels, source_file, source_recording.
Json import_covariance(nxr::io::SubjectWriter& sub, const std::filesystem::path& condition_dir, const std::string& condition, const CovarianceOptions& opt, const Emit& progress);

}  // namespace nxr::bst
