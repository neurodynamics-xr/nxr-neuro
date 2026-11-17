// nxr-pipelines — THE PCA STATISTIC (spec docs/specs/2026-11-16-meg-fingerprinting.md, 4.4): the principal components ACROSS the members of a table (members x features), stored as a
// statistic of method `pca`. A generalisation of the `eigendecomposition` statistic (one subject's second-moment matrix) to the usual case of a table of measurements, e.g. one row per
// member of a partition or per person. The maths is `nxr::pca` (compute/include/nxr/pca.h, a thin SVD of the centred table: never the features x features covariance); this file only
// prepares the table from what is stored (type-checked, refused with its reason), calls it and stores the result.
//
// THE TABLE comes from one of two stored forms, named by the caller:
//   a STATISTIC (`of`, with `members`): real values; the axis named `members` (by name, or its position) holds the members (the observations), the other axes together the features, in order
//   a MEASUREMENT (`of`, with `over` and `measure`, as `statistic compute`): the members are the partition's members, the features the measured rows (a channel, a vertex)
//
// STORED, as statistics of method `pca` (every component up to the table's rank, so any k is available; reconstruction at k is derived at read, never stored):
//   `<name>`             measure principal_components   [component x feature axes...]  unit-length loadings, the sign convention of nxr::pca (largest loading positive)
//   `<name>_scores`      measure principal_scores       [member x component]           the table's rows on the components, U S
//   `<name>_variance`    measure explained_variance     [component x 3]                singular value, explained variance (S^2 / (n - 1)), its ratio over the total
//   `<name>_centring`    measure pca_centring           [2 x feature axes...]          the mean removed from each feature, and the divisor after it (1 unless unit-variance)
// The precision of the components, scores and centring follows the input's stored precision (float32 in, float32 out); the variance table is float64.
#pragma once
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/io.h"

namespace nxr::pipelines {

struct PcaComputeOptions {
  std::string of;                            // what is decomposed: a stored statistic by name; or what was measured (a Field, or a band of a map statistic, as `statistic compute`)
  std::optional<std::string> members;        // statistic form: the axis whose entries are the members (observations): its name (when unique among the axes) or its position, from 0
  std::optional<std::string> over;           // measurement form: the partition (a stored Selection) whose members were measured
  std::string measure;                       // measurement form: the stored measure (a derived one is refused)
  bool own = false;                          // measurement form: the measure is the members' own (area, n_elements); `of` is then its parent
  std::string scale = "none";                // none: the features keep their units (the covariance's components) · unit-variance: each divided by its standard deviation (the correlation's)
  std::optional<std::string> name;           // default `<of>_pca`
  bool replace = false;
};
struct PcaComputeResult { nxr::io::Json info; bool hit = false; };
// Memory held: the table (members x features doubles) and its thin SVD (the same again, plus members^2). Refused with the reason: no or both forms named, a complex or non-scalar statistic, a
// non-finite value, fewer than two members, a table with no variance, a constant feature under unit-variance.
PcaComputeResult pca_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const PcaComputeOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

// A stored pca, read back as matrices (its feature axes flattened in order; `feature_shape` keeps their sizes). Reconstruction at k is
// nxr::pca::reconstruct(scores, components, mean, scale, k), the same function the computation is tested with.
struct StoredPca {
  Eigen::MatrixXd components, scores;        // [k x p], [n x k]
  Eigen::VectorXd singular_values, explained_variance, explained_variance_ratio;   // [k]
  Eigen::VectorXd mean, scale;               // [p]
  std::vector<std::int64_t> feature_shape;
  nxr::io::Json options;                     // scale, the input, the members' axis
};
StoredPca pca_read(nxr::io::SubjectWriter& sub, const std::string& name);

}  // namespace nxr::pipelines
