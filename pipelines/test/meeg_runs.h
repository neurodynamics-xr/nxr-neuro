// Test helpers: the head model, the noise covariance and the minimum-norm kernel as the DEFINITIONS `headmodel`, `noisecov` and `inverse` run through the runner (what `nxr-neuro run NAME` does), with the
// handful of options the tests state and a result read back from what was stored (the gain and the kernel as the float32 arrays stored, the covariance as the stored statistic). The hand-wired
// headmodel_compute, noisecov_compute and inverse_compute are gone; the parameters are the definitions'.
#pragma once
#include "nxr/io/subject_api.h"
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/io.h"
#include "nxr/io/array_reader.h"
#include "nxr/stats/covariance.h"
#include "nxr/meeg/inverse.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/runner.h"

namespace nxrtest {

enum class MegModel { OverlappingSpheres, SingleSphere, None };

struct HeadModelOptions {
  std::string name = "headmodel", channels = "sensors", sources = "cortex";
  std::optional<std::string> meg_surface, eeg_surface;
  MegModel meg = MegModel::OverlappingSpheres;
  bool eeg = true, replace = false;
};
struct HeadModelResult {
  bool hit = false;
  Eigen::MatrixXd gain;
  nxr::io::Json info;
};
struct NoiseCovOptions {
  std::string recording, name;
  std::optional<double> start, stop;
  std::optional<std::string> chunks;
  std::optional<double> block_seconds;
  nxr::stats::CovarianceOptions covariance;
  bool replace = false;
};
struct NoiseCovResult {
  bool hit = false;
  std::vector<std::int32_t> channels;
  Eigen::MatrixXd covariance;
  nxr::io::Json info;
};
struct InverseComputeOptions {
  std::string forward, noise_cov, name;
  nxr::meeg::KernelOptions kernel;
  bool mne_conventions = false;
  bool replace = false;
};
struct InverseComputeResult {
  bool hit = false;
  int whitener_rank = 0;
  double lambda = 0;
  nxr::io::Json info;
};

inline nxr::pipelines::graph::RunOptions span_options(const std::optional<std::string>& chunks, const std::optional<double>& start, const std::optional<double>& stop, bool replace) {
  nxr::pipelines::graph::RunOptions o;
  o.chunks = chunks;
  o.start = start;
  o.stop = stop;
  o.replace = replace;
  return o;
}

inline Eigen::MatrixXd read_matrix(nxr::io::SubjectWriter& sub, const std::string& path) {
  nxr::io::ArrayReader r = nxr::io::ArrayReader::open(sub.store(), path);
  const auto shape = r.shape();
  const auto raw = r.read();
  Eigen::MatrixXd m(shape.at(0), shape.at(1));
  for (std::int64_t i = 0; i < shape[0]; ++i)
    for (std::int64_t j = 0; j < shape[1]; ++j) {
      const std::size_t k = static_cast<std::size_t>(i * shape[1] + j);
      if (r.dtype() == nxr::io::DType::Float32) { float v; std::memcpy(&v, raw.data() + 4 * k, 4); m(i, j) = v; }
      else { double v; std::memcpy(&v, raw.data() + 8 * k, 8); m(i, j) = v; }
    }
  return m;
}

inline HeadModelResult headmodel_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const HeadModelOptions& o) {
  nxr::io::Json set = nxr::io::Json{{"channels", o.channels}, {"sources", o.sources}, {"name", o.name}, {"eeg", o.eeg},
                                    {"meg", o.meg == MegModel::OverlappingSpheres ? "overlapping_spheres" : o.meg == MegModel::SingleSphere ? "single_sphere" : "none"}};
  if (o.meg_surface) set["meg_surface"] = *o.meg_surface;
  if (o.eeg_surface) set["eeg_surface"] = *o.eeg_surface;
  const auto runs = nxr::pipelines::graph::run(ds, sub, nxr::pipelines::graph::load("headmodel"), set, span_options(std::nullopt, std::nullopt, std::nullopt, o.replace));
  HeadModelResult r;
  r.hit = !runs.empty() && runs[0].value("hit", false);
  r.gain = read_matrix(sub, o.name);
  r.info = nxr::io::Json{{"meg", o.meg == MegModel::OverlappingSpheres ? "overlapping spheres" : o.meg == MegModel::SingleSphere ? "single sphere" : "none"}, {"eeg", o.eeg ? "three shells" : "none"},
                         {"replaced", o.replace && !r.hit}};
  return r;
}

inline NoiseCovResult noisecov_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const NoiseCovOptions& o) {
  nxr::io::Json set = nxr::io::Json{{"recording", o.recording}, {"remove_mean", o.covariance.removeMean},
                                    {"divisor", o.covariance.divisor == nxr::stats::CovarianceDivisor::Samples ? "samples" : o.covariance.divisor == nxr::stats::CovarianceDivisor::SamplesMinusOne ? "samples_minus_one" : "samples_minus_blocks"}};
  if (!o.name.empty()) set["name"] = o.name;
  if (o.block_seconds) set["block_seconds"] = *o.block_seconds;
  const auto runs = nxr::pipelines::graph::run(ds, sub, nxr::pipelines::graph::load("noisecov"), set, span_options(o.chunks, o.start, o.stop, o.replace));
  NoiseCovResult r;
  r.hit = !runs.empty() && runs[0].value("hit", false);
  const std::string name = o.name.empty() ? o.recording + "_noise_cov" : o.name;
  const nxr::io::StoredStatistic st = nxr::io::read_statistic(sub, name);
  const std::int64_t M = st.shape.at(0);
  r.covariance.resize(M, M);
  for (std::int64_t i = 0; i < M; ++i) for (std::int64_t j = 0; j < M; ++j) r.covariance(i, j) = st.values[static_cast<std::size_t>(i * M + j)];
  if (const auto ch = nxr::io::selection::named_row(sub, name + "_channels"))                  // a pick of the recording's channels, found by its name (io lays the node out)
    for (const std::int32_t c : nxr::io::selection::read_element_members(sub, nxr::io::text(*ch, "id")).member_of) r.channels.push_back(c);
  r.info = st.producer;
  r.info["samples"] = st.producer.contains("run") ? st.producer["run"].value("samples_read", std::int64_t(0)) : std::int64_t(0);   // the run's record: the samples read
  r.info["blocks"] = st.row.count("n_members") ? nxr::io::integer(st.row, "n_members") : 1;
  r.info["replaced"] = o.replace && !r.hit;
  return r;
}

inline InverseComputeResult inverse_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const InverseComputeOptions& o) {
  nxr::io::Json set = nxr::io::Json{{"conventions", o.mne_conventions ? "mne" : "brainstorm"}};
  if (!o.forward.empty()) set["forward"] = o.forward;
  if (!o.noise_cov.empty()) set["noise_cov"] = o.noise_cov;
  if (!o.name.empty()) set["name"] = o.name;
  if (!o.kernel.depth.enabled) set["depth_weighting"] = false;
  const auto runs = nxr::pipelines::graph::run(ds, sub, nxr::pipelines::graph::load("inverse"), set, span_options(std::nullopt, std::nullopt, std::nullopt, o.replace));
  InverseComputeResult r;
  r.hit = !runs.empty() && runs[0].value("hit", false);
  if (const auto k = sub.by_name("operator", o.name)) {
    const nxr::io::Json pj = nxr::io::Json::parse(nxr::io::opt_text(*k, "producer_json").value_or("{}"), nullptr, false);
    r.whitener_rank = pj.is_object() ? pj.value("whitener_rank", 0) : 0;
    r.lambda = pj.is_object() ? pj.value("lambda", 0.0) : 0.0;
    r.info = pj;
    r.info["channels"] = static_cast<int>(nxr::io::integer(*k, "n_cols"));
  }
  return r;
}

}  // namespace nxrtest
