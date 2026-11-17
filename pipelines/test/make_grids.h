// Test helpers: the grids a test's fixture needs, made by the shipped MAKER DEFINITIONS through the runner (what `nxr-neuro run select_dyadic_surface` / `select_time_grid` / `select_bands` do). The hand-wired
// tiles_compute, frame_tiling_compute, constant_q_write and pyramid_spatial_bands are gone: a fixture runs the maker by name, its parameters quantities (a band in Hz, a tile in mm, a count of parts).
#pragma once
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/io/array_reader.h"
#include "nxr/io/subject_reads.h"
#include "nxr/io/writer.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/run_shipped.h"

namespace nxrtest {

/** the dyadic grid of the subject's primary surface (or `surface`), rule subject; `finest` a quantity ("parts 16", "20 mm"), empty: the finest every piece reaches */
inline std::vector<nxr::io::Json> make_tiles(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::string& finest = "", std::optional<std::string> surface = std::nullopt, bool replace = false) {
  const std::string name = surface ? *surface : nxr::pipelines::read_surface(sub, std::nullopt).name;
  nxr::io::Json set = nxr::io::Json{{"surface", name}, {"rule", "subject"}};
  if (!finest.empty()) set["finest"] = finest;
  return nxr::pipelines::run_shipped(ds, sub, "select_dyadic_surface", set, replace);
}

/** the line's time grid of every recording of the subject with the constant-Q ladder (rule constant_q: the Studio's ConstantQLevelInfo per level) */
inline std::vector<nxr::io::Json> make_time_grids(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::string& rule = "constant_q", const nxr::io::Json& more = nxr::io::Json::object()) {
  nxr::io::Json set = more;
  set["rule"] = rule;
  return nxr::pipelines::run_shipped(ds, sub, "select_time_grid", set, false);
}

/** the frames of `recording` for a band (a quantity, "8-12 Hz"), the frame density the definition's default (36 frames a cycle of the band's centre) unless `frames_per_cycle` is given */
inline std::vector<nxr::io::Json> make_frames(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::string& recording, const std::string& band, std::optional<double> frames_per_cycle = std::nullopt) {
  nxr::io::Json set = nxr::io::Json{{"recording", recording}, {"rule", "frames"}, {"band", band}};
  if (frames_per_cycle) set["frames_per_cycle"] = *frames_per_cycle;
  return nxr::pipelines::run_shipped(ds, sub, "select_time_grid", set, false);
}

/** the spatial band set of the subject's surface (rule levels, on the Eigenvalue line) */
inline std::vector<nxr::io::Json> make_spatial_bands(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::string& surface) {
  return nxr::pipelines::run_shipped(ds, sub, "select_bands", nxr::io::Json{{"rule", "levels"}, {"surface", surface}}, false);
}

/** a stored map by name as [n x components] (a Field of n values per component, as the tests store them); nullopt when absent or not float/int */
inline std::optional<Eigen::MatrixXd> stored_map(nxr::io::SubjectWriter& sub, const std::string& field_name, std::int64_t n) {
  using namespace nxr::io;
  const auto f = find_field(sub.db(), sub.id(), FieldQuery{field_name, std::nullopt, std::nullopt, true});
  if (!f) return std::nullopt;
  ArrayReader r = ArrayReader::open(sub.store(), text(*f, "path"));
  const std::vector<std::uint8_t> raw = r.read();
  const std::size_t total = raw.size() / r.itemsize();
  if (n <= 0 || total % static_cast<std::size_t>(n) != 0) return std::nullopt;
  Eigen::MatrixXd m(n, static_cast<Eigen::Index>(total / static_cast<std::size_t>(n)));
  for (std::size_t i = 0; i < total; ++i) {
    double v;
    if (r.dtype() == DType::Float32) { float x; std::memcpy(&x, raw.data() + i * 4, 4); v = x; }
    else if (r.dtype() == DType::Float64) std::memcpy(&v, raw.data() + i * 8, 8);
    else if (r.dtype() == DType::Int32) { std::int32_t x; std::memcpy(&x, raw.data() + i * 4, 4); v = x; }
    else return std::nullopt;
    m(static_cast<Eigen::Index>(i / static_cast<std::size_t>(m.cols())), static_cast<Eigen::Index>(i % static_cast<std::size_t>(m.cols()))) = v;
  }
  return m;
}

}  // namespace nxrtest
