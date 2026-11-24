// nxr-pipelines — WHAT THE RUNNER BINDS FOR A NODE (spec docs/specs/2026-11-17-compute-and-run.md, section 3.2 and PR C4: nodes lose the database).
//
// A node's `plan`, `prepare` and `resolve` take arrays, values and parameters; they never reach a database, a subject, a dataset or a file. What a node needs beside its inputs
// (the type of a manifold, the rows of a selection, a surface, a stored statistic, a kernel's matrix, the spatial band) is FOUND and READ by the runner, through io's records and
// store, and handed over as plain data through this class. `Bound` is the runner's side of the line: its methods are the only place a node's need is turned into a read; the
// node code (registry.cpp and the `*_steps.cpp`) includes no io header and holds no io handle (the layering guard `layering_nodes_without_io`).
//
// Each method is a constant the runner resolves for the node: the same arguments give the same data for the whole run. A refusal (a missing stored statistic, a surface of another
// subject) is the runner's, raised with the same message the node gave before.
// C5 turns these requests into declared input ports typed by `PortType` (the runner binds a found node as a port); until then they are named here, one method each, so the
// requests a node makes are listed in one place.
#pragma once

#include "nxr/io/measures.h"
#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "nxr/io.h"
#include "nxr/io/statistic.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/group_template.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/scale_inputs.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/pipelines/task.h"

namespace nxr::pipelines::graph {

/** raised when a run's node asks for a spatial band whose eigenmodes are not stored and the plan says to compute them: the RUNNER makes them first, as their own run, and starts again
 *  (docs/specs/2026-11-17-compute-and-run.md, question 6). A node never stores anything. */
struct MissingEigenmodes : std::runtime_error {
  std::string surface;                       // the surface whose package is missing
  SpatialBandPlan plan;                      // what will be computed (the package, its size, why)
  MissingEigenmodes(std::string s, SpatialBandPlan p) : std::runtime_error("the eigenmodes of " + s + " are not stored"), surface(std::move(s)), plan(std::move(p)) {}
};

/** a stored operator read back: its matrix [rows x cols] (float64) and the provenance it was stored with */
struct StoredOperator {
  Eigen::MatrixXd matrix;
  nxr::io::Json provenance;
};
/** an operator a node BUILT and offers to be stored: found next time by `key` (the operator's own provenance: the rows of its inputs, the request, the band's resolved choice), never rebuilt and
 *  never held in a process-wide cache. The runner stores each offered operator through io with the key and the provenance, in the run's one unit. `from_manifold_id`/`to_manifold_id`: the operator's ends. */
struct MadeOperator {
  nxr::io::Json key;
  std::shared_ptr<const Eigen::MatrixXd> matrix;
  nxr::io::Json provenance;
  std::string name, kind = "fused kernel", from_manifold_id, to_manifold_id, description;
};

class Bound {
 public:
  /** a binding that knows the subject's records only (a plan: before any run, nothing stored is read) */
  explicit Bound(const nxr::pipelines::task::Subject& subject) : subject_(subject) {}
  /** a binding for a subject's run: its records, its stored arrays, and the dataset a missing eigenmode package is made in (progress said through `say`) */
  Bound(nxr::io::SubjectWriter& sub, nxr::io::DatasetWriter* ds, std::function<void(const nxr::io::Json&)> say = {});

  // ── records (plan and run) ──
  std::optional<nxr::io::Row> manifold(const std::string& id) const;            // a manifold's row: name, type, n_vertices, subject
  std::optional<nxr::io::Row> selection(const std::string& id) const;           // a selection's row: params_json, array_path, member_of_id
  std::optional<nxr::io::Row> field(const std::string& id) const;               // a field's row
  std::string manifold_type(const std::string& id) const;                       // "" when unknown
  std::optional<std::string> glued_surface_id(const std::string& manifold_id) const;   // the id of `<name>_glued`, the surface of the same subject (one cortex, the bridge filled)
  std::vector<std::pair<double, double>> spans(const std::string& selection_id) const;   // a spans selection's members in order: start, stop (seconds)
  std::vector<nxr::io::Row> level_members(const std::string& ladder_id) const;  // a ladder's complete members in order: id, name, member_ordinal, params_json
  // ── stored operators, charts and lines (consolidation spec 3, 7; 25) ──
  /** a stored operator found by its provenance key (the fused operator a node would otherwise cache statically): the operator's row, its array read through `array_float`; nothing when it is not stored.
   *  The key is the one an operator output's producer record carries (`producer.cache`, stamped by the runner like every stored node's): a node that needs a fused operator asks for it here, and a pipeline that makes it declares it as an `operator` output. */
  std::optional<StoredOperator> stored_operator(const nxr::io::Json& key) const;
  /** a node offers an operator it built; the runner stores it. Offered twice under one key is kept once. */
  void offer_operator(MadeOperator op) const;
  /** what nodes offered in this run, in order (the runner drains it into the unit) */
  const std::vector<MadeOperator>& offered_operators() const { return *offered_; }
  /** the charts (coordinate maps) of a manifold's space, as io records them: an array of {name, unit, transition, params, reference}; the node hands them to compute's `quantity` as data. Empty when none. */
  nxr::io::Json charts(const std::string& manifold_id) const;
  /** the dataset's Frequency line (Hz, charts linear and log10): the manifold of the default subject every band set is spans on; nothing when the dataset has none (a plan without a dataset: the subject's own) */
  std::optional<nxr::io::Row> frequency_line() const;
  /** the dataset's Eigenvalue line (1/m2, charts eigenvalue and wavelength): the default subject's, on which spatial band sets are spans; nothing when the dataset has none */
  std::optional<nxr::io::Row> eigenvalue_line() const;
  /** the glue map of a manifold that is a derived glued surface (made by `glue`): glued vertex → original vertex (-1: the bridge); nothing for any other manifold. Results on it are restricted back to the original */
  std::optional<GlueMap> glue_of(const std::string& manifold_id) const;
  /** a labelling (a set per element: an atlas, a hemisphere labelling) as stored: each element's label and the dictionary's entries */
  StoredLabelling labelling(const std::string& selection_id) const;
  // ── the subject's stored constants (a run) ──
  std::optional<nxr::io::Row> manifold_named(const std::string& name) const;    // the run's subject's manifold by name
  SurfaceInput surface(const std::optional<std::string>& name = std::nullopt, bool as_stored = false) const;
  SurfaceLevel surface_level(const SurfaceInput& surface, int level) const;
  StoredFrames stored_frames(const std::string& surface_name, std::int64_t n_vertices) const;
  std::string subject_kind() const;                                              // the run's subject's kind: subject, template or group
  std::string coordinate_unit(const std::string& manifold_id) const;             // the length unit of a manifold's coordinate system ("m", "mm"; empty when it records none)
  std::string coordinate_system_key(const std::string& manifold_id) const;        // the NAME of a manifold's coordinate system ("scs"; empty when it records none): a surface made from it lies in the same one
  std::vector<std::uint8_t> medial_wall(const SurfaceInput& surface) const;      // a surface's medial wall [nV] (1 on it) from its stored Desikan-Killiany atlas; refused by name without one
  GroupTemplate group_template(const std::string& surface_name) const;           // the default subject's group ladder, sphere and surface, with this subject's registered sphere (group.h)
  internal::GluedDomain glued_domain(const SurfaceInput& surface) const;
  StoredCrossSpectrum cross_spectrum(const std::string& name) const;
  CrossSpectrumWindows cross_spectrum_windows(const std::string& name) const;       // an analytic series on time tiles, by its band's name (as stored)
  CrossSpectrumWindows cross_spectrum_windows(const std::string& name, const std::vector<std::int32_t>& channels) const;   // the same, on these recording channels in this order (refused for a channel it does not cover)
  nxr::io::StoredStatistic statistic(const std::string& name) const;
  /** a surface's frames as `frames` makes them (pipelines/frames.h make_frames): read through this binding (and a template subject of the dataset) */
  FramesMade make_frames(const FramesComputeOptions& opt) const;
  /** the values of a measure over a selection's members (io's measurement_values): of a field, of a map statistic's slice (by its name, a band of a map), or the members' own */
  nxr::io::MeasurementValues measurement_values(const std::string& selection_id, const std::optional<std::string>& of_field_id, const std::optional<std::string>& of_slice,
                                                const std::string& measure) const;
  /** the values of a measure over a selection's members OF A STORED STATISTIC (`measure` of a statistic, schema 54): [rows x members], rows the statistic's axes after the first */
  nxr::io::MeasurementValues measurement_values_of_statistic(const std::string& selection_id, const std::string& statistic_id, const std::string& measure) const;
  std::vector<std::int32_t> array_int32(const std::string& path) const;                         // a stored array by its store path (a kernel's channel list)
  std::vector<float> array_float(const std::string& path, std::vector<std::int64_t>* shape) const;   // (a kernel's matrix)
  std::vector<double> array_double(const std::string& path, std::vector<std::int64_t>* shape) const;   // (a covariance: precision follows the stored array)
  /** what scale selection reads: every (space level, time level) pair's stored joint Selection, labelling, cell amplitude and baseline, found and checked (scale_inputs.h) */
  nxr::pipelines::ScaleInputs scale_inputs(const std::string& recording_id, const std::string& recording_name, const std::vector<int>& space_levels, const std::vector<int>& time_levels,
                                           const std::string& source, const std::optional<std::string>& surface) const;
  // ── the spatial band: planned (its key is part of the run's cache key) and made (a missing eigenmode package is computed, said in `say`, and stored by the runner) ──
  SpatialBandPlan plan_spatial_band(const SurfaceInput& surface, const SurfaceLevel& level, const SpatialBandOptions& opt) const;
  SpatialBand make_spatial_band(const SurfaceInput& surface, const SurfaceLevel& level, const SpatialBandOptions& opt) const;

 private:
  nxr::io::SubjectWriter& writer() const;
  std::optional<nxr::io::Row> default_subject_manifold(const std::string& name) const;
  nxr::pipelines::task::Subject subject_;                       // the records, through io's task API (no SQL, no handle)
  nxr::io::SubjectWriter* sub_ = nullptr;
  nxr::io::DatasetWriter* ds_ = nullptr;
  std::function<void(const nxr::io::Json&)> say_;
  std::shared_ptr<std::vector<MadeOperator>> offered_ = std::make_shared<std::vector<MadeOperator>>();   // shared by the copies of this binding, so what a node offers reaches the runner
};

}  // namespace nxr::pipelines::graph
