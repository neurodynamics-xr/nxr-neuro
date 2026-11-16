// nxr-io — static MAPS from Brainstorm into a subject: `nxr.convert.results_map` (register D124, D158, D161; `nxr convert map`).
//
// A SOURCE MAP is a full `results_*.mat` that is not a kernel (`ImageGridAmp` [nVertices x nTime] on the surface its `SurfaceFile` names, no `ImagingKernel`; a
// volume projected onto a surface: a PET SUVR, an fMRI map): a Field on the cortex, `source/<stem>`, one value per vertex. A map over time (`Time` of more than two
// points, a BOLD series) is a Field on the surface x a time Line, stored as a recording is (see MapHooks). A REGION MAP is a `matrix_*.mat` of scout extraction
// (`Value`, `Description` the scout labels, `Atlas`, `SurfaceFile`): MEASUREMENTS on that atlas's Selection (`selection_measurement`, `code` the region's).
//
//   Json info = bst::export_map(sub, "results_surface_PET.mat", "protocol", "rest");
#pragma once
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "nxr/bst/mat.h"
#include "nxr/bst/subject.h"
#include "nxr/io/writer.h"

namespace nxr::bst {

// SESSION HOOKS. Two writers of the session port (`nxr.convert.timeseries`: `_recording` with `_write_blocks`, and `_time_line`) are needed by maps over time and by
// confounds matrices. Until the session port provides them, such a map is refused with NotImplementedError saying so; static and region maps do not need them.
struct SeriesRequest {                    // a [rows x T] series on rows x a time Line of its own, written by the recording writer (chunk grid, envelope)
  std::string stem;
  std::vector<float> x;                   // [rows x samples], row after row
  std::int64_t rows = 0, samples = 0;
  double t0 = 0, step = 0;                // the regular sampling of the time points (s)
  std::string rows_id;                    // the surface the rows are the vertices of
  std::optional<std::string> acquisition, unit;
  std::string comment, source, description;
  Json producer;
};
struct TimeLineRequest { std::string path; std::optional<std::string> acquisition_id; std::int64_t samples = 0; double sfreq = 0, t0 = 0; };
struct MapHooks {
  std::function<std::string(nxr::io::SubjectWriter&, const SeriesRequest&)> series;       // `_series`: returns the field's id
  std::function<std::string(nxr::io::SubjectWriter&, const TimeLineRequest&)> time_line;  // `_time_line`: returns the time Line manifold's id
};

// A results file that holds VALUES on its surface: a non-empty `ImageGridAmp` and no `ImagingKernel`. Recognised by content, not by `DataType`.
bool is_source_map(const std::map<std::string, Value>& mat);
// Every non-kernel `results_*.mat` in a condition folder.
std::vector<std::filesystem::path> find_source_maps(const std::filesystem::path& cond_dir);

// `provenance`: what made the map (merged over the file's own `Provenance`, into the field's producer_json). `replace`: a source map of the same name is replaced.
Json export_source_map(nxr::io::SubjectWriter& sub, const std::filesystem::path& results_file, const std::filesystem::path& bst_root, const std::optional<std::string>& acquisition,
                       bool replace = false, const std::optional<Json>& provenance = std::nullopt, const MapHooks* hooks = nullptr);

// A `matrix_*.mat` over time (`Time` of more than two points) that is not a region map: an fMRI confounds matrix.
bool is_matrix_series(const std::filesystem::path& path);
// One FIELD PER ROW on the time Line of the BOLD map it names (needs hooks->time_line).
Json export_matrix_series(nxr::io::SubjectWriter& sub, const std::filesystem::path& matrix_file, const std::optional<std::string>& acquisition, const MapHooks* hooks = nullptr);

// An atlas of the cortex at `cortex_path` as REGIONS: (its levels, `unassigned` then the scouts in Brainstorm's order; the code of each level; the atlas SELECTION's id).
// nullopt when the cortex holds no such atlas.
std::optional<std::tuple<std::vector<std::string>, std::vector<std::int64_t>, std::string>> atlas_regions(nxr::io::SubjectWriter& sub, const std::string& cortex_path,
                                                                                                          const std::filesystem::path& cortex_file, const std::string& atlas);
struct RegionValue { std::int64_t code = 0; double value = 0; std::optional<std::int64_t> within_code; };
// values over an atlas Selection's members as `selection_measurement` rows; returns how many were written
int region_measurements(nxr::io::SubjectWriter& sub, const std::string& selection_id, const std::vector<RegionValue>& rows, const std::string& measure,
                        const std::optional<std::string>& unit, const std::string& computed_by = "nxr convert");
// A `matrix_*.mat` that names the atlas and the surface its rows are regions of (scout extraction's output).
bool is_region_map(const std::filesystem::path& path);
Json export_region_map(nxr::io::SubjectWriter& sub, const std::filesystem::path& matrix_file, const std::filesystem::path& bst_root, const std::optional<std::string>& acquisition);

// A map file of any kind: `matrix_*` a matrix over time or a region map, else a source map (static or over time).
Json export_map(nxr::io::SubjectWriter& sub, const std::filesystem::path& path, const std::filesystem::path& bst_root, const std::optional<std::string>& acquisition,
                bool replace = false, const std::optional<Json>& provenance = std::nullopt, const MapHooks* hooks = nullptr);

// The static maps on the DEFAULT ANATOMY: every source map and region map in the protocol's shared folders (`@default_study`, `@inter`, every `Group_analysis`
// condition) whose `SurfaceFile` is a `@default_subject` surface. A patient's own folders are never read.
std::vector<std::filesystem::path> template_map_files(const std::filesystem::path& root);

}  // namespace nxr::bst
