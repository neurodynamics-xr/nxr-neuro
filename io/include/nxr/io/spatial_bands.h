// nxr-io — THE SPATIAL-SCALE (Weyl) AXIS OF THE MEASUREMENT PYRAMID. It is stored the way D158 stores the frequency bands:
//
//   THE LINE        the `Eigenvalue` manifold of the dataset's default subject (a template): a `line` in the LAPLACE-BELTRAMI eigenvalue λ (1/m², coordinate system `eigenvalue:Laplace-Beltrami`, whose
//                   second system is the wavelength 2π/√λ), origin 0, spacing 1000 1/m², 4096 samples (0 … 4.1e6 1/m², wavelengths down to 3 mm). Its folder is `eigenvalue`.
//   THE BANDS       a `spans` Selection on it, named `<surface> spatial bands`: one span [lo, hi) per level l of the surface's tile ladder, code = l — the level IS its Weyl band (the ladder records
//                   level l's λ range, Weyl's law with the mean piece area: [λ_l / 2, λ_l), level 0 from 0). The spans depend on the surface's area, so a bank belongs to a surface (as nxr's frequency bank
//                   belongs to the dataset): the Selection sits in the surface's subject, on the dataset's line, with a dictionary naming each level.
//   A MEASUREMENT'S SPATIAL BAND  is its `within` key: within_selection_id = the bank, within_code = the level.
//   BOTH BANDS      (a measurement that has a frequency band AND a spatial band; the measurement row has ONE within pair, and a conjunction is a labelling of its own, D30): a JOINT BANDS Selection, `set`
//                   with no members stored, on the product Frequency × Eigenvalue, params_json.band_key = {bands, spatial_bands, n_bands, n_spatial, layout}: code = band · n_spatial + spatial band.
//                   No schema change. `band_key_code` / `band_key_split` are the one place that arithmetic lives.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/writer.h"

namespace nxr::io {

constexpr const char* kEigenvalueName = "Eigenvalue";
constexpr double kEigenvalueSpacing = 1000.0;          // 1/m²
constexpr std::int64_t kEigenvalueSamples = 4096;

/** The Eigenvalue manifold of the dataset's default subject, or nullopt. */
std::optional<Row> find_eigenvalue(Database& db);
/** GET-OR-CREATE the Eigenvalue manifold in `sub`, which must be the default subject (a template); returns its id. */
std::string ensure_eigenvalue(SubjectWriter& sub);

struct SpatialBandEdge { std::int64_t level = 0; double lo = 0, hi = 0; };   // [lo, hi) in 1/m²
/** The surface's `<surface> spatial bands` Selection (code = level), made when absent (the manifold from the default subject; when `sub` is the template it is made there). An existing bank with other
 *  edges is refused with the reason (a code is the level, so a surface has one bank). Returns the Selection's id. */
std::string spatial_bands_selection(SubjectWriter& sub, const std::string& surface, const std::vector<SpatialBandEdge>& levels);
std::string spatial_bands_name(const std::string& surface);

/** The joint bands Selection of a frequency bank and a spatial bank, made when absent; returns its id. */
std::string band_key_selection(SubjectWriter& sub, const std::string& bands_id, const std::string& spatial_bands_id, const std::string& name);

/** What a measurement's `within_selection_id` is: the kind of the Selection a row's band key points to. */
struct WithinKind {
  enum Kind { None, Bands, SpatialBands, BandKey } kind = None;
  std::int64_t n_spatial = 1;                           // BandKey: the spatial bands per frequency band
};
WithinKind within_kind(const Database& db, const std::optional<std::string>& within_selection_id);
std::int64_t band_key_code(std::int64_t band, std::int64_t spatial_band, std::int64_t n_spatial);
void band_key_split(std::int64_t code, std::int64_t n_spatial, std::int64_t& band, std::int64_t& spatial_band);

}  // namespace nxr::io
