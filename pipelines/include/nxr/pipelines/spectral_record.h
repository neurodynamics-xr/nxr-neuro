#pragma once
// THE SCALING RECORD of a stored spectral quantity. Every producer of a
// spectrum (psd, sourcepsd, sourcetensor, crossspectrum, quadrature, hodgepower) writes `producer_json.spectral_scaling`: what is stored and the
// normalisers a conversion needs (nxr::spectral). A reader asked for a scaling reads it back; a quantity without it (a ratio, a relative power, nxr's
// chain scaling) is refused a conversion.
//
//   {"stored": "density" | "spectrum" | "band-power", "degree": 1 | 2, "band_width_hz": B, "bin_enbw_hz": e (Welch), "power_unit": "A²·m²",
//    "per_window_sum": true (the analytic series: a window's value is its SUM of z zᴴ; the band power is sum / (2 · window_samples))}
#include <optional>
#include <string>

#include "nxr/io/json_text.h"
#include "nxr/spectral_scaling.h"
#include "nxr/welch.h"

namespace nxr::pipelines {

/** A Welch estimate's record: density or spectrum as the options scale it; `band_width_hz` the bandwidth one stored value spans (a band of n bins
 *  averaged: n × the resolution; one bin: the resolution); `degree` 2 for a product of two spectra. Null for nxr's chain scaling (not converted). */
nxr::io::Json welch_scaling_record(const nxr::welch::WelchOptions& wo, double band_width_hz, const std::string& power_unit, int degree = 1);
/** The analytic estimator's record: a band power (the variance of the band-pass Re z), stored as per-window sums of z zᴴ; B the band-pass's noise
 *  bandwidth (timebank::noiseBandwidth). */
nxr::io::Json analytic_scaling_record(double noise_bandwidth_hz, const std::string& power_unit);

struct SpectralRecord {
  nxr::spectral::Scaling scaling;
  std::string power_unit;
  bool per_window_sum = false;
};
/** The record of a producer_json, if it has one. */
std::optional<SpectralRecord> spectral_record(const nxr::io::Json& producer);

}  // namespace nxr::pipelines
