// The scaling record of a stored spectral quantity (spectral_record.h).
#include "nxr/pipelines/spectral_record.h"

namespace nxr::pipelines {

using nxr::io::Json;

Json welch_scaling_record(const nxr::signal::spectral::WelchOptions& wo, double band_width_hz, const std::string& power_unit, int degree) {
  if (wo.scaling == nxr::signal::spectral::Scaling::Chain || wo.scaling == nxr::signal::spectral::Scaling::Brainstorm2018) return Json();   // no conversion is defined for these
  return Json{{"stored", wo.scaling == nxr::signal::spectral::Scaling::Density ? "density" : "spectrum"}, {"degree", degree}, {"band_width_hz", band_width_hz},
              {"bin_enbw_hz", nxr::signal::spectral::enbwPerBin(wo)}, {"power_unit", power_unit},
              {"note", "converted at read (nxr::spectral): density = stored (density) or stored / bin_enbw_hz (spectrum); band power = density x band_width_hz (the broadband reading)"}};
}

Json analytic_scaling_record(double noise_bandwidth_hz, const std::string& power_unit) {
  return Json{{"stored", "band-power"}, {"degree", 1}, {"band_width_hz", noise_bandwidth_hz}, {"power_unit", power_unit}, {"per_window_sum", true},
              {"note", "a window's value is its SUM of z z^H; the band power (the variance of the band-pass Re z) is sum / (2 window_samples); density = band power / band_width_hz (the band-pass's noise bandwidth)"}};
}

std::optional<SpectralRecord> spectral_record(const Json& pj) {
  if (!pj.is_object() || !pj.contains("spectral_scaling") || !pj["spectral_scaling"].is_object()) return std::nullopt;
  const Json& r = pj["spectral_scaling"];
  SpectralRecord out;
  out.scaling.stored = nxr::signal::spectral_scaling::parseStored(r.value("stored", std::string("density")));
  out.scaling.degree = r.value("degree", 1);
  out.scaling.bandWidthHz = r.contains("band_width_hz") && r["band_width_hz"].is_number() ? r["band_width_hz"].get<double>() : std::nan("");
  out.scaling.binEnbwHz = r.contains("bin_enbw_hz") && r["bin_enbw_hz"].is_number() ? r["bin_enbw_hz"].get<double>() : std::nan("");
  out.power_unit = r.value("power_unit", std::string("unit\xc2\xb2"));
  out.per_window_sum = r.value("per_window_sum", false);
  return out;
}

}  // namespace nxr::pipelines
