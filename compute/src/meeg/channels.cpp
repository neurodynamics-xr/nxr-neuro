// nxr::meeg::channelPositions and goodChannelsOfType (see the header).
#include "nxr/meeg/channels.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nxr::meeg {

std::vector<double> channelPositions(const std::vector<double>& points, const std::vector<double>& coils, const std::vector<std::int32_t>& indptr,
                                     const std::vector<std::int32_t>& indices, int coilPoints) {
  if (points.size() % 3 != 0) throw std::invalid_argument("channelPositions: the points are not [M x 3]");
  if (coilPoints < 1) throw std::invalid_argument("channelPositions: coilPoints is at least 1");
  std::vector<double> out = points;
  if (indptr.empty() && indices.empty() && coils.empty()) return out;
  const std::size_t M = points.size() / 3;
  if (coils.size() % 3 != 0) throw std::invalid_argument("channelPositions: the coils are not [Q x 3]");
  if (indptr.size() != M + 1) throw std::invalid_argument("channelPositions: indptr has " + std::to_string(indptr.size()) + " entries for " + std::to_string(M) + " channels (M + 1)");
  const std::int64_t Q = static_cast<std::int64_t>(coils.size() / 3);
  for (std::size_t m = 0; m < M; ++m) {
    const std::int64_t a = indptr[m], b = indptr[m + 1];
    if (a < 0 || b < a || b > static_cast<std::int64_t>(indices.size())) throw std::invalid_argument("channelPositions: indptr is not a CSR row pointer over the indices");
    const std::int64_t k = std::min<std::int64_t>(b - a, coilPoints);
    if (k == 0) continue;
    double s[3] = {0, 0, 0};
    for (std::int64_t j = a; j < a + k; ++j) {
      const std::int64_t q = indices[static_cast<std::size_t>(j)];
      if (q < 0 || q >= Q) throw std::invalid_argument("channelPositions: a coil index is outside the coils");
      for (int c = 0; c < 3; ++c) s[c] += coils[static_cast<std::size_t>(3 * q + c)];
    }
    for (int c = 0; c < 3; ++c) out[3 * m + static_cast<std::size_t>(c)] = s[c] / static_cast<double>(k);
  }
  return out;
}

std::vector<std::int32_t> goodChannelsOfType(const std::vector<std::string>& channelTypes, const std::vector<std::string>& channelQuality, const std::string& sensorType) {
  if (channelTypes.size() != channelQuality.size())
    throw std::invalid_argument("goodChannelsOfType: " + std::to_string(channelTypes.size()) + " channel types for " + std::to_string(channelQuality.size()) + " channel qualities");
  std::vector<std::string> measured;
  if (sensorType == "MEG") measured = {"MEG", "MEG MAG", "MEG GRAD"};
  else if (sensorType == "EEG") measured = {"EEG"};
  else throw std::invalid_argument("goodChannelsOfType: the sensor type is MEG or EEG, not '" + sensorType + "'");
  std::vector<std::int32_t> chosen;
  for (std::size_t channel = 0; channel < channelTypes.size(); ++channel) {
    const std::string& quality = channelQuality[channel];
    if (quality != "good" && quality != "bad")
      throw std::invalid_argument("goodChannelsOfType: channel " + std::to_string(channel) + " has the quality '" + quality + "', neither good nor bad");
    if (quality == "good" && std::find(measured.begin(), measured.end(), channelTypes[channel]) != measured.end()) chosen.push_back(static_cast<std::int32_t>(channel));
  }
  if (chosen.size() < 4)
    throw std::invalid_argument("goodChannelsOfType: " + std::to_string(chosen.size()) + " good " + sensorType + " channel(s), fewer than the four a sensor surface needs");
  return chosen;
}

}  // namespace nxr::meeg
