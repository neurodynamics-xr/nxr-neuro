// Unit strings as files write them, and their SI form: "uV", "µV" (U+00B5 or Greek mu, UTF-8 or Latin-1) and "mV" are volts times 1e-6 and
// 1e-3; "fT", "pT" and "T" are tesla. Any other unit ("mmHg", "%", "Boolean", "") is kept as the file writes it with scale 1: nothing is guessed.
#pragma once
#include <string>

namespace nxr::formats {

struct SiUnit {
  std::string unit;   // "V", "T", or the file's own string
  double scale;       // file unit -> unit
};

inline SiUnit si_unit(const std::string& file_unit) {
  std::size_t b = 0, e = file_unit.size();
  while (b < e && file_unit[b] == ' ') ++b;
  while (e > b && (file_unit[e - 1] == ' ' || file_unit[e - 1] == '\0')) --e;
  const std::string u = file_unit.substr(b, e - b);
  if (u.empty()) return {u, 1.0};
  const char base = u.back();
  if (base != 'V' && base != 'T') return {u, 1.0};
  const std::string prefix = u.substr(0, u.size() - 1);
  double s;
  if (prefix.empty()) s = 1.0;
  else if (prefix == "m") s = 1e-3;
  else if (prefix == "u" || prefix == "\xB5" || prefix == "\xC2\xB5" || prefix == "\xCE\xBC") s = 1e-6;
  else if (prefix == "n") s = 1e-9;
  else if (prefix == "p") s = 1e-12;
  else if (prefix == "f") s = 1e-15;
  else if (prefix == "k") s = 1e3;
  else return {u, 1.0};
  return {std::string(1, base), s};
}

}  // namespace nxr::formats
