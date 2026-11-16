// nxr-io — values as Python prints them (repr, str of a float), for the messages and lines nxr's Python writes.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace nxr::bst {

// repr(float): the shortest digits that read back the same, in Python's layout (1.0, 0.0001, 1e-05, 1.5e+16)
inline std::string float_repr(double x) {
  if (std::isnan(x)) return "NaN";
  if (std::isinf(x)) return x < 0 ? "-Infinity" : "Infinity";
  if (x == 0) return std::signbit(x) ? "-0.0" : "0.0";
  char buf[64];
  int prec = 1;
  for (; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof buf, "%.*e", prec - 1, x);
    if (std::strtod(buf, nullptr) == x) break;
  }
  std::string s = buf;   // d.ddde+XX
  const bool neg = s[0] == '-';
  if (neg) s.erase(0, 1);
  const auto epos = s.find('e');
  const int exp10 = std::atoi(s.c_str() + epos + 1);
  std::string digits;
  for (std::size_t i = 0; i < epos; ++i) if (s[i] != '.') digits += s[i];
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
  std::string out;
  if (exp10 >= -4 && exp10 < 16) {
    if (exp10 >= 0) {
      const std::size_t point = static_cast<std::size_t>(exp10) + 1;
      if (digits.size() <= point) out = digits + std::string(point - digits.size(), '0') + ".0";
      else out = digits.substr(0, point) + "." + digits.substr(point);
    } else {
      out = "0." + std::string(static_cast<std::size_t>(-exp10 - 1), '0') + digits;
    }
  } else {
    out = digits.substr(0, 1) + (digits.size() > 1 ? "." + digits.substr(1) : "");
    char e[16];
    std::snprintf(e, sizeof e, "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
    out += e;
  }
  return neg ? "-" + out : out;
}

// repr(str): single quotes, unless the text holds a single quote and no double quote; backslash, newline, return, tab and control characters escaped
inline std::string str_repr(const std::string& s) {
  const bool dq = s.find('\'') != std::string::npos && s.find('"') == std::string::npos;
  const char q = dq ? '"' : '\'';
  std::string out(1, q);
  for (unsigned char c : s) {
    if (c == static_cast<unsigned char>(q) || c == '\\') { out += '\\'; out += static_cast<char>(c); }
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else if (c < 0x20 || c == 0x7f) { char b[8]; std::snprintf(b, sizeof b, "\\x%02x", c); out += b; }
    else out += static_cast<char>(c);
  }
  out += q;
  return out;
}

}  // namespace nxr::bst
