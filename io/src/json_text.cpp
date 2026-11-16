#include "nxr/io/json_text.h"

#include <charconv>
#include <cmath>
#include <cstdio>

namespace nxr::io {

const char* const kEmptyGroup = "{\n  \"zarr_format\": 3,\n  \"node_type\": \"group\",\n  \"attributes\": {}\n}";

std::string js_number(double x) {
  if (!std::isfinite(x)) return "null";
  if (x == 0) return "0";
  const std::string sign = x < 0 ? "-" : "";
  char buf[64];
  // the shortest digits that round-trip, as d.ddde+XX
  auto r = std::to_chars(buf, buf + sizeof buf, std::fabs(x), std::chars_format::scientific);
  const std::string s(buf, r.ptr);
  const auto e_at = s.find('e');
  std::string mant = s.substr(0, e_at);
  const int e10 = std::stoi(s.substr(e_at + 1));
  std::string digits;
  for (char c : mant) if (c != '.') digits += c;
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
  const int k = static_cast<int>(digits.size());
  const int n = e10 + 1;  // value = 0.digits x 10^n
  std::string out;
  if (k <= n && n <= 21) out = digits + std::string(static_cast<std::size_t>(n - k), '0');
  else if (0 < n && n <= 21) out = digits.substr(0, static_cast<std::size_t>(n)) + "." + digits.substr(static_cast<std::size_t>(n));
  else if (-6 < n && n <= 0) out = "0." + std::string(static_cast<std::size_t>(-n), '0') + digits;
  else {
    const int e = n - 1;
    const std::string es = std::string(e > 0 ? "+" : "-") + std::to_string(std::abs(e));
    out = k == 1 ? digits + "e" + es : digits.substr(0, 1) + "." + digits.substr(1) + "e" + es;
  }
  return sign + out;
}

namespace {

void quote(const std::string& s, std::string& out) {
  out += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
        else out += static_cast<char>(c);
    }
  }
  out += '"';
}

void put(const Json& v, std::string& out) {
  switch (v.type()) {
    case Json::value_t::null: out += "null"; break;
    case Json::value_t::boolean: out += v.get<bool>() ? "true" : "false"; break;
    case Json::value_t::number_integer: out += std::to_string(v.get<std::int64_t>()); break;
    case Json::value_t::number_unsigned: out += std::to_string(v.get<std::uint64_t>()); break;
    case Json::value_t::number_float: out += js_number(v.get<double>()); break;
    case Json::value_t::string: quote(v.get_ref<const std::string&>(), out); break;
    case Json::value_t::array: {
      out += '[';
      bool first = true;
      for (const auto& x : v) { if (!first) out += ','; first = false; put(x, out); }
      out += ']';
      break;
    }
    case Json::value_t::object: {
      out += '{';
      bool first = true;
      for (auto it = v.begin(); it != v.end(); ++it) {
        if (!first) out += ',';
        first = false;
        quote(it.key(), out);
        out += ':';
        put(it.value(), out);
      }
      out += '}';
      break;
    }
    default: out += "null";
  }
}

}  // namespace

std::string js_json(const Json& v) {
  std::string out;
  put(v, out);
  return out;
}

std::string zarr_json_text(const Json& doc, const std::string& attributes) {
  Json rest = Json::object();
  for (auto it = doc.begin(); it != doc.end(); ++it) if (it.key() != "attributes") rest[it.key()] = it.value();
  const std::string head = js_json(rest);
  return head.substr(0, head.size() - 1) + (head.size() > 2 ? "," : "") + "\"attributes\":" + attributes + "}";
}

}  // namespace nxr::io
