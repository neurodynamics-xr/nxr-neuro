// json.dumps as Python writes it (ensure_ascii; ", " and ": " separators, or one item a line with `indent`; floats as repr), for the command line's output.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "nxr/bst/pyrepr.h"
#include "nxr/io/json_text.h"

namespace pyjson {

using nxr::io::Json;

using nxr::bst::float_repr;

inline void string_out(const std::string& s, std::string& out) {
  out += '"';
  for (std::size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c == '"') { out += "\\\""; ++i; }
    else if (c == '\\') { out += "\\\\"; ++i; }
    else if (c == '\n') { out += "\\n"; ++i; }
    else if (c == '\r') { out += "\\r"; ++i; }
    else if (c == '\t') { out += "\\t"; ++i; }
    else if (c == '\b') { out += "\\b"; ++i; }
    else if (c == '\f') { out += "\\f"; ++i; }
    else if (c < 0x20 || c == 0x7f) { char b[8]; std::snprintf(b, sizeof b, c == 0x7f ? "\x7f" : "\\u%04x", c); out += b; ++i; }
    else if (c < 0x80) { out += static_cast<char>(c); ++i; }
    else {
      std::uint32_t cp = 0; int n = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : 1;
      cp = c & (0x3F >> n);
      for (int k = 1; k <= n && i + k < s.size(); ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
      i += n + 1;
      char b[16];
      if (cp >= 0x10000) { cp -= 0x10000; std::snprintf(b, sizeof b, "\\u%04x\\u%04x", 0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF)); }
      else std::snprintf(b, sizeof b, "\\u%04x", cp);
      out += b;
    }
  }
  out += '"';
}

// `indent` < 0: one line, ", " and ": "; otherwise one item a line, `indent` spaces a level (json.dumps(indent=n))
inline void dump(const Json& v, std::string& out, int indent = -1, int level = 0) {
  auto newline = [&](int lvl) { out += '\n'; out.append(static_cast<std::size_t>(indent * lvl), ' '); };
  switch (v.type()) {
    case Json::value_t::null: out += "null"; return;
    case Json::value_t::boolean: out += v.get<bool>() ? "true" : "false"; return;
    case Json::value_t::number_integer: out += std::to_string(v.get<std::int64_t>()); return;
    case Json::value_t::number_unsigned: out += std::to_string(v.get<std::uint64_t>()); return;
    case Json::value_t::number_float: out += float_repr(v.get<double>()); return;
    case Json::value_t::string: string_out(v.get_ref<const std::string&>(), out); return;
    case Json::value_t::array: {
      if (v.empty()) { out += "[]"; return; }
      out += '[';
      bool first = true;
      for (const auto& x : v) {
        if (!first) out += indent < 0 ? ", " : ",";
        first = false;
        if (indent >= 0) newline(level + 1);
        dump(x, out, indent, level + 1);
      }
      if (indent >= 0) newline(level);
      out += ']';
      return;
    }
    case Json::value_t::object: {
      if (v.empty()) { out += "{}"; return; }
      out += '{';
      bool first = true;
      for (auto it = v.begin(); it != v.end(); ++it) {
        if (!first) out += indent < 0 ? ", " : ",";
        first = false;
        if (indent >= 0) newline(level + 1);
        string_out(it.key(), out);
        out += ": ";
        dump(it.value(), out, indent, level + 1);
      }
      if (indent >= 0) newline(level);
      out += '}';
      return;
    }
    default: out += "null";
  }
}

inline std::string dumps(const Json& v, int indent = -1) { std::string s; dump(v, s, indent); return s; }

}  // namespace pyjson
