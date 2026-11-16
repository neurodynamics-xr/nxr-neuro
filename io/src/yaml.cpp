// nxr-io — YAML read as JSON (yaml.h): the one translation unit that compiles rapidyaml's single header.
#define RYML_SINGLE_HDR_DEFINE_NOW
#define RYML_DEFAULT_CALLBACK_USES_EXCEPTIONS
#include "nxr/io/yaml.h"

#include <cmath>
#include <limits>
#include <regex>

#include "nxr/io/errors.h"
#include "ryml/ryml_all.hpp"

namespace nxr::io {

namespace {
std::string str(ryml::csubstr s) { return std::string(s.str ? s.str : "", s.len); }
// PyYAML's resolver (yaml/resolver.py, YAML 1.1 as safe_load reads it) for the plain scalars a definition holds
Json scalar(const std::string& s) {
  static const std::regex kNull("^(~|null|Null|NULL|)$"), kTrue("^(yes|Yes|YES|true|True|TRUE|on|On|ON)$"), kFalse("^(no|No|NO|false|False|FALSE|off|Off|OFF)$");
  static const std::regex kInt("^[-+]?(0|[1-9][0-9_]*)$"), kFloat("^[-+]?([0-9][0-9_]*)?\\.[0-9_]*([eE][-+][0-9]+)?$|^[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+$");
  static const std::regex kInf("^[-+]?\\.(inf|Inf|INF)$"), kNan("^\\.(nan|NaN|NAN)$");
  if (std::regex_match(s, kNull)) return nullptr;
  if (std::regex_match(s, kTrue)) return true;
  if (std::regex_match(s, kFalse)) return false;
  auto clean = [](std::string x) { x.erase(std::remove(x.begin(), x.end(), '_'), x.end()); return x; };
  if (std::regex_match(s, kInt)) return std::stoll(clean(s));
  if (std::regex_match(s, kFloat)) return std::stod(clean(s));
  if (std::regex_match(s, kInf)) return s[0] == '-' ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
  if (std::regex_match(s, kNan)) return std::numeric_limits<double>::quiet_NaN();
  return s;
}
Json convert(ryml::ConstNodeRef n) {
  if (n.is_map()) {
    Json o = Json::object();
    for (ryml::ConstNodeRef c : n.children()) o[str(c.key())] = convert(c);
    return o;
  }
  if (n.is_seq()) {
    Json a = Json::array();
    for (ryml::ConstNodeRef c : n.children()) a.push_back(convert(c));
    return a;
  }
  if (!n.has_val()) return nullptr;
  if (n.is_val_quoted()) return str(n.val());
  return scalar(str(n.val()));
}
}  // namespace

Json parse_yaml(const std::string& text, const std::string& where) {
  try {
    ryml::Tree t = ryml::parse_in_arena(ryml::to_csubstr(where), ryml::to_csubstr(text));
    t.resolve();                                                            // anchors and aliases
    if (t.empty() || !t.rootref().readable()) return nullptr;
    ryml::ConstNodeRef root = t.crootref();
    if (root.is_stream()) {
      if (root.num_children() == 0) return nullptr;
      if (root.num_children() > 1) throw Error(code::invalid_argument, where + ": several YAML documents (one is read)");
      root = root.first_child();
    }
    return convert(root);
  } catch (const Error&) {
    throw;
  } catch (const std::exception& e) {
    throw Error(code::invalid_argument, where + ": not valid YAML (" + e.what() + ")");
  }
}

}  // namespace nxr::io
