// nxr-io — private helpers of the `bst` importers that read MATLAB values the way nxr's Python does (pymatreader's squeezed values, `_empty`, `_scalar`,
// `_text`, `_jsonable`). Not installed; used by bst_inverse.cpp, bst_maps.cpp and bst_timefreq.cpp.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "nxr/bst/mat.h"
#include "nxr/bst/pyrepr.h"
#include "nxr/io/json_text.h"

namespace nxr::bst::detail {

using nxr::io::Json;

inline std::string strip(const std::string& s) {
  const char* ws = " \t\n\r\f\v";
  const auto a = s.find_first_not_of(ws);
  if (a == std::string::npos) return "";
  return s.substr(a, s.find_last_not_of(ws) - a + 1);
}
inline std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}
inline std::string upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return s;
}
inline bool starts_with(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }
inline bool ends_with(const std::string& s, const std::string& p) { return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0; }

// str(float) as Python prints it: the shortest text that reads back, with ".0" on an integer
inline std::string py_float(double x) {
  if (std::isnan(x)) return "nan";
  if (std::isinf(x)) return x < 0 ? "-inf" : "inf";
  return float_repr(x);
}

inline const Value* get(const std::map<std::string, Value>& m, const std::string& name) {
  const auto it = m.find(name);
  return it == m.end() ? nullptr : &it->second;
}
// a field of a struct value (the first element); nullptr when `st` is not a struct or has no such field
inline const Value* sfield(const Value* st, const std::string& name, std::size_t elem = 0) { return st && st->is_struct() ? st->field(name, elem) : nullptr; }

// `_empty`: None, "", an empty array / cell
inline bool py_empty(const Value* v) {
  if (!v) return true;
  if (v->is_char()) return v->text.empty() || v->text[0].empty();
  if (v->is_struct()) return false;
  return v->numel() == 0;
}
// the truth value of `x or y`: not None, not "", not 0, not an empty array
inline bool py_truthy(const Value* v) {
  if (py_empty(v)) return false;
  if (v->is_numeric()) return v->scalar() != 0;
  return true;
}
// `str(x)` of a char (the first row); a number as Python prints a numpy float; "" for anything else
inline std::string py_str(const Value* v) {
  if (!v) return "";
  if (v->is_char()) return v->text.empty() ? "" : v->text[0];
  if (v->is_numeric() && v->numel() > 0) {
    const double x = v->scalar();
    if (v->cls == Class::Double || v->cls == Class::Single) return py_float(x);
    if (v->is_logical) return x != 0 ? "True" : "False";
    return std::to_string(static_cast<std::int64_t>(x));
  }
  return "";
}
// `_text`: an optional text field as a str: "" when absent or empty, else str(first element).strip()
inline std::string text_field(const Value* v) { return py_empty(v) ? "" : strip(py_str(v)); }

// `_jsonable`: an integral float as an int (MATLAB's jsonencode writes 300, not 300.0)
inline Json jsonable(const Json& j) {
  if (j.is_object()) {
    Json o = Json::object();
    for (auto it = j.begin(); it != j.end(); ++it) o[it.key()] = jsonable(it.value());
    return o;
  }
  if (j.is_array()) {
    Json a = Json::array();
    for (const auto& e : j) a.push_back(jsonable(e));
    return a;
  }
  if (j.is_number_float()) {
    const double x = j.get<double>();
    if (std::isfinite(x) && x == std::floor(x) && std::fabs(x) < 9007199254740992.0) return Json(static_cast<std::int64_t>(x));
  }
  return j;
}

// A value as JSON the way pymatreader + `tolist()` give it: a char its text, a scalar a number, an array its squeezed nested lists, a struct an object.
inline Json to_json(const Value& v) {
  switch (v.cls) {
    case Class::Char: {
      if (v.text.empty()) return "";
      if (v.text.size() == 1) return v.text[0];
      Json a = Json::array();
      for (const auto& t : v.text) a.push_back(t);
      return a;
    }
    case Class::Struct: {
      auto obj = [&](std::size_t e) {
        Json o = Json::object();
        for (const auto& f : v.fields) { const Value* x = v.field(f, e); o[f] = x ? to_json(*x) : Json(nullptr); }
        return o;
      };
      if (v.numel() == 1) return obj(0);
      Json a = Json::array();
      for (std::size_t e = 0; e < v.numel(); ++e) a.push_back(obj(e));
      return a;
    }
    default: break;
  }
  const bool cell = v.is_cell();
  if (!cell && !v.is_numeric()) return nullptr;
  const std::vector<double> d = cell ? std::vector<double>() : v.doubles();
  const bool integer = !cell && v.cls != Class::Double && v.cls != Class::Single;
  auto leaf = [&](std::size_t i) -> Json {
    if (cell) return to_json(v.cell(i));
    if (v.is_logical) return d[i] != 0;
    if (integer) return static_cast<std::int64_t>(d[i]);
    return d[i];
  };
  std::vector<std::size_t> dims, stride;     // squeezed, with MATLAB's (column-major) strides
  std::size_t s = 1;
  for (std::size_t n : v.dims) { if (n != 1) { dims.push_back(n); stride.push_back(s); } s *= n; }
  if (dims.empty()) return v.numel() ? jsonable(leaf(0)) : Json(Json::array());
  auto rec = [&](auto&& self, std::size_t axis, std::size_t base) -> Json {
    Json a = Json::array();
    for (std::size_t i = 0; i < dims[axis]; ++i) {
      const std::size_t at = base + i * stride[axis];
      a.push_back(axis + 1 == dims.size() ? leaf(at) : self(self, axis + 1, at));
    }
    return a;
  };
  return jsonable(rec(rec, 0, 0));
}

// A numeric value as a squeezed numpy array: 0, 1 or 2 dimensions (`ndim`), the data row after row.
struct Mat {
  std::vector<double> d;
  std::size_t rows = 0, cols = 1;
  int ndim = 0;
  double at(std::size_t r, std::size_t c) const { return d[r * cols + c]; }
};
inline Mat as_mat(const Value* v) {
  Mat m;
  if (!v || !v->is_numeric()) return m;
  std::vector<std::size_t> sq;
  for (std::size_t n : v->dims) if (n != 1) sq.push_back(n);
  const std::vector<double> cm = v->doubles();
  m.ndim = static_cast<int>(sq.size());
  if (sq.size() <= 1) { m.d = cm; m.rows = cm.size(); m.cols = 1; if (sq.empty()) m.ndim = 0; return m; }
  m.rows = sq[0];
  m.cols = cm.size() / std::max<std::size_t>(1, sq[0]);
  m.d.resize(cm.size());
  for (std::size_t r = 0; r < m.rows; ++r) for (std::size_t c = 0; c < m.cols; ++c) m.d[r * m.cols + c] = cm[r + c * m.rows];
  return m;
}

// The strings of a cell of chars or a char array (np.atleast_1d(np.asarray(x, dtype=object)).ravel() -> str)
inline std::vector<std::string> strings_of(const Value* v) {
  std::vector<std::string> out;
  if (!v) return out;
  if (v->is_char()) for (const auto& t : v->text) out.push_back(t);
  else if (v->is_cell()) for (const auto& e : v->elems) out.push_back(py_str(&e));
  return out;
}

}  // namespace nxr::bst::detail
