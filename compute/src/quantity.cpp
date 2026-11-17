// nxr::quantity — the one parser of a stated quantity and the chart arithmetic (quantity.h).
#include "nxr/quantity.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>

namespace nxr::quantity {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

std::string trim(const std::string& s) {
  std::size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

void replace_all(std::string& s, const std::string& from, const std::string& to) {
  for (std::size_t p = 0; (p = s.find(from, p)) != std::string::npos; p += to.size()) s.replace(p, from.size(), to);
}

bool known_chart(const std::string& w) {
  static const char* names[] = {"linear", "log10", "clock", "eigenvalue", "wavelength", "area", "diameter", "parts", "index"};
  for (const char* n : names) if (w == n) return true;
  return false;
}

// the dimension and the factor to the dimension's base unit (m, s, Hz, m^2, 1/m^2)
struct UnitInfo { const char* dimension; double factor; };
const std::map<std::string, UnitInfo>& units() {
  static const std::map<std::string, UnitInfo> u = {
      {"m", {"length", 1}}, {"cm", {"length", 1e-2}}, {"mm", {"length", 1e-3}}, {"um", {"length", 1e-6}},
      {"s", {"time", 1}}, {"ms", {"time", 1e-3}}, {"us", {"time", 1e-6}},
      {"Hz", {"frequency", 1}}, {"kHz", {"frequency", 1e3}},
      {"m^2", {"area", 1}}, {"cm^2", {"area", 1e-4}}, {"mm^2", {"area", 1e-6}},
      {"1/m^2", {"eigenvalue", 1}}, {"1/cm^2", {"eigenvalue", 1e4}}, {"1/mm^2", {"eigenvalue", 1e6}},
  };
  return u;
}

// the chart a unit implies when the text names none
std::string chart_of_unit(const std::string& unit) {
  const auto it = units().find(unit);
  if (it == units().end()) return {};
  const std::string d = it->second.dimension;
  if (d == "frequency") return "linear";
  if (d == "time") return "clock";
  if (d == "length") return "diameter";
  if (d == "area") return "area";
  return "eigenvalue";
}

std::string num(double x) {
  if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
  char b[40];
  std::snprintf(b, sizeof b, "%.10g", x);
  return b;
}
}  // namespace

std::string canonical_unit(const std::string& unit) {
  std::string u = trim(unit);
  replace_all(u, "\xc2\xb2", "^2");          // ²
  replace_all(u, "\xc2\xb5", "u");           // µ (micro sign)
  replace_all(u, "\xce\xbc", "u");           // μ (greek mu)
  replace_all(u, "m2", "m^2");
  replace_all(u, "^^", "^");
  return u;
}

double unit_factor(const std::string& from, const std::string& to) {
  const std::string a = canonical_unit(from), b = canonical_unit(to);
  if (a == b) return 1.0;
  const auto fa = units().find(a), fb = units().find(b);
  if (fa == units().end()) throw std::invalid_argument("quantity: the unit '" + from + "' is not known (m, cm, mm, um, s, ms, Hz, kHz, mm^2, 1/m^2, ...)");
  if (fb == units().end()) throw std::invalid_argument("quantity: the unit '" + to + "' is not known (m, cm, mm, um, s, ms, Hz, kHz, mm^2, 1/m^2, ...)");
  if (std::string(fa->second.dimension) != fb->second.dimension)
    throw std::invalid_argument("quantity: '" + from + "' is a " + fa->second.dimension + " and '" + to + "' a " + fb->second.dimension + "; they do not convert");
  return fa->second.factor / fb->second.factor;
}

Quantity parse(const std::string& text) {
  const std::string t = trim(text);
  if (t.empty()) throw std::invalid_argument("quantity: empty text (write a number with a unit: '8-12 Hz', '2.5 s', 'tile 20 mm')");
  Quantity q;
  if (t.rfind("index:", 0) == 0) {                        // the one way to state an index
    char* end = nullptr;
    const std::string rest = trim(t.substr(6));
    const long long k = std::strtoll(rest.c_str(), &end, 10);
    if (rest.empty() || *end != '\0') throw std::invalid_argument("quantity: '" + text + "': an index is a whole number after 'index:'");
    q.lo = q.hi = static_cast<double>(k);
    q.chart = "index";
    q.unit = "index";
    return q;
  }
  std::size_t i = 0;
  std::string chart_word;
  if (std::isalpha(static_cast<unsigned char>(t[0]))) {                                  // a leading word: a chart (log10, tile, wavelength, parts)
    while (i < t.size() && std::isalnum(static_cast<unsigned char>(t[i]))) ++i;
    std::string w = t.substr(0, i);
    if (w == "tile") w = "diameter";
    if (known_chart(w)) { chart_word = w; while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i; }
    else throw std::invalid_argument("quantity: '" + text + "' starts with '" + w + "', which is not a chart (linear, log10, clock, eigenvalue, wavelength, area, tile, parts, index)");
  }
  std::string s = t.substr(i);
  bool at_least = false, at_most = false;
  for (const char* g : {">=", "\xe2\x89\xa5"}) if (s.rfind(g, 0) == 0) { at_least = true; s = trim(s.substr(std::string(g).size())); break; }
  if (!at_least) for (const char* g : {"<=", "\xe2\x89\xa4"}) if (s.rfind(g, 0) == 0) { at_most = true; s = trim(s.substr(std::string(g).size())); break; }
  char* end = nullptr;
  const double a = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || !std::isfinite(a)) throw std::invalid_argument("quantity: '" + text + "' holds no number (write '8-12 Hz', '2.5 s', 'tile 20 mm', 'parts 16')");
  std::string rest = trim(std::string(end));
  double b = a;
  bool range = false;
  for (const char* sep : {"-", "\xe2\x80\x93", "..", "to "}) {                         // "8-12", "8–12", "8..12", "8 to 12"
    if (rest.rfind(sep, 0) == 0) {
      const std::string after = trim(rest.substr(std::string(sep).size()));
      char* e2 = nullptr;
      const double v = std::strtod(after.c_str(), &e2);
      if (e2 != after.c_str()) { b = v; rest = trim(std::string(e2)); range = true; break; }
    }
  }
  if (range && b < a) throw std::invalid_argument("quantity: '" + text + "' is a range with its low end above its high end");
  if (at_least) b = kInf;
  if (at_most) b = a;
  q.lo = at_most ? -kInf : a;
  q.hi = b;
  std::string unit = canonical_unit(rest);
  if (!unit.empty() && unit[0] == '/') unit = unit.substr(1);                              // "16/hemisphere": parts per hemisphere
  q.unit = unit;
  q.chart = chart_word.empty() ? chart_of_unit(unit) : chart_word;
  if (q.chart.empty())
    throw std::invalid_argument("quantity: '" + text + "' has no chart: name its unit (Hz, s, mm, mm^2, 1/m^2) or its chart ('parts 16', 'wavelength 40 mm', 'tile 20 mm')");
  if (q.chart == "parts") { if (q.unit.empty()) q.unit = "parts"; }
  else if (q.chart == "index") { if (q.unit.empty()) q.unit = "index"; }
  else if (q.unit.empty()) throw std::invalid_argument("quantity: '" + text + "' has no unit (the " + q.chart + " chart is in a unit: Hz, s, mm, ...)");
  if (q.chart == "log10" && q.unit.empty()) q.unit = "Hz";
  return q;
}

std::string format(const Quantity& q) {
  std::string out;
  const std::string implied = chart_of_unit(q.unit);
  if (q.chart == "index") return "index:" + num(q.lo);
  if (q.chart != implied && q.chart != "diameter") out += q.chart + " ";
  else if (q.chart == "diameter" && implied != "diameter") out += "tile ";
  if (std::isinf(q.hi) && q.hi > 0) out += ">= " + num(q.lo);
  else if (std::isinf(q.lo) && q.lo < 0) out += "<= " + num(q.hi);
  else out += q.lo == q.hi ? num(q.lo) : num(q.lo) + "-" + num(q.hi);
  if (q.chart == "parts" && q.unit != "parts") out += "/" + q.unit;
  else if (!q.unit.empty() && q.chart != "parts") out += " " + q.unit;
  return out;
}

double to_reference(const Quantity& q, double v, const Chart& chart) {
  if (q.chart != chart.name) throw std::invalid_argument("quantity: the quantity is in the " + q.chart + " chart, not the " + chart.name + " chart it was given");
  double c = v;
  if (!chart.unit.empty() && !q.unit.empty() && canonical_unit(q.unit) != canonical_unit(chart.unit)) {
    try { c = v * unit_factor(q.unit, chart.unit); }
    catch (const std::invalid_argument&) { throw std::invalid_argument("quantity: '" + q.unit + "' is not a unit of the " + chart.name + " chart (its unit is " + chart.unit + ")"); }
  }
  const std::string& tr = chart.transition;
  if (tr == "identity") return c;
  if (tr == "affine") {
    if (chart.params.size() < 2 || chart.params[0] == 0) throw std::invalid_argument("quantity: the affine chart '" + chart.name + "' needs a non-zero scale and an offset");
    return (c - chart.params[1]) / chart.params[0];
  }
  if (tr == "log") {
    const double base = chart.params.empty() ? 10.0 : chart.params[0];
    if (!(base > 0) || base == 1) throw std::invalid_argument("quantity: the log chart '" + chart.name + "' needs a base > 0 and not 1");
    return std::pow(base, c);
  }
  if (tr == "power") {
    if (chart.params.size() < 2 || chart.params[0] == 0 || chart.params[1] == 0) throw std::invalid_argument("quantity: the power chart '" + chart.name + "' needs a non-zero factor and exponent");
    if (c == 0) return chart.params[1] < 0 ? kInf : 0.0;                                   // wavelength infinity is eigenvalue 0
    if (std::isinf(c)) return chart.params[1] < 0 ? 0.0 : kInf;
    return std::pow(c / chart.params[0], 1.0 / chart.params[1]);
  }
  throw std::invalid_argument("quantity: the chart '" + chart.name + "' is defined by '" + tr + "', which cannot be computed here (identity, affine, log or power)");
}

Quantity to_reference(const Quantity& q, const Chart& chart) {
  Quantity out = q;
  const double a = to_reference(q, q.lo, chart), b = to_reference(q, q.hi, chart);
  out.lo = std::min(a, b);
  out.hi = std::max(a, b);
  out.unit = chart.unit;
  out.chart = "reference";
  return out;
}

long long as_index(const Quantity& q) {
  if (q.chart != "index") throw std::invalid_argument("quantity: an index is stated through the index chart ('index:5'), not as '" + format(q) + "'");
  return static_cast<long long>(q.lo);
}

}  // namespace nxr::quantity
