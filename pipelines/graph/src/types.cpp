// nxr-graph — THE TYPE SYSTEM's terms: manifolds, domains, the printed form of a type and its parser, unification (types.h; spec docs/specs/2026-11-17-type-system.md, sections 3 and 4).
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>

#include "nxr/graph/types.h"

namespace nxr::graph {

const char* to_string(ManifoldKind k) {
  switch (k) {
    case ManifoldKind::points: return "points";
    case ManifoldKind::line: return "line";
    case ManifoldKind::surface: return "surface";
    case ManifoldKind::volume: return "volume";
    case ManifoldKind::curve: return "curve";
    case ManifoldKind::graph: return "graph";
    case ManifoldKind::members: return "members";
  }
  return "?";
}
const char* to_string(Role r) {
  switch (r) {
    case Role::none: return "none";
    case Role::time: return "time";
    case Role::frames: return "frames";
    case Role::time_tiles: return "time_tiles";
    case Role::frequency: return "frequency";
    case Role::modes: return "modes";
    case Role::levels: return "levels";
  }
  return "?";
}

// ───────────────────────────────────────────── manifold terms ─────────────────────────────────────────────

namespace {

ManifoldRef make_ref(Manifold m) { return std::make_shared<const Manifold>(std::move(m)); }

Role role_of_label(const std::string& label) {
  if (label == "time") return Role::time;
  if (label == "frames") return Role::frames;
  if (label == "time_tiles") return Role::time_tiles;
  if (label == "frequency") return Role::frequency;
  if (label == "modes") return Role::modes;
  if (label == "levels") return Role::levels;
  return Role::none;
}

std::string trim(const std::string& s) {
  std::size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

std::string number(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%g", v);
  return buf;
}

}  // namespace

ManifoldRef manifold_id(const std::string& label, const std::string& key, ManifoldKind kind, Role role, std::optional<CoordinateSystem> cs, std::int64_t n) {
  Manifold m; m.form = Manifold::Form::id; m.label = label; m.key = key; m.kind = kind; m.role = role; m.cs = std::move(cs); m.n = n;
  return make_ref(std::move(m));
}
ManifoldRef time_line(const std::string& key, double sfreq) {
  Manifold m; m.form = Manifold::Form::id; m.label = "time"; m.key = key; m.kind = ManifoldKind::line; m.role = Role::time; m.cs = CoordinateSystem{"time", "clock", "s"};
  if (sfreq > 0) m.spacing = 1.0 / sfreq;
  return make_ref(std::move(m));
}
ManifoldRef frame_line(const std::string& key) {
  Manifold m; m.form = Manifold::Form::id; m.label = "frames"; m.key = key; m.kind = ManifoldKind::line; m.role = Role::frames; m.cs = CoordinateSystem{"time", "clock", "s"};
  return make_ref(std::move(m));
}
ManifoldRef anonymous(int number_, ManifoldKind kind, Role role) {
  Manifold m; m.form = Manifold::Form::anonymous; m.key = std::to_string(number_); m.kind = kind; m.role = role;
  return make_ref(std::move(m));
}
ManifoldRef var(const std::string& name) {
  Manifold m; m.form = Manifold::Form::variable; m.key = name;
  return make_ref(std::move(m));
}
ManifoldRef time_var(const std::string& name) {
  Manifold m; m.form = Manifold::Form::variable; m.key = name; m.role = Role::time;
  return make_ref(std::move(m));
}
ManifoldRef frames_var(const std::string& name) {
  Manifold m; m.form = Manifold::Form::variable; m.key = name; m.role = Role::frames;
  return make_ref(std::move(m));
}
ManifoldRef var_rest(const std::string& name) {
  Manifold m; m.form = Manifold::Form::variable; m.key = name; m.rest = true;
  return make_ref(std::move(m));
}
ManifoldRef from_or_restriction_of(const ManifoldRef& v) {
  Manifold m = *v; m.subsume = true;
  return make_ref(std::move(m));
}
ManifoldRef made(const std::string& label, const std::string& key, ManifoldKind kind, Role role, std::optional<CoordinateSystem> cs) {
  Manifold m; m.form = Manifold::Form::made; m.label = label; m.key = key; m.kind = kind; m.role = role; m.cs = std::move(cs);
  return make_ref(std::move(m));
}
ManifoldRef made_line(const std::string& label, const std::string& key, Role role, const std::string& unit, double origin, std::optional<double> rate) {
  Manifold m; m.form = Manifold::Form::made; m.label = label; m.key = key; m.kind = ManifoldKind::line; m.role = role; m.unit = unit; m.origin = origin; m.rate = rate;
  return make_ref(std::move(m));
}
ManifoldRef restricted(const ManifoldRef& of, const std::string& by) {
  Manifold m; m.form = Manifold::Form::restricted; m.of = of; m.by = by; m.kind = of ? of->kind : ManifoldKind::points; m.role = of ? of->role : Role::none; m.cs = of ? of->cs : std::nullopt;
  return make_ref(std::move(m));
}
ManifoldRef members(const std::string& partition) {
  Manifold m; m.form = Manifold::Form::members; m.by = partition; m.kind = ManifoldKind::members; m.role = Role::time_tiles;
  return make_ref(std::move(m));
}
ManifoldRef members(const std::string& partition, const ManifoldRef& of, Role role, const std::string& label, bool irregular) {
  Manifold m; m.form = Manifold::Form::members; m.by = partition; m.kind = ManifoldKind::members; m.role = role; m.of = of; m.label = label; m.irregular = irregular;
  return make_ref(std::move(m));
}
ManifoldRef members_of_port(const std::string& port) { return members("$" + port); }
ManifoldRef derived(const std::string& relation, const ManifoldRef& of) {
  Manifold m; m.form = Manifold::Form::derived; m.key = relation; m.of = of; m.kind = of ? of->kind : ManifoldKind::points; m.cs = of ? of->cs : std::nullopt;
  return make_ref(std::move(m));
}

bool same_cs(const std::optional<CoordinateSystem>& a, const std::optional<CoordinateSystem>& b) {
  if (!a || !b) return true;                                   // a manifold without a recorded system is not told apart by it
  return a->space == b->space && a->name == b->name && (a->unit.empty() || b->unit.empty() || a->unit == b->unit);
}
std::string print_cs(const std::optional<CoordinateSystem>& c) { return c ? c->space + "/" + c->name + (c->unit.empty() ? "" : " (" + c->unit + ")") : std::string("none"); }

bool same(const ManifoldRef& a, const ManifoldRef& b) {
  if (a == b) return true;
  if (!a || !b) return false;
  if (a->form != b->form) return false;
  switch (a->form) {
    // the coordinate system is part of the manifold's type (T5): one manifold in two systems is two types, and no transition is inserted between them
    case Manifold::Form::id:
    case Manifold::Form::made: return a->key == b->key && a->label == b->label && same_cs(a->cs, b->cs);      // `sensors(rest)` and `time(rest)` are two manifolds of one recording
    case Manifold::Form::anonymous: return a->key == b->key && same_cs(a->cs, b->cs);
    case Manifold::Form::variable: return a->key == b->key;
    case Manifold::Form::restricted: return a->by == b->by && same(a->of, b->of);
    case Manifold::Form::members: return a->by == b->by;
    case Manifold::Form::derived: return a->key == b->key && same(a->of, b->of);
  }
  return false;
}

std::string print(const ManifoldRef& m) {
  if (!m) return "?";
  switch (m->form) {
    case Manifold::Form::id: return m->label.empty() ? m->key : m->label + "(" + m->key + ")";
    case Manifold::Form::anonymous: return "#" + m->key;
    case Manifold::Form::variable: return m->key;
    case Manifold::Form::made: return (m->label.empty() ? std::string("made") : m->label) + "(~" + m->key + ")";
    case Manifold::Form::restricted: return print(m->of) + "|" + m->by;
    case Manifold::Form::members: return "members(" + m->by + ")";
    case Manifold::Form::derived: return m->key + "(" + print(m->of) + ")";
  }
  return "?";
}

namespace {

bool is_name_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == ':' || c == '@' || c == '$' || c == '-' || c == '~' || c == '/'; }

// the balanced text after an opening parenthesis at s[i]; i moves past the closing one
std::string balanced(const std::string& s, std::size_t& i) {
  int depth = 0;
  const std::size_t start = i + 1;
  for (; i < s.size(); ++i) {
    if (s[i] == '(') ++depth;
    else if (s[i] == ')' && --depth == 0) { const std::string inner = s.substr(start, i - start); ++i; return inner; }
  }
  refuse("type: an unclosed parenthesis in '" + s + "'");
}

ManifoldRef parse_term(const std::string& s, std::size_t& i) {
  while (i < s.size() && s[i] == ' ') ++i;
  if (i >= s.size()) refuse("type: a manifold was expected in '" + s + "'");
  ManifoldRef m;
  if (s[i] == '#') {
    std::size_t j = ++i;
    while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
    m = anonymous(std::stoi(s.substr(i, j - i)));
    i = j;
  } else {
    std::size_t j = i;
    while (j < s.size() && is_name_char(s[j])) ++j;
    if (j == i) refuse("type: a manifold was expected at '" + s.substr(i) + "'");
    const std::string name = s.substr(i, j - i);
    i = j;
    if (i < s.size() && s[i] == '(') {
      const std::string inner = balanced(s, i);
      if (name == "members") m = members(inner);
      else if (inner.find('(') != std::string::npos) {
        std::size_t k = 0;
        m = derived(name, parse_term(inner, k));
      } else if (!inner.empty() && inner[0] == '~') {
        const Role r = role_of_label(name);
        m = made(name, inner.substr(1), r == Role::none ? ManifoldKind::points : ManifoldKind::line, r);
      } else {
        const Role r = role_of_label(name);
        m = manifold_id(name, inner, r == Role::none ? ManifoldKind::points : ManifoldKind::line, r);
      }
    } else {
      m = var(name);
    }
  }
  if (i < s.size() && s[i] == '|') {
    std::size_t j = ++i;
    int depth = 0;
    for (; j < s.size(); ++j) {
      if (s[j] == '(') ++depth;
      else if (s[j] == ')') --depth;
      else if (s[j] == ' ' && depth == 0) break;
    }
    m = restricted(m, s.substr(i, j - i));
    i = j;
  }
  return m;
}

}  // namespace

ManifoldRef parse_manifold(const std::string& text) {
  std::size_t i = 0;
  ManifoldRef m = parse_term(text, i);
  while (i < text.size() && text[i] == ' ') ++i;
  if (i != text.size()) refuse("type: unexpected '" + text.substr(i) + "' after the manifold in '" + text + "'");
  return m;
}

// ───────────────────────────────────────────── domains, resolutions ─────────────────────────────────────────────

const Manifold* Domain::streamed() const {
  if (factors.empty()) return nullptr;
  const Manifold& l = *factors.back();
  return l.role == Role::time || l.role == Role::frames ? &l : nullptr;
}
bool PortType::per_block() const {
  if (streamed() || axis != Axis::none || domain.factors.size() < 2) return false;
  const Manifold& l = *domain.factors.back();
  return l.form == Manifold::Form::made && l.kind == ManifoldKind::line && l.role == Role::frequency;
}
bool same(const Domain& a, const Domain& b) {
  if (a.cells != b.cells || a.factors.size() != b.factors.size()) return false;
  for (std::size_t i = 0; i < a.factors.size(); ++i)
    if (!same(a.factors[i], b.factors[i])) return false;
  return true;
}
std::string print(const Domain& d) {
  std::string s;
  for (std::size_t i = 0; i < d.factors.size(); ++i) s += (i ? " × " : "") + print(d.factors[i]);
  if (!d.cells.empty() && d.cells != "0") s += " {cells " + d.cells + "}";
  return s;
}

namespace {

std::string levels_text(const Resolution& r) {
  std::string s = r.ladder + "@";
  if (r.levels.size() == 1) return s + std::to_string(r.levels[0]);
  s += "[";
  for (std::size_t i = 0; i < r.levels.size(); ++i) s += (i ? "," : "") + std::to_string(r.levels[i]);
  return s + "]";
}
std::string band_text(const Resolution& r) {
  std::string s = "band " + number((*r.band)[0]) + "-" + number((*r.band)[1]) + " Hz";
  if (!r.band_shape.empty()) s += " " + r.band_shape;
  return s;
}
std::string level_text(const Resolution& r) {
  std::string s = "level " + levels_text(r);
  if (!r.spatial_shape.empty()) s += " " + r.spatial_shape;
  return s;
}

std::vector<std::string> split(const std::string& s, const std::string& sep) {
  std::vector<std::string> v;
  std::size_t a = 0;
  for (;;) {
    const std::size_t b = s.find(sep, a);
    if (b == std::string::npos) { v.push_back(trim(s.substr(a))); break; }
    v.push_back(trim(s.substr(a, b - a)));
    a = b + sep.size();
  }
  return v;
}
bool starts(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }

bool parse_band(const std::string& part, Resolution& r) {   // "band 8-12 Hz [shape]"
  if (!starts(part, "band ")) return false;
  std::istringstream in(part.substr(5));
  std::string range, hz, shape;
  in >> range >> hz >> shape;
  const std::size_t dash = range.find('-', 1);
  if (dash == std::string::npos) refuse("type: band '" + part + "': lo-hi Hz");
  r.band = std::array<double, 2>{std::stod(range.substr(0, dash)), std::stod(range.substr(dash + 1))};
  r.band_shape = shape;
  return true;
}
bool parse_level(const std::string& part, Resolution& r) {   // "level ladder@5 [shape]" or "ladder@[4,5]"
  if (!starts(part, "level ")) return false;
  std::istringstream in(part.substr(6));
  std::string at, shape;
  in >> at >> shape;
  const std::size_t a = at.find('@');
  if (a == std::string::npos) refuse("type: level '" + part + "': ladder@level");
  r.ladder = at.substr(0, a);
  r.levels.clear();
  std::string rest = at.substr(a + 1);
  if (!rest.empty() && rest[0] == '[') rest = rest.substr(1, rest.size() - 2);
  for (const auto& l : split(rest, ",")) if (!l.empty()) r.levels.push_back(std::stoi(l));
  r.spatial_shape = shape;
  return true;
}

std::string value_text(const PortType& t) { return std::string(t.complex_parts && t.value != ValueType::complex ? "complex " : "") + to_string(t.value); }
void parse_value(const std::string& text, PortType& t, bool from = false) {
  std::string v = text;
  bool cp = false;
  if (starts(v, "complex ")) { cp = true; v = v.substr(8); }
  static const std::pair<const char*, ValueType> names[] = {{"real", ValueType::real}, {"complex", ValueType::complex}, {"vector3", ValueType::vector3}, {"quaternion", ValueType::quaternion},
                                                              {"hermitian3", ValueType::hermitian3}, {"monogenic", ValueType::monogenic}, {"integer", ValueType::integer}};
  for (const auto& [n, vt] : names)
    if (v == n) {
      if (from) t.from_value = vt; else { t.value = vt; t.complex_parts = cp; }
      return;
    }
  refuse("type: '" + text + "' is not a value type");
}

}  // namespace

std::string print(const Resolution& r) {
  std::string s;
  if (r.band) s += band_text(r);
  if (!r.ladder.empty() || !r.levels.empty()) s += (s.empty() ? "" : "; ") + level_text(r);
  return s;
}

// ───────────────────────────────────────────── quantities ─────────────────────────────────────────────

const std::vector<std::string>& quantities() {
  static const std::vector<std::string> q = {
      // a field's kind: what the values are
      "signal", "recording", "envelope", "amplitude", "power", "relative power", "phase_current", "current", "curl_tan", "div_tan", "normal", "characteristic level", "surrogate",
      "square", "monogenic",
      // a measurement's measure, a statistic's measure
      "mean", "mean_square", "max", "min", "abs_max", "median", "envelope_max", "duration", "interval_next", "trace", "frame", "sense", "centre_vertex", "centre", "area", "peak",
      "diameter_cm", "characteristic_level", "size_cm", "relative_peak", "burst", "first_frame", "last_frame", "n_frames", "path_length", "displacement", "track", "crossings",
      "t_on", "t_off", "tau_rise", "tau_fall", "fit_baseline", "fit_height", "fit_r2", "plateau_duration", "summaries", "histogram", "home", "summary", "welch", "power_tiles",
      "n_cells", "n_tiles", "first_tile", "last_tile", "burst_ordinal", "covariance", "cross_spectrum", "psd",
      // an operator's kind
      "inverse kernel", "forward", "interpolation", "spatial band", "eigenvectors", "transition", "whitener",
      // a selection's type
      "point", "span", "spans", "indices", "kernel", "set", "ladder", "tiling"};
  return q;
}
bool is_quantity(const std::string& q) {
  const auto& v = quantities();
  return q.empty() || std::find(v.begin(), v.end(), q) != v.end();
}

// ───────────────────────────────────────────── the printed form of a type ─────────────────────────────────────────────

std::string print(const PortType& t) {
  std::vector<std::string> p;
  auto add_res = [&](const Resolution& r) {
    if (r.band) p.push_back(band_text(r));
    if (!r.ladder.empty() || !r.levels.empty()) p.push_back(level_text(r));
  };
  switch (t.cls) {
    case PortClass::measurement:
      p.push_back(print(t.domain));
      if (!t.quantity.empty()) p.push_back(t.quantity);
      add_res(t.res);
      if (t.value != ValueType::real || t.complex_parts) p.push_back(value_text(t));
      break;
    case PortClass::selection:
      p.push_back((t.sel ? t.sel->type : t.quantity) + " on " + print(t.domain));
      if (t.sel && t.sel->partition) p.push_back("partition");
      if (t.sel && !t.sel->detected_over.empty()) p.push_back("detected over " + t.sel->detected_over);
      if (t.sel && !t.sel->identity.empty()) p.push_back("as " + t.sel->identity);
      add_res(t.res);
      break;
    case PortClass::statistic: {
      p.push_back(t.quantity.empty() ? "summaries" : t.quantity);
      p.push_back("axes " + print(t.domain));
      if (t.of) {
        std::string o = "of " + t.of->quantity;
        if (t.of->res.band) o += " " + band_text(t.of->res);
        if (!t.of->res.ladder.empty() || !t.of->res.levels.empty()) o += " " + level_text(t.of->res);
        if (!t.of->on.empty()) o += " on " + print(t.of->on);
        p.push_back(o);
      }
      break;
    }
    case PortClass::operator_:
      if (t.transition) {
        p.push_back("transition " + t.transition->first.space + ": " + t.transition->first.name + " → " + t.transition->second.name);
      } else {
        p.push_back(print(t.domain) + " → " + print(t.target));
        p.push_back(std::string(to_string(t.from_value)) + " → " + value_text(t));
        if (!t.quantity.empty()) p.push_back(t.quantity);
      }
      if (t.spin != 0) p.push_back("spin " + std::to_string(t.spin));
      if (!t.gauge.empty()) p.push_back("gauge " + t.gauge);
      break;
    case PortClass::manifold: p.push_back(print(t.domain)); break;
    case PortClass::constant:
    case PortClass::partition:
      p.push_back(value_text(t));
      if (!t.quantity.empty()) p.push_back(t.quantity);
      break;
    case PortClass::field:
      p.push_back(print(t.domain));
      p.push_back(value_text(t));
      if (!t.quantity.empty()) p.push_back(t.quantity);
      add_res(t.res);
      if (t.nonlinear) p.push_back("non-linear");
      if (t.spin != 0) p.push_back("spin " + std::to_string(t.spin));
      if (!t.gauge.empty()) p.push_back("gauge " + t.gauge);
      if (!t.unit.empty()) p.push_back("unit " + t.unit);
      break;
  }
  std::string s = std::string(to_string(t.cls)) + "<";
  for (std::size_t i = 0; i < p.size(); ++i) s += (i ? "; " : "") + p[i];
  return s + ">";
}

namespace {

Domain parse_domain(const std::string& text) {
  Domain d;
  std::string s = text;
  const std::size_t c = s.find(" {cells ");
  if (c != std::string::npos) {
    d.cells = s.substr(c + 8, s.size() - c - 9);
    s = s.substr(0, c);
  }
  for (const auto& f : split(s, " × ")) d.factors.push_back(parse_manifold(f));
  return d;
}

}  // namespace

PortType parse_type(const std::string& text) {
  const std::size_t lt = text.find('<');
  if (lt == std::string::npos || text.empty() || text.back() != '>') refuse("type: '" + text + "' is not a printed type (class<...>)");
  const std::string cls = trim(text.substr(0, lt));
  const auto parts = split(text.substr(lt + 1, text.size() - lt - 2), ";");
  PortType t;
  t.typed = true;
  static const std::pair<const char*, PortClass> classes[] = {{"field", PortClass::field}, {"measurement", PortClass::measurement}, {"selection", PortClass::selection},
      {"statistic", PortClass::statistic}, {"operator", PortClass::operator_}, {"manifold", PortClass::manifold}, {"constant", PortClass::constant}, {"partition", PortClass::partition}};
  bool known = false;
  for (const auto& [n, c] : classes) if (cls == n) { t.cls = c; known = true; }
  if (!known) refuse("type: '" + cls + "' is not a class (field, measurement, selection, statistic, operator, manifold, constant)");
  auto resolution = [&](const std::string& part, Resolution& r) { return parse_band(part, r) || parse_level(part, r); };
  auto need = [&](std::size_t n) { if (parts.size() < n) refuse("type: '" + text + "' has too few members for a " + cls); };
  switch (t.cls) {
    case PortClass::field:
      need(2);
      t.domain = parse_domain(parts[0]);
      parse_value(parts[1], t);
      for (std::size_t i = 2; i < parts.size(); ++i) {
        const std::string& q = parts[i];
        if (resolution(q, t.res)) continue;
        if (q == "non-linear") t.nonlinear = true;
        else if (starts(q, "spin ")) t.spin = std::stoi(q.substr(5));
        else if (starts(q, "gauge ")) t.gauge = q.substr(6);
        else if (starts(q, "unit ")) t.unit = q.substr(5);
        else t.quantity = q;
      }
      break;
    case PortClass::measurement:
      need(1);
      t.domain = parse_domain(parts[0]);
      for (std::size_t i = 1; i < parts.size(); ++i) {
        const std::string& q = parts[i];
        if (resolution(q, t.res)) continue;
        if (q == "real" || q == "complex" || starts(q, "complex ") || q == "vector3" || q == "quaternion" || q == "hermitian3" || q == "monogenic" || q == "integer") parse_value(q, t);
        else t.quantity = q;
      }
      break;
    case PortClass::selection: {
      need(1);
      SelectionShape sh;
      const std::size_t on = parts[0].find(" on ");
      if (on == std::string::npos) refuse("type: a selection is printed 'TYPE on DOMAIN'");
      sh.type = parts[0].substr(0, on);
      t.quantity = sh.type;
      t.domain = parse_domain(parts[0].substr(on + 4));
      for (std::size_t i = 1; i < parts.size(); ++i) {
        const std::string& q = parts[i];
        if (resolution(q, t.res)) continue;
        if (q == "partition") sh.partition = true;
        else if (starts(q, "detected over ")) sh.detected_over = q.substr(14);
        else if (starts(q, "as ")) sh.identity = q.substr(3);
        else refuse("type: '" + q + "' is not a member of a selection");
      }
      t.sel = sh;
      break;
    }
    case PortClass::statistic:
      need(2);
      t.quantity = parts[0];
      if (!starts(parts[1], "axes ")) refuse("type: a statistic's second member is 'axes DOMAIN'");
      t.domain = parse_domain(parts[1].substr(5));
      if (parts.size() > 2) {
        if (!starts(parts[2], "of ")) refuse("type: '" + parts[2] + "' is not 'of QUANTITY ...'");
        OfType o;
        std::string s = parts[2].substr(3);
        auto cut = [&](const std::string& key) { return s.find(" " + key + " "); };
        std::size_t on = cut("on"), lv = cut("level"), bd = cut("band");
        const std::size_t end = std::min({on, lv, bd, s.size()});
        o.quantity = trim(s.substr(0, end));
        if (bd != std::string::npos) parse_band(trim(s.substr(bd + 1, std::min(std::min(lv, on), s.size()) - bd - 1)), o.res);
        if (lv != std::string::npos) parse_level(trim(s.substr(lv + 1, std::min(on, s.size()) - lv - 1)), o.res);
        if (on != std::string::npos) o.on = parse_domain(trim(s.substr(on + 4)));
        t.of = o;
      }
      break;
    case PortClass::operator_: {
      need(1);
      if (starts(parts[0], "transition ")) {
        const std::string rest = parts[0].substr(11);
        const std::size_t colon = rest.find(": "), arrow = rest.find(" → ");
        if (colon == std::string::npos || arrow == std::string::npos) refuse("type: a transition is printed 'transition SPACE: FROM → TO'");
        const std::string space = rest.substr(0, colon);
        t.transition = std::make_pair(CoordinateSystem{space, rest.substr(colon + 2, arrow - colon - 2), ""}, CoordinateSystem{space, rest.substr(arrow + 5), ""});
        t.quantity = "transition";
      } else {
        need(2);
        const std::size_t arrow = parts[0].find(" → ");
        if (arrow == std::string::npos) refuse("type: an operator is printed 'FROM → TO; VALUE → VALUE; KIND'");
        t.domain = parse_domain(parts[0].substr(0, arrow));
        t.target = parse_domain(parts[0].substr(arrow + 5));
        const std::size_t va = parts[1].find(" → ");
        if (va == std::string::npos) refuse("type: an operator's second member is 'VALUE → VALUE'");
        parse_value(parts[1].substr(0, va), t, true);
        parse_value(parts[1].substr(va + 5), t);
      }
      for (std::size_t i = t.transition ? 1 : 2; i < parts.size(); ++i) {
        const std::string& q = parts[i];
        if (starts(q, "spin ")) t.spin = std::stoi(q.substr(5));
        else if (starts(q, "gauge ")) t.gauge = q.substr(6);
        else t.quantity = q;
      }
      break;
    }
    case PortClass::manifold:
      need(1);
      t.domain = parse_domain(parts[0]);
      break;
    case PortClass::constant:
    case PortClass::partition:
      need(1);
      parse_value(parts[0], t);
      if (parts.size() > 1) t.quantity = parts[1];
      break;
  }
  return t;
}

// ───────────────────────────────────────────── unification ─────────────────────────────────────────────

namespace {

// one manifold has one size: two references to the same stored (or anonymous) manifold whose vertex counts are both known and differ are refused, never unified
bool sizes_conflict(const ManifoldRef& a, const ManifoldRef& b, std::string& why) {
  if (!a || !b || a->n <= 0 || b->n <= 0 || a->n == b->n) return false;
  if (a->form != Manifold::Form::id && a->form != Manifold::Form::anonymous) return false;
  why = print(b) + " has " + std::to_string(b->n) + " vertices where " + print(a) + " has " + std::to_string(a->n) + ": one manifold cannot have two sizes";
  return true;
}

bool role_allows(Role want, const Manifold& m) {
  if (want == Role::none) return true;
  if (want == Role::time) return m.role == Role::time;
  if (want == Role::frames) return m.role == Role::time || m.role == Role::frames;
  return m.role == want;
}

}  // namespace

bool unify(const ManifoldRef& pattern, const ManifoldRef& actual, Bindings& b, std::string& why) {
  if (!pattern || !actual) { why = "a manifold is missing"; return false; }
  if (pattern->form == Manifold::Form::variable) {
    if (!role_allows(pattern->role, *actual)) {
      why = std::string(pattern->role == Role::time ? "a time line" : "a time line or a frame line") + " was expected where " + print(actual) + " lies";
      return false;
    }
    if (pattern->kinded && actual->kind != pattern->kind) {           // a variable restricted to a kind refuses another kind where it meets it, not at a later port
      why = print(actual) + " is a " + to_string(actual->kind) + "; this node reads one on a " + to_string(pattern->kind) + " (" + pattern->key + ")";
      return false;
    }
    auto it = b.vars.find(pattern->key);
    if (it == b.vars.end()) { b.vars[pattern->key] = actual; return true; }
    const ManifoldRef& bound = it->second;
    if (same(bound, actual)) return !sizes_conflict(bound, actual, why);
    if (bound->key == actual->key && bound->label == actual->label && !same_cs(bound->cs, actual->cs)) {
      why = print(actual) + " is the manifold " + print(bound) + " in another coordinate system (" + print_cs(actual->cs) + ", not " + print_cs(bound->cs) +
            "); no transition is applied by the checker: add the node `meeg.positions_transition` with the transition operator " + print_cs(actual->cs) + " to " + print_cs(bound->cs);
      return false;
    }
    if (pattern->subsume) {
      // the one implicit move: a source restricted to S accepts the whole manifold it is of (the kernel reads its channels of the recording) and a restriction to the same S
      if (actual->form == Manifold::Form::restricted && same(bound, actual->of)) return true;
      if (actual->form == Manifold::Form::restricted && actual->of && bound->key == actual->of->key && bound->label == actual->of->label && !same_cs(bound->cs, actual->of->cs)) {
        why = print(actual) + " is a selection of the manifold " + print(bound) + " in another coordinate system (" + print_cs(actual->of->cs) + ", not " + print_cs(bound->cs) +
              "); no transition is applied by the checker: add the node `meeg.positions_transition` with the transition operator " + print_cs(actual->of->cs) + " to " + print_cs(bound->cs);
        return false;
      }
      if (actual->form == Manifold::Form::restricted && bound->form == Manifold::Form::restricted && same(bound->of, actual->of)) {
        why = "it reads " + print(actual) + "; the value lies on " + print(bound) + " (another selection of the same manifold)";
        return false;
      }
    }
    why = print(actual) + " is not " + print(bound) + " (variable " + pattern->key + " is already bound to it)";
    return false;
  }
  if (pattern->form != actual->form) { why = print(actual) + " is not " + print(pattern); return false; }
  switch (pattern->form) {
    case Manifold::Form::restricted:
      if (!pattern->by.empty() && pattern->by != actual->by) { why = print(actual) + " is not " + print(pattern); return false; }
      return unify(pattern->of, actual->of, b, why);
    case Manifold::Form::derived:
      if (pattern->key != actual->key) { why = print(actual) + " is not " + print(pattern); return false; }
      return unify(pattern->of, actual->of, b, why);
    default:
      if (!same(pattern, actual)) {
        why = print(actual) + " is not " + print(pattern);
        if (pattern->key == actual->key && pattern->label == actual->label)
          why += " (the same manifold in another coordinate system: " + print_cs(actual->cs) + ", not " + print_cs(pattern->cs) + "; no transition is applied by the checker: add the node `meeg.positions_transition` with the transition operator)";
        return false;
      }
      return !sizes_conflict(pattern, actual, why);
  }
}

bool unify(const Domain& pattern, const Domain& actual, Bindings& b, std::string& why) {
  const auto& pf = pattern.factors;
  const auto& af = actual.factors;
  std::size_t rest = pf.size();
  for (std::size_t i = 0; i < pf.size(); ++i)
    if (pf[i]->form == Manifold::Form::variable && pf[i]->rest) rest = i;
  const std::size_t fixed = rest == pf.size() ? pf.size() : pf.size() - 1;
  if (rest == pf.size() ? af.size() != fixed : af.size() < fixed) {
    why = "the domain " + print(actual) + " has " + std::to_string(af.size()) + " factor(s); " + print(pattern) + " wants " + (rest == pf.size() ? "" : "at least ") + std::to_string(fixed);
    return false;
  }
  for (std::size_t i = 0; i < pf.size(); ++i) {
    if (i == rest) continue;
    const std::size_t j = i < rest ? i : af.size() - (pf.size() - i);
    if (!unify(pf[i], af[j], b, why)) return false;
  }
  if (rest != pf.size()) {
    std::vector<ManifoldRef> mid(af.begin() + static_cast<std::ptrdiff_t>(rest), af.end() - static_cast<std::ptrdiff_t>(pf.size() - rest - 1));
    auto it = b.rests.find(pf[rest]->key);
    if (it == b.rests.end()) b.rests[pf[rest]->key] = mid;
    else {
      Domain x, y; x.factors = it->second; y.factors = mid;
      if (!same(x, y)) { why = "the other factors " + print(y) + " are not " + print(x) + " (" + pf[rest]->key + " is already bound)"; return false; }
    }
  }
  if (!pattern.cells.empty() && pattern.cells != actual.cells) { why = "the cells are " + actual.cells + "; " + pattern.cells + " is wanted"; return false; }
  return true;
}

ManifoldRef instantiate(const ManifoldRef& p, const Bindings& b, const std::map<std::string, std::string>& ports) {
  if (!p) return p;
  switch (p->form) {
    case Manifold::Form::variable: {
      const auto it = b.vars.find(p->key);
      if (it == b.vars.end()) refuse("the variable " + p->key + " is not bound by any input of the signature");
      return it->second;
    }
    case Manifold::Form::members:
      if (!p->by.empty() && p->by[0] == '$') {
        const auto it = ports.find(p->by.substr(1));
        if (it == ports.end()) refuse("members of port '" + p->by.substr(1) + "': that port is not read");
        return members(it->second);
      }
      return p;
    case Manifold::Form::restricted: return restricted(instantiate(p->of, b, ports), p->by);
    case Manifold::Form::derived: return derived(p->key, instantiate(p->of, b, ports));
    default: return p;
  }
}

Domain instantiate(const Domain& p, const Bindings& b, const std::map<std::string, std::string>& ports) {
  Domain d;
  d.cells = p.cells;
  for (const auto& f : p.factors) {
    if (f->form == Manifold::Form::variable && f->rest) {
      const auto it = b.rests.find(f->key);
      if (it == b.rests.end()) refuse("the variable " + f->key + " is not bound by any input of the signature");
      d.factors.insert(d.factors.end(), it->second.begin(), it->second.end());
    } else {
      d.factors.push_back(instantiate(f, b, ports));
    }
  }
  return d;
}

}  // namespace nxr::graph
