// The exporters' shared reading of a cortical surface and its hemispheres: see cortex.h.
#include "cortex.h"

#include <algorithm>
#include <cmath>

#include "nxr/io/errors.h"

namespace nxr::exporters::detail {

namespace code = nxr::io::code;
using nxr::io::ArrayReader;
using nxr::io::Error;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// "left" / "right" (as the schema's `label` says, or L / R)
std::string hemisphere_of(const std::string& label) {
  const std::string l = lower(label);
  if (l == "left" || l == "l" || l == "lh") return "L";
  if (l == "right" || l == "r" || l == "rh") return "R";
  return "";
}

}  // namespace

std::vector<std::int32_t> read_int32(const ArrayReader& a) {
  const std::vector<double> v = widen(a.read(), a.dtype());
  std::vector<std::int32_t> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (v[i] != std::floor(v[i]) || std::abs(v[i]) > 2147483647.0) refuse("an integer array holds the value " + std::to_string(v[i]) + ", which an int32 cannot");
    out[i] = static_cast<std::int32_t>(v[i]);
  }
  return out;
}

// (documented in cortex.h) a hemisphere component of a glued cortex reads its slice of the union's arrays (`from_index`)
Surface load_surface(const nxr::io::SubjectPackage& pkg, const task::Subject& subject, const task::Node& m) {
  if (!m.type || *m.type != "surface") refuse("the manifold '" + m.name + "' is not a surface (it is a " + (m.type ? *m.type : std::string("manifold of no type")) + "): an export of a cortical field lies on a cortical surface");
  Surface s;
  s.node = m;
  s.n = m.n_vertices.value_or(0);
  const task::Geometry g = subject.geometry(m.id);
  if (!g.positions_path) refuse("the surface '" + m.name + "' has no stored positions");
  s.coordinate_unit = g.coordinate_unit.value_or("");
  const ArrayReader pos = ArrayReader::open(pkg.store(), *g.positions_path);
  if (pos.shape().size() != 2 || pos.shape()[1] != 3) refuse("the positions of '" + m.name + "' are not [vertices x 3]");
  const std::int64_t total = pos.shape()[0];
  std::int64_t from = 0;
  const bool sliced = total != s.n;
  if (sliced) {
    const Json f = m.fact("from_index");
    if (f.is_null()) refuse("the surface '" + m.name + "' has " + std::to_string(s.n) + " vertices but its positions hold " + std::to_string(total));
    from = f.get<std::int64_t>();
    if (from < 0 || from + s.n > total) refuse("the surface '" + m.name + "' lies outside its positions");
  }
  s.xyz = widen(pos.read({from, 0}, {from + s.n, 3}), pos.dtype());
  const std::string* cells = nullptr;
  for (const auto& c : g.cells) if (c.first == 2) cells = &c.second;
  if (!cells) refuse("the surface '" + m.name + "' has no stored triangles");
  const ArrayReader fa = ArrayReader::open(pkg.store(), *cells);
  if (fa.shape().size() != 2 || fa.shape()[1] != 3) refuse("the triangles of '" + m.name + "' are not [triangles x 3]");
  const std::vector<std::int32_t> all = read_int32(fa);
  if (!sliced) s.faces = all;
  else
    for (std::size_t t = 0; t + 2 < all.size(); t += 3) {
      const bool in = all[t] >= from && all[t] < from + s.n && all[t + 1] >= from && all[t + 1] < from + s.n && all[t + 2] >= from && all[t + 2] < from + s.n;
      if (in) for (int c = 0; c < 3; ++c) s.faces.push_back(static_cast<std::int32_t>(all[t + c] - from));
    }
  const Json label = m.fact("label");
  if (label.is_string()) {
    s.hemi = hemisphere_of(label.get<std::string>());
    s.structure = s.hemi == "L" ? "CortexLeft" : s.hemi == "R" ? "CortexRight" : "";
  }
  return s;
}

std::vector<Surface> hemispheres_of(const nxr::io::SubjectPackage& pkg, const task::Subject& subject, const task::Node& whole) {
  std::vector<std::pair<std::int64_t, task::Node>> comps;
  task::Want w;
  w.cls = task::Cls::manifold;
  w.type = "surface";
  for (const auto& m : subject.find_all(w)) {
    const Json of = m.fact("component_of_id");
    if (of.is_string() && of.get<std::string>() == whole.id) comps.emplace_back(m.fact("component_ordinal").is_number() ? m.fact("component_ordinal").get<std::int64_t>() : 0, m);
  }
  std::sort(comps.begin(), comps.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  if (comps.empty()) refuse("the surface '" + whole.name + "' has no hemispheres as components: --hemispheres needs a glued cortex that stores them (one file per hemisphere)");
  std::vector<Surface> out;
  for (const auto& c : comps) {
    Surface s = load_surface(pkg, subject, c.second);
    if (s.hemi.empty()) refuse("the component '" + c.second.name + "' of '" + whole.name + "' is not labelled left or right, so its file cannot be named");
    out.push_back(std::move(s));
  }
  return out;
}


}  // namespace nxr::exporters::detail
