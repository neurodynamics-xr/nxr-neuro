// nxr.io.mesh — OBJ, PLY, STL read without a dependency.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

#include "nxr/io/surface.h"

namespace fs = std::filesystem;

namespace nxr::io {

namespace {
Error value_error(const std::string& m) { return Error(code::invalid_argument, m); }

std::string read_all(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// faces of any size as triangles: a fan from each face's first vertex
void fan(const std::vector<std::vector<std::int64_t>>& polys, const std::string& file, std::vector<std::int64_t>& out) {
  for (const auto& poly : polys) {
    if (poly.size() < 3) throw value_error(file + ": a face has " + std::to_string(poly.size()) + " vertices (a triangle needs 3)");
    for (std::size_t k = 1; k + 1 < poly.size(); ++k) { out.push_back(poly[0]); out.push_back(poly[k]); out.push_back(poly[k + 1]); }
  }
}

Mesh checked(const std::string& file, const std::vector<double>& v, const std::vector<std::int64_t>& f) {
  if (v.empty() || v.size() % 3) throw value_error(file + ": vertices are not [n × 3]");
  if (f.empty() || f.size() % 3) throw value_error(file + ": no triangles (" + std::to_string(f.size() / 3) + ")");
  const std::int64_t nv = static_cast<std::int64_t>(v.size() / 3);
  for (auto i : f) if (i < 0 || i >= nv) throw value_error(file + ": a face indexes a vertex outside 0.." + std::to_string(nv - 1));
  for (double x : v) if (!std::isfinite(x)) throw value_error(file + ": a vertex is not finite");
  Mesh m;
  m.vertices = v;
  m.faces.assign(f.begin(), f.end());
  return m;
}

Mesh read_obj(const fs::path& p) {
  std::ifstream fh(p);
  std::vector<double> verts;
  std::vector<std::vector<std::int64_t>> polys;
  std::string line;
  while (std::getline(fh, line)) {
    if (line.rfind("v ", 0) == 0) {
      std::istringstream ss(line.substr(2));
      double x, y, z;
      if (!(ss >> x >> y >> z)) throw value_error("could not convert string to float: '" + line + "'");
      verts.push_back(x); verts.push_back(y); verts.push_back(z);
    } else if (line.rfind("f ", 0) == 0) {
      std::istringstream ss(line.substr(2));
      std::string tok;
      std::vector<std::int64_t> idx;
      while (ss >> tok) {
        const std::int64_t i = std::stoll(tok.substr(0, tok.find('/')));
        idx.push_back(i > 0 ? i - 1 : static_cast<std::int64_t>(verts.size() / 3) + i);  // negative: relative to the vertices read so far
      }
      polys.push_back(idx);
    }
  }
  std::vector<std::int64_t> faces;
  fan(polys, p.filename().string(), faces);
  return checked(p.filename().string(), verts, faces);
}

// ── PLY ──
struct PlyProp { std::string name; bool list = false; std::string ctype, itype; };  // ctype: scalar type, or the count's type for a list
struct PlyElement { std::string name; std::int64_t n = 0; std::vector<PlyProp> props; };

std::string ply_type(const std::string& t) {
  static const std::map<std::string, std::string> m = {{"char", "i1"}, {"int8", "i1"}, {"uchar", "u1"}, {"uint8", "u1"}, {"short", "i2"}, {"int16", "i2"}, {"ushort", "u2"}, {"uint16", "u2"},
                                                       {"int", "i4"}, {"int32", "i4"}, {"uint", "u4"}, {"uint32", "u4"}, {"float", "f4"}, {"float32", "f4"}, {"double", "f8"}, {"float64", "f8"}};
  auto it = m.find(t);
  if (it == m.end()) throw value_error("PLY property type '" + t + "' not read");
  return it->second;
}
std::size_t ply_size(const std::string& t) { return static_cast<std::size_t>(t[1] - '0'); }

double ply_get(const std::uint8_t* p, const std::string& t, bool big) {
  std::uint8_t b[8];
  const std::size_t n = ply_size(t);
  for (std::size_t i = 0; i < n; ++i) b[i] = big ? p[n - 1 - i] : p[i];
  switch (t[0]) {
    case 'i': { if (n == 1) { std::int8_t x; std::memcpy(&x, b, 1); return x; } if (n == 2) { std::int16_t x; std::memcpy(&x, b, 2); return x; } std::int32_t x; std::memcpy(&x, b, 4); return x; }
    case 'u': { if (n == 1) { std::uint8_t x; std::memcpy(&x, b, 1); return x; } if (n == 2) { std::uint16_t x; std::memcpy(&x, b, 2); return x; } std::uint32_t x; std::memcpy(&x, b, 4); return x; }
    default: { if (n == 4) { float x; std::memcpy(&x, b, 4); return x; } double x; std::memcpy(&x, b, 8); return x; }
  }
}

Mesh read_ply(const fs::path& p) {
  const std::string raw = read_all(p);
  const std::string nm = p.filename().string();
  const auto end = raw.find("end_header");
  if (raw.rfind("ply", 0) != 0 || end == std::string::npos) throw value_error(nm + ": not a PLY file");
  const auto nl = raw.find('\n', end);
  std::istringstream hs(raw.substr(0, nl));
  std::string fmt, ln;
  std::vector<PlyElement> els;
  std::getline(hs, ln);
  while (std::getline(hs, ln)) {
    ln.erase(std::remove(ln.begin(), ln.end(), '\r'), ln.end());
    std::istringstream ws(ln);
    std::vector<std::string> w;
    for (std::string t; ws >> t;) w.push_back(t);
    if (w.empty() || w[0] == "comment" || w[0] == "obj_info" || w[0] == "end_header") continue;
    if (w[0] == "format") fmt = w[1];
    else if (w[0] == "element") els.push_back({w[1], std::stoll(w[2]), {}});
    else if (w[0] == "property") {
      if (w[1] == "list") els.back().props.push_back({w[4], true, ply_type(w[2]), ply_type(w[3])});
      else els.back().props.push_back({w[2], false, ply_type(w[1]), ""});
    }
  }
  if (fmt != "ascii" && fmt != "binary_little_endian" && fmt != "binary_big_endian") throw value_error(nm + ": PLY format '" + fmt + "' not read");
  const bool ascii = fmt == "ascii", big = fmt == "binary_big_endian";
  std::size_t pos = nl + 1;
  std::vector<std::string> tok;
  std::size_t tp = 0;
  if (ascii) { std::istringstream ts(raw.substr(pos)); for (std::string t; ts >> t;) tok.push_back(t); }
  auto next = [&](const std::string& type) -> double {
    if (ascii) return std::stod(tok.at(tp++));
    if (pos + ply_size(type) > raw.size()) throw value_error(nm + ": PLY data is truncated");
    const double v = ply_get(reinterpret_cast<const std::uint8_t*>(raw.data()) + pos, type, big);
    pos += ply_size(type);
    return v;
  };
  std::vector<double> vertices;
  std::vector<std::vector<std::int64_t>> polys;
  bool have_v = false, have_f = false;
  for (const auto& el : els) {
    int xi = -1, yi = -1, zi = -1;
    for (std::size_t i = 0; i < el.props.size(); ++i) { if (el.props[i].name == "x") xi = static_cast<int>(i); if (el.props[i].name == "y") yi = static_cast<int>(i); if (el.props[i].name == "z") zi = static_cast<int>(i); }
    for (std::int64_t r = 0; r < el.n; ++r) {
      std::vector<double> sc(el.props.size(), 0);
      bool took_list = false;
      std::vector<std::int64_t> poly;
      for (std::size_t i = 0; i < el.props.size(); ++i) {
        const auto& pr = el.props[i];
        if (!pr.list) { sc[i] = next(pr.ctype); continue; }
        const std::int64_t k = static_cast<std::int64_t>(next(pr.ctype));
        for (std::int64_t j = 0; j < k; ++j) { const double x = next(pr.itype); if (el.name == "face" && !took_list) poly.push_back(static_cast<std::int64_t>(x)); }
        if (el.name == "face") took_list = true;
      }
      if (el.name == "vertex" && xi >= 0 && yi >= 0 && zi >= 0) { vertices.push_back(sc[xi]); vertices.push_back(sc[yi]); vertices.push_back(sc[zi]); have_v = true; }
      if (el.name == "face") { polys.push_back(poly); have_f = true; }
    }
  }
  if (!have_v || !have_f) throw value_error(nm + ": PLY without vertex and face elements");
  std::vector<std::int64_t> faces;
  fan(polys, nm, faces);
  return checked(nm, vertices, faces);
}

// ── STL ──
Mesh read_stl(const fs::path& p) {
  const std::string raw = read_all(p);
  const std::string nm = p.filename().string();
  std::vector<std::array<float, 3>> corners;
  std::uint32_t n = 0;
  const bool binary = raw.size() >= 84 && (std::memcpy(&n, raw.data() + 80, 4), raw.size() == 84 + 50ull * n);
  if (binary) {
    for (std::uint32_t i = 0; i < n; ++i) {
      const char* t = raw.data() + 84 + 50ull * i + 12;  // skip the normal
      for (int c = 0; c < 3; ++c) { std::array<float, 3> v; std::memcpy(v.data(), t + 12 * c, 12); corners.push_back(v); }
    }
  } else {
    std::size_t s = 0;
    while (s < raw.size() && std::isspace(static_cast<unsigned char>(raw[s]))) ++s;
    if (raw.compare(s, 5, "solid") != 0) throw value_error(nm + ": not an STL file");
    std::istringstream ss(raw);
    for (std::string t; ss >> t;) {
      if (t != "vertex") continue;
      double x, y, z;
      ss >> x >> y >> z;
      corners.push_back({static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)});
    }
    if (corners.empty()) throw value_error(nm + ": not an STL file");
  }
  if (corners.size() % 3) throw value_error(nm + ": STL vertices are not in threes");
  // STL repeats every shared corner: merged, exactly (np.unique's order: lexicographic)
  std::map<std::array<float, 3>, std::int64_t> uniq;
  for (const auto& c : corners) uniq.emplace(c, 0);
  std::int64_t k = 0;
  std::vector<double> vertices;
  for (auto& [c, i] : uniq) { i = k++; vertices.push_back(c[0]); vertices.push_back(c[1]); vertices.push_back(c[2]); }
  std::vector<std::int64_t> faces;
  for (const auto& c : corners) faces.push_back(uniq.at(c));
  return checked(nm, vertices, faces);
}
}  // namespace

std::string mesh_name(const fs::path& path) {
  const std::string n = path.filename().string();
  for (const char* suf : {".surf.gii", ".obj", ".ply", ".stl", ".gii"}) {
    const std::string s = suf;
    if (n.size() >= s.size() && lower(n.substr(n.size() - s.size())) == s) return n.substr(0, n.size() - s.size());
  }
  return n;
}

Mesh read_mesh(const fs::path& path) {
  if (!fs::is_regular_file(path)) throw Error(code::not_found, "not a file: " + path.string());
  const std::string suffix = lower(path.extension().string());
  if (suffix == ".obj") return read_obj(path);
  if (suffix == ".ply") return read_ply(path);
  if (suffix == ".stl") return read_stl(path);
  if (suffix == ".gii") throw Error(code::unsupported, path.filename().string() + ": reading a GIfTI surface needs nibabel (nxr's Python reader); nxr-io reads OBJ, PLY and STL");
  const std::string n = lower(path.filename().string());
  if (n.rfind("lh.", 0) == 0 || n.rfind("rh.", 0) == 0) return read_freesurfer_surface(path);
  throw value_error(path.filename().string() + ": not a mesh format nxr reads (OBJ, PLY, STL, GIfTI .gii, FreeSurfer lh.pial …)");
}

// FreeSurfer's triangle file (nibabel.freesurfer.read_geometry): the magic 0xFFFFFE, a "created by" line and an empty line, the vertex and face counts
// (int32, big-endian), the vertices (float32, big-endian, mm) and the faces (int32, big-endian, 0-based). What follows the faces (the volume
// geometry tags) is not read. The quadrangle formats (magic 0xFFFFFF, 0xFFFFFD) are refused.
Mesh read_freesurfer_surface(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error(code::not_found, "not a file: " + path.string());
  const std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const std::string name = path.filename().string();
  if (b.size() < 3) throw value_error(name + ": not a FreeSurfer surface (too short)");
  const std::uint32_t magic = (static_cast<std::uint32_t>(b[0]) << 16) | (static_cast<std::uint32_t>(b[1]) << 8) | b[2];
  if (magic != 0xFFFFFEu)
    throw Error(magic == 0xFFFFFFu || magic == 0xFFFFFDu ? code::unsupported : code::invalid_argument,
                name + (magic == 0xFFFFFFu || magic == 0xFFFFFDu ? ": a FreeSurfer QUADRANGLE surface; only the triangle format is read" : ": not a FreeSurfer triangle surface"));
  std::size_t i = 3;
  for (int line = 0; line < 2; ++line) {            // the "created by" line, then the empty one
    while (i < b.size() && b[i] != '\n') ++i;
    ++i;
  }
  auto be32 = [&](std::size_t at) {
    if (at + 4 > b.size()) throw value_error(name + ": truncated FreeSurfer surface");
    return (static_cast<std::uint32_t>(b[at]) << 24) | (static_cast<std::uint32_t>(b[at + 1]) << 16) | (static_cast<std::uint32_t>(b[at + 2]) << 8) | b[at + 3];
  };
  const std::int64_t nv = static_cast<std::int32_t>(be32(i)), nf = static_cast<std::int32_t>(be32(i + 4));
  i += 8;
  if (nv <= 0 || nf <= 0 || i + static_cast<std::size_t>(12 * nv + 12 * nf) > b.size()) throw value_error(name + ": truncated FreeSurfer surface");
  Mesh m;
  m.vertices.resize(static_cast<std::size_t>(3 * nv));
  for (std::size_t k = 0; k < m.vertices.size(); ++k, i += 4) {
    const std::uint32_t u = be32(i);
    float x;
    std::memcpy(&x, &u, 4);
    m.vertices[k] = static_cast<double>(x);
  }
  m.faces.resize(static_cast<std::size_t>(3 * nf));
  for (std::size_t k = 0; k < m.faces.size(); ++k, i += 4) {
    const auto v = static_cast<std::int32_t>(be32(i));
    if (v < 0 || v >= nv) throw value_error(name + ": face index " + std::to_string(v) + " outside its " + std::to_string(nv) + " vertices");
    m.faces[k] = v;
  }
  return m;
}

// nibabel.freesurfer.read_annot (see surface.h): the vertex ids are not read (the labels are in vertex order); a value not in the table is placed as
// numpy's searchsorted places it (nibabel's behaviour), past the end refused.
FsAnnot read_freesurfer_annot(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error(code::not_found, "not a file: " + path.string());
  const std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const std::string name = path.filename().string();
  std::size_t i = 0;
  auto i32 = [&]() {
    if (i + 4 > b.size()) throw value_error(name + ": truncated FreeSurfer annotation");
    const std::uint32_t u = (static_cast<std::uint32_t>(b[i]) << 24) | (static_cast<std::uint32_t>(b[i + 1]) << 16) | (static_cast<std::uint32_t>(b[i + 2]) << 8) | b[i + 3];
    i += 4;
    return static_cast<std::int32_t>(u);
  };
  auto str = [&](std::int32_t n) {                   // numpy's |S<n>: trailing NULs dropped
    if (n < 0 || i + static_cast<std::size_t>(n) > b.size()) throw value_error(name + ": truncated FreeSurfer annotation");
    std::string s(reinterpret_cast<const char*>(b.data() + i), static_cast<std::size_t>(n));
    i += static_cast<std::size_t>(n);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
  };
  FsAnnot a;
  const std::int32_t vnum = i32();
  if (vnum < 0) throw value_error(name + ": a negative vertex count");
  std::vector<std::int32_t> values(static_cast<std::size_t>(vnum));
  for (auto& v : values) { (void)i32(); v = i32(); }
  if (i32() == 0) throw value_error(name + ": Color table not found in annotation file");
  const std::int32_t n_entries = i32();
  if (n_entries > 0) {                               // the old format
    str(i32());
    a.ctab.assign(static_cast<std::size_t>(n_entries), {0, 0, 0, 0, 0});
    for (std::int32_t k = 0; k < n_entries; ++k) {
      a.names.push_back(str(i32()));
      for (int c = 0; c < 4; ++c) a.ctab[static_cast<std::size_t>(k)][static_cast<std::size_t>(c)] = i32();
    }
  } else {
    if (-n_entries != 2) throw value_error(name + ": Unrecognised .annot file version (" + std::to_string(-n_entries) + ")");
    const std::int32_t max_index = i32();
    a.ctab.assign(static_cast<std::size_t>(std::max(max_index, 0)), {0, 0, 0, 0, 0});
    str(i32());                                      // the original table's path
    const std::int32_t to_read = i32();
    for (std::int32_t k = 0; k < to_read; ++k) {
      const std::int32_t idx = i32();
      a.names.push_back(str(i32()));
      if (idx < 0 || idx >= max_index) throw value_error(name + ": colour-table index " + std::to_string(idx) + " outside its " + std::to_string(max_index) + " rows");
      for (int c = 0; c < 4; ++c) a.ctab[static_cast<std::size_t>(idx)][static_cast<std::size_t>(c)] = i32();
    }
  }
  for (auto& r : a.ctab) r[4] = r[0] + r[1] * (1 << 8) + r[2] * (1 << 16);
  // labels -> rows: ord = argsort(packed); a label's row is ord[searchsorted(packed[ord], label)]
  std::vector<std::int32_t> ord(a.ctab.size());
  for (std::size_t k = 0; k < ord.size(); ++k) ord[k] = static_cast<std::int32_t>(k);
  std::stable_sort(ord.begin(), ord.end(), [&](std::int32_t x, std::int32_t y) { return a.ctab[static_cast<std::size_t>(x)][4] < a.ctab[static_cast<std::size_t>(y)][4]; });
  std::vector<std::int32_t> sorted(ord.size());
  for (std::size_t k = 0; k < ord.size(); ++k) sorted[k] = a.ctab[static_cast<std::size_t>(ord[k])][4];
  a.labels.resize(values.size());
  for (std::size_t v = 0; v < values.size(); ++v) {
    if (values[v] == 0) { a.labels[v] = -1; continue; }
    const auto pos = static_cast<std::size_t>(std::lower_bound(sorted.begin(), sorted.end(), values[v]) - sorted.begin());
    if (pos >= ord.size()) throw value_error(name + ": vertex " + std::to_string(v) + "'s label " + std::to_string(values[v]) + " is in no colour-table row");
    a.labels[v] = ord[pos];
  }
  return a;
}

std::string sanitize_node_name(const std::string& name) {
  std::string out;
  bool run = false;
  for (unsigned char c : name) {
    const bool ok = (std::isalnum(c) && c < 128) || c == '_' || c == '.' || c == '-';
    if (ok) { out += static_cast<char>(c); run = false; }
    else if (!run) { out += '_'; run = true; }
  }
  const auto a = out.find_first_not_of("_.-");
  if (a == std::string::npos) throw value_error("name sanitizes to nothing: " + std::string("'") + name + "'");
  return out.substr(a, out.find_last_not_of("_.-") - a + 1);
}

}  // namespace nxr::io
