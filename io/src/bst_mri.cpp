// nxr::bst — nxr.convert.mri: a subject's MRI (subjectimage_*.mat) as `mri/<name>` entities.
#include "nxr/bst/mri.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <regex>
#include <set>

#include "nxr/bst/protocol.h"
#include "nxr/io/layout.h"
#include "nxr/io/surface.h"

namespace nxr::bst {

namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::Cols;
using nxr::io::SubjectWriter;
using nxr::io::view;

namespace {

[[noreturn]] void value_error(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

// repr(float): the shortest digits that read back the same, in Python's layout
std::string py_float(double x) {
  if (std::isnan(x)) return "nan";
  if (std::isinf(x)) return x < 0 ? "-inf" : "inf";
  if (x == 0) return std::signbit(x) ? "-0.0" : "0.0";
  char buf[64];
  int prec = 1;
  for (; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof buf, "%.*e", prec - 1, x);
    if (std::strtod(buf, nullptr) == x) break;
  }
  std::string s = buf;
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

// np.round(x, d).tolist() printed as Python prints a list of floats
std::string rounded_list(const std::array<double, 3>& v, int decimals) {
  const double f = std::pow(10.0, decimals);
  std::string s = "[";
  for (int i = 0; i < 3; ++i) s += (i ? ", " : "") + py_float(std::nearbyint(v[i] * f) / f);
  return s + "]";
}

std::string trim(const std::string& s) {
  const char* ws = " \t\n\r\f\v";
  const auto a = s.find_first_not_of(ws);
  return a == std::string::npos ? "" : s.substr(a, s.find_last_not_of(ws) - a + 1);
}

std::string char_text(const Value* v) { return v && v->is_char() && !v->text.empty() ? v->text[0] : ""; }

std::string np_dtype(const Value& v) {
  if (v.is_logical) return "bool";
  switch (v.cls) {
    case Class::Double: return "float64";
    case Class::Single: return "float32";
    case Class::Int8: return "int8";
    case Class::UInt8: return "uint8";
    case Class::Int16: return "int16";
    case Class::UInt16: return "uint16";
    case Class::Int32: return "int32";
    case Class::UInt32: return "uint32";
    case Class::Int64: return "int64";
    case Class::UInt64: return "uint64";
    default: return class_name(v.cls);
  }
}
bool is_integer_class(const Value& v) {
  if (v.is_logical) return false;
  switch (v.cls) {
    case Class::Int8: case Class::UInt8: case Class::Int16: case Class::UInt16: case Class::Int32: case Class::UInt32: case Class::Int64: case Class::UInt64: return true;
    default: return false;
  }
}

// a field of a struct (element 0) as numbers, when it is a nonempty numeric array: `_vec` / `_mat`
std::optional<std::vector<double>> field_numbers(const Value* st, const char* key, std::size_t expect = 0) {
  if (!st || !st->is_struct() || st->numel() < 1) return std::nullopt;
  const Value* f = st->field(key, 0);
  if (!f || !f->is_numeric()) return std::nullopt;
  std::vector<double> v = f->dims.size() <= 2 ? f->row_major() : f->doubles();
  if (expect && v.size() != expect) return std::nullopt;     // reshape(3, 3) fails
  if (v.empty()) return std::nullopt;
  return v;
}

std::array<double, 3> arr3(const std::vector<double>& v) { return {v.at(0), v.at(1), v.at(2)}; }

// the Labels cell [N x 3] of [id, name, rgb]: `_read_labels`
std::optional<MriLabels> read_labels(const Value* raw) {
  if (!raw || !raw->is_cell() || raw->numel() == 0 || raw->dims.size() != 2 || raw->dims[1] < 3) return std::nullopt;
  const std::size_t N = raw->dims[0];
  MriLabels lab;
  for (std::size_t r = 0; r < N; ++r) {
    const Value& id = raw->cell(r);
    const Value& nm = raw->cell(r + N);
    const Value& rgb = raw->cell(r + 2 * N);
    if (!id.is_numeric() || id.numel() == 0) value_error("index 0 is out of bounds for axis 0 with size 0");
    lab.ids.push_back(static_cast<std::int32_t>(r));      // replaced by the codes 0..N-1 in to_codes
    lab.source_ids.push_back(static_cast<std::int32_t>(static_cast<std::int64_t>(id.doubles().at(0))));
    lab.names.push_back(nm.is_char() ? char_text(&nm) : "");
    const std::vector<double> c = rgb.is_numeric() ? rgb.doubles() : std::vector<double>{};
    Rgb col{0, 0, 0};
    for (int i = 0; i < 3 && i < static_cast<int>(c.size()); ++i) col[i] = static_cast<int>(static_cast<std::uint8_t>(static_cast<std::int64_t>(c[i])));
    lab.colors.push_back(col);
  }
  return lab;
}

// Rewrite an atlas's voxels as CODES — the row of its label table (D123). `values` are the cube's label values (C order).
std::vector<std::uint16_t> to_codes(const std::vector<std::int64_t>& values, MriLabels& lab) {
  std::vector<std::int64_t> src(lab.source_ids.begin(), lab.source_ids.end());
  std::set<std::int64_t> present(values.begin(), values.end());
  std::vector<std::int64_t> missing;
  for (auto v : present) if (std::find(src.begin(), src.end(), v) == src.end()) missing.push_back(v);   // np.setdiff1d: sorted, unique
  for (auto v : missing) {
    src.push_back(v);
    lab.names.push_back("unlabelled " + std::to_string(v));
    lab.colors.push_back({128, 128, 128});
  }
  std::set<std::int64_t> uniq(src.begin(), src.end());
  if (uniq.size() != src.size()) {
    std::vector<std::int64_t> sorted(src);
    std::sort(sorted.begin(), sorted.end());
    std::string l = "[";
    for (std::size_t i = 0; i < sorted.size(); ++i) l += (i ? ", " : "") + std::to_string(sorted[i]);
    value_error("the label table repeats a value: " + l + "]");
  }
  std::vector<std::size_t> order(src.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return src[a] < src[b]; });
  std::vector<std::int64_t> sorted_src(order.size());
  for (std::size_t i = 0; i < order.size(); ++i) sorted_src[i] = src[order[i]];
  std::vector<std::uint16_t> codes(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto pos = std::lower_bound(sorted_src.begin(), sorted_src.end(), values[i]) - sorted_src.begin();
    codes[i] = static_cast<std::uint16_t>(order[static_cast<std::size_t>(pos)]);
  }
  lab.source_ids.assign(src.begin(), src.end());
  lab.ids.resize(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) lab.ids[i] = static_cast<std::int32_t>(i);
  lab.unlabelled = missing;
  return codes;
}

// the voxel -> SCS transform as JSON: {"r": [9], "t": [3]}
Json frame_key(const std::map<std::string, std::vector<double>>& scs) {
  return Json{{"r", scs.at("R")}, {"t", scs.at("T")}};
}

// `scs_mm = R @ (voxel * voxsize) + T` as ONE 3x4 matrix over voxel INDEX, row after row
std::vector<double> affine(const std::map<std::string, std::vector<double>>& tf, const std::array<double, 3>& voxsize) {
  const auto& R = tf.at("R");
  const auto& T = tf.at("T");
  std::vector<double> m(12);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) m[r * 4 + c] = R[r * 3 + c] * voxsize[c];
    m[r * 4 + 3] = T[r];
  }
  return m;
}

std::optional<nxr::io::Row> anatomical_volume(SubjectWriter& sub) {
  for (const auto& r : sub.db().all("SELECT * FROM manifold WHERE subject_id = ? AND type = 'volume' ORDER BY path", {sub.id()})) {
    const std::string p = nxr::io::text(r, "path");
    if (p.rfind("mri/", 0) == 0 && p.size() >= 7 && p.compare(p.size() - 7, 7, "_volume") == 0) return r;
  }
  return std::nullopt;
}

std::string py_int_list(const std::vector<std::int64_t>& v) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
  return s + "]";
}

}  // namespace

std::string mri_name(const fs::path& file, const std::string& comment) {
  std::string stem = file.stem().string();
  stem = std::regex_replace(stem, std::regex("^subjectimage_"), "");
  stem = std::regex_replace(stem, std::regex("_?volatlas$"), "");
  stem = std::regex_replace(stem, std::regex("^MRI_"), "");
  if (std::regex_match(stem, std::regex("\\d{6}_\\d{4}(_\\d+)?")) && !trim(comment).empty()) return nxr::io::sanitize_node_name(comment);
  return !stem.empty() ? nxr::io::sanitize_node_name(stem) : "MRI";
}

std::array<double, 3> to_scs(const std::array<double, 3>& voxel, const std::map<std::string, std::vector<double>>& scs, const std::array<double, 3>& voxsize) {
  const auto& R = scs.at("R");
  const auto& T = scs.at("T");
  std::array<double, 3> v{voxel[0] * voxsize[0], voxel[1] * voxsize[1], voxel[2] * voxsize[2]}, out{};
  for (int r = 0; r < 3; ++r) out[r] = (R[r * 3] * v[0] + R[r * 3 + 1] * v[1] + R[r * 3 + 2] * v[2]) + T[r];
  return out;
}

void verify_scs(const std::map<std::string, std::vector<double>>& scs, const std::array<double, 3>& voxsize) {
  std::string missing;
  for (const char* k : {"R", "T", "NAS", "LPA", "RPA"}) if (!scs.count(k)) missing += std::string(missing.empty() ? "" : ", ") + "'" + k + "'";
  if (!missing.empty()) value_error("SCS is missing [" + missing + "] \xe2\x80\x94 cannot verify orientation, so cannot import");
  if (const auto it = scs.find("Origin"); it != scs.end()) {
    const auto o = to_scs(arr3(it->second), scs, voxsize);
    if (std::sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]) > kScsTolMm) value_error("SCS origin does not map to the SCS origin: " + rounded_list(o, 4) + " mm");
  }
  const auto nas = to_scs(arr3(scs.at("NAS")), scs, voxsize), lpa = to_scs(arr3(scs.at("LPA")), scs, voxsize), rpa = to_scs(arr3(scs.at("RPA")), scs, voxsize);
  if (nas[0] <= 0) value_error("NAS is not anterior (+x): " + rounded_list(nas, 3) + " mm");
  if (lpa[1] <= 0 || rpa[1] >= 0)
    value_error("LEFT-RIGHT ORIENTATION IS WRONG \xe2\x80\x94 LPA must map to +y and RPA to -y. Got LPA " + rounded_list(lpa, 3) + ", RPA " + rounded_list(rpa, 3) +
                " mm. Importing this would produce a mirrored brain that looks entirely correct.");
  const double worst = std::max({std::fabs(nas[2]), std::fabs(lpa[2]), std::fabs(rpa[2])});
  if (worst > kScsTolMm) {
    char b[64];
    std::snprintf(b, sizeof b, "%.4g", worst);
    value_error(std::string("the fiducials do not define the z=0 plane (worst |z| = ") + b + " mm)");
  }
}

Mri read_subjectimage(const fs::path& p) {
  const std::string fname = p.filename().string();
  std::map<std::string, Value> m;
  try {
    MatFile f = MatFile::open(p);
    if (!f.complete()) value_error(fname + ": not a readable MATLAB file");
    for (const char* n : {"Cube", "Labels", "Voxsize", "SCS", "NCS", "Comment"})
      if (auto v = f.read(n)) m.emplace(n, std::move(*v));
  } catch (const nxr::io::Error& e) {
    if (std::string(e.code()) != code::invalid_argument) throw;
    value_error(fname + ": not a readable MATLAB file");   // truncated or corrupt: the same words as nxr's, not matio's
  }
  const auto get = [&](const char* k) -> const Value* { const auto it = m.find(k); return it == m.end() ? nullptr : &it->second; };
  const Value* cube = get("Cube");
  if (!cube) value_error(fname + ": no Cube \xe2\x80\x94 not a Brainstorm subject image");
  std::vector<std::size_t> dims = cube->dims;
  if (dims.size() == 4 && dims[3] == 1) dims.pop_back();
  if (dims.size() != 3) value_error(fname + ": Cube is " + std::to_string(dims.size()) + "-D, expected 3");
  Mri img;
  for (int i = 0; i < 3; ++i) img.shape[i] = static_cast<std::int64_t>(dims[i]);
  const std::size_t ni = dims[0], nj = dims[1], nk = dims[2], total = ni * nj * nk;
  auto c_index = [&](std::size_t i, std::size_t j, std::size_t k) { return (i * nj + j) * nk + k; };
  auto cm_index = [&](std::size_t i, std::size_t j, std::size_t k) { return i + ni * (j + nj * k); };

  auto labels = read_labels(get("Labels"));
  if (labels) {
    if (!is_integer_class(*cube)) value_error(fname + ": an atlas Cube must hold integers, not " + np_dtype(*cube));
    const std::vector<std::int64_t> cm = cube->int64s();
    std::vector<std::int64_t> v(total);
    for (std::size_t i = 0; i < ni; ++i) for (std::size_t j = 0; j < nj; ++j) for (std::size_t k = 0; k < nk; ++k) v[c_index(i, j, k)] = cm[cm_index(i, j, k)];
    img.codes = to_codes(v, *labels);
    img.kind = "atlas";
  } else if (cube->cls == Class::UInt8 && !cube->is_logical) {
    img.u8.resize(total);
    for (std::size_t i = 0; i < ni; ++i) for (std::size_t j = 0; j < nj; ++j) for (std::size_t k = 0; k < nk; ++k) img.u8[c_index(i, j, k)] = cube->re[cm_index(i, j, k)];
    img.kind = "anatomical";
  } else if (cube->cls == Class::Double || cube->cls == Class::Single) {
    const std::vector<double> cm = cube->doubles();
    img.f32.resize(total);
    for (std::size_t i = 0; i < ni; ++i) for (std::size_t j = 0; j < nj; ++j) for (std::size_t k = 0; k < nk; ++k) img.f32[c_index(i, j, k)] = static_cast<float>(cm[cm_index(i, j, k)]);
    img.kind = "scalar";
  } else {
    value_error(fname + ": Cube is " + np_dtype(*cube) + "; only uint8 anatomy, integer atlases and float maps are supported");
  }
  img.labels = std::move(labels);

  if (const Value* vs = get("Voxsize")) {
    const std::vector<double> v = vs->is_numeric() ? vs->doubles() : std::vector<double>{};
    if (v.size() != 3) value_error(fname + ": Voxsize has " + std::to_string(v.size()) + " entries, expected 3");
    img.voxsize = {v[0], v[1], v[2]};
  }
  const Value* scs_raw = get("SCS");
  if (!scs_raw) value_error(fname + ": no SCS \xe2\x80\x94 the registration is the reason to import this");
  for (const char* k : {"R", "T", "NAS", "LPA", "RPA", "Origin"})
    if (auto v = field_numbers(scs_raw, k, std::string(k) == "R" ? 9 : 0)) img.scs[k] = *v;
  verify_scs(img.scs, img.voxsize);
  if (const Value* ncs_raw = get("NCS"))
    for (const char* k : {"R", "T", "AC", "PC", "IH"})
      if (auto v = field_numbers(ncs_raw, k, std::string(k) == "R" ? 9 : 0)) img.ncs[k] = *v;

  const std::string comment = char_text(get("Comment"));
  img.name = mri_name(p, get("Comment") && get("Comment")->is_char() ? comment : "");
  img.comment = comment;
  img.source = fname;
  return img;
}

std::string write_mri(SubjectWriter& sub, const Mri& img) {
  const std::string node = "mri/" + img.name;
  for (const char* table : {"manifold", "field", "operator", "selection"})
    if (sub.find(table, node)) value_error(img.name + ": " + node + " is already in the subject \xe2\x80\x94 remove it first");
  const nxr::io::Shape chunks{std::min(kMriChunk, img.shape[0]), std::min(kMriChunk, img.shape[1]), std::min(kMriChunk, img.shape[2])};
  const nxr::io::Shape shape{img.shape[0], img.shape[1], img.shape[2]};
  const std::int64_t total = img.shape[0] * img.shape[1] * img.shape[2];
  const std::string axes = "+x anterior, +y left, +z superior; fiducials at z=0";

  if (img.kind == "anatomical") {
    const std::string vol = node + "_volume";
    const std::string vname = vol.substr(vol.rfind('/') + 1);
    nxr::io::ManifoldSpec ms;
    ms.name = vname;
    ms.path = vol;
    ms.type = "volume";
    ms.n_vertices = total;
    ms.topology = {"lattice", 3, 1, std::nullopt, {}};
    const auto lc = nxr::io::lattice_counts({img.shape[0], img.shape[1], img.shape[2]});
    if (lc.n_edges) ms.cols.set("n_edges", *lc.n_edges); else ms.cols.null("n_edges");
    if (lc.n_faces) ms.cols.set("n_faces", *lc.n_faces); else ms.cols.null("n_faces");
    if (lc.n_volumes) ms.cols.set("n_volumes", *lc.n_volumes); else ms.cols.null("n_volumes");
    nxr::io::GeometrySpec g;
    g.form = "closed_form";
    g.cs_key = "voxel:" + vname;
    const char* ax = "ijk";
    for (int d = 0; d < 3; ++d) {
      nxr::io::DimensionSpec ds;
      ds.name = std::string(1, ax[d]);
      ds.n_vertices = img.shape[d];
      ds.spacing = img.voxsize[d];
      ds.unit = "mm";
      g.dimensions.push_back(ds);
    }
    ms.geometry = g;
    ms.cols.set("source", img.source);
    ms.cols.json("producer_json", Json{{"scs", frame_key(img.scs)}, {"convention", "scs_mm = R @ (voxel * voxsize) + T"}, {"axes", axes}});
    const std::string vid = sub.manifold(std::move(ms));

    nxr::io::FieldSpec f;
    f.name = img.name;
    f.path = node;
    f.kind = "T1-weighted";
    f.manifold_id = vid;
    f.data = view(img.u8, shape);
    f.array.chunks = chunks;
    f.array.compress = true;
    f.array.dense = true;
    f.cols.set("description", "T1-weighted intensity per voxel").set("source", img.source);
    if (!img.comment.empty()) f.cols.set("comment", img.comment); else f.cols.null("comment");
    sub.field(std::move(f));

    struct Frame { const char* key; const std::map<std::string, std::vector<double>>* tf; std::vector<const char*> extra; };
    for (const Frame& fr : {Frame{"scs", &img.scs, {"NAS", "LPA", "RPA", "Origin"}}, Frame{"ncs", &img.ncs, {"AC", "PC", "IH"}}}) {
      if (fr.tf->empty() || !fr.tf->count("R") || !fr.tf->count("T")) continue;
      const std::string frame = fr.key, up = frame == "scs" ? "SCS" : "NCS";
      const std::vector<double> mat = affine(*fr.tf, img.voxsize);
      nxr::io::OperatorSpec op;
      op.name = img.name + "_" + frame;
      op.path = op.name;
      op.kind = "transition";
      op.from_manifold_id = vid;
      op.cols.set("from_coordinate_system_id", sub.cs("voxel:" + vname));
      op.cols.set("to_coordinate_system_id", sub.cs(frame));
      op.n_rows = 3;
      op.n_cols = 4;
      op.data = view(mat, {3, 4});
      op.cols.set("description", "voxel index \xe2\x86\x92 " + up + " millimetres (3x4 affine)");
      Json prod = Json{{"convention", frame + "_mm = R @ (voxel * voxsize) + T"}};
      if (frame == "scs") prod["axes"] = axes;
      for (const char* k : fr.extra) {
        std::string lower = k;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (const auto it = fr.tf->find(k); it != fr.tf->end()) prod[lower] = it->second;
      }
      op.cols.json("producer_json", prod);
      sub.operator_(std::move(op));
    }
    return node;
  }

  const auto va = anatomical_volume(sub);
  if (!va) value_error(img.name + ": an atlas or map lives on the anatomical volume \xe2\x80\x94 import the T1 first");
  const std::string vpath = nxr::io::text(*va, "path");
  std::vector<std::int64_t> dims;
  for (const auto& d : sub.db().all("SELECT d.n_vertices FROM geometry_dimension d WHERE d.geometry_id = ? ORDER BY d.ordinal", {nxr::io::text(*va, "geometry_id")}))
    dims.push_back(nxr::io::integer(d, "n_vertices"));
  Json stored = nullptr;
  if (const auto pj = nxr::io::opt_text(*va, "producer_json")) {
    const Json parsed = Json::parse(*pj);
    if (parsed.contains("scs")) stored = parsed["scs"];
  }
  const Json want = Json::parse(nxr::io::jdump(frame_key(img.scs)));
  if (stored != want || dims != std::vector<std::int64_t>(shape.begin(), shape.end()))
    value_error(img.name + ": its voxel grid or SCS frame differs from " + vpath + "'s \xe2\x80\x94 one Volume per subject (decision 2), and this volume is not on it");

  if (img.kind == "atlas") {
    const MriLabels& lab = *img.labels;
    nxr::io::DictionarySpec d;
    d.name = img.name;
    d.scope = "app";
    std::vector<nxr::io::DictionaryEntry> entries;
    for (std::size_t i = 0; i < lab.ids.size(); ++i) {
      nxr::io::DictionaryEntry e;
      e.code = lab.ids[i];
      e.name = lab.names[i];
      e.ordinal = lab.ids[i];
      e.color = lab.colors[i];
      e.cols.json("attributes_json", Json{{"source_id", static_cast<std::int64_t>(lab.source_ids[i])}});
      entries.push_back(e);
    }
    d.entries = entries;
    const auto dr = sub.dictionary(d);
    const std::int64_t max_code = *std::max_element(dr.codes.begin(), dr.codes.end());
    std::vector<std::uint16_t> mapped(img.codes.size());
    std::set<std::uint16_t> distinct;
    for (std::size_t i = 0; i < mapped.size(); ++i) { mapped[i] = static_cast<std::uint16_t>(dr.codes.at(img.codes[i])); distinct.insert(mapped[i]); }
    nxr::io::SelectionSpec s;
    s.name = img.name;
    s.path = node;
    s.type = "set";
    s.manifold_id = nxr::io::text(*va, "id");
    s.no_elements = true;
    s.dense = true;
    std::vector<std::uint8_t> narrow;
    if (max_code < 256) {
      narrow.assign(mapped.begin(), mapped.end());
      s.data = view(narrow, shape);
      s.meta = nxr::io::array_meta(nxr::io::DType::UInt8, shape, {.chunks = chunks, .compress = true});
    } else {
      s.data = view(mapped, shape);
      s.meta = nxr::io::array_meta(nxr::io::DType::UInt16, shape, {.chunks = chunks, .compress = true});
    }
    std::string desc = "one label per voxel \xe2\x80\x94 the CODE, i.e. the dictionary entry; its source_id is Brainstorm's label value (D123)";
    if (!lab.unlabelled.empty()) desc += "; unlabelled values " + py_int_list(lab.unlabelled);
    s.cols.set("dictionary_id", dr.id).set("n_elements", total).set("n_members", static_cast<std::int64_t>(distinct.size())).set("description", desc);
    sub.selection(std::move(s));
    return node;
  }

  std::string lower = img.name + " " + img.comment;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  nxr::io::FieldSpec f;
  f.name = img.name;
  f.path = node;
  f.kind = img.comment.empty() ? img.name : img.comment;
  f.manifold_id = nxr::io::text(*va, "id");
  f.data = view(img.f32, shape);
  f.array.chunks = chunks;
  f.array.compress = true;
  f.array.dense = true;
  f.cols.set("description", "a value per voxel of the anatomical volume (resliced to its grid)").set("source", img.source);
  if (!img.comment.empty()) f.cols.set("comment", img.comment); else f.cols.null("comment");
  if (lower.find("suvr") != std::string::npos) f.cols.set("unit", "SUVR"); else f.cols.null("unit");
  sub.field(std::move(f));
  return node;
}

std::vector<std::string> find_subjectimages(const fs::path& anat_dir) {
  std::vector<std::string> found;
  for (const auto& n : glob_names(anat_dir, "subjectimage_", ".mat")) if (n.rfind("._", 0) != 0) found.push_back(n);
  std::stable_sort(found.begin(), found.end(), [](const std::string& a, const std::string& b) {
    const bool va = a.find("volatlas") != std::string::npos, vb = b.find("volatlas") != std::string::npos;
    return va != vb ? vb : a < b;
  });
  return found;
}

std::vector<Json> import_subjectimages(SubjectWriter& sub, const fs::path& anat_dir, const std::optional<std::vector<std::string>>& only, const Emit& progress) {
  std::vector<Json> out;
  std::vector<std::pair<std::string, std::string>> skipped;   // (file, reason)
  std::vector<Mri> images;
  auto skip = [&](const std::string& file, const std::string& reason) {
    skipped.push_back({file, reason});
    if (progress) progress(Json{{"stage", "skipped"}, {"file", file}, {"reason", reason}});
  };
  for (const auto& n : find_subjectimages(anat_dir)) {
    const fs::path p = anat_dir / n;
    if (only && !only->empty() && std::find(only->begin(), only->end(), mri_name(p)) == only->end()) continue;
    try {
      images.push_back(read_subjectimage(p));
    } catch (const nxr::io::Error& e) {
      if (std::string(e.code()) != code::invalid_argument) throw;
      skip(n, e.what());
    }
  }
  const auto rank = [](const Mri& m) { return m.kind == "anatomical" ? 0 : m.kind == "atlas" ? 1 : 2; };
  std::stable_sort(images.begin(), images.end(), [&](const Mri& a, const Mri& b) { return rank(a) != rank(b) ? rank(a) < rank(b) : a.source < b.source; });
  for (const Mri& img : images) {
    std::string base;
    try {
      base = write_mri(sub, img);
    } catch (const nxr::io::Error& e) {
      if (std::string(e.code()) != code::invalid_argument) throw;
      skip(img.source, e.what());
      continue;
    }
    Json rec = Json{{"stage", "mri"}, {"name", img.name}, {"kind", img.kind}, {"shape", std::vector<std::int64_t>(img.shape.begin(), img.shape.end())}, {"path", base}};
    if (img.labels) rec["labels"] = static_cast<std::int64_t>(img.labels->ids.size());
    out.push_back(rec);
    if (progress) progress(rec);
  }
  if (out.empty()) {
    std::string why;
    for (const auto& [f, r] : skipped) why += (why.empty() ? "" : "; ") + f + ": " + r;
    if (why.empty()) why = "none found";
    throw nxr::io::Error(code::not_found, "no subject image imported from " + py_path(anat_dir.string()) + " (" + why + ")");
  }
  return out;
}

}  // namespace nxr::bst
