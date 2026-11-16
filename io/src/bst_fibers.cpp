// nxr::bst — nxr.convert.fibers: tractography as fibre bundles (curve manifolds), their ends on the cortex and the connectomes.
#include "nxr/bst/fibers.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>

#include "nxr/bst/mat.h"
#include "nxr/bst/maps.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/layout.h"
#include "nxr/io/surface.h"

namespace nxr::bst {

namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::SubjectWriter;
using nxr::io::view;

namespace {

[[noreturn]] void value_error(const std::string& m) { throw Error(code::invalid_argument, m); }

// a numpy shape as Python prints it: (3, 4) · (5,) · ()
std::string shape_repr(const std::vector<std::size_t>& d) {
  std::string s = "(";
  for (std::size_t i = 0; i < d.size(); ++i) s += (i ? ", " : "") + std::to_string(d[i]);
  if (d.size() == 1) s += ",";
  return s + ")";
}

// pymatreader squeezes the singleton dimensions of an array
std::vector<std::size_t> squeezed(const Value& v) {
  std::vector<std::size_t> d;
  for (std::size_t x : v.dims) if (x != 1) d.push_back(x);
  return d;
}

// a numeric array as MATLAB stores it (column-major) -> the values in C order of the same dims (numpy's `ravel()` of the array pymatreader returns)
std::vector<double> c_order(const Value& v) {
  const std::vector<double> col = v.doubles();
  const std::vector<std::size_t> d = v.dims;
  std::vector<double> out(col.size());
  std::vector<std::size_t> idx(d.size(), 0);
  for (std::size_t k = 0; k < out.size(); ++k) {   // k runs in C order; the column-major offset of the same index
    std::size_t off = 0, stride = 1;
    for (std::size_t a = 0; a < d.size(); ++a) { off += idx[a] * stride; stride *= d[a]; }
    out[k] = col[off];
    for (std::size_t a = d.size(); a-- > 0;) { if (++idx[a] < d[a]) break; idx[a] = 0; }
  }
  return out;
}

// scipy.spatial.cKDTree(points).query(q) -> the nearest point's index (3-D, Euclidean)
class KdTree {
 public:
  explicit KdTree(const std::vector<double>& pts) : p_(pts), order_(pts.size() / 3) {
    for (std::size_t i = 0; i < order_.size(); ++i) order_[i] = static_cast<std::int64_t>(i);
    build(0, static_cast<std::int64_t>(order_.size()), 0);
  }
  std::int64_t nearest(const double* q) const {
    best_ = -1;
    best_d_ = std::numeric_limits<double>::infinity();
    search(0, static_cast<std::int64_t>(order_.size()), 0, q);
    return best_;
  }

 private:
  const std::vector<double>& p_;
  std::vector<std::int64_t> order_;
  mutable std::int64_t best_ = -1;
  mutable double best_d_ = 0;
  double at(std::int64_t i, int axis) const { return p_[static_cast<std::size_t>(i) * 3 + static_cast<std::size_t>(axis)]; }
  void build(std::int64_t lo, std::int64_t hi, int axis) {
    if (hi - lo < 2) return;
    const std::int64_t mid = (lo + hi) / 2;
    std::nth_element(order_.begin() + lo, order_.begin() + mid, order_.begin() + hi, [&](std::int64_t a, std::int64_t b) {
      return at(a, axis) < at(b, axis) || (at(a, axis) == at(b, axis) && a < b);
    });
    build(lo, mid, (axis + 1) % 3);
    build(mid + 1, hi, (axis + 1) % 3);
  }
  void search(std::int64_t lo, std::int64_t hi, int axis, const double* q) const {
    if (hi <= lo) return;
    const std::int64_t mid = (lo + hi) / 2, i = order_[static_cast<std::size_t>(mid)];
    double d = 0;
    for (int a = 0; a < 3; ++a) { const double t = at(i, a) - q[a]; d += t * t; }
    if (d < best_d_ || (d == best_d_ && i < best_)) { best_d_ = d; best_ = i; }
    const double diff = q[axis] - at(i, axis);
    const int next = (axis + 1) % 3;
    if (diff < 0) { search(lo, mid, next, q); if (diff * diff <= best_d_) search(mid + 1, hi, next, q); }
    else { search(mid + 1, hi, next, q); if (diff * diff <= best_d_) search(lo, mid, next, q); }
  }
};

// an index array in the narrowest of nxr's two widths: int32 unless a value reaches 2**31
struct Ints {
  std::vector<std::int32_t> a;
  std::vector<std::int64_t> b;
  Ints(const std::vector<std::int64_t>& v, bool wide) {
    if (wide) b = v; else for (auto x : v) a.push_back(static_cast<std::int32_t>(x));
  }
  nxr::io::ArrayView view_(std::int64_t n) const { return b.empty() ? view(a, {n}) : view(b, {n}); }
};

std::string text_of(const Value* v) { return v && v->is_char() ? v->string() : ""; }

}  // namespace

bool is_fibers_file(const fs::path& path) { return path.filename().string().rfind("tess_fibers", 0) == 0; }

Json export_fibers(SubjectWriter& sub, const fs::path& fibers_file, const std::string& subject, const std::string& cortex_path, const fs::path& cortex_file,
                   const Emit& progress) {
  const fs::path& p = fibers_file;
  const std::string pname = p.filename().string();
  const auto mat = read_all_variables(p);
  auto var = [&](const char* k) -> const Value* { auto it = mat.find(k); return it == mat.end() ? nullptr : &it->second; };

  const Value* pv = var("Points");
  if (!pv || !pv->is_numeric()) value_error(pname + ": Points is (), expected [nFibers, nPoints, 3]");
  const std::vector<std::size_t> sq = squeezed(*pv);
  if (sq.size() != 3 || sq[2] != 3) value_error(pname + ": Points is " + shape_repr(sq) + ", expected [nFibers, nPoints, 3]");
  const std::int64_t n_f = static_cast<std::int64_t>(sq[0]), n_p = static_cast<std::int64_t>(sq[1]);
  const std::vector<double> pts = c_order(*pv);   // [nF, nP, 3], C order
  const std::string name = nxr::io::sanitize_node_name(surface_node_name(pname));
  const std::string node = nxr::io::folder_for("curve") + "/" + name;
  const auto cortex = sub.find("manifold", cortex_path);
  if (!cortex) value_error(pname + ": the cortex " + cortex_path + " is not in the subject \xe2\x80\x94 export it first");

  // ---- the curves: a bundle of 1-manifolds (D125) ----
  std::vector<float> flat(pts.size());
  for (std::size_t i = 0; i < pts.size(); ++i) flat[i] = static_cast<float>(pts[i]);
  const std::int64_t n_fp = n_f * n_p;
  std::vector<std::uint8_t> colors;
  if (const Value* cv = var("Colors"); cv && cv->is_numeric() && static_cast<std::int64_t>(cv->numel()) == n_fp * 3)
    for (double x : c_order(*cv)) colors.push_back(static_cast<std::uint8_t>(static_cast<std::int64_t>(x)));

  nxr::io::ManifoldSpec ms;
  ms.name = name;
  ms.path = node;
  ms.type = "curve";
  ms.n_vertices = n_fp;
  ms.topology = {"path", 1, static_cast<int>(n_f), std::nullopt, {}};
  nxr::io::GeometrySpec g;
  g.form = "stored";
  g.cs_key = "scs";
  g.positions_path = node + "/points";
  ms.geometry = g;
  ms.cols.set("n_edges", n_f * (n_p - 1));
  if (const std::string c = text_of(var("Comment")); !c.empty()) ms.cols.set("comment", c); else ms.cols.null("comment");
  ms.cols.set("source", pname);
  ms.cols.json("producer_json", Json{{"points_per_component", n_p},
                                     {"description", std::to_string(n_f) + " fibres of " + std::to_string(n_p) + " points each (fibre-major: fibre f is rows f\xc2\xb7" +
                                                         std::to_string(n_p) + "\xe2\x80\xa6)"}});
  ms.populate = [&](nxr::io::Store&, const std::string&) {
    nxr::io::ArrayOptions ao;
    ao.compress = true;
    sub.write_array(node + "/points", view(flat, {n_fp, 3}), ao);
    if (!colors.empty()) sub.write_array(node + "/colors", view(colors, {n_fp, 3}), ao);
  };
  const std::string fid = sub.manifold(std::move(ms));
  if (progress) progress(Json{{"stage", "fibers"}, {"name", name}, {"n_fibers", n_f}, {"n_points", n_p}});

  // ---- the ends on the cortex: nearest vertex to each fibre's first and last point ----
  std::vector<double> verts;
  {
    const auto rd = nxr::io::ArrayReader::open(sub.store(), cortex_path + "/vertices");
    const auto bytes = rd.read();
    const std::size_t n = bytes.size() / rd.itemsize();
    verts.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      if (rd.dtype() == nxr::io::DType::Float32) { float x; std::memcpy(&x, bytes.data() + i * 4, 4); verts[i] = x; }
      else { double x; std::memcpy(&x, bytes.data() + i * 8, 8); verts[i] = x; }
    }
  }
  const std::int64_t nv = static_cast<std::int64_t>(verts.size() / 3);
  const KdTree tree(verts);
  std::vector<std::int64_t> nearest(static_cast<std::size_t>(2 * n_f));
  for (std::int64_t f = 0; f < n_f; ++f) {
    nearest[static_cast<std::size_t>(f)] = tree.nearest(&pts[static_cast<std::size_t>(f * n_p * 3)]);                              // first point
    nearest[static_cast<std::size_t>(n_f + f)] = tree.nearest(&pts[static_cast<std::size_t>((f * n_p + n_p - 1) * 3)]);          // last point
  }
  std::vector<std::int64_t> end_rows(static_cast<std::size_t>(2 * n_f));   // fibre vertices
  for (std::int64_t f = 0; f < n_f; ++f) { end_rows[static_cast<std::size_t>(f)] = f * n_p; end_rows[static_cast<std::size_t>(n_f + f)] = f * n_p + (n_p - 1); }
  std::vector<std::int64_t> order(end_rows.size());   // CSR over cortex vertices: lexsort((end_rows, nearest))
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<std::int64_t>(i);
  std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
    const auto na = nearest[static_cast<std::size_t>(a)], nb = nearest[static_cast<std::size_t>(b)];
    return na != nb ? na < nb : end_rows[static_cast<std::size_t>(a)] < end_rows[static_cast<std::size_t>(b)];
  });
  std::vector<std::int64_t> indptr(static_cast<std::size_t>(nv + 1), 0), indices(order.size());
  for (auto v : nearest) ++indptr[static_cast<std::size_t>(v) + 1];
  for (std::int64_t i = 0; i < nv; ++i) indptr[static_cast<std::size_t>(i) + 1] += indptr[static_cast<std::size_t>(i)];
  for (std::size_t i = 0; i < order.size(); ++i) indices[i] = end_rows[static_cast<std::size_t>(order[i])];
  const std::string op = name + "_ends";
  {
    const std::int64_t lim = static_cast<std::int64_t>(1) << 31;
    const Ints ip(indptr, indptr.back() >= lim), ix(indices, n_fp >= lim);
    const std::vector<float> ones(static_cast<std::size_t>(2 * n_f), 1.0f);
    nxr::io::OperatorSpec os;
    os.name = op;
    os.path = op;
    os.kind = "fibre ends";
    os.from_manifold_id = fid;
    os.layout = "csr";
    os.n_rows = nv;
    os.n_cols = n_fp;
    os.cols.set("to_manifold_id", nxr::io::text(*cortex, "id")).set("nnz", 2 * n_f)
        .set("description", "each fibre's two ends -> its nearest cortex vertex (1 per end)").set("source", pname);
    os.cols.json("producer_json", Json{{"method", "nearest vertex (scipy cKDTree), SCS metres"}});
    os.sparse = nxr::io::SparseArrays{{{"indptr", ip.view_(nv + 1)}, {"indices", ix.view_(2 * n_f)}, {"data", view(ones, {2 * n_f})}}};
    sub.operator_(std::move(os));
  }
  if (progress) progress(Json{{"stage", "fibre-ends"}, {"name", op}, {"nnz", 2 * n_f}});

  // ---- THE CONNECTOME: a sparse field on cortex x cortex (D159), from the ends above: W = E1^T E2 + E2^T E1, CSC ----
  std::map<std::pair<std::int64_t, std::int64_t>, float> cells;   // (column, row) -> streamlines: ordered by column, then row
  for (std::int64_t f = 0; f < n_f; ++f) {
    const std::int64_t a = nearest[static_cast<std::size_t>(f)], b = nearest[static_cast<std::size_t>(n_f + f)];
    cells[{b, a}] += 1.0f;   // E1^T E2
    cells[{a, b}] += 1.0f;   // E2^T E1
  }
  std::vector<std::int64_t> w_indptr(static_cast<std::size_t>(nv + 1), 0), w_indices;
  std::vector<float> w_data;
  for (const auto& [cr, v] : cells) {
    ++w_indptr[static_cast<std::size_t>(cr.first) + 1];
    w_indices.push_back(cr.second);
    w_data.push_back(v);
  }
  for (std::int64_t i = 0; i < nv; ++i) w_indptr[static_cast<std::size_t>(i) + 1] += w_indptr[static_cast<std::size_t>(i)];
  const std::int64_t nnz = static_cast<std::int64_t>(w_data.size());
  const std::string cname = name + "_connectome";
  {
    const std::string cid = nxr::io::text(*cortex, "id");
    const std::string pid = sub.product({cid, cid});
    const Ints ip(w_indptr, nnz >= (static_cast<std::int64_t>(1) << 31)), ix(w_indices, false);
    nxr::io::FieldSpec fs_;
    fs_.name = cname;
    fs_.path = nxr::io::node_path(sub.db(), pid, cname);
    fs_.kind = "connectome";
    fs_.manifold_id = pid;
    fs_.cols.set("unit", "streamlines").set("source", pname)
        .set("description", "streamline counts between cortex vertices: each fibre adds 1 to the pair of its two end vertices (nearest cortex vertex), symmetric");
    fs_.cols.json("producer_json", Json{{"measure", "streamline_count"}, {"ends", op}, {"fibres", n_f}, {"method", "W = E1\xe1\xb5\x80" "E2 + E2\xe1\xb5\x80" "E1"}});
    fs_.sparse = nxr::io::SparseArrays{{{"indptr", ip.view_(nv + 1)}, {"indices", ix.view_(nnz)}, {"data", view(w_data, {nnz})}}};
    sub.field(std::move(fs_));
  }
  if (progress) progress(Json{{"stage", "connectome"}, {"name", cname}, {"nnz", nnz}});

  // ---- the connectomes Brainstorm assigned (Scouts(k).Assignment, per atlas): measurements on the atlas's Selection ----
  Json connectomes = Json::array();
  if (const Value* scouts = var("Scouts"); scouts && scouts->is_struct() && scouts->has_field("ConnectFile")) {
    for (std::size_t e = 0; e < scouts->numel(); ++e) {
      const Value* av = scouts->field("Assignment", e);
      std::string label = text_of(scouts->field("ConnectFile", e));
      if (label.empty()) label = text_of(scouts->field("Label", e));
      // pymatreader squeezes: a [1, 2] assignment is 1-D and is skipped
      if (!av || !av->is_numeric() || av->numel() == 0 || av->dims.size() != 2 || av->dims[1] != 2 || static_cast<std::int64_t>(av->dims[0]) != n_f || n_f == 1) continue;
      const std::string atlas = label.rfind(subject + "_", 0) == 0 ? label.substr(subject.size() + 1) : label;
      const std::string fam = nxr::io::sanitize_node_name(atlas);
      // Brainstorm's assignment is a region INDEX (the scout's row, 1-based; 0 = none); the atlas's CODES are its dictionary's
      const auto regions = atlas_regions(sub, cortex_path, cortex_file, atlas);
      if (!regions) {
        if (progress) progress(Json{{"stage", "skipped"}, {"file", pname}, {"reason", "connectome '" + label + "': no atlas '" + fam + "' on " + cortex_path}});
        continue;
      }
      const auto& [levels, code_of, selection_id] = *regions;
      const std::int64_t n_levels = static_cast<std::int64_t>(levels.size());
      std::vector<std::int64_t> a;
      for (double x : c_order(*av)) a.push_back(static_cast<std::int64_t>(x));   // [nF, 2], C order
      const std::int64_t amax = *std::max_element(a.begin(), a.end()), amin = *std::min_element(a.begin(), a.end());
      if (amax >= n_levels || amin < 0)
        value_error(pname + ": " + label + " assigns region " + std::to_string(amax) + " of a " + std::to_string(n_levels - 1) + "-region atlas");
      for (auto& x : a) x = code_of[static_cast<std::size_t>(x)];
      std::map<std::pair<std::int64_t, std::int64_t>, double> C;   // the tally import_fibers_subject.m makes, code 0 kept
      std::int64_t assigned = 0;
      for (std::int64_t f = 0; f < n_f; ++f) {
        const std::int64_t r = a[static_cast<std::size_t>(2 * f)], c = a[static_cast<std::size_t>(2 * f + 1)];
        C[{r, c}] += 1;
        if (r != c) C[{c, r}] += 1;
        if (r > 0 && c > 0) ++assigned;
      }
      if (sub.db().one("SELECT 1 FROM selection_measurement WHERE selection_id = ? AND measure = 'streamlines' AND of_field_id IS NULL LIMIT 1", {nxr::io::V(selection_id)}))
        value_error(pname + ": the " + fam + " atlas already holds Brainstorm's streamline counts of another fibres file");
      std::vector<RegionValue> rows;
      for (const auto& [rc, v] : C) rows.push_back(RegionValue{rc.first, v, rc.second});
      region_measurements(sub, selection_id, rows, "streamlines", std::string("streamlines"));
      connectomes.push_back(fam);
      if (progress) progress(Json{{"stage", "connectome"}, {"atlas", fam}, {"regions", n_levels - 1}, {"assigned", assigned}});
    }
  }
  return Json{{"name", name}, {"id", fid}, {"n_fibers", n_f}, {"n_points", n_p}, {"connectome", cname}, {"connectomes", connectomes}};
}

}  // namespace nxr::bst
