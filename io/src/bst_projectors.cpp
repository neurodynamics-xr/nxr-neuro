// nxr::bst — nxr.convert.projectors: the channel file's projectors and CTF compensation (register D168).
// Products are NOT fused with the sums that follow them: nxr's numpy rounds each.
#pragma STDC FP_CONTRACT OFF
#include "nxr/bst/projectors.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "nxr/io/array_reader.h"

#include "nxr/io/layout.h"

namespace nxr::bst {

namespace code = nxr::io::code;
using nxr::io::Json;
using nxr::io::view;

namespace {

std::string upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return s;
}
std::string strip(const std::string& s) {
  const char* ws = " \t\n\r\f\v";
  const auto a = s.find_first_not_of(ws);
  if (a == std::string::npos) return "";
  return s.substr(a, s.find_last_not_of(ws) - a + 1);
}
std::string text_of(const Value* v) { return v && v->is_char() && !v->text.empty() ? v->text[0] : ""; }

using Mat = std::vector<double>;   // row-major

// sum over j ascending of a[:, j] w[j, :]  (a [R x J], w [J x C]); `cols` selects the j (all when empty)
Mat outer_sum(const Mat& a, std::size_t r, std::size_t j_total, const Mat& w, std::size_t c, const std::vector<std::size_t>& cols) {
  Mat s(r * c, 0.0);
  for (std::size_t j : cols)
    for (std::size_t i = 0; i < r; ++i)
      for (std::size_t l = 0; l < c; ++l) {
        const double pr = a[i * j_total + j] * w[j * c + l];
        s[i * c + l] = s[i * c + l] + pr;
      }
  return s;
}

// A = W^T (W W^T)^-1 for W [K x M]; returns [M x K]
Mat pseudo_inverse(const Mat& w, std::size_t k, std::size_t m) {
  Mat g(k * k, 0.0);
  for (std::size_t j = 0; j < m; ++j)
    for (std::size_t i = 0; i < k; ++i)
      for (std::size_t l = 0; l < k; ++l) {
        const double pr = w[i * m + j] * w[l * m + j];
        g[i * k + l] = g[i * k + l] + pr;
      }
  Mat inv(k * k, 0.0);
  for (std::size_t i = 0; i < k; ++i) inv[i * k + i] = 1.0;
  for (std::size_t c = 0; c < k; ++c) {
    std::size_t p = c;
    for (std::size_t r = c + 1; r < k; ++r)
      if (std::fabs(g[r * k + c]) > std::fabs(g[p * k + c])) p = r;
    if (g[p * k + c] == 0.0) throw nxr::io::Error(code::invalid_argument, "ICA projector: the unmixing matrix is rank deficient");
    if (p != c)
      for (std::size_t x = 0; x < k; ++x) { std::swap(g[c * k + x], g[p * k + x]); std::swap(inv[c * k + x], inv[p * k + x]); }
    const double d = g[c * k + c];
    for (std::size_t x = 0; x < k; ++x) { g[c * k + x] = g[c * k + x] / d; inv[c * k + x] = inv[c * k + x] / d; }
    for (std::size_t r = 0; r < k; ++r) {
      if (r == c) continue;
      const double f = g[r * k + c];
      for (std::size_t x = 0; x < k; ++x) {
        const double p1 = f * g[c * k + x];
        g[r * k + x] = g[r * k + x] - p1;
        const double p2 = f * inv[c * k + x];
        inv[r * k + x] = inv[r * k + x] - p2;
      }
    }
  }
  Mat a(m * k, 0.0);
  for (std::size_t j = 0; j < k; ++j)
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t l = 0; l < k; ++l) {
        const double pr = w[j * m + i] * inv[j * k + l];
        a[i * k + l] = a[i * k + l] + pr;
      }
  return a;
}

}  // namespace

std::vector<ProjectorEntry> projector_entries(const std::map<std::string, Value>& channel_mat) {
  std::vector<ProjectorEntry> out;
  const auto it = channel_mat.find("Projector");
  if (it == channel_mat.end() || !it->second.is_struct() || it->second.numel() == 0) return out;
  const Value& p = it->second;
  for (std::size_t i = 0; i < p.numel(); ++i) {
    const Value* comp = p.field("Components", i);
    if (!comp || !comp->is_numeric()) throw nxr::io::Error(code::invalid_argument, "projector " + std::to_string(i + 1) + ": no Components");
    ProjectorEntry e;
    e.order = static_cast<int>(i + 1);
    e.m = comp->rows();
    e.k = comp->numel() / std::max<std::size_t>(e.m, 1);
    e.components = comp->row_major();
    const Value* mk = p.field("CompMask", i);
    e.mask.assign(e.k, false);
    if (mk && mk->is_numeric()) {
      const auto d = mk->doubles();
      if (d.size() != e.k) throw nxr::io::Error(code::invalid_argument, "projector " + std::to_string(i + 1) + ": CompMask has " + std::to_string(d.size()) + " entries for " + std::to_string(e.k) + " components");
      for (std::size_t j = 0; j < e.k; ++j) e.mask[j] = d[j] != 0.0;
    }
    e.comment = text_of(p.field("Comment", i));
    e.method = text_of(p.field("Method", i));
    const Value* st = p.field("Status", i);
    e.status = st && st->is_numeric() && st->numel() ? static_cast<int>(st->scalar()) : 0;
    e.kind = upper(e.method).rfind("ICA", 0) == 0 ? "ica" : "ssp";
    out.push_back(std::move(e));
  }
  return out;
}

std::vector<double> projector_matrix(const ProjectorEntry& e) {
  const std::size_t m = e.m, k = e.k;
  std::vector<std::size_t> sel;
  for (std::size_t j = 0; j < k; ++j) if (e.mask[j]) sel.push_back(j);
  Mat s;
  if (e.kind == "ica") {
    Mat w(k * m);                                           // W = Components^T
    for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < k; ++j) w[j * m + i] = e.components[i * k + j];
    const Mat a = pseudo_inverse(w, k, m);
    s = outer_sum(a, m, k, w, m, sel);
  } else {
    Mat ut(k * m);
    for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < k; ++j) ut[j * m + i] = e.components[i * k + j];
    s = outer_sum(e.components, m, k, ut, m, sel);
  }
  Mat p(m * m);
  for (std::size_t i = 0; i < m; ++i) for (std::size_t l = 0; l < m; ++l) p[i * m + l] = (i == l ? 1.0 : 0.0) - s[i * m + l];
  return p;
}

namespace {
// the eigen decomposition of a small symmetric matrix g [k x k] (cyclic Jacobi): eigenvalues descending in `w`, eigenvectors as the columns of `e`
void symmetric_eigen(Mat g, std::size_t k, std::vector<double>& w, Mat& e) {
  e.assign(k * k, 0.0);
  for (std::size_t i = 0; i < k; ++i) e[i * k + i] = 1.0;
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0, diag = 0;
    for (std::size_t i = 0; i < k; ++i) for (std::size_t j = 0; j < k; ++j) (i == j ? diag : off) += g[i * k + j] * g[i * k + j];
    if (off <= 1e-30 * diag) break;
    for (std::size_t p = 0; p < k; ++p)
      for (std::size_t q = p + 1; q < k; ++q) {
        const double apq = g[p * k + q];
        if (apq == 0.0) continue;
        const double theta = (g[q * k + q] - g[p * k + p]) / (2 * apq);
        const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
        const double c = 1 / std::sqrt(t * t + 1), sn = t * c;
        for (std::size_t r = 0; r < k; ++r) {                       // g := J^T g J, columns then rows
          const double gp = g[r * k + p], gq = g[r * k + q];
          g[r * k + p] = c * gp - sn * gq;
          g[r * k + q] = sn * gp + c * gq;
        }
        for (std::size_t r = 0; r < k; ++r) {
          const double gp = g[p * k + r], gq = g[q * k + r];
          g[p * k + r] = c * gp - sn * gq;
          g[q * k + r] = sn * gp + c * gq;
        }
        for (std::size_t r = 0; r < k; ++r) {
          const double ep = e[r * k + p], eq = e[r * k + q];
          e[r * k + p] = c * ep - sn * eq;
          e[r * k + q] = sn * ep + c * eq;
        }
      }
  }
  std::vector<std::size_t> idx(k);
  for (std::size_t i = 0; i < k; ++i) idx[i] = i;
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return g[a * k + a] > g[b * k + b]; });
  w.resize(k);
  Mat es(k * k);
  for (std::size_t j = 0; j < k; ++j) {
    w[j] = g[idx[j] * k + idx[j]];
    for (std::size_t r = 0; r < k; ++r) es[r * k + j] = e[r * k + idx[j]];
  }
  e = std::move(es);
}
// b := a b for [n x n] row-major
Mat matmul(const Mat& a, const Mat& b, std::size_t n) {
  Mat s(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t l = 0; l < n; ++l) {
      const double x = a[i * n + l];
      if (x == 0.0) continue;
      for (std::size_t c = 0; c < n; ++c) s[i * n + c] += x * b[l * n + c];
    }
  return s;
}
}  // namespace

std::vector<double> joint_projector(const std::vector<StoredProjector>& projectors, const ProjectorRule& rule, std::vector<std::string>* used) {
  if (projectors.empty()) return {};
  const std::size_t m = projectors.front().m;
  std::vector<std::size_t> ch = rule.channels;
  if (ch.empty()) for (std::size_t i = 0; i < m; ++i) ch.push_back(i);
  const std::size_t n = ch.size();
  for (std::size_t c : ch) if (c >= m) throw nxr::io::Error(code::invalid_argument, "a projector's channel is outside its channels");
  // the selected entries, in the file's order; the SSP vectors stacked as the columns of V [n x k]
  std::vector<const StoredProjector*> sel;
  for (const auto& p : projectors) {
    if (p.m != m) throw nxr::io::Error(code::invalid_argument, "the projectors " + projectors.front().name + " and " + p.name + " are on different channels");
    if (std::find(rule.statuses.begin(), rule.statuses.end(), p.status) != rule.statuses.end()) sel.push_back(&p);
  }
  std::stable_sort(sel.begin(), sel.end(), [](const StoredProjector* a, const StoredProjector* b) { return a->order < b->order; });
  std::vector<std::vector<double>> cols;
  std::vector<int> items;                                            // -1: the joint SSP (at its first entry's place), else the index in `sel` of a matrix projector
  bool ssp_placed = false;
  for (std::size_t i = 0; i < sel.size(); ++i) {
    const StoredProjector& p = *sel[i];
    if (used) used->push_back(p.name);
    if (p.kind != "ica" && !p.vectors.empty()) {
      for (const auto& v : p.vectors) {
        if (v.size() != m) throw nxr::io::Error(code::invalid_argument, p.name + ": a vector does not match its channels");
        std::vector<double> c(n);
        for (std::size_t r = 0; r < n; ++r) c[r] = v[ch[r]];
        if (rule.renormalise) {
          double s = 0;
          for (double x : c) s += x * x;
          s = std::sqrt(s);
          if (s > 0) for (double& x : c) x /= s;
        }
        cols.push_back(std::move(c));
      }
      if (!ssp_placed) { items.push_back(-1); ssp_placed = true; }
    } else {
      if (p.matrix.size() != m * m) throw nxr::io::Error(code::invalid_argument, p.name + ": its matrix does not match its channels");
      items.push_back(static_cast<int>(i));
    }
  }
  Mat ssp;
  bool any = false;
  for (const auto& c : cols) for (double x : c) any = any || x != 0.0;
  if (any) {
    const std::size_t k = cols.size();
    Mat g(k * k, 0.0);                                               // V^T V: its eigenvalues are V's squared singular values, its eigenvectors V's right singular vectors
    for (std::size_t a = 0; a < k; ++a) for (std::size_t b = 0; b < k; ++b) { double s = 0; for (std::size_t r = 0; r < n; ++r) s += cols[a][r] * cols[b][r]; g[a * k + b] = s; }
    std::vector<double> w;
    Mat e;
    symmetric_eigen(g, k, w, e);
    const double s1 = std::sqrt(std::max(w[0], 0.0));
    ssp.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) ssp[i * n + i] = 1.0;
    for (std::size_t j = 0; j < k; ++j) {
      const double sj = std::sqrt(std::max(w[j], 0.0));
      if (!(sj >= 0.01 * s1) || sj == 0.0) break;                    // the linearly dependent vectors: a singular value below 1% of the largest
      std::vector<double> u(n, 0.0);                                 // U[:, j] = V e_j / s_j
      for (std::size_t a = 0; a < k; ++a) for (std::size_t r = 0; r < n; ++r) u[r] += cols[a][r] * e[a * k + j];
      for (double& x : u) { x /= sj; if (rule.zero_tiny && std::fabs(x) < 1e-15) x = 0.0; }
      for (std::size_t r = 0; r < n; ++r) for (std::size_t c = 0; c < n; ++c) ssp[r * n + c] -= u[r] * u[c];
    }
  }
  Mat out;
  for (int it : items) {
    Mat p;
    if (it < 0) { if (ssp.empty()) continue; p = ssp; }
    else {
      const StoredProjector& q = *sel[static_cast<std::size_t>(it)];
      p.assign(n * n, 0.0);
      for (std::size_t r = 0; r < n; ++r) for (std::size_t c = 0; c < n; ++c) p[r * n + c] = q.matrix[ch[r] * m + ch[c]];
    }
    out = out.empty() ? p : matmul(p, out, n);
  }
  bool identity = !out.empty();
  for (std::size_t r = 0; r < n && identity; ++r) for (std::size_t c = 0; c < n && identity; ++c) identity = out[r * n + c] == (r == c ? 1.0 : 0.0);
  if (identity) out.clear();
  return out;
}

std::vector<StoredProjector> stored_projectors(const nxr::io::Database& db, const nxr::io::Store& store, const std::string& channels_manifold_id) {
  std::vector<StoredProjector> out;
  for (const auto& r : db.all("SELECT name, path, n_rows, params_json FROM operator WHERE kind = 'projector' AND from_manifold_id = ? AND status = 'complete'", {nxr::io::V(channels_manifold_id)})) {
    StoredProjector p;
    p.name = nxr::io::text(r, "name");
    p.m = static_cast<std::size_t>(nxr::io::integer(r, "n_rows"));
    const Json pj = Json::parse(nxr::io::opt_text(r, "params_json").value_or("{}"), nullptr, false);
    if (pj.is_object()) {
      p.order = pj.value("order", 0);
      p.status = pj.value("status", 0);
      p.kind = pj.value("kind", std::string("ssp"));
      p.method = pj.value("method", std::string());
      if (pj.contains("vectors") && pj["vectors"].is_array()) for (const auto& v : pj["vectors"]) p.vectors.push_back(v.get<std::vector<double>>());
    }
    if (p.vectors.empty() || p.kind == "ica") {
      const nxr::io::ArrayReader ar = nxr::io::ArrayReader::open(store, nxr::io::text(r, "path"));
      const std::vector<std::uint8_t> raw = ar.read();
      p.matrix.resize(p.m * p.m);
      if (ar.dtype() == nxr::io::DType::Float64 && raw.size() == p.matrix.size() * 8) std::memcpy(p.matrix.data(), raw.data(), raw.size());
      else if (ar.dtype() == nxr::io::DType::Float32 && raw.size() == p.matrix.size() * 4) for (std::size_t i = 0; i < p.matrix.size(); ++i) { float f; std::memcpy(&f, raw.data() + 4 * i, 4); p.matrix[i] = f; }
      else throw nxr::io::Error(code::invalid_argument, p.name + ": a projector is a dense [channels x channels] array of floats");
    }
    out.push_back(std::move(p));
  }
  std::stable_sort(out.begin(), out.end(), [](const StoredProjector& a, const StoredProjector& b) { return a.order < b.order; });
  return out;
}

std::optional<CtfGrade> ctf_grade(const Value* sfile) {
  if (!sfile || !sfile->is_struct()) return std::nullopt;
  const Value* prop = sfile->field("prop");
  if (!prop || !prop->is_struct()) return std::nullopt;
  const Value* cur = prop->field("currCtfComp");
  const Value* dest = prop->field("destCtfComp");
  if (!cur || !dest || !cur->is_numeric() || !dest->is_numeric() || cur->numel() != 1 || dest->numel() != 1) return std::nullopt;
  return CtfGrade{static_cast<int>(cur->scalar()), static_cast<int>(dest->scalar())};
}

std::vector<std::string> export_projectors(nxr::io::SubjectWriter& sub, const std::vector<ProjectorEntry>& entries, const std::string& name, const std::string& channels_id,
                                           const std::string& acquisition) {
  std::vector<std::string> names;
  for (const auto& e : entries) {
    const Mat p = projector_matrix(e);
    const std::string nm = name + "_projector_" + std::to_string(e.order);
    nxr::io::OperatorSpec op;
    op.name = nm;
    op.path = nm;
    op.kind = "projector";
    op.from_manifold_id = channels_id;
    op.layout = "dense";
    op.n_rows = static_cast<std::int64_t>(e.m);
    op.n_cols = static_cast<std::int64_t>(e.m);
    op.data = view(p, {static_cast<std::int64_t>(e.m), static_cast<std::int64_t>(e.m)});
    op.acquisition = acquisition;
    Json comps = Json::array();
    for (std::size_t j = 0; j < e.k; ++j) if (e.mask[j]) comps.push_back(static_cast<std::int64_t>(j));
    op.cols.set("to_manifold_id", channels_id);
    Json pj{{"status", e.status}, {"comment", e.comment}, {"method", e.method}, {"kind", e.kind}, {"components", comps}, {"order", e.order}};
    if (e.kind != "ica") {                                         // an SSP's removed vectors, as the file holds them: the references compose projectors from these
      Json vecs = Json::array();
      for (std::size_t j = 0; j < e.k; ++j) {
        if (!e.mask[j]) continue;
        std::vector<double> v(e.m);
        for (std::size_t i = 0; i < e.m; ++i) v[i] = e.components[i * e.k + j];
        vecs.push_back(v);
      }
      pj["vectors"] = vecs;
    }
    op.cols.json("params_json", pj);
    op.cols.set("description", "Brainstorm projector " + std::to_string(e.order) + " (" + upper(e.kind) + "): x -> P x on the channels; status " + std::to_string(e.status) +
                                   " (0 inactive, 1 active, 2 already applied to the samples)");
    sub.operator_(std::move(op));
    names.push_back(nm);
  }
  return names;
}

std::optional<std::string> export_compensation(nxr::io::SubjectWriter& sub, const std::map<std::string, Value>& channel_mat, const std::string& name, const std::string& channels_id,
                                               const std::string& acquisition, const std::optional<CtfGrade>& grade) {
  const auto ci = channel_mat.find("MegRefCoef");
  if (ci == channel_mat.end() || !ci->second.is_numeric() || ci->second.numel() == 0) return std::nullopt;
  const auto chi = channel_mat.find("Channel");
  if (chi == channel_mat.end() || !chi->second.is_struct()) throw nxr::io::Error(code::invalid_argument, "Channel: not a struct array");
  const Value& ch = chi->second;
  const std::size_t n = ch.numel();
  std::vector<std::size_t> meg, ref;
  for (std::size_t i = 0; i < n; ++i) {
    const std::string t = upper(strip(text_of(ch.field("Type", i))));
    if (t == "MEG") meg.push_back(i);
    else if (t == "MEG REF") ref.push_back(i);
  }
  const Value& cv = ci->second;
  if (cv.rows() != meg.size() || cv.numel() != meg.size() * ref.size())
    throw nxr::io::Error(code::invalid_argument, "MegRefCoef does not match the channel file's MEG and MEG REF channels");
  const Mat c = cv.row_major();
  std::vector<std::int32_t> indptr{0}, indices;
  std::vector<double> vals;
  std::size_t next = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (next < meg.size() && meg[next] == i) {
      for (std::size_t j = 0; j < ref.size(); ++j)
        if (c[next * ref.size() + j] != 0.0) { indices.push_back(static_cast<std::int32_t>(ref[j])); vals.push_back(c[next * ref.size() + j]); }
      ++next;
    }
    indptr.push_back(static_cast<std::int32_t>(vals.size()));
  }
  const std::string nm = name + "_ctf_compensation";
  const std::int64_t N = static_cast<std::int64_t>(n), nnz = static_cast<std::int64_t>(vals.size());
  nxr::io::OperatorSpec op;
  op.name = nm;
  op.path = nm;
  op.kind = "ctf compensation";
  op.from_manifold_id = channels_id;
  op.layout = "csr";
  op.n_rows = N;
  op.n_cols = N;
  op.acquisition = acquisition;
  Json params{{"n_meg", static_cast<std::int64_t>(meg.size())}, {"n_ref", static_cast<std::int64_t>(ref.size())}};
  if (grade) params["grade"] = Json{{"current", grade->current}, {"destination", grade->destination}};
  op.cols.set("to_manifold_id", channels_id).set("nnz", nnz);
  op.cols.json("params_json", params);
  op.cols.set("description", "CTF third-order compensation: entry (MEG channel, reference channel) = MegRefCoef, so a compensated gain is G - C G (rows MEG, columns reference), i.e. (I - C) G");
  op.sparse = nxr::io::SparseArrays{{{"indptr", view(indptr, {N + 1})}, {"indices", view(indices, {nnz})}, {"data", view(vals, {nnz})}}};
  sub.operator_(std::move(op));
  return nm;
}

Json recording_record(const std::vector<ProjectorEntry>& entries, const std::map<std::string, Value>& channel_mat, const std::string& name, const std::optional<CtfGrade>& grade) {
  Json out = Json::object();
  if (!entries.empty()) {
    Json rec = Json::array();
    for (const auto& e : entries) {
      const char* applied = e.status == 1 ? "applied on load" : e.status == 2 ? "already applied to the samples" : "not applied";
      rec.push_back(Json{{"order", e.order}, {"comment", e.comment}, {"method", e.method}, {"status", e.status}, {"applied", applied},
                         {"operator", name + "_projector_" + std::to_string(e.order)}});
    }
    out["projectors"] = rec;
  }
  const auto ci = channel_mat.find("MegRefCoef");
  if (ci != channel_mat.end() && ci->second.is_numeric() && ci->second.numel() > 0) {
    Json c{{"operator", name + "_ctf_compensation"}};
    if (grade) c["grade"] = Json{{"current", grade->current}, {"destination", grade->destination}};
    out["ctf_compensation"] = c;
  }
  return out;
}

}  // namespace nxr::bst
