// nxr::bst — nxr.convert.projectors: the channel file's projectors and CTF compensation (register D168).
// Products are NOT fused with the sums that follow them: nxr's numpy rounds each.
#pragma STDC FP_CONTRACT OFF
#include "nxr/bst/projectors.h"

#include <algorithm>
#include <cmath>

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

std::vector<double> applied_matrix(const std::vector<ProjectorEntry>& entries) {
  Mat out;
  std::size_t m = 0;
  for (const auto& e : entries) {
    if (e.status != 1) continue;
    const Mat p = projector_matrix(e);
    if (out.empty()) { out = p; m = e.m; continue; }
    Mat s(m * m, 0.0);
    for (std::size_t l = 0; l < m; ++l)
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t c = 0; c < m; ++c) {
          const double pr = p[i * m + l] * out[l * m + c];
          s[i * m + c] = s[i * m + c] + pr;
        }
    out = std::move(s);
  }
  return out;
}

void project(const std::vector<double>& p, std::size_t m, const std::vector<float>& x, std::size_t n, std::vector<float>& y) {
  y = x;
  std::vector<std::size_t> act;
  for (std::size_t i = 0; i < m; ++i) {
    bool touched = false;
    for (std::size_t l = 0; l < m && !touched; ++l) {
      const double e = i == l ? 1.0 : 0.0;
      touched = p[i * m + l] != e || p[l * m + i] != e;
    }
    if (touched) act.push_back(i);
  }
  if (act.empty()) return;
  std::vector<double> acc(act.size() * n, 0.0);
  for (std::size_t j = 0; j < act.size(); ++j) {
    const std::size_t l = act[j];
    for (std::size_t r = 0; r < act.size(); ++r) {
      const double c = p[act[r] * m + l];
      for (std::size_t t = 0; t < n; ++t) {
        const double pr = c * static_cast<double>(x[l * n + t]);
        acc[r * n + t] = acc[r * n + t] + pr;
      }
    }
  }
  for (std::size_t r = 0; r < act.size(); ++r)
    for (std::size_t t = 0; t < n; ++t) y[act[r] * n + t] = static_cast<float>(acc[r * n + t]);
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
    op.cols.json("params_json", Json{{"status", e.status}, {"comment", e.comment}, {"method", e.method}, {"kind", e.kind}, {"components", comps}, {"order", e.order}});
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
      const char* applied = e.status == 1 ? "applied by the import" : e.status == 2 ? "already applied to the samples" : "not applied";
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
