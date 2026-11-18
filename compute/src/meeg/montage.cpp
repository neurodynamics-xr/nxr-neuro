// nxr::montage — see include/nxr/montage.h for the rule and what it cannot resolve.
#include "nxr/meeg/montage.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace nxr::meeg {

namespace {

using V3 = Eigen::Vector3d;

[[noreturn]] void refuse(const std::string& m) { throw std::invalid_argument("montage: " + m); }

// ── the names ──
// Rows are the 5% steps along the midline, front to back (index 0 = Nz, 10 = Cz, 20 = Iz). Away from the midline, rows 7 to 13 are named differently (FFT, FT, FTT, T, TTP, TP, TPP)
// than near it (FFC, FC, FCC, C, CCP, CP, CPP).
const char* const kLateral[21] = {"", "", "", "AFp", "AF", "AFF", "F", "FFT", "FT", "FTT", "T", "TTP", "TP", "TPP", "P", "PPO", "PO", "POO", "", "", ""};
const char* const kMedial[21] = {"", "", "", "AFp", "AF", "AFF", "F", "FFC", "FC", "FCC", "C", "CCP", "CP", "CPP", "P", "PPO", "PO", "POO", "", "", ""};

std::string midline(int i) {
  if (i == 0) return "Nz";
  if (i == 1) return "NFpz";
  if (i == 2) return "Fpz";
  if (i == 18) return "Oz";
  if (i == 19) return "OIz";
  if (i == 20) return "Iz";
  return std::string(kMedial[i]) + "z";
}
// the contours that run front to back at a fixed side: 7/8 (Fpz - T7 - Oz), 9/10 (Nz - T9 - Iz) and 9h/10h (NFpz - T9h - OIz); side 1 is the left, 2 the right
std::string column(int kind, int side, int i) {
  const std::string n = std::to_string(side);
  const bool L = side == 1;
  if (kind == 0) {         // 7 / 8
    if (i == 0) return "Fpz";
    if (i == 20) return "Oz";
    if (i == 1) return "Fp" + n + "h";
    if (i == 2) return "Fp" + n;
    if (i == 18) return "O" + n;
    if (i == 19) return "O" + n + "h";
    return std::string(kLateral[i]) + (L ? "7" : "8");
  }
  if (kind == 1) {         // 9 / 10
    if (i == 0) return "Nz";
    if (i == 20) return "Iz";
    if (i == 1) return "N" + n + "h";
    if (i == 2) return "N" + n;
    if (i == 18) return "I" + n;
    if (i == 19) return "I" + n + "h";
    return std::string(kLateral[i]) + (L ? "9" : "10");
  }
  if (i == 0) return "NFpz";   // 9h / 10h
  if (i == 20) return "OIz";
  if (i == 1) return "NFp" + n + "h";
  if (i == 2) return "NFp" + n;
  if (i == 18) return "OI" + n;
  if (i == 19) return "OI" + n + "h";
  return std::string(kLateral[i]) + (L ? "9h" : "10h");
}
const char* const kSeventeen[17] = {"7", "7h", "5", "5h", "3", "3h", "1", "1h", "z", "2h", "2", "4h", "4", "6h", "6", "8h", "8"};
bool lateralToken(const std::string& t) { return t == "7" || t == "7h" || t == "8" || t == "8h" || t == "9" || t == "9h" || t == "10" || t == "10h"; }
// a coronal row (index 3..17, not 10): 17 names from the left end to the right end; the middle one is the row's midline name
std::vector<std::string> row(int i) {
  std::vector<std::string> out;
  for (const char* t : kSeventeen) {
    const std::string tok = t;
    out.push_back(tok == "z" ? midline(i) : std::string(lateralToken(tok) ? kLateral[i] : kMedial[i]) + tok);
  }
  return out;
}
// the coronal contour T9 - Cz - T10 (the row of C): 21 names
std::vector<std::string> coronal() {
  std::vector<std::string> out = {"T9", "T9h"};
  for (const char* t : kSeventeen) {
    const std::string tok = t;
    out.push_back(tok == "z" ? "Cz" : std::string(lateralToken(tok) ? "T" : "C") + tok);
  }
  out.push_back("T10h");
  out.push_back("T10");
  return out;
}
std::vector<std::string> columnNames(int kind, int side) {
  std::vector<std::string> out;
  for (int i = 0; i < 21; ++i) out.push_back(column(kind, side, i));
  return out;
}
// every contour in the order the rule draws them
std::vector<std::vector<std::string>> contours() {
  std::vector<std::vector<std::string>> c;
  std::vector<std::string> mid;
  for (int i = 0; i < 21; ++i) mid.push_back(midline(i));
  c.push_back(mid);
  c.push_back(columnNames(1, 2));
  c.push_back(columnNames(1, 1));
  c.push_back(coronal());
  c.push_back(columnNames(2, 2));
  c.push_back(columnNames(2, 1));
  c.push_back(columnNames(0, 2));
  c.push_back(columnNames(0, 1));
  for (int i = 3; i <= 17; ++i) if (i != 10) c.push_back(row(i));
  return c;
}

// ── the mesh ──
struct Chain {
  std::vector<V3> pts;
  std::vector<double> cum;   // arclength at each point
  bool closed = false;
  double total = 0;
};

V3 closestOnTriangle(const V3& p, const V3& a, const V3& b, const V3& c) {   // Ericson, Real-Time Collision Detection 5.1.5
  const V3 ab = b - a, ac = c - a, ap = p - a;
  const double d1 = ab.dot(ap), d2 = ac.dot(ap);
  if (d1 <= 0 && d2 <= 0) return a;
  const V3 bp = p - b;
  const double d3 = ab.dot(bp), d4 = ac.dot(bp);
  if (d3 >= 0 && d4 <= d3) return b;
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + ab * (d1 / (d1 - d3));
  const V3 cp = p - c;
  const double d5 = ab.dot(cp), d6 = ac.dot(cp);
  if (d6 >= 0 && d5 <= d6) return c;
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + ac * (d2 / (d2 - d6));
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
  const double denom = 1.0 / (va + vb + vc);
  return a + ab * (vb * denom) + ac * (vc * denom);
}

class Scalp {
 public:
  Scalp(const std::vector<double>& v, const std::vector<std::int32_t>& f) : v_(v), f_(f) {
    if (v.size() % 3 != 0 || v.size() < 9) refuse("the scalp's vertices are [V x 3]");
    if (f.size() % 3 != 0 || f.empty()) refuse("the scalp's faces are [F x 3]");
    nv_ = static_cast<std::int64_t>(v.size() / 3);
    for (const auto i : f) if (i < 0 || i >= nv_) refuse("a face names a vertex that does not exist");
  }
  V3 vertex(std::int64_t i) const { return V3(v_[3 * i], v_[3 * i + 1], v_[3 * i + 2]); }
  std::int64_t nVertices() const { return nv_; }
  V3 closestPoint(const V3& p) const {
    V3 best = vertex(0);
    double bd = std::numeric_limits<double>::infinity();
    for (std::size_t t = 0; t < f_.size(); t += 3) {
      const V3 q = closestOnTriangle(p, vertex(f_[t]), vertex(f_[t + 1]), vertex(f_[t + 2]));
      const double d = (q - p).squaredNorm();
      if (d < bd) { bd = d; best = q; }
    }
    return best;
  }
  // The scalp cut by the plane through a, b and c: the chain of the cut that passes through a.
  Chain section(const V3& a, const V3& b, const V3& c) const {
    V3 n = (b - a).cross(c - a);
    const double nn = n.norm();
    if (!(nn > 1e-12 * (b - a).norm() * (c - a).norm())) refuse("three points of a contour are on a line");
    n /= nn;
    std::vector<double> d(static_cast<std::size_t>(nv_));
    for (std::int64_t i = 0; i < nv_; ++i) d[static_cast<std::size_t>(i)] = n.dot(vertex(i) - a);
    // an edge the plane crosses is a node; a face it cuts joins two nodes. A vertex ON the plane counts as below it, so no face is degenerate.
    std::unordered_map<std::int64_t, V3> where;
    std::unordered_map<std::int64_t, std::vector<std::int64_t>> next;
    auto node = [&](std::int32_t i, std::int32_t j) {
      const std::int64_t key = static_cast<std::int64_t>(std::min(i, j)) * nv_ + std::max(i, j);
      if (!where.count(key)) {
        const double di = d[static_cast<std::size_t>(i)], dj = d[static_cast<std::size_t>(j)];
        where[key] = vertex(i) + (vertex(j) - vertex(i)) * (di / (di - dj));
      }
      return key;
    };
    for (std::size_t t = 0; t < f_.size(); t += 3) {
      const std::int32_t w[3] = {f_[t], f_[t + 1], f_[t + 2]};
      bool up[3];
      int nUp = 0;
      for (int k = 0; k < 3; ++k) { up[k] = d[static_cast<std::size_t>(w[k])] > 0; nUp += up[k]; }
      if (nUp == 0 || nUp == 3) continue;
      const int lone = nUp == 1 ? (up[0] ? 0 : up[1] ? 1 : 2) : (!up[0] ? 0 : !up[1] ? 1 : 2);   // the vertex alone on its side
      const std::int32_t x = w[lone], y = w[(lone + 1) % 3], z = w[(lone + 2) % 3];
      const std::int64_t k1 = node(x, y), k2 = node(x, z);
      next[k1].push_back(k2);
      next[k2].push_back(k1);
    }
    if (next.empty()) refuse("a contour's plane does not cut the scalp");
    for (const auto& kv : next) if (kv.second.size() > 2) refuse("a contour's cut branches (the scalp is not a manifold surface there)");
    std::set<std::int64_t> seen;
    std::vector<Chain> chains;
    auto walk = [&](std::int64_t start, bool closed) {
      Chain ch;
      ch.closed = closed;
      std::int64_t prev = -1, cur = start;
      while (true) {
        seen.insert(cur);
        ch.pts.push_back(where[cur]);
        std::int64_t nx = -1;
        for (const auto q : next[cur]) if (q != prev && !seen.count(q)) { nx = q; break; }
        if (nx < 0) break;
        prev = cur;
        cur = nx;
      }
      chains.push_back(std::move(ch));
    };
    for (const auto& kv : next) if (kv.second.size() == 1 && !seen.count(kv.first)) walk(kv.first, false);
    for (const auto& kv : next) if (!seen.count(kv.first)) walk(kv.first, true);
    for (auto& ch : chains) {
      ch.cum.assign(ch.pts.size(), 0.0);
      for (std::size_t i = 1; i < ch.pts.size(); ++i) ch.cum[i] = ch.cum[i - 1] + (ch.pts[i] - ch.pts[i - 1]).norm();
      ch.total = ch.cum.back() + (ch.closed ? (ch.pts.front() - ch.pts.back()).norm() : 0.0);
    }
    // the chain that passes through a
    std::size_t best = 0;
    double bd = std::numeric_limits<double>::infinity();
    for (std::size_t c0 = 0; c0 < chains.size(); ++c0) {
      double dist = std::numeric_limits<double>::infinity(), s = 0;
      nearestOn(chains[c0], a, dist, s);
      if (dist < bd) { bd = dist; best = c0; }
    }
    return chains[best];
  }
  static void nearestOn(const Chain& ch, const V3& p, double& dist, double& s) {
    dist = std::numeric_limits<double>::infinity();
    s = 0;
    const std::size_t n = ch.pts.size();
    const std::size_t segs = ch.closed ? n : n - 1;
    for (std::size_t i = 0; i < segs; ++i) {
      const V3& a = ch.pts[i];
      const V3& b = ch.pts[(i + 1) % n];
      const V3 ab = b - a;
      const double len2 = ab.squaredNorm();
      const double t = len2 > 0 ? std::min(1.0, std::max(0.0, (p - a).dot(ab) / len2)) : 0.0;
      const double dd = (a + ab * t - p).norm();
      if (dd < dist) { dist = dd; s = ch.cum[i] + t * std::sqrt(len2); }
    }
  }

 private:
  const std::vector<double>& v_;
  const std::vector<std::int32_t>& f_;
  std::int64_t nv_ = 0;
};

V3 pointAt(const Chain& ch, double s) {
  const std::size_t n = ch.pts.size();
  if (ch.closed) { s = std::fmod(s, ch.total); if (s < 0) s += ch.total; }
  else s = std::min(std::max(s, 0.0), ch.total);
  const auto it = std::upper_bound(ch.cum.begin(), ch.cum.end(), s);
  std::size_t i = it == ch.cum.begin() ? 0 : static_cast<std::size_t>(it - ch.cum.begin()) - 1;
  if (i >= n) i = n - 1;
  const V3& a = ch.pts[i];
  const bool wrap = i + 1 >= n;
  if (wrap && !ch.closed) return a;
  const V3& b = ch.pts[wrap ? 0 : i + 1];
  const double len = (b - a).norm();
  return len > 0 ? V3(a + (b - a) * ((s - ch.cum[i]) / len)) : a;
}

// An arc from p1 through p2 to p3 along the scalp's cut by the plane of the three, and the points at fractions of its length.
class Arc {
 public:
  Arc(const Scalp& scalp, const V3& p1, const V3& p2, const V3& p3) : ch_(scalp.section(p1, p2, p3)) {
    double dist = 0, s1 = 0, s2 = 0, s3 = 0;
    Scalp::nearestOn(ch_, p1, dist, s1);
    Scalp::nearestOn(ch_, p2, dist, s2);
    Scalp::nearestOn(ch_, p3, dist, s3);
    s1_ = s1;
    if (ch_.closed) {
      const double L = ch_.total;
      auto fwd = [&](double a, double b) { double x = std::fmod(b - a, L); return x < 0 ? x + L : x; };
      const double d12 = fwd(s1, s2), d13 = fwd(s1, s3);
      if (d12 <= d13) { dir_ = 1; len_ = d13; }
      else { dir_ = -1; len_ = L - d13; }
    } else {
      dir_ = s3 >= s1 ? 1 : -1;
      len_ = std::fabs(s3 - s1);
      if (!((s2 - s1) * dir_ >= -1e-12 && (s3 - s2) * dir_ >= -1e-12)) refuse("a contour's middle point is not between its ends (the cut is open there)");
    }
    if (!(len_ > 0)) refuse("a contour has no length");
  }
  V3 at(double fraction) const { return pointAt(ch_, s1_ + dir_ * fraction * len_); }

 private:
  Chain ch_;
  double s1_ = 0, len_ = 0;
  int dir_ = 1;
};

V3 toV(const Point& p) { return V3(p[0], p[1], p[2]); }
Point fromV(const V3& v) { return {v.x(), v.y(), v.z()}; }

}  // namespace

const std::vector<std::string>& labels1005() {
  static const std::vector<std::string> out = [] {
    std::vector<std::string> l;
    std::set<std::string> seen;
    for (const auto& c : contours()) for (const auto& s : c) if (seen.insert(s).second) l.push_back(s);
    return l;
  }();
  return out;
}

const std::vector<std::string>& labels1020() {
  static const std::vector<std::string> out = {"Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"};
  return out;
}

const std::vector<std::string>& labels1010() {
  // Nz · Fp1 Fpz Fp2 · AF7 AF3 AFz AF4 AF8 · the rows F, FT, T, TP, P from the 9 to the 10 column (11 each) · PO9 PO7 PO3 POz PO4 PO8 PO10 · O1 Oz O2 · I1 Iz I2
  static const std::vector<std::string> out = {
      "Nz", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10", "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
      "T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10", "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P9", "P7", "P5", "P3", "P1", "Pz",
      "P2", "P4", "P6", "P8", "P10", "PO9", "PO7", "PO3", "POz", "PO4", "PO8", "PO10", "O1", "Oz", "O2", "I1", "Iz", "I2"};
  return out;
}

const std::map<std::string, std::string>& oldNames() {
  static const std::map<std::string, std::string> out = {{"T3", "T7"}, {"T4", "T8"}, {"T5", "P7"}, {"T6", "P8"}};
  return out;
}

Point inionFromLandmarks(const std::vector<double>& vertices, const std::vector<std::int32_t>& faces, const Point& nasion, const Point& lpa, const Point& rpa) {
  const Scalp scalp(vertices, faces);
  const V3 mid = (toV(lpa) + toV(rpa)) / 2;
  V3 dir = mid - toV(nasion);
  if (!(dir.norm() > 0)) refuse("the nasion is at the midpoint of the preauricular points");
  dir.normalize();
  double tBest = -1;
  for (std::size_t t = 0; t < faces.size(); t += 3) {   // Moller and Trumbore (1997), the ray's origin at the midpoint of the preauricular points
    const V3 a = scalp.vertex(faces[t]), b = scalp.vertex(faces[t + 1]), c = scalp.vertex(faces[t + 2]);
    const V3 e1 = b - a, e2 = c - a, h = dir.cross(e2);
    const double det = e1.dot(h);
    if (std::fabs(det) < 1e-18) continue;
    const V3 sv = mid - a;
    const double u = sv.dot(h) / det;
    if (u < 0 || u > 1) continue;
    const V3 q = sv.cross(e1);
    const double v = dir.dot(q) / det;
    if (v < 0 || u + v > 1) continue;
    const double tt = e2.dot(q) / det;
    if (tt > tBest) tBest = tt;
  }
  if (!(tBest > 0)) refuse("the line from the nasion through the midpoint of the preauricular points does not leave the scalp behind the ears");
  return fromV(mid + dir * tBest);
}

Placement tenFive(const std::vector<double>& vertices, const std::vector<std::int32_t>& faces, const Landmarks& lm, const Options& opt) {
  const Scalp scalp(vertices, faces);
  Placement out;
  // the landmarks are the caller's: each moves to its nearest point of the scalp, and how far it moved is reported
  const V3 nz = scalp.closestPoint(toV(lm.nasion)), iz = scalp.closestPoint(toV(lm.inion)), lpa = scalp.closestPoint(toV(lm.lpa)), rpa = scalp.closestPoint(toV(lm.rpa));
  out.landmark_shift[0] = (nz - toV(lm.nasion)).norm(); out.landmark_shift[1] = (iz - toV(lm.inion)).norm();
  out.landmark_shift[2] = (lpa - toV(lm.lpa)).norm(); out.landmark_shift[3] = (rpa - toV(lm.rpa)).norm();
  out.on_scalp = Landmarks{fromV(nz), fromV(iz), fromV(lpa), fromV(rpa)};
  const double head = (nz - iz).norm();
  if (!(head > 0) || !((rpa - lpa).norm() > 0)) refuse("the landmarks coincide");
  // Cz: the point at 50% of the midline Nz - Cz - Iz and of the coronal LPA - Cz - RPA. Start at the scalp's vertex highest above the landmarks' plane, then alternate.
  V3 up = (rpa - lpa).cross(nz - iz);
  if (!(up.norm() > 0)) refuse("the landmarks are on a line");
  up.normalize();
  V3 cz = scalp.vertex(0);
  double hi = -std::numeric_limits<double>::infinity();
  for (std::int64_t i = 0; i < scalp.nVertices(); ++i) { const double h = up.dot(scalp.vertex(i)); if (h > hi) { hi = h; cz = scalp.vertex(i); } }
  bool settled = false;
  for (int it = 1; it <= opt.max_iterations; ++it) {
    const V3 s = Arc(scalp, nz, cz, iz).at(0.5);
    const V3 c = Arc(scalp, lpa, s, rpa).at(0.5);
    out.cz_residual = std::max((s - cz).norm(), (c - s).norm()) / head;
    out.iterations = it;
    cz = c;
    if (out.cz_residual < opt.tolerance) { settled = true; break; }
  }
  if (!settled) refuse("Cz did not settle in " + std::to_string(opt.max_iterations) + " iterations (it still moved " + std::to_string(out.cz_residual) + " of the nasion - inion distance)");
  // the contours, in the rule's order; a point an earlier contour made is not moved by a later one
  std::unordered_map<std::string, V3> pos;
  pos["Nz"] = nz; pos["Iz"] = iz; pos["T9"] = lpa; pos["T10"] = rpa; pos["Cz"] = cz;
  for (const auto& c : contours()) {
    const std::size_t n = c.size();
    const auto need = [&](const std::string& name) -> const V3& {
      const auto f = pos.find(name);
      if (f == pos.end()) refuse("contour of " + c.front() + " - " + c.back() + ": the point " + name + " is not placed");
      return f->second;
    };
    const Arc arc(scalp, need(c.front()), need(c[n / 2]), need(c.back()));
    for (std::size_t i = 0; i < n; ++i)
      if (!pos.count(c[i])) pos[c[i]] = arc.at(static_cast<double>(i) / static_cast<double>(n - 1));
  }
  out.labels = labels1005();
  out.positions.reserve(3 * out.labels.size());
  for (const auto& l : out.labels) { const V3& p = pos.at(l); out.positions.push_back(p.x()); out.positions.push_back(p.y()); out.positions.push_back(p.z()); }
  return out;
}

}  // namespace nxr::meeg
