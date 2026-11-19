#include "nxr/signal/intrinsic_forms.h"
#include "nxr/operators/dec.h"
#include "geometrycentral/surface/intrinsic_geometry_interface.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::signal {

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
Eigen::Vector3d row3(const MatrixXd& X, int v) { return X.row(v).transpose(); }
// the vertex's tangent basis: e1 the reference neighbour's direction projected on the tangent plane, e2 = n × e1
void basisAt(const Complex& c, int v, Eigen::Vector3d& e1, Eigen::Vector3d& e2) {     // an orthonormal basis of the vertex's tangent plane
    const Eigen::Vector3d n = row3(c.normals, v);
    const Eigen::Vector3d a = std::abs(n.x()) < 0.9 ? Eigen::Vector3d(1, 0, 0) : Eigen::Vector3d(0, 1, 0);
    e1 = (a - a.dot(n) * n).normalized();
    e2 = n.cross(e1);
}
}  // namespace

Complex complexOf(const nxr::geometry::SurfaceArrays& arrays) {
    nxr::geometry::IntrinsicSurface surface(arrays);
    const auto K = surface.complex();
    Complex c;
    c.nV = surface.nV(); c.nE = static_cast<int>(K.length.size()); c.nF = static_cast<int>(K.faces.size() / 3);
    c.edges = K.edges;
    // d0, d1, ⋆1 (the cotangent weights) and ⋆2 (1 / area) are the operators module's DEC of the triangulation; ⋆0 its recorded mass
    const nxr::operators::DecMatrices dec = nxr::operators::decMatrices(arrays);
    c.d0 = dec.d0; c.d1 = dec.d1;
    c.length = Eigen::Map<const VectorXd>(K.length.data(), c.nE);
    c.star0 = nxr::operators::vertexMass(arrays, nxr::operators::MassMatrixVariant::MixedVoronoi).diagonal();
    c.star1 = dec.hodge1.diagonal();
    c.star2 = dec.hodge2.diagonal();
    c.angleTail = VectorXd::Constant(c.nE, std::numeric_limits<double>::quiet_NaN());
    c.angleTip = c.angleTail;
    for (int h = 0; h < 3 * c.nF; ++h) {
        const int se = K.faceEdges[static_cast<std::size_t>(h)], e = std::abs(se) - 1;
        (se > 0 ? c.angleTail : c.angleTip)(e) = K.angle[static_cast<std::size_t>(h)];
    }
    c.coneAngle = Eigen::Map<const VectorXd>(K.angleSum.data(), c.nV);
    // a boundary edge has one face: its halfedge runs along the edge (tail -> tip), and the end at its tip has no corner. Its direction there is the
    // last of the tip's fan, at the tip's angle sum (a boundary vertex's angles run from its first boundary edge, 0, to its other one, the angle sum)
    for (int e = 0; e < c.nE; ++e) {
        if (std::isnan(c.angleTip(e))) c.angleTip(e) = c.coneAngle(c.edges[static_cast<std::size_t>(2 * e + 1)]);
        if (std::isnan(c.angleTail(e))) c.angleTail(e) = c.coneAngle(c.edges[static_cast<std::size_t>(2 * e)]);
    }
    c.reference.assign(K.reference.begin(), K.reference.end());
    attachEmbedding(c, surface);
    return c;
}

void attachEmbedding(Complex& c, const double* positions, const std::int32_t* inputFaces, int nFin) {
    // the input as it is (the same vertices; its own signposts: the same reference halfedges, the same cone angles as the intrinsic mesh's)
    nxr::geometry::IntrinsicOptions o;
    o.intrinsic = false;
    nxr::geometry::IntrinsicSurface input(positions, c.nV, inputFaces, nFin, o);
    attachEmbedding(c, input);
}

void attachEmbedding(Complex& c, nxr::geometry::IntrinsicSurface& surface) {
    namespace gcs = geometrycentral::surface;
    auto& E = surface.embedded();
    E.requireVertexNormals();
    E.requireVertexPositions();
    c.positions = MatrixXd(c.nV, 3);
    c.normals = MatrixXd(c.nV, 3);
    for (gcs::Vertex v : surface.inputMesh().vertices()) {
        const auto i = static_cast<Eigen::Index>(v.getIndex());
        c.positions.row(i) << E.vertexPositions[v].x, E.vertexPositions[v].y, E.vertexPositions[v].z;
        c.normals.row(i) << E.vertexNormals[v].x, E.vertexNormals[v].y, E.vertexNormals[v].z;
    }
    const std::vector<double> ang = surface.inputSignpostAngles();
    c.fans.assign(static_cast<std::size_t>(c.nV), {});
    for (gcs::Halfedge he : surface.inputMesh().halfedges()) {
        const int v = static_cast<int>(he.vertex().getIndex()), w = static_cast<int>(he.tipVertex().getIndex());
        const Eigen::Vector3d n = row3(c.normals, v);
        Eigen::Vector3d d = row3(c.positions, w) - row3(c.positions, v);
        d -= d.dot(n) * n;
        c.fans[static_cast<std::size_t>(v)].push_back({ang[he.getIndex()], d.normalized()});
    }
    for (auto& fan : c.fans) std::sort(fan.begin(), fan.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
}


Eigen::Vector3d tangentAt(const Complex& c, int e, bool atTip) {
    const int v = c.edges[static_cast<std::size_t>(2 * e + (atTip ? 1 : 0))];
    const auto& fan = c.fans[static_cast<std::size_t>(v)];
    if (fan.empty()) throw std::invalid_argument("intrinsic: no embedding of the signposts (attachEmbedding)");
    const double theta = atTip ? c.angleTip(e) : c.angleTail(e), cone = c.coneAngle(v);
    // the input edges bracketing theta (cyclically)
    std::size_t i = 0;
    while (i + 1 < fan.size() && fan[i + 1].first <= theta) ++i;
    const auto& lo = fan[i];
    const auto& hi = i + 1 < fan.size() ? fan[i + 1] : fan[0];
    double span = hi.first - lo.first, off = theta - lo.first;
    if (i + 1 >= fan.size()) span += cone;
    // an angle within rounding of an input edge's IS that edge (exact): at a boundary vertex the last input edge sits at the angle sum, and an intrinsic
    // edge there (the boundary edge, at the cone angle) must not wrap round through the exterior, where the boundary vertex's fan has no gap to cross
    if (std::abs(off) <= 1e-9) return atTip ? Eigen::Vector3d(-lo.second) : lo.second;
    const double s = span > 0 ? off / span : 0.0;
    // rotate lo toward hi in the tangent plane by s of the angle between them (the actual angle, not the signpost's)
    const Eigen::Vector3d n = row3(c.normals, v);
    double phi = std::atan2(n.dot(lo.second.cross(hi.second)), lo.second.dot(hi.second));
    if (phi <= 0) phi += 2 * M_PI;                     // counter-clockwise from lo to hi
    const double a = s * phi;
    const Eigen::Vector3d t = std::cos(a) * lo.second + std::sin(a) * n.cross(lo.second);
    return atTip ? Eigen::Vector3d(-t) : t;           // the tip's signpost is the twin's (pointing tip → tail): reversed to point along the edge
}

MatrixXd flat(const Complex& c, const MatrixXd& J) {
    if (J.rows() != 3 * c.nV) throw std::invalid_argument("intrinsic::flat: J has " + std::to_string(J.rows()) + " rows, 3 x " + std::to_string(c.nV) + " expected");
    MatrixXd w(c.nE, J.cols());
    for (int e = 0; e < c.nE; ++e) {
        const int a = c.edges[static_cast<std::size_t>(2 * e)], b = c.edges[static_cast<std::size_t>(2 * e + 1)];
        const Eigen::Vector3d ta = tangentAt(c, e, false), tb = tangentAt(c, e, true);
        w.row(e) = 0.5 * c.length(e) * (ta.transpose() * J.middleRows(3 * a, 3) + tb.transpose() * J.middleRows(3 * b, 3));
    }
    return w;
}

MatrixXd sharp(const Complex& c, const MatrixXd& w) {
    if (w.rows() != c.nE) throw std::invalid_argument("intrinsic::sharp: a 1-form of " + std::to_string(c.nE) + " edges expected");
    // per vertex: Σ_e (t_e t_eᵀ) v = Σ_e t_e (ω_e / L_e), on the tangent plane (2 x 2 in the vertex's basis)
    std::vector<Eigen::Matrix2d> A(static_cast<std::size_t>(c.nV), Eigen::Matrix2d::Zero());
    std::vector<MatrixXd> rhs(static_cast<std::size_t>(c.nV), MatrixXd::Zero(2, w.cols()));
    std::vector<Eigen::Vector3d> E1(static_cast<std::size_t>(c.nV)), E2(static_cast<std::size_t>(c.nV));
    for (int v = 0; v < c.nV; ++v) basisAt(c, v, E1[static_cast<std::size_t>(v)], E2[static_cast<std::size_t>(v)]);
    for (int e = 0; e < c.nE; ++e)
        for (int end = 0; end < 2; ++end) {
            const int v = c.edges[static_cast<std::size_t>(2 * e + end)];
            const Eigen::Vector3d t = tangentAt(c, e, end == 1);
            const Eigen::Vector2d t2(t.dot(E1[static_cast<std::size_t>(v)]), t.dot(E2[static_cast<std::size_t>(v)]));
            A[static_cast<std::size_t>(v)] += t2 * t2.transpose();
            rhs[static_cast<std::size_t>(v)] += t2 * (w.row(e) / c.length(e));
        }
    MatrixXd out(3 * c.nV, w.cols());
    for (int v = 0; v < c.nV; ++v) {
        const MatrixXd x = A[static_cast<std::size_t>(v)].ldlt().solve(rhs[static_cast<std::size_t>(v)]);
        out.middleRows(3 * v, 3) = E1[static_cast<std::size_t>(v)] * x.row(0) + E2[static_cast<std::size_t>(v)] * x.row(1);
    }
    return out;
}

MatrixXd area2form(const Complex& c, const MatrixXd& rho) {
    // a face's vertices: from d1 (its edges) and d0 (their ends); a face's area: 1 / ⋆2
    MatrixXd s = MatrixXd::Zero(c.nF, rho.cols());
    const Eigen::SparseMatrix<double, Eigen::RowMajor> D1(c.d1);
    for (int f = 0; f < c.nF; ++f) {
        std::vector<int> vs;
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(D1, f); it; ++it)
            for (int end = 0; end < 2; ++end) { const int v = c.edges[static_cast<std::size_t>(2 * it.col() + end)]; bool seen = false; for (int x : vs) seen |= x == v; if (!seen) vs.push_back(v); }
        for (int v : vs) s.row(f) += rho.row(v);
        s.row(f) *= (1.0 / c.star2(f)) / static_cast<double>(vs.size());
    }
    return s;
}

MatrixXd vertexDensity(const Complex& c, const MatrixXd& sigma) {
    MatrixXd out = MatrixXd::Zero(c.nV, sigma.cols());
    VectorXd wsum = VectorXd::Zero(c.nV);
    const Eigen::SparseMatrix<double, Eigen::RowMajor> D1(c.d1);
    for (int f = 0; f < c.nF; ++f) {
        const double area = 1.0 / c.star2(f);
        std::vector<int> vs;
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(D1, f); it; ++it)
            for (int end = 0; end < 2; ++end) { const int v = c.edges[static_cast<std::size_t>(2 * it.col() + end)]; bool seen = false; for (int x : vs) seen |= x == v; if (!seen) vs.push_back(v); }
        for (int v : vs) { out.row(v) += sigma.row(f); wsum(v) += area; }    // the density × area = σ_f, summed; ÷ the area round the vertex
    }
    for (int v = 0; v < c.nV; ++v) if (wsum(v) > 0) out.row(v) /= wsum(v);
    return out;
}

VectorXd vertexOutflow(const Complex& c, const VectorXd& u) {
  if (u.size() != c.nE) throw std::invalid_argument("vertexOutflow: u has one value an intrinsic edge");
  return -(c.d0.transpose() * (c.star1.asDiagonal() * u));                    // the outflow of each vertex's region (the divergence times mass, negated)
}

VectorXd vertexCirculation(const Complex& c, const VectorXd& w) {
  if (w.size() != c.nE) throw std::invalid_argument("vertexCirculation: w has one value an intrinsic edge");
  const VectorXd curl = c.d1 * w;                                            // each face's circulation
  VectorXd out = VectorXd::Zero(c.nV);
  const Eigen::SparseMatrix<double, Eigen::RowMajor> D1(c.d1);
  for (int f = 0; f < c.nF; ++f) {
    std::vector<int> vs;
    for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(D1, f); it; ++it)
      for (int end = 0; end < 2; ++end) { const int v = c.edges[static_cast<std::size_t>(2 * it.col() + end)]; bool seen = false; for (int x : vs) seen |= x == v; if (!seen) vs.push_back(v); }
    for (int v : vs) out(v) += curl(f) / static_cast<double>(vs.size());
  }
  return out;
}

MatrixXd read1(const Complex& c, const Modes& m, const MatrixXd& w) {
    const MatrixXd s1w = c.star1.asDiagonal() * w;                       // ⋆1 ω
    const MatrixXd ex = m.P.transpose() * (c.d0.transpose() * s1w);       // ⟨dP_k, ω⟩⋆1
    const MatrixXd co = m.X2.transpose() * (c.star2.asDiagonal() * (c.d1 * w));   // ⟨⋆1⁻¹ d1ᵀ ⋆2 X2_k, ω⟩⋆1 = X2_kᵀ ⋆2 d1 ω
    MatrixXd out(m.mu.size(), w.cols());
    for (Eigen::Index k = 0; k < m.mu.size(); ++k) out.row(k) = 2.0 * (m.kind[static_cast<std::size_t>(k)] == 0 ? ex.row(k) : co.row(k)) / m.mu(k);
    return out;
}

MatrixXd synth1(const Complex& c, const Modes& m, const MatrixXd& coef) {
    MatrixXd ce = coef, cc = coef;
    for (Eigen::Index k = 0; k < m.mu.size(); ++k) {
        const bool exact = m.kind[static_cast<std::size_t>(k)] == 0;
        ce.row(k) = exact ? MatrixXd(coef.row(k) / m.mu(k)) : MatrixXd::Zero(1, coef.cols());
        cc.row(k) = exact ? MatrixXd::Zero(1, coef.cols()) : MatrixXd(coef.row(k) / m.mu(k));
    }
    return c.d0 * (m.P * ce) + c.star1.cwiseInverse().asDiagonal() * (c.d1.transpose() * (c.star2.asDiagonal() * (m.X2 * cc)));
}

MatrixXd read2(const Complex& c, const Modes& m, const MatrixXd& sigma) {
    MatrixXd out = 2.0 * (m.X2.transpose() * (c.star2.asDiagonal() * sigma));
    for (Eigen::Index k = 0; k < m.mu.size(); ++k) if (m.kind[static_cast<std::size_t>(k)] == 0) out.row(k).setZero();
    return out;
}

MatrixXd synth2(const Complex& c, const Modes& m, const MatrixXd& coef) {
    (void)c;
    MatrixXd cc = coef;
    for (Eigen::Index k = 0; k < m.mu.size(); ++k) if (m.kind[static_cast<std::size_t>(k)] == 0) cc.row(k).setZero();
    return m.X2 * cc;
}

MatrixXd potential(const Modes& m, const MatrixXd& coef) {
    MatrixXd ce = MatrixXd::Zero(coef.rows(), coef.cols());
    for (Eigen::Index k = 0; k < m.mu.size(); ++k) if (m.kind[static_cast<std::size_t>(k)] == 0) ce.row(k) = coef.row(k) / m.mu(k);
    return m.P * ce;
}

Eigen::VectorXd weightedPotentialMesh(const Complex& c, const VectorXd& w, const VectorXd& omega) {
    if (w.size() != c.nE || omega.size() != c.nE) throw std::invalid_argument("intrinsic::weightedPotentialMesh: weights and the 1-form on the " + std::to_string(c.nE) + " edges");
    const VectorXd W = c.star1.cwiseProduct(w);
    Eigen::SparseMatrix<double> L = c.d0.transpose() * W.asDiagonal() * c.d0;
    const double eps = 1e-10 * L.diagonal().mean() / c.star0.mean();
    for (int v = 0; v < c.nV; ++v) L.coeffRef(v, v) += eps * c.star0(v);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(L);
    if (ldlt.info() != Eigen::Success) throw std::runtime_error("intrinsic::weightedPotentialMesh: the weighted Laplacian could not be factored");
    const VectorXd phi = ldlt.solve(c.d0.transpose() * W.cwiseProduct(omega));
    return phi.array() - phi.dot(c.star0) / c.star0.sum();
}

Eigen::VectorXd weightedPotentialModes(const Complex& c, const Modes& m, const std::vector<char>& use, const VectorXd& w, const VectorXd& omega) {
    std::vector<int> ks;
    for (Eigen::Index k = 0; k < m.mu.size(); ++k) if (use[static_cast<std::size_t>(k)] && m.kind[static_cast<std::size_t>(k)] == 0) ks.push_back(static_cast<int>(k));
    if (ks.empty()) return VectorXd::Zero(c.nV);
    MatrixXd Pu(c.nV, static_cast<Eigen::Index>(ks.size()));
    for (std::size_t j = 0; j < ks.size(); ++j) Pu.col(static_cast<Eigen::Index>(j)) = m.P.col(ks[j]);
    const MatrixXd A = c.d0 * Pu;                                       // [nE x n_use]
    const VectorXd W = c.star1.cwiseProduct(w);
    MatrixXd G = A.transpose() * W.asDiagonal() * A;
    G.diagonal().array() += 1e-12 * G.trace() / static_cast<double>(ks.size());
    const VectorXd a = G.ldlt().solve(A.transpose() * W.cwiseProduct(omega));
    return Pu * a;
}

PatternFields patternFields(const Complex& c, const Modes& m, const std::vector<std::pair<int, int>>& rows, const Eigen::VectorXcd& coef) {
  const Eigen::Index K = m.mu.size();
  if (static_cast<Eigen::Index>(rows.size()) != coef.size()) throw std::invalid_argument("patternFields: one coefficient a row");
  Eigen::VectorXcd cs = Eigen::VectorXcd::Zero(K), ca = Eigen::VectorXcd::Zero(K), cn = Eigen::VectorXcd::Zero(K);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const int k = rows[i].first, part = rows[i].second;
    if (k < 0 || k >= K) throw std::invalid_argument("patternFields: a row's mode is outside the modes");
    const bool coexact = m.kind[static_cast<std::size_t>(k)] == 2;
    if (part == 2) {
      if (!coexact) throw std::invalid_argument("patternFields: a normal (2-form) row of an exact mode");
      cn(k) += coef(static_cast<Eigen::Index>(i));
    } else if (part == 1) {
      (coexact ? cs : ca)(k) += coef(static_cast<Eigen::Index>(i)) / m.mu(k);
    } else {
      throw std::invalid_argument("patternFields: a row's part is 1 (tangent) or 2 (normal)");
    }
  }
  PatternFields out;
  const Eigen::MatrixXcd P = m.P.cast<std::complex<double>>();
  out.S = P * cs;
  out.A = P * ca;
  Eigen::MatrixXd re(K, 2);
  re.col(0) = cn.real();
  re.col(1) = cn.imag();
  const Eigen::MatrixXd r = vertexDensity(c, synth2(c, m, re));
  out.rho = r.col(0).cast<std::complex<double>>() + std::complex<double>(0, 1) * r.col(1).cast<std::complex<double>>();
  return out;
}

PhaseSplit splitPhase(const Complex& c, const Modes& m, const Eigen::VectorXd& w) {
  if (w.size() != c.nE) throw std::invalid_argument("splitPhase: w has one value an intrinsic edge");
  const Eigen::Index K = m.mu.size();
  PhaseSplit out;
  out.coef = read1(c, m, w).col(0);
  Eigen::VectorXd ce = out.coef, cc = out.coef;
  for (Eigen::Index k = 0; k < K; ++k) (m.kind[static_cast<std::size_t>(k)] == 0 ? cc : ce)(k) = 0;
  out.exact = synth1(c, m, ce).col(0);
  out.coexact = synth1(c, m, cc).col(0);
  const double E = w.dot(c.star1.asDiagonal() * w), Eex = out.exact.dot(c.star1.asDiagonal() * out.exact), Eco = out.coexact.dot(c.star1.asDiagonal() * out.coexact);
  out.shareTiming = E > 0 ? Eex / E : NAN;
  out.shareRotation = E > 0 ? Eco / E : NAN;
  out.shareResidual = E > 0 ? std::max(0.0, 1 - out.shareTiming - out.shareRotation) : NAN;
  out.potential = potential(m, ce).col(0);
  out.propagation = sharp(c, -out.exact);
  Eigen::VectorXd cs = cc;
  for (Eigen::Index k = 0; k < K; ++k) if (m.kind[static_cast<std::size_t>(k)] == 2) cs(k) /= m.mu(k);
  out.stream = vertexDensity(c, m.X2 * cs).col(0);
  Eigen::VectorXd num = Eigen::VectorXd::Zero(c.nV), den = Eigen::VectorXd::Zero(c.nV);
  for (int e = 0; e < c.nE; ++e) {
    const double r2 = c.star1(e) * std::pow(w(e) - out.exact(e), 2), w2 = c.star1(e) * w(e) * w(e);
    for (int end = 0; end < 2; ++end) { const int v = c.edges[2 * static_cast<std::size_t>(e) + static_cast<std::size_t>(end)]; num(v) += r2; den(v) += w2; }
  }
  out.inconsistency = num.cwiseQuotient(den);
  return out;
}

Eigen::VectorXd phaseOneForm(const Complex& c, const Eigen::VectorXcd& x) {
  if (x.size() != c.nV) throw std::invalid_argument("phaseOneForm: x has one value a vertex");
  Eigen::VectorXd w(c.nE);
  for (int e = 0; e < c.nE; ++e) w(e) = std::arg(x(c.edges[2 * static_cast<std::size_t>(e) + 1]) * std::conj(x(c.edges[2 * static_cast<std::size_t>(e)])));
  return w;
}

}  // namespace nxr::signal
