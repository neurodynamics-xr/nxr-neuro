// nxr::signal — see opticalflow.h. A port of Rheome's +dynamics/opticalflow_scalar, tangent_basis, flow_readout and +flow/apparent, activation.
#include "nxr/signal/opticalflow.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/core/errors.h"
#include "nxr/signal/fourier.h"
#include "nxr/signal/opticalflow_connection.h"

namespace nxr::signal {

using Eigen::MatrixXd;
using Eigen::VectorXd;
constexpr double EPS = 2.220446049250313e-16;   // MATLAB eps

TangentBasis tangentBasis(const RowMat3& vertexNormals, const Eigen::Vector3d& axis) {
    const bool rheome = axis == Eigen::Vector3d::Ones();
    const Eigen::Index V = vertexNormals.rows();
    TangentBasis tb;
    tb.normal = vertexNormals; tb.e1.resize(V, 3); tb.e2.resize(V, 3);
    for (Eigen::Index i = 0; i < V; ++i) {
        Eigen::Vector3d n = vertexNormals.row(i);
        n /= std::max(n.norm(), EPS);
        tb.normal.row(i) = n;
        Eigen::Vector3d e1(n[2] - n[1], n[0] - n[2], n[1] - n[0]);
        if (rheome) { if (std::abs(n.dot(Eigen::Vector3d::Ones() / std::sqrt(3.0))) > 0.97) e1 = Eigen::Vector3d(n[1], -n[0], 0); }   // normal ~ [1 1 1]
        else {
            e1 = axis.cross(n);
            if (std::abs(n.dot(axis.normalized())) > 0.97) e1 = n.cross(std::abs(n[0]) < 0.9 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY());
        }
        e1 /= std::max(e1.norm(), EPS);
        Eigen::Vector3d e2 = n.cross(e1);
        e2 /= std::max(e2.norm(), EPS);
        tb.e1.row(i) = e1; tb.e2.row(i) = e2;
    }
    return tb;
}

RowMat3 vertexNormals(const RowMat3& positions, const operators::FaceMat& faces) { return nxr::operators::FaceOperators(positions, faces, false).normals; }

// The embedded cotan Laplacian and the lumped Galerkin mass (row sums = a third of each incident face's area).
void embeddedLaplacian(const RowMat3& positions, const operators::FaceMat& faces, Eigen::SparseMatrix<double>& K, VectorXd& mass) {
    std::vector<double> P(static_cast<std::size_t>(positions.size()));
    for (Eigen::Index i = 0; i < positions.rows(); ++i) for (int d = 0; d < 3; ++d) P[static_cast<std::size_t>(3 * i + d)] = positions(i, d);
    std::vector<int32_t> F(static_cast<std::size_t>(faces.size()));
    for (Eigen::Index f = 0; f < faces.rows(); ++f) for (int d = 0; d < 3; ++d) F[static_cast<std::size_t>(3 * f + d)] = faces(f, d);
    nxr::geometry::IntrinsicOptions io;
    io.intrinsic = false;                                                 // the embedded mesh: no mollification, no flips
    nxr::geometry::IntrinsicSurface s(P.data(), static_cast<int>(positions.rows()), F.data(), static_cast<int>(faces.rows()), io);
    K = s.cotanLaplacian();
    K = (0.5 * (K + Eigen::SparseMatrix<double>(K.transpose()))).eval();
    const Eigen::SparseMatrix<double> M = s.galerkinMassMatrix();
    mass = M * VectorXd::Ones(M.cols());
}

OpticalflowScalar::OpticalflowScalar(const RowMat3& positions, const operators::FaceMat& faces, const RowMat3& normals, double alpha, double maxAbs, const FlowForm& form)
    : nV_(positions.rows()), form_(form), faces_(faces), scale_(maxAbs + EPS), fg_(positions, faces, false), tb_(tangentBasis(normals, form.frameAxis)) {
    if (form_.dataFit == DataFit::Auto) form_.dataFit = form_.regulariser == Regulariser::Connection ? DataFit::Gauge : DataFit::Triangle;
    if ((form_.regulariser == Regulariser::Connection) != (form_.dataFit == DataFit::Gauge))
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "OpticalflowScalar: the gauge data term goes with the connection regulariser, and only with it");
    if (form_.regulariser == Regulariser::Connection) { conn_ = std::make_unique<connection::ConnectionFlow>(positions, faces, alpha, scale_); return; }
    Eigen::SparseMatrix<double> K;
    embeddedLaplacian(positions, faces, K, mass_);
    const Eigen::SparseMatrix<double> aK = alpha * K;
    std::vector<Eigen::Triplet<double>> trip;                          // the smoothness, and the data's four diagonals as explicit zeros
    for (int k = 0; k < aK.outerSize(); ++k) for (Eigen::SparseMatrix<double>::InnerIterator it(aK, k); it; ++it) {
        {                                                              // the ambient Dirichlet energy: K_ij (e_k(i) . e_l(j))
            const Eigen::Vector3d e[2] = {Eigen::Vector3d(tb_.e1.row(it.row())), Eigen::Vector3d(tb_.e2.row(it.row()))};
            const Eigen::Vector3d f[2] = {Eigen::Vector3d(tb_.e1.row(it.col())), Eigen::Vector3d(tb_.e2.row(it.col()))};
            for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) trip.emplace_back(a * nV_ + it.row(), b * nV_ + it.col(), it.value() * e[a].dot(f[b]));
        }
    }
    for (Eigen::Index f = 0; f < faces.rows(); ++f) for (int p = 0; p < 3; ++p) for (int q = 0; q < 3; ++q)
        for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) trip.emplace_back(a * nV_ + faces(f, p), b * nV_ + faces(f, q), 0.0);   // every pair of a face is in the pattern
    for (Eigen::Index i = 0; i < nV_; ++i) for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) trip.emplace_back(a * nV_ + i, b * nV_ + i, 0.0);
    system_.resize(2 * nV_, 2 * nV_);
    system_.setFromTriplets(trip.begin(), trip.end());
    smoothValues_.assign(system_.valuePtr(), system_.valuePtr() + system_.nonZeros());
    slot_.resize(static_cast<std::size_t>(4 * nV_));
    for (Eigen::Index i = 0; i < nV_; ++i) for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b)
        slot_[static_cast<std::size_t>(4 * i + 2 * a + b)] = &system_.coeffRef(a * nV_ + i, b * nV_ + i) - system_.valuePtr();
    {                                                                  // the triangle data term (ambient)
        const Eigen::Index nF = faces.rows();
        area_.resize(nF);
        faceSlot_.resize(static_cast<std::size_t>(36 * nF));
        for (Eigen::Index f = 0; f < nF; ++f) {
            const Eigen::Vector3d p0 = positions.row(faces(f, 0)), p1 = positions.row(faces(f, 1)), p2 = positions.row(faces(f, 2));
            area_[f] = 0.5 * (p1 - p0).cross(p2 - p0).norm();
            for (int p = 0; p < 3; ++p) for (int q = 0; q < 3; ++q) for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b)
                faceSlot_[static_cast<std::size_t>(36 * f + 12 * p + 4 * q + 2 * a + b)] = &system_.coeffRef(a * nV_ + faces(f, p), b * nV_ + faces(f, q)) - system_.valuePtr();
        }
    }
}

OpticalflowScalar::~OpticalflowScalar() = default;

void OpticalflowScalar::assemble(const VectorXd& previous, const VectorXd& next, double scaleIn, VectorXd& rhs, double* dataScale) {
    const Eigen::Index nV = nV_;
    if (previous.size() != nV || next.size() != nV) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "OpticalflowScalar: a map has " + std::to_string(previous.size()) + " rows, the mesh " + std::to_string(nV) + " vertices");
    const double scale = std::isnan(scaleIn) ? scale_ : scaleIn;
    const VectorXd act = previous / scale, dt = next / scale - act;
    {
        // int_t (dI/dt + V . grad I)^2 with grad I the triangle's own (P1 of the previous map), V and dI/dt interpolated by the hat functions: A/12 (1 + delta_pq) r_p r_q,
        // r_p = dI_p + (a_p e1_p + b_p e2_p) . g_t
        const Eigen::Index nF = faces_.rows();
        VectorXd gf[3];
        for (int k = 0; k < 3; ++k) gf[k] = fg_.grad[k] * act;
        double* val = system_.valuePtr();
        std::copy(smoothValues_.begin(), smoothValues_.end(), val);
        rhs = VectorXd::Zero(2 * nV);
        double dsum = 0;
        for (Eigen::Index f = 0; f < nF; ++f) {
            const Eigen::Vector3d gr(gf[0][f], gf[1][f], gf[2][f]);
            double h[2][3], d[3];
            for (int p = 0; p < 3; ++p) {
                const Eigen::Index i = faces_(f, p);
                h[0][p] = gr.dot(tb_.e1.row(i).transpose()); h[1][p] = gr.dot(tb_.e2.row(i).transpose()); d[p] = dt[i];
            }
            for (int p = 0; p < 3; ++p) for (int q = 0; q < 3; ++q) {
                const double w = area_[f] / 12.0 * (p == q ? 2.0 : 1.0);
                for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) val[faceSlot_[static_cast<std::size_t>(36 * f + 12 * p + 4 * q + 2 * a + b)]] += w * h[a][p] * h[b][q];
                for (int a = 0; a < 2; ++a) rhs[a * nV + faces_(f, p)] -= w * d[q] * h[a][p];
            }
            dsum += area_[f] * (h[0][0] * h[0][0] + h[1][0] * h[1][0] + h[0][1] * h[0][1] + h[1][1] * h[1][1] + h[0][2] * h[0][2] + h[1][2] * h[1][2]) / 3.0;
        }
        if (dataScale) *dataScale = 0.5 * dsum / area_.sum();
        return;
    }
}

VectorXd OpticalflowScalar::toVelocity(const VectorXd& ab) const {
    const Eigen::Index nV = nV_;
    VectorXd out(3 * nV);
    for (Eigen::Index i = 0; i < nV; ++i) out.segment<3>(3 * i) = ab[i] * tb_.e1.row(i).transpose() + ab[nV + i] * tb_.e2.row(i).transpose();
    return out;
}

VectorXd OpticalflowScalar::step(const VectorXd& previous, const VectorXd& next) {
    if (conn_) return conn_->step(previous, next);
    VectorXd rhs;
    assemble(previous, next, std::numeric_limits<double>::quiet_NaN(), rhs);
    VectorXd ab;
    if (!analysed_) { ldlt_.analyzePattern(system_); analysed_ = true; }
    ldlt_.factorize(system_);
    if (ldlt_.info() == Eigen::Success) ab = ldlt_.solve(rhs);
    if (ldlt_.info() != Eigen::Success || !ab.allFinite()) {             // a singular system (flat map): the sparse LU, as MATLAB's backslash
        Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(system_);
        ab = lu.solve(rhs);
    }
    return toVelocity(ab);
}

void OpticalflowScalar::makePreconditioner(const PcgOptions& o, double c) {
    pre_ = system_;
    double* val = pre_.valuePtr();
    std::copy(smoothValues_.begin(), smoothValues_.end(), val);
    for (Eigen::Index i = 0; i < nV_; ++i) {
        val[slot_[static_cast<std::size_t>(4 * i)]] += c * mass_[i];
        val[slot_[static_cast<std::size_t>(4 * i + 3)]] += c * mass_[i];
    }
    preKind_ = o.preconditioner;
    if (preKind_ == PcgOptions::Preconditioner::Ldlt) { preLdlt_.analyzePattern(pre_); preLdlt_.factorize(pre_); }
    else preIc_.compute(pre_);
    preBuilt_ = true;
}

void OpticalflowScalar::buildPreconditioner(const VectorXd& map, const PcgOptions& o, double scale) {
    if (conn_) { conn_->buildPreconditioner(map, o, scale); return; }
    if (preBuilt_) return;
    VectorXd rhs; double c = 0;
    assemble(map, map, scale, rhs, &c);
    makePreconditioner(o, std::max(c, 1e-12));
}

VectorXd OpticalflowScalar::stepPcg(const VectorXd& previous, const VectorXd& next, VectorXd& ab, const PcgOptions& o, PcgInfo* info, double scale) {
    if (conn_) return conn_->stepPcg(previous, next, ab, o, info, scale);
    VectorXd rhs; double c = 0;
    assemble(previous, next, scale, rhs, &c);
    if (!preBuilt_) makePreconditioner(o, std::max(c, 1e-12));
    auto precond = [&](const VectorXd& r) { return preKind_ == PcgOptions::Preconditioner::Ldlt ? VectorXd(preLdlt_.solve(r)) : VectorXd(preIc_.solve(r)); };
    VectorXd x = ab.size() == rhs.size() ? ab : VectorXd::Zero(rhs.size());
    const double bn = rhs.norm();
    PcgInfo pi;
    if (!(bn > 0)) { x.setZero(); pi.residual = 0; }
    else {
        VectorXd r = rhs - system_ * x, z = precond(r), p = z;
        double rz = r.dot(z);
        while (r.norm() > o.tolerance * bn && pi.iterations < o.maxIterations) {
            const VectorXd Ap = system_ * p;
            const double pAp = p.dot(Ap);
            if (!(pAp > 0)) break;
            const double a = rz / pAp;
            x += a * p; r -= a * Ap;
            z = precond(r);
            const double rz2 = r.dot(z);
            p = z + (rz2 / rz) * p; rz = rz2;
            ++pi.iterations;
        }
        pi.residual = r.norm() / bn;
        pi.converged = pi.residual <= o.tolerance;
    }
    ab = x;
    if (info) *info = pi;
    return toVelocity(x);
}

MatrixXd opticalflowScalar(const MatrixXd& activityMapIn, const RowMat3& positions, const operators::FaceMat& faces, const RowMat3& normals, double alpha, const FlowForm& form) {
    const Eigen::Index nV = positions.rows(), nT = activityMapIn.cols();
    if (activityMapIn.rows() != nV) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "opticalflowScalar: the map has " + std::to_string(activityMapIn.rows()) + " rows, the mesh " + std::to_string(nV) + " vertices");
    if (nT < 2) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "opticalflowScalar: optical flow needs at least two frames");
    OpticalflowScalar flow(positions, faces, normals, alpha, activityMapIn.cwiseAbs().maxCoeff(), form);   // unit amplitude (Brainstorm's bst_opticalflow)
    MatrixXd out(3 * nV, nT - 1);
    for (Eigen::Index t = 0; t < nT - 1; ++t) out.col(t) = flow.step(activityMapIn.col(t), activityMapIn.col(t + 1));
    return out;
}

FlowReader::FlowReader(const RowMat3& positions, const operators::FaceMat& faces) : fg_(positions, faces, false) {}

void FlowReader::read(const VectorXd& velocity, VectorXd& divergence, VectorXd& vorticity, VectorXd& speed) const {
    const Eigen::Index V = fg_.nV;
    if (velocity.size() != 3 * V) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "FlowReader: the velocity has " + std::to_string(velocity.size()) + " rows, 3 x " + std::to_string(V) + " expected");
    MatrixXd J[3];
    for (int d = 0; d < 3; ++d) { J[d].resize(V, 1); for (Eigen::Index v = 0; v < V; ++v) J[d](v, 0) = velocity[3 * v + d]; }
    VectorXd d = VectorXd::Zero(fg_.nF);
    for (int k = 0; k < 3; ++k) d += fg_.grad[k] * J[k].col(0);
    divergence = fg_.toVertex * d;
    MatrixXd dummyDiv, curl;
    fg_.divCurl(J, dummyDiv, curl);                                     // curl of the tangent part
    vorticity = curl.col(0);
    speed.resize(V);
    for (Eigen::Index v = 0; v < V; ++v) speed[v] = std::sqrt(J[0](v, 0) * J[0](v, 0) + J[1](v, 0) * J[1](v, 0) + J[2](v, 0) * J[2](v, 0));
}

FlowReadout flowReadout(const MatrixXd& velocity, const RowMat3& positions, const operators::FaceMat& faces) {
    const nxr::operators::FaceOperators fg(positions, faces, false);
    const Eigen::Index V = fg.nV, T = velocity.cols();
    if (velocity.rows() != 3 * V) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "flowReadout: the velocity has " + std::to_string(velocity.rows()) + " rows, 3 x " + std::to_string(V) + " expected");
    MatrixXd J[3], Jt[3];
    for (int d = 0; d < 3; ++d) { J[d].resize(V, T); for (Eigen::Index v = 0; v < V; ++v) J[d].row(v) = velocity.row(3 * v + d); }
    FlowReadout r;
    MatrixXd d = MatrixXd::Zero(fg.nF, T);
    for (int k = 0; k < 3; ++k) d += fg.grad[k] * J[k];
    r.divergence = fg.toVertex * d;
    MatrixXd dummyDiv;
    fg.divCurl(J, dummyDiv, r.vorticity);                               // curl of the tangent part
    r.speed.resize(V, T);
    for (Eigen::Index v = 0; v < V; ++v) r.speed.row(v) = (J[0].row(v).array().square() + J[1].row(v).array().square() + J[2].row(v).array().square()).sqrt();
    return r;
}

Eigen::MatrixXcd hilbertRows(const MatrixXd& X) {
    const Eigen::Index n = X.cols();
    Eigen::MatrixXcd F = fftColumns(Eigen::MatrixXcd(X.transpose().cast<std::complex<double>>()), true);   // [T x rows]
    VectorXd h = VectorXd::Zero(n);                                      // MATLAB hilbert
    h[0] = 1;
    if (n % 2 == 0) { h.segment(1, n / 2 - 1).setConstant(2); h[n / 2] = 1; }
    else h.segment(1, (n - 1) / 2).setConstant(2);
    for (Eigen::Index j = 0; j < F.cols(); ++j) F.col(j) = F.col(j).cwiseProduct(h.cast<std::complex<double>>());
    return fftColumns(F, false).transpose();
}

std::int64_t hilbertHalo(double sfreq, double lowestHz, double cycles) {
    if (!(sfreq > 0) || !(lowestHz > 0) || !(cycles > 0)) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "hilbertHalo: the rate, the band's lowest frequency and the cycles are positive");
    return static_cast<std::int64_t>(std::ceil(cycles * sfreq / lowestHz));
}

Eigen::MatrixXcd hilbertRowsOwn(const MatrixXd& block, Eigen::Index left, Eigen::Index own) {
    if (left < 0 || own < 1 || left + own > block.cols()) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "hilbertRowsOwn: the block holds " + std::to_string(block.cols()) + " columns, the own columns are [" + std::to_string(left) + ", " + std::to_string(left + own) + ")");
    return hilbertRows(block).middleCols(left, own);
}

MatrixXd activation(const Eigen::MatrixXcd& Jin, ActivationMethod method, bool alreadyAnalytic) {
    if (Jin.rows() % 3 != 0) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "activation: J must be [3V x T] interleaved; got " + std::to_string(Jin.rows()) + " rows");
    Eigen::MatrixXcd J = Jin;
    if (method == ActivationMethod::Analytic && !alreadyAnalytic) {
        if (J.cols() < 8) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "activation: the analytic signal needs at least 8 frames");
        J = hilbertRows(Jin.real());
    }
    const Eigen::Index V = J.rows() / 3;
    MatrixXd A(V, J.cols());
    for (Eigen::Index v = 0; v < V; ++v) A.row(v) = (J.row(3 * v).array().abs2() + J.row(3 * v + 1).array().abs2() + J.row(3 * v + 2).array().abs2()).sqrt();
    return A;
}

Apparent apparent(const MatrixXd& A, const RowMat3& positions, const operators::FaceMat& faces, const ApparentOptions& o) {
    if ((A.array() < 0).any()) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "apparent: A must be a real non-negative activation map; build it with activation()");
    if (A.rows() != positions.rows()) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "apparent: A has " + std::to_string(A.rows()) + " vertices, the surface " + std::to_string(positions.rows()));
    if (A.cols() < 2) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "apparent: optical flow needs at least two frames; got " + std::to_string(A.cols()));
    if (!(o.alpha > 0)) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "apparent: alpha must be positive");
    Apparent r;
    r.velocity = opticalflowScalar(A, positions, faces, vertexNormals(positions, faces), o.alpha, o.form);
    FlowReadout rd = flowReadout(r.velocity, positions, faces);
    r.divergence = std::move(rd.divergence); r.vorticity = std::move(rd.vorticity); r.speedPerFrame = std::move(rd.speed);
    if (std::isfinite(o.rate)) r.speed = r.speedPerFrame * o.rate;
    return r;
}

MatrixXd flowCoefficients(const nxr::operators::Basis& basis, const MatrixXd& velocity) {
    const MatrixXd w = nxr::operators::hodgeWeights(basis, velocity);
    return w.bottomRows(w.rows() / 2);
}

}  // namespace nxr::signal
