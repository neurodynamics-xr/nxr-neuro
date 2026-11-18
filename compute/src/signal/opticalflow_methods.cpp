// nxr::signal — alternative optical-flow methods; see opticalflow_methods.h.
#include "nxr/signal/opticalflow_methods.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/opticalflow_connection.h"

namespace nxr::signal {

using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace fb = nxr::signal::graphfilterbank;
constexpr double EPS = 2.220446049250313e-16;

Regulariser parseRegulariser(const std::string& n) {
    if (n == "ambient") return Regulariser::Ambient;
    if (n == "connection") return Regulariser::Connection;
    throw std::invalid_argument("opticalflow: the regulariser '" + n + "' is not connection or ambient");
}
const char* regulariserName(Regulariser r) { return r == Regulariser::Connection ? "connection" : "ambient"; }
DataFit parseDataFit(const std::string& n) {
    if (n == "triangle") return DataFit::Triangle;
    if (n == "gauge") return DataFit::Gauge;
    if (n == "auto") return DataFit::Auto;
    throw std::invalid_argument("opticalflow: the datafit '" + n + "' is not gauge, triangle or auto");
}
const char* dataFitName(DataFit d) { return d == DataFit::Gauge ? "gauge" : d == DataFit::Triangle ? "triangle" : "auto"; }
DataFit resolveDataFit(Regulariser r, DataFit d) {
    if (d == DataFit::Auto) d = r == Regulariser::Connection ? DataFit::Gauge : DataFit::Triangle;
    if ((r == Regulariser::Connection) != (d == DataFit::Gauge))
        throw std::invalid_argument(std::string("opticalflow: the datafit '") + dataFitName(d) + "' does not go with the regulariser '" + regulariserName(r) + "' (the gauge data term is the connection regulariser's, and only its)");
    return d;
}

FlowMethod parseFlowMethod(const std::string& n) {
    if (n == "direct") return FlowMethod::Direct;
    if (n == "pcg") return FlowMethod::Pcg;
    if (n == "perscale") return FlowMethod::Perscale;
    throw std::invalid_argument("opticalflow: method '" + n + "' is not direct, pcg or perscale");
}
const char* flowMethodName(FlowMethod m) {
    switch (m) {
        case FlowMethod::Direct: return "direct";
        case FlowMethod::Pcg: return "pcg";
        case FlowMethod::Perscale: return "perscale";
    }
    return "direct";
}


struct FlowEngine::Impl {
    FlowMethodOptions o;
    RowMat3 pos; operators::FaceMat faces; RowMat3 normals;
    double alpha = 1, maxAbs = 0;
    optimisation::PcgOptions pcg;
    std::unique_ptr<OpticalflowScalar> flow;
    // perscale
    std::unique_ptr<fb::ChebyshevOperator> cheb;
    std::vector<fb::Gain> gains;                  // perscale: the selected members'
    std::vector<VectorXd> ab;                     // warm starts: per series
    std::vector<double> seriesMax;                // perscale: each member's max |filtered map|
    std::vector<double> centres;
    VectorXd cacheRaw;                            // the last `next`, and its filtered versions
    std::vector<VectorXd> cacheFilt;

    OpticalflowScalar& scalar(double scale) {
        if (!flow) flow = std::make_unique<OpticalflowScalar>(pos, faces, normals, alpha, scale, FlowForm{o.regulariser, resolveDataFit(o.regulariser, o.dataFit)});
        return *flow;
    }
    // the maps filtered by every gain: columns of X [V x 2] -> per gain [V x 2]
    std::vector<MatrixXd> filtered(const MatrixXd& X) const {
        std::vector<MatrixXd> out;
        for (const fb::Gain& g : gains) out.push_back(fb::chebyshev(*cheb, g, X));
        return out;
    }
    // previous and next filtered, reusing the last step's `next` as this step's `previous`
    std::vector<MatrixXd> filteredPair(const VectorXd& prev, const VectorXd& next) {
        const bool reuse = cacheFilt.size() == gains.size() && cacheRaw.size() == prev.size() && (cacheRaw - prev).isZero(0);
        std::vector<MatrixXd> out;
        if (reuse) {
            MatrixXd n(next.size(), 1); n.col(0) = next;
            const std::vector<MatrixXd> fn = filtered(n);
            for (std::size_t l = 0; l < gains.size(); ++l) { MatrixXd p(prev.size(), 2); p.col(0) = cacheFilt[l]; p.col(1) = fn[l].col(0); out.push_back(std::move(p)); }
        } else {
            MatrixXd X(prev.size(), 2); X.col(0) = prev; X.col(1) = next;
            out = filtered(X);
        }
        cacheRaw = next;
        cacheFilt.resize(gains.size());
        for (std::size_t l = 0; l < gains.size(); ++l) cacheFilt[l] = out[l].col(1);
        return out;
    }
};

FlowEngine::FlowEngine(const FlowMethodOptions& options, const RowMat3& positions, const operators::FaceMat& faces, const RowMat3& normals, double alpha) : p_(std::make_unique<Impl>()) {
    Impl& s = *p_;
    s.o = options; s.pos = positions; s.faces = faces; s.normals = normals; s.alpha = alpha;
    s.pcg.tolerance = options.tolerance; s.pcg.maxIterations = options.maxIterations; s.pcg.preconditioner = options.preconditioner;
    if (!(options.tolerance > 0)) throw std::invalid_argument("opticalflow: the pcg tolerance must be positive");
    if (options.maxIterations < 1) throw std::invalid_argument("opticalflow: the pcg maximum iterations must be at least 1");
    const bool needsCheb = options.method == FlowMethod::Perscale;
    if (needsCheb) {
        Eigen::SparseMatrix<double> L; VectorXd mass;
        embeddedLaplacian(positions, faces, L, mass);
        Eigen::SparseMatrix<double> M(mass.size(), mass.size());
        std::vector<Eigen::Triplet<double>> tr;
        for (Eigen::Index i = 0; i < mass.size(); ++i) tr.emplace_back(i, i, mass[i]);
        M.setFromTriplets(tr.begin(), tr.end());
        s.cheb = std::make_unique<fb::ChebyshevOperator>(L, M);
        const double lmax = s.cheb->lmax();
        fb::Family fam;
        if (options.scaleFamily == "mexhat") fam = fb::Family::MexHat;
        else if (options.scaleFamily == "itersine") fam = fb::Family::Itersine;
        else if (options.scaleFamily == "logitersine") fam = fb::Family::LogItersine;
        else throw std::invalid_argument("opticalflow perscale: the scale family is required (mexhat, itersine or logitersine)");
        fb::DesignOptions d;
        d.voices = options.scaleVoices; d.numFilters = options.scaleFilters;
        const fb::GraphFilterbank bank = fb::design(fam, fb::Range::fromLmax(lmax), d);
        const VectorXd cw = fb::centerWavenumbers(bank);
        for (int m = 0; m < bank.size(); ++m) {
            const fb::Role r = bank.members[static_cast<std::size_t>(m)].role;
            if (r == fb::Role::Scaling || r == fb::Role::Top) continue;
            s.gains.push_back(bank.members[static_cast<std::size_t>(m)].gain);
            s.centres.push_back(cw[m]);
        }
        if (s.gains.empty()) throw std::invalid_argument("opticalflow perscale: the bank has no band-pass member");
        s.ab.resize(s.gains.size());
        s.seriesMax.assign(s.gains.size(), 0.0);
    }
}
FlowEngine::~FlowEngine() = default;

int FlowEngine::series() const { return p_->o.method == FlowMethod::Perscale ? static_cast<int>(p_->gains.size()) : 1; }
std::vector<std::string> FlowEngine::seriesNames() const {
    std::vector<std::string> n;
    if (p_->o.method != FlowMethod::Perscale) return {""};
    for (int m = 0; m < series(); ++m) { char b[16]; std::snprintf(b, sizeof b, "m%02d", m); n.push_back(b); }
    return n;
}
std::vector<double> FlowEngine::seriesCentres() const { return p_->centres; }
double FlowEngine::maxAbs() const { return p_->maxAbs; }

void FlowEngine::observe(const VectorXd& map) {
    Impl& s = *p_;
    s.maxAbs = std::max(s.maxAbs, map.cwiseAbs().maxCoeff());
    if (s.o.method == FlowMethod::Perscale) {
        MatrixXd X(map.size(), 1); X.col(0) = map;
        const std::vector<MatrixXd> f = s.filtered(X);
        for (std::size_t m = 0; m < f.size(); ++m) s.seriesMax[m] = std::max(s.seriesMax[m], f[m].cwiseAbs().maxCoeff());
    }
}

FlowStep FlowEngine::step(const VectorXd& prev, const VectorXd& next) {
    Impl& s = *p_;
    const auto t0 = std::chrono::steady_clock::now();
    const double mx = s.maxAbs > 0 ? s.maxAbs : std::max(prev.cwiseAbs().maxCoeff(), next.cwiseAbs().maxCoeff()), scale = mx + EPS;
    FlowStep out;
    switch (s.o.method) {
        case FlowMethod::Direct:
            out.velocity.push_back(s.scalar(mx).step(prev, next));
            break;
        case FlowMethod::Pcg: {
            if (s.ab.empty()) s.ab.resize(1);
            optimisation::PcgInfo info;
            out.velocity.push_back(s.scalar(mx).stepPcg(prev, next, s.ab[0], s.pcg, &info));
            out.iterations.push_back(info.iterations);
            break;
        }
        case FlowMethod::Perscale: {
            OpticalflowScalar& f = s.scalar(1.0);
            const std::vector<MatrixXd> fm = s.filteredPair(prev, next);
            for (std::size_t m = 0; m < s.gains.size(); ++m) {
                optimisation::PcgInfo info;
                const double sc = s.seriesMax[m] > 0 ? s.seriesMax[m] + EPS : 1.0;
                out.velocity.push_back(f.stepPcg(fm[m].col(0), fm[m].col(1), s.ab[m], s.pcg, &info, sc));
                out.iterations.push_back(info.iterations);
            }
            break;
        }
    }
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace nxr::signal
