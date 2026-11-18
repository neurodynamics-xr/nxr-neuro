#pragma once
// nxr::signal gwt / igwt — the wavelet transform of a field on a graph (a vertex field through a GraphFilterbank), exact through the operator's eigenbasis or by a Chebyshev series in the pencil (L, M).
// A graph bank only DESIGNS (nxr/signal/graphfilterbank.h); these are the transforms (Rheome's wt / iwt of @graphfilterbank, GSPBox's gsp_filter_analysis / synthesis).
#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/graphtransforms.h"

namespace nxr::signal {

/** W_m = Φ g_m(Λ) Φᵀ B F, one [n × nT] block per member (EXACT: the operator's modes). Trailing time/trial columns pass through untouched; a complex field is transformed part by part. */
graphfilterbank::Coefficients gwt(const graphfilterbank::GraphFilterbank& bank, const graphtransforms::EigenTransform& t, const Eigen::MatrixXd& F);
graphfilterbank::CoefficientsC gwt(const graphfilterbank::GraphFilterbank& bank, const graphtransforms::EigenTransform& t, const Eigen::MatrixXcd& F);
/** W_m = g_m(M⁻¹L) F, by the series. */
graphfilterbank::Coefficients gwt(const graphfilterbank::GraphFilterbank& bank, const graphtransforms::ChebyshevOperator& t, const Eigen::MatrixXd& F);
graphfilterbank::CoefficientsC gwt(const graphfilterbank::GraphFilterbank& bank, const graphtransforms::ChebyshevOperator& t, const Eigen::MatrixXcd& F);
/** ONE member's band: W_m alone (the spatial band of a level: member l of `weyl` is the level l's band), without the other members' cost. Throws on a member outside the bank. */
Eigen::MatrixXd gwt(const graphfilterbank::GraphFilterbank& bank, int member, const graphtransforms::EigenTransform& t, const Eigen::MatrixXd& F);
Eigen::MatrixXd gwt(const graphfilterbank::GraphFilterbank& bank, int member, const graphtransforms::ChebyshevOperator& t, const Eigen::MatrixXd& F);

/** F = Σ_m w_m(L) W_m (exact eigenbasis). */
Eigen::MatrixXd igwt(const graphfilterbank::GraphFilterbank& bank, const graphtransforms::EigenTransform& t, const graphfilterbank::Coefficients& W, graphfilterbank::Synthesis mode = graphfilterbank::Synthesis::Dual);
Eigen::MatrixXcd igwt(const graphfilterbank::GraphFilterbank& bank, const graphtransforms::EigenTransform& t, const graphfilterbank::CoefficientsC& W, graphfilterbank::Synthesis mode = graphfilterbank::Synthesis::Dual);
/** F = Σ_m w_m(M⁻¹L) W_m; the dual weights g/S are themselves expanded in Chebyshev series (S must stay positive on [0, λmax]). */
Eigen::MatrixXd igwt(const graphfilterbank::GraphFilterbank& bank, const graphtransforms::ChebyshevOperator& t, const graphfilterbank::Coefficients& W, graphfilterbank::Synthesis mode = graphfilterbank::Synthesis::Dual);
Eigen::MatrixXcd igwt(const graphfilterbank::GraphFilterbank& bank, const graphtransforms::ChebyshevOperator& t, const graphfilterbank::CoefficientsC& W, graphfilterbank::Synthesis mode = graphfilterbank::Synthesis::Dual);

}  // namespace nxr::signal
