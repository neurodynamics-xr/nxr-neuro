// fromOperator: Rheome's graphfilterbank.fromOperator (lambda_max by Lanczos, the bank on [0, lambda_max], the Chebyshev transform).
#include "nxr/signal/graphfilterbank.h"

#include <utility>

namespace nxr::signal::graphfilterbank {

OperatorBank fromOperator(const Eigen::SparseMatrix<double>& L, Family family, const Eigen::SparseMatrix<double>& mass, int order, const DesignOptions& options) {
    ChebyshevOperator transform(L, mass, 0.0, order);   // estimates λmax (Lanczos × 1.01) and factors the mass once
    GraphFilterbank bank = design(family, Range::fromLmax(transform.lmax()), options);
    return OperatorBank{std::move(bank), std::move(transform)};
}

}  // namespace nxr::signal::graphfilterbank
