function eig = eigen(mctx, k, sigma)
%EIGEN  Generalised eigenproblem K φ = λ M φ for the k smallest modes.
%
%   eig = nxr.manifold.solve.eigen(mctx, k)
%   eig = nxr.manifold.solve.eigen(mctx, k, sigma)
%
%   Returns a struct with fields `eigenvectors` (nV × k), `eigenvalues`
%   (k × 1, sorted ascending), and `nV`. By default (sigma omitted or NaN)
%   the solve uses the scale-aware shift ladder (sigma = -1e-7*lmax, then
%   -1e-4*lmax), never a fixed shift on the null space; an explicit sigma
%   is honoured as a fixed shift.
    if nargin < 3 || isempty(sigma)
        sigma = NaN;
    end
    eig = nxr_compute('operators.eigen', mctx.K, mctx.M, k, sigma);
end
