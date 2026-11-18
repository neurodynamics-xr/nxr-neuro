function eig = precompute(mctx, k)
%PRECOMPUTE  One-shot assemble + solve + normalize + remove-DC bundle.
%
%   eig = nxr.manifold.solve.precompute(mctx, k)
%
%   Equivalent to:
%       e = nxr.manifold.solve.eigen(mctx, k);
%       e.eigenvectors = nxr_compute('operators.normalize', e.eigenvectors, mctx.M);
%       e = nxr_compute('operators.removeDC', e);
%
%   One call of `operators.eigen` with its normalize and removeDC options.
    eig = nxr_compute('operators.eigen', mctx.K, mctx.M, k, NaN, true, true);
end
