function r = pca(x, opts)
%NXR.OPS.PCA.PCA  Principal components of a table of observations x features (nxr::stats::pca), by the thin singular value decomposition.
%   r = nxr.compute.pca.pca(x)
%   r = nxr.compute.pca.pca(x, scale="unit_variance")
%
%   Each column's mean is removed; optionally each column is then divided by its standard deviation (ddof 1; the correlation matrix's components). Never
%   the p x p covariance matrix: p can be tens of thousands where n is a few hundred. Every component up to the rank of the centred table is returned.
%   Sign convention (fixed; scikit-learn's svd_flip with u_based_decision=False): each component's sign is chosen so that its entry of largest absolute
%   value (the first, if several are equal) is positive; the scores' column takes the same sign.
%   What it cannot show: a component is a direction of variance in THIS sample; with n observations at most n - 1 components exist, and the leading
%   ones of a small sample are not stable estimates of a population's.
%
%   Inputs
%     x        [n x p], n >= 2: observations by features, single or double. A non-finite value, a table with no variance, or (scale="unit_variance") a
%              constant column is refused with an error saying why.
%     scale    "none" (default): the columns keep their units (the covariance's components). "unit_variance": divided by their standard deviation
%              after centring (ddof 1; the correlation's components).
%
%   Output r (struct; the input's precision)
%     components                [k x p], unit rows, the sign convention above
%     singular_values           [1 x k], of the centred (and scaled) table, descending
%     explained_variance        [1 x k], singular value^2/(n - 1)
%     explained_variance_ratio  [1 x k], over the table's total variance (sums to 1 at full rank)
%     scores                    [n x k], U S
%     mean                      [1 x p], the columns' means removed
%     scale                     [1 x p], the divisor of each column after centring (all 1 for scale="none")
%     rank                      k
%   Matches scikit-learn's PCA(svd_solver="full") (sign aside, which the convention above fixes).
    arguments
        x {mustBeNumericOrLogical}
        opts.scale {mustBeTextScalar} = "none"
    end
    [c, s, ev, evr, sc, mu, scl, rk] = nxr_compute('compute.pca.pca', x, char(opts.scale));
    r = struct('components', c, 'singular_values', s, 'explained_variance', ev, 'explained_variance_ratio', evr, 'scores', sc, 'mean', mu, ...
               'scale', scl, 'rank', rk);
end
