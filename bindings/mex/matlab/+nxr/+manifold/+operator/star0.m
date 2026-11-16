function out = star0(mctx)
%STAR0  Hodge ★₀ as a sparse [nV x nV] matrix: the mixed Voronoi mass of the
%   mollified input (the same matrix as the default mass).
%   out = nxr.manifold.operator.star0(mctx)
    dec = nxr.manifold.impl.withHandle(mctx, @(h) nxr_compute('assembleDECOperators', h));
    out = dec.hodge0;
end
