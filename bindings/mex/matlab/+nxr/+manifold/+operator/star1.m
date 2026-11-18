function out = star1(mctx)
%STAR1  Hodge ★₁ — edge dual/primal length ratios, diagonal sparse [nE x nE].
%   out = nxr.manifold.operator.star1(mctx)
    dec = nxr_compute('operators.assembleDECOperators', mctx.surface);
    out = dec.hodge1;
end
