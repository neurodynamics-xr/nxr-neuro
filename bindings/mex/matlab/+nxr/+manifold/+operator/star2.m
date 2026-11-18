function out = star2(mctx)
%STAR2  Hodge ★₂ — inverse face areas as a diagonal sparse [nF x nF].
%   out = nxr.manifold.operator.star2(mctx)
    dec = nxr_compute('operators.assembleDECOperators', mctx.surface);
    out = dec.hodge2;
end
