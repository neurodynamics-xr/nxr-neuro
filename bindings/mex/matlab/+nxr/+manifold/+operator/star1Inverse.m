function out = star1Inverse(mctx)
%STAR1INVERSE  Hodge ★₁⁻¹ — inverse edge ratios, diagonal sparse [nE x nE].
%   out = nxr.manifold.operator.star1Inverse(mctx)
    dec = nxr_compute('operators.assembleDECOperators', mctx.surface);
    out = dec.hodge1Inverse;
end
