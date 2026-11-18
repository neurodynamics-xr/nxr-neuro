function out = d1(mctx)
%D1  DEC edge→face derivative d1 [nF x nE sparse].
%   out = nxr.manifold.operator.d1(mctx)
    dec = nxr_compute('operators.assembleDECOperators', mctx.surface);
    out = dec.d1;
end
