function out = d0(mctx)
%D0  DEC vertex→edge derivative d0 [nE x nV sparse].
%   out = nxr.manifold.operator.d0(mctx)
    dec = nxr_compute('operators.assembleDECOperators', mctx.surface);
    out = dec.d0;
end
