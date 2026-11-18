function s = stripeFreq(mctx, vertexFieldRaw, frequencies, connectOnSingularities)
%STRIPEFREQ  Stripe pattern with per-vertex target frequencies.
%   s = nxr.manifold.uv.stripeFreq(mctx, vertexFieldRaw, frequencies)
%   s = nxr.manifold.uv.stripeFreq(mctx, vertexFieldRaw, frequencies, connectOnSingularities)
%
%   `frequencies` is a length-V vector of per-vertex stripe frequencies.
    if nargin < 4
        connectOnSingularities = true;
    end
    s = nxr_compute('geometry.computeFreq', mctx.surface, ...
                    vertexFieldRaw, frequencies, ...
                    logical(connectOnSingularities));
end
