function out = heat(mctx, sources, sourceValues, timesteps, alpha, k)
%HEAT  Spectral heat diffusion over a list of timesteps.
%   out = nxr.manifold.solve.heat(mctx, sources, sourceValues, timesteps)
%   out = nxr.manifold.solve.heat(mctx, sources, sourceValues, timesteps, alpha)
%   out = nxr.manifold.solve.heat(mctx, sources, sourceValues, timesteps, alpha, k)
%
%   sources       source vertex indices (1-based)
%   sourceValues  initial values at those vertices
%   timesteps     vector of t values
%   alpha         diffusion coefficient (default 1.0)
%   k             number of eigenmodes for the spectral evolution
%                 (default min(64, nV-1))
%
%   Returns a [T x nV] time series of the diffused field.
%
%   Unlike the WASM binding — where eigenmodes persist on the context —
%   this stateless functional form runs an internal eigensolve (k modes,
%   normalized, DC removed) before the spectral diffusion. Pass k to
%   control accuracy. (nxr.manifold.measure.distance is the geodesic sibling.)
    if nargin < 5 || isempty(alpha), alpha = 1.0; end
    if nargin < 6 || isempty(k),     k = min(64, mctx.nV - 1); end
    eig = nxr_compute('operators.eigen', mctx.K, mctx.M, k, NaN, true, true);
    u0  = nxr_compute('synthesis.delta', mctx.nV, int32(sources), sourceValues);
    out = nxr_compute('synthesis.heatDiffusion', mctx.M, eig, u0, timesteps, alpha);
end
