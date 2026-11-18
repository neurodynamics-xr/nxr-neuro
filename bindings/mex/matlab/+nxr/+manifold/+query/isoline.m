function out = isoline(mctx, field, level)
%ISOLINE  Single-level isoline of a per-vertex scalar field.
%   out = nxr.manifold.query.isoline(mctx, field, level)
%
%   field  per-vertex scalar [nV x 1]; level  contour value.
%   Returns a struct {positions [2*segmentCount x 3], segmentCount}.
%   Mirrors the WASM wrapper (numLevels = 1 over [level, level]).
    out = nxr_compute('signal.isoline', mctx.surface, field, 1, level, level);
end
