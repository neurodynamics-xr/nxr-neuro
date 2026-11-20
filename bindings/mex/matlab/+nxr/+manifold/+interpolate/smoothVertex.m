function r = smoothVertex(mctx, nSym, alignToCurvature)
%SMOOTHVERTEXFIELD  Smoothest vertex-based n-RoSy direction field.
%   r = nxr.manifold.interpolate.smoothVertex(mctx)
%   r = nxr.manifold.interpolate.smoothVertex(mctx, nSym)
%   r = nxr.manifold.interpolate.smoothVertex(mctx, nSym, alignToCurvature)
%
%   Returns a struct with `vertexVectors` (V*3 world-space) and
%   `vertexFieldRaw` (V*2 raw nRoSy field).
    if nargin < 2, nSym = 2;             end
    if nargin < 3, alignToCurvature = false; end
    r = nxr_compute('geometry.smoothVertex', mctx.surface, ...
                    int32(nSym), logical(alignToCurvature));
end
