function out = frame(mctx)
%FRAME  Per-face orthonormal tangent frame.
%   out = nxr.manifold.measure.frame(mctx)
%
%   Returns a struct {e1, e2, normals}, each [nF x 3], with e1 ⟂ e2 and
%   both unit-length (e2 = n × e1).
    out = nxr_compute('geometry.frames', mctx.surface);
end
