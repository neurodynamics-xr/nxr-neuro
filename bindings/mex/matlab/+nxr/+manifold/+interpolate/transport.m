function out = transport(mctx, sourceVerts, sourceVectors)
%TRANSPORT  Parallel-transport tangent vectors via the vector heat method.
%   out = nxr.manifold.interpolate.transport(mctx, sourceVerts, sourceVectors)
%
%   sourceVectors is N*3 world-space (one row per source vertex).
%   Returns V*3 row-major tangent vectors.
    out = nxr_compute('geometry.parallel', mctx.surface, ...
                      int32(sourceVerts), sourceVectors);
end
