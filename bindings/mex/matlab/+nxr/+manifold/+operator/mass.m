function M = mass(mctx)
%MASS  V × V mass matrix M (mixed Voronoi default).
%   M = nxr.manifold.operator.mass(mctx)
%
%   Diagonal mixed Voronoi mass of the mollified input, recorded before the
%   flips of the intrinsic Delaunay mesh. Eagerly computed
%   by `nxr.manifold.context(V, F)` and held on the struct; returning
%   the field directly so this is O(1).
    M = mctx.M;
end
