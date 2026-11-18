function mctx = context(V, F)
%CONTEXT  Build the surface data for the nxr.manifold.* functions.
%
%   mctx = nxr.manifold.context(V, F)
%
%   V    nV-by-3 double, vertex positions
%   F    nF-by-3 integer (one-based), triangle indices
%
%   Returns a struct holding the mesh, the SURFACE (`surface`: the data of
%   nxr_compute('geometry.surfaceArrays', V, F), the intrinsic Delaunay
%   triangulation made once, which every surface command takes as its first
%   argument), the assembled cotangent stiffness
%   `K`, the mixed Voronoi mass `M` (diagonal; Meyer et al. 2003, of the
%   mollified input, recorded before the flips), and the per-axis sizes
%   `nV / nE / nF`.
%   Each nxr.manifold.* leaf takes mctx as its first argument, mirroring
%   the JS / WASM / N-API binding's six-group surface:
%
%       mctx = nxr.manifold.context(V, F);
%       eig  = nxr.manifold.solve.eigen(mctx, 6);
%       d    = nxr.manifold.measure.signedDistance(mctx, [0,1,2], false);
%
%   Setup:
%     addpath('bindings/mex/matlab')                   % this package tree
%     addpath('build/bindings/mex/Release')            % the MEX binary
%
%   Build the MEX with `bash scripts/build.sh Release`.
%
%   Lifetime: MATLAB is value-typed, so the struct is copied at every
%   call boundary. The MEX dispatcher is stateless: it keeps nothing between
%   calls, and rebuilds the library's surface from `mctx.surface` inside
%   each command. The intrinsic triangulation (the expensive step) and the
%   cotangent / mass assembly are paid once per `nxr.manifold.context` call;
%   downstream leaves operate on the resulting surface, K and M.
%
%   See also: nxr.manifold.solve.eigen, nxr.manifold.operator.stiffness.

    validateattributes(V, {'double'}, {'2d', 'ncols', 3, 'finite'}, 'context', 'V');
    if ~isa(F, 'int32')
        F = int32(F);
    end
    validateattributes(F, {'int32'}, {'2d', 'ncols', 3}, 'context', 'F');

    surface = nxr_compute('geometry.surfaceArrays', V, F);
    ops = nxr_compute('operators.assembleManifoldOperators', surface);

    mctx        = struct();
    mctx.V      = V;
    mctx.F      = F;
    mctx.surface = surface;
    mctx.K      = ops.cotanLaplacian;
    mctx.M      = ops.mass;
    mctx.nV     = ops.nV;
    mctx.nE     = ops.nE;
    mctx.nF     = ops.nF;
    mctx.totalArea       = ops.totalArea;
    mctx.vertexDualAreas = ops.vertexDualAreas;
    mctx.vertexNormals   = ops.vertexNormals;
end
