function r = eigen(vertices, faces, k, opts)
%NXR.COMPUTE.OPERATORS.EIGEN  The Laplace-Beltrami eigenmodes of a triangle mesh (nxr::geometry).
%   r = nxr.compute.operators.eigen(vertices, faces, k)
%   r = nxr.compute.operators.eigen(vertices, faces, k, normalize=true, remove_dc=true)
%
%   The k smallest eigenpairs of K phi = lambda M phi on a mesh (the cotangent Laplacian K and the mixed-Voronoi mass M of the operator mesh).
%   (The pre-existing nxr_compute commands 'create' and 'solve' give the same modes on a kept mesh handle.)
%
%   Inputs
%     vertices   [n_vertices x 3]  Positions (double; other types are converted).
%     faces      [n_faces x 3]     Vertex indices, 1-based (the Python wheel's are 0-based).
%     k          The number of eigenpairs (1 <= k <= n_vertices - 1).
%   Name-value options
%     sigma      Unset (default []): the scale-aware shift ladder (sigma = -1e-7 lambda_max, then -1e-4 lambda_max, then a Cholesky-mode last resort),
%                never a fixed shift on the null space. A finite value is a FIXED shift-invert centre: the k eigenvalues nearest sigma are returned.
%     normalize  true: M-orthonormalize the eigenvectors so that U' M U = I. Default false: as the solver produces them.
%     remove_dc  true: strip the null space (every mode with |lambda| within a relative 1e-8 of 0). Default false.
%     intrinsic_delaunay  true (default): the operators are built on the intrinsic mollified Delaunay mesh. false: on the input triangulation as it is
%                ("embedded").
%
%   Output r (struct)
%     eigenvalues   [k x 1], ascending
%     eigenvectors  [n_vertices x k], the columns
%     n_converged
    arguments
        vertices {mustBeNumeric}
        faces {mustBeNumeric}
        k (1,1) double
        opts.sigma = []
        opts.normalize (1,1) logical = false
        opts.remove_dc (1,1) logical = false
        opts.intrinsic_delaunay (1,1) logical = true
    end
    [ev, evec, n] = nxr_compute('compute.operators.eigen', vertices, faces, k, opts.sigma, opts.normalize, opts.remove_dc, opts.intrinsic_delaunay);
    r = struct('eigenvalues', ev, 'eigenvectors', evec, 'n_converged', n);
end
