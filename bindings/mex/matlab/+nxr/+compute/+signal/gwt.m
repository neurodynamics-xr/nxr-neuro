function w = gwt(F, bank, opts)
%NXR.COMPUTE.SIGNAL.GWT  The wavelet transform of a field on a graph (nxr::signal::gwt): [members x vertices x samples], real.
%   w = nxr.compute.signal.gwt(F, struct(family="weyl", area=A, levels=4), eigenvalues=lambda, eigenvectors=phi, mass=M)
%   w = nxr.compute.signal.gwt(F, struct(family="itersine", lmax=lmax), laplacian=L, mass=M, order=40)
%
%   Inputs
%     F      [vertices x samples], real.
%     bank   a graph bank struct: family and the settings of nxr.compute.signal.graphBank (lmax, lmin, num_filters, voices, lp_factor, fine_limit, scale_limits, edges, area,
%            levels, half_width).
%   The field is transformed through the operator's modes (eigenvalues and eigenvectors [vertices x modes], mass-orthonormal; mass a sparse matrix or its diagonal) or by a Chebyshev
%   series in the pencil (laplacian and mass sparse; order; lmax, the series' bound, 0 to estimate it). A bank given no lmax is designed on the transform's own spectrum.
%   Name-value options: eigenvalues, eigenvectors, mass, laplacian (default []), order (default 40), lmax (default 0).
    arguments
        F {mustBeNumeric}
        bank (1,1) struct
        opts.eigenvalues = []
        opts.eigenvectors = []
        opts.mass = []
        opts.laplacian = []
        opts.order (1,1) double = 40
        opts.lmax (1,1) double = 0
    end
    w = nxr_compute('compute.signal.gwt', F, nxr.impl.bank(bank), opts.eigenvalues, opts.eigenvectors, opts.mass, opts.laplacian, opts.order, opts.lmax);
end
