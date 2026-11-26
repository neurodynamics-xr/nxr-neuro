function F = igwt(w, bank, opts)
%NXR.COMPUTE.SIGNAL.IGWT  The field back from coefficients (nxr::signal::igwt): [vertices x samples].
%   F = nxr.compute.signal.igwt(w, bank, eigenvalues=lambda, eigenvectors=phi, mass=M)
%   w is [members x vertices x samples] as nxr.compute.signal.gwt gives it; arguments and options as gwt, and
%     synthesis   "dual" (default; exact wherever the bank covers) or "tight".
%   The field is transformed through the operator's modes (eigenvalues and eigenvectors [vertices x modes], mass-orthonormal; mass a sparse matrix or its diagonal) or by a Chebyshev
%   series in the pencil (laplacian and mass sparse; order; lmax, the series' bound, 0 to estimate it). A bank given no lmax is designed on the transform's own spectrum.
    arguments
        w
        bank (1,1) struct
        opts.eigenvalues = []
        opts.eigenvectors = []
        opts.mass = []
        opts.laplacian = []
        opts.order (1,1) double = 40
        opts.lmax (1,1) double = 0
        opts.synthesis {mustBeTextScalar} = "dual"
    end
    F = nxr_compute('compute.signal.igwt', w, nxr.impl.bank(bank), opts.eigenvalues, opts.eigenvectors, opts.mass, opts.laplacian, opts.order, opts.lmax, char(opts.synthesis));
end
