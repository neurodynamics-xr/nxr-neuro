function r = jwt(F, sfreq, graph_bank, time_bank, opts)
%NXR.COMPUTE.SIGNAL.JWT  The joint time-vertex wavelet transform (nxr::signal::jwt): complex [members x rows x samples] and the member index table.
%   r = nxr.compute.signal.jwt(modes, 256, struct(family="weyl", area=A, levels=4), struct(family="meyer"), eigenvalues=lambda)
%   r = nxr.compute.signal.jwt(F, 256, gbank, tbank, eigenvalues=lambda, eigenvectors=phi, mass=M, kernels={struct(type="speed", c=1, d=2)})
%
%   A member is the product of a graph-bank member, a time-bank member and (optionally) a joint kernel K(lambda, f); the members run over all combinations, the graph index fastest.
%   graph_bank and time_bank are the structs of nxr.compute.signal.graph_bank and nxr.compute.signal.time_bank (see wt for the time bank's fields). The time bank is designed on F's
%   own number of samples, at sfreq. Kernels: a cell array (or struct array) of structs, type "speed" (c, d: a ridge on omega = c sqrt(lambda), positive f; c in length/s, d the angular
%   width in rad/s) or "ridge" (v, width, sense +1 or -1: a ridge on f = sense v sqrt(lambda) / 2 pi, width in Hz; a two-sided time bank).
%
%   The domain is given by the transform arguments: with only eigenvalues, F is a [modes x samples] series of mode coefficients; with eigenvectors too (or a laplacian for the Chebyshev
%   series, separable banks only), F is a field [vertices x samples]. A real series needs an analytic time bank, a complex one a two-sided bank.
%   Name-value options: eigenvalues, eigenvectors, mass, laplacian (default []), order (40), lmax (0), kernels (default []).
%
%   Output r (struct): coefficients [members x rows x samples] complex; index [members x 3], the graph member g, the time member t and the kernel k of each (1-based; g varies fastest).
    arguments
        F
        sfreq (1,1) double
        graph_bank (1,1) struct
        time_bank (1,1) struct
        opts.eigenvalues = []
        opts.eigenvectors = []
        opts.mass = []
        opts.laplacian = []
        opts.order (1,1) double = 40
        opts.lmax (1,1) double = 0
        opts.kernels = []
    end
    [w, index] = nxr_compute('compute.signal.jwt', F, sfreq, nxr.impl.bank(graph_bank), nxr.impl.bank(time_bank), opts.eigenvalues, opts.eigenvectors, opts.mass, opts.laplacian, ...
                             opts.order, opts.lmax, nxr.impl.kernels(opts.kernels));
    r = struct('coefficients', w, 'index', index);
end
