function x = ijwt(w, sfreq, n, graph_bank, time_bank, opts)
%NXR.COMPUTE.SIGNAL.IJWT  The series back from joint coefficients (nxr::signal::ijwt, ijwtComplex): [rows x n], real for an analytic time bank, complex for a two-sided one.
%   x = nxr.compute.signal.ijwt(w, 256, n, gbank, tbank, eigenvalues=lambda)
%   w is [members x rows x n] as nxr.compute.signal.jwt gives it (its coefficients field); n the number of samples the time bank was designed on; the other arguments as jwt, and
%     synthesis   "dual" (default; exact wherever the bank covers) or "tight".
    arguments
        w
        sfreq (1,1) double
        n (1,1) double
        graph_bank (1,1) struct
        time_bank (1,1) struct
        opts.eigenvalues = []
        opts.eigenvectors = []
        opts.mass = []
        opts.laplacian = []
        opts.order (1,1) double = 40
        opts.lmax (1,1) double = 0
        opts.kernels = []
        opts.synthesis {mustBeTextScalar} = "dual"
    end
    x = nxr_compute('compute.signal.ijwt', w, sfreq, n, nxr.impl.bank(graph_bank), nxr.impl.bank(time_bank), opts.eigenvalues, opts.eigenvectors, opts.mass, opts.laplacian, ...
                    opts.order, opts.lmax, nxr.impl.kernels(opts.kernels), char(opts.synthesis));
end
