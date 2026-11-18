function x = iwt(w, sfreq, n, bank, opts)
%NXR.COMPUTE.SIGNAL.IWT  The record back from coefficients (nxr::signal::iwt, iwtComplex): [channels x n], real for an analytic bank, complex for a two-sided one.
%   x = nxr.compute.signal.iwt(w, 256, n, bank)
%   x = nxr.compute.signal.iwt(w, 256, n, bank, synthesis="tight")
%
%   w is [members x channels x n] as nxr.compute.signal.wt gives it; bank and sfreq are those of wt, and n the number of samples the bank was designed on.
%   Name-value options
%     synthesis   "dual" (default; g/S, exact wherever the bank covers) or "tight" (g, exact when the bank is tight).
    arguments
        w
        sfreq (1,1) double
        n (1,1) double
        bank (1,1) struct
        opts.synthesis {mustBeTextScalar} = "dual"
    end
    x = nxr_compute('compute.signal.iwt', w, sfreq, n, nxr.impl.bank(bank), char(opts.synthesis));
end
