function w = wt(x, sfreq, bank)
%NXR.COMPUTE.SIGNAL.WT  The wavelet transform of x by a time bank (nxr::signal::wt): complex [members x channels x samples], at the full rate.
%   w = nxr.compute.signal.wt(x, 256, struct(family="meyer"))
%   w = nxr.compute.signal.wt(z, 256, struct(family="morlet", sides="two_sided", centres=[6 10 16], cycles=6, gain=2))
%
%   A time bank is a struct of the library's design fields: family ("itersine", "meyer", "morlet", "morse", "amor", "bump", "power_partition"), sides ("analytic" for a real
%   signal, "two_sided" for a complex one; default "analytic"), gain (the peak in-band response, default 1) and the family's fields: centres, cycles, bands ([nb x 2]), edges,
%   half_width, voices, anchor, limits, min_bins, gamma, time_bandwidth, edge_bins ("single" or "drop"). A field not given keeps the library's default. The bank is DESIGNED on the
%   number of samples of the signal given: that window is the caller's, and nothing is cut or padded beyond what the design says.
%
%   Inputs
%     x       [channels x samples]: real for an analytic bank, complex (or real) for a two-sided bank.
%     sfreq   Hz.
%     bank    the time bank struct above.
    arguments
        x
        sfreq (1,1) double
        bank (1,1) struct
    end
    w = nxr_compute('compute.signal.wt', x, sfreq, nxr.impl.bank(bank));
end
