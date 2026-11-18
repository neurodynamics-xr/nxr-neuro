function z = band_analytic(x, sfreq, bands, opts)
%NXR.COMPUTE.SIGNAL.BAND_ANALYTIC  The analytic signal of bands (nxr::signal::bandAnalytic): complex [bands x channels x samples].
%   z = nxr.compute.signal.band_analytic(x, 256, [8 12; 12 30])
%   z = nxr.compute.signal.band_analytic(x, 256, [8 12], shape="meyer")
%
%   One member per band [lo, hi] Hz of shape (morlet: a Gaussian of full width at half maximum hi - lo; meyer: Meyer edges, half_width octaves; power_partition: squared gains adding
%   to 1), gain 2 (the one-sided analytic signal), DC and Nyquist dropped, designed on x's own length: x is the window the caller hands in (periodic on it; the caller's halo holds
%   the edge effect).
%
%   Inputs
%     x        [channels x samples], real.
%     sfreq    Hz.
%     bands    [bands x 2]: the low and high edge of each, Hz.
%   Name-value options
%     shape        "morlet" (default), "meyer" or "power_partition".
%     half_width   Meyer's transition half-width in octaves (default 1/3).
    arguments
        x {mustBeNumericOrLogical}
        sfreq (1,1) double
        bands (:,2) double
        opts.shape {mustBeTextScalar} = "morlet"
        opts.half_width (1,1) double = 1/3
    end
    z = nxr_compute('compute.signal.band_analytic', x, sfreq, bands, char(opts.shape), opts.half_width);
end
