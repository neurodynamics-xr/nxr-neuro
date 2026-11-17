function r = cross_spectrum(x, sfreq, bins, opts)
%NXR.OPS.WELCH.CROSS_SPECTRUM  Welch's cross-spectral density between every pair of channels, per group of bins (nxr::welch::CrossWelch).
%   r = nxr.ops.welch.cross_spectrum(x, sfreq, bins)
%   r = nxr.ops.welch.cross_spectrum(x, sfreq, [1 1; 13 13; 31 41], quadrature=true)
%
%   S_g = scale/segments x (mean or sum over the group's bins of w_k Re(X_k X_k^H)), w_k the one-sided doubling. The same segments, detrending, window,
%   scaling and bad-segment rule as nxr.ops.welch.psd, which is its diagonal. Memory: channels^2 per group, never a spectrum per pair.
%
%   Inputs
%     x        [channels x samples]  The record, single or double.
%     sfreq    Sampling frequency, Hz.
%     bins     [groups x 2]  The groups: each a run of bins first..last (inclusive, 1-based) of the one-sided spectrum. Bin k is at (k-1)*sfreq/segment_samples
%              Hz (bin 1 is DC; the Python wheel counts from 0).
%   Name-value options
%     segment_samples, overlap, window, periodic, detrend, scaling, good   As in nxr.ops.welch.psd.
%     mean         true (default): a group's value is the mean of its bins (a band). false: their sum (a total).
%     quadrature   Also accumulate the imaginary part (the quadrature spectrum), Im(X_k X_k^H) = B A' - A B' for X = A + iB. Default false. The complex
%                  density of a group is real + 1i*imag.
%
%   Output r (struct)
%     frequencies      [1 x bins]  Hz, of all bins.
%     real             [groups x channels x channels]: the real part of the cross-spectral density of each group of bins.
%     imag             [groups x channels x channels], antisymmetric: its imaginary part; [] unless quadrature is true.
%     segments, segments_skipped
    arguments
        x {mustBeNumericOrLogical}
        sfreq (1,1) double
        bins {mustBeNumeric}
        opts.segment_samples (1,1) double = 256
        opts.overlap (1,1) double = 0.5
        opts.window {mustBeTextScalar} = "hann"
        opts.periodic (1,1) logical = true
        opts.detrend {mustBeTextScalar} = "constant"
        opts.scaling {mustBeTextScalar} = "density"
        opts.mean (1,1) logical = true
        opts.quadrature (1,1) logical = false
        opts.good = []
    end
    [f, re, im, n, skipped] = nxr_compute('ops.welch.cross_spectrum', x, sfreq, opts.segment_samples, opts.overlap, char(opts.window), opts.periodic, ...
                                          char(opts.detrend), char(opts.scaling), bins, opts.mean, opts.quadrature, opts.good);
    r = struct('frequencies', f, 'real', re, 'imag', im, 'segments', n, 'segments_skipped', skipped);
end
