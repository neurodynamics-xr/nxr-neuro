function r = psd(x, sfreq, opts)
%NXR.OPS.WELCH.PSD  Welch's power spectral density of every channel (nxr::welch::Welch).
%   r = nxr.ops.welch.psd(x, sfreq)
%   r = nxr.ops.welch.psd(x, sfreq, segment_samples=512, window="hamming", ...)
%
%   A record x(0..T) of C channels is cut into segments of segment_samples (n) samples, a new segment every hop = n - floor(n*overlap) samples, from
%   the first sample, as many as fit whole (a partial last segment is dropped, as scipy.signal.welch and Brainstorm do). Each segment is detrended,
%   multiplied by a window w and transformed (rfft), and |X_k|^2 is averaged over the segments. The one-sided spectrum doubles every bin but DC (and,
%   for an even n, Nyquist). Frequencies are k*sfreq/n, k = 0 .. n/2.
%
%   Inputs
%     x                [channels x samples]  The record, single or double. The result comes back in the same precision.
%     sfreq            Sampling frequency, Hz. It sets the frequencies and the density.
%   Name-value options (the library's WelchOptions; the same names as in the Python wheel)
%     segment_samples  n, samples per segment (default 256).
%     overlap          Fraction of a segment shared with the next, in [0, 1): hop = n - floor(n*overlap) (default 0.5).
%     window           "hann" (default), "hamming", "blackman" or "rectangular".
%     periodic         true (default): the DFT-even form of the window (w_j = f(2 pi j/n), scipy). false: the symmetric form (w_j = f(2 pi j/(n-1)),
%                      MATLAB's hamming(n)).
%     detrend          Removed from each segment before the window: "constant" (default, scipy's), "linear" or "none".
%     scaling          "density" (default): |X_k|^2/(sfreq*sum w^2), power per Hz (scipy "density"; Brainstorm's physical units).
%                      "spectrum": |X_k|^2/(sum w)^2, power (scipy "spectrum").
%                      "chain": |X_k|^2/(n*sum w^2), power per sample, per bin (the Studio's and nxr's chain Welch).
%     good             logical [1 x samples]: a segment that touches a sample that is not good is left out (Brainstorm's bad segments). The segment grid
%                      does not move. Default [] (all good).
%
%   Output r (struct)
%     frequencies      [1 x (segment_samples/2 + 1)]  Hz.
%     psd              [channels x frequencies], in the input's precision.
%     segments         The segments averaged (0: the record held none; psd is then zero).
%     segments_skipped Whole segments left out for touching a bad sample.
%   The values are those of scipy.signal.welch for the same segmentation, window, detrend and scaling.
    arguments
        x {mustBeNumericOrLogical}
        sfreq (1,1) double
        opts.segment_samples (1,1) double = 256
        opts.overlap (1,1) double = 0.5
        opts.window {mustBeTextScalar} = "hann"
        opts.periodic (1,1) logical = true
        opts.detrend {mustBeTextScalar} = "constant"
        opts.scaling {mustBeTextScalar} = "density"
        opts.good = []
    end
    [f, p, n, skipped] = nxr_compute('ops.welch.psd', x, sfreq, opts.segment_samples, opts.overlap, char(opts.window), opts.periodic, ...
                                     char(opts.detrend), char(opts.scaling), opts.good);
    r = struct('frequencies', f, 'psd', p, 'segments', n, 'segments_skipped', skipped);
end
