function r = detectBursts(amplitude, step, opts)
%NXR.COMPUTE.SELECT.DETECTBURSTS  The bursts of an amplitude series (nxr::select::detectBursts; Rheome's labelalpha in envelope mode, nxr's burst_stages / spindle_envelope).
%   r = nxr.compute.select.detectBursts(amplitude, step)
%   r = nxr.compute.select.detectBursts(amplitude, step, enter_sd=2, min_s=0.2, baseline=[m s])
%
%   The robust baseline (the median and 1.4826 * MAD over the finite values, or the caller's), enter at median + enter_sd * SD, extend while >= median + exit_sd * SD,
%   gaps joined, spans kept by length. The series is ANY amplitude (a sensor's envelope per sample, a tile measurement per tile, a region's mean). A non-finite value
%   (a bad sample or tile) ends a span and is left out of the baseline.
%
%   Inputs
%     amplitude   vector: the series.
%     step        s between its samples (1/sfreq per sample, a tile's width per tile).
%   Name-value options (the library's BurstOptions; the same names as in the Python wheel)
%     enter_sd    robust SDs above the median to enter (default 1.5).
%     exit_sd     robust SDs above the median to extend (default 1.0).
%     merge_s     s: gaps lasting up to this are joined (default 0.1; compared in whole samples).
%     min_s       s: the bursts kept last at least this (default 0.4).
%     max_s       s: and at most this (default Inf: no bound).
%     baseline    [median sd] in the series' units, the caller's (a span-wide measurement). Default [] (computed over the finite values).
%
%   Output r (struct)
%     spans       the bursts, in samples of the series (start, stop, peak, value; 1-based, stop the last sample, inclusive).
%   Spans are a struct of column vectors: start (1-based first sample), stop (the LAST sample, inclusive), peak (1-based sample of the extreme value inside) and value.
%   The library holds them 0-based with stop exclusive (the Python face's Spans); the conversion is made at the boundary: start + 1, stop unchanged, peak + 1.
%     median, sd  the baseline used.        enter, exit   the two levels, in the series' units.
%     good        finite values in the series.
%     gap, min_length, max_length   the durations as compared, in samples (max_length is the largest int64 when unbounded).
%     skipped     empty, or why nothing was detected (fewer than 2 good samples, no spread).
    arguments
        amplitude {mustBeNumeric}
        step (1,1) double
        opts.enter_sd (1,1) double = 1.5
        opts.exit_sd (1,1) double = 1.0
        opts.merge_s (1,1) double = 0.1
        opts.min_s (1,1) double = 0.4
        opts.max_s (1,1) double = Inf
        opts.baseline = []
    end
    r = nxr_compute('compute.select.detectBursts', amplitude, step, opts.enter_sd, opts.exit_sd, opts.merge_s, opts.min_s, opts.max_s, opts.baseline);
end
