function r = risefall(amplitude, sfreq, t0, t1, opts)
%NXR.COMPUTE.SELECT.RISEFALL  Where an amplitude episode switches on and off: the double-sigmoid fit of Rheome's +rheome/+detect/risefall.m (nxr::select::fit).
%   r = nxr.compute.select.risefall(amplitude, sfreq, t0, t1)
%   r = nxr.compute.select.risefall(amplitude, sfreq, t0, t1, model="quadrature", min_r2=0.8)
%
%     a(t) = b + h S((t - t_on)/tau_rise) S(-(t - t_off)/tau_fall)                    model "linear"       S(x) = 1/(1 + e^-x)
%     a(t) = sqrt(b^2 + (h S(..) S(..))^2)                                            model "quadrature"   an episode on a background adds in power
%   fitted by bounded nonlinear least squares to the series, smoothed by a symmetric moving mean and scaled to unit maximum, over [t0 - margin, t1 + margin].
%   t_on and t_off are the HALF-rise and HALF-fall times. What it cannot show: the fit is of the amplitude's shape, not of its sources.
%
%   Inputs
%     amplitude   vector: any amplitude series (a region's mean, the maximum over tiles).
%     sfreq       samples per second.
%     t0, t1      s: the candidate episode.
%   Name-value options
%     model       "linear" (default) or "quadrature". Rheome's "posterior" is not built and is refused.
%     margin      s either side of [t0, t1] included in the fit (default 1.0).
%     smooth      s: the symmetric moving mean applied first (default 0.1).
%     tau_max     s: the upper bound of the rise and fall times (default 1.0).
%     min_r2      when given, r.kept is the fit gate (ok, r2 >= min_r2, not at a bound); default [] (r.kept is []).
%
%   Output r (struct): t_on, t_off, tau_rise, tau_fall (s, on the series' clock: sample i, 1-based, is at (i-1)/sfreq), baseline, height, r2 (on the unit scale), ok (false: a
%   window under 11 samples, no amplitude, or t_off <= t_on; the numbers are then NaN), at_bound (a time within 1% of tau_max), buildup, plateau, decay ([start end] s; plateau NaN
%   when they cross), has_plateau, kept.
    arguments
        amplitude {mustBeNumeric}
        sfreq (1,1) double
        t0 (1,1) double
        t1 (1,1) double
        opts.model {mustBeTextScalar} = "linear"
        opts.margin (1,1) double = 1.0
        opts.smooth (1,1) double = 0.1
        opts.tau_max (1,1) double = 1.0
        opts.min_r2 = []
    end
    r = nxr_compute('compute.select.risefall', amplitude, sfreq, t0, t1, char(opts.model), opts.margin, opts.smooth, opts.tau_max, opts.min_r2);
end
