function r = graphBank(family, opts)
%NXR.COMPUTE.SIGNAL.GRAPHBANK  A graph filterbank, for inspection (nxr::signal::graphfilterbank::make): its members, and its gains and frame bounds at the eigenvalues given.
%   r = nxr.compute.signal.graphBank("itersine", eigenvalues=lambda, lmax=max(lambda))
%   r = nxr.compute.signal.graphBank("weyl", area=4*pi, levels=4)
%
%   family: "mexhat", "heat", "itersine", "logitersine" (designed on a spectral range), "meyer" (on edges), "weyl" (on area and levels: one octave a level) or "rank" (on levels).
%   The range is lmin and lmax, or lmax (lmin = lmax / lp_factor), or the eigenvalues themselves.
%
%   Name-value options (the same names as in the Python wheel)
%     eigenvalues    the spectrum the gains and frame bounds are taken on (default []: the bank's own range).
%     lmax, lmin     the spectral range. Default [].
%     num_filters    wavelet count; 0 (default) derives it from voices.   voices   members per octave when derived (default 3).
%     lp_factor      lmin = lmax / lp_factor when lmin is unknown (default 20).
%     fine_limit     "basis" (default; the finest scale tiles the whole spectrum) or "usable" (every member responds fully).
%     scale_limits   [t_min t_max] overrides the policy above.      edges   meyer: ascending edges in lambda.
%     area, levels   weyl (area in the units of 1/lambda) and rank.  half_width   the transition half-width in octaves (default 1/3).
%
%   Output r (struct): gains [members x K] at eigenvalues ([] without them), roles (cell: "member", "scaling", "wavelet", "top"), scales (NaN where the family has none),
%   centre_wavenumbers (NaN for a bank with no range: meyer, weyl, rank), frame_bounds [A B] (of sum g^2 on the eigenvalues, else on the bank's own range; NaN for a bank with neither: meyer, weyl, rank).
    arguments
        family {mustBeTextScalar}
        opts.eigenvalues = []
        opts.lmax = []
        opts.lmin = []
        opts.num_filters (1,1) double = 0
        opts.voices (1,1) double = 3
        opts.lp_factor (1,1) double = 20
        opts.fine_limit {mustBeTextScalar} = "basis"
        opts.scale_limits = []
        opts.edges = []
        opts.area = []
        opts.levels = []
        opts.half_width (1,1) double = 1/3
    end
    bank = nxr.impl.given('family', family, 'lmax', opts.lmax, 'lmin', opts.lmin, 'num_filters', opts.num_filters, 'voices', opts.voices, 'lp_factor', opts.lp_factor, ...
                          'fine_limit', opts.fine_limit, 'scale_limits', opts.scale_limits, 'edges', opts.edges, 'area', opts.area, 'levels', opts.levels, 'half_width', opts.half_width);
    [g, roles, s, c, a, b] = nxr_compute('compute.signal.graphBank', bank, opts.eigenvalues);
    r = struct('gains', g, 'roles', {roles}, 'scales', s, 'centre_wavenumbers', c, 'frame_bounds', [a b]);
end
