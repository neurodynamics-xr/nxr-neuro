function r = time_bank(sfreq, n, family, opts)
%NXR.COMPUTE.SIGNAL.TIME_BANK  A time filterbank designed on n samples at sfreq, for inspection (nxr::signal::timefilterbank::design): the grid, the gains and the frame bounds.
%   r = nxr.compute.signal.time_bank(256, 1024, "meyer")
%   r = nxr.compute.signal.time_bank(256, 1024, "morlet", centres=[6 10 16], cycles=6, sides="two_sided", gain=2)
%
%   family has no default, and neither has the gain's meaning: the PEAK in-band response of each member is gain (default 1). A field left [] keeps the library's default.
%
%   Name-value options (the library's Design; the same names as in the Python wheel)
%     sides        "analytic" (default; a real signal, bins f >= 0) or "two_sided" (a complex signal, signed f).
%     gain         the peak in-band response (default 1).
%     centres      Hz: the member centres (morlet, morse, amor, bump).        cycles    morlet: width fc / cycles.
%     bands        [nb x 2] Hz (meyer, morlet or power_partition as bands).   edges     meyer: ascending edges, Hz.
%     half_width   meyer: transition half-width in octaves.                   voices, anchor, limits, min_bins   itersine.
%     gamma, time_bandwidth   morse.                                          edge_bins   "single" (default) or "drop": what DC and Nyquist get.
%
%   Output r (struct)
%     frequencies         [1 x bins] Hz.            gains              [members x bins], the DC / Nyquist rule applied.
%     centre_frequencies  [1 x members] Hz.         frame_bounds       [A B]: the minimum and maximum of the frame operator sum g^2 on the interior bins.
    arguments
        sfreq (1,1) double
        n (1,1) double
        family {mustBeTextScalar}
        opts.sides {mustBeTextScalar} = "analytic"
        opts.gain (1,1) double = 1
        opts.centres = []
        opts.cycles = []
        opts.bands = []
        opts.edges = []
        opts.half_width = []
        opts.voices = []
        opts.anchor = []
        opts.limits = []
        opts.min_bins = []
        opts.gamma = []
        opts.time_bandwidth = []
        opts.edge_bins = []
    end
    bank = nxr.impl.given('family', family, 'sides', opts.sides, 'gain', opts.gain, 'centres', opts.centres, 'cycles', opts.cycles, 'bands', opts.bands, 'edges', opts.edges, ...
                          'half_width', opts.half_width, 'voices', opts.voices, 'anchor', opts.anchor, 'limits', opts.limits, 'min_bins', opts.min_bins, 'gamma', opts.gamma, ...
                          'time_bandwidth', opts.time_bandwidth, 'edge_bins', opts.edge_bins);
    [f, g, c, a, b] = nxr_compute('compute.signal.time_bank', sfreq, n, bank);
    r = struct('frequencies', f, 'gains', g, 'centre_frequencies', c, 'frame_bounds', [a b]);
end
