function y = filter_run(kind, x, sfreq, lower, upper, frequencies, opts)
%NXR.IMPL.FILTER_RUN  The one call behind the five filters (nxr.ops.filter.*): arguments in the MEX's order, nothing else.
    y = nxr_compute(['ops.filter.' kind], x, sfreq, lower, upper, frequencies, nxr.impl.text(opts.method), opts.ripple, opts.transition_bandwidth, ...
                    char(opts.edge), opts.transient_energy, opts.transient_samples, opts.remove_mean, opts.notch_bandwidth, opts.bandstop_order);
end
