function e = bandEnvelope(x, sfreq, bands, opts)
%NXR.COMPUTE.SIGNAL.BANDENVELOPE  The envelopes of bands, |bandAnalytic| (nxr::signal::bandEnvelope): real [bands x channels x samples], in x's precision.
%   e = nxr.compute.signal.bandEnvelope(x, 256, [8 12; 12 30])
%   Arguments and options as nxr.compute.signal.bandAnalytic.
    arguments
        x {mustBeNumericOrLogical}
        sfreq (1,1) double
        bands (:,2) double
        opts.shape {mustBeTextScalar} = "morlet"
        opts.half_width (1,1) double = 1/3
    end
    e = nxr_compute('compute.signal.bandEnvelope', x, sfreq, bands, char(opts.shape), opts.half_width);
end
