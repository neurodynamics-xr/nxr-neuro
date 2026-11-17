function [x, header] = read_edf(path, opts)
%NXR.IO.READ_EDF  Read samples first..last (or all) of a EDF, EDF+, BDF or BDF+ file: [data, header].
%   [x, header] = nxr.io.read_edf(path, first=1, last=10000)
%   [x, header] = nxr.io.read_edf(path, all=true)
%
%   The span is the caller's: give first and last (samples, 1-based, inclusive), or all=true, not both and not neither. data is [channels x samples],
%   in SI units, in the file's precision. first_channel and last_channel (1-based, inclusive) pick the channels; default all. See nxr.io.open_edf and
%   nxr.io.Recording.read.
%   sampling_rate   As in nxr.io.open_edf (default 0: the file's only rate).
    arguments
        path {mustBeTextScalar}
        opts.first = []
        opts.last = []
        opts.all (1,1) logical = false
        opts.first_channel (1,1) double = 1
        opts.last_channel = []
        opts.sampling_rate (1,1) double = 0
    end
    rec = nxr.io.open_edf(path, sampling_rate=opts.sampling_rate);
    [x, header] = nxr.impl.read_span(rec, opts);
end
