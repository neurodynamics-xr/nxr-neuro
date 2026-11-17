function [x, header] = read_fif(path, opts)
%NXR.IO.READ_FIF  Read samples first..last (or all) of a FIF raw recording: [data, header].
%   [x, header] = nxr.io.read_fif(path, first=1, last=10000)
%   [x, header] = nxr.io.read_fif(path, all=true)
%
%   The span is the caller's: give first and last (samples, 1-based, inclusive), or all=true, not both and not neither. data is [channels x samples],
%   in SI units, in the file's precision. first_channel and last_channel (1-based, inclusive) pick the channels; default all. See nxr.io.open_fif and
%   nxr.io.Recording.read.
%   allow_maxshield   As in nxr.io.open_fif (default false).
    arguments
        path {mustBeTextScalar}
        opts.first = []
        opts.last = []
        opts.all (1,1) logical = false
        opts.first_channel (1,1) double = 1
        opts.last_channel = []
        opts.allow_maxshield (1,1) logical = false
    end
    rec = nxr.io.open_fif(path, allow_maxshield=opts.allow_maxshield);
    [x, header] = nxr.impl.read_span(rec, opts);
end
