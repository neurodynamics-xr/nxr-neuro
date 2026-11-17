function [x, header] = read_brainvision(path, opts)
%NXR.IO.READ_BRAINVISION  Read samples first..last (or all) of a BrainVision recording (.vhdr): [data, header].
%   [x, header] = nxr.io.read_brainvision(path, first=1, last=10000)
%   [x, header] = nxr.io.read_brainvision(path, all=true)
%
%   The span is the caller's: give first and last (samples, 1-based, inclusive), or all=true, not both and not neither. data is [channels x samples],
%   in SI units, in the file's precision. first_channel and last_channel (1-based, inclusive) pick the channels; default all. See nxr.io.open_brainvision and
%   nxr.io.Recording.read.
    arguments
        path {mustBeTextScalar}
        opts.first = []
        opts.last = []
        opts.all (1,1) logical = false
        opts.first_channel (1,1) double = 1
        opts.last_channel = []
    end
    rec = nxr.io.open_brainvision(path);
    [x, header] = nxr.impl.read_span(rec, opts);
end
