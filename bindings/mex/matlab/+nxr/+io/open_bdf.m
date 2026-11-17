function rec = open_bdf(path, opts)
%NXR.IO.OPEN_BDF  Open a BDF or BDF+ file: the same reader as nxr.io.open_edf (the format is recognised from the header).
%   rec = nxr.io.open_bdf("rest.bdf", sampling_rate=0)
    arguments
        path {mustBeTextScalar}
        opts.sampling_rate (1,1) double = 0
    end
    rec = nxr.io.open_edf(path, sampling_rate=opts.sampling_rate);
end
