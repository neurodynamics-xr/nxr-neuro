function rec = open_edf(path, opts)
%NXR.IO.OPEN_EDF  Open an EDF, EDF+, BDF or BDF+ file (continuous or discontinuous; the format is recognised from the header).
%   rec = nxr.io.open_edf("rest.edf")
%   rec = nxr.io.open_edf("mixed.edf", sampling_rate=100)
%
%   sampling_rate (Hz): the rate of the signals to read; 0 (default) = the file's only rate (a file with several rates is refused, the refusal lists
%   them). Annotation signals become header.annotations and are not channels; the discontinuous records' gaps are header.gaps.
%   Returns a nxr.io.Recording.
    arguments
        path {mustBeTextScalar}
        opts.sampling_rate (1,1) double = 0
    end
    rec = nxr.io.Recording(nxr_compute('io.open', 'edf', char(path), opts.sampling_rate));
end
