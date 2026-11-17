function rec = open_eeglab(path)
%NXR.IO.OPEN_EEGLAB  Open an EEGLAB .set recording (with its .fdt, or the data embedded in the .set).
%   rec = nxr.io.open_eeglab("rest.set")
%
%   Events and channel locations are in the header; compressed variables and epoched datasets are refused. Returns a nxr.io.Recording.
    arguments
        path {mustBeTextScalar}
    end
    rec = nxr.io.Recording(nxr_compute('io.open', 'eeglab', char(path), 0));
end
