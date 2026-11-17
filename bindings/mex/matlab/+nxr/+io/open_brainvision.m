function rec = open_brainvision(path)
%NXR.IO.OPEN_BRAINVISION  Open a BrainVision recording by its .vhdr (with the .vmrk markers and the .eeg data).
%   rec = nxr.io.open_brainvision("rest.vhdr")
%
%   INT_16, INT_32 or IEEE_FLOAT_32, multiplexed or vectorized; an ASCII file is refused. Markers become header.annotations.
%   Returns a nxr.io.Recording.
    arguments
        path {mustBeTextScalar}
    end
    rec = nxr.io.Recording(nxr_compute('io.open', 'brainvision', char(path), 0));
end
