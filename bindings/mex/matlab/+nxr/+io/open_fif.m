function rec = open_fif(path, opts)
%NXR.IO.OPEN_FIF  Open a Neuromag / MNE raw .fif recording (MEG and EEG): the header now, the samples by block.
%   rec = nxr.io.open_fif("sub-01_raw.fif")
%   rec = nxr.io.open_fif("shielded_raw.fif", allow_maxshield=true)
%
%   A split file's later parts (named by the file) are found next to it and read as one sample axis. Samples are in SI units (T, T/m, V, ... per
%   channel; header.channels(i).unit), as MNE reads them: stored value x cal x range. int16, int32 and float32 buffers read as single, float64 as
%   double; complex buffers, epochs, evoked and processed-data files are refused with the reason. The first sample of the file is not added to the
%   sample axis (it starts at 1; header.info.first_samp). Active projectors are listed, not applied.
%   allow_maxshield: read internal-active-shielding raw data, which MNE refuses by default (default false). Skipped stretches are header.gaps, read
%   as zeros; annotations carry their onset in seconds from the first sample; bad channels are header.bad_channels. Returns a nxr.io.Recording.
    arguments
        path {mustBeTextScalar}
        opts.allow_maxshield (1,1) logical = false
    end
    rec = nxr.io.Recording(nxr_compute('io.open', 'fif', char(path), double(opts.allow_maxshield)));
end
