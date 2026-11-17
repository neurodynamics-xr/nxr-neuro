classdef Recording < handle
%NXR.IO.RECORDING  An open recording: the header now, the samples by block (nxr::formats::RecordingReader).
%   rec = nxr.io.open_edf("rest.edf");
%   rec.header.sampling_rate, rec.header.n_samples
%   x = rec.read(1, 10000);              % samples 1..10000: [channels x samples]
%
%   A recording is never loaded whole unless the caller says so: the span is always the caller's (first and last sample, or read_all).
%   Samples are 1-based and inclusive, as everywhere in MATLAB (the Python wheel's read(start, stop) is read(start + 1, stop) here).
%   The file closes when the object is deleted or close is called.

    properties (SetAccess = private)
        % The parsed header (struct): format, channels (struct array: name, kind, unit, file_unit, scale_to_si, gain, offset, sampling_rate, transducer,
        % prefilter, reference, location, orientation, coil_type, bad), sampling_rate (Hz), n_samples (per channel), precision ("single" or "double"),
        % start_time, annotations (struct array: onset and duration in seconds from the first sample, label, channels), bad_channels (cell array of
        % names), gaps ([first count] per row: first sample, 1-based, and count), position_unit, and info (every other field, by the file's own name,
        % as a valid MATLAB field name).
        header
    end
    properties (Access = private)
        handle
    end

    methods
        function obj = Recording(handle)
            % Internal: use nxr.io.open_edf, open_brainvision, open_eeglab or open_fif.
            obj.handle = handle;
            obj.header = nxr_compute('io.header', handle);
        end

        function x = read(obj, first, last, opts)
            %READ  Samples first..last of channels first_channel..last_channel as [channels x samples], in SI units.
            %   x = rec.read(first, last)
            %   x = rec.read(first, last, first_channel=2, last_channel=10)
            %   single for a file that holds integers or 32-bit floats, double for a 64-bit recording (precision follows the file). Samples in a gap
            %   (header.gaps) read as zeros. The result never depends on the block read. A span outside the recording raises nxr:invalidArgument.
            arguments
                obj
                first (1,1) double
                last (1,1) double
                opts.first_channel (1,1) double = 1
                opts.last_channel = []
            end
            if isempty(opts.last_channel), opts.last_channel = numel(obj.header.channels); end
            x = nxr_compute('io.read', obj.handle, first, last, opts.first_channel, opts.last_channel);
        end

        function x = read_all(obj, opts)
            %READ_ALL  The whole recording (the span stated as everything): memory is the recording's size.
            arguments
                obj
                opts.first_channel (1,1) double = 1
                opts.last_channel = []
            end
            if isempty(opts.last_channel), opts.last_channel = numel(obj.header.channels); end
            x = nxr_compute('io.read', obj.handle, 1, obj.header.n_samples, opts.first_channel, opts.last_channel);
        end

        function close(obj)
            %CLOSE  Release the file.
            if ~isempty(obj.handle)
                nxr_compute('io.close', obj.handle);
                obj.handle = [];
            end
        end

        function delete(obj)
            try
                obj.close();
            catch
            end
        end
    end
end
