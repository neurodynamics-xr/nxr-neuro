function r = run(pipeline, dataset, subject, varargin)
%NXR.RUN  Run a pipeline on a subject of a dataset: the command line's `nxr-neuro run` (the same library calls, so the same cached result).
%   r = nxr.run('psd', dataset, 'sub-01', 'recording', 'rest', 'all', true)
%   r = nxr.run('global_bursts', dataset, 'sub-01', 'from', 0, 'to', 300, 'band', [15 29])
%
%   The time range is REQUIRED and the caller's: 'from' and 'to' (seconds on the recording's clock; both) or 'all', true (the whole recording, still read one chunk at a
%   time). A call without one, or with both, is refused (nxr:invalidArgument).
%
%   Inputs
%     pipeline   A built-in's name ('psd'), a shipped definition's name, or the path of a .yaml definition. nxr.list.pipelines() lists them with each parameter's meaning,
%                default and unit; nxr.list.help(name) prints that help.
%     dataset    The dataset folder (a catalogue of subject packages, or a one-database dataset).
%     subject    The subject's name.
%   Name-value arguments
%     'from', 'to'   The time range in seconds.
%     'all'          The whole recording (default false).
%     'replace'      Another job's outputs of the same names are moved aside (never deleted) and replaced; without it they are refused (default false).
%     'progress'     A function handle called with each progress line, a struct (the command line's JSON lines), e.g. @(s) disp(s.stage).
%     any other name is a parameter of the pipeline: psd's are recording, name, set, segment_seconds, segment_samples, segment_level, overlap, window, periodic,
%                detrend, scaling, exclude_bad_segments, through.
%   A rerun with the same pipeline, parameters, inputs and range is a cache hit: nothing is computed.
%
%   Output r (struct)
%     pipeline, subject, dataset
%     cache      'hit' when everything was found stored, else 'computed'.
%     stored     struct array: output, name, id, path of each stored result (nxr.read(dataset, subject, name) reads one back).
%     runs       The library's own report of each run (a cell array of structs).
%     seconds    The run's duration.
    if ~(ischar(pipeline) || isstring(pipeline)) || ~(ischar(dataset) || isstring(dataset)) || ~(ischar(subject) || isstring(subject))
        error('nxr:invalidArgument', 'nxr.run(pipeline, dataset, subject, ...): the three are text');
    end
    from = []; to = []; wholeRun = false; replace = false; progress = []; params = struct();
    if mod(numel(varargin), 2) ~= 0
        error('nxr:invalidArgument', 'nxr.run: the options come as name-value pairs');
    end
    for k = 1:2:numel(varargin)
        key = char(varargin{k});
        val = varargin{k+1};
        switch key
            case 'from', from = val;
            case 'to', to = val;
            case 'all', wholeRun = logical(val);
            case 'replace', replace = logical(val);
            case 'progress', progress = val;
            otherwise, params.(key) = val;
        end
    end
    if ~isempty(progress)
        fcn = progress;
        progress = @(text) fcn(jsondecode(text));
    end
    text = nxr_compute('run.run', char(dataset), char(subject), char(pipeline), jsonencode(params), from, to, wholeRun, replace, progress);
    r = jsondecode(text);
    r.dataset = char(dataset);
    if ~isfield(r, 'stored') || isempty(r.stored), r.stored = struct('output', {}, 'name', {}, 'id', {}, 'path', {}); end
    if isstruct(r.runs), r.runs = num2cell(r.runs); end
end
