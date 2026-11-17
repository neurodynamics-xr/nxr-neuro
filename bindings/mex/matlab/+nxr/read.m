function s = read(dataset, subject, name, opts)
%NXR.READ  A stored field, statistic, operator or selection of a subject, read whole: its arrays and its metadata.
%   s = nxr.read(dataset, subject, 'rest_psd_welch')      % s.values [channels x frequencies], s.meta.measure, s.meta.axes ...
%   t = nxr.read(dataset, subject, 'segments', measure='power_sum')   % a selection's measurement table: t.code, t.value, t.weight, t.members
%
%   Inputs
%     dataset, subject   As in nxr.run.
%     name               The stored thing's name (or id): a statistic, field, operator or selection.
%   Name-value options
%     measure   With it, name is a selection and the answer is its measurement table (what the measure carries on each member).
%     of        The field the measurement is of, when several are measured.
%
%   Output s (struct): name, meta (what the node records about itself: class, id, kind, unit; a statistic's measure, method, axes, options and provenance) and its arrays by
%   name, as MATLAB arrays of the stored type (values for a field, statistic or operator; a sparse operator: data, indices, indptr). Nothing is written. A recording is read
%   whole by this call: for a long one use nxr.io or nxr.run.
    arguments
        dataset {mustBeTextScalar}
        subject {mustBeTextScalar}
        name {mustBeTextScalar}
        opts.measure = []
        opts.of = []
    end
    if ~isempty(opts.measure), opts.measure = char(opts.measure); end
    if ~isempty(opts.of), opts.of = char(opts.of); end
    [metaText, arrays] = nxr_compute('run.read', char(dataset), char(subject), char(name), opts.measure, opts.of);
    meta = jsondecode(metaText);
    s = struct('name', meta.name, 'meta', meta);
    keys = fieldnames(arrays);
    for k = 1:numel(keys)
        s.(keys{k}) = arrays.(keys{k});
    end
end
