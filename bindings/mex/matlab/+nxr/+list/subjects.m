function s = subjects(dataset, opts)
%NXR.LIST.SUBJECTS  The subjects of a dataset.
%   s = nxr.list.subjects(dataset)
%   s = nxr.list.subjects(dataset, where="age > 60")     % a catalogue dataset: a filter over the participant variables (version 1 of the filter language)
%   s = nxr.list.subjects(dataset, names=["sub-01" "sub-02"])
%   'where' and 'names' (by name or id) need a catalogue dataset and are alternatives. A struct array.
    arguments
        dataset {mustBeTextScalar}
        opts.where = []
        opts.names = string.empty
    end
    if ~isempty(opts.where), opts.where = char(opts.where); end
    names = cellstr(string(opts.names));
    namesJson = jsonencode(names(:)');
    if isempty(names), namesJson = '[]'; end
    r = jsondecode(nxr_compute('run.subjects', char(dataset), namesJson, opts.where));
    s = r.subjects;
end
