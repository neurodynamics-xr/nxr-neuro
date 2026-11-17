function r = results(dataset, subject, opts)
%NXR.LIST.RESULTS  What a subject holds, by kind: a struct array of name, kind, table.
%   r = nxr.list.results(dataset, subject)                         % every complete field and statistic a computation stored
%   r = nxr.list.results(dataset, subject, kind="recordings")      % or "kernels"
    arguments
        dataset {mustBeTextScalar}
        subject {mustBeTextScalar}
        opts.kind {mustBeTextScalar} = "results"
    end
    r = jsondecode(nxr_compute('run.results', char(dataset), char(subject), char(opts.kind)));
end
