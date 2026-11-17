function d = datasets(datastore)
%NXR.LIST.DATASETS  The datasets of a datastore.
%   d = nxr.list.datasets(datastore)   % a struct array
    arguments
        datastore {mustBeTextScalar}
    end
    r = jsondecode(nxr_compute('run.datasets', char(datastore)));
    d = r.datasets;
end
