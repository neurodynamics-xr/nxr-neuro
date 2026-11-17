function [t, names] = read_tsv(path)
%NXR.IO.READ_TSV  Read a BIDS-style TSV table: a struct from column name to the cell array of its cells, as text.
%   [t, names] = nxr.io.read_tsv("participants.tsv")
%   t.age{1}
%
%   `n/a` is kept as written; no cell is empty. A row whose cell count differs from the header's is refused with its line number. The fields are the
%   column names as valid MATLAB field names (characters other than letters, digits and _ become _); names is the cell array of the names as the
%   file has them, in order.
    arguments
        path {mustBeTextScalar}
    end
    [names, columns] = nxr_compute('io.read_tsv', char(path));
    t = struct();
    for i = 1:numel(names)
        t.(matlab.lang.makeValidName(names{i})) = columns{i};
    end
end
