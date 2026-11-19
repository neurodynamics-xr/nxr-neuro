function s = bank(s)
%NXR.IMPL.BANK  A bank (or a kernel) as the struct the MEX takes: string scalars become character vectors; fields left [] are "not given".
%   Internal: converts only; the design fields and their names are the library's (checked by the library).
    if ~isstruct(s) || ~isscalar(s)
        error('nxr:invalidArgument', 'a bank is a scalar struct of the design fields (family, ...)');
    end
    f = fieldnames(s);
    for i = 1:numel(f)
        if isstring(s.(f{i})), s.(f{i}) = char(s.(f{i})); end
    end
end
