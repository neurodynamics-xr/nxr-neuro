function c = kernels(k)
%NXR.IMPL.KERNELS  A list of joint kernels as the cell array of structs the MEX takes; [] (none) stays [].
%   Internal: k is a struct array or a cell array of scalar structs with type "speed" (c, d) or "ridge" (v, width, sense).
    if isempty(k)
        c = [];
        return
    end
    if isstruct(k), k = num2cell(k); end
    if ~iscell(k), error('nxr:invalidArgument', 'kernels is a struct array or a cell array of structs'); end
    c = cellfun(@nxr.impl.bank, k, 'UniformOutput', false);
end
