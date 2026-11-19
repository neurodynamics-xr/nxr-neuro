function s = given(varargin)
%NXR.IMPL.GIVEN  A struct of the name-value pairs whose value is not [] (a design field left [] is not given and keeps the library's default); strings become char.
%   Internal.
    s = struct();
    for i = 1:2:numel(varargin)
        v = varargin{i+1};
        if isempty(v), continue, end
        if isstring(v), v = char(v); end
        s.(varargin{i}) = v;
    end
end
