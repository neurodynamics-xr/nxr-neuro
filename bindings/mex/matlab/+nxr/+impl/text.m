function c = text(v)
%NXR.IMPL.TEXT  A text option as the character vector the MEX takes; [] (not given) stays [].
%   Internal: converts a string scalar or char vector for nxr_compute. It checks nothing else.
    if isempty(v)
        c = [];
    else
        c = char(v);
    end
end
