function m = median(x)
%NXR.COMPUTE.SELECT.MEDIAN  The median of a 1-D series (nxr::select::median).
%   m = nxr.compute.select.median(x)
%     x   a vector (any series). Returns the median over its values.
    arguments
        x {mustBeNumeric}
    end
    m = nxr_compute('compute.select.median', x);
end
