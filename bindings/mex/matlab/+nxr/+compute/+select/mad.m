function s = mad(x)
%NXR.COMPUTE.SELECT.MAD  The robust SD of a 1-D series: 1.4826 * median |x - median(x)| (nxr::select::mad).
%   s = nxr.compute.select.mad(x)
    arguments
        x {mustBeNumeric}
    end
    s = nxr_compute('compute.select.mad', x);
end
