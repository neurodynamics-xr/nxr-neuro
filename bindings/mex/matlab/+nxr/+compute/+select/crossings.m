function spans = crossings(x, entry, exit)
%NXR.COMPUTE.SELECT.CROSSINGS  The spans where x >= exit, each holding at least one sample >= entry (nxr::select::crossings).
%   spans = nxr.compute.select.crossings(x, entry, exit)
%
%   Hysteresis: a span is entered high and extended while above the low level; entry = exit is a plain threshold. peak and value are the maximum inside.
%   A non-finite sample ends a span.
%   Spans are a struct of column vectors: start (1-based first sample), stop (the LAST sample, inclusive), peak (1-based sample of the extreme value inside) and value.
%   The library holds them 0-based with stop exclusive (the Python face's Spans); the conversion is made at the boundary: start + 1, stop unchanged, peak + 1.
%
%   Inputs
%     x       vector: any series.
%     entry   the level a span must reach.
%     exit    the level a span extends down to.
    arguments
        x {mustBeNumeric}
        entry (1,1) double
        exit (1,1) double
    end
    spans = nxr_compute('compute.select.crossings', x, entry, exit);
end
