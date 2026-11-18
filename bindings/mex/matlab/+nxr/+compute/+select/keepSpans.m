function spans = keepSpans(spans, min_length, max_length)
%NXR.COMPUTE.SELECT.KEEPSPANS  The spans whose length is in [min_length, max_length] samples kept (nxr::select::keepSpans).
%   spans = nxr.compute.select.keepSpans(spans, 20)
%   spans = nxr.compute.select.keepSpans(spans, 20, 300)
%
%   Spans are a struct of column vectors: start (1-based first sample), stop (the LAST sample, inclusive), peak (1-based sample of the extreme value inside) and value.
%   The library holds them 0-based with stop exclusive (the Python face's Spans); the conversion is made at the boundary: start + 1, stop unchanged, peak + 1.
%     min_length   samples (default 0).
%     max_length   samples (default Inf: no upper bound).
    arguments
        spans (1,1) struct
        min_length (1,1) double = 0
        max_length (1,1) double = Inf
    end
    spans = nxr_compute('compute.select.keepSpans', spans, min_length, max_length);
end
