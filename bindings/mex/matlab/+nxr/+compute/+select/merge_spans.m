function spans = merge_spans(spans, gap)
%NXR.COMPUTE.SELECT.MERGE_SPANS  The spans whose gap (next start - previous stop) is below gap samples joined (nxr::select::mergeSpans).
%   spans = nxr.compute.select.merge_spans(spans, gap)
%
%   Spans are a struct of column vectors: start (1-based first sample), stop (the LAST sample, inclusive), peak (1-based sample of the extreme value inside) and value.
%   The library holds them 0-based with stop exclusive (the Python face's Spans); the conversion is made at the boundary: start + 1, stop unchanged, peak + 1.
%   The gap is the samples BETWEEN two spans (next start - previous last sample - 1 in this face's inclusive stop), compared with gap as the library does.
    arguments
        spans (1,1) struct
        gap (1,1) double
    end
    spans = nxr_compute('compute.select.merge_spans', spans, gap);
end
