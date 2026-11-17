function [x, header] = read_span(rec, opts)
%NXR.IMPL.READ_SPAN  The span rule of the read_<format> functions: first and last, or all, never a default span.
    if opts.all
        if ~isempty(opts.first) || ~isempty(opts.last)
            error('nxr:invalidArgument', 'give first and last, or all=true, not both');
        end
        x = rec.read_all(first_channel=opts.first_channel, last_channel=opts.last_channel);
    else
        if isempty(opts.first) || isempty(opts.last)
            error('nxr:invalidArgument', 'the span is the caller''s: give first and last (samples), or all=true');
        end
        x = rec.read(opts.first, opts.last, first_channel=opts.first_channel, last_channel=opts.last_channel);
    end
    header = rec.header;
end
