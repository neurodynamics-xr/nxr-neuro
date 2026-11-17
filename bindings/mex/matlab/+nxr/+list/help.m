function text = help(name)
%NXR.LIST.HELP  A pipeline's help: its parameters' meaning, default and unit (what `nxr-neuro run NAME --help` prints).
%   nxr.list.help('psd')
    p = nxr.list.pipelines();
    hit = find(strcmp({p.name}, char(name)), 1);
    if isempty(hit)
        error('nxr:invalidArgument', 'no pipeline named %s (nxr.list.pipelines() lists them)', char(name));
    end
    p = p(hit);
    lines = {sprintf('%s: %s', p.name, p.summary), ''};
    params = p.parameters;
    for k = 1:numel(params)
        q = params(k);
        if iscell(params), q = params{k}; end
        def = 'none';
        if ~isempty(q.default), def = strtrim(formattedDisplayText(q.default)); end
        choices = '';
        if ~isempty(q.choices), choices = [' (one of: ' strjoin(cellstr(string(q.choices)), ', ') ')']; end
        unit = '';
        if ~isempty(q.unit) && ~strcmp(q.unit, '-'), unit = [' [' q.unit ']']; end
        lines{end+1} = sprintf('  %s%s: %s%s; default %s', q.name, unit, q.meaning, choices, def); %#ok<AGROW>
    end
    text = strjoin(lines, newline);
    if nargout == 0, disp(text); clear text; end
end
