function p = pipelines()
%NXR.LIST.PIPELINES  The pipelines nxr.run takes: the built-in psd and every shipped definition, by name.
%   p = nxr.list.pipelines()
%   A struct array: name, form ('built-in' or 'definition'), summary, and parameters (a struct array: name, default, meaning, unit, choices; each parameter's
%   meaning, default and unit as the command line's `run NAME --help` gives them). nxr.list.help(name) prints them.
    p = jsondecode(nxr_compute('run.pipelines'));
end
