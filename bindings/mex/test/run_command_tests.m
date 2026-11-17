% Runs the nxr_compute command tests (the test_*.m function files and the smoke test in this folder; the first slice's tests are in first_slice/).
% The MEX is found under <repo>/build. Each file runs in its own try; the failures are listed at the end and make the run fail.
here = fileparts(mfilename('fullpath'));
files = [dir(fullfile(here, 'test_*.m')); dir(fullfile(here, 'nxr_compute_mex_smoke.m'))];
failed = {};
for i = 1:numel(files)
    [~, name] = fileparts(files(i).name);
    try
        runOne(name);
        fprintf('  pass  %s\n', name);
    catch e
        fprintf('  FAIL  %s: %s\n', name, strtok(e.message, newline));
        failed{end+1} = name; %#ok<AGROW>
    end
    clear nxr_compute
end
fprintf('%d of %d command tests passed\n', numel(files) - numel(failed), numel(files));
assert(isempty(failed), 'failed: %s', strjoin(failed, ', '));

function runOne(name__)
    % in a workspace of its own: the smoke test is a script that clears its variables
    evalc([name__ ';']);
end
