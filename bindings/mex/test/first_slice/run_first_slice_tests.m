% Runs the first slice's tests (this folder) from MATLAB: the MEX is found under <repo>/build, or in NXR_MEX_DIR.
%   matlab -batch "cd bindings/mex/test/first_slice; run_first_slice_tests"
here = fileparts(mfilename('fullpath'));
results = runtests(here);
disp(table(results));
assert(all([results.Passed]), '%d of %d tests failed', nnz(~[results.Passed]), numel(results));
