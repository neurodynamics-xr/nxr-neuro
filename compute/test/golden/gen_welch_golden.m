function gen_welch_golden(work)
% Brainstorm's PSD (bst_psd) on the synthetic signals of gen_welch_golden.py (work/input.csv, work/good.csv), written to work/bst_out.txt.
% Only Brainstorm's computational function is called (no database, no GUI): addpath of its toolbox, then bst_psd(F, sfreq, WinLength, WinOverlap, BadSegments, ImagingKernel, WinFunc, PowerUnits, IsRelative)
% with WinLength in seconds, WinOverlap in PERCENT (50), WinFunc the mean over windows, PowerUnits 'physical' (U^2/Hz). Brainstorm's source is not read.
% Run:  matlab -batch "gen_welch_golden('<work>')"   
addpath(genpath(fullfile(getenv('BRAINSTORM3_HOME'), 'toolbox')));   % Brainstorm, as a path only
F = readmatrix(fullfile(work, 'input.csv')); fs = 200;
fid = fopen(fullfile(work, 'bst_out.txt'), 'w');
[TF, f, nw] = bst_psd(F, fs, 2, 50, [], [], 'mean', 'physical', 0);
put(fid, 'bst_default_2s', squeeze(TF), nw);
bs = {[901; 1250], [900; 1250], [900; 1249]};   % candidate conventions for a bad span of samples 900..1249 (0-based)
for i = 1:numel(bs)
  [TF, f, nw] = bst_psd(F, fs, 2, 50, bs{i}, [], 'mean', 'physical', 0);
  put(fid, sprintf('bst_bad_%d', i), squeeze(TF), nw);
end
[TF, f, nw] = bst_psd(F, fs, 1, 50, [], [], 'mean', 'physical', 0);
put(fid, 'bst_default_1s', squeeze(TF), nw);
fclose(fid);
end
function put(fid, name, A, nw)
fprintf(fid, '%s %d %d %d\n', name, size(A, 1), size(A, 2), nw);
fprintf(fid, '%.17g ', A.'); fprintf(fid, '\n');
end
