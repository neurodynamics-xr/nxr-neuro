function gen_fingerprint_features_golden(out)
% THE PER-SUBJECT FEATURES OF MEG FINGERPRINTING with Brainstorm's release of 2018-10-12 (the version the paper used), on synthetic region signals, written to
% fingerprint_features_golden.txt beside this file (or to `out`). Read by compute/test/test_fingerprint_features.cpp.
%
% The release (megFingerprinting v1.0.0, 06fdba8) calls, on each band's region signals over a half of the run:
%   process_psd (2_prerprocessing_PSD.m: win_length 2 s, win_overlap 50 %, measure power)  ->  bst_psd(F, sfreq, 2, 50, BadSegments, [], 0) of that release
%   process_aec1n (2_preprocessing_AEC_ortho.m: isorth 0)  ->  bst_connectivity's AEC: process_bandpass('Compute', F, sfreq, lo, hi) (its default filter), hilbert, abs, bst_corrn
% Only those computational functions are called here (no database, no GUI): Brainstorm is a path, found through the environment variable BRAINSTORM_2018_HOME (a checkout of
% github.com/brainstorm-tools/brainstorm3 at the last commit on or before 2018-10-12). Its source is not copied.
%
% THE CASE (closed form: the test makes the same samples): R = 4 region signals at 200 Hz, 40 s; each a sum of 16 sinusoids (1.7 + 3.1 k Hz) with slow amplitude modulations, a slow drift and
% an offset (so the mean removal matters); the rows share some modulations (their envelopes correlate). The HALF is samples 1001..4001 (MATLAB indices; 5 s to 20 s inclusive, 3001 samples),
% what process_aec1n and process_psd load for a time window [5, 20]. The bands: [1 4], [8 13], [30 50] Hz.
% Written: the filters' orders, the band-passed half at every 21st sample, the envelope at every 21st sample, the AEC [R x R] a band, the PSD [R x bins] of the half and its window count.
%
% Run (one MATLAB batch run at a time):  BRAINSTORM_2018_HOME=<checkout> matlab -batch "gen_fingerprint_features_golden"
bst = getenv('BRAINSTORM_2018_HOME');
if isempty(bst) || ~exist(fullfile(bst, 'toolbox', 'timefreq', 'bst_psd.m'), 'file')
  error('gen_fingerprint_features_golden: set BRAINSTORM_2018_HOME to a checkout of Brainstorm at its release of 2018-10-12');
end
addpath(genpath(fullfile(bst, 'toolbox')));
here = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(out), out = fullfile(here, 'fingerprint_features_golden.txt'); end
ver = fileread(fullfile(bst, 'doc', 'version.txt'));
ver = strtrim(regexprep(ver, '[\r\n%]+', ' '));
[st, commit] = system(['git -C "' bst '" rev-parse HEAD']);
if st ~= 0, commit = 'unknown'; end
commit = strtrim(commit);

fs = 200; N = 8000; R = 4; K = 16;
t = (0:N-1) / fs;
x = zeros(R, N);
for r = 0:R-1
  for k = 0:K-1
    f = 1.7 + 3.1 * k;
    m = 1 + 0.6 * cos(2 * pi * 0.13 * (k + 1) * t + 0.7 * mod(r, 2) + k);       % rows 0, 2 and 1, 3 share their modulations
    x(r + 1, :) = x(r + 1, :) + m .* cos(2 * pi * f * t + 1.3 * k + 0.5 * r) / (1 + 0.2 * k);
  end
  x(r + 1, :) = x(r + 1, :) + 0.3 * cos(2 * pi * 0.05 * t + r) + 2 + r;
end
half = 1001:4001;
xw = x(:, half);
bands = [1 4; 8 13; 30 50];

fid = fopen(out, 'w');
fprintf(fid, '# Brainstorm %s, commit %s; MATLAB %s; hilbert: the Signal Processing Toolbox''s (%d); UseSigProcToolbox: %d\n', ver, commit, version, exist('hilbert', 'file') == 2, bst_get('UseSigProcToolbox'));
put(fid, 'fs', fs); put(fid, 'half', [half(1) - 1, numel(half)]);   % 0-based first sample, count
put(fid, 'bands', bands);
for b = 1:size(bands, 1)
  [y, spec] = process_bandpass('Compute', xw, fs, bands(b, 1), bands(b, 2));
  env = abs(hilbert(y')');
  put(fid, sprintf('order_%d', b), spec.order);
  put(fid, sprintf('bandpassed_%d', b), y(:, 1:21:end));
  put(fid, sprintf('envelope_%d', b), env(:, 1:21:end));
  put(fid, sprintf('aec_%d', b), bst_corrn(env, env));
end
[TF, freqs, nwin] = bst_psd(xw, fs, 2, 50, [], [], 0);
put(fid, 'psd', squeeze(TF)); put(fid, 'psd_freq', freqs(:).'); put(fid, 'psd_windows', nwin);
fclose(fid);
end
function put(fid, name, A)
fprintf(fid, '%s %d %d\n', name, size(A, 1), size(A, 2));
fprintf(fid, '%.17g ', A.'); fprintf(fid, '\n');
end
