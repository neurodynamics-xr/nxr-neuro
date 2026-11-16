function gen_source_psd_golden(outfile)
% Brainstorm's output for source band maps on synthetic data: bst_psd on K x (Brainstorm's imaging-kernel option: the PSD computed ON SOURCES), then
% process_tf_bands('Compute') ("Group in frequency bands", default bands, function mean), and process_tf_norm('Compute', ..., 'relative') ("Spectrum normalization: relative").
% Only Brainstorm's computational functions are called, on matrices (no database, no GUI); its source is not read.
% Run:  matlab -batch "gen_source_psd_golden('<outfile>')"   (one MATLAB run at a time)
addpath(genpath(fullfile(getenv('BRAINSTORM3_HOME'), 'toolbox')));
rng(7);
fs = 250; M = 20; N = 50; T = 2000;
t = (0:T-1) / fs;
mix = randn(M, 4);
x = mix * [sin(2*pi*10*t + 1); sin(2*pi*20*t); sin(2*pi*40.5*t + 2); sin(2*pi*3.1*t)] + 0.5 * randn(M, T);
K = randn(3*N, M);
% PSD on sources: 4 s windows, 50 % overlap, Brainstorm's defaults
[TF, freqs, nw] = bst_psd(x, fs, 4, 50, [], K, 'mean', 'physical', 0);
TF = squeeze(TF);                         % [3N x bins], power per Hz, each orientation apart
% also the sensor PSD, for the equality of the cross-spectrum route
bandsDef = {'delta', '2, 4', 'mean'; 'theta', '5, 7', 'mean'; 'alpha', '8, 12', 'mean'; 'beta', '15, 29', 'mean'; 'gamma1', '30, 59', 'mean'; 'gamma2', '60, 90', 'mean'};
Fm = struct('TF', reshape(TF, size(TF,1), 1, size(TF,2)), 'Freqs', freqs(:)', 'Time', 0, 'Measure', 'power', 'Method', 'psd', 'RowNames', {arrayfun(@num2str, 1:size(TF,1), 'UniformOutput', false)});
ok = true; msg = '';
try
  Fb = process_tf_bands('Compute', Fm, bandsDef, []);
  B = squeeze(Fb.TF);                     % [3N x nBands]
catch e
  ok = false; msg = e.message; B = nan(size(TF,1), 6);
end
% the documented rule: mean of the bins from lo to hi inclusive
lo = [2 5 8 15 30 60]; hi = [4 7 12 29 59 90];
R = zeros(size(TF,1), 6);
for b = 1:6, R(:,b) = mean(TF(:, freqs >= lo(b) & freqs <= hi(b)), 2); end
% relative power: Brainstorm's normalisation of the full PSD, then the bands
[TFn, errMsg] = process_tf_norm('Compute', reshape(TF, size(TF,1), 1, size(TF,2)), 'power', freqs(:)', 'relative');
Bn = zeros(size(TF,1), 6);
TFn = squeeze(TFn);
for b = 1:6, Bn(:,b) = mean(TFn(:, freqs >= lo(b) & freqs <= hi(b)), 2); end
tot = sum(TF, 2);
fid = fopen(outfile, 'w');
fprintf(fid, '# bands_function_ok %d %s\n', ok, strrep(msg, newline, ' '));
put(fid, 'fs', fs); put(fid, 'x', x); put(fid, 'K', K); put(fid, 'freqs', freqs(:)');
put(fid, 'psd', TF); put(fid, 'bands_bst', B); put(fid, 'bands_rule', R); put(fid, 'relative_bst', Bn); put(fid, 'total_sum', tot); put(fid, 'nwin', nw);
fclose(fid);
end
function put(fid, name, A)
fprintf(fid, '%s %d %d\n', name, size(A, 1), size(A, 2));
fprintf(fid, '%.10g ', A.'); fprintf(fid, '\n');
end
