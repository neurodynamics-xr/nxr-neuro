% Writes timebank_golden.txt: Rheome's @timefilterbank (itersine, gain 2, analytic) and MATLAB's cwtfilterbank (Morse / Amor / Bump, Rheome's reference
% path in +ingest/bank.m) on SYNTHETIC signals, for test_timebank.cpp. Synthetic data only.
% Run:  RHEOME=<path to the rheome checkout> matlab -batch "run('compute/test/golden/gen_timebank_golden.m')"   (one MATLAB at a time)
rh = getenv('RHEOME');  if isempty(rh), error('set RHEOME to the Rheome checkout'); end
addpath(rh);
here = fileparts(mfilename('fullpath'));
fid = fopen(fullfile(here, 'timebank_golden.txt'), 'w');
w = @(name, A) i_write(fid, name, A);

% ── Rheome's timefilterbank ──
cases = { 192, 100, 4, 1, [];   300, 250, 2, 1, [2 60];   151, 128, 3, 0.5, [];   240, 200, 6, 1, [] };
for c = 1:size(cases, 1)
    [N, fs, V, a, lim] = deal(cases{c, :});
    if isempty(lim), tfb = rheome.timefilterbank(N, 'SamplingFrequency', fs, 'VoicesPerOctave', V, 'Anchor', a);
    else,            tfb = rheome.timefilterbank(N, 'SamplingFrequency', fs, 'VoicesPerOctave', V, 'Anchor', a, 'FrequencyLimits', lim); end
    rng(10 + c);  t = (0:N-1)' / fs;
    x = randn(N, 1) + 0.7 + 2 * cos(2*pi*5.3*t) + 0.5 * sin(2*pi*(0.9*fs/2)*t);
    fprintf(fid, 'itersine %d %.17g %d %.17g %d %.17g %.17g\n', N, fs, V, a, ~isempty(lim), i_get(lim, 1), i_get(lim, 2));
    [H, f] = freqz(tfb);
    b = framebounds(tfb);  [r, Nm] = rates(tfb);
    w('H', H.');  w('fc', centerFrequencies(tfb));  w('bins', bins(tfb));
    w('bounds', [b.A b.B b.Tightness b.Uncovered]);  w('rates', r(:)');  w('Nm', Nm(:)');
    w('x', x);
    X = fft(x);  nb = numel(f);  M = size(H, 1);
    Cr = zeros(N, M);  Ci = zeros(N, M);
    for m = 1:M
        Y = zeros(N, 1);  Y(1:nb) = X(1:nb) .* H(m, :).';
        cm = ifft(Y);  Cr(:, m) = real(cm);  Ci(:, m) = imag(cm);
    end
    w('Cre', Cr);  w('Cim', Ci);
    fprintf(fid, 'end\n');
end

% ── MATLAB's cwtfilterbank ──
cw = { 'morse', 128, 200, 6, [4 60], {};  'morse', 128, 200, 5, [3 40], {'TimeBandwidth', 30};  'amor', 128, 200, 6, [4 60], {};  'bump', 128, 200, 6, [4 60], {} };
for c = 1:size(cw, 1)
    [wv, N, fs, V, lim, extra] = deal(cw{c, :});
    fbk = cwtfilterbank('SignalLength', N, 'SamplingFrequency', fs, 'Wavelet', wv, 'VoicesPerOctave', V, 'FrequencyLimits', lim, extra{:});
    [H, f] = freqz(fbk);  fc = centerFrequencies(fbk);
    rng(30 + c);  t = (0:N-1)' / fs;
    x = randn(N, 1) + 0.7 + 2 * cos(2*pi*12.3*t);
    xe = [x; flipud(x)];  Ne = 2 * N;                  % cwtfilterbank extends the record symmetrically to 2N and keeps the first N samples
    X = fft(xe);  nb = numel(f);  M = size(H, 1);
    Cr = zeros(N, M);  Ci = zeros(N, M);
    for m = 1:M
        Y = zeros(Ne, 1);  Y(1:nb) = X(1:nb) .* H(m, :).';
        cm = ifft(Y);  Cr(:, m) = real(cm(1:N));  Ci(:, m) = imag(cm(1:N));
    end
    W = wt(fbk, x);                                   % [M x N] complex
    fprintf(fid, 'cwt %s %d %.17g %d %.17g %.17g %s\n', wv, N, fs, V, lim(1), lim(2), strjoin(cellfun(@i_s, extra, 'UniformOutput', false), ' '));
    w('fc', fc(:)');  w('H', H.');  w('x', x);  w('Cre', Cr);  w('Cim', Ci);  w('Wre', real(W).');  w('Wim', imag(W).');
    fprintf(fid, 'end\n');
end
fclose(fid);  disp('wrote timebank_golden.txt');

function i_write(fid, name, A)
    A = full(A);
    fprintf(fid, '%s %d %d\n', name, size(A, 1), size(A, 2));
    fprintf(fid, '%.13g ', A.');  fprintf(fid, '\n');
end
function v = i_get(lim, k), if isempty(lim), v = 0; else, v = lim(k); end, end
function s = i_s(x), if ischar(x), s = x; else, s = sprintf('%.17g', x); end, end
