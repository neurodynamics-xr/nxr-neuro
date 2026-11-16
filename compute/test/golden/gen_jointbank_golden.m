% Writes jointbank_golden.txt: Rheome's @jointfilterbank (+ +jtv) on the SPECTRUM of a synthetic sphere (lambda = l(l+1), multiplicity 2l+1, l = 0..5) and
% a synthetic joint spectrum, for test_jointbank.cpp. The time factors are the gains of Rheome's @timefilterbank (itersine, default Gain 2: peak 2) passed as TimeFilters (f in Hz).
% Run:  RHEOME=<path to the rheome checkout> matlab -batch "run('compute/test/golden/gen_jointbank_golden.m')"   (one MATLAB at a time)
rh = getenv('RHEOME');  if isempty(rh), error('set RHEOME to the Rheome checkout'); end
addpath(rh);
here = fileparts(mfilename('fullpath'));
fid = fopen(fullfile(here, 'jointbank_golden.txt'), 'w');
w = @(name, A) i_write(fid, name, A);

lam = [];  for l = 0:5, lam = [lam; repmat(l*(l+1), 2*l+1, 1)]; end   %#ok<AGROW>
N = 96;  fs = 64;  V = 2;
tfb = rheome.timefilterbank(N, 'SamplingFrequency', fs, 'VoicesPerOctave', V, 'Anchor', 1);
M = tfb.NumMembers;
nb = floor(N/2) + 1;  K = numel(lam);
rng(21);  C = randn(K, nb) + 1i * randn(K, nb);
tf = arrayfun(@(m) @(f) gains(tfb, m, f), 1:M, 'UniformOutput', false);      % time factors take f in HERTZ
w('lambda', lam);  w('Cre', real(C));  w('Cim', imag(C));
fprintf(fid, 'params %d %.17g %d\n', N, fs, V);

cases = { 'itersine-sep', 'itersine', {};  'mexhat-speed', 'mexhat', {rheome.jointfilterbank.speedkernel(40, 6)} };
for c = 1:size(cases, 1)
    gfb = rheome.graphfilterbank(lam, 'Wavelet', cases{c, 2});
    kern = cases{c, 3};
    if isempty(kern), jfb = rheome.jointfilterbank(gfb, 'SignalLength', N, 'SamplingFrequency', fs, 'TimeFilters', tf);
    else,             jfb = rheome.jointfilterbank(gfb, 'SignalLength', N, 'SamplingFrequency', fs, 'TimeFilters', tf, 'JointKernels', kern); end
    fprintf(fid, 'case %s %s %d %d %d\n', cases{c, 1}, cases{c, 2}, gfb.NumMembers, M, max(1, numel(kern)));
    Y = wt(jfb, C);  fb = framebounds(jfb);
    w('S', fb.S);  w('bounds', [fb.A fb.B fb.Tightness fb.Uncovered]);
    pick = unique(round(linspace(1, size(Y, 3), 5)));  w('pick', pick);                  % five members spread over the bank (the file stays small)
    for m = pick, w(sprintf('Yre%d', m), real(Y(:, :, m))); w(sprintf('Yim%d', m), imag(Y(:, :, m))); end
    Cd = iwt(jfb, Y, 'dual');  Ct = iwt(jfb, Y, 'tight');
    w('Cdre', real(Cd));  w('Cdim', imag(Cd));  w('Ctre', real(Ct));  w('Ctim', imag(Ct));
    % +jtv on the materialised bank
    G = jointfilters(jfb);  bj = rheome.jtv.bounds(G);  [Gd, ~] = rheome.jtv.dual(G);
    Cs = rheome.jtv.synthesis(rheome.jtv.analysis(C, G), Gd);
    w('jtvS', bj.S);  w('jtvCre', real(Cs));  w('jtvCim', imag(Cs));
    fprintf(fid, 'end\n');
end
fclose(fid);  disp('wrote jointbank_golden.txt');

function i_write(fid, name, A)
    A = full(A);
    fprintf(fid, '%s %d %d\n', name, size(A, 1), size(A, 2));
    fprintf(fid, '%.13g ', A.');  fprintf(fid, '\n');
end
