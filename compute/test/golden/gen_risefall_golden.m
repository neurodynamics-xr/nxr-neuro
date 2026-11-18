function gen_risefall_golden(out)
% Rheome's rheome.detect.risefall on synthetic episodes with known parameters: the golden for nxr::select (test_risefall.cpp). Rheome is called as a path only (the Rheome code).
% Cases: Model "linear" (episodes a = b + h S S, with noise), Model "quadrature" (a = sqrt(b^2 + (h S S)^2), with noise), a short window (not ok), and a flat series (not ok).
% File: one line a case: name, then the inputs and Rheome's outputs as `key n v1 .. vn`.
% Run:  matlab -batch "gen_risefall_golden('<file>')"   
addpath(getenv('RHEOME_HOME'));
rng(5);
fs = 50; T = 40*fs; t = (0:T-1)/fs;
S = @(x) 1 ./ (1 + exp(-x));
fid = fopen(out, 'w');
cases = {
  % name   model        b     h    tOn  tOff  tauR  tauF  noise  t0   t1
  'lin1',  'linear',    0.20, 1.0, 12.0, 18.5, 0.30, 0.45, 0.02, 11.0, 19.5;
  'lin2',  'linear',    0.05, 2.5, 22.0, 30.0, 0.15, 0.60, 0.05, 21.5, 31.0;
  'quad1', 'quadrature',0.30, 1.0, 10.0, 17.0, 0.25, 0.35, 0.02,  9.0, 18.0;
  'quad2', 'quadrature',0.50, 1.5, 24.0, 28.5, 0.50, 0.20, 0.03, 23.0, 29.5;
};
for i = 1:size(cases,1)
  c = cases(i,:);  [nm, mdl, b, h, on, off, tr, tf, nz, t0, t1] = c{:};
  g = S((t - on)/tr) .* S(-(t - off)/tf);
  if strcmp(mdl, 'linear'), a = b + h*g; else, a = sqrt(b^2 + (h*g).^2); end
  a = a + nz*randn(1, T);
  R = rheome.detect.risefall(a, fs, t0, t1, Model=mdl);
  put(fid, [nm '_model'], double(strcmp(mdl, 'quadrature')));
  put(fid, [nm '_fs'], fs); put(fid, [nm '_span'], [t0 t1]); put(fid, [nm '_a'], a);
  put(fid, [nm '_truth'], [b h on off tr tf]);
  put(fid, [nm '_fit'], [R.tOn R.tOff R.tauR R.tauF R.b R.h R.r2 R.ok R.atBound]);
  put(fid, [nm '_plateau'], [numel(R.plateau) R.plateau(:).']);
end
% a window under 11 samples, and a series of no amplitude: not ok
a = abs(randn(1, 300));
R = rheome.detect.risefall(a, 50, 2, 2.1, Margin=0);
put(fid, 'short_a', a); put(fid, 'short_fit', [R.ok]);
R = rheome.detect.risefall(zeros(1, 600), 50, 3, 6);
put(fid, 'zero_fit', [R.ok]);
fclose(fid);
end

function put(fid, k, v)
  fprintf(fid, '%s %d', k, numel(v));
  fprintf(fid, ' %.17g', v);
  fprintf(fid, '\n');
end
