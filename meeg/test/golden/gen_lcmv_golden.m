function gen_lcmv_golden(out)
% THE LCMV BEAMFORMER (pseudo neural activity index, fixed orientation) AND THE PCA SCOUT of Brainstorm's release of 2018-10-12, on a synthetic case,
% written to lcmv_golden.txt beside this file (or to `out`). Read by meeg/test/test_beamformer_parity.cpp.
%
% Brainstorm is used as a path only (no database, no GUI): its computational functions bst_inverse_linear_2016(HeadModel, OPTIONS) (the 2016 source
% estimation that process_inverse_2016 calls) and bst_scout_value(F, 'pca', Orient, 1, 'none', 1) (the scout function PCA with the sign flip, as
% process_extract_scout calls it before a spectrum). It is found through the environment variable BRAINSTORM_2018_HOME (a checkout of
% github.com/brainstorm-tools/brainstorm3 at the last commit on or before 2018-10-12); nothing else is read.
%
% THE CASE (synthetic): 200 sources on a sphere of radius 6-7 cm (a Fibonacci lattice) with oblique orientations, some pointing inward; 26 MEG
% channels on a helmet at 12 cm, radial point magnetometers (16 labelled MEG MAG, 10 MEG GRAD with a gain 5 times larger: two channel types, so the
% covariances are regularised type by type); the gain by Sarvas' formula in a sphere centred at the origin, rounded to single precision (the stored
% gain is float32). A RECORDING of 1200 samples at 600 Hz: one planted source (11 Hz, amplitude-modulated) plus a background of 34 sinusoids per
% channel; an EMPTY-ROOM recording of 34 other sinusoids. Both are closed-form, so the test makes the same samples from the gain and orientations stored
% here. The data and noise covariances are Brainstorm's (bst_noisecov: one block, its mean removed, divided by N - 1). The regions: 8 sectors of the
% sphere (a partition of the sources).
%
% Run (one MATLAB batch run at a time):  BRAINSTORM_2018_HOME=<checkout> matlab -batch "gen_lcmv_golden"
bst = getenv('BRAINSTORM_2018_HOME');
if isempty(bst) || ~exist(fullfile(bst, 'toolbox', 'inverse', 'bst_inverse_linear_2016.m'), 'file')
  error('gen_lcmv_golden: set BRAINSTORM_2018_HOME to a checkout of Brainstorm at its release of 2018-10-12');
end
addpath(fullfile(bst, 'toolbox', 'inverse'), fullfile(bst, 'toolbox', 'math'), fullfile(bst, 'toolbox', 'misc'));
here = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(out), out = fullfile(here, 'lcmv_golden.txt'); end
ver = fileread(fullfile(bst, 'doc', 'version.txt'));
ver = strtrim(regexprep(ver, '[\r\n%]+', ' '));
[st, commit] = system(['git -C "' bst '" rev-parse HEAD']);
if st ~= 0, commit = 'unknown'; end
commit = strtrim(commit);

% ── the geometry ──
N = 200; M = 26; nMag = 16; ga = pi * (3 - sqrt(5));
P = zeros(N, 3); U = zeros(N, 3); Q = zeros(N, 3); region = zeros(N, 1);
for i = 1:N
  z = 1 - 2 * (i - 0.5) / N; rho = sqrt(1 - z^2); phi = (i - 1) * ga;
  r = 0.06 + 0.01 * mod((i - 1) * 0.618034, 1);
  u = [rho * cos(phi), rho * sin(phi), z];
  P(i, :) = r * u; U(i, :) = u;
  t1 = cross(u, [0.3 0.2 1]); t1 = t1 / norm(t1); t2 = cross(u, t1);
  th = 0.2 + 0.6 * mod((i - 1) * 0.7548776662, 1); psi = 1.3 * (i - 1);
  q = cos(th) * u + sin(th) * (cos(psi) * t1 + sin(psi) * t2);
  if mod(floor((i - 1) / 3), 4) == 0, q = -q; end                      % some sources point inward: the scout's sign flip has work to do
  Q(i, :) = q / norm(q);
  region(i) = floor(mod(phi, 2 * pi) / (pi / 2)) + 4 * (z < 0);          % 8 sectors, numbered 0 .. 7
end
S = zeros(M, 3); O = zeros(M, 3); gscale = ones(M, 1); types = cell(M, 1);
for c = 1:M
  z = 0.15 + 0.85 * (c - 0.5) / M; rho = sqrt(1 - z^2); phi = (c - 1) * ga;
  S(c, :) = 0.12 * [rho * cos(phi), rho * sin(phi), z]; O(c, :) = S(c, :) / norm(S(c, :));
  if c <= nMag, types{c} = 'MEG MAG'; else, types{c} = 'MEG GRAD'; gscale(c) = 5; end
end
% Sarvas (1987): the field of a current dipole q at r0 in a conducting sphere centred at the origin, at r, along o
mu = 4e-7 * pi;
G = zeros(M, 3 * N);
for c = 1:M
  r = S(c, :); rn = norm(r);
  for i = 1:N
    r0 = P(i, :); a = r - r0; an = norm(a);
    F = an * (rn * an + rn^2 - dot(r0, r));
    gF = (an^2 / rn + dot(a, r) / an + 2 * an + 2 * rn) * r - (an + 2 * rn + dot(a, r) / an) * r0;
    for j = 1:3
      q = zeros(1, 3); q(j) = 1;
      B = mu / (4 * pi * F^2) * (F * cross(q, r0) - dot(cross(q, r0), r) * gF);
      G(c, 3 * (i - 1) + j) = gscale(c) * dot(B, O(c, :));
    end
  end
end
G = double(single(G));

% ── the recordings (closed form; the test makes the same samples) ──
fs = 600; T = 1200; p = 37;                                             % the planted source (0-based index)
j = 0:(T - 1);
lp = G(:, 3 * p + (1:3)) * Q(p + 1, :)';
s = 2e-8 * sin(2 * pi * 11 * j / fs) .* (1 + 0.5 * sin(2 * pi * 0.7 * j / fs));
X = lp * s; Xn = zeros(M, T);
for c = 0:(M - 1)
  b = zeros(1, T); bn = zeros(1, T);
  for k = 1:34
    b = b + cos(0.37 * c * k + 0.31 * k) * sin(2 * pi * (2.5 + 2.9 * k) * j / fs + 0.29 * c * k + 1.3 * k);
    bn = bn + cos(0.23 * c * k + 0.53 * k) * sin(2 * pi * (1.9 + 3.1 * k) * j / fs + 0.41 * c * k + 0.7 * k);
  end
  X(c + 1, :) = X(c + 1, :) + 1e-13 * gscale(c + 1) * b;
  Xn(c + 1, :) = 1e-13 * gscale(c + 1) * bn;
end
covOf = @(Y) ((Y - mean(Y, 2)) * (Y - mean(Y, 2))') / (size(Y, 2) - 1);   % bst_noisecov on one block: its mean removed, / (N - 1)
Cd = covOf(X); Cn = covOf(Xn);

% ── the beamformer, as process_inverse_2016 calls it (the authors' options: LCMV, NAI, fixed, median, 0.1; depth and SNR are ignored by LCMV) ──
HM = struct('Gain', G, 'GridLoc', P, 'GridOrient', Q, 'HeadModelType', 'surface');
variants = {'median', 'reg', 'none', 'diagnoise'};
res = struct();
for v = 1:numel(variants)
  Ov = bst_inverse_linear_2016();
  Ov.InverseMethod = 'lcmv'; Ov.InverseMeasure = 'nai'; Ov.SourceOrient = {'fixed'}; Ov.Loose = 0.2;
  Ov.UseDepth = 1; Ov.WeightExp = 0.5; Ov.WeightLimit = 10;
  Ov.NoiseMethod = 'median'; Ov.NoiseReg = 0.1;
  Ov.SnrMethod = 'rms'; Ov.SnrRms = 1e-6; Ov.SnrFixed = 3; Ov.ComputeKernel = 1;
  Ov.ChannelTypes = types';
  CnV = Cn;
  switch variants{v}
    case 'reg', Ov.NoiseMethod = 'reg';
    case 'none', Ov.NoiseMethod = 'none';
    case 'diagnoise', CnV = diag(diag(Cn));                              % a diagonal noise covariance: the 2016 code then whitens BOTH covariances by their diagonal
  end
  Ov.NoiseCovMat = struct('NoiseCov', CnV, 'nSamples', T);
  Ov.DataCovMat = struct('NoiseCov', Cd, 'nSamples', T);
  [R, ~] = bst_inverse_linear_2016(HM, Ov);
  res.(variants{v}) = struct('K', full(R.ImagingKernel), 'W', full(R.Whitener), 'DW', full(R.DataWhitener));
  fprintf('%s: kernel %dx%d\n', variants{v}, size(R.ImagingKernel, 1), size(R.ImagingKernel, 2));
end

% ── the PCA scout over a span, after the median kernel, as process_extract_scout calls it (isflip = 1 for sources) ──
span0 = 150; spanN = 900;                                               % samples span0 .. span0 + spanN - 1 (0-based)
K = res.median.K;
Y = zeros(8, spanN);
for rr = 0:7
  iv = find(region == rr);
  Fr = K(iv, :) * X(:, span0 + (1:spanN));
  [u, ~, ~] = svd(Q(iv, :), 0); fm = sign(u(:, 1));
  if nnz(fm > 0) == nnz(fm < 0), error('gen_lcmv_golden: region %d has a tie in its orientation majority (the sign of an SVD would decide)', rr); end
  Y(rr + 1, :) = bst_scout_value(Fr, 'pca', Q(iv, :), 1, 'none', 1, sprintf('region %d', rr));
end

% ── written row-major: "name rows cols" then the values on one line ──
fid = fopen(out, 'w');
fprintf(fid, '# lcmv golden: %s, commit %s (bst_inverse_linear_2016: LCMV, NAI, fixed; bst_scout_value: pca with sign flip); see gen_lcmv_golden.m\n', ver, commit);
fprintf(fid, '# MATLAB %s\n', version);
put(fid, 'params', [N M T fs p span0 spanN nMag], '%.17g');
put(fid, 'G', G, '%.9g');                                               % float32 values: exact in 9 digits
put(fid, 'orient', Q, '%.17g');
put(fid, 'types', [zeros(1, nMag), ones(1, M - nMag)], '%d');
put(fid, 'gscale', gscale', '%.17g');
put(fid, 'region', region', '%d');
put(fid, 'Cd', Cd, '%.17g');
put(fid, 'Cn', Cn, '%.17g');
for v = 1:numel(variants)
  put(fid, [variants{v} '.K'], res.(variants{v}).K, '%.12g');
  put(fid, [variants{v} '.W'], res.(variants{v}).W, '%.12g');
  put(fid, [variants{v} '.DW'], res.(variants{v}).DW, '%.12g');
end
put(fid, 'scout_pca', Y, '%.12g');
fclose(fid);
fprintf('wrote %s\n', out);
end

function put(fid, name, A, fmt)
  fprintf(fid, '%s %d %d\n', name, size(A, 1), size(A, 2));
  fprintf(fid, [fmt ' '], A');
  fprintf(fid, '\n');
end
