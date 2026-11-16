% Writes kernels_golden.txt: Rheome's +filters (space-time kernels, sphere_kernel, steer, spin, scatter) and jointfilterbank.speedkernel / band, on a SYNTHETIC sphere
% spectrum and a synthetic icosphere, for test_spacetime_kernels.cpp. Synthetic data only.
% Run:  RHEOME=<path to the rheome checkout> matlab -batch "run('compute/test/golden/gen_kernels_golden.m')"   (one MATLAB at a time)
rh = getenv('RHEOME');  if isempty(rh), error('set RHEOME to the Rheome checkout'); end
addpath(rh);
here = fileparts(mfilename('fullpath'));
fid = fopen(fullfile(here, 'kernels_golden.txt'), 'w');
w = @(name, A) i_write(fid, name, A);

lam = [];  for l = 0:5, lam = [lam; repmat(l*(l+1), 2*l+1, 1)]; end   %#ok<AGROW>
N = 48;  fs = 40;  grid = (0:N-1) * fs / N;                           % a FULL FFT grid, in Hz (the way +filters reads it)
half = (0:floor(N/2)) * fs / N;                                        % a ONE-sided grid (bins 0 ... fs/2)
ws = grid - fs * (grid >= fs/2);
t = linspace(-2, 3, 17);
w('lambda', lam);  w('grid', grid);  w('half', half);  w('t', t);
w('signedHalf', rheome.filters.signedfrequencies(half, 'positive'));                  % signedfrequencies on the one-sided grid: unchanged
w('signedFull', rheome.filters.signedfrequencies(grid, 'full'));
w('gabor', rheome.filters.gabor(lam, grid, 20, 10, 2));
w('travwave', rheome.filters.travwave(lam, grid, 3, 2, 0));
w('travwave_p', rheome.filters.travwave(lam, grid, 3, 2, 1));  w('travwave_m', rheome.filters.travwave(lam, grid, 3, 2, -1));
w('travwave_half', rheome.filters.travwave(lam, half, 3, 2, 1, 'positive'));
w('gabor_half', rheome.filters.gabor(lam, half, 20, 10, 2, 'positive'));
R = rheome.filters.resonator(lam, grid, 10, 6);  w('resRe', real(R));  w('resIm', imag(R));
w('stmatern', rheome.filters.stmatern(lam, grid, 5, 1.5, 2));
w('bandpass', rheome.filters.bandpass(grid, 10, 3));
% the CONTINUUM propagators, physical units (not normalised by lambda_max)
w('diffusion', rheome.filters.diffusion(lam, t, 1.3));
w('wave', rheome.filters.wave(lam, t, 1.7));
w('dampedwave', rheome.filters.dampedwave(lam, t, 1.7, 0.2));
w('kleingordon', rheome.filters.kleingordon(lam, t, 1.2, 0.3));
S = rheome.filters.schrodinger(lam, t, 1.5);  w('schRe', real(S));  w('schIm', imag(S));
% the Chebyshev-angle ('discrete') variants: the numbers the earlier definitions gave, kept
w('wave_cheb', rheome.filters.wave(lam, t, 1.7, 'Variant', 'discrete'));
w('wave_cheb_lmax', rheome.filters.wave(lam, t, 1.7, 'Variant', 'discrete', 'LambdaMax', 50));
w('dampedwave_cheb', rheome.filters.dampedwave(lam, t, 1.7, 0.2, 'Variant', 'discrete'));
w('kleingordon_cheb', rheome.filters.kleingordon(lam, t, 1.2, 0.3, 'Variant', 'discrete'));
w('heat', rheome.filters.heat(lam, [0.05 0.2 0.5], []));
w('heat_lmax', rheome.filters.heat(lam, [0.05 0.2 0.5], 30));
w('mexhat', rheome.filters.mexhat(lam, [0.05 0.2 0.5]));
w('diffgauss', rheome.filters.diffgauss(lam, [0.05 0.1], [0.2 0.5]));
gam = linspace(0, pi, 15)';  gl = exp(-(0:8)' * 0.4);
w('gamma', gam);  w('gl', gl);  w('sphere_kernel', rheome.filters.sphere_kernel(gam, gl));
band = rheome.jointfilterbank.band(8, 14, 1.5);  f1 = linspace(0, 20, 41);
w('f1', f1);  w('psitime', rheome.filters.psitime(f1, [8 14], 1.5));  w('band_hz', band(f1));
sk = rheome.jointfilterbank.speedkernel(40, 6);  w('speed', sk(lam, f1));          % f in Hz; omega = 2 pi f inside
% quaternion fields
rng(4);  Wq = randn(12, 3);  q0 = [0.3; -0.5; 0.2; 0.7];
w('Wq', Wq);  w('q0', q0);  w('steer', rheome.filters.steer(Wq, q0));
V = randn(7, 3);  ax = randn(7, 3);  th = randn(7, 1);
w('V', V);  w('axis', ax);  w('theta', th);  w('spin', rheome.filters.spin(V, ax, th));

% ── scatter on a synthetic icosphere (level 1, 42 vertices) ──
tt = (1 + sqrt(5)) / 2;
Vt = [-1 tt 0; 1 tt 0; -1 -tt 0; 1 -tt 0; 0 -1 tt; 0 1 tt; 0 -1 -tt; 0 1 -tt; tt 0 -1; tt 0 1; -tt 0 -1; -tt 0 1];
Fa = [0 11 5; 0 5 1; 0 1 7; 0 7 10; 0 10 11; 1 5 9; 5 11 4; 11 10 2; 10 7 6; 7 1 8; 3 9 4; 3 4 2; 3 2 6; 3 6 8; 3 8 9; 4 9 5; 2 4 11; 6 2 10; 8 6 7; 9 8 1] + 1;
Vt = Vt ./ vecnorm(Vt, 2, 2);
cache = containers.Map('KeyType', 'char', 'ValueType', 'double');  Gm = zeros(0, 3);
for f = 1:size(Fa, 1)
    a = Fa(f,1); b = Fa(f,2); c = Fa(f,3);  m = zeros(1, 3);
    for e = 1:3
        i = Fa(f, e); j = Fa(f, mod(e, 3) + 1);  key = sprintf('%d_%d', min(i, j), max(i, j));
        if ~isKey(cache, key), Vt(end+1, :) = (Vt(i, :) + Vt(j, :)) / 2;  Vt(end, :) = Vt(end, :) / norm(Vt(end, :));  cache(key) = size(Vt, 1); end
        m(e) = cache(key);
    end
    Gm = [Gm; a m(1) m(3); m(1) b m(2); m(3) m(2) c; m(1) m(2) m(3)]; %#ok<AGROW>
end
Fa = Gm;  n = size(Vt, 1);
L = sparse(n, n);  Mm = sparse(n, n);
for f = 1:size(Fa, 1)
    p = Fa(f, :);  P = Vt(p, :);  A = norm(cross(P(2,:) - P(1,:), P(3,:) - P(1,:))) / 2;
    for e = 1:3
        i = p(e); j = p(mod(e,3)+1); k = p(mod(e+1,3)+1);  u = Vt(i,:) - Vt(k,:);  v = Vt(j,:) - Vt(k,:);
        cw = dot(u, v) / norm(cross(u, v)) / 2;
        L(i,i) = L(i,i) + cw; L(j,j) = L(j,j) + cw; L(i,j) = L(i,j) - cw; L(j,i) = L(j,i) - cw;
    end
    Mm(p, p) = Mm(p, p) + A / 12 * (ones(3) + eye(3));
end
L = (L + L') / 2;  Mm = (Mm + Mm') / 2;
[Phi, D] = eig(full(L), full(Mm));  [lm, o] = sort(real(diag(D)));  Phi = Phi(:, o);  Phi = Phi ./ sqrt(sum(Phi .* (Mm * Phi), 1));  lm(abs(lm) < 1e-10) = 0;
basis = struct('Phi', Phi, 'Lambda', lm, 'Mass', Mm);
rng(9);  Fc = randn(n, 3) + 1i * randn(n, 3);
w('icoL', L);  w('icoM', Mm);  w('icoLambda', lm);  w('FcRe', real(Fc));  w('FcIm', imag(Fc));
for fam = {'mexhat', 'itersine', 'heat'}
    fr = rheome.filters.frame(fam{1}, [], lm, 'warn', false);
    Sc = rheome.filters.scatter(basis, Fc, fr, struct('Order', 2));
    fprintf(fid, 'scatter %s %d\n', fam{1}, size(Sc.Pairs, 1));
    w('Sigma', Sc.Sigma);  w('S0re', real(Sc.S0));  w('S0im', imag(Sc.S0));
    w('S1', squeeze(Sc.S1));  w('Pairs', Sc.Pairs);
    S2 = zeros(size(Sc.Pairs, 1), 3);  for q = 1:size(Sc.Pairs, 1), S2(q, :) = squeeze(Sc.S2(Sc.Pairs(q,1), Sc.Pairs(q,2), 1, :)).'; end
    w('S2', S2);
end
fprintf(fid, 'end\n');
fclose(fid);  disp('wrote kernels_golden.txt');

function i_write(fid, name, A)
    A = full(A);
    if isvector(A) && size(A,2) == 1 && ~strcmp(name, 'q0'), end
    fprintf(fid, '%s %d %d\n', name, size(A, 1), size(A, 2));
    fprintf(fid, '%.13g ', A.');  fprintf(fid, '\n');
end
