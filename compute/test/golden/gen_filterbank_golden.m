% Writes filterbank_golden.txt: Rheome's @graphfilterbank + +graphtransform on a SYNTHETIC icosphere (level 1, 42 vertices), for
% test_filterbank.cpp. The operator is assembled here (cotan stiffness, Galerkin mass) and written with the results, so the C++ test
% needs no mesh code. Synthetic data only.
% Run:  RHEOME=<path to the rheome checkout> matlab -batch "run('compute/test/golden/gen_filterbank_golden.m')"   (one MATLAB at a time)
rh = getenv('RHEOME');  if isempty(rh), error('set RHEOME to the Rheome checkout'); end
addpath(rh);
here = fileparts(mfilename('fullpath'));
fid = fopen(fullfile(here, 'filterbank_golden.txt'), 'w');

% ── a unit icosphere, level 1 ──
t = (1 + sqrt(5)) / 2;
V = [-1 t 0; 1 t 0; -1 -t 0; 1 -t 0; 0 -1 t; 0 1 t; 0 -1 -t; 0 1 -t; t 0 -1; t 0 1; -t 0 -1; -t 0 1];
F = [0 11 5; 0 5 1; 0 1 7; 0 7 10; 0 10 11; 1 5 9; 5 11 4; 11 10 2; 10 7 6; 7 1 8; 3 9 4; 3 4 2; 3 2 6; 3 6 8; 3 8 9; 4 9 5; 2 4 11; 6 2 10; 8 6 7; 9 8 1] + 1;
V = V ./ vecnorm(V, 2, 2);
cache = containers.Map('KeyType', 'char', 'ValueType', 'double');
G = zeros(0, 3);
for f = 1:size(F, 1)
    a = F(f,1); b = F(f,2); c = F(f,3);
    m = zeros(1, 3);
    for e = 1:3
        i = F(f, e); j = F(f, mod(e, 3) + 1);
        key = sprintf('%d_%d', min(i, j), max(i, j));
        if ~isKey(cache, key)
            V(end+1, :) = (V(i, :) + V(j, :)) / 2;  V(end, :) = V(end, :) / norm(V(end, :));
            cache(key) = size(V, 1);
        end
        m(e) = cache(key);
    end
    G = [G; a m(1) m(3); m(1) b m(2); m(3) m(2) c; m(1) m(2) m(3)]; %#ok<AGROW>
end
F = G;  n = size(V, 1);

% ── cotan stiffness L and Galerkin mass M ──
L = sparse(n, n);  M = sparse(n, n);
for f = 1:size(F, 1)
    p = F(f, :);  P = V(p, :);
    A = norm(cross(P(2,:) - P(1,:), P(3,:) - P(1,:))) / 2;
    for e = 1:3
        i = p(e); j = p(mod(e,3)+1); k = p(mod(e+1,3)+1);          % edge ij, opposite vertex k
        u = V(i,:) - V(k,:);  w = V(j,:) - V(k,:);
        cw = dot(u, w) / norm(cross(u, w)) / 2;
        L(i,i) = L(i,i) + cw; L(j,j) = L(j,j) + cw; L(i,j) = L(i,j) - cw; L(j,i) = L(j,i) - cw;
    end
    M(p, p) = M(p, p) + A / 12 * (ones(3) + eye(3));
end
L = (L + L') / 2;  M = (M + M') / 2;

% ── its modes, mass-orthonormal ──
[Phi, D] = eig(full(L), full(M));  [lam, o] = sort(real(diag(D)));  Phi = Phi(:, o);
Phi = Phi ./ sqrt(sum(Phi .* (M * Phi), 1));
lam(abs(lam) < 1e-10) = 0;

rng(7);  X = randn(n, 2);
w = @(name, A) i_write(fid, name, A);
w('L', L); w('M', M); w('lambda', lam); w('X', X);

% ── the four banks, on the exact spectrum ──
cases = {
    'mexhat',      {'Wavelet', 'mexhat'}
    'mexhat-usable', {'Wavelet', 'mexhat', 'FineLimit', 'usable', 'VoicesPerOctave', 2}
    'heat',        {'Wavelet', 'heat'}
    'heat-6',      {'Wavelet', 'heat', 'NumFilters', 6}
    'itersine',    {'Wavelet', 'itersine'}
    'itersine-5',  {'Wavelet', 'itersine', 'NumFilters', 5}
    'logitersine', {'Wavelet', 'logitersine'}
    'logitersine-v2', {'Wavelet', 'logitersine', 'VoicesPerOctave', 2}
    'mexhat-limits', {'Wavelet', 'mexhat', 'ScaleLimits', [0.002 0.2]}
    'mexhat-lmax-lp10', {'Wavelet', 'mexhat', 'LPFactor', 10}      % built on lambda_max alone (no spectrum): lambda_min = lmax/10
    'heat-lmax',   {'Wavelet', 'heat'}
};
byLmax = {'mexhat-lmax-lp10', 'heat-lmax'};
lmaxC = 1.01 * max(lam);
Te = rheome.graphtransform.eigen(Phi, M, lam);
Tc = rheome.graphtransform.chebyshev(L, lmaxC, 'Order', 60, 'Mass', M);
for i = 1:size(cases, 1)
    nm = cases{i, 1};
    if any(strcmp(nm, byLmax)), gfb = rheome.graphfilterbank(max(lam), cases{i, 2}{:}); rng_ = 'lmax';
    else,                       gfb = rheome.graphfilterbank(lam, cases{i, 2}{:});       rng_ = 'spectrum';
    end
    fprintf(fid, 'bank %s %s %s\n', nm, gfb.Wavelet, rng_);
    fprintf(fid, 'opts %s\n', strjoin(cellfun(@i_opt, cases{i, 2}, 'UniformOutput', false), ' '));
    H = graphfilters(gfb);   tt = scales(gfb);
    b = framebounds(gfb);
    w('H', H);  w('t', tt(:)');  w('bounds', [b.A b.B b.Tightness b.Uncovered]);
    w('cwn', centerWavenumbers(gfb));  w('massLost', gfb.MassLost);
    % transforms act on the exact spectrum (the variants are checked on their gains and bounds only)
    isMain = ismember(nm, {'mexhat', 'heat', 'itersine', 'logitersine'});
    if isMain
        gfb.Transform = Te;
        W = wt(gfb, X);
        for m = 1:size(W, 3), w(sprintf('W%d', m), W(:, :, m)); end
        w('Xdual', iwt(gfb, W, 'dual'));
        w('Xtight', iwt(gfb, W, 'tight'));
        w('scaleEnergy', scaleSpectrum(gfb, X));  w('memberEnergy', vertexSpectrum(gfb, X));
        % chebyshev transform (analysis only: Rheome's iwt has no polynomial route)
        gfb.Transform = Tc;
        Wc = wt(gfb, X);
        for m = 1:size(Wc, 3), w(sprintf('Wc%d', m), Wc(:, :, m)); end
    end
    fprintf(fid, 'end\n');
end
fprintf(fid, 'cheb %d %.17g\n', 60, lmaxC);
fclose(fid);
disp('wrote filterbank_golden.txt');

function i_write(fid, name, A)
    A = full(A);
    fprintf(fid, '%s %d %d\n', name, size(A, 1), size(A, 2));
    fprintf(fid, '%.15g ', A.');  fprintf(fid, '\n');
end
function s = i_opt(x)
    if ischar(x), s = x; else, s = mat2str(x); end
end
