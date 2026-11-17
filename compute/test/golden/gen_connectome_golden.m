function gen_connectome_golden(out)
% The structural connectome's vertex level against the maintainers' MATLAB reference, tess_connectome (Brainstorm fork), on SYNTHETIC data: the golden for test_connectome_golden.cpp.
% Run ONCE by a developer, never by a test:   matlab -batch "gen_connectome_golden('<file>')"      (BRAINSTORM3_HOME = the Brainstorm checkout; read only)
%
% What it builds is what test_connectome.cpp builds (compute/test/connectome_helpers.h, mesh_helpers.h): a sphere of radius 0.08 m (the icosphere at level 2: 162 vertices, 320
% triangles) and, for 0, 1 and 3 hops, a bundle of 240 fibers of 6 points (6 with both ends on one vertex, 8 repeated; ends drawn from the same portable generator, seed 100 + hops).
% tess_connectome is only CALLED, on the surface and the fibers (its "own fibers" path, the vertex resolution): nothing of Brainstorm is copied here. The reference reads a protocol
% database (bst_get, in_tess_bst, file_fullpath); this script answers those three calls from files of its own in a throwaway folder (first on the path, removed at the end), so no
% Brainstorm protocol is made and no setting of the user's Brainstorm changes. The gate is none: every fiber counts.
%
% Written (text: "name rows cols" and the values row by row, 17 significant digits; sorted triplets i j value, 1-based, the upper triangle of a symmetric matrix):
%   vertices, faces                      the sphere
%   ends_<h>      [240 x 6]              each fiber's first end then its last end (x y z, x y z): what the reference reads of the fibers
%   counts_<h>    triplets               the fiber counts: the reference's W at SmoothHops 0 and EdgeThr 0 (no smoothing, nothing pruned; the diagonal removed)
%   weights_<h>_<k>, keep_<h>_<k>        W (smoothed over h hops, entries below the fraction EdgeThr of the largest removed) and the largest component's vertices,
%                                        for EdgeThr = 0, 1e-4 and 0.05 (k = 0, 1, 2)
if nargin < 1, out = 'connectome_golden.txt'; end
home = getenv('BRAINSTORM3_HOME');
assert(~isempty(home), 'set BRAINSTORM3_HOME to the Brainstorm checkout');
addpath(genpath(fullfile(home, 'toolbox')));                       % Brainstorm, as a path only
shims = tempname; mkdir(shims);
cleanup = onCleanup(@() remove_shims(shims));
write_text(fullfile(shims, 'file_fullpath.m'), ['function f = file_fullpath(f)' newline 'end' newline]);
write_text(fullfile(shims, 'in_tess_bst.m'), ['function s = in_tess_bst(f)' newline 'm = load(f, ''Vertices'', ''Faces''); s = struct(''Vertices'', m.Vertices, ''Faces'', m.Faces, ''Atlas'', struct(''Name'', {}, ''Scouts'', {}));' newline 'end' newline]);
write_text(fullfile(shims, 'bst_get.m'), ['function s = bst_get(~, f)' newline 's = struct(''Surface'', struct(''SurfaceType'', ''Fibers'', ''FileName'', [f ''.fibers.mat''], ''Comment'', ''synthetic''));' newline 'end' newline]);
addpath(shims); rehash;                                              % first on the path: it answers bst_get, in_tess_bst and file_fullpath for the reference

[V, F] = icosphere(2, 0.08);
nV = size(V, 1);
surfFile = fullfile(shims, 'sphere.mat');
Vertices = V; Faces = F;
save(surfFile, 'Vertices', 'Faces');

fid = fopen(out, 'w');
put(fid, 'vertices', V);
put(fid, 'faces', F);
nF = 240; nP = 6;
for hops = [0 1 3]
    g = lcg(100 + hops);
    [Points, g] = make_bundle(g, nF, nP, 0.08, 6, 8);              %#ok<ASGLU>
    save([surfFile '.fibers.mat'], 'Points');
    ends = [squeeze(Points(:, 1, :)), squeeze(Points(:, nP, :))];
    put(fid, sprintf('ends_%d', hops), ends);
    % the fiber counts: no smoothing, nothing pruned
    R = tess_connectome(surfFile, 'Resolution', 'vertex', 'SmoothHops', 0, 'EdgeThr', 0);
    put(fid, sprintf('counts_%d', hops), triplets(R.W));
    for k = 0:2
        thr = [0 1e-4 0.05];
        R = tess_connectome(surfFile, 'Resolution', 'vertex', 'SmoothHops', hops, 'EdgeThr', thr(k + 1));
        put(fid, sprintf('weights_%d_%d', hops, k), triplets(R.W));
        put(fid, sprintf('keep_%d_%d', hops, k), R.keep(:));
    end
end
fclose(fid);
fprintf('wrote %s (%d vertices)\n', out, nV);
end

% ---- the portable generator of connectome_helpers.h (Lcg): s = s * 6364136223846793005 + 1442695040888963407 (mod 2^64); next = (s >> 11) / 2^53 ----
function g = lcg(seed)
g = struct('s', uint64(seed));
end
function [x, g] = next(g)
M32 = uint64(4294967295);
a = bitshift(uint64(1481765933), 32) + uint64(1284865837);
c = bitshift(uint64(335903614), 32) + uint64(4150755663);
s = g.s;
al = bitand(s, M32); ah = bitshift(s, -32); bl = bitand(a, M32); bh = bitshift(a, -32);
lo = al * bl;                                                       % below 2^64: exact
cross = bitand(ah * bl, M32) + bitand(al * bh, M32);
hi = bitshift(lo, -32) + cross;
s = bitor(bitshift(bitand(hi, M32), 32), bitand(lo, M32));          % the product mod 2^64
lo = bitand(s, M32) + bitand(c, M32);
hi = bitshift(s, -32) + bitshift(c, -32) + bitshift(lo, -32);
g.s = bitor(bitshift(bitand(hi, M32), 32), bitand(lo, M32));        % + the increment, mod 2^64
x = double(bitshift(g.s, -11)) / 9007199254740992;
end
function [x, g] = range(g, a, b)
[u, g] = next(g);
x = a + (b - a) * u;
end

% ---- connectome_helpers.h: unitRandom and makeBundle ----
function [p, g] = unit_random(g)
while true
    p = zeros(1, 3);
    for k = 1:3, [p(k), g] = range(g, -1, 1); end
    n = sqrt(p(1) * p(1) + p(2) * p(2) + p(3) * p(3));
    if n >= 0.1 && n <= 1, break; end
end
p = p / n;
end
function [P, g] = make_bundle(g, nF, nP, radius, self, dup)
P = zeros(nF, nP, 3);
for f = 1:nF
    [a, g] = unit_random(g);
    [c, g] = unit_random(g);
    bend = zeros(1, 3);
    for k = 1:3
        [r, g] = range(g, 0.97, 1.03); a(k) = a(k) * (radius * r);
        [r, g] = range(g, 0.97, 1.03); c(k) = c(k) * (radius * r);
        [r, g] = range(g, -1, 1); bend(k) = radius * 0.2 * r;
    end
    if f <= self
        for k = 1:3, [r, g] = range(g, -1, 1); c(k) = a(k) * (1 + 1e-6 * r); end
    end
    for p = 1:nP
        t = (p - 1) / (nP - 1); w = 4 * t * (1 - t);
        for k = 1:3, P(f, p, k) = (1 - t) * a(k) + t * c(k) + w * bend(k); end
    end
end
for d = 1:dup, P(nF - d + 1, :, :) = P(d, :, :); end              % the last `dup` fibers repeat fibers 1 .. dup
end

% ---- mesh_helpers.h: the icosphere ----
function [V, F] = icosphere(level, radius)
t = (1 + sqrt(5)) / 2;
V = [-1 t 0; 1 t 0; -1 -t 0; 1 -t 0; 0 -1 t; 0 1 t; 0 -1 -t; 0 1 -t; t 0 -1; t 0 1; -t 0 -1; -t 0 1];
F = [0 11 5; 0 5 1; 0 1 7; 0 7 10; 0 10 11; 1 5 9; 5 11 4; 11 10 2; 10 7 6; 7 1 8; 3 9 4; 3 4 2; 3 2 6; 3 6 8; 3 8 9; 4 9 5; 2 4 11; 6 2 10; 8 6 7; 9 8 1] + 1;
for i = 1:12, V(i, :) = V(i, :) / sqrt(V(i, 1) * V(i, 1) + V(i, 2) * V(i, 2) + V(i, 3) * V(i, 3)); end
for s = 1:level
    mid = containers.Map('KeyType', 'double', 'ValueType', 'double');
    G = zeros(4 * size(F, 1), 3);
    for f = 1:size(F, 1)
        a = F(f, 1); b = F(f, 2); c = F(f, 3);
        [ab, V, mid] = midpoint(V, mid, a, b); [bc, V, mid] = midpoint(V, mid, b, c); [ca, V, mid] = midpoint(V, mid, c, a);
        G(4 * f - 3:4 * f, :) = [a ab ca; ab b bc; ca bc c; ab bc ca];
    end
    F = G;
end
V = V * radius;
end
function [n, V, mid] = midpoint(V, mid, a, b)
key = min(a, b) * 1e6 + max(a, b);
if isKey(mid, key), n = mid(key); return; end
n = size(V, 1) + 1;
m = 0.5 * (V(a, :) + V(b, :));
V(n, :) = m / sqrt(m(1) * m(1) + m(2) * m(2) + m(3) * m(3));
mid(key) = n;
end

function remove_shims(folder)
rmpath(folder); rmdir(folder, 's');
end

% ---- output ----
function T = triplets(W)
[i, j, v] = find(triu(W, 1));                                       % a symmetric matrix with no diagonal: the upper triangle
[~, o] = sortrows([i j]);
T = [i(o) j(o) v(o)];
end
function put(fid, name, A)
fprintf(fid, '%s %d %d\n', name, size(A, 1), size(A, 2));
for r = 1:size(A, 1), fprintf(fid, '%.17g ', A(r, :)); fprintf(fid, '\n'); end
end
function write_text(file, text)
fid = fopen(file, 'w'); fwrite(fid, text); fclose(fid);
end
