function gen_vertex_normals_golden(out)
% BRAINSTORM'S VERTEX NORMALS (tess_normals of the release of 2018-10-12: the surfaces' VertNormals, which process_inverse_2016 reads as the fixed orientations) on a synthetic mesh,
% written to vertex_normals_golden.txt beside this file (or to `out`). Read by compute/test/test_vertex_normals_patch.cpp. Brainstorm is used as a path only, found through the environment
% variable BRAINSTORM_2018_HOME (a checkout at the last commit on or before 2018-10-12); its code is run, never copied.
% THE MESH: the convex hull of 162 points of a sphere with radii between 6 and 9 cm (uneven triangles, so the weighting of the faces shows), its faces in the order convhulln gives them.
% Run (one MATLAB batch run at a time):  BRAINSTORM_2018_HOME=<checkout> matlab -batch "gen_vertex_normals_golden"
bst = getenv('BRAINSTORM_2018_HOME');
if isempty(bst) || ~exist(fullfile(bst, 'toolbox', 'anatomy', 'tess_normals.m'), 'file')
  error('gen_vertex_normals_golden: set BRAINSTORM_2018_HOME to a checkout of Brainstorm at its release of 2018-10-12');
end
addpath(fullfile(bst, 'toolbox', 'anatomy'), fullfile(bst, 'toolbox', 'math'), fullfile(bst, 'toolbox', 'core'), fullfile(bst, 'toolbox', 'misc'));
here = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(out), out = fullfile(here, 'vertex_normals_golden.txt'); end
[st, commit] = system(['git -C "' bst '" rev-parse HEAD']);
if st ~= 0, commit = 'unknown'; end
n = 162; ga = pi * (3 - sqrt(5)); V = zeros(n, 3);
for i = 1:n
  z = 1 - 2 * (i - 0.5) / n; rho = sqrt(1 - z^2); phi = (i - 1) * ga;
  V(i, :) = (0.06 + 0.03 * mod((i - 1) * 0.618034, 1)) * [rho * cos(phi), rho * sin(phi), z];
end
F = convhulln(V);
N = tess_normals(V, F);
fid = fopen(out, 'w');
fprintf(fid, '# vertex normals golden: Brainstorm tess_normals, commit %s; MATLAB %s; see gen_vertex_normals_golden.m\n', strtrim(commit), version);
put(fid, 'V', V, '%.17g'); put(fid, 'F', F - 1, '%d'); put(fid, 'N', N, '%.12g');
fclose(fid);
fprintf('wrote %s\n', out);
end
function put(fid, name, A, fmt)
  fprintf(fid, '%s %d %d\n', name, size(A, 1), size(A, 2));
  fprintf(fid, [fmt ' '], A'); fprintf(fid, '\n');
end
