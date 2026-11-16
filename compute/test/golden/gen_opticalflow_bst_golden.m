function gen_opticalflow_bst_golden(out)
% Brainstorm's bst_opticalflow (Lefevre and Baillet 2008) on the same tiny perturbed icosphere and bump as gen_opticalflow_golden: the golden for the ambient / triangle flow
% (test_opticalflow_ambient.cpp). Brainstorm (GPLv3) is CALLED here, never read: its function is on the path and run directly (`brainstorm nogui` hangs).
% Run:  matlab -batch "gen_opticalflow_bst_golden('<file>')"   (one MATLAB run at a time)
addpath(fullfile(getenv('BRAINSTORM3_HOME'), 'toolbox', 'math'));
rng(5);
[V, F] = icosphere1();
V = V .* (1 + 0.05*randn(size(V,1),1));
nV = size(V,1);
c0 = [0.6 -0.2 0.77]; c0 = c0/norm(c0); dirv = [-0.25 0.4 0];
A = zeros(nV, 4);
for t = 1:4
  c = c0 + (t-1)*0.12*dirv; c = c/norm(c);
  d2 = sum((V./vecnorm(V,2,2) - c).^2, 2);
  A(:,t) = 0.2 + exp(-d2/0.35);
end
fn = cross(V(F(:,2),:) - V(F(:,1),:), V(F(:,3),:) - V(F(:,1),:), 2);      % area-weighted face normals (twice the area)
vn = zeros(nV, 3);
for k = 1:3, for d = 1:3, vn(:,d) = vn(:,d) + accumarray(F(:,k), fn(:,d), [nV 1]); end, end
vn = vn ./ vecnorm(vn, 2, 2);
assert(all(sum(vn .* V, 2) > 0), 'normals point inward');
FV = struct('Vertices', V, 'Faces', F, 'VertNormals', vn);
Time = 0:3;
% Brainstorm's GUI helpers (progress bar, ...) are not needed: any undefined helper it calls gets a no-op stub in a temporary folder (our own code, not Brainstorm's)
stubs = tempname; mkdir(stubs); addpath(stubs);
fid = fopen(out, 'w');
put(fid, 'positions', V); put(fid, 'faces', F-1); put(fid, 'A', A);
for alpha = [1 0.1]
  nout = 3;
  while true
    o = cell(1, nout);
    try, [o{1:nout}] = bst_opticalflow(A, FV, Time, 0, 3, alpha); break
    catch e
      tok = regexp(e.message, 'Undefined function ''(\w+)''', 'tokens', 'once');
      if ~isempty(tok), fid2 = fopen(fullfile(stubs, [tok{1} '.m']), 'w'); fprintf(fid2, 'function varargout = %s(varargin)\nvarargout = cell(1, max(nargout,1));\nend\n', tok{1}); fclose(fid2); rehash; fprintf('stub %s\n', tok{1}); continue; end
      fprintf('nargout %d: %s\n', nout, e.message); nout = nout - 1; if nout < 1, rethrow(e); end
    end
  end
  for k = 1:nout
    x = o{k}; fprintf('alpha %g out %d: %s %s\n', alpha, k, class(x), mat2str(size(x)));
    if isnumeric(x), put(fid, sprintf('bst_%g_%d', alpha, k), reshape(x, size(x,1), []));
    elseif isstruct(x), disp(x); end
  end
end
fclose(fid);
end
function put(fid, name, M)
  fprintf(fid, '%s %d %d\n', name, size(M,1), size(M,2));
  fprintf(fid, '%.17g ', M.');  fprintf(fid, '\n');
end
function [V, F] = icosphere1()
  t = (1+sqrt(5))/2;
  V = [-1 t 0; 1 t 0; -1 -t 0; 1 -t 0; 0 -1 t; 0 1 t; 0 -1 -t; 0 1 -t; t 0 -1; t 0 1; -t 0 -1; -t 0 1];
  F = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; 2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; 4 10 5; 4 5 3; 4 3 7; 4 7 9; 4 9 10; 5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
  V = V ./ vecnorm(V,2,2);
  M = containers.Map('KeyType','char','ValueType','double'); newF = zeros(0,3);
  for f = 1:size(F,1)
    a=F(f,1); b=F(f,2); c=F(f,3);
    ab=mid(a,b); bc=mid(b,c); ca=mid(c,a);
    newF = [newF; a ab ca; ab b bc; ca bc c; ab bc ca]; %#ok<AGROW>
  end
  F = newF;
  function k = mid(a, b)
    key = sprintf('%d_%d', min(a,b), max(a,b));
    if isKey(M, key), k = M(key); return; end
    p = (V(a,:)+V(b,:))/2; p = p/norm(p); V(end+1,:) = p; k = size(V,1); M(key) = k; %#ok<AGROW>
  end
end
