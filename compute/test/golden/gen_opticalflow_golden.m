function gen_opticalflow_golden(out)
% Rheome's scalar optical flow on a tiny perturbed icosphere: the golden for nxr::signal (test_opticalflow.cpp).
% Calls Rheome (MIT-licensed code, on the path): rheome.dynamics.tangent_basis, opticalflow_scalar, flow_readout, rheome.flow.activation.
% Run:  matlab -batch "gen_opticalflow_golden('<file>')"   (one MATLAB run at a time)
addpath(getenv('RHEOME_HOME'));
rng(5);
[V, F] = icosphere1();
V = V .* (1 + 0.05*randn(size(V,1),1));                        % irregular, still a closed sphere-like surface
nV = size(V,1);
fg = rheome.operators.face_gradient(V, F);
vn = fg.W * fg.FaceNormal;  vn = vn ./ vecnorm(vn, 2, 2);
S = struct('Vertices', V, 'Faces', F, 'VertNormals', vn, 'nV', nV);
% a bump moving along +x/+y over 4 frames
c0 = [0.6 -0.2 0.77]; c0 = c0/norm(c0); dirv = [-0.25 0.4 0];
Aact = zeros(nV, 4);
for t = 1:4
  c = c0 + (t-1)*0.12*dirv; c = c/norm(c);
  d2 = sum((V./vecnorm(V,2,2) - c).^2, 2);
  Aact(:,t) = 0.2 + exp(-d2/0.35);
end
tb = rheome.dynamics.tangent_basis(S);
v1 = rheome.dynamics.opticalflow_scalar(Aact, S, struct('alpha', 1));
v2 = rheome.dynamics.opticalflow_scalar(Aact, S, struct('alpha', 0.1));
rd = rheome.dynamics.flow_readout(v1, S);
J = randn(3*nV, 12) .* repmat(1 + (1:12)/12, 3*nV, 1);
Jan = rheome.flow.activation(J, Method="analytic");
Jnm = rheome.flow.activation(J, Method="norm");
fid = fopen(out, 'w');
put(fid, 'positions', V); put(fid, 'faces', F-1); put(fid, 'normals', vn); put(fid, 'A', Aact);
put(fid, 'e1', tb.e1); put(fid, 'e2', tb.e2);
put(fid, 'v1', v1); put(fid, 'v2', v2);
put(fid, 'div', rd.divergence); put(fid, 'curl', rd.vorticity); put(fid, 'speed', rd.speed);
put(fid, 'J', J); put(fid, 'act_analytic', Jan); put(fid, 'act_norm', Jnm);
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
