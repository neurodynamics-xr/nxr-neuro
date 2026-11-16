% Golden for nxr::frames::canonicalFrames: Rheome's rheome.geom.sphereframe on a deterministic synthetic "brain" (a bumpy ellipsoid with a warped registered sphere).
% Run:  matlab -batch "addpath('<rheome>'); run('gen_frames_golden.m')"   (rheome on the path: +rheome/+geom/sphereframe.m)
n = 400; k = (0:n-1)' + 0.5; z = 1 - 2*k/n; phi = pi*(1 + sqrt(5))*k; r = sqrt(1 - z.^2);
u = [r.*cos(phi) r.*sin(phi) z];
Sph = u + 0.06*[sin(3*u(:,2)) cos(2*u(:,3)) sin(2*u(:,1))];  Sph = Sph ./ vecnorm(Sph,2,2);
V = (u .* [2.0 1.0 1.5]) .* (1 + 0.08*sin(4*u(:,1)).*cos(3*u(:,3)));
F = convhull(u);                                            % outward-oriented triangles on the sphere (same connectivity on the cortex)
c = mean(V); fn = cross(V(F(:,2),:)-V(F(:,1),:), V(F(:,3),:)-V(F(:,1),:), 2);
if mean(sum(fn .* (V(F(:,1),:)-c),2)) < 0, F = F(:,[1 3 2]); fn = -fn; end
N = zeros(n,3); for j = 1:3, for d = 1:3, N(:,d) = N(:,d) + accumarray(F(:,j), fn(:,d), [n 1]); end, end
N = N ./ vecnorm(N,2,2);
fid = fopen('frames_golden.txt','w');
w = @(name, A) fprintf(fid, '%s %d %d\n%s\n', name, size(A,1), size(A,2), strtrim(sprintf('%.17g ', A.')));
w('positions', V); w('faces', F-1); w('normals', N); w('sphere', Sph);
for ax = {[0 0 1], [1 0 0]}
    fr = rheome.geom.sphereframe(V, F, N, Sph, Axis=ax{1});
    w('e1', fr.e1); w('e2', fr.e2); w('colat', fr.colat); w('lon', fr.lon); w('condition', fr.condition);
    w('singular', double(fr.singular)); w('poles', [fr.north fr.south] - 1);
end
fclose(fid);
