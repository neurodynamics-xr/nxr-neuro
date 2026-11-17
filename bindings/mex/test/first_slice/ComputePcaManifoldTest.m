classdef ComputePcaManifoldTest < SliceTestBase
%OPSPCAMANIFOLDTEST  nxr.compute.pca against the C++ golden (numpy's SVD) and the wheel's test_pca.py; nxr.compute.manifold.eigenmodes against the sphere's
%   spectrum (the wheel's test_manifold.py).

    methods (Test)
        function pcaEqualsTheCppGolden(tc)
            g = tc.readGolden('pca_golden.txt');
            for name = {'wide', 'tall', 'scaled'}
                p = name{1};
                scale = "none"; if g.([p '_unit_variance']) ~= 0, scale = "unit_variance"; end
                r = nxr.compute.pca.pca(g.([p '_X']), scale=scale);
                tc.verifyEqual(r.rank, g.([p '_rank']), p);
                tc.verifyEqual(r.components, g.([p '_components']), 'AbsTol', 1e-9, p);
                tc.verifyEqual(r.singular_values, g.([p '_S']), 'RelTol', 1e-10, p);
                tc.verifyEqual(r.explained_variance, g.([p '_ev']), 'RelTol', 1e-10, p);
                tc.verifyEqual(r.explained_variance_ratio, g.([p '_ratio']), 'RelTol', 1e-10, p);
                tc.verifyEqual(r.scores, g.([p '_scores']), 'AbsTol', 1e-8, p);
                tc.verifyEqual(r.mean, g.([p '_mean']), 'AbsTol', 1e-12, p);
                tc.verifyEqual(r.scale, g.([p '_scale']), 'RelTol', 1e-12, p);
                for k = g.([p '_ks'])
                    y = nxr.compute.pca.reconstruct(r.scores, r.components, r.mean, r.scale, k);
                    tc.verifyEqual(y, g.(sprintf('%s_recon%d', p, k)), 'AbsTol', 1e-8, p);
                end
            end
        end

        function pcaEqualsTheSvdWithTheSignConvention(tc)
            s = RandStream('twister', 'Seed', 3);
            for unit = [false true]
                X = randn(s, 40, 12) * randn(s, 12, 12) + 3;
                r = nxr.compute.pca.pca(X, scale=ifelse(unit, "unit_variance", "none"));
                mu = mean(X, 1);
                Xc = X - mu;
                sc = ones(1, 12); if unit, sc = std(Xc, 0, 1); end
                [U, S, V] = svd(Xc ./ sc, 'econ');
                Vt = V.'; S = diag(S).';
                [~, idx] = max(abs(Vt), [], 2);
                sign_ = sign(Vt(sub2ind(size(Vt), (1:size(Vt, 1))', idx)));
                comp = Vt .* sign_;
                tc.verifyEqual(r.rank, 12);
                tc.verifyEqual(r.components, comp, 'AbsTol', 1e-10);
                tc.verifyEqual(r.singular_values, S, 'RelTol', 1e-10);
                tc.verifyEqual(r.scores, (U .* S) .* sign_.', 'AbsTol', 1e-9);
                tc.verifyEqual(r.mean, mu, 'AbsTol', 1e-12);
                tc.verifyEqual(r.scale, sc, 'RelTol', 1e-12);
                tc.verifyEqual(r.explained_variance, S.^2 / 39, 'RelTol', 1e-10);
                tc.verifyEqual(sum(r.explained_variance_ratio), 1, 'AbsTol', 1e-12);
                tc.verifyEqual(r.components * r.components.', eye(12), 'AbsTol', 1e-10);
            end
        end

        function rankIsAtMostNMinusOne(tc)
            s = RandStream('twister', 'Seed', 4);
            tc.verifyEqual(nxr.compute.pca.pca(randn(s, 6, 20)).rank, 5);
        end

        function reconstructIsTheBestRankKApproximation(tc)
            s = RandStream('twister', 'Seed', 5);
            X = randn(s, 30, 8) + 5;
            r = nxr.compute.pca.pca(X);
            tc.verifyEqual(nxr.compute.pca.reconstruct(r.scores, r.components, r.mean, r.scale, r.rank), X, 'AbsTol', 1e-10);
            tc.verifyEqual(nxr.compute.pca.reconstruct(r.scores, r.components, r.mean, r.scale, 0), repmat(r.mean, 30, 1), 'AbsTol', 1e-12);
            err = norm(X - nxr.compute.pca.reconstruct(r.scores, r.components, r.mean, r.scale, 2), 'fro');
            tc.verifyEqual(err, sqrt(sum(r.singular_values(3:end).^2)), 'RelTol', 1e-9);   % the error is the discarded singular values
            tc.verifyError(@() nxr.compute.pca.reconstruct(r.scores, r.components, r.mean, r.scale, 99), 'nxr:invalidArgument');
        end

        function precisionAndRefusals(tc)
            s = RandStream('twister', 'Seed', 6);
            X = single(randn(s, 20, 5));
            r = nxr.compute.pca.pca(X);
            tc.verifyClass(r.components, 'single'); tc.verifyClass(r.scores, 'single'); tc.verifyClass(r.mean, 'single');
            bad = double(X); bad(3, 2) = NaN;
            tc.verifyError(@() nxr.compute.pca.pca(bad), 'nxr:invalidArgument');
            const = ones(10, 3); const(:, 1) = 1:10;
            tc.verifyError(@() nxr.compute.pca.pca(const, scale="unit_variance"), 'nxr:invalidArgument');   % a constant column
        end

        function sphereSpectrumIsLTimesLPlusOne(tc)
            [v, f] = icosphere(3);                          % 642 vertices, unit radius
            r = nxr.compute.manifold.eigenmodes(v, f, 25);
            want = [0, repmat(2, 1, 3), repmat(6, 1, 5), repmat(12, 1, 7), repmat(20, 1, 9)]';
            tc.verifyEqual(r.n_converged, 25);
            tc.verifyEqual(size(r.eigenvectors), [642 25]);
            tc.verifyEqual(size(r.eigenvalues), [25 1]);
            tc.verifyEqual(r.eigenvalues(2:16), want(2:16), 'RelTol', 0.03);   % l = 1..3
            % the last two shells (l = 4, 5): the eigensolver's random start can leave a mode of a degenerate shell out (seen on Linux: an l = 5 value
            % in place of the last l = 4 one), so each value must lie on a shell, not on a given one
            shells = [12 20 30];
            tc.verifyLessThan(max(min(abs(r.eigenvalues(17:end) ./ shells - 1), [], 2)), 0.03);
            tc.verifyLessThan(abs(r.eigenvalues(1)), 1e-6);
            tc.verifyGreaterThanOrEqual(min(diff(r.eigenvalues)), -1e-9);
        end

        function removeDcDropsTheConstantMode(tc)
            [v, f] = icosphere(2);
            r = nxr.compute.manifold.eigenmodes(v, f, 10, remove_dc=true, normalize=true);
            tc.verifyGreaterThan(r.eigenvalues(1), 1);
            tc.verifyEqual(size(r.eigenvectors), [162 9]);   % the constant mode is gone
        end

        function agreesWithTheKeptHandleCommands(tc)
            [v, f] = icosphere(2);
            r = nxr.compute.manifold.eigenmodes(v, f, 8);
            h = nxr_compute('create', v, f);
            c = onCleanup(@() nxr_compute('destroy', h));
            e = nxr_compute('solve', h, 8);
            tc.verifyEqual(r.eigenvalues, e.eigenvalues(:), 'AbsTol', 1e-8);
        end

        function meshErrorsAreValuesNotCrashes(tc)
            [v, f] = icosphere(1);
            bad = f; bad(f == 4) = 999;
            tc.verifyError(@() nxr.compute.manifold.eigenmodes(v, bad, 3), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.compute.manifold.eigenmodes(v(:, 1:2), f, 3), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.compute.manifold.eigenmodes(v, f - 1, 3), 'nxr:invalidArgument');   % 0-based indices are refused: faces are 1-based
            threw = false;
            try
                nxr.compute.manifold.eigenmodes(v, f, 0);     % the library's own refusal: an error with a code, not a crash
            catch e
                threw = true;
                tc.verifyTrue(startsWith(e.identifier, 'nxr:'));
            end
            tc.verifyTrue(threw);
        end
    end
end

function v = ifelse(c, a, b)
    if c, v = a; else, v = b; end
end

function [v, f] = icosphere(levels)
    t = (1 + sqrt(5)) / 2;
    v = [-1 t 0; 1 t 0; -1 -t 0; 1 -t 0; 0 -1 t; 0 1 t; 0 -1 -t; 0 1 -t; t 0 -1; t 0 1; -t 0 -1; -t 0 1];
    f = [1 12 6; 1 6 2; 1 2 8; 1 8 11; 1 11 12; 2 6 10; 6 12 5; 12 11 3; 11 8 7; 8 2 9; 4 10 5; 4 5 3; ...
         4 3 7; 4 7 9; 4 9 10; 5 10 6; 3 5 12; 7 3 11; 9 7 8; 10 9 2];
    v = v ./ vecnorm(v, 2, 2);
    for lev = 1:levels
        cache = containers.Map('KeyType', 'double', 'ValueType', 'double');
        faces = zeros(size(f, 1) * 4, 3);
        for i = 1:size(f, 1)
            a = f(i, 1); b = f(i, 2); c = f(i, 3);
            [v, cache, ab] = midpoint(v, cache, a, b);
            [v, cache, bc] = midpoint(v, cache, b, c);
            [v, cache, ca] = midpoint(v, cache, c, a);
            faces(4*i-3:4*i, :) = [a ab ca; b bc ab; c ca bc; ab bc ca];
        end
        f = faces;
    end
end

function [v, cache, m] = midpoint(v, cache, a, b)
    key = min(a, b) * 1e7 + max(a, b);
    if isKey(cache, key)
        m = cache(key);
    else
        p = v(a, :) + v(b, :);
        v(end+1, :) = p / norm(p);
        m = size(v, 1);
        cache(key) = m;
    end
end
