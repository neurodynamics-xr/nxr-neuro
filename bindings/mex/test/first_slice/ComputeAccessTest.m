classdef ComputeAccessTest < SliceTestBase
%COMPUTEACCESSTEST  The event, wavelet and filterbank functions through the MEX: a call, its shapes, names, 1-based spans and errors (the maths is the library's); the wheel's test_compute_access.py.

    methods (Test)
        function detectBurstsFindsThePlantedBurstWithOneBasedSpans(tc)
            t = 0:999;
            x = 1 + 0.1 * sin(2*pi*t/37);
            x(401:460) = 3;                                    % samples 400..459, 0-based, as the wheel's test
            r = nxr.compute.select.detect_bursts(x, 0.01);
            tc.verifyEqual(r.spans.start, 401);                % 1-based first sample
            tc.verifyEqual(r.spans.stop, 460);                 % the last sample, inclusive
            tc.verifyEqual(r.spans.peak, 401);
            tc.verifyEqual(r.spans.value, 3);
            tc.verifyEqual(r.skipped, '');
            tc.verifyEqual([r.good r.min_length r.gap], [1000 40 11]);
            tc.verifyEqual(r.median, nxr.compute.select.median(x), 'AbsTol', 1e-12);
            % crossings, merge and keep compose to the same burst
            s = nxr.compute.select.crossings(x, r.enter, r.exit);
            tc.verifyEqual([s.start s.stop], [401 460]);
            y = x; y(431:435) = 1;                              % a dip splits it
            split = nxr.compute.select.crossings(y, r.enter, r.exit);
            tc.verifyEqual(numel(split.start), 2);
            m = nxr.compute.select.merge_spans(split, 11);
            tc.verifyEqual([m.start m.stop], [401 460]);
            tc.verifyEmpty(nxr.compute.select.keep_spans(split, 40).start);
            tc.verifyEqual(numel(nxr.compute.select.keep_spans(m, 40, 60).start), 1);
        end

        function bandEnvelopeOfASineIsFlatInside(tc)
            fs = 256; t = (0:2047) / fs;
            x = sin(2*pi*10*t);
            e = nxr.compute.signal.band_envelope(x, fs, [8 12; 20 30]);
            tc.verifySize(e, [2 1 2048]);
            inner = squeeze(e(1, 1, 257:end-256));
            tc.verifyLessThan(std(inner) / mean(inner), 0.02);
            tc.verifyEqual(mean(inner), 1, 'RelTol', 0.1);
            z = nxr.compute.signal.band_analytic(x, fs, [8 12], shape="meyer");
            tc.verifySize(z, [1 1 2048]);
            tc.verifyFalse(isreal(z));
            tc.verifyClass(nxr.compute.signal.band_envelope(single(x), fs, [8 12]), 'single');
        end

        function jwtHasTheJointBanksShapeAndRoundTrips(tc)
            nt = 512; fs = 128; lam = linspace(0, 4, 8).';
            C = reshape(sin((1:8*nt) * 0.37), 8, nt);
            g = struct(family="itersine", lmax=4, num_filters=3);
            tb = struct(family="meyer", sides="analytic");
            r = nxr.compute.signal.jwt(C, fs, g, tb, eigenvalues=lam);
            ng = numel(nxr.compute.signal.graph_bank("itersine", lmax=4, num_filters=3).roles);
            nT = numel(nxr.compute.signal.time_bank(fs, nt, "meyer").centre_frequencies);
            tc.verifySize(r.coefficients, [ng*nT 8 nt]);
            tc.verifySize(r.index, [ng*nT 3]);
            tc.verifyEqual(r.index(1:2, :), [1 1 1; 2 1 1]);   % 1-based, g varies fastest
            x = nxr.compute.signal.ijwt(r.coefficients, fs, nt, g, tb, eigenvalues=lam);
            tc.verifyEqual(x, C, 'AbsTol', 1e-8);
            % Morlet x itersine on a two-sided bank
            tm = struct(family="morlet", sides="two_sided", centres=[4 8], cycles=6, gain=2);
            k = nxr.compute.signal.jwt(C + 1i * flipud(C), fs, g, tm, eigenvalues=lam);
            tc.verifySize(k.coefficients, [3*4 8 nt]);
        end

        function wtRoundTripsAndGraphBanksDescribeThemselves(tc)
            x = reshape(cos((1:3*1024) * 0.21) + sin((1:3*1024) * 1.3), 3, 1024);
            bank = struct(family="meyer");
            w = nxr.compute.signal.wt(x, 256, bank);
            tc.verifySize(w(:, :, :), [size(w, 1) 3 1024]);
            tc.verifyEqual(nxr.compute.signal.iwt(w, 256, 1024, bank), x, 'AbsTol', 1e-9);
            b = nxr.compute.signal.graph_bank("weyl", area=4*pi, levels=4);
            tc.verifyEqual(numel(b.roles), 5);
            tc.verifyTrue(all(isnan(b.frame_bounds)));
            tb = nxr.compute.signal.time_bank(256, 1024, "meyer");
            tc.verifySize(tb.gains, [numel(tb.centre_frequencies) 513]);
        end

        function namesTheLibraryRefusesAreErrors(tc)
            tc.verifyError(@() nxr.compute.signal.graph_bank("nonsense", lmax=1), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.compute.signal.time_bank(256, 256, "meyer", sides="sideways"), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.compute.signal.wt(zeros(1, 256), 256, struct(family="meyer", colour=3)), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.compute.select.risefall((1:300).', 100, 1, 2, model="posterior"), 'nxr:invalidArgument');
        end
    end
end
