classdef ComputeWelchTest < SliceTestBase
%OPSWELCHTEST  nxr.compute.welch against the C++ goldens (scipy.signal.welch) and a plain fft reference; the wheel's test_welch.py.

    methods (Test)
        function psdEqualsScipyGoldens(tc)
            g = tc.readGolden('welch_golden.txt');
            fs = g.fs; x = g.x;
            check = @(r, name) tc.verifyLessThan(rowRel(r.psd, g.(name)), 1e-10, name);
            check(nxr.compute.welch.psd(x, fs, segment_samples=400, overlap=0.5), 'scipy_hann_const_n400');
            check(nxr.compute.welch.psd(x, fs, segment_samples=301, overlap=100/301, window="hamming", detrend="linear"), 'scipy_hamming_linear_n301_ov100');
            check(nxr.compute.welch.psd(x, fs, segment_samples=256, overlap=64/256, window="blackman", detrend="none", scaling="spectrum"), ...
                  'scipy_blackman_none_spectrum_n256_ov64');
        end

        function badSamplesLeaveSegmentsOut(tc)
            g = tc.readGolden('welch_golden.txt');
            r = nxr.compute.welch.psd(g.x, g.fs, segment_samples=400, overlap=0.5, good=g.good > 0.5);
            tc.verifyLessThan(rowRel(r.psd, g.scipy_hann_const_n400_bad), 1e-10);
            tc.verifyEqual(r.segments, g.scipy_hann_const_n400_bad_segments);
            % the wheel's case: 8 s at 1000 Hz, samples 1000..1009 (0-based) bad, segments of 256 every 128: those at 768 and 896 touch them
            [x, fs] = tc.signal();
            good = true(1, size(x, 2)); good(1001:1010) = false;
            r = nxr.compute.welch.psd(x, fs, segment_samples=256, overlap=0.5, good=good);
            full = nxr.compute.welch.psd(x, fs, segment_samples=256, overlap=0.5);
            tc.verifyEqual(r.segments_skipped, 2);
            tc.verifyEqual(r.segments, full.segments - 2);
        end

        function psdEqualsAnFftReference(tc)
            [x, fs] = tc.signal();
            r = nxr.compute.welch.psd(x, fs, segment_samples=256, overlap=0.5);
            C = referenceCross(x, fs, 256, 0.5);
            want = zeros(4, 129);
            for k = 1:129, want(:, k) = real(diag(C(:, :, k))); end
            tc.verifyEqual(r.psd, want, 'RelTol', 1e-9, 'AbsTol', 1e-18);
            tc.verifyEqual(r.frequencies, (0:128) * fs / 256, 'RelTol', 1e-12);
            tc.verifyEqual(r.segments, floor((size(x, 2) - 256) / 128) + 1);
            tc.verifyEqual(r.segments_skipped, 0);
        end

        function precisionFollowsTheInput(tc)
            [x, fs] = tc.signal();
            r64 = nxr.compute.welch.psd(x, fs);
            r32 = nxr.compute.welch.psd(single(x), fs);
            tc.verifyClass(r64.psd, 'double');
            tc.verifyClass(r32.psd, 'single');
            tc.verifyEqual(double(r32.psd), r64.psd, 'RelTol', 2e-4);
            tc.verifyClass(nxr.compute.welch.psd(round(x * 100), fs).psd, 'double');
            tc.verifyClass(nxr.compute.welch.psd(int32(round(x * 100)), fs).psd, 'double');   % an integer input is computed as double
        end

        function refusesWhatItCannotDo(tc)
            [x, fs] = tc.signal();
            tc.verifyError(@() nxr.compute.welch.psd(x, fs, window="triangle"), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.compute.welch.psd(x, fs, good=true(1, 10)), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.compute.welch.psd(x, fs, detrend="quadratic"), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.compute.welch.psd(x, fs, scaling="power"), 'nxr:invalidArgument');
        end

        function crossSpectrumEqualsAReference(tc)
            [x, fs] = tc.signal();
            x = x(1:3, :);
            bins = [1 1; 13 13; 31 41];                 % 1-based: the wheel's (0, 0), (12, 12), (30, 40)
            r = nxr.compute.welch.cross_spectrum(x, fs, bins, segment_samples=256, quadrature=true);
            C = referenceCross(x, fs, 256, 0.5);
            for g = 1:3
                want = mean(C(:, :, bins(g, 1):bins(g, 2)), 3);
                tc.verifyEqual(squeeze(r.real(g, :, :)), real(want), 'RelTol', 1e-9, 'AbsTol', 1e-18);
                tc.verifyEqual(squeeze(r.imag(g, :, :)), imag(want), 'RelTol', 1e-9, 'AbsTol', 1e-18);
            end
            tc.verifyEqual(size(r.real), [3 3 3]);
            tc.verifyEqual(r.imag, -permute(r.imag, [1 3 2]), 'AbsTol', 1e-18);   % antisymmetric
        end

        function crossSpectrumDiagonalIsThePsd(tc)
            [x, fs] = tc.signal();
            bins = [(1:129)' (1:129)'];
            r = nxr.compute.welch.cross_spectrum(x, fs, bins, segment_samples=256);
            p = nxr.compute.welch.psd(x, fs, segment_samples=256).psd;
            d = zeros(4, 129);
            for k = 1:129, d(:, k) = diag(squeeze(r.real(k, :, :))); end
            tc.verifyEqual(d, p, 'RelTol', 1e-10);
            tc.verifyEmpty(r.imag);
        end

        function crossSpectrumSumVersusMean(tc)
            [x, fs] = tc.signal();
            a = nxr.compute.welch.cross_spectrum(x, fs, [11 15], mean=true).real;
            b = nxr.compute.welch.cross_spectrum(x, fs, [11 15], mean=false).real;
            tc.verifyEqual(b, a * 5, 'RelTol', 1e-12);
            tc.verifyError(@() nxr.compute.welch.cross_spectrum(x, fs, [0 3]), 'nxr:invalidArgument');   % bins are 1-based
        end
    end
end

function e = rowRel(a, b)
    % max |a - b| / max |b| per channel (a row), the worst of them (the C++ tests' measure)
    if ~isequal(size(a), size(b)), e = Inf; return; end
    e = max(max(abs(a - b), [], 2) ./ max(abs(b), [], 2));
end

function C = referenceCross(x, fs, n, overlap)
    % Welch's cross-spectrum by plain fft: periodic Hann, constant detrend, density, one-sided; C(i, j, k) = X_i conj(X_j) at bin k
    hop = n - floor(n * overlap);
    w = 0.5 - 0.5 * cos(2 * pi * (0:n-1)' / n);
    nseg = floor((size(x, 2) - n) / hop) + 1;
    c = size(x, 1); nb = n / 2 + 1;
    acc = zeros(c, c, nb);
    for s = 0:nseg-1
        seg = x(:, s*hop + (1:n));
        seg = seg - mean(seg, 2);
        X = fft(seg .* w.', n, 2);
        X = X(:, 1:nb);
        for k = 1:nb, acc(:, :, k) = acc(:, :, k) + X(:, k) * X(:, k)'; end
    end
    C = acc / nseg / (fs * sum(w.^2));
    C(:, :, 2:end-1) = 2 * C(:, :, 2:end-1);
end
