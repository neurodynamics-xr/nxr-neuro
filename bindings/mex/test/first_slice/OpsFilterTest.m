classdef OpsFilterTest < SliceTestBase
%OPSFILTERTEST  nxr.ops.filter against the Brainstorm goldens of the C++ tests and the wheel's tone tests (test_filter.py).

    properties (Constant)
        FS = 1000
    end

    methods (Test)
        function bandpassPassesTheBandAndStopsTheRest(tc)
            x = tones([3 15 80]);
            g = gain(nxr.ops.filter.bandpass(x, 10, 20, tc.FS), x);
            tc.verifyEqual(g(2), 1, 'AbsTol', 0.02);
            tc.verifyLessThan(g(1), 1e-2);
            tc.verifyLessThan(g(3), 1e-2);
        end

        function lowpassAndHighpassSplitTheSpectrum(tc)
            x = tones([5 100]);
            lo = gain(nxr.ops.filter.lowpass(x, 30, tc.FS), x);
            hi = gain(nxr.ops.filter.highpass(x, 30, tc.FS), x);
            tc.verifyEqual(lo(1), 1, 'AbsTol', 0.02); tc.verifyLessThan(lo(2), 1e-2);
            tc.verifyEqual(hi(2), 1, 'AbsTol', 0.02); tc.verifyLessThan(hi(1), 1e-2);
        end

        function notchRemovesTheLineAndItsHarmonic(tc)
            x = tones([10 50 100]);
            g = gain(nxr.ops.filter.notch(x, [50 100], tc.FS), x);
            tc.verifyGreaterThan(g(1), 0.98); tc.verifyLessThan(g(2), 0.05); tc.verifyLessThan(g(3), 0.05);
        end

        function bandstopRemovesTheBand(tc)
            x = tones([5 45 120]);
            g = gain(nxr.ops.filter.bandstop(x, 40, 50, tc.FS), x);
            tc.verifyGreaterThan(g(1), 0.98); tc.verifyLessThan(g(2), 0.05); tc.verifyGreaterThan(g(3), 0.98);
        end

        function zeroPhaseKeepsShapeAndPrecision(tc)
            [x, fs] = tc.signal();
            y = nxr.ops.filter.bandpass(x, 8, 16, fs);
            tc.verifyEqual(size(y), size(x)); tc.verifyClass(y, 'double');
            y32 = nxr.ops.filter.bandpass(single(x), 8, 16, fs);
            tc.verifyClass(y32, 'single');
            tc.verifyEqual(double(y32), y, 'AbsTol', 2e-4 * max(abs(y(:))));
            pulse = zeros(1, 4001); pulse(2001) = 1;   % a symmetric pulse stays symmetric about its centre
            out = nxr.ops.filter.lowpass(pulse, 40, tc.FS, remove_mean=false);
            tc.verifyEqual(out(2001-300:2000), out(2001+300:-1:2002), 'AbsTol', 1e-12);
        end

        function optionsAreNamedAndChecked(tc)
            x = tones([5 50]);
            tc.verifyEqual(size(nxr.ops.filter.bandstop(x, 40, 60, tc.FS, method="iir")), size(x));
            tc.verifyEqual(size(nxr.ops.filter.lowpass(x, 30, tc.FS, edge="zeros")), size(x));
            tc.verifyError(@() nxr.ops.filter.lowpass(x, 30, tc.FS, method="elliptic"), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.ops.filter.bandpass(x, 10, 20, tc.FS, method="iir"), 'nxr:invalidArgument');   % the IIR designs are the notch and the band-stop
            tc.verifyError(@() nxr.ops.filter.bandpass(x, 20, 10, tc.FS), 'nxr:invalidArgument');                 % lower edge above the upper
            tc.verifyError(@() nxr.ops.filter.lowpass(x, 600, tc.FS), 'nxr:invalidArgument');                     % above Nyquist
        end

        function firEqualsBrainstormGoldens(tc)
            g = tc.readGolden('filter_bst_golden.txt');
            fs = g.fs; x = g.x; scale = max(abs(x(:)));
            cases = {'lp30', 'lowpass', 0, 30, false; 'hp5', 'highpass', 5, 0, false; 'bp1_20', 'bandpass', 1, 20, false; ...
                     'bp8_40', 'bandpass', 8, 40, false; 'lp5', 'lowpass', 0, 5, false; 'hp0p5', 'highpass', 0.5, 0, false; ...
                     'bp1_20_relax', 'bandpass', 1, 20, true};
            for i = 1:size(cases, 1)
                [name, kind, lo, hi, relax] = cases{i, :};
                K = g.([name '_transient']);               % Brainstorm's K stated: the record's ends are then compared exactly
                ripple = 1e-3 + relax * (1e-2 - 1e-3);
                switch kind
                    case 'lowpass',  y = nxr.ops.filter.lowpass(x, hi, fs, transient_samples=K, ripple=ripple);
                    case 'highpass', y = nxr.ops.filter.highpass(x, lo, fs, transient_samples=K, ripple=ripple);
                    otherwise,       y = nxr.ops.filter.bandpass(x, lo, hi, fs, transient_samples=K, ripple=ripple);
                end
                tc.verifyLessThan(max(abs(y(:) - g.(name)(:))), 1e-11 * scale, name);
            end
        end

        function iirEqualsBrainstormGoldens(tc)
            g = tc.readGolden('filter_bst_golden.txt');
            fs = g.fs2; x = g.x2; scale = max(abs(x(:)));
            tc.verifyLessThan(max(abs(nxr.ops.filter.notch(x, [50 100], fs) - g.notch50_100_bw1), [], 'all'), 1e-11 * scale);
            tc.verifyLessThan(max(abs(nxr.ops.filter.notch(x, 50, fs, notch_bandwidth=2.5) - g.notch50_bw2p5), [], 'all'), 1e-11 * scale);
            tc.verifyLessThan(max(abs(nxr.ops.filter.bandstop(x, 49.25, 50.75, fs) - g.bandstop50_w1p5), [], 'all'), 1e-7 * scale);
        end
    end
end

function x = tones(freqs)
    n = 8000; t = (0:n-1) / 1000;
    x = zeros(numel(freqs), n);
    for i = 1:numel(freqs), x(i, :) = sin(2 * pi * freqs(i) * t); end
end

function g = gain(y, x)
    lo = 1501; hi = size(x, 2) - 1500;   % amplitude ratio of output to input away from the ends
    g = sqrt(mean(y(:, lo:hi).^2, 2) ./ mean(x(:, lo:hi).^2, 2));
end
