classdef RunTest < SliceTestBase
%RUNTEST  The pipeline level: nxr.run, nxr.list and nxr.read on a synthetic dataset made in the test; the wheel's test_run.py.
%   nxr.run('psd', ...) stores what `nxr-neuro run psd` stores (the C++ test test_face runs both and compares them); here the stored PSD is compared with the pure function
%   nxr.compute.signal.psd on the stored samples, the stored recording's bad span left out as the pipeline leaves it out.

    properties
        ds
    end

    methods (TestMethodSetup)
        function makeDataset(tc)
            root = tempname;
            mkdir(root);
            tc.addTeardown(@() rmdir(root, 's'));
            tc.ds = nxr_compute('run.synthetic_dataset', root, 'demo', 'S1', 'rest', 3, 100, 40);
        end
    end

    methods (Test)
        function theTimeRangeIsRequired(tc)
            tc.verifyError(@() nxr.run('psd', tc.ds, 'S1', 'recording', 'rest'), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.run('psd', tc.ds, 'S1', 'recording', 'rest', 'from', 0), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.run('psd', tc.ds, 'S1', 'recording', 'rest', 'from', 0, 'to', 10, 'all', true), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.run('psd', tc.ds, 'S1', 'recording', 'rest', 'all', true, 'bogus', 1), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.run('psd', tc.ds, 'S1', 'all', true), 'nxr:invalidArgument');
        end

        function psdIsThePureFunctionsResult(tc)
            r = nxr.run('psd', tc.ds, 'S1', 'recording', 'rest', 'set', 'scipy', 'segment_seconds', 2, 'all', true, 'progress', @(s) assignin('base', 'nxrRunProgress', s));
            tc.verifyEqual(r.cache, 'computed');
            tc.verifyEqual({r.stored.name}, {'rest_psd_welch'});
            tc.verifyTrue(evalin('base', 'isstruct(nxrRunProgress)'));
            evalin('base', 'clear nxrRunProgress');
            got = nxr.read(tc.ds, 'S1', 'rest_psd_welch');
            tc.verifyEqual(got.meta.class, 'statistic');
            tc.verifyEqual(got.meta.measure, 'power_density');
            tc.verifyEqual(size(got.values), [3 101]);
            tc.verifyClass(got.values, 'double');
            x = nxr.read(tc.ds, 'S1', 'rest').values;
            tc.verifyEqual(size(x), [3 4000]);
            tc.verifyClass(x, 'single');
            t = (0:size(x, 2) - 1) / 100;
            good = ~(t >= 0.3 * 40 & t < 0.375 * 40);
            want = nxr.compute.signal.psd(x, 100, segment_samples=200, good=good);   % scipy's set: hann (periodic), constant detrend, 50 % overlap, density
            tc.verifyGreaterThan(want.segments_skipped, 0);
            tc.verifyLessThan(max(abs(got.values ./ want.psd - 1), [], 'all'), 1e-6);
            r2 = nxr.run('psd', tc.ds, 'S1', 'recording', 'rest', 'set', 'scipy', 'segment_seconds', 2, 'all', true);
            tc.verifyEqual(r2.cache, 'hit');
        end

        function aRangeIsARange(tc)
            a = nxr.run('psd', tc.ds, 'S1', 'recording', 'rest', 'segment_seconds', 2, 'from', 0, 'to', 20);
            tc.verifyEqual(a.cache, 'computed');
            s = nxr.read(tc.ds, 'S1', a.stored(1).name);
            tc.verifyEqual(s.meta.span.stop_sample, 2048);   % 20 s = 2000 samples, snapped to the 2^k-sample tile boundary of the default set
        end

        function theLists(tc)
            p = nxr.list.pipelines();
            names = {p.name};
            tc.verifyTrue(all(ismember({'psd', 'global_bursts'}, names)));
            tc.verifyEqual(p(strcmp(names, 'psd')).form, 'built-in');
            tc.verifyNotEmpty(strfind(nxr.list.help('psd'), 'segment_seconds'));
            d = nxr.list.datasets(fileparts(tc.ds));
            tc.verifyEqual({d.name}, {'demo'});
            s = nxr.list.subjects(tc.ds);
            tc.verifyEqual({s.name}, {'S1'});
            tc.verifyError(@() nxr.list.subjects(tc.ds, where="age > 3"), 'nxr:invalidArgument');
            nxr.run('psd', tc.ds, 'S1', 'recording', 'rest', 'all', true);
            r = nxr.list.results(tc.ds, 'S1');
            tc.verifyEqual({r.name}, {'rest_psd_welch'});
            rec = nxr.list.results(tc.ds, 'S1', kind="recordings");
            tc.verifyEqual({rec.name}, {'rest'});
        end

        function readRefusesWhatIsNotStored(tc)
            tc.verifyError(@() nxr.read(tc.ds, 'S1', 'nothing_of_that_name'), 'nxr:noSuchRow');
        end
    end
end
