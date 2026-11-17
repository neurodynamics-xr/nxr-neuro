classdef SliceTestBase < matlab.unittest.TestCase
%SLICETESTBASE  The paths, the goldens and the synthetic signal the first slice's tests share.
%   The MEX is found under <repo>/build (or in NXR_MEX_DIR); the +nxr package in bindings/mex/matlab. The goldens are the C++ tests' (compute/test/golden,
%   io/formats/test/golden); the synthetic inputs and expected values are those of the Python wheel's tests (bindings/python/tests).

    properties (SetAccess = protected)
        repo
        computeGolden
        formatsTest
    end

    methods (TestClassSetup)
        function setUpPaths(tc)
            here = fileparts(mfilename('fullpath'));
            tc.repo = fullfile(here, '..', '..', '..', '..');
            tc.computeGolden = fullfile(tc.repo, 'compute', 'test', 'golden');
            tc.formatsTest = fullfile(tc.repo, 'io', 'formats', 'test');
            mexDir = getenv('NXR_MEX_DIR');
            if isempty(mexDir)
                hits = dir(fullfile(tc.repo, 'build', '**', ['nxr_compute.' mexext]));
                tc.assertNotEmpty(hits, ['nxr_compute.' mexext ' not found under <repo>/build (set NXR_MEX_DIR)']);
                mexDir = hits(1).folder;
            end
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(mexDir));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(tc.repo, 'bindings', 'mex', 'matlab')));
        end
    end

    methods
        function g = readGolden(tc, name)
            %READGOLDEN  A golden text file ("name rows cols" then rows*cols numbers, row after row) as a struct of matrices.
            txt = fileread(fullfile(tc.computeGolden, name));
            tok = strsplit(strtrim(txt));
            g = struct();
            i = 1;
            while i <= numel(tok)
                key = tok{i}; r = str2double(tok{i+1}); c = str2double(tok{i+2});
                vals = str2double(tok(i+3 : i+2+r*c));
                g.(key) = reshape(vals, c, r).';
                i = i + 3 + r*c;
            end
        end

        function j = readJson(tc, name)
            j = jsondecode(fileread(fullfile(tc.formatsTest, 'golden', [name '_golden.json'])));
        end

        function x = readF32(tc, name, nch, n)
            %READF32  A golden's float32 samples, channel after channel, as [channels x samples] single.
            fid = fopen(fullfile(tc.formatsTest, 'golden', name), 'r');
            c = onCleanup(@() fclose(fid));
            v = fread(fid, Inf, 'single=>single', 0, 'l');
            x = reshape(v, n, nch).';
        end

        function p = dataFile(tc, name)
            p = fullfile(tc.formatsTest, 'data', name);
        end

        function [x, fs] = signal(~)
            %SIGNAL  Four channels of coloured noise plus a 12 Hz and a 40 Hz sinusoid, 1000 Hz, 8 s (the wheel's fixture; MATLAB's own generator).
            fs = 1000; n = 8000; t = (0:n-1) / fs;
            s = RandStream('twister', 'Seed', 20261117);
            x = cumsum(randn(s, 4, n), 2) * 0.05 + randn(s, 4, n);
            x = x + 2.0 * sin(2*pi*12*t) + 1.0 * sin(2*pi*40*t + 0.3);
        end
    end
end
