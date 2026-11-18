classdef IoTest < SliceTestBase
%IOTEST  nxr.io against the repository's synthetic files and their goldens (MNE-Python, nibabel, pandas); the wheel's test_io.py.

    methods (Test)
        function edfBrainvisionEeglabEqualTheMneGoldens(tc)
            for fmt = {'edf', 'brainvision', 'eeglab'}
                golden = tc.readJson(fmt{1});
                for name = fieldnames(golden.cases)'
                    c = golden.cases.(name{1});
                    if ~isfield(c, 'data') || (isfield(c, 'gaps') && ~isempty(c.gaps)), continue; end
                    switch fmt{1}
                        case 'edf', rec = nxr.io.open_edf(tc.dataFile(c.file));
                        case 'brainvision', rec = nxr.io.open_brainvision(tc.dataFile(c.file));
                        otherwise, rec = nxr.io.open_eeglab(tc.dataFile(c.file));
                    end
                    tc.checkRecording(rec, c, [fmt{1} '/' name{1}]);
                end
            end
        end

        function fifEqualsTheMneGoldens(tc)
            golden = tc.readJson('fif');
            seen = 0;
            for name = fieldnames(golden.cases)'
                c = golden.cases.(name{1});
                if ~endsWith(c.data, '.f32') || (isfield(c, 'data_has_projector') && c.data_has_projector), continue; end
                rec = nxr.io.open_fif(tc.dataFile(c.file));
                h = rec.header;
                tc.verifyEqual(h.format, 'fif'); tc.verifyEqual(h.precision, 'single');
                tc.verifyEqual(h.sampling_rate, c.sfreq, 'RelTol', 1e-12);
                tc.verifyEqual(h.n_samples, c.n_times);
                names = {h.channels.name};
                tc.verifyEqual(names(:), c.info.ch_names(:), name{1});
                tc.verifyEqual(sort(asCellstr(h.bad_channels)), sort(asCellstr(c.info.bads)), name{1});
                x = rec.read_all();
                tc.verifyClass(x, 'single');
                tc.verifyEqual(x, tc.readF32(c.data, numel(h.channels), c.n_times), 'RelTol', 2e-6, 'AbsTol', 1e-30, name{1});
                a = floor(c.n_times / 3) + 1; b = floor(2 * c.n_times / 3);
                tc.verifyEqual(rec.read(a, b), x(:, a:b));
                if isempty(c.gaps), tc.verifyEmpty(h.gaps); else, tc.verifyEqual(h.gaps, [[c.gaps.first]' + 1, [c.gaps.count]'], name{1}); end
                seen = seen + 1;
            end
            tc.verifyGreaterThanOrEqual(seen, 6);
        end

        function fifFloat64AndRefusals(tc)
            tc.verifyClass(nxr.io.open_fif(tc.dataFile('fif_double_raw.fif')).read_all(), 'double');
            tc.verifyEqual(nxr.io.open_fif(tc.dataFile('fif_basic_raw.fif')).header.info.first_samp, '250');
            tc.verifyError(@() nxr.io.open_fif(tc.dataFile('fif_refuse_epo.fif')), 'nxr:unsupported');
            tc.verifyError(@() nxr.io.open_fif(tc.dataFile('fif_complex_raw.fif')).read(1, 10), 'nxr:unsupported');   % complex buffers: refused by the real read
            tc.verifyError(@() nxr.io.open_fif(tc.dataFile('fif_refuse_notfif.fif')), 'nxr:formatError');
            [x, header] = nxr.io.read_fif(tc.dataFile('fif_basic_raw.fif'), first=11, last=30);
            tc.verifyEqual(size(x), [10 20]);
            tc.verifyEqual(header.format, 'fif');
        end

        function edfGapsReadAsZeros(tc)
            c = tc.readJson('edf').cases.plus_d;
            rec = nxr.io.open_edf(tc.dataFile(c.file));
            tc.verifyEqual(rec.header.gaps, [c.gaps.first + 1, c.gaps.count]);
            x = rec.read_all();
            tc.verifyFalse(any(x(:, c.gaps.first + 1 : c.gaps.first + c.gaps.count), 'all'));
        end

        function readHelpersStateTheSpan(tc)
            path = tc.dataFile('edf_plain.edf');
            [x, header] = nxr.io.read_edf(path, first=101, last=200);
            tc.verifyEqual(size(x, 2), 100);
            tc.verifyTrue(startsWith(header.format, 'edf'));
            tc.verifyEqual(size(nxr.io.read_edf(path, all=true), 2), header.n_samples);
            tc.verifyError(@() nxr.io.read_edf(path), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.io.read_edf(path, first=1, last=10, all=true), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.io.read_edf(path, first=1, last=1e9), 'nxr:invalidArgument');
            tc.verifyError(@() nxr.io.read_edf(path, first=0, last=10), 'nxr:invalidArgument');   % samples are 1-based
        end

        function refusalsAreErrorsWithCodes(tc)
            d = @(n) tc.dataFile(n);
            tc.verifyError(@() nxr.io.open_edf(d('edf_bad_version.edf')), 'nxr:formatError');
            tc.verifyError(@() nxr.io.open_edf(d('edf_mixed_rates.edf')), 'nxr:unsupported');
            tc.verifyError(@() nxr.io.open_brainvision(d('bv_ascii.vhdr')), 'nxr:unsupported');
            tc.verifyError(@() nxr.io.open_edf(d('no_such_file.edf')), 'nxr:io');
            tc.verifyEqual(nxr.io.open_edf(d('edf_mixed_rates.edf'), sampling_rate=100).header.sampling_rate, 100);   % a rate named lets a mixed-rate file be read
        end

        function precisionFollowsTheFile(tc)
            tc.verifyClass(nxr.io.open_eeglab(tc.dataFile('eeglab_fdt.set')).read_all(), 'single');
            tc.verifyClass(nxr.io.open_edf(tc.dataFile('edf_plain.edf')).read_all(), 'single');
        end

        function closedRecordingIsAnError(tc)
            rec = nxr.io.open_edf(tc.dataFile('edf_plain.edf'));
            rec.close();
            tc.verifyError(@() rec.read(1, 10), ?MException);
        end

        function niftiEqualsTheNibabelGoldens(tc)
            golden = tc.readJson('nifti');
            for name = fieldnames(golden.cases)'
                c = golden.cases.(name{1});
                if ~isfield(c, 'data'), continue; end
                [data, h] = nxr.io.read_nifti(tc.dataFile(c.file));
                tc.verifyEqual(size(data), c.shape(:)', name{1});
                tc.verifyEqual(double(data(:)), c.data(:), 'RelTol', 1e-6, 'AbsTol', 1e-6, name{1});   % the file's x axis first, as the golden lists them
                A = h.affine.';                                                                       % the golden's 3 x 4 row-major
                tc.verifyEqual(A(:)', [c.best_affine(:)', 0 0 0 1], 'RelTol', 1e-6, 'AbsTol', 1e-9, name{1});
                tc.verifyEqual(h.version, c.version); tc.verifyEqual(h.descrip, c.descrip);
            end
            [data, h] = nxr.io.read_nifti(tc.dataFile('nifti_pair.hdr'));
            tc.verifyEqual(ndims(data), 3); tc.verifyEqual(h.version, 1);
            tc.verifyError(@() nxr.io.read_nifti(tc.dataFile('nifti_complex64.nii')), 'nxr:unsupported');
            tc.verifyError(@() nxr.io.read_nifti(tc.dataFile('not_nifti.nii')), 'nxr:formatError');
        end

        function giftiEqualsTheNibabelGoldens(tc)
            golden = tc.readJson('gifti');
            for name = fieldnames(golden.cases)'
                c = golden.cases.(name{1});
                img = nxr.io.read_gifti(tc.dataFile(c.file));
                wantArrays = c.arrays; if isstruct(wantArrays), wantArrays = num2cell(wantArrays); end
                tc.verifyEqual(numel(img.arrays), numel(wantArrays), name{1});
                for i = 1:numel(wantArrays)
                    want = wantArrays{i}; got = img.arrays(i);
                    tc.verifyEqual(got.intent, want.intent); tc.verifyEqual(got.datatype, want.datatype);
                    dims = want.dims(:)'; if isscalar(dims), dims = [dims 1]; end
                    tc.verifyEqual(size(got.data), dims, name{1});
                    % the golden lists the values in the file's row-major order; a triangle's indices are 1-based here
                    vals = permute(got.data, ndims(got.data):-1:1);
                    if strcmp(want.intent, 'NIFTI_INTENT_TRIANGLE'), vals = vals - 1; end
                    tc.verifyEqual(double(vals(:)), want.data(:), 'RelTol', 1e-6, 'AbsTol', 1e-9, name{1});
                end
            end
            img = nxr.io.read_gifti(tc.dataFile('gifti_surf_ascii.surf.gii'));
            tc.verifyClass(img.arrays(1).data, 'single');
            tc.verifyClass(img.arrays(2).data, 'int32');
            % a mesh read from a file goes straight into the eigenmodes
            r = nxr.compute.operators.eigen(double(img.arrays(1).data), img.arrays(2).data, 2);
            tc.verifyEqual(r.n_converged, 2);
        end

        function tsvEqualsThePandasGolden(tc)
            golden = tc.readJson('tsv');
            for name = {'lf', 'crlf', 'bom'}
                want = golden.(name{1});
                [t, names] = nxr.io.read_tsv(tc.dataFile(['tsv_' name{1} '.tsv']));
                tc.verifyEqual(names(:), want.columns(:));
                rows = cellfun(@(r) r(:).', want.rows, 'UniformOutput', false);
                rows = vertcat(rows{:});
                for j = 1:numel(want.columns)
                    tc.verifyEqual(t.(matlab.lang.makeValidName(want.columns{j}))(:), rows(:, j), name{1});
                end
            end
            tc.verifyError(@() nxr.io.read_tsv(tc.dataFile('tsv_ragged.tsv')), 'nxr:formatError');
        end
    end

    methods
        function checkRecording(tc, rec, c, label)
            h = rec.header;
            nch = numel(c.mne_channels);
            if isfield(c, 'n_times'), n = c.n_times; else, n = c.n_samples; end
            names = {h.channels.name};
            tc.verifyEqual(names(:), c.mne_channels(:), label);
            tc.verifyEqual(h.sampling_rate, c.sfreq, 'RelTol', 1e-12);
            tc.verifyEqual(h.n_samples, n, label);
            want = tc.readF32(c.data, nch, n);
            x = rec.read_all();
            tc.verifyEqual(size(x), [nch n], label);
            tc.verifyEqual(double(x), double(want), 'RelTol', 2e-6, 'AbsTol', 1e-12, label);   % a file that holds doubles reads as double
            % a block read does not depend on the block, nor on the channels asked
            a = floor(n / 3) + 1; b = floor(2 * n / 3);
            tc.verifyEqual(rec.read(a, b), x(:, a:b), label);
            tc.verifyEqual(rec.read(a, b, first_channel=2, last_channel=nch), x(2:end, a:b), label);
            % BrainVision's own "New Segment" marker is a marker like any other here; MNE keeps it out of the annotations
            ann = h.annotations;
            keep = true(1, numel(ann));
            for i = 1:numel(ann), keep(i) = ~startsWith(ann(i).label, 'New Segment'); end
            ann = ann(keep);
            wantAnn = c.annotations;
            if isstruct(wantAnn), wantAnn = num2cell(wantAnn); end
            tc.verifyEqual(numel(ann), numel(wantAnn), label);
            for i = 1:numel(wantAnn)
                tc.verifyEqual(round(ann(i).onset, 6), round(wantAnn{i}.onset, 6), label);
                tc.verifyEqual(round(ann(i).duration, 6), round(wantAnn{i}.duration, 6), label);
                tc.verifyEqual(ann(i).label, wantAnn{i}.label, label);
            end
        end
    end
end

function c = asCellstr(v)
    % a JSON list of names (or none: []) and the header's cell array as the same column of text
    if isempty(v), c = cell(0, 1); elseif ischar(v), c = {v}; else, c = v(:); end
end
