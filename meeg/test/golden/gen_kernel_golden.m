function gen_kernel_golden(work)
% Brainstorm's imaging kernels for the synthetic cases of gen_kernel_golden.py (work/cases.mat), written to work/bst_out.mat.
% Only Brainstorm's computational function is called (no database, no GUI): addpath of its toolbox, bst_inverse_linear_2018(HeadModel, OPTIONS).
% Run:  matlab -batch "gen_kernel_golden('<work>')"   
addpath(genpath(fullfile(getenv('BRAINSTORM3_HOME'), 'toolbox')));   % Brainstorm, as a path only
S = load(fullfile(work, 'cases.mat'));
names = {'full_megeeg', 'full_meg', 'deficient_megeeg', 'deficient_meg'};
variants = {'default', 'nodepth', 'diag', 'none'};
out = struct();
for i = 1:numel(names)
  c = S.(names{i});
  for v = 1:numel(variants)
    if v > 1 && ~strcmp(names{i}, 'full_megeeg'), continue; end
    G = c.G; N = size(G, 2) / 3;
    HM = struct('Gain', G, 'GridLoc', zeros(N, 3), 'GridOrient', [], 'HeadModelType', 'volume');
    O = bst_inverse_linear_2018();
    O.NoiseCovMat = struct('NoiseCov', c.C, 'nSamples', 1000 * ones(1, 1));
    O.ChannelTypes = cellstr(c.types(:));
    O.SourceOrient = {'free'};
    O.InverseMethod = 'minnorm'; O.InverseMeasure = 'amplitude';
    O.UseDepth = 1; O.WeightExp = 0.5; O.WeightLimit = 10;
    O.NoiseMethod = 'reg'; O.NoiseReg = 0.1;
    O.SnrMethod = 'fixed'; O.SnrFixed = 3;
    switch variants{v}
      case 'nodepth', O.UseDepth = 0;
      case 'diag', O.NoiseMethod = 'diag';
      case 'none', O.NoiseMethod = 'none';
    end
    [R, O2] = bst_inverse_linear_2018(HM, O);
    r = struct('K', full(R.ImagingKernel), 'W', full(R.Whitener));
    if isfield(R, 'Options') && isfield(R.Options, 'Lambda'), r.lambda = R.Options.Lambda; end
    out.([names{i} '__' variants{v}]) = r;
    fprintf('%s %s K %dx%d\n', names{i}, variants{v}, size(r.K, 1), size(r.K, 2));
  end
end
save(fullfile(work, 'bst_out.mat'), '-struct', 'out');
end
