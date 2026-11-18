function gen_kernel_projectors_golden(work)
% Brainstorm's imaging kernels for the case of gen_kernel_projectors_golden.py (work/cases.mat) WITH THE CHANNEL FILE'S PROJECTOR, written to work/bst_out.mat.
% Only Brainstorm's own functions are called (no database, no GUI): the projector's lines of process_inverse_2018 (process_ssp2 BuildProjector with status [1 2],
% Proj * Gain, Proj * NoiseCov * Proj'), then bst_inverse_linear_2018(HeadModel, OPTIONS) with Brainstorm's defaults.
% Run:  matlab -batch "gen_kernel_projectors_golden('<work>')"
addpath(genpath(fullfile(getenv('BRAINSTORM3_HOME'), 'toolbox')));   % Brainstorm, as a path only
S = load(fullfile(work, 'cases.mat'));
G = S.G; N = size(G, 2) / 3;
ChannelMat.Projector = struct('Comment', 'ssp', 'Components', S.u(:), 'CompMask', 1, 'Status', 1, 'SingVal', [], 'Method', 'SSP');
out = struct();
for k = {'raw', 'proj'}
  C = S.(['C_' k{1}]);
  Gain = G;
  % process_inverse_2018: apply the current SSP projectors
  Proj = process_ssp2('BuildProjector', ChannelMat.Projector, [1 2]);
  iGainSensors = find(sum(isnan(Gain), 2) == 0);
  Gain(iGainSensors,:) = Proj(iGainSensors,iGainSensors) * Gain(iGainSensors,:);
  C = Proj * C * Proj';
  HM = struct('Gain', Gain, 'GridLoc', zeros(N, 3), 'GridOrient', [], 'HeadModelType', 'volume');
  O = bst_inverse_linear_2018();
  O.NoiseCovMat = struct('NoiseCov', C, 'nSamples', 1000);
  O.ChannelTypes = cellstr(S.types(:));
  O.SourceOrient = {'free'};
  O.InverseMethod = 'minnorm'; O.InverseMeasure = 'amplitude';
  O.UseDepth = 1; O.WeightExp = 0.5; O.WeightLimit = 10;
  O.NoiseMethod = 'reg'; O.NoiseReg = 0.1;
  O.SnrMethod = 'fixed'; O.SnrFixed = 3;
  R = bst_inverse_linear_2018(HM, O);
  out.(['K_bst_' k{1}]) = full(R.ImagingKernel);
  fprintf('%s K %dx%d\n', k{1}, size(R.ImagingKernel, 1), size(R.ImagingKernel, 2));
end
save(fullfile(work, 'bst_out.mat'), '-struct', 'out');
end
