function gen_forward_golden(work)
% Brainstorm's sphere forward models for the case of gen_forward_golden.py (work/cases.mat), written to work/bst_out.mat.
% Only Brainstorm's computational functions are called (no database, no GUI): bst_os, bst_meg_sph, bst_eeg_sph, bst_berg.
% Run:  matlab -batch "gen_forward_golden('<work>')"   
addpath(genpath(fullfile(getenv('BRAINSTORM3_HOME'), 'toolbox')));
addpath(fullfile(getenv('BRAINSTORM3_HOME'), 'external', 'mosher'));   % helpers of bst_os
S = load(fullfile(work, 'cases.mat'));
ch = S.chs(:)';
out = struct();
Param = struct('Center', num2cell(repmat(S.c0(:), 1, numel(ch)), 1));
out.meg_single = bst_meg_sph(S.src', ch, Param);
out.meg_single_out = bst_meg_sph(S.src_out', ch, Param);
Sp = bst_os(ch, S.inner, S.F);                 % overlapping spheres on the inner skull
out.os_centre = cell2mat(arrayfun(@(s) s.Center(:)', Sp(:), 'UniformOutput', false));
out.os_radius = [Sp.Radius]';
Sc = bst_os(ch, S.scalp, S.F);
out.os_scalp_centre = cell2mat(arrayfun(@(s) s.Center(:)', Sc(:), 'UniformOutput', false));
out.os_scalp_radius = [Sc.Radius]';
P2 = Param;
for i = 1:numel(ch), P2(i).Center = Sp(i).Center; P2(i).Radius = Sp(i).Radius; end
out.meg_os = bst_meg_sph(S.src', ch, P2);
R = [0.88 0.93 1.0]' * 0.09; sg = [0.33 0.0042 0.33]';
out.eeg_bst = bst_eeg_sph(S.src, S.eeg, S.ec(:)', R, sg);
out.eeg_bst_out = bst_eeg_sph(S.src_out, S.eeg, S.ec(:)', R, sg);
out.eeg_bst_1 = bst_eeg_sph(S.src, S.eeg, S.ec(:)', 0.09, 0.33);
[mu, lam, f] = bst_berg(R, sg);
out.berg_mu = mu; out.berg_lam = lam; out.berg_f = f;
save(fullfile(work, 'bst_out.mat'), '-struct', 'out');
end
