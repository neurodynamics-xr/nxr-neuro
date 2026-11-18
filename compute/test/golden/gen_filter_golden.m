function gen_filter_golden(out)
% Brainstorm's temporal filters on synthetic signals: the golden for nxr::signal::filters's Brainstorm option set (test_filter.cpp).
% Only Brainstorm's computational functions are called (no database, no GUI): addpath of its toolbox, then
%   bst_bandpass_hfilter(x, Fs, HighPass, LowPass, isMirror, isRelax)   linear-phase FIR (Kaiser), band-pass / low-pass / high-pass  [also its FiltSpec: coefficients, transient]
%   process_notch('Compute', x, Fs, FreqList, '', bandWidth)            2nd-order IIR notch, zero phase (filtfilt), one frequency after the other
%   process_bandstop('Compute', x, Fs, FreqList, FreqWidth, 'fieldtrip_butter')   Butterworth band-stop, zero phase
% Brainstorm's source is not read (GPLv3); only the help headers, the online tutorial, and the outputs of these calls.
% Run:  matlab -batch "gen_filter_golden('<file>')"   
addpath(genpath(fullfile(getenv('BRAINSTORM3_HOME'), 'toolbox')));   % Brainstorm, as a path only
rng(11);
fs = 100; T = 1500; t = (0:T-1)/fs;
x = [cumsum(randn(1,T))*0.3 + 2*sin(2*pi*7*t) + 10; randn(1,T) + 1.5*sin(2*pi*23*t) - 4];
fid = fopen(out, 'w');
put(fid, 'fs', fs); put(fid, 'x', x);
cases = {'lp30',[0 30],0,0; 'hp5',[5 0],0,0; 'bp1_20',[1 20],0,0; 'bp8_40',[8 40],0,0; 'lp5',[0 5],0,0; 'hp0p5',[0.5 0],0,0; 'bp1_20_relax',[1 20],0,1; 'bp1_20_mirror',[1 20],1,0};
for i = 1:size(cases,1)
  hl = cases{i,2}; [y, S] = bst_bandpass_hfilter(x, fs, hl(1), hl(2), cases{i,3}, cases{i,4});
  put(fid, cases{i,1}, y);
  put(fid, [cases{i,1} '_transient'], round(S.transient*fs));
  put(fid, [cases{i,1} '_order'], S.order);
  if i <= 4 || i == 7, put(fid, [cases{i,1} '_b'], S.b(:).'); end
end
% the notch and the band-stop at 250 Hz
fs2 = 250; T2 = 2000; t2 = (0:T2-1)/fs2;
x2 = [cumsum(randn(1,T2))*0.2 + 3*sin(2*pi*50*t2) + 1.5*sin(2*pi*100*t2) + 8; randn(1,T2) + 2*sin(2*pi*50*t2) + 5];
put(fid, 'fs2', fs2); put(fid, 'x2', x2);
put(fid, 'notch50_100_bw1', process_notch('Compute', x2, fs2, [50 100], '', 1));
put(fid, 'notch50_bw2p5', process_notch('Compute', x2, fs2, 50, '', 2.5));
put(fid, 'bandstop50_w1p5', process_bandstop('Compute', x2, fs2, 50, 1.5, 'fieldtrip_butter'));
put(fid, 'bandstop50_100_w3', process_bandstop('Compute', x2, fs2, [50 100], 3, 'fieldtrip_butter'));
fclose(fid);
end
function put(fid, name, A)
fprintf(fid, '%s %d %d\n', name, size(A, 1), size(A, 2));
fprintf(fid, '%.15g ', A.'); fprintf(fid, '\n');
end
