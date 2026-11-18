function gen_wcoherence_coi_golden(out)
% The cone of influence of MATLAB's wcoherence (Wavelet Toolbox), the golden for nxr::signal::coneOfInfluence, coneOfInfluenceFrequency and insideConeOfInfluence (test_signal_grid.cpp).
% Synthetic signals only. For two series lengths (even and odd) at one sampling rate, wcoherence's fourth output (the cone, as frequencies in Hz, truncated at the highest centre frequency), its
% centre frequencies (Hz), and the wavelet's constants (the Fourier factor and the standard deviation of the default analytic Morlet's envelope in time, from the toolbox's own wavCFandSD).
% Run:  matlab -batch "gen_wcoherence_coi_golden('<file>')"
rng(5);
fs = 10;
[ff, sg] = wavelet.internal.cwt.wavCFandSD('amor');
fid = fopen(out, 'w');
put(fid, 'fs', fs); put(fid, 'fourier_factor', ff); put(fid, 'sigma_t', sg);
for N = [64 63]
  t = (0:N-1)'/fs;
  x = sin(2*pi*1*t) + 0.1*randn(N,1);
  y = sin(2*pi*1*t + 0.5) + 0.1*randn(N,1);
  [~, ~, freq, coi] = wcoherence(x, y, fs);
  put(fid, sprintf('N%d.n', N), N);
  put(fid, sprintf('N%d.freq_hz', N), freq(:).');
  put(fid, sprintf('N%d.coi_hz', N), coi(:).');
end
fclose(fid);
end
function put(fid, name, A)
fprintf(fid, '%s ', name); fprintf(fid, '%.17g ', A); fprintf(fid, '\n');
end
