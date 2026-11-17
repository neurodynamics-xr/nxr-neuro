function y = notch(x, frequencies, sfreq, opts)
%NXR.OPS.FILTER.NOTCH  Notch filter at the line frequency and its harmonics (nxr::filter::notch).
%   y = nxr.ops.filter.notch(x, [50 100], 1000)
%   Zero-phase filters of a whole in-memory record [channels x samples]: Brainstorm's FIR (Kaiser window) and IIR (Butterworth notch and band-stop)
%   filters as documented in nxr/filter.h. The result has the same shape and precision.
%
%   Inputs
%     x         [channels x samples]  The record, single or double.
%     frequencies   Hz: the line frequency and its harmonics.
%     sfreq     Sampling frequency, Hz.
%   Name-value options (the library's FilterOptions; the same names as in the Python wheel)
%     method               "fir" or "iir". Default [] (unset): the reference's (FIR, but IIR for bandstop and notch).
%     ripple               Kaiser: the passband and stopband deviation (linear); the stopband attenuation is -20 log10(ripple) = 60 dB for the default
%                          1e-3 (Brainstorm "relax": 1e-2).
%     transition_bandwidth Hz, each transition band (the lower edge's with firwin). Default [] (unset): the reference's rule.
%     edge                 FIR: the two ends of the record are padded with the mean of the first and last K samples ("mean_of_transient", Brainstorm;
%                          default), or with nothing ("zeros").
%     transient_energy     mean_of_transient: K = the shortest one-sided impulse response holding this fraction of its energy (default 0.99).
%     transient_samples    mean_of_transient: K stated (samples), instead of the energy rule. Default [] (unset).
%     remove_mean          Subtract the record's mean before filtering; added back only when the filter has no high-pass part (Brainstorm; default true).
%     notch_bandwidth      Hz: Brainstorm the 3-dB bandwidth of each notch (default 1 Hz), MNE the width of each stop band (default f/200). Default [] (unset).
%     bandstop_order       Butterworth prototype order N (the filter has 2N poles; Brainstorm: fieldtrip's 'but', N = 4; default 4).
%
%   Output
%     y   [channels x samples], in the input's precision.
    arguments
        x {mustBeNumericOrLogical}
        frequencies {mustBeNumeric}
        sfreq (1,1) double
        opts.method = []
        opts.ripple (1,1) double = 1e-3
        opts.transition_bandwidth = []
        opts.edge {mustBeTextScalar} = "mean_of_transient"
        opts.transient_energy (1,1) double = 0.99
        opts.transient_samples = []
        opts.remove_mean (1,1) logical = true
        opts.notch_bandwidth = []
        opts.bandstop_order (1,1) double = 4
    end
    y = nxr.impl.filter_run('notch', x, sfreq, 0, 0, frequencies, opts);
end
