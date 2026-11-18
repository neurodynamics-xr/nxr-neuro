function r = stages(t_on, t_off, tau_rise, tau_fall)
%NXR.COMPUTE.SELECT.STAGES  The stages of an episode from a fit's four numbers, s (nxr::select::stagesOf): onset, within (the plateau) and offset.
%   r = nxr.compute.select.stages(t_on, t_off, tau_rise, tau_fall)
%   onset [t_on - 2 tau_rise, t_on + 2 tau_rise]; within [t_on + 2 tau_rise, t_off - 2 tau_fall] (NaN, has_within false, when they meet); offset [t_off - 2 tau_fall, t_off + 2 tau_fall].
%   valid: the four numbers were finite and the rise and fall times above 0. NaN in: no stage.
    arguments
        t_on (1,1) double
        t_off (1,1) double
        tau_rise (1,1) double
        tau_fall (1,1) double
    end
    r = nxr_compute('compute.select.stages', t_on, t_off, tau_rise, tau_fall);
end
