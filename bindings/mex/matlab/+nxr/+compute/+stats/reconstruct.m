function y = reconstruct(scores, components, mean, scale, k)
%NXR.COMPUTE.STATS.RECONSTRUCT  The rank-k reconstruction scores(:, 1:k)*components(1:k, :).*scale + mean, [n x p] (nxr::stats::reconstruct).
%   y = nxr.compute.stats.reconstruct(r.scores, r.components, r.mean, r.scale, k)
%
%   The best rank-k approximation of the centred table in the least-squares sense (Eckart-Young). k runs from 0 (the mean) to the components held.
    arguments
        scores {mustBeNumeric}
        components {mustBeNumeric}
        mean {mustBeNumeric}
        scale {mustBeNumeric}
        k (1,1) double
    end
    y = nxr_compute('compute.stats.reconstruct', scores, components, mean, scale, k);
end
