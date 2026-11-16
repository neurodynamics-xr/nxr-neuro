# The band's cross-spectrum, bin by bin (frequency-resolved within a band)

Status: **asked for by the maintainers (2026-11-15: "We don't have to materialize it for the full spectrum necessarily but let's do it for a band")**.

## Why

`crossspectrum compute` stores one complex channels × channels matrix per band: the MEAN of the band's Welch bins. Within a band that mean
keeps a phase, which says a lag only modulo a whole cycle. Kept bin by bin, the band's cross-spectrum also gives:

- the **coherency spectrum** 𝒞_ab(f) = C_ab(f) / √(C_aa(f) C_bb(f)) and its imaginary part, drawn like a PSD across the band;
- the **slope of the phase over frequency**, a time delay τ = +(1/2π) dφ/df in ms, with no cycle ambiguity inside the band (the group delay; if a leads b by d, C_ab = |X_a|² e^{+i2πfd}, so τ = +d);
- the **phase-slope index** (Nolte et al. 2008), Ψ_ab = Im Σ_f conj(𝒞_ab(f)) 𝒞_ab(f + δf), the sign of the slope from the coherency itself.

Size: alpha 8–12 Hz at the default 4 s segment (0.25 Hz bins) is 17 bins; 268 channels: 268² × 17 × 16 bytes ≈ 20 MB a band.

## The graph (the same as today; one stage changes its partition)

    sensors block (chunk + halo: io's chunk service, the segment tiles)
      → window + FFT per segment                       (pointwise: per channel, per bin)
      → X Xᴴ per bin per segment                       (pointwise: per channel pair, per bin, per segment)
      → reduce over the SEGMENT partition (sum, count) (the Welch mean over time: as today)
      → reduce over a FREQUENCY partition              (TODAY: the band's bins → one member, their mean)
      → gather → io (a statistic)

The only change is the frequency partition the last reduction uses:
- `--resolve band` (today's behaviour, the default; unchanged, the same cache key as before): the band is ONE member of the frequency
  partition, its bins averaged.
- `--resolve bins` (new, named): each bin of the band is its own member, so nothing is reduced over frequency. This is the pointwise
  output; the band mean is a reduction of it.

In compute, both are `nxr::welch::CrossWelch` with its `BinGroup`s: one group spanning the band (`mean = true`) for `band`, one group a bin
for `bins`. No new compute function; `CrossWelch` is already tested against its reference. Checked against the Studio's chain: the
Studio's cross-spectrum (`gpu/crossSpectrum.ts`) is band-integrated only, so `bins` has no Studio counterpart yet; the stages and their
order are the Studio's.

## What is stored

Per band, a statistic `<stem>_<band>_bins` (stem as today), measure `cross_spectral_density`, method `welch`, value type `complex`, OF the
recording, from the segment partition `<stem>_segments`:
- axes channel × channel × frequency: the channels as the band statistic's (`<stem>_channels`), then the band's bins as a frequency axis
  (their centres in Hz, the same frequency line and bin indices as the recording's PSD: `bins: [first, last]` in the options);
- float64 [M × M × F × 2] (re, im), Hermitian in the first two axes at every bin;
- `producer_json` and the cache key record `resolve: bins` and every Welch option, as the band statistic does.
`--resolve bins` writes the `_bins` statistic IN ADDITION to the band statistic (the band one is the mean of the bins; both from the same
pass), so readers of `<stem>_<band>` are unchanged.

## Reading it (derived at read, never stored)

`crossspectrum_bins_read(sub, name, channels?)` returns C(f) [F × M × M] and the bins' frequencies. Derived from it by pure functions in
`meeg/` (matrices in, matrices out; called by the reader, by `nxr.scopes`, by the Studio's twin later):

| function | gives | options (named) |
|---|---|---|
| `coherency(C)` | 𝒞(f) per pair and bin | |
| `phaseSlope(C, f)` | τ per pair (ms), its fit's residual | `weights: coherence \| none` (least squares of the unwrapped phase on f, weighted by |𝒞|² or not); `unwrap: adjacent` |
| `phaseSlopeIndex(C)` | Ψ per pair | `normalise: none \| std` (Nolte's Ψ / std(Ψ) by jackknife over segments needs the segments' C: a later option) |

Sign: τ > 0 and Ψ > 0 when the ROW channel leads the column channel (the same sign convention as the stored phase: arg C_ab > 0 when a leads).

A command for development and the parity tests: `nxr-neuro crossspectrum read --dataset D/N --subject S --name <stem>_<band>_bins
[--pair A,B | --seed A] [--reading coherency|imag|phase|delay|psi]`, JSON out.

Source space follows from linearity (T_ij(f) = K_i C(f) K_jᵀ, per bin as per band): the lag map's delay reading (Studio spec
`docs/superpowers/specs/2026-11-15-lag-and-polarisation-views.md`) uses it there; not part of this change.

## References and tests

- The per-bin C(f): `CrossWelch` (already golden-tested); a new test that the mean over the `_bins` statistic's bins equals the band
  statistic exactly, and that the result does not depend on the chunk size.
- `phaseSlopeIndex`: Nolte et al. 2008 (PRL 100, 234101), written from the paper; golden against MNE-Connectivity's `phase_slope_index`
  on a synthetic pair if it can be installed in nxr's environment for the generator, else against a numpy implementation written from the
  paper in the generator.
- `phaseSlope`: synthetic signals with a known delay (a broadband source and its copy delayed by d ms plus independent noise): τ recovered.
- Synthetic data only. A real-data recording (`NXR_FIXTURE_DATASET`, alpha, chunk 0) is the by-hand real-data check.

## What it cannot resolve (README, not in the tool)

The delay is the slope of the phase across the band's bins: one delay per pair for the whole band, from a band only a few Hz wide; within
that range a constant delay and a phase that is not linear in frequency are not told apart beyond the fit's residual.
