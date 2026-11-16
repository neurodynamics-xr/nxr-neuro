# The scaling of a spectral quantity, chosen at read (spec, 2026-11-13; the maintainers approved the design the same day)

## Why

A spectral quantity can be read as a density (units²/Hz), a band power (units²), an amplitude (units), or in decibels. These are one quantity rescaled
or transformed, not different computations. The cardinal rule already says conversions and ratios of stored quantities are derived at read. So each
estimator stores ONE quantity, with the normalisers a conversion needs in its provenance, and the reader picks the scaling. No variant is stored twice,
and every result states its units.

## What is stored (`producer_json.spectral_scaling`)

| producer | stored | band_width_hz | other |
|---|---|---|---|
| `psd` (per bin) | density or spectrum (the Welch option) | the resolution | bin_enbw_hz = fs Σw² / (Σw)² |
| `sourcepsd`, `sourcetensor`, `crossspectrum` (Welch), `hodgepower` (per band, the mean of its bins) | density or spectrum | bins × resolution | bin_enbw_hz |
| `quadrature` energy (Σ (Im C)²) | density or spectrum, degree 2 | bins × resolution | bin_enbw_hz |
| `crossspectrum --estimator analytic` (per window) | band-power, as per-window sums of z zᴴ | the band-pass's noise bandwidth B_n = ∫ (g/2)² df | per_window_sum |

There is no record (so no conversion) for a relative power, nxr's chain scaling, or anything computed before 2026-11-13; the reader refuses those any
scaling but `stored`.

## The conversions (`nxr::spectral`, pure)

- density = stored (density); stored / bin_enbw_hz (spectrum); band power / band_width_hz (band-power)
- power = density × band_width_hz (the broadband reading of a Welch band); band-power as is
- amplitude = √power; amplitude-density = √density; db = 10 log10(power); db-density = 10 log10(density)
- density and power are LINEAR (a factor, raised to the degree): they apply to cross-spectra and commute with sums, means and maxima. The others apply
  only to a real power (degree 1).
- The analytic band power is ⟨|z|²⟩ / 2 (Re z is the band-pass; the analytic signal carries twice its variance).

## Where the scaling is chosen

- `crossspectrum_read(…, scaling)` and `crossspectrum_windows_read(…, scaling)`: stored · density · power.
- `measure export --scaling`: every value of a Field from its record. Linear targets scale the statistics (mean_square and variance by the factor
  squared; ratios such as normal_share unchanged); the non-linear ones apply to mean, max and min only (of the stored power: the amplitude of a
  member's mean power). A gradient's sums are not converted.
- The Welch estimators keep `--scaling density|spectrum`: it reproduces Brainstorm and scipy (the method rule's references).

## Tests

- `test_spectral_scaling`: every conversion and refusal; white noise three ways (Welch density, Welch spectrum ÷ ENBW, analytic band power ÷ B_n)
  read as the same density (σ² / (fs/2) = 0.02250: 0.02239, 0.02239, 0.02236).
- `test_crossspectrum`: the readers; the Welch and analytic band powers of a 10 Hz cosine (0.5): 0.4994 and 0.4978.
- `test_measure_pipeline`: `--scaling power` multiplies a density's mean and sd by the band's width; refusals.

## Not done

`speedspectrum` (the joint tiles' energy) has no record yet; nor do the cycle patterns' eigenvalues (their shares are scale-free).
