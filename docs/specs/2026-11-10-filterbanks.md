# Filterbanks in nxr-neuro: the FFT and the graph filterbank (part 1), the time and joint time-vertex banks and the space-time kernels (part 2) (2026-11-10)

maintainers' goal (2026-11-10): nxr-neuro has the filterbanks Rheome has. Part 1 (this PR): the FFT in `compute/` and the graph filterbank. Part 2 (second half of this note): the time (cwt) and joint
time-vertex banks and the space-time kernels, built on part 1. API: `compute/include/nxr/fft.h` (`nxr::fft`), `compute/include/nxr/filterbank.h` (`nxr::filterbank`).

## The three implementations

| | Rheome | nxr (Python) | Studio (TypeScript) |
|---|---|---|---|
| graph bank | `@graphfilterbank` (`designbank.m`); `+filters/frame.m` is a shim over it | `filters/frames.py::graph_bank` ("line for line" from `designbank`) + `dyadic_frame`, `weyl_frame`, `rank_frame` | `wavelets/dyadicFrame.ts` (Meyer; D108) |
| families | `mexhat`, `heat`, `itersine`, `logitersine` | the same four, plus Meyer | Meyer only (`kernels.ts` has single `heat`/`mexhat`/level-hat gains, not banks) |
| scale placement | mexhat: `t` log-spaced from `1/λmax` (`basis`) or `4.744/λmax` (`usable`) to `2/λmin`, `λmin = λmax/20` unless the spectrum is given (then the smallest nonzero λ); heat: same, one octave coarser at the top; itersine: `Nf` linear windows on `[0, λmax]`; logitersine: `V` voices per octave of √λ below `√λmax` | identical to Rheome | Weyl: edges `4π·2^j/A`, one wavelet per octave; `rank_frame` on the pattern index for first-order (Dirac) bases |
| tightness | itersine, logitersine: Σg² = 1 on `[0, λmax]`; mexhat, heat: not tight (B/A 2.8 and 225 on the test pencil) so synthesis uses the canonical dual `g/S` | same | Meyer: Σg² = 1 for every λ >= 0 (tight) |
| low-pass / high-pass | mexhat: GSPBox's quartic scaling function (peak `1.2/e`); logitersine: a low- and a high-pass member | same | scaling member (1 at λ = 0) and a top member |
| transforms | `+graphtransform.eigen` (exact: `Φ g(Λ) Φᵀ B f`) and `.chebyshev` (order 40, `M⁻¹L`, DCT at K+1 nodes, Clenshaw, power-iteration λmax × 1.01, no Jackson) | no bank transform (the graph bank is a design only; `spectral/envelope.py` has a Chebyshev mexhat operator) | `chebyshev.ts`: polynomial route (FFT-DCT coefficients, degree searched to a tolerance) and a pinned resolvent route (shift-invert, for coarse bands), applied on the lumped-mass pencil |
| synthesis | `iwt`: `dual` (default) or `tight`; `filters.frame_synthesis`: `tight` (default) | tight (Σg² = 1 frames) | tight |
| time bank gain | analytic gain 2 (Σg² = 2) | `itersine_time_frame` keeps 2; `itersine_frame` divides it out | 1 |

## What was checked (this PR's tests)

- C++ `design` equals Rheome's `designbank` and nxr's `graph_bank` on all four families and 11 option sets (default; `usable`; `NumFilters`; `VoicesPerOctave`; `ScaleLimits`;
  `LPFactor` on a `λmax`-only range): gains, scales, frame bounds, centre wavenumbers and mass lost agree to 2e-14 with Rheome (MATLAB, on a synthetic icosphere) and 2e-14 with nxr.
- Analysis (exact and Chebyshev, order 60, mass pencil), dual and tight synthesis, and the energies equal Rheome's to 3e-14.
- Meyer / Weyl / rank frames and `rankCoordinates` equal the Studio's own `dyadicFrame.ts` (run with the monorepo's `tsx`) to 2e-14.
- `nxr::fft` equals `numpy.fft` bit for bit on 111 cases (all lengths 1..257 incl. primes, 1-3 dimensions, any axis, three norms).

## Where they differ (for the maintainers; none was resolved silently)

1. **No Meyer family in Rheome's graph bank, no mexhat/heat/itersine in the Studio's.** nxr-neuro implements all five; "the Studio's graph bank" and "Rheome's graph bank"
   are different families, not two versions of one. The project's pipeline (nxr, Studio) uses Meyer on the Weyl ladder; Rheome's published graph banks are mexhat/heat/itersine.
   Decision needed: which family is the project default for graph analysis (suggestion: Meyer/Weyl, as the Studio and nxr's `tight_frame` use; Rheome's remain for parity).
2. **Scale placement depends on the basis in Rheome/nxr, not in the Studio.** With a spectrum given, `λmin` is the smallest nonzero eigenvalue, so the coarse end of a mexhat/heat
   bank moves with the number of modes (Rheome's own `frame.m` documents this and offers `sigmaMax` to pin it; `graphfilterbank` has no such option, `ScaleLimits` is the equivalent
   and is implemented). Weyl edges depend on the area only.
3. **Gain conventions.** Graph banks: Σg² = 1 (itersine, logitersine, Meyer) or not constant (mexhat, heat), in all three. Time banks: Rheome's analytic gain 2 versus the
   Studio's 1 (known; part 2 must state the convention per bank, as nxr's `Frame.total` does). The Studio's `levelHatGain` is peak-normalised to 1; Rheome's mexhat has peak `1/e`.
4. **Rheome's `iwt` has no polynomial route.** `iwt` needs the transform's coefficient space; with the Chebyshev transform it multiplies a K-vector gain into a vertex array
   (an error, or wrong). `wt` works. nxr-neuro adds Chebyshev synthesis: `g_m/S` (dual) or `g_m` (tight) expanded as a series. `iwt` defaults to the dual, `frame_synthesis` to
   tight; nxr-neuro defaults to `Synthesis::Dual` (exact for every bank that covers the spectrum, and equal to tight for the tight ones).
5. **Chebyshev details.** Rheome: fixed order 40, `λmax` by Lanczos (`fromOperator`) or power iteration × 1.01, a supplied `λmax` below 0.99 of a cheap estimate warns. Studio:
   degree searched per filter to a tolerance; a resolvent route for coarse bands. nxr-neuro follows Rheome (fixed order, error on a too-small `λmax` instead of a warning, because the
   recursion diverges); the adaptive degree and the resolvent route are part 2 or later (the FFT makes the DCT cheap). Measured here: itersine/logitersine converge slowly in the
   series (finite smoothness; at order 60 on 42 vertices logitersine is within 7e-2 of the exact eigenbasis, itersine 2e-3, mexhat 1e-3; heat reaches 1e-14), as does mexhat's quartic low-pass (Rheome's own warning).
6. **Rheome rejects a signed spectrum** (first-order Dirac): the bank is built on `D²` or on the rank coordinate (Studio `rank_frame`). `design` throws on `λmin < 0`; `rank` and `rankCoordinates` are in.
7. nxr's `graph_bank` marks itersine `total = 1`; true on `[0, λmax]` only (zero above): matches Rheome's "tight on the interior". No action.

## Decisions taken in this PR

- **FFT dependency: pocketfft C++ header, vendored** (`compute/deps/pocketfft/pocketfft_hdronly.h`, pinned commit and hash in its README, license kept). Vendored rather than
  `FetchContent`: a build is offline-capable, the wasm build has no network step, and the bytes are fixed. It is the library inside NumPy >= 2 and SciPy, so equality with
  `numpy.fft` is the acceptance test. Built single-threaded (`POCKETFFT_NO_MULTITHREADING`) in `src/fft.cpp` only; the header is not exposed to consumers.
- **Eigen/Spectra stay the linear algebra**; no new dependency beyond pocketfft.
- A bank is a list of `std::function` gains (a data type, not an enum switch), so part 2's time and joint banks use the same `Bank`, `gains`, `bounds`, `dualGains`, `Coefficients` and
  `Synthesis`; a joint bank is a time `Bank` on `nxr::fft::rfftfreq` bins times a graph `Bank`.
- Bindings and pipelines: none in this PR (as agreed).

## Part 2: the time bank, the joint time-vertex bank, the space-time kernels

API: `compute/include/nxr/timebank.h` (`nxr::timebank`), `jointbank.h` (`nxr::jointbank`), `spacetime_kernels.h` (`nxr::kernels`). Pure maths, C++ only (no bindings, no pipelines).
Tests: `test_timebank`, `test_jointbank`, `test_spacetime_kernels`; goldens and their generators in `compute/test/golden/` (`gen_timebank_golden.m`, `gen_jointbank_golden.m`,
`gen_kernels_golden.m` for Rheome and MATLAB; `gen_timebank_nxr.py` for nxr; `gen_timebank_studio.mts` for the Studio).

### Three required choices

`timebank::Design`'s only constructor takes `family`, `gain` and `sides`; the bank records them. The families: `Itersine` (Rheome's `@timefilterbank`), `Meyer` (Studio
`dyadicFrame.ts` / nxr `window_frame`), `Morlet` (Studio `kernels.ts`, nxr `morlet_hat`), `Morse` / `Amor` / `Bump` (MATLAB `cwtfilterbank`, Rheome's reference path `+ingest/bank.m`).
Morlet and the cwt families need their centres given (`logCentres`, `cwtCentres`); nothing derives them silently.

**`sides`.** `Analytic`: one-sided bank on `rfftfreq` bins, REAL signal in, complex coefficients out (envelope and phase). A complex input is refused (`std::invalid_argument`).
`TwoSided`: members at +f and at -f on `fftfreq` bins, tight over the whole axis, no factor 2; for complex signals (complex eigenmode coefficients, rotating patterns,
x + iy fields) and the joint bank. Every member lies on one sign only, the low-pass/scaling too (a rotation slower than the first band keeps its sense); the DC bin, which has
no sense, is a member of its own (`Role::Dc`). `TwoSided` takes `Gain::One` only (a complex signal has no mirror half to double; `Gain::Two` is refused, the maintainers to say if a
two-sided "2" is wanted). Measured: a counter-clockwise tone puts all its energy in the +f members (2.6e2 against 2e-27 at -f), a clockwise one in the -f members; the analytic bank
cannot tell them apart (real parts identical).

**`gain`.** `Two` (Rheome): Sum g^2 = 2 on 0 < f < fs/2, sum of the members' energies = sum x^2 (Parseval direct), reconstruction x = Re ifft(Sum g w). `One` (Studio, nxr): Sum g^2 = 1,
energies add to half of sum x^2, reconstruction x = 2 Re ifft(Sum g w); in general x = (2/G) Re ifft(Sum g w). For the tight families (`Itersine`, `Meyer`) `gain` fixes Sum g^2 = G.
For the families that are not tight (Morlet, Morse, Amor, Bump) there is no constant sum, so `gain` fixes the PEAK gain of each member (`Two`: a tone of amplitude A gives |c| = A, which
is the analytic amplitude of the Studio's `2 x Morlet` and of `cwtfilterbank`; `One`: |c| = A/2). **(Superseded by the gain ruling below; kept as found.)** "gain 2" therefore meant Sum g^2 = 2 (peak sqrt 2) for Rheome's own
`timefilterbank` and peak 2 for Rheome's own Morse reference; the maintainers' sentence "gain 2: |c| = physical amplitude" holds for the second, not the first (for the itersine bank a tone at a
member's centre gives |c| = A sqrt(2)/2).

**DC and Nyquist** (real, unmirrored bins) are handled identically for every family and both gains, by `EdgeBins`: `Single` (default) gives each the interior gain / sqrt 2, i.e.
S = G/2 there: a bin that stands for itself, not for itself and a mirror. Then the bank is exactly invertible for any record (tested: a DC offset and a Nyquist component reconstruct to
3e-16 for both gains, both families, even and odd lengths, periodic and padded), Parseval holds for the whole record, and nothing is doubled. `Drop` gives them gain 0 (Rheome's
`timefilterbank`: "tight up to the DC and Nyquist bins"; the reconstruction then misses exactly that content, tested to 9e-16). Parity with Rheome uses `Drop`.

**Padding.** `Padding::None` (the record is periodic: Rheome, nxr), `FastLen` (zero padding to `nextFastLen(2n)`, no wrap; the Studio pads to `nextPow2(2n)`). `extendSymmetric`
is `cwtfilterbank`'s symmetric extension (`[x; flip(x)]`; its `wt` keeps the first n rows; verified bit for bit in MATLAB).
Coefficients live on the padded grid (`nfft` rows); `synthesis` crops to the record.

**Frame operations**: `gains`, `dualGains` (g/S, with the edge rule), `frameOperator`, `bounds` (interior bins; raw Sum g^2, Rheome prints S/2), `isTight`, `centerFrequencies`,
`rates` and `analysisOwnRate` (Rheome's `wt`: each member at its own rate), `analysis`, `synthesis` / `synthesisComplex` (Dual | Tight), `scalogram`, `energies`.

### The joint bank (`jointbank`)

`W_i(lambda, f) = K_k(lambda, f) g_g(lambda) h_t(f)`, every combination, `ig` fastest (Rheome's `ndgrid`). Graph side: any `filterbank::Bank`; time side: any `TimeBank`, signed f when it is
two-sided; K optional. Without kernels the bank is separable, S = S_g x S_t, and `productBounds` equals `bounds` (tested to 4e-16): tight iff both factors are. Domains: spectrum
(`analysisSpectrum` / `synthesisSpectrum`, Rheome `wt` / `iwt`, `jtv.analysis` / `jtv.synthesis`), modes x time, vertices x time through an `EigenTransform` (exact, any kernel) or a
`ChebyshevOperator` (separable banks only; a bank with a kernel is refused, because a kernel couples lambda and f). `axes` / `compatible` are `jtv.axes` / `jtv.compatible`. Kernels
carry f in hertz; `speedKernel(c, d)` is Rheome's (omega = 2 pi f); `ridgeKernel(v, width, sense)` is nxr's `ridge_filter` (a signed ridge, which needs a two-sided time axis).
Measured: a +ridge / -ridge pair separates a counter-clockwise from a clockwise travelling pattern by 19:1 on a 36-mode sphere (the lambda = 0 mode sits at f = 0 on both).

### Parity (all tests pass; rel = max |a-b| / max(1, max |b|))

| source | what | agreement |
|---|---|---|
| Rheome `@timefilterbank` (4 banks: N 151-300, 2-6 voices, a limited range, an odd length) | gains, centres, bins, bounds (S/2), rates, full-rate coefficients, own-rate `wt` | 5e-13 |
| MATLAB `cwtfilterbank` Morse (default and TimeBandwidth 30), Amor, Bump | centres, gains, coefficients, `wt` (symmetric extension) | 4e-13 (the Nyquist rule changes nothing visible) |
| Rheome `@jointfilterbank` + `+jtv` (sphere spectrum x itersine, separable; mexhat x speed kernel) | frame operator, bounds, `wt` (5 members), `iwt` dual and tight, `jtv.bounds` / `jtv.dual` | 4e-13 |
| Rheome `+filters` | gabor, travwave, resonator, stmatern, bandpass, diffusion, wave, dampedwave, kleingordon, schrodinger, heat, mexhat, diffgauss, sphere_kernel, steer, spin, psitime / band, speedkernel, scatter (mexhat 55 pairs; itersine 0 pairs) | 3e-12 |
| nxr `filters/frames.py` | `window_frame` edges and gains (Meyer, gain One), `itersine_time_frame` (gain Two, Drop) | 1e-14 |
| nxr `filters/kernels.py`, `spectral/joint.py` | `morlet_hat`; `joint.morlet` at +-f0 (two-sided members); `ridge_filter` both senses | 4e-15 / 1.4e-15 |
| nxr `spectral/rotary.py` | A+ = abs(z_+), A- = abs(z_-) from the +f / -f members of a two-sided bank (tone on a bin) | 4e-15 |
| Studio `dyadicFrame.ts`, `chain.ts` | `windowFrame` gains and edges; `bankGain` Morlet (= 2 x hat = gain Two); `waveletFreqs` | 8e-15 |

### Where they differ (flagged for the maintainers; none was decided silently)

1. **"Gain 2" was two things — RESOLVED (see "Gain, one meaning"):** constant sum of squares for tight banks, peak 2 for the cwt/Morlet banks. Rheome's own Morse path and its own itersine bank do not share a
   normalisation (peak 2 against peak sqrt 2).
2. **DC and Nyquist, all seven ways.** Rheome `timefilterbank` and nxr `itersine_time_frame`: gain 0 at both. `cwtfilterbank`: full H at both (negligible). Studio `waveletTransform`:
   bins j < N/2 (DC kept at its Morlet/Meyer gain, which is ~0; Nyquist dropped). Studio `envelope()`: DC and Nyquist x1 while the rest are x2 (the analytic-signal convention: half of
   the interior in amplitude, 1/sqrt 2 in energy split between analysis and synthesis, which is what `Single` does for the tight banks). Studio `meyerFilter`: a real two-sided gain, DC once.
   nxr `rotary_amplitudes` (via `hilbert`): a complex series' DC bin is split half to each sense (|z_dc|/2 in A+ and in A-); the two-sided bank gives DC no sense (own member).
   They agree wherever a record has no DC or Nyquist content; they differ by exactly that content otherwise.
3. **Padding**: Rheome and nxr periodic; `cwtfilterbank` symmetric extension to 2N; Studio zero padding to `nextPow2(2T)` (and `nextPow2(T)` in `envelope`); here `None` or `nextFastLen(2n)`.
4. **The cwt's lowest scale** is not the user's: `cwtfilterbank` raises the bottom to where the wavelet's support fits the record (`cwtfreqbounds`; 22 of 24 requested scales on a 128-sample
   record). Not ported: `cwtCentres(fmin, fmax, voices)` builds the geometric grid from fmax, which MATLAB's centres equal to 1e-13, and the caller supplies fmin.
5. **The Studio's Meyer wavelet bank** (`chain.ts`, `meyerHat`, centres `voices` per octave, scaled 1/sqrt(voices)) is not tight for more than one voice (measured Sum (g/2)^2 in
   [0.947, 1] at 4 voices) and is a different family from the dyadic Meyer frame (`window_frame`), which is tight. Only the dyadic one is ported.
6. **Morlet**: the Studio's width is fc/cycles; nxr floors it at max(|f0|, 0.5)/cycles (so it allows f0 = 0); identical for |f0| >= 0.5. The two-sided Morlet keeps the Gaussian tail across
   f = 0 (e^(-cycles^2/2), nxr's behaviour) rather than cutting it, so the Morlet members at +fc and -fc overlap by that much; the Morse, Amor and Bump are exactly zero on the other sign.
7. **Studio `jointTime.ts` is not a bank**: analytic Morlet atoms on dyadic tiles (`morletAt`) and a nonseparable gain cos(sqrt(lambda) c (t - tc)) under an envelope. It has no frame
   bounds and no dual; not ported. Its dispersion (omega = c sqrt(lambda), continuum) differs from Rheome's `wave` (below).
8. **nxr `joint_power`** is a Welch two-sided power estimate, not a bank; not ported (estimation, not filtering). nxr's `ridge_filter` is Rheome's `speedkernel` made signed.
9. **Rheome's joint bank, 'full' half**: bins 0..N-1 are unsigned (0 to fs), not signed; its kernels then see f and fs - f. Here a two-sided bank is on signed frequencies.

### Ambiguities in Rheome's `+filters`: RESOLVED (maintainers' ruling 2026-11-10; the same fix is in Rheome, PR `fix/filter-kernels`, and here)

1. **The unit of the frequency argument. RESOLVED: Hz, signed, everywhere.** `jtv.bank` and `jointfilterbank` pass `f` in hertz to every time factor and (λ, f) kernel (they passed ω = 2πf); `band(f1, f2)`'s handle takes Hz;
   `speedkernel(c, d)` takes `f` in Hz and forms ω = 2πf inside (a speed stays in length/s; `d` is the angular width, rad/s); `forward.simulate` no longer hands 2πf to kernels that read hertz. `nxr::kernels` and `nxr::jointbank`
   already took Hz; every kernel's header and help now states its units.
2. **`i_signed` on a one-sided grid. RESOLVED.** The grid alone cannot say whether it is 0 … fs−df (full) or 0 … fs/2 (one-sided), so the half is stated: `signedFrequencies(grid, Half::Full | Half::Positive)`
   (Rheome: `filters.signedfrequencies(f, 'full' | 'positive')`, a last argument of `gabor`, `travwave`, `resonator`, `stmatern`, `bandpass`). Positive leaves the grid unchanged (a 20 Hz bin is +20 Hz; the old rule made it −0.83 Hz);
   Full is as before. Tested on both grids.
3. **`wave`, `dampedwave`, `kleingordon` (and `diffusion`, `schrodinger`). RESOLVED: the continuum forms are the definitions, in physical units.** `wave = cos(t·c√λ)`; `dampedwave = e^(−β|t|)·cos(t·c√λ)` (β a given decay
   rate); `kleingordon = cos(t·√(c²λ + m²))` (m, the rest frequency, only through ω²); `diffusion = exp(−D λ |t|)`; `schrodinger = exp(−i λ t/ħ)`; none normalised by λmax. The continuum wave equals the exact sphere spectrum
   ω_l = c√(l(l+1))/R (tested). The Chebyshev-angle form (θ = acos(1 − 2λ/λmax), a discrete-time propagator) is the named `Variant::Chebyshev` / Rheome `'Variant','discrete'`, never the default, its numbers unchanged
   (goldens `wave_cheb`, `wave_cheb_lmax`, `dampedwave_cheb`, `kleingordon_cheb`). `diffusion` and `schrodinger` have no discrete variant (their λmax normalisation only set a relative time unit).
4. **(λ, t) kernels are not joint-plane kernels. RESOLVED (documented and refused).** They act on a time lag and cannot multiply a joint (λ, f) spectrum; every kernel's help says so, Rheome's `jtv.bank` 'ts' error says why,
   and `jointbank::make(..., std::vector<LagKernel>)` throws with the same reason.
5. **`travwave` was sense-blind. RESOLVED:** it takes the SIGNED f and a `sense` (+1 | −1 | 0 = |f|, the previous behaviour). On a complex coefficient a wave one way sits at +f and the other way at −f; tested with a synthetic
   travelling pattern each way.
6. **`scatter`. RESOLVED:** a member's scale is 1/(its centre wavenumber), defined for every family; the low-pass member (Role::Scaling, or a member that is ≥ half its peak at λ = 0, not in a heat bank) is excluded from
   the coarser-second pairing, as first or second member. Pair counts: mexhat M members → (M−1)(M−2)/2 (one scaling function out), itersine 8 windows → 21 (it had 0), heat unchanged (all members paired by centre).
7. **`steer` and `spin`** take a Hamilton quaternion [w x y z]; `spin` applies the half angle itself (the argument is the full rotation). Unchanged (a convention, stated).

### Gain, one meaning (maintainers' ruling 2026-11-10)

`gain` G is the PEAK in-band response of every member, in every family (any positive number, recorded on the bank; `Design`'s constructor takes a `double`, the `One`/`Two` enum is gone). A real tone of amplitude A inside a member's
passband gives |c| = (G/2)·A on an analytic bank (G = 2: the physical amplitude), G·A for a complex tone on a two-sided bank. The tight families (Itersine, Meyer) are G times the unit-sum frame: Σ g² = G², Parseval
Σ‖w‖² = (G²/2)Σx² (analytic) or G²Σ|z|² (two-sided), reconstruction x = (2/G²) Re ifft(Σ g W) (still exact, dual or tight, any G). The non-tight families (Morlet, Morse, Amor, Bump) already had peak G. Rheome's
`@timefilterbank` takes `'Gain'` (default 2) and follows the same rule (`framebounds` reports Gain²/2 = 2; was 1). **What changed numerically:** Rheome's itersine gains ×√2 at the default (peak √2 → 2) and its frame operator ×2;
nxr-neuro's `Gain::Two` itersine/Meyer likewise (Σg² 2 → 4); Gain One is unchanged. Rheome's `ingest.bank` pins `Gain = √2` so its stored tile energies stay Parseval-direct. **nxr (Python) `itersine_time_frame` still has Σg² = 2
(peak √2)**; the parity test now builds the bank at gain √2 to match it: nxr differs from the ruling and needs the same change (not made here: separate repo).

### Not done (and why)

- Bindings (MEX/N-API/wasm) and pipelines: out of scope for this PR, as agreed.
- `cwtfreqbounds` (the support-fit lower limit), the Studio's Meyer mother bank, `jointTime.ts`'s atoms and nxr's `joint_power`: see the differences above.
- Adaptive Chebyshev degree and the resolvent route (part 1's list) remain open.
