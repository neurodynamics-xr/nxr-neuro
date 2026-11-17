# Surrogates: the lag-free nulls as typed, selected runs (2026-11-17)

Status: decided structure, implemented in the same change. Builds on the vortex-cores port (`2026-11-14-vortex-cores-port.md`, piece 10), the
spatial events (`2026-11-14-spatial-events.md`, "Statistics and nulls"), the type system (`2026-11-17-type-system.md`) and compute and run
(`2026-11-17-compute-and-run.md`, 8.1: `lagfree_surrogate` becomes `surrogate.lagfree`).

## 1. What a surrogate is for

A vortex core, a direction consistency in a tile, a repeat between two runs, a patch of high relative power or a lagged share of a burst's second
moment is evidence of a travelling pattern only against what the instrument and the signal's own zero-lag structure produce without one. The
**lag-free surrogate** is that reference: Gaussian noise with the data's own spectrum, mixed by the square root of the band's real (zero-lag)
covariance, scaled to the data's band envelope. It keeps the zero-lag covariance, the spectrum and the envelope, and nothing else. A result beyond
it is beyond what mixing and amplitude modulation produce; it is never by itself proof of a wave (every output says so in its warnings).

The references are Rheome's `+rheome/+cores/lagfreenull.m` (the lag-free draw), `envelopenull.m` (rescaled to the data's envelope), `pairnull.m`
(the lag-free Wishart second moment), `pairshare.m` (the lagged share), `bh.m` and `tilepq.m` (the p and q values), and the Python reference they port.

## 2. The names

The surrogates are used in the statistical evaluation of the vortex and lag analyses, so every name says *surrogate*, *lag-free* and what is drawn.
The verbs stay `compute` and `run`; there is no new top-level verb, and the old names are not kept as aliases.

| was | now | what it is |
|---|---|---|
| definition `vortex_null` | `surrogate_lagfree_vortex` | ONE draw: the vortex chain on a lag-free surrogate (`draw: seed`) |
| `nxr-neuro run null` | `nxr-neuro run surrogate_lagfree_vortex_compare` | the draws in batches with early stopping, and the comparison with the data |
| definition `spatial_events_null` | `surrogate_lagfree_spatial_events` | ONE draw: the spatial-events chain on a lag-free surrogate |
| `nxr-neuro run spatialnull` | `nxr-neuro run surrogate_lagfree_spatial_events_compare` | the draws and the comparison of the home map and the patch sizes |
| definition `wishart` | `surrogate_lagfree_pairs` | each burst's lagged share against lag-free Wishart draws of its own second moment |
| step `lagfree_surrogate` | node `surrogate.lagfree` | the unmixed, shaped and scaled noise of one draw (section 4) |
| (inside `lagfree_surrogate`) | node `surrogate.lagfree_mixing` | the zero-lag mixing operator √(Re C) of the band's cross-spectrum |
| step `wishart_null` | node `surrogate.lagfree_pairs` | the Wishart draws and the per-burst comparison |
| `compute/include/nxr/surrogate.h`, `nxr::surrogate` | `compute/include/nxr/synthesis/lagfree.h`, `nxr::synthesis` | the generators (pure functions of arrays) |
| the tests inside `null_compute.cpp`, `spatial_null.cpp` | `compute/include/nxr/nulltest.h`, `nxr::nulltest` | the comparison's maths (pure functions of arrays) |

A draw's stored nodes carry the pipeline's name and the draw: `<pipeline>.draw<seed>.<output>.<recording>`, e.g.
`surrogate_lagfree_vortex.draw3.null_crossings.<recording>`, `surrogate_lagfree_spatial_events.draw3.home.<recording>`.
`lagfree_envelope` (the data's band power in dyadic time tiles, which the draws are scaled to) is a measurement of the data, not a surrogate,
and keeps its name.

### The compute functions (`nxr::synthesis`, `compute/include/nxr/synthesis/lagfree.h`)

| function | was | what it computes |
|---|---|---|
| `standardNormalAt(seed, stream, index)` | `normalAt` | one standard normal of a counter-based generator (splitmix64, Box-Muller): the same value however samples are cut into blocks |
| `whiteNoiseBlock(seed, rows, first, n)` | `whiteBlock` | [rows x n] standard normals, column j the sample `first + j` |
| `spectrumShapingTaps(freq, psd, fs, half)` | `shapingTaps` | zero-phase, Hann-windowed, unit-energy taps whose amplitude response follows √psd |
| `convolveValid(white, taps)` | `filterValid` | the taps along the columns, the valid part |
| `symmetricSquareRoot(A)` | `sqrtPsd` | √A of a symmetric positive semi-definite matrix (negative eigenvalues clipped; Rheome `pairsqrt`) |
| `shapedNoiseBlock(taps, seed, rows, first, n, scale)` | (new) | the UNMIXED draw: white → taps → per-sample scale |
| `lagfreeBlock(mixing, taps, seed, first, n, scale)` | `lagfreeBlock` | the mixed draw, `mixing · shapedNoiseBlock(…)` (kept: the reference form the equivalence is tested against) |
| `zeroLagMixing(realCrossSpectrum)` | (inside the step) | √(Re C / (tr Re C / M)): the mixing at unit mean channel variance |
| `channelMeanSpectrum(values, channels, frequencies)` | (inside the step) | the mean of a [channels x frequencies] spectrum over the named channels |
| `analyticGainPower(freq, psd, gain)` | (inside the step) | the expected mean \|z\|² of unit-variance noise of this spectrum through the band's analytic gain: 2 Σ psd g² / Σ psd |
| `envelopeScale(power, expected)` | (inside the step) | the per-sample factor √(max(power, 0) / expected): pointwise |
| `interpolateTileSeries(values, stride, first, n)` | `interpolateSeries` | a series stored every `stride` samples read at samples `first …`, linear, held beyond the ends |
| `effectiveBandwidth(freq, psd, gain)` | `effectiveBandwidth` | (Σ P)² dF / Σ P² |
| `lagfreeSecondMomentDraw(ReQ0, dof, seed, draw)` | `wishartDraw` | Q ~ complex Wishart(Re Q0 / k, k), k = max(round(dof), 2) (Rheome `pairnull`) |
| `laggedShare(Q)` | (in the step) | ‖Im Q‖²_F / ‖Q‖²_F (Rheome `pairshare` with L = I, squared) |

### The comparison's functions (`nxr::nulltest`, `compute/include/nxr/nulltest.h`)

| function | was | what it computes |
|---|---|---|
| `sequentialP(exceedances, draws, h)` | `besagCliffordP` | Besag and Clifford (1991): h / n when stopped (exceedances ≥ h), else (e + 1) / (n + 1) |
| `sequentialPs(data, exceedances, draws, h)` | (inline loops) | the same per member, NaN where the data has no statistic |
| `fixedDrawPs(data, exceedances, draws)` | (in the Wishart step) | (e + 1) / (n + 1) of a fixed number of draws (Rheome `tilepq`) |
| `benjaminiHochbergQ(p)` | `benjaminiHochberg` | the q-values (NaN stays NaN and is not counted; Rheome `bh`) |
| `exceedanceIndicator(data, draw, side)` | (inline loops) | per member, 1 where the draw is at least (upper) or at most (lower) the data's, both finite; NaN-safe |
| `Tally`, `tallyDraw(tally, data, draw)` | (inline accumulators) | per member: the upper and lower exceedances, the finite draws, the draws' sum (the roll-up of one draw into the batch's) |
| `undecidedMembers(tally, data, h)` | (the stopping loop) | the members with a statistic and fewer than h exceedances: the stopping rule's question |
| `nullMean(tally)` | (inline) | the draws' mean per member (NaN with none) |
| `occupancyFraction(sumsAndCounts, untested)` | (inline) | per vertex, frames in a patch / frames (NaN or 0 when a vertex has none, as asked) |
| `patchShares(counts, patches, frames)` | `shares_of` | the share of the patches per entry and the patches per frame |
| `unbiasedConsistency(sum, n, minCount)` | `rho2` | (\|S\|² − n) / (n (n − 1)) per tile, NaN under `minCount` (or 2) |
| `meanResultants(sum, n)` | (inline) | each tile's sum / n (a zero row where n is 0) |
| `countRatio(dataCounts, nullCountsSum, draws)` | (inline) | the data's count over the null's mean count (NaN where the null has none) |
| `compareMeanOverMembers(data, draws)` | (in the Wishart step) | a `summary` mean over an irregular selection's members (the bursts) against the draws' means: mean, null mean, null SD, p |

## 3. The graphs

Each draw is ONE RUN of a typed definition (the runner, the checker, the cache, the provenance: nothing special), with `seed` its draw
parameter. The stored PSD, the band's cross-spectrum and the data's band-power tiles are found by name and type-checked like any input.

### `surrogate_lagfree_vortex` (one draw)

```
inputs: recording (read: shape) · kernel · tiles (the ladder) · tiling (frame tiling) · bursts · threshold (the DATA's) ·
        psd · cross_spectrum · power_series (`lagfree_envelope`) · bad · onset · within · offset
before:  surrogate.lagfree_mixing  {cross_spectrum}                                   -> mixing        [M x M], over the cross-spectrum's channels
chunks over recording (its SHAPE only, section 6):
         surrogate.lagfree         {x: recording, psd, cross_spectrum, power_series}   -> y   white -> taps -> envelope scale (UNMIXED)
         analytic                  {x: y}                                               -> z   the band's analytic signal
         frames                    {x: z, tiling, within: bursts}                       -> z_frames
         apply_kernel              {x: z_frames, kernel, tiles, pre_operator: mixing}  -> curl_frames   ONE fused operator (section 5)
         cores                     {x: curl_frames, threshold, exclude: bad}           -> cores, frame, sense, centre, ...
after:   tracks -> tile_crossings (as: statistic)                                       -> null_crossings over the level's tiles
```

### `surrogate_lagfree_spatial_events` (one draw)

The same head (`surrogate.lagfree_mixing` before; `surrogate.lagfree` → `analytic` → `frames` → `apply_kernel` with `levels`, `glued` and
`pre_operator: mixing`), then the spatial-events tail unchanged (`power`, `relative` to the DATA's baseline, `at_level`, `scale`, `patches` at
the DATA's threshold), storing the draw's home map and patch summary.

### `surrogate_lagfree_pairs`

`before: surrogate.lagfree_pairs {recording, bursts, psd}` with the band's analytic window series by name, as `wishart` was: per burst, the
second moment Q of its windows, its lagged share, and the shares of `draws` lag-free Wishart draws of its own Re Q with dof = duration × the band's
effective bandwidth. The maths are `nxr::synthesis` (`lagfreeSecondMomentDraw`, `laggedShare`, `effectiveBandwidth`) and `nxr::nulltest`
(`exceedanceIndicator`, `benjaminiHochbergQ`); the node only arranges arrays.

### The comparisons (`…_compare`)

Orchestration only (`pipelines/src/surrogate_compare.cpp`; stored as `surrogate_lagfree_vortex_compare.<rec>.consistency_p | .crossing_ratio | .resultant`,
`vortex_cores.run_repeat.<a>.<b>.p | .whole_p`, and `surrogate_lagfree_spatial_events_compare.<rec>.home_p | .home_null | .size_null`): for i = 0, 1, …, run the draw definition with `seed: i` through the runner (each draw a
cached node of its own), read the draw's small statistic through io, check it against the data's (section 7), fold it into the tally with
`nxr::nulltest::tallyDraw`, and after each batch ask `undecidedMembers`; stop when none is left after `min_draws`, or at `max_draws`. Then
`sequentialP`, `benjaminiHochbergQ`, `nullMean`, `countRatio`, and store. The loop over draws is the orchestration's; every number is compute's.

## 4. The order of the surrogate chain, and why it is the same draw

The draw was written `x = scale · Rh · (taps * w)` per channel (Rheome `lagfreenull` then `envelopenull`: mixed, then rescaled), followed by the
analytic band filter, the frame sampling and the kernel. It is now written

```
w (white, one stream per cross-spectrum channel) -> taps (spectrum shaping) -> scale(t) -> analytic band -> frames -> [form . band . kernel . mixing]
```

The two are equal, step by step:

1. **The per-sample scale commutes with the mixing.** scale(t) multiplies every channel of sample t by the same number: `scale(t) · (Rh v_t) =
   Rh · (scale(t) v_t)`.
2. **The mixing commutes with the time filters.** The analytic band filter is one FIR filter applied to every channel alike (row-wise
   convolution): for any channel matrix A, `filter(A X) = A filter(X)`. The frame sampling is one Kaiser-sinc interpolation applied to every
   channel alike: `frames(A X) = A frames(X)`. Both are linear and act on time only.
3. **The mixing then joins the spatial operator.** After the frames, every remaining step up to the first non-linear one is a linear map of the
   channels: the kernel K, the tangent part and the `curl_tan` form (`nxr::forms::formOfFreeColumns`: a linear scalar form of the free current,
   folded into the kernel's columns), and the spatial band (the level's hat, applied to the operator's columns). So `M = band · form · K` is
   one [targets x channels] operator (as `apply_kernel` already builds it), and `M · Rh` is one operator too.

Hence `y = M · frames(analytic(scale · Rh · (taps * w))) = (M · Rh) · frames(analytic(scale · (taps * w)))`, exactly in real arithmetic. The
values differ only by rounding: the unmixed block is carried in float32 between the steps (as the mixed one was), and the product is formed in
double. The first non-linear step is `cores` (a threshold on the curl) in the vortex draw and `power` (|·|²) in the spatial-events draw; nothing
after them is fused.

The unit variance of each unmixed stream (unit-energy taps of white noise) and `zeroLagMixing`'s unit mean channel variance make the expected
band power of the mixed draw what it was, so `envelopeScale` is unchanged.

## 5. The fused operator (`apply_kernel`'s `pre_operator`)

Generic, not analysis-specific: `apply_kernel` takes an optional input `pre_operator`, a linear operator that acts on x's rows BEFORE the kernel
(the type system's `compose` move: operator<A → A>, operator<A → B> → operator<A → B>). In preparation the node builds its operator as before
(the kernel padded to x's channels, the form, the spatial band) and composes it ONCE with the pre-operator: `M' = M · P`, [targets x channels].
It is applied per block as the kernel was. Any node may make a pre-operator: an array [M x M] with `meta.channels` (the x rows it acts on and
lands on, in order) and `meta.provenance` (what it is made from). The surrogate's is `surrogate.lagfree_mixing`.

**Typing.** `surrogate.lagfree` marks its output `awaits_operator: {of: <the cross-spectrum>}`; the facts are carried by the linear time steps
(`analytic`, `frames`). A node that consumes a value awaiting an operator must be handed that operator as `pre_operator` (the operator's
`operator_of` names the same cross-spectrum), and the mark is then cleared. Any node that makes a non-linear value or leaves the loop (a
measurement, a statistic) from a value still awaiting its operator is refused before any data is read, with the reason: the unmixed noise is
not the surrogate until it is mixed.

**Cached by its inputs' provenance.** The bound operator is a constant of (the kernel's id, x's channel manifold, the parameters, the form, the
spatial band's resolved choice) and, composed, of the pre-operator's provenance (the cross-spectrum's id and how the mixing was made). It is
kept in a small process cache under that key (the last two operators, double precision: [targets x channels] each), so the draws of one
comparison, which share every one of those inputs and differ only in the seed, build it once. A different kernel, cross-spectrum, level, form
or band choice is a different key. MEMORY: two operators of [glued vertices (x levels) x channels] doubles.

The data's chain uses the same node and the same fusion (kernel · form · spatial band, one operator) without a pre-operator.

## 6. A draw never reads the recording

The recording input of a draw is declared `read: shape` in the definition (`recording: {field: recording, each: true, where: …, read: shape}`).
The runner then asks the chunk service for the recording's SHAPE only (`nxr::io::ChunkSource` in `Reading::shape`): its channels, rate, span
and chunk grid, and each chunk's good-sample mask; no sample is read (the array's header gives the shape). The blocks handed to the graph have
zero rows and the chunk's width and halo. Only a node that declares it reads no sample of its streamed input (`StepSpec::reads_shape_only`:
`surrogate.lagfree`) may take such a value; any other is refused at plan with the reason. The draw's values are a function of the sample index
(the counter-based generator), so the result is the same as before.

## 7. Selections and types of the comparison

A surrogate draw is as typed and as selected as the data's run: that is what makes its provenance whole.

- **The draws form a line.** The draw index i = 0, 1, … is a line; a batch is a member of a grid on it (`batch` draws a member). The tally of a
  batch is a measurement over that member (`tallyDraw` folded draw by draw), and the total is the roll-up of the batches (a sum through the
  grid). The draws' seeds are the line's coordinates (`seed_offset + i`; the second run of a repeat test is offset by 100000).
- **The same selection.** The comparison takes the data's measurement and each draw's measurement over the SAME selection, and refuses a
  mismatch, naming both:
  - vortex: the data's crossings are measured over a level's tiles (a Selection); each draw's `null_crossings` statistic records the Selection
    it is over (`partition_selection_id`, written by `tile_crossings` as a statistic) and its tile count. A draw over another level or ladder, or
    with another number of tiles or stages, is refused.
  - spatial events: the data's home map lies on the glued surface (its vertex axis's manifold); a draw's home map must lie on the same manifold
    with the same vertices, and its patch summary must name the same entries (the same levels and area bins) in the same order.
  - every draw must have been made by the draw definition the comparison asked for (its producer's `pipeline`, version and definition hash).
- **The outputs** are statistics over that selection (`partition_selection_id` the data's level for the vortex; the vertex axis on the glued
  surface for the spatial events), and their provenance names the generator (the draw definition's name, version and sha256), the seeds (first,
  stride and count), the draws made, the stopping rule's outcome and the data's statistics compared.

## 8. What it cannot say

The surrogate has the data's zero-lag covariance (the band's cross-spectrum's real part, the same at every frequency in the band), its mean
spectrum over the channels the cross-spectrum covers, and its band envelope (up to the noise's own sampling variation: the scale is the expected,
not the realised, power). A tile, a vertex or a burst beyond it is beyond what mixing and amplitude modulation produce under the same threshold,
burst frames and baseline (the data's), never by itself proof of a travelling wave. The Wishart test is at the sensors.

## 9. Not in scope

Dynamical families of surrogates, an impulse engine, vortex atoms, forward simulation of sources, parallel draws.

## 10. Tests

The old C++ tests stay as they are, with the renames only (`test_surrogate`, `test_wishart_null`, the vortex and spatial-events pipeline tests);
they are ported in the final round. To add then: the equivalence of section 4 on synthetic blocks (the mixed draw through a linear chain equals
the unmixed draw through the same chain with the mixing composed into the operator, to float32 rounding); the comparison's refusals of a draw over
another selection; a shape-only source reads no array chunk (`ChunkSource::reads()` stays 0).
