# The consistent cycle pattern of a band at a spatial scale (spec, 2026-11-13; APPROVED by the maintainers)

## Why

The question is what spatiotemporal pattern repeats from one cycle of a rhythm to the next. Phases rotate from cycle to cycle and cannot be averaged. Products z zᴴ can: they cancel the common rotation. The
cross-spectrum over a window therefore keeps what is consistent across the window's cycles.

The leading eigenvector m of the band's cross-spectrum is the most consistent cycle pattern. Re(m e^{iφ}), φ over [0, 2π), is that pattern through
one cycle, and its eigenvalue's share of the trace says how much of the band's power it holds.

Experiment 13 (scratch, 2026-11-13, chunk 0 of the fixture, six 4 s windows) found:
- one stable pattern in the coarse spatial bands (Weyl 0–4, about 18 cm and up): 37–77 % of the power, repeated from window to window;
- for Weyl 3, a mostly standing oscillation: at 90° the pattern has 8–13 % of its energy at 0°.

This spec turns that experiment into a verb. It changes no method of the experiment. It fixes two things the scratch code did by hand: windows that
cross a chunk edge, and the decimation grid.

## What is computed

For a free-orientation current J (the inverse's 3-vector at each vertex, no orientation assumed), split at each vertex by the surface normal and read
in the subject's Hodge–Dirac modes with their full forms on the intrinsic complex (`nxr::intrinsic`, as `phasegradient --spatial-band`):
- the normal part ρ = n·J, a 2-form, read into the coexact modes' 2-form parts: coefficients a = A2 · x (A2 [K x M], from the kernel);
- the tangent part, a 1-form, read into every mode's 1-form part: coefficients c = A1 · x (A1 [K x M]).

For a spatial band (a Weyl member, gains g_k), the band's coefficient kernel B stacks the normal rows of its coexact modes and the tangent rows of all its
modes, each weighted by g_k, keeping the modes whose |g_k| ≥ 1e-3 (n rows). For a window's sensor cross-spectrum C (M x M, the mean of z zᴴ over the
window's samples), C_Φ = B C Bᵀ (n x n) is the band's cross-spectrum in the coefficients.

From C_Φ:

| quantity | definition | meaning |
|---|---|---|
| pattern m_i | i-th eigenvector of C_Φ (unit norm), i = 1..q | the i-th most consistent cycle pattern |
| share s_i | λ_i / tr C_Φ | how much of the band's cross-spectral power the pattern holds (the consistency) |
| similarity | \|⟨m, m′⟩\| between two windows' (or a window's and the whole span's) patterns | does the same pattern recur (phase-free) |
| standing share | after the phase rotation that maximises ‖Re(m e^{iθ})‖: ‖Im‖² / ‖Re‖² | 0: a standing oscillation; 1: a pattern that travels (a rotating or travelling wave) |

The **phase origin** of a pattern is free (an eigenvector is defined up to e^{iθ}). Every stored pattern is rotated to the frame of largest variance
(θ = −½ arg Σ m_k²), so phase 0 is that frame. This is stated in the provenance.

**On the cortex**, a pattern's coefficients give three complex vertex fields:
- S, the **stream function** of the tangent current's divergence-free part: Σ_coexact (c_k / μ_k) P_k;
- A, the **potential** of its curl-free part: Σ_exact (c_k / μ_k) P_k (`intrinsic::potential`);
- ρ, the **normal current**: vertexDensity(synth2(a)).

## The graph

Two pipelines, as the architecture asks: the time reduction (per-window cross-spectra) is a measurement over a window partition and is stored; the
pattern analysis reads stored nodes and has no loop.

### Stage 1: per-window cross-spectra (an estimator of `csd compute`)

`csd compute --estimator analytic` (a new named option; today's Welch estimator is `--estimator welch`):

```
inputs:     recording (by name), channels (of a kernel, --channels-of, as today)
parameters: band [lo, hi] Hz, family meyer, rate (Hz, the analytic signal's frame rate), window-seconds
chunks:  over recording (the chunk service, with the filter's halo)
  analytic_band   chunk       ℝ [M x T] → ℂ [M x T]   the band's analytic signal (timebank::analysisOfBlock, Meyer)       the Studio's filter → transform "complex"
  downsample        chunk       ℂ [M x T] → ℂ [M x T/d] every d-th sample on a GLOBAL grid (sample index ≡ 0 mod d from the span's start; d = round(fs / rate))
  cross_spectrum  accumulate  ℂ [M x T'] → per window: Σ z zᴴ [M x M] and the count   partition::crossSpectrum over the window Selection
outputs:
  <stem>_<band>_windows   Operator "cross-spectral density" series [W x M x M x 2] (the per-window sums), on `<stem>_channels`
  <stem>_windows          Selection (spans on the recording's time): the window tiling, with each window's sample count
```

- **Windows carried across chunk edges.** The window grid is fixed from the span's start, and a window that crosses a chunk edge keeps accumulating in
  the next chunk (the accumulator holds the partial sum), as Welch segments are. Only full windows are stored, and their count is stored with them.
  The result must not depend on the chunk size (tested).
- **Bad samples** are excluded as in Welch (`--exclude-bad-segments`): a window that holds a bad sample is dropped and counted in the provenance.
- **The span** is the caller's (`--chunks`, `--start/--stop`): chunk 0 by default, as every verb.
- **The memory held:** the open window's sum (M x M complex) plus the stored windows (W x M x M complex); 268 channels, 6 windows: 7 MB.
- **Decimation is not a filter.** The signal is already band-limited to the band (8–12 Hz at 300 Hz is far below Nyquist), so keeping every d-th sample
  aliases nothing. Σ z zᴴ at the decimated rate equals the full-rate sum divided by d, up to the band-limited signal's sampling error (tested). The
  rate is an option (`--rate`); it changes the time frames, not the estimate.
- **Port:** the Studio's chain has the filter and the complex transform; it has no decimation stage. `downsample` is a new step (registered in the
  chain-graph spec's table), flagged here for that reason.

### Stage 2: the cycle patterns (`cyclepattern compute`)

```
inputs:     csd windows (a stored series by name, --csd), kernel (--kernel), the Hodge–Dirac package with full forms and its intrinsic complex
            (--package, as phasegradient), the glued surface (from the package)
parameters: spatial band (weyl:J | weyl:all | modes), patterns q
before (setup, bound once; the same code as phasegradient --spatial-band):
  glued_kernel     operator → the kernel on the glued surface
  split_normal     vector3 kernel → ρ rows and tangent rows (the vertex normals of the embedding)
  read_forms       → A2 [K x M] (read2 ∘ area2form), A1 [K x M] (read1 ∘ flat)
after (per spatial band):
  band_rows        gains, A2, A1 → B [n x M]                                       nxr::filterbank gains; rows kept at |g| ≥ 1e-3
  congruence       B, C_w → C_Φ,w [n x n] per window; and for the whole span        algebra::congruence, dense (a new overload: the whole product)
  cycle_patterns   C_Φ → m [n x q], λ [q], trace                                    new: nxr::patterns::leading
  pattern_rotate   m → m e^{iθ}, the frame of largest variance at phase 0           new: nxr::patterns::rotateToLargest
  similarity       m_w, m_span → |⟨m_w, m_span⟩|, |⟨m_w, m_1⟩|                      new: nxr::patterns::similarity
  pattern_fields   m → S, A, ρ [V] complex                                          new: nxr::intrinsic::patternFields (potential, synth2, vertexDensity)
outputs: (below)
```

- **The span's C_Φ** is the window partition's measurement (the sum of the stored window sums over their counts), never a re-read of the recording.
- **Shares and the standing share** are ratios: derived at read from the stored eigenvalues, trace and pattern (the cardinal rule), not stored as
  reductions of a ratio.
- **No loop over samples or chunks** in stage 2: every step is an `after` step on stored nodes.

## Stored outputs (stage 2)

On the glued surface (the bridge is not cortex: NaN, as phasegradient), per spatial band `<b>` (`weyl<J>`) and pattern `<i>` = 1..q, under the stem
(`--name`, default `<csd>_cycle`):

| name | kind | type | on |
|---|---|---|---|
| `<stem>_<b>_p<i>_stream` | cycle pattern | complex | glued surface vertices |
| `<stem>_<b>_p<i>_potential` | cycle pattern | complex | glued surface vertices |
| `<stem>_<b>_p<i>_normal` | cycle pattern | complex | glued surface vertices |
| `<stem>_<b>_patterns` | cycle pattern coefficients | complex [n x q] | the band's rows (mode index and part: normal / tangent) |
| `<stem>_<b>_eigenvalues` | cycle pattern eigenvalues | scalar [W+1 x q], and the trace [W+1] | the windows Selection, plus the span |
| `<stem>_<b>_similarity` | cycle pattern similarity | scalar [W x 2] (to the span's, to window 1's) | the windows Selection |

The provenance records every option, the phase convention, the windows dropped, and the warnings:
- "a pattern's phase origin is arbitrary: phase 0 is its frame of largest variance";
- "a window of B Hz and T s holds about B·T independent complex samples: a band with more coefficients than that has at most that many patterns per
  window, so a single window's shares are bounded below by about 1 / (B·T), not 1 / n".

**Amendment (2026-11-13, at build; storage only):** the per-window eigenvalues, traces and similarities, and each pattern's standing share, are stored in
the provenance of the pattern fields (they are a few numbers per window; the io model has no manifold for "the windows plus the span"); the coefficients
are stored per pattern on the package's eigen axis as two complex fields (`_p<i>_tangent_modes`, `_p<i>_normal_modes`, 0 outside the band) rather than
one [n x q] array on the band's rows. Nothing computed changes.

## Compute functions (new, pure, each with a unit test)

| function | in → out | test |
|---|---|---|
| `nxr::timebank::downsample(Z, d, offset)` | ℂ [M x T] → ℂ [M x ⌈(T − offset)/d⌉] | a band-limited signal: Σ z zᴴ decimated × d = full rate (relative error < 1e-3) |
| `nxr::algebra::congruence(A, M)` (dense overload) | A [n x m], M [m x m] → A M Aᴴ | equals the pattern call on every pair |
| `nxr::patterns::leading(C, q)` | Hermitian [n x n] → vectors [n x q], λ [q], trace | a planted rank-1 + noise: the vector recovered (similarity > 0.999) |
| `nxr::patterns::rotateToLargest(m)` | ℂ [n] → ℂ [n] | Re of the result has the largest norm over θ |
| `nxr::patterns::similarity(U, V)` | ℂ [n x q] × 2 → [q] | invariant to e^{iθ}; 1 on itself |
| `nxr::intrinsic::patternFields(c, modes, rows, m)` | coefficients → S, A, ρ [V] ℂ | a single coexact mode's tangent coefficient gives its own P_k / μ_k as S |

## Tests (scoped, `tools/tests.py`)

- **Stage 1** (`pipelines/test/test_csd_analytic.cpp`, a synthetic recording):
  - the windows are the same with chunks of two widths (700 and 1777 samples: windows crossing chunk edges), to the halo's truncation (< 1e-4; 1.6e-5
    measured; a 4x halo gives 5e-9). Amendment 2026-11-13: the spec said 1e-12, which only a transform with no truncated tail could give;
  - the decimated estimate equals the full-rate one (÷ d);
  - a window with a bad sample is dropped and counted.
- **Stage 2** (`pipelines/test/test_cyclepattern.cpp`, the two glued spheres of `test_phasegradient_band`):
  - a planted STANDING pattern (one spatial pattern swinging at 10 Hz) plus noise: recovered (similarity > 0.99), its share as planted, standing
    share < 0.05;
  - a planted ROTATING pattern (two patterns in quadrature): standing share > 0.9;
  - two halves of the recording with different planted patterns: the windows' similarity to window 1 falls in the second half;
  - refusals: a csd with no windows, a package without full forms, a kernel/channels mismatch.
- **The fixture** (by hand, never a ctest): chunk 0 reproduces experiment 13's numbers (shares and similarities per band to 1e-6 where the windows
  coincide, which they do on chunk 0: no window crosses a chunk edge in one chunk).

## README

A section "Cycle patterns" (why, what is stored, the phase convention, what a single window cannot resolve) and the two command lines:

```
nxr-neuro csd compute --dataset D --subject S --recording R --channels-of KERNEL --estimator analytic --band 8,12 --rate 300 --window-seconds 4 [--chunks …]
nxr-neuro cyclepattern compute --dataset D --subject S --csd NAME --kernel KERNEL [--package NAME] [--spatial-band weyl:J|weyl:all|modes] [--patterns Q] [--name STEM] [--replace]
```

## What this spec does not do

- No choice of spatial band: `weyl:all` stores every band, and the caller picks (as a frequency band is picked from a spectrum).
- No time-resolved projection of the pattern (how strongly each frame expresses it, ⟨m, x_t⟩): a later step, once the patterns are stored; the 300 Hz
  frames of stage 1 are what it would read.
- No cross-subject comparison.

## Checklist (the architecture policy)

- Can each new compute function be called with a matrix from anywhere? Yes: arrays in, arrays out; none knows a window, a chunk or a recording.
- Can the pipeline run the same chain on another input of the same kind just by naming it? Yes: `--csd`, `--kernel`, `--package` are names; any stored
  windowed cross-spectrum of the kernel's channels is accepted.
- Is every loop over samples or chunks, and every file access, inside io? Yes: stage 1 loops through the chunk service; stage 2 has no loop.
- Is every choice of which data (signal, source, span) made by the caller? Yes: the recording, span, band, rate and window are command-line options.

## Amendment (2026-11-13): windows, decimation and spans on the dyadic time tiles (APPROVED by the maintainers)

**Why.** Settled by the maintainers (2026-11-13): time is ONE partition into tiles of 2^k samples on the line's grid. The chunks, the FFT windows, the
time-windowed measurements, the constant-Q tiling and the multirate all sit on it, and "arbitrary time spans would 'snap' to the nearest division".
Stage 1 cut its windows in seconds from the span's start and decimated by `--rate`. This amendment moves both onto the tiles, together with the Welch
segments of every Welch producer and the spans of every verb. What is computed inside a window does not change: Σ z zᴴ, the Meyer analytic signal,
Welch's estimate.

**What is ported.** This amendment designs nothing new. Every rule below is an existing one:
- the Studio's `backend/src/db/constant-q.ts`, already ported as `io/include/nxr/io/constant_q.h`:
  - `anchorsFor`;
  - the band window `pow2ceil(cycles · fs / lo)`;
  - `strideFor` (the largest power of two leaving ≥ 24 samples a cycle at the band's low edge);
  - hop = window;
  - the closed form {origin, width, hop, count};
- the Studio's envelope (`envelope.ts`, D96), for coarser windows: one array at the finest window, and a coarser window is the dyadic roll-up at read
  (`code >> shift`);
- the Studio's chain Welch (`toolbox/signal/chain.ts`, `welch` / `welchSeg`): power-of-two segments, half overlapping;
- nxr's grid (`nxr/io/grid.py`): window 2^5 ⊂ chunk 2^15 ⊂ block 2^17, every boundary shared.

### Words

| word | meaning |
|---|---|
| frame | a sample |
| tile of level k | samples [m·2^k, (m+1)·2^k) of the recording's line, counted from the line's first sample (not the span's); m is the tile's ordinal and its code (D87) |
| chunk | level 15 |
| block | level 17 |
| window | the level a measurement uses |
| span | a run of whole tiles; seconds snap to the nearest tile boundary, and the snapped extent goes in the provenance |

### 1. Stage 1 (`crossspectrum --estimator analytic`): windows are tiles

```
parameters: band [lo, hi] Hz (Meyer, as before); level k (the window), default: the constant-Q level of the band (below)
chunks:  over recording (the chunk service, with the filter's halo)
  analytic_band   chunk       ℝ [M x T] → ℂ [M x T]        unchanged
  downsample        chunk       ℂ [M x T] → ℂ [M x T/d]      d = constantq::strideFor(fs, lo), a power of two; the samples kept are those whose index on the LINE is ≡ 0 mod d
  cross_spectrum  accumulate  ℂ → per tile m of level k: Σ z zᴴ [M x M] and its count (2^k / d when whole)   partition::crossSpectrum over the level-k tiles
outputs:
  <stem>_<band>_windows   "cross-spectrum series" [W x M x M x 2], the sums over the tiles kept; producer_json.window_index = their ordinals m on the line
  <stem>_windows          the level-k tiling as a closed-form `spans` Selection on the recording's time line (params_json.tiling {origin, width, hop, count}, the
                          envelope's form, D96/D87): the tiling of the whole line. The stored tiles are the series' window_index
```

- **The window.**
  - By default it is the band's constant-Q window: 2^k = `pow2ceil(cycles · fs / lo)`, with `cycles` from `anchorsFor(hertz, fs)`, as the ladder sizes a
    band. At 1200 Hz, alpha 8–12 Hz gives level 10 (1024 samples, 0.85 s, 6.8 cycles of 8 Hz), and beta 15–29 Hz also gives level 10.
  - With several bands, the series is stored at the finest of their levels. One window Selection then serves every band.
  - `--level k` asks for another level, and `--window-seconds S` snaps to the nearest level, k = round(log2(S · fs)) (the `anchorsFor` rounding).
  - A stored level is at most 15 (a chunk). A coarser window is a sum at read (below).
- **The decimation.**
  - The stride is the ladder's, per band: `strideFor(fs, lo)`. Alpha at 1200 Hz gives 4 (300 Hz); beta 15 Hz gives 2 (600 Hz).
  - It replaces `--rate`. Each band's producer records its own `decimation`, `rate_hz` and `points_per_window` (2^k / d).
  - A band whose high edge reaches the decimated Nyquist rate is refused (unchanged). This cannot happen for hi < 12·lo.
  - Since d ≤ 2^k and both are powers of two, a tile holds exactly 2^k / d decimated samples, on the line's grid.
- **No chunk-edge carry.**
  - The fold is one sum by tile ordinal, m = (line sample) >> k.
  - On the real grid no tile crosses a chunk: the level is ≤ 15 = the chunk, and chunks are level-15 tiles. So every tile is completed inside one chunk,
    and the separate partial-window bookkeeping goes.
  - A tile is kept when it is whole (count = 2^k / d) and holds no bad sample (unchanged).
- **Coarser windows are sums at read.**
  - The series reader (`meeg.cpp`, the cross-spectrum series reader used by `cyclepattern`) takes a level L ≥ k. Window M of level L is the sum of the
    stored tiles m with m >> (L − k) = M, and its count is the sum of their counts.
  - A level-L window is kept only when all 2^(L−k) of its tiles are stored (whole and free of bad samples).
  - This is the envelope's roll-up: exact, and nothing more is stored.
- **What is stored, and the memory.**
  - Finer windows store more: (span / 2^k) · M² · 16 bytes a band.
  - Chunk 0 of the fixture (268 channels), alpha at level 10: 32 windows, 37 MB.
  - A 10-minute recording at level 10: about 700 windows, 0.8 GB a band, all held in memory until stored (as today's accumulator does). A caller who wants
    less asks for a coarser `--level`. The options comment states this.

### 2. Stage 2 (`cyclepattern compute`): the window it reads

- `--level L`, or `--window-seconds S` snapped to the nearest level, reads the stored series rolled up to level L (§1).
- The default is `--window-seconds 4`, the window of experiment 13 (the reference). At 1200 Hz it snaps to level 12: 4096 samples, 3.41 s.
- The provenance records the level read, its seconds, and the windows dropped by the roll-up.
- The fixture check (by hand) compares with experiment 13's code run on 4096-sample tiles (the same tiles), to 1e-6. Its old 4800-sample windows no
  longer coincide with any tile.

### 3. Welch (`psd`, `sourcepsd`, `sourcetensor`, `crossspectrum --estimator welch`, `quadrature`, `hodgepower`, and `speedspectrum`, which shares the request): segments are tiles

- **A new named option set, `--set tiles`, becomes the default.**
  - Its segment is 2^k samples, with k = round(log2(`--segment-seconds` · fs)), or `--segment-level k`.
  - The default `--segment-seconds 4` gives level 12 at 1200 Hz (4096 samples, 3.41 s, 0.29 Hz bins), level 13 at 2400 Hz, and level 12 at 1000 Hz.
  - The overlap is ½, so the hop is the tile level k − 1. Segments start on the line's grid at multiples of the hop.
  - The overlap must keep the hop a power of two (0, ½, ¾, …). Any other overlap is refused, and the message names the valid ones.
  - Taper, detrend and scaling are Brainstorm's (Hamming, symmetric; constant; density). Only the segment changes from today's default, so a stored
    spectrum differs from the old default by its segment alone.
- **`--set brainstorm` stays as the named option set that reproduces Brainstorm.** It keeps exactly today's behaviour (method rule 2): 4 s = round(4 · fs)
  samples, 50 %, from the span's start.
  - `--set scipy` and `--set nxr-chain` are unchanged: they reproduce their references.
- **Segments still cross chunk edges.** At ½ overlap, the segment that starts at 2^15 − 2^(k−1) ends past the chunk edge. `nxr::welch::Welch` keeps its
  carry (it is compute's streaming estimator and identical to the last bit for any chunk size). Only its segment length and its first segment's place
  change.
- **The Studio's `welchSeg` rounds a segment UP to a power of two.** Seconds here snap to the NEAREST level, by the maintainers' snap rule and `anchorsFor`. An
  explicit `--segment-level` is exact.

### 4. Spans (`--start/--stop`) snap to tiles, in io

- `nxr::io::SpanRequest` gains `level` (default 0, a sample: today's rounding).
  - `resolve_span` snaps start and stop to the nearest multiple of 2^level on the line's grid, clamped to the recording.
  - A span that snaps to nothing is refused, and the message names the level and its seconds.
  - `span_record` writes the seconds asked, the snapped extent and the level.
- `--chunks` spans are level-15 tiles already and are unchanged. The recording's last chunk may be partial, so a span can end on the line's end rather
  than on a tile boundary. Measurements over whole tiles leave that remainder out, as today.
- The level of a run is the pipeline's (its preparation, never compute):

  | run | level it snaps to |
  |---|---|
  | `crossspectrum --estimator analytic` | the stored window level |
  | the Welch producers | the segment level |
  | `measure` over time | its tiling's width, when it is a power of two (a ladder level, `<recording>_chunks`) |
  | pointwise verbs (filter, frames, opticalflow, the wavelet transforms) | level 0, unchanged |

### Tests (scoped, `tools/tests.py`)

- `test_crossspectrum`:
  - the analytic windows are the same with chunk widths 2^9 and 2^12 (one finer than the level-10 tile, one coarser), to the halo's truncation
    (< 1e-4, as before). The halo, not the tiles, sets this tolerance, so the check is not exact;
  - the stride is `strideFor`;
  - a level-12 read equals the level-10 tiles summed by fours;
  - a tile with a bad sample drops its level-12 window.
- `test_cyclepattern`: the planted patterns are recovered with tile windows (`--level` / `--window-seconds`).
- The Welch producers' tests:
  - `--set brainstorm` reproduces today's numbers to the bit;
  - `--set tiles` uses 2^k samples on the line's grid;
  - an overlap of 0.3 under `tiles` is refused.
- `test_measure_time`: a `--start/--stop` span snaps to the tiling's width, and its windows are whole (no NaN window at a span edge).
- io: a `resolve_span` unit test of the snap (nearest, the clamp, the refusal).

### Decided

- **The analytic estimator's `--segment-seconds` / `--rate` are retired**, not kept as a named option. They were added the same day (#106), no reference
  reproduces them, and the only results made with them are the dev nodes, which are recomputed anyway (old kind). `--segment-samples` or a Welch option
  given with `--estimator analytic` is refused, and the message names `--level` / `--window-seconds`.

### Checklist (the architecture policy)

- **Compute functions are called with matrices from anywhere:** yes. No compute function changes its signature. `downsample` and
  `partition::crossSpectrum` are called with a stride and a width.
- **The same chain runs on another input by naming it:** yes, unchanged.
- **Loops and file access are in io:** yes. The tiles and the snap are io's closed forms (`constant_q.h`, `resolve_span`), and the roll-up at read is
  the series reader's.
- **The caller chooses the data:** yes. The level is a parameter of the method. The span is the caller's, snapped.
