# The bispectrum in eigenmode coordinates, and the flow across frequencies (2026-11-24): the graphs

**Status:** design written before the code (CONTRIBUTING, "Adding a node or a pipeline", step 0), then built: `sensor_bispectrum` and `sensor_flow_across_frequencies`. The
single-event shape and the per-event flow across frequencies are graphs only (not built). Sensor space; the same nodes read any surface's chunk spectra. It replaces the separate
"phase-amplitude coupling", "harmonics", "speed across frequencies" and "waveform shape" sketches of `docs/specs/2026-11-24-sensor-flow.md` ("Pipelines that read the same chunk
spectra").

## Why

The sensor flow pipeline reads second-order statistics: one frequency at a time. The relations BETWEEN frequencies that survive a sum over a run are third-order: triple products
x(f1) x(f2) conj(x(f1 + f2)), whose phases cancel unless the three components keep a fixed phase relation. Three questions are readings of the same object:
- **phase-amplitude coupling**: a high rhythm's amplitude following a low rhythm's phase puts power at f_high +- f_low with a fixed phase to f_low;
- **harmonics of non-sinusoidal waveforms**: a waveform that is not a sine has harmonics phase-locked to its fundamental: bicoherence at (f, f, 2f);
- **waveform shape**: the biphase at (f, f, 2f) says how: its real part the peak-trough asymmetry, its imaginary part the rise-decay asymmetry.
So there is ONE pipeline, the bispectrum in the eigenmode coordinates of the stored chunk spectra, weighted by the flow's atoms, with these readings. Whether the FLOW moves at
the same speed and direction across frequencies is a different question, second-order per atom: it is a readout of the flow pipeline's own per-atom results (below).

## Notation and sizes (a 268-sensor helmet, 1200 Hz, a 300 s run)

K = 268 field coordinates (the constant and the exact modes), S = 268 vertices, T = 11 chunks, N = 65 536 (each chunk's transform), B = 3 277 bins of [0, 60) Hz (0.0183 Hz
apart), A = 17 atoms (the flow's: 4 x 2^(k/5) Hz, standard deviation centre / 12), Q = 7 eigenvalue bands. x_t(k) the chunk spectra [T x B x K] (stored, float32, 77 MB).

## The bispectrum (`sensor_bispectrum`)

### The graph

| part | piece | in → out |
|---|---|---|
| loading (io) | the stored chunk spectra, by name (`<name>_chunk_spectra` of `sensor_flow` or `sensor_flow_bins`); the atoms (a kernel set); the Hodge-Dirac package and the bands (surface only) | read by name; never the recording |
| preparation | type checks: the spectra are [chunks x bins x coordinates] with their frequency line and coordinates' modes; the atoms are a kernel set on that line; `space surface` needs the package, bands only on the surface | refused with the reason before any data |
| graph | `signal.bispectrum` (NEW; compute `nxr::signal::atomTriples`, `nxr::signal::atomBispectrum`) | spectra, atoms [, package, bands] → [elements x triples (x bands) x measures] |
| readback | one statistic `<name>`: `bispectrum_real`, `bispectrum_imaginary` and the normalisation's sums per element and triple, each triple's atoms, centres and weight | derived at read: bicoherence, biphase, asymmetry |

### The maths

Per triple (g1, g2, g3: one gain per axis, each a row of the atoms' gains on the bins), per element r:

    B_r = Σ_t Σ_{i, j} g1(i) x_t,r(i) · g2(j) x_t,r(j) · conj(g3(i + j) x_t,r(i + j))     (bins i, j, i + j all kept)

The direct, FFT-based bispectrum (Hagihira et al. 2001; Sigl and Chamoun 1994; the classic estimate at one pair of bins when each gain is one bin) weighted by atoms. Its time-domain
form, with z = the atom's band signal (the inverse transform of its weighted bins, as `signal.atom_time_cells` makes it): B = N² Σ over the transform's samples of z1 z2 conj(z3).
- **phase_amplitude_coupling**: (low, high, high), every pair with the high centre above the low: B = N² Σ z_low |z_high|². Its modulus is the coupling, its argument the low
  rhythm's phase at which the high amplitude peaks. The bispectral view of Zandvoort and Nolte (2021): coupling is read only over the pairs (f1, f2) with f2 and f1 + f2 both in the
  high atom, so the high atom must be wider than the low frequency; with the flow's narrow atoms (standard deviation centre / 12) only pairs about five times apart are read. Each
  triple's weight Σ g1 g2 g3 is stored, so a reader sees where no pair carries it.
- **harmonics**: (a, a, a at twice its frequency), the third gain g_a(f / 2) (the same atom, normalised the same way, evaluated at half the frequency): bicoherence at (f, f, 2f).
  The biphase gives the waveform's shape (Bartz et al. 2019): with numpy's forward transform, Re(B) > 0 for sharp peaks and broad troughs, < 0 for sharp troughs; Im(B) > 0 for
  a slow rise and fast decay, < 0 for a fast rise and slow decay (`test_bispectrum` checks all four on generated waveforms). B is invariant to each chunk's time origin.
- **Normalisations** (named options; ingredients stored, ratios at read), with w = g1 g2 g3 >= 0 (the gains are refused if negative):
  - `threenorm` (default: Shahbazi Avarvand et al. 2018, used by Zandvoort and Nolte 2021 and Bartz et al. 2019): the cube sums Σ w|x(i)|³, Σ w|x(j)|³, Σ w|x(i + j)|³;
    bicoherence = |B| / cbrt(their product) <= 1 (Hölder over the chunks and pairs).
  - `kim_powers` (Kim and Powers 1979; Hagihira et al. 2001; the BIS literature, Sigl and Chamoun 1994): Σ w|x(i) x(j)|² and Σ w|x(i + j)|²; bicoherence² = |B|² over their
    product <= 1 (Cauchy-Schwarz).
  - `none`: B only.
- **Space** (`--space`): `coordinates`: each field coordinate with itself (K rows); the reading per eigenvalue band is a selection of these rows at read (the measurement system).
  `surface`: each vertex (S rows) of the field synthesised from the coordinates; with `--banded true`, one field per band: the synthesis composed with the band's gains FIRST
  (one S x K operator per band, Psi diag(h_q)), applied to each chunk's coefficients. The coupling BETWEEN coordinates or vertices (the spatial bispectrum) is not read.
- The atoms are the flow's own weighting (`frequency_normalisation: sum_of_squared_gains`, `support_floor`), so a coupling found here is between the flow's atoms.

### Compose first; the cheapest order (sizes and ranks)

| object | shape | size | how |
|---|---|---|---|
| the full spatial bispectrum | K³ x B² | 2 x 10^14 entries | never formed |
| per pair of bins, per element | K x (pairs of an atom triple: up to ~10^6) per chunk | ~2 x 10^9 products per triple | never: the sums below give the same numbers |
| a chunk's coefficients | K x B complex | 14 MB double | read from the stored spectra, one chunk at a time |
| surface: the composed operator per band | S x K | 0.6 MB | Psi diag(h_q), once; applied to K x B (never S x S) |
| the correlation of one (second, sum) pair along the frequency line | B x rows complex | 14 MB | one transform of length >= the two supports (nxr's FFT), rows at a time |
| the gains' own correlations and convolutions (data-free) | B x triples, three | 12 MB | once per run |
| the stored result | rows x 153 triples x 5 measures | 1.6 MB (coordinates) | sums |

Triples that share their (second, sum) gains share ONE correlation W(i) = Σ_j g2 x(j) conj(g3 x(i + j)), read with every first gain by a dot product; triples that share their
(first, second) gains share one convolution, read with every sum gain. `atomTriples` picks the grouping with fewer distinct pairs: with both readings on the flow's 17 atoms, 34
distinct (second, sum) pairs against 153 (first, second) pairs. Cost: per chunk and pair, three transforms of about 2B points per row: about 3 x 10^8 operations, so about
10^11 for the run (minutes on one core); the direct sum over pairs of bins would cost about 3 x 10^11 for the coupling triples alone. The cube sums and the sum power need only
the gains' correlations (no data transform). Nothing is formed per sample, per pair of bins or as a matrix of a single chunk.

### Checked against the Studio's chain

members (the stored coordinates: the serving operator already applied) → filter (the atoms as gains on the frequency line; on the surface, the bands' gains composed with the
synthesis: one linear map, applied before any product) → transform (none: the spectra are the chunk's transform) → the forms (triple products, the non-linear step, after every
linear map) → the measures (sums over the chunks and pairs of bins, folded chunk by chunk). The GPU form: per chunk, batched FFT correlations of the weighted spectra (GEMM-shaped
reads per first atom); the data-free correlations on the CPU once.

## The asymmetry of single events (`sensor_event_shape`; graph only, not built)

The whole-run bispectrum cancels a shape whose phase drifts; single events are the burst pipelines'. From the same stored spectra:
- loading: the chunk spectra by name; the events, a selection on the time line from `band_bursts`, `gabor_bursts` or `global_bursts` (REUSED);
- graph: per chunk, the atoms' band signals z (the inverse transform of each atom's weighted bins, the overlap dropped: the step `signal.atom_time_cells` already does, factored as
  a node `signal.atom_signals`, NEW) → a POINTWISE node `signal.triple_product` (NEW: z1 z2 conj(z3) per sample, and |z|³ of each for the threenorm) → measured over the events
  partition (the measure `sum`: the measurement system, never a loop in the node);
- readback: per event, per element, the sums; bicoherence and biphase derived at read, as for the run. The (f, f, 2f) triple's biphase per event is its waveform's asymmetry.
- what it cannot resolve: an event shorter than the atom's time width (its band signal spreads beyond the event); one event's bicoherence has a large bias (a null of shuffled
  events or noise of the same length).

## The flow across frequencies (`sensor_flow_across_frequencies`)

Whether a flow moves at the same speed and in the same direction at every frequency is a statistic over the members of the atoms' level, read from the flow pipeline's stored
per-atom results: never the recording.

| part | piece | in → out |
|---|---|---|
| loading | `<name>_flow_edges`, `<name>_flow_modes`, `<name>_flow_vertices` (`signal.spectral_flow`'s whole field), the package, the atoms | by name |
| graph | `signal.flow_across_frequencies` (NEW; compute `nxr::signal::flowAcrossFrequencies`) | edges [E x A], modes [M x A], vertices [S x A x 6] → atom pairs [A x A x 3], dispersion [S x 4] |
| readback | `<name>_atom_pairs`, `<name>_dispersion` | statistics |

- per pair of atoms (a, b): the cosine of their edge phase currents in the Hodge inner product (weights star1), of their exact (travel) and of their coexact (rotation)
  Hodge-Dirac coefficients (an orthonormal basis: plain dot products). 1: the same flow; 0: unrelated; -1: opposite.
- per vertex: the direction consistency |Σ_a u_a| / n, u_a the unit vertex phase current of atom a; the least-squares slope and intercept of log apparent speed against
  log frequency, the apparent speed 2 pi f_a power / |vertex phase current| (an upper bound). Slope 0: the same speed at every frequency (no dispersion); 1: the same
  wavelength. The two hypotheses are the two ends of one number, so neither has to be chosen before the data.
- Sizes: A x A x 3 and S x 4: negligible. The sign convention of the phase current is common to every atom, so nothing depends on it.
- **An event** (graph only): the stored `<name>_time_cells` of `sensor_flow` measured over an event selection (the sum of each atom's cells inside the event: the measurement
  system) give each atom's edge and vertex currents and power within the event; the same node compares them across atoms, its mode coefficients projected from the edge currents
  (`projectOneForm`; a `modes` port made optional). Over events, the cosines and slopes are distributions with events as replicates (a sign-flip null over events).
- What it cannot resolve: the apparent speed is an upper bound; a whole-run flow keeps the net quadrature only; one run gives descriptions, not a test.

## Reused, changed, new

- Reused: the stored chunk spectra and the flow's outputs (`sensor_flow`), the kernel sets (`select_kernels`), the package binding (`bind_hodge_package`), nxr's FFT,
  `normaliseBySumOfSquaredGains`, `cosineSquaredPartition`.
- New: `nxr::signal::atomTriples`, `atomBispectrum`, `flowAcrossFrequencies`; the nodes `signal.bispectrum`, `signal.flow_across_frequencies`; the two definitions.
- Not changed: every sensor-flow node. The chunk-spectra reader and the kernel-gain helper are read the same way as in `sensor_flow_steps.cpp` (one shared reader to follow).
- Ports: nxr and Rheome have no bispectrum; Rheome has a time-domain mean-vector-length coupling per tile (Canolty et al. 2006), not a bispectral one. Nothing was ported.

## What the measurement cannot resolve

- The field on the helmet only: no source, no field spread, no cortical claim.
- A whole-run sum: coupling or shape whose phase drifts over the run cancels.
- Coupling only where the high atom is wider than the low frequency (Zandvoort and Nolte 2021); the weight is stored.
- A harmonic whose centre lies above the span is not read (left out); bicoherence at (f, f, 2f) does not tell a harmonic from a coupling of f with itself.
- The recording's ends are reflected into the first and last overlaps: time runs backwards there and a rise-decay asymmetry is reversed in those samples.
- Each element with itself only: coupling between coordinates or between vertices is not read.
- Bicoherence is biased upward by few chunks: compare with a null of the same length.
- The apparent speed is an upper bound; the flow across frequencies of one run is a description.

## Open questions (for the maintainers)

1. Whether coupling BETWEEN coordinates (the low frequency's phase on coarse modes, the high frequency's amplitude on fine modes) is wanted: K² rows per triple instead of K.
2. The null for the bicoherence: chunks' spectra shuffled between the axes (the classic surrogate for bispectra), phase-randomised chunks, or the empty-room recording.
