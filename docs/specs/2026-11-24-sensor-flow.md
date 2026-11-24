# The sensor-space flow analysis (2026-11-24): the graph

**Status:** design, written before the code (CONTRIBUTING, "Adding a node or a pipeline", step 0). Sensor space only: no inverse kernel, no cortex.

## Why

The cross-spectrum of the MEG channels holds the lags between them, but by itself it has no spatial meaning. Read on the sensor surface (the helmet
triangulated through its sensors, `sensor_surface`) with that surface's Hodge-Dirac eigenmodes (`eigenmodes`), the lags become a field on a surface with a
geometry: a phase current on the surface's edges, split into its travelling part (exact: curl-free) and its rotating part (coexact: divergence-free). This
describes the magnetic field on the helmet only. It says nothing about sources: without an inverse model the field gives what it gives.

## What the pipeline does, in one paragraph

Per chunk of the recording (2^15 samples, at the recording's own rate: no resampling), the chunk with an **overlap** of half a chunk on each side (taken from
the neighbouring chunks, mirrored at the recording's two ends) is read through a composed spatial operator into the **field coordinates** of the sensor
surface (the constant and the exact eigenmodes' vertex rows) and Fourier-transformed once: 2^16 samples of real data, no window, no taper and no zero
padding. Its bins in the caller's frequency span are kept. These chunk spectra ARE the stored cross-spectrum, in its exact factored form (the cross-spectrum
of one bin is the sum over the chunks of each chunk's vector times its conjugate). Then, once per run, **atoms** are applied: cells of the joint frequency x
eigenvalue level, each a frequency gain (a wavelet on the frequency line, which is also its taper in time) times a band of eigenmodes (a gain on the Dirac
eigenvalue mu). An atom is a weighting of the cross-spectrum, never a cross-spectrum of its own. Per atom the pipeline reads power, the phase current per
intrinsic edge, its Hodge-Dirac coefficients (travel and rotation), the potential and the stream, the power-weighted phase gradient at the vertices, and the
same for every PAIR of eigenvalue bands. Per atom it also reads the flow in the atom's own time cells, and tests each edge's and vertex's direction over
those cells with a sign-flip statistic. Optionally the aperiodic background of every sensor's spectrum is fitted (specparam's aperiodic model, with and
without a knee).

## The graph

Each piece placed in the four parts of the architecture (loading · preparation · graph · readback). K = the field coordinates (the constant and the exact
modes), S = the sensors (vertices of the sensor surface), E = its intrinsic edges, F = its faces, M = the Hodge-Dirac modes (exact and coexact), C = 2^15
(a chunk), O = the overlap (C/2 on each side), N = C + 2 O = 2^16 (the transform), B = the bins in the frequency span, T = the chunks of the span,
A = the frequency atoms, Q = the eigenvalue bands.

### Loading (io)
- The recording, read only by io's chunk service with an overlap of O samples on each side (the chunk loop's halo, the word retired for overlap), clipped at
  the recording's ends; the kept projectors applied by the service as for every chunked read.
- Stored nodes, read by name: the sensor surface's Hodge-Dirac eigenvectors operator (`<stem>_eigenvectors`, with its 1- and 2-form parts and the complex's
  options: `reads_package_forms`), the surface's vertex-to-channel map (`<surface>_map`), two kernel sets (`select_kernels`): the frequency atoms on the
  chunk spectra's frequency line and the eigenvalue bands on the eigenvalue line, and the recording's bad segments (optional).

### Preparation (host, `pipelines/`)
- Type checks before any data: the operator's channels against the recording's rows; the map's surface against the eigenvectors' surface; the kernel sets'
  lines against the frequency line the serving node makes and the eigenvalue line of the eigenvectors; the span snapped to the chunk grid.
- Constants bound once: the composed serving operator (below); the intrinsic complex and the modes (`bind_package`, as `signal.flow_map` binds them).

### The graph of pure nodes

| step | node | kind | in → out | reused / new |
|---|---|---|---|---|
| before | `signal.mode_analysis` | array | eigenvectors + channel map → the serving operator [K x channels]: the field coordinates' analysis rows (Psiᵀ star0, each coordinate at unit star0 norm) composed with the channel map | REUSED; one new choice of `modes`: `exact_and_constant` (the constant 1/sqrt(area) first, mu 0) |
| chunks | `signal.overlapped_spectrum` | chunk (windowed: overlap O on each side) | recording block with its overlap, serving operator → the chunk's spectrum [K x B] complex, once a block, on the frequency line it makes (spacing fs / N) | NEW (compute `nxr::signal::overlappedSpectrum`) |
| chunks | `signal.chunk_spectra` | accumulate | the per-block spectra → `chunk_spectra` [T x B x K] complex: one value per member of the chunk grid (a measurement over the chunk partition; nothing summed) | NEW |
| after | `signal.spectral_flow` | array (statistic) | chunk spectra, eigenvectors (+ forms), a frequency weighting (the atoms: a kernel set; or sharp sets of bins; or every bin), eigenvalue bands (optional) → per frequency member: whole field and per band pair (power, edge phase current, flow coefficients on the M modes, potential, stream, vertex phase current) | NEW, shared unchanged by the whole-recording pipeline below; reuses `modeCrossSpectrum`, `projectOneForm`, `exactPotential`, `coexactStream`, `sharp` |
| after | `signal.atom_time_cells` | array (measurement over time cells) | chunk spectra, eigenvectors, atoms, bands → per atom x band, per time cell of the atom's own width: edge phase current, vertex phase current (3 components), power | NEW |
| after | `stats.sign_flip_consistency` | array (statistic) | the per-cell fields of one atom x band → R1 per element, p, family-wise p, threshold, median and its null | NEW (with `nxr::stats::NumpyGenerator`, a port of numpy's `default_rng`) |
| after (option) | `signal.sensor_power` + `stats.aperiodic_fit` | array (statistic) | chunk spectra, serving operator's synthesis → each sensor's power per bin (mean over chunks) → offset, exponent, knee, r2 per sensor, fixed and knee models | NEW (port of nxr `nxr.spectral.aperiodic`) |

The kernel sets are made by their own runs of the REUSED `select_kernels`, which gains two profiles: `gaussian` with `width_rule: proportional` (the golden's
constant-Q atoms: the standard deviation is the centre / Q) and `cos2_partition` (the golden's amplitude partition in log eigenvalue: neighbouring gains sum
to one, the outer members flat beyond their centres). The Mexican hat graph bank (`nxr::signal::graphBank`, `mexhat`) divided by the sum of its gains is
the named alternative `mexican_hat_partition`.

### Readback (io)
- `chunk_spectra`: a statistic of the recording, axes chunk (the recording's chunk grid), frequency (the line the serving node made: N, fs), coordinate (the
  eigenvalue line's members, the constant first). Precision follows the input (float32).
- `spectral_flow`: measurements over the product (frequency atoms x eigenvalue bands x eigenvalue bands) and over the frequency atoms alone (the whole field).
- `atom_time_cells`: measurements over each atom's time cells (a dyadic level of the time line), per band.
- `sign_flip`: a statistic per atom x band.
- `aperiodic` (option): a statistic per sensor.
- Derived at read, never stored: shares (travel and rotation of the flow energy, band pairs of the whole), the apparent speed 2 pi f power / |vertex phase
  current| (an upper bound), the power-weighted phase gradient (vertex phase current / power), the periodic share 1 - aperiodic / power.

## The maths of each new node

**Serving** (`overlappedSpectrum`). In: the chunk with its overlap, x [channels x (left + n + right)], the overlap O, the serving operator P [K x channels],
the bins [first, first + B). The transform's samples are x[:, first - O .. first + n + O); where the recording ends, the missing side is the recording's end
reflected about its last sample (numpy `reflect`: the edge sample not repeated), so every transform is N = n_chunk + 2 O = 2^16 samples long (the last,
partial chunk is extended on its right by reflection to the same N: one frequency line for every chunk). Compose first: P is applied to the samples (K x N,
K = S at most) before the transform, and the transform is nxr-neuro's FFT, once a chunk. No window, no taper, no zero padding: the frequency atoms are the
taper. Out: [K x B] complex.

**Atoms.** Frequency gains g_a(f) on the bins (the golden's: centres 4 x 2^(k/5) Hz, Gaussian, standard deviation centre / 12, gain 2 on the positive
frequencies), NORMALISED by the sum of the squared gains of the set at each bin (a named choice, `frequency_normalisation: sum_of_squared_gains | none`; with
it the atoms add back to the whole span exactly, without it they add back to within the ripple). Eigenvalue gains h_q(mu) on the coordinates: an AMPLITUDE
partition (gains sum to one at every mu), the constant (mu = 0) in the coarsest band.

**The atom's weighting of the cross-spectrum** (`spectral_flow`). Y_a = the atom's bins of every chunk, each column times g_a(f) [K x T B_a]; the weighted sum
C_a = Y_a Y_aᴴ [K x K] (a SUM over chunks and bins: forming it is allowed, rank at most min(K, T B_a)). With conj(C_a) = conj(Y_a) Y_aᵀ and Psi [S x K] the
coordinates' synthesis (each coordinate at unit star0 norm), G [S x 3 K] their tangent gradients at the vertices (`sharp(d0 Psi)`):
- the phase current per intrinsic edge e = (tail, tip) of the band pair (p, q): J_pq(e) = Im(Psi_tail diag(h_p) conj(C_a) diag(h_q) Psi_tipᵀ), which is
  Σ over the columns of Im(conj(z_p,tail) z_q,tip) with z_p = Psi diag(h_p) Y_a: `phaseCurrent`'s definition, so travel from tail to tip gives a NEGATIVE
  phase current (the forward transform's sign), and the direction of travel is -J;
- the vertex phase current (3 components) P_pq(v) = Im(Psi_v diag(h_p) conj(C_a) diag(h_q) G_vᵀ) (the golden's |w|^2 grad(theta));
- the power per vertex Re(Psi_v diag(h_p) conj(C_a) diag(h_q) Psi_vᵀ);
- the flow's Hodge-Dirac coefficients (`projectOneForm` on the stored edge rows), the exact potential and the coexact stream (`exactPotential`,
  `coexactStream`): on an open surface the potential is Neumann and the stream Dirichlet (recorded in the provenance).
The pairs sum to the whole field's values (the bands are an amplitude partition); the whole field (no eigenvalue gain) is stored beside them, so the
description "frequency atom x the flow's own modes" adds back exactly. Cheapest order: per atom one product C_a (K² T B_a), then per band q one product
R_q = conj(C_a) diag(h_q) [Psiᵀ | Gᵀ] (K² 4 S), then per pair only the entries needed (E K + 4 S K). Never Psi C Psiᵀ as a dense S x S matrix per pair.

**Time cells** (`atom_time_cells`). Per atom, per chunk: the atom's bins times g_a, the composed synthesis Psi diag(h_q) applied in the frequency domain on
the atom's bins only (S x K by K x B_a), then the inverse transform of length N per sensor; the overlap is dropped, leaving the chunk's own samples. The time
cells are the cells of the dyadic level of the time line whose width is the power of two nearest 2 sigma_t fs (sigma_t = 1 / (2 pi sigma_f), the atom's time
width): no windowing step, the atom decides its cells. Per cell: the sum over its samples of Im(conj(z_tail) z_tip) per edge, of Im(conj(z_v) grad z_v) per
vertex, of |z_v|^2 per vertex (no per-cell matrix). A cell beyond the span's last whole cell is left out (stated); cells never straddle chunks (the cell
width divides the chunk). The sums are a measurement of the per-sample products over the time cells (the measure `sum`): the products themselves (E x n per
atom and band) are never materialised beyond one chunk.

**The sign-flip statistic** (`sign_flip_consistency`, the golden's step 11). In: per cell s and element e (an edge: 1 component; a vertex: 3), the value
x_se. U_se = x_se / |x_se| (0 where |x_se| = 0); R1_e = |Σ_s U_se| / S. Null: `patterns` random sign vectors over the cells, drawn as numpy
`default_rng(seed).random((patterns, S)) < 0.5`, R1 recomputed per pattern (one matrix product: signs [patterns x S] by U [S x elements]). Out: R1 [elements],
p_e = (#(null R1_e >= R1_e) + 1) / (patterns + 1), the family-wise p_e = (#(null max over elements >= R1_e) + 1) / (patterns + 1), the threshold (the null
maximum's 95th percentile), the median R1 over elements, its null's 95th percentile and its p. Also stated: the measured correlation of neighbouring cells'
unit fields (Σ_e U_se · U_(s+1)e / elements, mean over s), the check on the exchangeability of single cells. The exchangeable unit is a group of
`cells_per_flip` consecutive cells (the definition states it).

**The aperiodic fit** (option). Each sensor's power per bin from the same chunk spectra: diag(Psi C_bin Psiᵀ) / T, the mean over the chunk partition. Fit
of log10 power against log10 frequency in the caller's range: offset - exponent log10 f, or offset - log10(knee + f^exponent) with a knee (grid search, as
the reference); 3 iterations dropping points above the fit by more than 2 standard deviations. Reference: nxr `nxr.spectral.aperiodic.fit`.

## A second pipeline from the same nodes: the whole-recording flow, no wavelets (`sensor_flow_bins`)

The flow of the whole recording per frequency bin (the net quadrature), with no atoms and no time cells. The SAME serving nodes (`signal.overlapped_spectrum`,
`signal.chunk_spectra`; or the stored `chunk_spectra` of an earlier run, found by name) and the SAME readout node `signal.spectral_flow`, with its frequency
weighting set to `every_bin` (each bin its own member, weight 1: C_k = Σ over the chunks of x_tk x_tkᴴ, the dense cross-spectrum eigenmode x eigenmode x bin
summed over every chunk) or to a SHARP set of bins the caller names (a partition of the frequency line: hard spans, weight 1 inside, 0 outside; `select_bands`).
Space: the whole field (every coordinate) and, when the caller names eigenvalue bands, the band-pair matrix. Outputs per bin (or per sharp set): power per
vertex, phase current per edge, its Hodge-Dirac coefficients (travel and rotation), potential and stream, the vertex phase current. Sizes: per bin one
K x K product of rank at most T (K² T per bin, about 2.6 GFlop for 3 277 bins); stored per bin x edge (767 x 3 277 x 8 bytes, 20 MB) and per bin x mode;
the band pairs multiply this by Q² and are only made when asked. Caveat stated with every result: whole-run cross-spectra keep the net quadrature only.

So that both pipelines use the readout node unchanged, its frequency weighting is a port taking any selection on the chunk spectra's frequency line whose
members carry weights (a kernel set: soft; a span set: sharp) or the parameter `every_bin`, and its eigenvalue bands port is optional (absent: the whole
field only). The atoms' normalisation is a parameter of the node (`frequency_normalisation`), meaningful for a soft set only (refused for a sharp set).

## Pipelines that read the same chunk spectra, beyond second order and across frequencies (graphs only; not built in this round)

Each is its own definition reading the stored `chunk_spectra` (never the recording again), with its own compute nodes. Notation as above; P = the frequency
pairs asked for.

**Phase-amplitude coupling** (`sensor_pac`). Nodes: `signal.atom_signals` (NEW, shared with the time cells: per chunk the atom's band signal z = Psi diag(h)
Y_a back in time, the overlap dropped; one atom at a time, never stored), `signal.phase_amplitude_coupling` (NEW: per sensor or per coordinate, the
mean-vector length or modulation index of a low atom's phase against a high atom's amplitude envelope, sums over the time cells as measurements: Σ |z_high|
exp(i arg z_low) and Σ |z_high|, the ratio derived at read), a surrogate null by circular shifts of whole chunks (REUSED surrogate machinery,
`docs/specs/2026-11-17-surrogates.md`). Stores: per (low atom, high atom) pair, per sensor, the complex sum and the amplitude sum. Cost: one band signal per
atom per chunk (S x N), the products S x N per pair. References: Tort et al. (2010) modulation index, Canolty et al. (2006) mean-vector length; nxr has no
port, MNE-connectivity none in core; Rheome to be checked. PI decisions: which coupling measure is the default; whether coupling is read per sensor, per
coordinate or as a flow (the high atom's phase current against the low atom's phase).

**Harmonics of non-sinusoidal waveforms** (`sensor_harmonics`). Nodes: `signal.cross_frequency_spectrum` (NEW: per chunk, the cross-spectrum between bin k
and bin n k, Σ_t x_t,k conj(x_t,nk)ᵀ for n = 2, 3, a K x K sum per pair, from the factored spectra with no inverse transform), its flow readout by the same
edge rule as `signal.spectral_flow` (J between the fundamental's and the harmonic's phase scaled by n). Stores: per fundamental bin (or sharp set) and
harmonic order, the coupling's power and phase per coordinate and the edge values. Cost: K² T per pair of bins. References: the n:m phase locking of
Tass et al. (1998); harmonic detection in specparam's residual peaks. PI decisions: the harmonic orders, and whether harmonics are reported as a coupling
or removed before the flow.

**Speed across frequencies** (`sensor_flow_dispersion`). No new compute: per broadband event (an event of the existing detectors, `global_bursts` or
`band_bursts`, a selection on the time line) the time cells of every atom that fall inside it (`signal.atom_time_cells`, REUSED) give each atom's phase
current and power; the apparent speed per atom (2 pi f power / |phase current|, derived at read) and the direction per atom compared across atoms: a NEW
statistic node `stats.cross_atom_consistency` (the circular spread of the direction across atoms per event, and the slope of speed against frequency, with a
sign-flip null over events). Stores: per event, per atom, the sums (measurements over the event's cells). PI decisions: which events, and whether "the same
speed" is tested as a constant speed or a constant wavelength (speed proportional to frequency).

**Waveform shape: the bispectrum and biphase in eigenmode coordinates** (`sensor_bispectrum`). Node: `signal.bispectrum` (NEW: per chunk and bin pair (k, l),
B_klm = Σ_t x_t,k x_t,l conj(x_t,k+l) for chosen coordinates, a third-order SUM; restricted to the coordinates and bin pairs asked, since the full object is
K³ B² and never formed), then the biphase (arg B) and the bicoherence (|B|² over the product of powers, derived at read). Peak-trough asymmetry is the real
part of the biphase at (f, f), rise-decay asymmetry its imaginary part (Bartz et al. 2019; Cole and Voytek 2017 for the time-domain cycle-by-cycle measures).
Stores: per coordinate (diagonal: k = l coordinates equal) and bin pair. Cost: B_pairs x K x T per chunk; the bin pairs limited to the fundamental band x
itself. References: the `bycycle` package (Cole and Voytek), Bartz et al. (2019) biphase. PI decisions: diagonal only (each coordinate with itself) or the
spatial bispectrum (pairs of coordinates), and which bins.

**The asymmetry of single events** (`sensor_event_shape`). Nodes: the existing event pipelines (`band_bursts`, `gabor_bursts`, `global_bursts`) give the
events as selections on the time line (REUSED); per event, the atom's band signal (`signal.atom_signals`) and a NEW `signal.cycle_shape` node (per cycle
inside the event: peak-trough and rise-decay ratios from the band signal's extrema, Cole and Voytek's cycle-by-cycle definitions), measured over the
event's cycles. Stores: per event, per sensor or coordinate, the sums of the ratios' ingredients. References: `bycycle`. PI decisions: whether the shape is
read on the broadband signal (bycycle's way, which needs the recording's samples in time: the chunk spectra's inverse transform over the whole span) or on
the atom's band signal (which removes the shape by construction for a narrow atom).

## Sizes and ranks (a 268-sensor helmet, 1200 Hz, a 300 s run: T = 11, B = 3277 bins below 60 Hz)

| intermediate | shape | size | rank |
|---|---|---|---|
| the serving operator (composed once) | K x channels = 268 x 268 | 0.6 MB | K |
| a chunk with its overlap, in coordinates | K x N = 268 x 65 536 | 140 MB double, transient | <= K |
| a chunk's spectrum (stored) | K x B = 268 x 3 277 complex64 | 7 MB; 77 MB for 11 | <= K |
| an atom's weighted columns Y_a | K x T B_a (4 Hz: 11 x ~150; 36 Hz: 11 x ~1 300) | <= 60 MB | <= min(K, T B_a) |
| an atom's weighted cross-spectrum C_a (a sum) | K x K | 1.1 MB | <= min(K, T B_a) |
| R_q per band | K x 4 S | 2.3 MB | <= rank C_a |
| a time-cell step: the synthesised band signal | S x N complex | 280 MB transient (one atom, one band, one chunk) | <= K |
| time cells stored | cells x (E + 4 S) per atom and band | about 0.5 GB for 17 atoms x 7 bands, float32 | |

The dense cross-spectrum (B x K x K, about 1 GB single precision) is never built on the CPU: per atom the weighted sum is formed directly from the factored
form (the same numbers; cheaper when the atoms are fewer than the bins). A GPU backend may build the dense form once and read every atom from it (the
Studio's choice); which order is cheaper is the executor's to decide from the sizes above, not the graph's.

## Checked against the Studio's chain

members (the serving operator: a linear spatial map, applied first) → the transform (the chunk's spectrum, overlap included) → the atoms as gains on the
frequency line and the eigenvalue line (a Meyer- or Morlet-like band of the chain's filter stage, here a weighting) → the field's forms (power, phase current,
the Hodge split: `fieldForms.ts`) → the measures (sums over the chunk partition and the time cells, folded across blocks). The linear map is read before any
non-linear step (D106); the two linear maps (the channel map and the analysis rows) are composed once into the serving operator.

## Three places, one graph

- **Command line** (a laptop or a server): `nxr-neuro run sensor_flow ...`; the CPU backend.
- **The Studio**: its typed WebGPU executor reads the stored `chunk_spectra` (nxr-neuro's served item) and runs `spectral_flow` and `atom_time_cells` as GEMMs
  and batched inverse transforms on the device; the statistics (sign flip, aperiodic) on the CPU.
- **A remote node with a GPU**: the same definition, the executor choosing the backend at the dispatch point.

What a GPU backend in nxr-neuro would need (not built now): (1) `Backend::gpu` in `dispatch.cpp` with a device implementation of the phases of
`signal.overlapped_spectrum` (a batched real FFT of K x N), `signal.spectral_flow` (complex GEMMs: Y Yᴴ, conj(C) diag(h) [Psiᵀ | Gᵀ], gathered edge entries)
and `signal.atom_time_cells` (a GEMM per atom and band, a batched inverse FFT of length N, per-cell segmented sums); (2) device-resident constants (Psi, G,
the edges' tail and tip, the gains) bound once in `prepare`; (3) a readback node gathering the measurements to the host for io; (4) the plan's size and cost
check choosing CPU or GPU per node from the sizes stated above. The node boundaries above already fit this: every node is a function of arrays, GEMM-shaped,
with no file, chunk loop or subject inside.

## Reused, changed, new

- Reused as they are: `sensor_surface`, `eigenmodes`, `select_kernels` (two new profiles), `signal.mode_analysis` (one new `modes` choice), the intrinsic
  forms (`phaseCurrent`, `projectOneForm`, `exactPotential`, `coexactStream`, `sharp`), `modeCrossSpectrum`, the chunk service's halo (the overlap).
- Not reused, and why: `spectral_coefficients` (its tiles have no overlap; an accumulate node can take context only to its right), `flowmap` /
  `signal.flow_map` (synthesises every cell's field on the surface, V x columns per cell, where the cheaper order forms C_a once; no band pairs, no
  normalised atoms, no vertex phase current), `crossspectrum` (Welch: windows and tapers), `speedspectrum` and `hodgepower` (source space, windowed).
- New: the serving per chunk with overlap, the chunk spectra statistic, the atoms' normalisation, the cos² and Mexican hat partitions as kernel profiles,
  the band pairs, the per-atom readout, the time cells, the sign-flip statistic with a numpy-compatible generator, the aperiodic fit.

## What the measurement cannot resolve

- The field on the helmet only: no source, no field spread, no cortical claim.
- The sensor surface's spacing is its resolution: patterns shorter than about two sensor spacings are not resolved.
- A whole-run cross-spectrum keeps only the net quadrature: motion that cancels over the run appears as standing (the time cells test it).
- The apparent speed is an upper bound.
- An incomplete Hodge-Dirac solve leaves coordinates and flow modes out: the field's energy outside the coordinates and the flow's energy outside the
  modes are stated with every result (issue filed).
- The phase current depends on the eigenvalue band (the band pairs show how much flow lies between bands): the bands are always stated with a result.

## Open questions (for the maintainers)

1. The 1-form the split reads by default: the edge phase current or the vertex phase gradient. Both are computed until it is decided.
2. How many consecutive time cells one sign flip groups, from the measured correlation of neighbouring cells.
3. Which eigenvalue weighting the time-cell test uses: each band alone (the golden's cells), the whole field, or both.
4. The empty-room recording as a sensor-space control: the pipeline runs on any recording by name; no control step is wired.
