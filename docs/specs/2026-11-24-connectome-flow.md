# The flow through the fibre network and its connectome eigenmodes (2026-11-24): the graphs

> **Storage, since the flow consolidation (2026-11-24):** the per-member flow results named below as statistics are stored as MEASUREMENTS over the atoms and their products, and the time cells are not stored; see `docs/reference.md`, "How the flow results are stored".


**Status:** design written before the code (CONTRIBUTING, "Adding a node or a pipeline", step 0), then built: `tile_network` and `connectome_flow`. A port of nxr's
flow through the fibre network (`staging/bursts/analysis/flownet.py`, with `nxr.measure.tile_connectome.surface_ladder`): its design and names are the starting point;
every deviation is listed below with its reason. The combined geometric-connectome eigenmodes of nxr (`flowmodes.py`) are a graph only here (last section).

## Why

Activity measured on the cortex can be coordinated between regions that are far apart. Whether part of that coordination travels ALONG the white-matter fibres (a lag
that grows with the bundle, specific to joined regions, repeatable) is a question the surface flow cannot answer: the surface flow moves between neighbouring vertices.
The fibre network gives the long-range routes; this analysis reads the phase relation of the source activity on those routes, with a control of regions at the same
distance that no fibre joins, and describes the network-wide pattern in the connectome's own eigenmodes.

## Notation and sizes (a 268-channel helmet, 1200 Hz, a 300 s run; a cortex of 15 002 vertices; 10^6 streamlines; tiles of level 5: 32 a hemisphere)

T = 64 tiles, P = T (T - 1) / 2 = 2016 tile pairs, C = 268 channels, K = 268 field coordinates of the served spectra, V = 15 002 vertices, t = 11 chunks,
B = 3 277 bins of [0, 60) Hz, A = 17 atoms.

## The graph: the fibre network on the tiles (`tile_network`)

| part | piece | in → out |
|---|---|---|
| loading (io) | the fibre bundle (a curve, fibre-major points), the end surface (the kernel's cortex), the dyadic tiles of that surface (a grid selection: `select_dyadic_surface`) | by name |
| preparation | the level of the tiles from a stated quantity (`parts 32` per hemisphere: level 5); the grid lies on the end surface (one label per vertex) | refused with the reason before any data |
| graph | `meeg.fiber_ends` → `meeg.nearest_vertex` → `meeg.end_gate` (REUSED, the `connectome` definition's); `meeg.fiber_lengths` (NEW); `meeg.tile_pairs` (NEW); `meeg.laplacian_eigenmodes` (NEW) | fibres, surface, tiles → each tile pair's count and median length, each tile's centroid and piece; the connectome Laplacian's eigenmodes |
| readback | three statistics: `<name>_tiles`, `<name>_pairs`, `<name>_eigenmodes` | |

- **the tiles** (`meeg.tile_pairs`): the grid's finest code of each vertex shifted to the stated level (`code >> (depth - level)`); a tile is a code some vertex
  holds. nxr's `surface_ladder` and the grid maker port the same ladder (equal-mass halving across the geodesic diameter, every tile one piece), so the codes agree:
  2^level x piece + tile.
- **the pairs**: every pair of tiles (i < j, in code order) is an element; for each, the streamlines whose two ends lie in the two tiles (each end the nearest
  vertex of the end surface; an end outside the tiles, or both ends in one tile, is left out), their count and their median length; the Euclidean distance of the
  tiles' centroids (the mean position of their vertices) and whether they lie on different pieces (across the hemispheres). The EDGES of the network are a
  selection of the pairs: count >= `min_streamlines` (nxr: 10), made at read and by the nodes that need it, never a separate stored array.
- **the connectome Laplacian** (`meeg.laplacian_eigenmodes`): L = D - W on the tiles, W the edges' streamline counts (symmetric), D its row sums; ALL its
  eigenmodes by one dense symmetric eigensolve (T x T: the tile graph is small), eigenvalues ascending. Its null space has one mode per connected component of the
  network (stated: eigenvalues below `1e-10 x the largest`). These are the connectome eigenmodes of the reference.
- **a fibre's length** (`meeg.fiber_lengths`): the sum of its segments' lengths, in the bundle's unit (SCS metres), so a speed is metres per second without a factor.

## The graph: the flow through the network (`connectome_flow`)

| part | piece | in → out |
|---|---|---|
| loading (io) | the stored chunk spectra (`sensor_flow`'s `<name>_chunk_spectra`: the cross-spectrum in its factored form); the atoms (a kernel set); the sensor surface's eigenvectors and channel map (to make the serving operator again); the inverse kernel (by name: the caller's, free orientation); the tiles; the network's `<name>_pairs` and `<name>_eigenmodes` | by name; never the recording |
| preparation | `signal.mode_analysis` (REUSED, as `sensor_flow` binds it: the serving operator, `exact_and_constant`, `unit_mass`); `signal.tile_operator` (NEW): the kernel averaged over each tile's vertices, composed with the synthesis of the served coordinates back to the kernel's channels: ONE operator [3T x K], built once | the spectra's coordinates must be the serving operator's rows; the kernel's channels must be the serving operator's channels; the kernel's surface the tiles' |
| graph | `signal.tile_flow` (NEW): the whole run's tile cross-spectrum per atom (the sensor chain's form, for comparison); `signal.tile_time_cells` (NEW): the same sums per cell of a level of the time line (the reference's 4 s segments by default); `stats.network_sign_flip` (NEW, the reference's nulls); `meeg.network_flow` (NEW): divergence, graph potential, its connectome-eigenmode coefficients, the gradient part's sums; `stats.run_replication` (NEW, definition `connectome_flow_replication`) | spectra, operator, atoms → per atom: pair sums, tile power; per cell: pair sums; per pair: mean edge flow, p, family-wise p, control, phase; per tile and mode: divergence, potential, coefficients; per atom: the runs' correlations |
| readback | `<name>_tile_flow`, `<name>_tile_cells`, `<name>_sign_flip`, `<name>_network_flow` (statistics) | derived at read: edge flow, lag, speed, shares, counts over the edges and the control |

### The maths

The served chunk spectra are y_t(k) [K] per chunk t and bin k. The tile operator M [3T x K] gives each tile's current, three Cartesian components (no orientation
assumed): z_t(k) = M y_t(k).
- **the whole-run tile cross-spectrum per atom a**: S_ij = Σ_t Σ_k g_a(k)² Σ_c z_t,(i,c)(k) conj(z_t,(j,c)(k)), the components traced (the reference's
  `trace(axis1=1, axis2=3)`). Stored: Re S_ij and Im S_ij per pair (i < j), S_ii per tile.
- **the edge flow** (derived at read): F_ij = Im S_ij / sqrt(S_ii S_jj), the signed imaginary coherency: > 0 when the activity of tile j lags that of tile i (the
  phase flows i → j); zero-lag leakage contributes nothing to it. Its sign convention is the reference's (z_i conj(z_j)); the sensor chain's edge phase current
  Im(conj(z_tail) z_tip) is its negative for the pair's orientation.
- **the lag and the speed** (derived at read): lag = -arg(S_ij) / (2 pi f_a), f_a the atom's centre; speed = median length / |lag| on the consistent edges. An upper
  bound: leakage pulls lags towards zero.
- **the cells** (`signal.tile_time_cells`): per atom and chunk, the atom's band signal back in time: the tile operator composed with the atom's gains on its
  bins FIRST (M y g_a: [3T x the atom's bins]), one inverse transform per row, the chunk's own samples kept (the overlap dropped), then per cell the sums
  Σ_c Σ_samples z_i conj(z_j) for every pair (real and imaginary parts) and |z_i|² per tile. The cells are a level of the time line anchored at the run's first
  sample; a cell cut by a chunk's edge is added up from the two chunks and only whole cells are kept. `cells segments` (DEFAULT, the reference's): non-overlapping
  segments of `segment_seconds` (4 s), the same for every atom, S = N // L; `cells atom_width`: each atom's own cells (the sensor flow's). The per-cell sums over all
  pairs are one product of [T x cell] blocks per component (every pair is an element); no matrix of a single sample is formed.
- **the nulls** (`stats.network_sign_flip`, the reference's, ported exactly; numpy's generator bit for bit, one for the run, drawn in the reference's order): per atom,
  each cell's edge flow F_s = Im C_ij / sqrt(C_ii C_jj) for every pair (kept in float32, as the reference's array); the mean edge flow Σ_s F_s / S; the sign flip
  (time reversal) over the cells: `patterns` sign vectors random((patterns, S)) < 0.5 → -1, p = (#(|null| >= |observed|) + 1) / (patterns + 1), and the
  family-wise p over all pairs on the sums normalised by Σ_s |F_s| (the null maximum); then the CONTROL: for each distance bin (np.digitize over
  `distance_bin_edges`, 0 30 45 60 80 100 130 170 250 mm) the joined pairs occupy, as many not-joined pairs of that bin as joined ones, drawn by numpy's
  `choice(replace=False)`; the phase arg Σ_s C_ij. Checked against numpy and the reference's code on generated values (`test_network_flip`: every p, family-wise p
  and control flag equal).
- **the replication** (`stats.run_replication`, the reference's `netfig`): per atom Pearson's r of two runs' mean edge flows over the joined pairs and over the
  not-joined pairs.
- **the network** (`meeg.network_flow`), per atom, on the edges (count >= `min_streamlines`), weights w = their counts, B the incidence (edge e = (i → j):
  -1 at i, +1 at j):
  - divergence d = Bᵀ F (per tile; the reference's sign: the flow into a tile minus the flow out of it);
  - the graph potential phi: the weighted least-squares fit of F by a gradient, Bᵀ W B phi = Bᵀ W F, i.e. L phi = Bᵀ W F with L the connectome Laplacian above.
    In the connectome's eigenmodes (complete: the dense solve) this solve is a FILTER: phi = Σ_k (u_kᵀ Bᵀ W F) / lambda_k u_k over the modes with lambda_k > 0, the
    null modes left out (the minimum-norm solution, the reference's `lstsq`). Stored: phi per tile and its coefficients u_kᵀ phi per mode;
  - the gradient part's sums: Σ_e w_e (B phi)_e² and Σ_e w_e F_e² (their ratio, the gradient share, derived at read); the energy of phi over the connectome's
    eigenvalues (the reference's `(Uᵀ (phi - mean phi))²`) derived at read from the coefficients.

### Compose first; the cheapest order (sizes and ranks)

| object | shape | size | how |
|---|---|---|---|
| the kernel | 3V x C | 45 006 x 268, 96 MB double | read once; never applied to data |
| the tile averaging | 3T x 3V | sparse, 3V entries | applied to the kernel's rows FIRST: [3T x C] |
| the synthesis of the coordinates back to the channels | C x K | 268 x 268 | the pseudo-inverse of the serving operator (its rank stated: K when complete) |
| **the tile operator** M | 3T x K | 192 x 268, 0.4 MB | (averaging x kernel) x synthesis: two products at the smallest size, once |
| a chunk's tile coefficients | 3T x B | 192 x 3 277 complex, 10 MB | M Y_t, once per chunk, shared by every atom (whole run) |
| the whole-run tile cross-spectrum per atom | T x T | 64 x 64 complex | Σ over chunks and the atom's bins of three [T x bins] products (a sum: a matrix that holds what vectors do not) |
| an atom's band signal per chunk | 3T x 32 768 | 100 MB complex double | (M diag over the atom's bins) Y_t, one inverse transform of 65 536 per row, in blocks of rows |
| per cell, every pair | T x T per cell | | three products [T x cell] [cell x T] per cell (every pair is an element) |
| the stored results | atoms x (2P + T); cells x (P + T); atoms x (3P + 7); atoms x (3T + 2) | a few MB | |

Nothing is resampled, windowed, tapered or padded: the spectra are the sensor chain's (each chunk transformed once with its overlap); the atoms are the taper.

### Checked against the Studio's chain

members (the tile operator: a linear spatial map, composed once) → filter (the atoms as gains on the frequency line) → transform (none: the stored chunk spectra are the
transform) → the forms (products z_i conj(z_j): the non-linear step, after every linear map) → the measures (sums over the chunks and bins, or over the cells).
The network step reads the sums only: a function of the reduced quantities (a derived measure), as the sensor chain's Hodge-Dirac flow coefficients are.

## Deviations from the reference, and the decision behind each

| the reference (nxr flownet) | here | why |
|---|---|---|
| the recording decimated to 300 Hz, the whole run transformed at once | the stored chunk spectra at the native rate | no resampling (2026-11-23, restated 2026-11-24); recordings streamed one chunk at a time, the cross-spectrum stored factored (2026-11-24) |
| 4 s segments: the sign-flip unit and the mean of per-segment normalised flows | the same by default (`cells segments`, 4 s at the native rate, from the run's first sample, whole segments only); the atoms' own cells an option (`cells atom_width`), and the whole-run ratio of sums (`signal.tile_flow`) stored beside it | nulls are ported exactly; where atoms and time cells would change a null, the reference's behaviour is the default and the difference is reported (2026-11-24) |
| its own flip and its distance-matched control, one numpy generator | the same (`stats.network_sign_flip`), bit for bit | ported exactly (2026-11-24); not the sensor chain's `stats.sign_flip_consistency` (R1 of the signs), whose statistic differs |
| Gaussians at 4 x 2^(k/5) Hz with standard deviation f / 12, unnormalised | the flow pipeline's atoms (the same centres and widths), normalised by the sum of their squared gains | the frequency grid = the flow pipeline's atoms (2026-11-24) |
| the tile current: the kernel averaged over the tile, applied to the decimated recording | the same average composed with the synthesis of the served coordinates: one [3T x K] operator | compose first (2026-11-24) |
| the edge flow the network reads: the mean over segments of the per-segment ratios | the same by default (`meeg.network_flow` reads the nulls' mean edge flow); the whole-run ratio an input it also reads | the reference's behaviour as the default |
| phi by `lstsq` on the weighted incidence | the 1/eigenvalue gain on the connectome's complete eigenmodes | Poisson solves are filters in eigenmodes (2026-11-24) |
| the kernel's long wavelengths removed (`--keep`) | not built | the source mapping's band readouts (source flow) compose with the tile averaging when they land |

## What the measurement cannot resolve

- A tile's current is the kernel's mean over its vertices: activity of opposite sign within a tile cancels; the tile's size sets the network's scale.
- Tractography ends are not cortical terminations (streamlines stop at the grey/white boundary); the nearest vertex is a location on the surface, not a depth.
- Leakage of the inverse correlates neighbouring tiles at zero lag: the imaginary part is blind to it but lags are pulled towards zero, so speeds are upper bounds.
- A consistent lag between two joined tiles does not show that the fibres carry it: the distance-matched not-joined pairs are the control; a common drive with
  different delays gives the same signature.
- A whole-run cross-spectrum keeps the net quadrature only; the time cells test its consistency.

## The combined geometric-connectome eigenmodes (nxr `flowmodes.py`): graph only, not built

Whether the flows on the cortex are described more compactly by the geometry's eigenmodes, the connectome's, or both: L_geo + w L_conn (the cotan Laplacian per
hemisphere, the vertex connectome's graph Laplacian from the fibres' end vertices, w with w tr(L_conn) = mix x tr(L_geo)), a generalised eigensolve with the lumped mass,
and the share of a travel potential's energy captured by the first K modes of each basis. Pieces: `meeg.end_connectome` (REUSED, gate none, no smoothing) → a node
`connectome.combined_laplacian` (NEW) → the eigensolve the `eigenmodes` definition runs (REUSED, given an operator) → a readout of the sensor or source flow's stored
potential (the measurement system). Waits for the source flow's cortical potential.
