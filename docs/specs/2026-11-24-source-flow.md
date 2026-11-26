# The source mapping of the sensor-space flow (2026-11-24): the graph

> **Storage, since the flow consolidation (2026-11-24):** the per-member flow results named below as statistics are stored as MEASUREMENTS over the atoms and their products, and the time cells are not stored; see `docs/reference.md`, "How the flow results are stored".


**Status:** design, written before the code (CONTRIBUTING, "Adding a node or a pipeline", step 0). It plugs into the sensor chain of `docs/specs/2026-11-24-sensor-flow.md`.

## Why

The sensor-space flow describes the magnetic field on the helmet only: it says nothing about sources. To read the flow on the cortex, the same cross-spectrum is
carried onto the cortex through an existing minimum-norm kernel. The source step is that kernel PROJECTED ONTO THE CORTEX'S EIGENMODES and nothing else: a change of
coordinates, composed once. The lead field, the noise covariance, the regularisation and the depth weighting stay inside the kernel; no inverse is derived here and no
prior over modes is added. The kernel is always the caller's named input.

## The operator

Names (symbols only inside formulas): the minimum-norm kernel W [3 V x channels] (vertex-major, three Cartesian components per vertex); the cortex's coordinates Phi
[V x K_c] (the constant and the exact Hodge-Dirac modes' vertex parts, each at unit vertex-mass norm), M the vertex masses; the sensor surface's served field coordinates:
their analysis rows P_s [K_s x channels] and their synthesis back to the channels Q_s = M_s⁻¹ P_sᵀ [channels x K_s].

- **Option B** (`columns: channels`): the kernel in the cortex's mode coordinates, (I_3 ⊗ Phiᵀ M) W, [3 K_c x channels]. It acts on the channels (the chunk spectra are
  served through the identity on the kernel's channels). This is the reference's convention (its band kernels are Phi g_b Phiᵀ M W), so it is the default.
- **Option A** (`columns: sensor_coordinates`): option B times Q_s, [3 K_c x K_s]. It acts on the served sensor field coordinates (the sensor chain's chunk spectra
  as they are: no round trip back to the channels). A applied to P_s x is B (Q_s P_s) x: the two are equal when the sensor basis is complete, and otherwise differ by
  B (I - Q_s P_s) x, the part of the field in the coordinates the sensor solve left out; its share is stated with every option-A result.
- **Free orientation:** each Cartesian component is transformed as a scalar field and the three are summed in every reading: no tangent, normal or orientation prior.
- **The cortex: per hemisphere or glued** (`cortex: per_hemisphere | glued`, both named options to compare). Per hemisphere: each hemisphere's rows of W (its vertex
  range in the cortex, a selection) with that hemisphere's modes; the reference's choice, so the default. Glued: W's rows SELECTED by the glue's vertex map (the kept
  vertices' rows), the bridge rows zero (no current is estimated on the bridge), then the glued modes; the medial wall's rows have no counterpart and are dropped.
- **Cheapest order, built once:** (I_3 ⊗ Phiᵀ M) W first (3 K_c x V x channels per hemisphere, about 10 G multiply-adds for a 10 000-vertex hemisphere, 1 300 modes,
  268 channels), then x Q_s (3 K_c x channels x K_s, about 0.3 G). W Q_s first would add work for nothing.

## The graph

K_c the cortical coordinates, K_s the sensor coordinates (option A) or the channels (option B), V the cortex package's vertices, E its intrinsic edges, Q the cortical
bands, A the frequency members, T the chunks, B_a a member's bins.

### Loading (io)
- The recording, read only by io's chunk service with an overlap of half a chunk on each side, as the sensor chain.
- Stored nodes, read by name: the kernel (an inverse kernel operator), the cortex's Hodge-Dirac eigenvectors (each hemisphere's, or the glued cortex's; with their 1- and
  2-form parts), for option A the sensor surface's eigenvectors and vertex-to-channel map, the frequency atoms and the cortical eigenvalue bands (kernel sets).
- A hemisphere's package is handed over with its place in the cortex (the surface it is a component of and its first vertex), so a node can select the kernel's rows.

### Preparation (host)
- Type checks before any data: the kernel is free-orientation and lies on the package's input surface or on the surface that surface is a component of; the mode operator
  was made from this kernel (same channels); option A: the sensor surface's channels are the kernel's, its coordinates at unit mass; the chunk spectra's coordinates are
  the operator's columns (by their members); the kernel's selected rows are the package's vertices.
- Constants bound once: the mode operator, the readouts of the whole field and of each band, the sparse sharp operator and the factored Poisson systems of the cortex.

### The graph of pure nodes

| step | node | kind | in -> out | reused / new |
|---|---|---|---|---|
| before | `signal.mode_analysis` | array | cortex eigenvectors + kernel -> (I_3 ⊗ Phiᵀ M) W [3 K_c x channels] | REUSED; the kernel's rows now SELECTED (a hemisphere's range, or the glued rows with the bridge zero) |
| before (option A) | `signal.mode_analysis` | array | sensor eigenvectors + channel map -> the serving operator P_s | REUSED, as the sensor chain |
| before | `signal.source_mode_operator` | array | the mode operator, the kernel, the cortex package, (A: P_s and the sensor masses) -> the operator [3 K_c x K_s], the kernel's rows in the same columns [3 V x K_s], the serving operator | NEW |
| chunks | `signal.overlapped_spectrum`, `signal.chunk_spectra` | chunk, accumulate | recording + serving operator -> chunk spectra [T x B x K_s] | REUSED unchanged |
| after | `signal.cortical_flow` | array (statistic) | chunk spectra, cortex package, operator, kernel rows, frequency weighting, cortical bands -> per frequency member: whole field and every band pair (power, edge and vertex phase current, flow coefficients, potential, stream, the remainder's potential and stream, energies) | NEW (compute `bandReadouts`, `readoutPairFlow`, `OneFormRemainderSplit`; reuses `projectOneForm`, `exactPotential`, `coexactStream`, `synthesizeOneForm`) |
| after | `signal.cortical_time_cells` | array (measurement over time cells) | the same inputs + atoms -> per atom x readout (the whole field, then each band) per time cell: edge and vertex phase current, power | NEW (compute `readoutTimeCellFlow`) |
| after | `stats.sign_flip_consistency` | array (statistic) | the cells -> R1 and its sign-flip null | REUSED unchanged |

### Readback (io)
- `<side>_flow_vertices`, `_flow_edges`, `_flow_modes` (whole field) and `<side>_band_pair_*` (band pairs) per hemisphere (`left`, `right`) or for the glued cortex,
  measurements over the frequency members (x band x band); `<side>_time_cells`, `<side>_sign_flip`. The chunk spectra are stored as in the sensor chain.
- Derived at read, never stored: travel and rotation shares, the remainder's share (its star1 energy over the flow's), the apparent speed (an upper bound), the
  power-weighted phase gradient, the potential and stream of the whole flow (the modes' part plus the remainder's).

## The maths of each new node

**Band readouts.** U_q = (I_3 ⊗ Phi diag(g_q)) times the mode operator for q below the last band, [3 V x K_s] (component-major); the LAST band is the complement,
W Q_s - the sum of the others: its own gain and every content outside the kept modes. The bands add back to the kernel exactly; the whole field's readout is the kernel
itself. Formed once per run (compose first: Phi diag(g_q) [V x K_c] times the operator's K_c rows per component; never the V x V filter).

**The readings of a frequency member.** Y_a = the member's weighted columns of the chunk spectra [K_s x T B_a]; C_a = Y_a Y_aᴴ is a sum and stays in the K_s
coordinates: it is never transformed to 3 K_c x 3 K_c or 3 V x 3 V. Per band q one right factor R_q = conj(C_a) U_qᵀ [K_s x 3 V] (dense: conj(C) applied to the real
readout as two real products) or conj(Y_a) (Y_aᵀ U_qᵀ) when 2 T B_a < K_s (factored); per pair (p, q) only the entries read, summed over the three components:
- power per vertex Re(U_p,v R_q[:, v]); phase current per edge Im(U_p,tail R_q[:, tip]) (travel from tail to tip gives a negative value);
- phase current per vertex Im(U_p,v conj(C) (grad U_q)_vᵀ), with the gradient the sparse sharp operator S [3 V x E] on the pair's entries along the edges:
  sum over the edges at v of S(v, e) (Im Z(v, tip_e) - Im Z(v, tail_e)), Z(a, b) = U_p,a conj(C) U_q,bᵀ; no gradient readout is formed;
- the flow coefficients: `projectOneForm` on the cortex's STORED edge rows; potential sum (coefficient / mu) P_k, stream likewise (`exactPotential`, `coexactStream`):
  the split is a filter in eigenmodes and equals a full solve when the basis is complete;
- the remainder: the phase current minus its synthesis from the coefficients. Only it is split by sparse Poisson solves: its potential from
  d0ᵀ ⋆1 d0 alpha = d0ᵀ ⋆1 r and its stream from d1 ⋆1⁻¹ d1ᵀ (⋆2 beta) = d1 r (Neumann potential and Dirichlet stream on an open surface; the hemispheres with their
  medial wall and the glued cortex are closed). It belongs to the complement of the flow's own scale; its star1 energy is stored beside the flow's.

**Time cells.** Per atom and chunk, the atom's spectrum is inverted on the columns' side (K_s rows, one inverse transform), the chunk's own samples kept; per cell
z = U Y_cell [3 V x cell samples] and the sums of Im(conj(z_tail) z_tip) per edge, of |z_v|² per vertex, and the vertex phase current S times the edge sums (within one
readout Im(conj(z_v) z_v) = 0). The cells are the atom's own (as the sensor chain); nothing per sample is kept beyond one cell.

## Sizes and ranks (a 10 242-vertex hemisphere, 1 300 cortical coordinates, 268 channels, 17 atoms, 9 bands)

| intermediate | shape | size (double) | rank |
|---|---|---|---|
| the mode operator | 3 K_c x K_s = 3 900 x 268 | 8 MB | at most K_s (option A: the sensor coordinates kept; B: the kernel's rank) |
| a band readout U_q | 3 V x K_s = 30 726 x 268 | 66 MB (10 readouts: 660 MB) | the same bound |
| a member's cross-spectrum C_a (a sum) | K_s x K_s | 1.1 MB | at most min(K_s, T B_a) |
| a right factor R_q | K_s x 3 V complex | 132 MB, one at a time | at most rank C_a |
| a time cell's band signal | 3 V x cell samples complex | 0.5 MB per sample column | at most K_s |

Work: per member and band 2 K_s² 3 V real multiply-adds (about 4.4 G), per pair 3 (V + 2 E) K_s complex entries; for 17 atoms x 9 bands per hemisphere about
0.7 T multiply-adds (CPU minutes). The time cells: 3 V K_s per sample per readout, about 0.5 T multiply-adds per chunk, atom and band (a GPU's work).

## Checked against the Studio's chain

members (the serving operator: a linear map, applied first) -> the transform (the chunk's spectrum) -> the atoms as gains on the frequency line -> the source operator
and the cortical band gains (linear maps, read BEFORE the non-linear products: the cross-spectrum is weighted in sensor coordinates and the readout applied to its
factors, A C Aᴴ = (A y)(A y)ᴴ) -> the field's forms on the cortex (power, phase current, the Hodge split) -> the measures (sums over the chunk partition and the time
cells). The two linear maps of the source step (the kernel and the analysis onto the modes) are composed once.

## Three places, one graph

The command line (CPU), the Studio's WebGPU executor (inputs: the chunk spectra, the operator, Phi, the gains, the edges; U_q built on the device per band) and a node
with a GPU run the same definition; every node is a GEMM or a gathered dot product, and the executor picks the backend. The definition does not.

## Reused, changed, new

- Reused unchanged: the sensor chain's serving nodes, `stats.sign_flip_consistency`, `projectOneForm`, `exactPotential`, `coexactStream`, `synthesizeOneForm`,
  `cosineSquaredPartition` and the kernel-set readers.
- Changed: `signal.mode_analysis` selects a hemisphere's or the glued cortex's rows of the kernel (it had refused a hemisphere and blended the bridge); the glued carry of a
  kernel (`gluedKernel`) is a selection with the bridge rows zero.
- New: `signal.source_mode_operator`, `signal.cortical_flow`, `signal.cortical_time_cells`; compute `selectVertexRows`, `synthesisFromAnalysis`, `bandReadouts`,
  `sharpOperator`, `readoutPairFlow`, `readoutTimeCellFlow`, `OneFormRemainderSplit` (since merged with the sensor and connectome copies into `flow_readout.h`: `pairFlow`, `timeCellPairFlow`, `flowSplit`; `sharpOperator` is in `intrinsic_forms.h`).
- Not built here: storing the composed operator as an operator of the dataset (it is bound per run), and carrying the cortical maps to a template (a selection of rows by
  a registration match at readback).

## What the measurement cannot resolve

- It is the kernel's estimate in other coordinates: its point spread, depth bias and field spread are inherited. Apparent flow at cortical wavelengths the kernel cannot
  resolve is leakage, not travel; the kernel's resolution in mode coordinates is the diagnostic.
- Rank at most the sensor coordinates (option A) or the kernel's rank (option B), whatever K_c is.
- Wavelengths shorter than the last kept mode appear only in the complement band; the flow beyond the kept modes only in the remainder.
- Per hemisphere: no flow across the midline. Glued: no medial-wall sources.
- The three components are summed: no claim about the normal or tangent direction of the current. The apparent speed is an upper bound.
