# The flow verbs as definitions: quadrature, optical flow, cycle patterns, the spatial-band phase gradient (2026-11-17)

Status: `quadrature` is converted (definition `quadrature`, node `signal.source_planes`). The other three are designed here as graphs, checked against the Studio's chain,
and wait on the points under "What remains". Each graph is written first, as CONTRIBUTING.md asks: nodes, typed ports, the inputs it is handed, the outputs it gives back.

The Studio's chain (`studio/src/engine/toolbox/signal/chain.ts`, its device form `deviceChain.ts`, the field forms `fieldForms.ts`) is: members (a linear spatial map) -> filter
(zero-phase sections, a band, a Meyer or Morlet wavelet band) -> transform (none, envelope, wavelet power or complex, spectrum) -> the field's forms -> the measures. A linear map reads the
chain BEFORE any non-linear transform (D106). The names below are the Studio's where it has one.

## 1. quadrature (done)

```
recording [channels x samples, streamed]
  --signal.xpsd (accumulate; partition = the Welch segments, a Rule on the time line; channels_of = the kernel's)--> spectrum [bands x M x M, complex]
  --signal.source_planes (array; F = K L, S = L L^H; Im(F F^H) in factored form)--> energy [vertex x band], pairs [vertex x pattern (x 3)]
```
- Loading: the chunk service reads the recording; the cross-spectrum node is the only node on the stream. Preparation: the kernel's channels pick the rows (type-checked: the
  cross-spectrum must be bands x M x M on the kernel's M channels). Graph: two pure nodes. Readback: two statistics over the segment partition.
- No reduction is the node's own: the reduction over time is the cross-spectrum's (a statistic over the partition); the node is pointwise in sources (one member each), with no neighbourhood.
- Stored names are the importer's: `<name>_energy` (`quadrature_energy`) and `<name>` (`quadrature_pattern`). The `--tiles` summaries of the hand-wired verb are not part of the definition:
  a map's tile summaries are measurements over the tiles of the surface's grid (`run patches`, `run measure`).

### The grouped congruence norms and the disc neighbourhood take the resolved forms (spec 2026-11-17-selection-module, section 8)
- `quadrature::groupedCongruenceNorms` and `pairCongruenceNorms` have overloads taking `selection::MemberOfElement` (the member number of each row, the member count): the spatial tiles of a
  level are a GRID (a level of the surface's dyadic grid), resolved by `resolve_member_of`; an irregular set with disjoint members (a parcellation) takes the same form. `cellquadrature` calls them
  with the tiles' resolved form. The `std::vector<int> + count` signatures remain, the overloads being thin.
- `quadrature::weightedPhaseGradient` has an overload taking the disc as `selection::Elements` (one member: a `distance` selection, centre vertex plus radius, resolved by `resolve_distance`, which
  always holds the centre). The radius is no longer a parameter of the overload: the neighbourhood is the selection. The old signature builds the disc with the chart kit and calls the same code.

## 2. optical flow (designed)

```
recording [channels x samples, streamed]
  --signal.member_analytic (chunk, windowed; RETIMING to the frame grid's frames; halo = the member's taps, declared)--> z [channels x frames, complex]
  --meeg.apply_kernel (chunk, pointwise; linear map BEFORE the non-linear step, D106)--> J [3N x frames, complex]
  --signal.flow_activation (chunk, pointwise: amplitude |J|, the sum of squares over a source's orientations)--> A [N x frames, real >= 0]
  --flow.horn_schunck (chunk, stateful: the previous frame is its carried state; the amplitude scale is a window constant)--> velocity [3N x frames - 1]
  --flow.readout (chunk, pointwise per frame: divergence, curl, speed)--> d, c, s [N x frames - 1]
  --measure over (leaf grid x frame grid): sums per leaf and per frame, weighted by the vertex's area--> the pyramid's leaf x frame sums (measurements)
  --hodge flow weights (chunk, pointwise: the glued carry, then the modes)--> flow_modes [K x frames - 1] (a field on the eigen axis x the made line)
```
- (a) The neighbourhood is explicit. The flow's activation is the amplitude of an ANALYTIC signal made by a band node with its own declared halo (the wavelet's margin; `timebank::haloOf`
  for a Meyer band), never an FFT Hilbert over "the block passed in". Where a Hilbert of a real block is wanted, `signal::hilbertOwn(block, left, own)` returns the own columns of a block that
  carries a halo, and `flow::hilbertHalo(sfreq, lowestHz, cycles = 4)` is the halo a node declares (the Studio's `supportOf` / `fitLevelOf`: 4 cycles of the band's low edge). The result does not
  depend on the chunk size beyond the band's own response tail, which the halo bounds, and the planner reads the halo from the node instead of finding it in a chunk edge.
- (c) Every reduction is a measurement over a selection: the leaf sums are `measure` over the product (leaves, frames), a JOINT grid (the Studio's D129: first factor = frames, radix rounded to
  2^bits, code = (leaf << bits) | frame); the time means are derived at read from the stored sums and counts; there is no per-vertex time mean.
- (d) The band node's output carries its made line as a domain factor (types.md of this round): z lies on channels x frames, the frames being the frame grid's line (origin, rate, unit).
- The two-pass scale (Rheome scales the activation to unit amplitude over the WHOLE window) is not expressible in one chunk loop. It needs either a declared scale (a window constant the caller
  states) or a runner that runs the loop twice; see the decisions.

## 3. cycle patterns and the phasegradient spatial-band route (designed)

```
analytic cross-spectrum series (a statistic over the time tiles, stored by crossspectrum_analytic) + kernel (free) + Hodge-Dirac package (+ its forms)
  --hodge.kernel_forms (array: the kernel read into the modes' full forms: A2 [K x M] coexact normal part, A1 [K x M] tangent part)-->
  --spatial band (a grid: Weyl's bank of the ladder, one band per level, level l -> gain on lambda = mu^2; coupled to the surface's dyadic level by `weyl_level`)--> rows B [n x M]
  --patterns.leading (array: the q leading eigenvectors of B C B^H over the span, and per window)--> vectors, values, similarity (per window)
  --patterns.fields (array: stream function, potential, normal current on the glued surface)--> complex fields [vertex x pattern x quantity]
```
- A single band selects (item 11): the spatial band is a PARAMETER of one run, and a run has one band, as a time band is one band. The hand-wired verb's `weyl:all` loop over bands becomes
  one run per band (`--spatial-band weyl:J`), or a spatial-band axis on the output statistic.
- The window partition is the series' tile grid; the span is the sum over its tiles (a measurement over the partition), never a re-read.

## 4. Optical flow, built (2026-11-17): the nodes and the two definitions

The nodes are in `pipelines/src/flow_steps.cpp`, registered by `register_flow_nodes()`:
`signal.member_analytic` (retiming, windowed: one filter-bank member's full-rate analytic signal evaluated at the frames of a frame grid, input `tiling` made by `select.time_grid` rule frames; the halo declared = the member's taps; frame j starts at sample origin + j hop, so frames do not depend on the chunk), `signal.flow_activation` (J = K z, then |J|), `signal.flow_max_amplitude` (accumulate), `signal.flow_velocity` (stateful: the previous map; the unit amplitude is the stored
maximum), `signal.flow_hodge_modes` (pointwise), `signal.flow_series` (accumulate: the weights [K x steps] and their line).

- `run flow_amplitude` stores the window's maximum amplitude as a statistic; `run flow_modes` reads it by name (input `scale`) and stores the flow weights as a field on (eigen axis x the made time line).
- The frame cells are no longer made inside the run, and the leaf x frame sums (divergence, curl, speed, velocity components per leaf and frame) are NOT part of the definition: they are measurements over
  a grid (cortex x time), which is being redesigned as plain selections and measurements. They follow the grid. Until then the hand-wired `opticalflow` stays (its sums and tile summaries).
- Not built: `perscale` (one flow per spatial band: a band set); the bad segments are not excluded (as before); the line's start is relative to the recording's first sample (the recording's origin is not
  added: to do with the run's facts); the product output `modes_time` has no producing node (to check against the runner's declared outputs).

## 5. cyclepattern and the phasegradient spatial-band route: graph, not built

Both read the package's intrinsic complex and full forms (`kernel_forms`: A2, A1 [K x M]) and a spatial band (a gain on lambda = mu^2). Graph, as one definition each:

```
cross-spectrum series (windows) + kernel (free) + package (+ stored intrinsic complex, forms2)
  --hodge.kernel_forms (array: the kernel read into the modes' full forms)--> A2, A1 [K x M]
  --select.bands (a band SET: Weyl's bank of the ladder; one selection)-->
  --patterns.leading (array, once per band member and per window: q leading eigenvectors of B C B^H)--> vectors, values, similarity
  --patterns.fields (array)--> stream, potential, normal on the glued surface
outputs: ONE statistic whose axes include the band member (not one run per band)
```
New `Bound` services (generic, reading only): the stored intrinsic complex of a surface (d0, d1, edges, lengths, stars, angles) and a package's full forms (`forms2`). `kernel_forms` and the phase-lag split
are compute functions already (`nxr::intrinsic`, `nxr::hodge`). Not built in this round: the work waits on the grid redesign for the leaves of `phasegradient --spatial-band` and on the band-set output of rule 14.
