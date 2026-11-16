# Direction and timing of a band's activity at a spatial scale (spec, 2026-11-12, revision 3; APPROVED by the maintainers; amended 2026-11-13, below)

## Why

The question is which way activity moves across the cortex, at which spatial scale, and in what order places are reached. Exploratory experiments of 2026-11-12 on synthetic and real recordings pointed to:

- **Scale:** at the mesh's own scale the lags are tiny; lag content lives at coarser scales. A spatial band is chosen, as a time-frequency band is.
- **Sign:** a fixed (normal) kernel flips sign across folds. The free estimate does not, and the maintainers ruled that direction analysis makes NO orientation
  assumption.
- **Graph:** the magnetic Laplacian needs a graph (edges, length, weights), and that choice changes the answer.

The mean **phase current** of the band-filtered free current answers the direction question without those choices. Read in the subject's Hodge–Dirac
modes, the same field also gives the **timing map** and its **consistency**, with no further solves.

How it relates to the magnetic Laplacian (approved spec `2026-11-12-magnetic-laplacian.md`): both summarise the same averaged lags.

- **The phase current** is the lags at vanishing edge length.
- **The timing map** fitted to it is, for small lags, the map the magnetic Laplacian fits.
- **Its unexplained part** plays the role of the frustration λ₁.

The magnetic Laplacian stays the tool for large (wrapping) lags and for pairs that are not neighbours (interhemispheric edges). This spec does not
replace it.

## What the readout means

For one FFT segment (or one analytic sample) X, the source field z_v = k_v X has the phase gradient ∇φ = Im(z̄ ∇z) / |z|². Phases cannot be averaged
over segments (they rotate), but z̄ ∇z cancels the common rotation and is bilinear in X. Its average is therefore a quadratic form of the cross-spectrum
C = ⟨X Xᴴ⟩:

- **P = ⟨|z|²⟩:** the power;
- **H = Im⟨z̄ ∇z⟩:** the phase current;
- **Q = ⟨|∇z|²⟩:** the gradient power;
- **H / P = Σ_s |z_s|² ∇φ_s / Σ_s |z_s|²:** the power-weighted mean of the segments' phase gradients.

These are the typical direction of phase travel over the cross-spectrum's span: strong segments count more, and opposite directions cancel.

**Two consistencies**, different questions:

| measure | asks | 1 (or 0) means |
|---|---|---|
| travelling index \|H\|/√(PQ) (exists) | is the direction the same over TIME? | 1: one wave in every segment; 0: standing, or segments cancelling |
| the coexact share of the phase field (new) | do the averaged directions fit one timing map over SPACE? | 0: one timing map explains every lag; > 0: rotation, overlapping patterns |

## What exists (ported, tested)

| part | where | what it gives |
|---|---|---|
| the band's sensor cross-spectrum, stored | `csd compute`, `csd_read` | C = S + i Sq per time-frequency band |
| the phase sums | `nxr::meeg::phaseSums` (`meeg/wave.h`; nxr's `PhaseReader`) | P, H, Q per vertex, summed over signals; readouts H/P, velocity, speed, travelling index |
| the verb | `phasegradient compute` | the sums and readouts of one kernel and one stored C |
| the fused kernel | `nxr::hodge::hodgeWeights` | a free kernel read in the Hodge–Dirac modes, B [2K × channels] |
| band gains | `nxr::filterbank` (`weyl`, `design`) | a member's gain on λ = μ² |
| synthesis | `nxr::hodge::hodgeSynthesis` | weights back to the current on the glued surface; and the potentials A, S |
| a vector field read in the modes | `nxr::hodge::WeightsReader::flow` (as `opticalflow` reads its velocities) | exact and coexact coefficients of a tangent field |
| the glue, the modes | `eigen solve` (PR #101) | the subject's glued cortex and its 2,941 Hodge–Dirac modes |

## What is missing

1. **The spatial band as an input** to the phase sums: the band's current kernel J_b = synthesis(G_b · B). It is a kernel like any other; the phase
   sums take it unchanged.
2. **The modes reading of the phase field,** and the scalars read from its coefficients:

   | scalar | made from | meaning |
   |---|---|---|
   | **timing map** φ = −A, A = Σ_exact b_k P_k | the exact coefficients | the field's gradient part is −∇A; τ = φ / ω, with ω at the band's centre. A larger φ LEADS (z ∝ e^{+iωt}), so propagation runs from high φ to low |
   | **stream function** S = Σ_coexact b_k P_k | the coexact coefficients | extrema: vortex cores |
   | **divergence**, **curl** | the same sums, each b_k × μ_k² | the sources and sinks of propagation; its rotation |
   | **consistency** | the coexact energy, in total and per vertex | the part no timing map explains |

   There is no harmonic part: the glued cortex is one closed surface. The eigensolve already did the Poisson-type work; reading a field is one K × K
   system, factored once (`WeightsReader`).

## The graph

**Loading (io):** the stored band cross-spectrum (`csd_read`, on the kernel's channels in its order), the stored free kernel, and the stored Hodge–Dirac
package with its glued surface. No recording is read: `csd compute` read it chunk by chunk (one chunk by default, `--chunks all` the whole), and this
route inherits its span.

**Preparation (host):**
- type checks: a FREE kernel [3V × channels] (a fixed one refused); C covers the kernel's channels; the package is on the kernel's surface; the band
  lies within the modes' range (a band past the last mode is flagged as truncated);
- constants bound once: the fused kernel B, the band gains g_b, the glued mesh's gradient operators, the `WeightsReader`.

**The graph, pure nodes:**

```
B ──bandGain(g_b)──► G_b B ──synthesize──► J_b [3V × M] ──┐
                                                           ├──phaseSums (x, y, z of J_b and their gradients; S, Sq)──► P, H, Q   (pointwise sums)
C = S + i Sq ──────────────────────────────────────────────┘
```

Every output is pointwise. The sums are linear in C, so they are time sums over the cross-spectrum's own time partition: the cardinal rule holds.

**Readback:**
- stored as Fields on the glued surface (the bridge masked): `power` P, `phaseCurrent` H (vector3), `gradientPower` Q;
- derived at read, never stored as ratios: H/P, velocity, speed, the travelling index;
- per region (DK, tiles) through the measure system: Σ H, Σ P, Σ Q, with ratios derived;
- **the modes reading of the phase field:** an analysis of the stored sums, not a sum.
  - **Stored:** its coefficients, the exact b and the coexact b, as Fields on the eigen axis (as `hodgepower` stores per mode), with their provenance.
  - **Synthesised at read:** the timing map, stream function, divergence, curl and consistency.
  - **Never added** over windows or regions: for windows, sum P and H first, then read.

## Which field is read

| option | read | behaviour |
|---|---|---|
| `field=gradient` (default) | the mean phase gradient H/P, an unweighted projection on the modes (diagonal: free) | `opticalflow` reads its velocities the same way. Weak places count as much as strong ones |
| `field=current` | the phase current H with weight P: the potential minimising ∫ P \|∇φ − H/P\|², a weighted solve on the exact modes (≈ 1,500 × 1,500, per band) | the magnetic Laplacian's weighted fit in field form. For ANY single coherent pattern z(u, t) = s(t) m(u), the timing map explains everything (consistency 0); vortices and overlapping patterns give > 0. The unweighted reading has no such guarantee |

The stream function, divergence and curl come from the unweighted reading in both cases; only the timing map and the consistency differ.

## What it cannot resolve

These go into `producer_json.warnings` and `docs/reference.md`.

- **Kinematic:** the phase's movement, not the current's.
- **A time average** over the cross-spectrum's span: directions that change cancel. The travelling index says how much; windows resolve it (next).
- **Linear in the lags:** wrapping is not handled (the magnetic Laplacian handles it).
- **Band-limited:** the finest mode is 2.1 cm. The coexact vertex potential S is a face-to-vertex average, approximate at the finest scale.
- **Polarisation:** for an elliptically polarised current whose axes turn across space, Im⟨J, ∇J⟩ carries a geometric term that no timing map
  explains, so the consistency is > 0 without a timing inconsistency. The free estimate is close to linearly polarised at the mesh's scale (0.97–0.98,
  experiment 9); the term's size at band scales is to be measured.
- **The bridge** is not cortex: masked in every readout.

## The command

```
nxr-neuro phasegradient compute --dataset D/N --subject S --csd NAME --kernel NAME [--package STEM]
                                [--spatial-band FAMILY:MEMBER | --spatial-band all] [--field gradient|current] [--name STEM] [--replace]
```

- **`--spatial-band`:** a member of a graph filterbank on the package's μ², e.g. `weyl:5` (Weyl's Meyer octave j = 5), or `all`. Without it, the kernel
  is used as it is (today's behaviour) and the modes reading is skipped unless `--package` is named.
- **`--field`:** the timing fit: `gradient` (default) or `current`.
- Every option goes into the provenance and the cache key.

## Options and defaults

Defaults reproduce the references; everything new is a named option (the method rule). No default of an existing verb changes.

- **The signal:** this route passes `--signal ambient` (the free current's x y z components, no orientation assumed: the maintainers' ruling of 2026-11-12).
  The verb's default (`normal`) is unchanged.
- **The timing fit:** `--field gradient` (default: the unweighted reading, as `opticalflow` reads its velocities) · `--field current` (the
  power-weighted fit).
- **Where the sums live:** follows from the input. A band kernel is synthesised on the glued surface, so its sums are stored there (the bridge masked); a
  plain kernel's stay on the kernel's surface, as today.
- **The fused kernel:** stored once as a cached Operator (channels → the eigen axis, [2K × M]) with its provenance; `hodgepower` is unchanged.
- **The cross-spectrum:** the caller names it (`--csd`), as today.

## Order of work (each its own PR, after approval)

1. `phasegradient compute`: the spatial band, the ambient signal, the modes reading and its stored coefficients.
   Tests on synthetic fields on two glued spheres:
   - a plane wave: a linear timing map, consistency ≈ 0, travelling index ≈ 1;
   - a rotating wave: consistency > 0, the vortex core at an extremum of S;
   - two patterns with different timing: consistency > 0;
   - a standing pattern: a flat timing map, travelling index ≈ 0;
   - a pattern inside the band, unchanged by the band.
2. The weighted fit, `--field current` (a compute function: the exact modes, weights P, field H; its own unit test).
3. README: the method's section (why, the default and its reference, what is stored, what it cannot resolve).
4. Then the time windows (the next discussion).


## Amendment: one triangulation

The modes are solved on the intrinsic Delaunay complex; reading and synthesising with the input mesh's operators mixed two triangulations (the fused
div/curl reading was not a projection and failed on a real phase field). Now:
- `eigen solve` stores the full forms: the intrinsic complex (d0, d1, lengths, stars, signposts) and each mode's 2-form (PR #104).
- The band, the phase field and the split are all on that complex (`nxr::intrinsic`): the current read into the modes by inner products with their
  forms (flat / sharp through the signposts, embedded via the input's own edges), the phase 1-form as each intrinsic edge's lag arg T, the split as
  inner products with the modes' 1-forms. No K × K solve, no input-mesh operator.
- The readings of the earlier draft (`--field current`, `divcurl`, the least-squares `weightedFlow`) are gone; the travelling index is not computed
  on this route (it needs ⟨|∇z|²⟩, a vertex-gradient quantity).
