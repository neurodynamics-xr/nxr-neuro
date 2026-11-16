# MEG brain fingerprinting: telling people apart from their source spectra (spec, 2026-11-16)

Status: **proposed; no code yet.** It needs `sourcepsd` to keep its bins (section 3.1), the subject-centred dataset module's cohort selection and group
packages (`2026-11-16-subject-centred-datasets.md`, sections 3.1 and 5), and the verb-first command line (`2026-11-16-command-line.md`).

**The method:** da Silva Castanheira, Orozco Perez et al. 2021, *Brief segments of neurophysiological activity enable individual differentiation*,
Nat Commun 12:5713 (code: github.com/neurohazardous/megFingerprinting, notebooks `4_PSD_fingerprinting_*`), after the connectome fingerprinting of
Finn et al. 2015 (Nat Neurosci 18:1664) and Amico & Goñi 2018 (Sci Rep 8:8254). **The reference for the default** is the maintainers' earlier
implementation of it (a per-subject stage and a numpy cohort stage, section 6), whose departures from the papers are listed in section 6.2.

## 1. Why

A person's resting brain activity has a spectral signature: how power is distributed over frequency in each cortical region. If that signature is
more alike between two recordings of the same person than between recordings of different people, a recording can be matched to its owner: a
**fingerprint**. The papers showed that the region spectra of a few minutes of resting MEG, and even of 30 seconds, identify most people in a cohort
of 158, within a session and across sessions months apart, and that the features that differentiate people best are not the ones the group shares.

For the measurement science of brain dynamics this is a **test of reliability at the level of the individual**: before a pattern measured in one
person can be related to their cognition, age or disease, it has to be stable within that person and distinct between people. The fingerprint
gives three numbers for that, all on one footing:
- **identification accuracy**: the share of people whose second recording is most like their first (chance is 1/n);
- **self-identifiability** (the papers' differentiability, D_self): how far each person's self-similarity stands above their similarity to others, in
  standard deviations of the latter;
- **edgewise reliability** per feature (a region at a frequency): the intraclass correlation ICC(1) of the two measurements across people, Finn's
  differential power and group consistency: which features carry the individual and which the group.

**What it measures and what it cannot show.** It measures whether two recordings are more similar within a person than between people. It does not
say *why*: anatomy (head size and shape, cortical folding, the head's position in the helmet), the inverse kernel built from that anatomy, recurring
artefacts (muscle, heart, eyes) and the day's instrument noise are all individual and all enter the spectra (section 8). It is a reliability
measurement, not a finding about brain function by itself.

## 2. The graph, in one view

    PER SUBJECT (one subject package; run once per time range: A and B of a session, or one range per session)
      io: the recording's sensor blocks over the caller's span (chunk service), the bad-segment mask, the kernel, the labelling
      → [graph] kernel (linear map) → window + FFT per segment → power per vertex per bin                 (pointwise)
      → reduce over the SEGMENT partition (Welch mean)                                                   (time partition)
      → readback: statistic `<stem>_<band>_bins` (vertex × bin)
      → region summary over the PARCELLATION partition:
          mean (default): measurement `mean` per region per bin                                          (space partition)
          pca  (option):  the span's covariance (additive over time) → the region's leading component at read
                          → the region map (W·K) as the chain's members stage → Welch per bin            (a second pass)

    COHORT (a group package; members chosen by --subjects or --where)
      io: each member's region × bin spectra for the target and the database (by name), the members' content hashes
      → preparation: type checks (same partition, same frequency line, same units, same Welch options)
      → feature vectors (normalisation at read: relative · log10 · none), region-major
      → statistics over the members: target × database correlation; edgewise ICC(1), differential power, group consistency;
        the across-member PCA, the reconstruction curve, and the edgewise statistics at the best k
      → readings derived at read: accuracy both ways, self-identifiability, self-correlation

Checked against the Studio's chain: per subject, the stages are the Studio's `SignalChain` order (members → transform `xf` spectrum →
measures), with the linear map before the non-linear power (D106); the region summary is `measureOverPartition` for `mean`. The cohort stage has no
Studio counterpart: it is a statistic layer over stored measurements, as `statistic compute` and `pyramid reduce` are.

## 3. Per subject: source spectra per frequency bin, and the region summary

### 3.1 `sourcepsd --resolve bins`

`sourcepsd` today stores a band's power as the mean of its Welch bins (vertex × band). Fingerprinting needs every bin. As `crossspectrum --resolve bins`
does (`2026-11-15-band-resolved-cross-spectrum.md`), only the **frequency partition of the last reduction changes**:
- `--resolve band` (today's behaviour, the default; its output and cache key unchanged): each band is one member of the frequency partition.
- `--resolve bins` (new, named): each bin of each band is its own member; nothing is reduced over frequency. The band map is the mean of its bins
  (tested to rounding).

**Stored**, per band, a statistic `<stem>_<band>_bins`: measure `power_density` (`power` for the other scalings), method `welch`, OF the recording,
from its segment partition `<stem>_segments`, axes vertex (the kernel's target) × frequency (the bins' centres in Hz, `bins: [first, last]`, the
PSD's frequency line), float32, unit as `sourcepsd`'s, `resolve: bins` and every Welch option in `options_json`, the producer and the cache key. A free
kernel's three orientations are summed in power, as today. The fingerprint's band is the caller's (`--bands spectrum:1-150`); the bins kept are
that band's, so a 1–150 Hz band at 0.5 Hz on 15 000 vertices is 299 bins, about 18 MB.

**Memory.** Today `sourcepsd` accumulates the sensors' cross-spectrum per band (M² complex) and takes diag(K C Kᵀ) at the end. Per bin that is
M² · F · 16 bytes (270 channels, 299 bins: ~350 MB). The kernel is linear, so the same numbers come from applying K to each segment's Fourier
coefficients and accumulating |K X|² per vertex per bin (N · F · 8 bytes: ~36 MB), at the price of one N × M product per segment. Which path is used
(or a threshold between them) is open (question 4); either way the options struct states the memory held. Results must agree to rounding.

**Two time ranges.** The caller runs it once per range, naming each result: `--from 5 --to 150 --name fp_A`, then `--from 150 --to 295 --name fp_B`
(within a session), or once per session's recording (between sessions). The ranges are always stated; there is no "halves" default and no
"first session" rule: which data is the target and which the database is the caller's (section 5). Segments carry across chunk edges, the bad
segments of the recording are left out (a segment touching a bad sample is dropped, the grid does not move), and the result does not depend on the
chunk size (section 7).

**The kernel is required here.** `sourcepsd` today falls back to "the subject's only kernel"; that is a choice of data, so this work makes
`--kernel` required for `--resolve bins` (and proposes it for every `sourcepsd` run; question 6).

### 3.2 The region summary: a measurement over the parcellation partition

The papers and the earlier implementation compare region spectra (Desikan-Killiany, 68 regions), not vertex spectra. The region summary is a
**named option**, `--region-summary mean|pca`, with `--labelling L` (a `set` Selection: any imported parcellation or a tile ladder level).

**`mean` (the default; reproduces the earlier implementation).** The region spectrum at bin f is the mean of the region's vertex spectra at f: the
measurement `mean` over the partition (`measureOverPartition`), one `selection_measurement` row a region a bin (`of_statistic_id`, `of_index` = the
bin), with the measurement's weight stated by the caller (`--weight none|barycentric|voronoi`; the earlier implementation's is `none`, the plain
vertex mean). No new compute: it is `measure compute` on a map statistic, run by `sourcepsd`'s readback when `--labelling` is given, with the same recipe
record and refusals.

**`pca` (the papers' scout first component; a named option).** The papers collapsed each region's source time series to its first principal component
before the spectrum (Brainstorm's PCA scout, applied before the PSD), so a region's spectrum is the spectrum of one signal. Built here without a
special loop:
1. **The region's covariance is a measurement over time.** Pass one reduces x xᵀ of the sensor block over the span's time partition (the sum and the
   sample count; additive across chunks, as `nxr::partition::crossSpectrum` is) and stores it as a statistic `<stem>_covariance` (sensors ×
   sensors, OF the recording, over the same span and bad-sample mask as the spectrum). The kernel being linear, each region's source covariance is
   K_R Σ K_Rᵀ, never formed for the whole cortex.
2. **The leading component is derived at read**: w_R, the leading eigenvector of K_R Σ K_Rᵀ, signed by the region's majority orientation, is
   `region::collapse(pca_flip, …)` (already in `compute/region.h`), with the share of the region's power it explains. Nothing about it is stored
   beyond its ingredients and that share (in the provenance).
3. **The region map is the chain's members stage**: W K (regions × sensors), bound as a constant in preparation, then the same Welch per bin. Stored as
   `<stem>_<band>_bins` with axes region (the partition's members, by dictionary name) × frequency. Equivalent by linearity to w_Rᵀ K_R C(f) K_Rᵀ w_R
   from a per-bin sensor cross-spectrum; a test checks the two.

The component is computed per range from that range's own covariance, as the papers' scouts were computed per time window. For power the sign of w_R
does not matter; its scale does (question 3).

## 4. The cohort: statistics over the members

### 4.1 Selection, reading and type checks (io and preparation)

- **Members** come from the catalogue by `--subjects NAME,…` or `--where FILTER` (exactly one; neither or both refused; no "all"), each with its
  package content hash; the matched list is printed before anything is computed and recorded in the result.
- **Inputs** per member: the stored region spectra named `--target` and `--database` (the names the caller gave in the per-subject stage, the same in
  every member), read through the lookup on `--labelling`. A member missing either is **refused, by name** (no silent exclusion; question 7).
- **Type checks, before any value is read**: every member's target and database share the partition's member names (joined by dictionary name, as
  `measure export` does; `unassigned` left out), the frequency line (bin centres; a member at another sampling rate or segment length has other
  bins and is refused, naming its line), the unit and spectral scaling, the region summary, and the Welch options; the kernel kind and the inverse
  method are compared and a mismatch is refused unless the caller passes `--allow-mixed-kernels` (recorded). `--comparison` is checked against the
  data: `within-session` needs target and database from one recording (or one acquisition) with spans that do not overlap; `between-session` needs
  them from different sessions. The label chooses no data.

### 4.2 Features

Per member and measurement, the region × bin matrix inside `--frequencies LO-HI` (default 1–150 Hz, the earlier implementation's), normalised by
`--normalisation`, with the earlier implementation's names:
- `relative` (default): each region's spectrum divided by its sum over the range's bins. A ratio of reduced quantities, so it is **derived at read**
  when the features are assembled, never stored;
- `log10`: pointwise;
- `none`: the stored values (the papers' notebooks use the raw spectrum, 0–150 Hz).

then flattened region-major (region, then bin) into one vector of p = regions × bins features. The features are assembled at read and not stored.

### 4.3 The statistics (each a statistic node of the group package, over the cohort's members)

T and D are the n × p target and database matrices, row i = member i; z(·) z-scores a row across its p features (population sd, as the notebooks).

| statistic (measure) | what | axes |
|---|---|---|
| `identification_correlation` | r_ij = z(t_i) · z(d_j) / (p − 1), the notebooks' estimator (it is Pearson's r × p/(p − 1); argmax and self-identifiability are unchanged) | target member × database member |
| `edgewise_reliability` | per feature: ICC(1) with k = 2 measurements on the z-scored features (as the notebooks); differential power (Finn 2015: Σ_i −ln p_i, p_i the share of ρ_ij, ρ_ji above ρ_ii, floored at 1/(2(n − 1))); group consistency (mean_i z_t,i · z_d,i) | feature (region × bin) × quantity |
| `principal_components` (method `pca`, section 4.4) | the across-member PCA of [T; D] | see 4.4 |
| `identification_curve` | per k: accuracy both ways and the mean self-identifiability of the reconstruction at k components | k × quantity |
| `edgewise_reliability` at the best k | the edgewise statistics of the reconstruction at the k of highest mean self-identifiability (the earlier implementation's rule) | feature × quantity |

**Derived at read** from `identification_correlation` (never stored as values of their own; the verb prints them, `export measurements` writes them):
- accuracy target → database: the share of rows whose maximum is on the diagonal; database → target: the same for columns; chance 1/n;
- self-identifiability per member: the diagonal of the row-wise z-score of r (the papers' D_self);
- self-correlation per member: r_ii.

**Comparisons.** `--comparison within-session` (two ranges of one session) and `between-session` (two sessions) run the same statistics; each run is
one comparison, stored under its own stem (`<stem>_<comparison>_…`). The papers' 30-second "shortened" challenges are more runs of the per-subject
stage on shorter ranges, compared pair by pair: nothing in the cohort stage changes.

**Stored**, in the group package named by `--into GROUP` (never a default one; dataset spec section 5): each statistic with `value_type`, axes, unit,
the cohort partition (a Selection of the members, in order, with each member's subject row and content hash as anchor rows), `n_members`, every
option (comparison, target, database, labelling, frequencies, normalisation, PCA steps), the selection (`--subjects` list or `--where` text and the
filter language's version), the per-member inputs read (statistic ids, recipes), the producer and the cache key. A rerun with the same members (same
hashes) and options is a cache hit.

### 4.4 A general `pca` statistic across members

`eigendecomposition` today decomposes a Hermitian second-moment statistic of one subject (the cycle patterns of a windowed cross-spectrum). The
**`pca` statistic** generalises it to a table of members × features, the usual case across people:
- **Input:** a matrix X [rows × p] assembled at read from stored measurements: rows are members (or member × measurement, as [T; D] here), columns the
  features. Column means μ are removed (scikit-learn's `PCA`); no scaling unless the caller asks (`--scale none|unit-variance`, a named option).
- **Computed** by a thin SVD of X − μ (never the p × p covariance: p is ~20 000 here, rows ~300), the general algebra in `compute/` (a pure function,
  matrices in and out, callable with any matrix).
- **Stored**, one statistic, method `pca`: the components V [component × feature], the singular values and the explained variance and its ratio
  [component], the scores U S [row × component], μ [feature]; every component up to rank(X − μ), so any k is available. The rows' identities and the
  features' axes are recorded as for any statistic.
- **Reconstruction at k is derived at read**: X̂_k = (U S)[:, :k] V[:k] + μ. Nothing per k is stored by the `pca` statistic itself.
- **Same layout as `eigendecomposition`** (vectors, values, shares), so readers treat both alike; `eigendecomposition` of a stored covariance and
  `pca` of the table that made it give the same components (a test).
- Usable on its own: `compute statistic --method pca --of PATTERN --over GROUP_MEMBERS` on any measurement table of a group package.

**The fingerprinting curve** is built on it: for each k in the steps (default the earlier implementation's and the notebooks': 2, 5, then 10, 20, …
below 2n, capped below min(2n, p)), the reconstruction at k is split back into T̂ and D̂ and identified as in 4.3; the curve and the edgewise
statistics at the best k are stored (`OF` the `pca` statistic).

## 5. The command line (verb-first; required inputs, no data defaults)

Per subject, once per time range (the reference's Welch is 2-s segments, 0.5-Hz bins: section 6):

    nxr-neuro compute sourcepsd --dataset D/N --subject S --recording R --kernel K (--from SEC --to SEC | --all)
        --bands spectrum:1-150 --resolve bins --name fp_A
        [--labelling L --region-summary mean --weight none | --labelling L --region-summary pca]
        [--set brainstorm --segment-seconds 2 --overlap 0.5 --window hamming --symmetric] [--no-bad-segments] [--replace]

Required: dataset, subject, recording, kernel, the time range, `--name` (the name the cohort stage reads, so it must be the same in every member).
With `--labelling`, `--region-summary` and (for `mean`) `--weight` are required.

The cohort:

    nxr-neuro compute fingerprint --dataset D/N (--subjects A,B,C | --where FILTER) --into GROUP
        --comparison within-session|between-session --target fp_A --database fp_B
        --features region-spectra --labelling L
        [--frequencies 1-150] [--normalisation relative|log10|none] [--pca-steps 2,5,10,...] [--name STEM] [--allow-mixed-kernels] [--replace]

Required: dataset, the selection, `--into`, `--comparison`, `--target`, `--database`, `--features` (only `region-spectra` now; the papers' amplitude
envelope connectome is a later value, question 9), `--labelling`. Defaults exist only for the method's reference parameters (frequencies,
normalisation, PCA steps) and the output stem (`fingerprint`). The result line prints n, both accuracies against chance, the mean
self-identifiability, the best k and its accuracies; `--json` gives the usual JSON lines. A `run` definition (`fingerprint_subject.yaml`) may chain the
per-subject runs for two ranges; it takes the ranges as required `--set` values.

## 6. References and defaults

### 6.1 The default reproduces the earlier implementation

The maintainers' earlier implementation computed the per-subject spectra with a Brainstorm-based stage and the cohort with a numpy script. Its
choices are the defaults here (the method rule): region **mean** of the vertex spectra, 1–150 Hz, **relative** normalisation, the notebooks' estimators
(correlation, ICC on z-scored features, differential power with its floor, group consistency), full (not randomised) SVD for the PCA, the PCA steps
and the best-k rule. Brainstorm's Welch (`--set brainstorm`: 2-s segments, 50 % overlap, symmetric Hamming, the segment mean removed, density) gives
0.5-Hz bins at any sampling rate, so members at different rates still share a frequency line; the `tiles` set (2^k samples) does not, and the cohort
stage refuses its mixed lines.

**Parity is proven on synthetic data only:**
- **Cohort stage:** synthetic region spectra (section 7) are written in the earlier script's input form; a developer runs the earlier script once to
  make the golden (correlations, accuracies, self-identifiability, ICC, differential power, group consistency, the PCA curve and the best k), committed
  with its generator under `pipelines/test/golden/`. The generator finds the earlier script through an environment variable and is skipped, with a
  reason, without it; the committed golden is what the test reads. Because the earlier script is not public, a second generator transcribes the
  public notebooks' estimators in numpy so anyone can regenerate; both goldens must agree.
- **Per-subject stage:** `--resolve bins` against `--resolve band` (the band is the bins' mean) and against scipy's Welch of the projected sources
  (synthetic recordings and kernel); `pca` against the first principal component computed directly from the synthetic source time series.

### 6.2 Departures from the papers

| | papers | default here (earlier implementation) |
|---|---|---|
| source model | LCMV beamformer (Brainstorm 2016, NAI-normalised, orientation normal to the cortex, 15 000 sources, empty-room noise covariance), one model per frequency band | **the caller's kernel**; the earlier implementation used a depth-weighted dSPM kernel constrained normal to the cortex. **The beamformer is not in nxr-neuro**, so the papers' source model cannot be reproduced here |
| region signal | the first principal component of each region's source time series, before the spectrum | the **mean** of the region's vertex spectra (after the spectrum); the papers' component is the named option `pca` |
| frequency range | 0–150 Hz, 0.5-Hz bins (301 bins) | 1–150 Hz, 0.5-Hz bins (299 bins); the range is an option |
| normalisation | none in the notebooks (the spectra as computed) | `relative`; `none` is the papers' |
| the two within-session halves | [5 s, T/2] and [T/2, T − 5 s] of one run | the earlier implementation used the kept 4-s epochs of each half after epoch rejection; here, the caller's two ranges with the recording's bad segments left out (the papers' halves are `--from 5 --to T/2` and `--from T/2 --to T−5`) |
| PCA | scikit-learn's `PCA`, whose automatic solver is randomised at this size | a full SVD; the notebooks' curve matches to the randomised solver's tolerance only |
| preprocessing | notch at 60 Hz and 10 harmonics, 0.3-Hz high-pass, SSP for heart, eyes, saccades and muscle | not part of this pipeline: the recording the caller names carries its own preprocessing and projectors (recorded in the provenance) |

## 7. Tests (synthetic data only)

- **Planted individuals:** n synthetic subjects, each with its own spectrum per region (a 1/f exponent, a peak frequency and a peak height drawn per
  subject per region), and two measurements per subject made of that spectrum plus independent noise well below the between-subject spread:
  identification is perfect both ways, every self-identifiability is high, ICC is high on the planted features.
- **Null cohort:** the target and the database drawn independently per subject: accuracy at chance (within a binomial bound around 1/n over many
  draws), self-identifiability centred on 0, ICC near 0.
- **Parity:** the cohort stage against the earlier script's golden, and against the notebooks' transcription (section 6.1).
- **End to end on a synthetic sphere:** recordings generated from planted region sources through a synthetic forward and kernel, both stages run
  through the command, and the planted subjects identified.
- **Chunk-size independence** of the per-subject stage: `sourcepsd --resolve bins` (both accumulation paths) and the `pca` covariance give the same
  values with chunks of 128, 700 and 1777 samples and with one chunk over the span.
- **Units:** `--resolve bins` against `--resolve band`; `pca` through the members stage against w_Rᵀ K_R C(f) K_Rᵀ w_R from a per-bin cross-spectrum;
  the `pca` statistic against numpy's SVD and its reconstruction at k against `PCA.inverse_transform` with the full solver; `pca` of a table against
  `eigendecomposition` of its covariance.
- **Refusals:** a missing target or database in a member, mismatched frequency lines or partitions, overlapping spans under `within-session`, the
  same session under `between-session`, neither or both of `--subjects` and `--where`.

## 8. What it cannot resolve (for `docs/reference.md`, not the tool's output)

- **Identification is not neural by construction.** Anatomy, the head's position in the helmet, the kernel built from them, recurring artefacts
  (muscle in the gamma range, the heart, the eyes) and the day's environmental noise are individual and enter the spectra. Within a session the head
  position and the noise are shared by both ranges, so within-session accuracy is an upper bound of between-session accuracy. The papers' controls
  (identification from empty-room recordings through each person's kernel; correlation of self-identifiability with artefact amplitude) are not run
  by the tool; they can be run as further comparisons (question 8).
- **Accuracy depends on the cohort:** it falls as n grows and depends on how alike the members are; numbers from cohorts of different size or make-up
  are not comparable. Self-identifiability is relative to the cohort it was computed in.
- **Regions are not independent** through a minimum-norm or beamformer kernel: neighbouring regions share signal (the kernel's spread), so the p
  features are far fewer independent measurements than p.
- **The PCA curve and the best k are fitted on the data they score** (both measurements of every member enter the PCA, and k is chosen for the highest
  mean self-identifiability), so the reconstructed accuracy is optimistic.
- **ICC is computed on features z-scored within each vector** (the notebooks' choice), so it measures the reliability of a feature's position within a
  person's spectrum profile, not of its absolute power.
- **A spectrum is not a pattern's speed or direction.** The fingerprint uses time-averaged power per region; it says nothing about the spatiotemporal
  organisation of the activity.

## 9. Phases

| phase | what | tests |
|---|---|---|
| 1 | `sourcepsd --resolve bins` (both accumulation paths), `--kernel` required with it | bins vs band, chunk-size independence, memory stated |
| 2 | the region summary: `mean` through the measurement at readback; the span covariance statistic and `pca` | mean vs `measure compute`; pca vs direct PCA; the two pca paths agree |
| 3 | the `pca` statistic (compute + io + `compute statistic --method pca`) | numpy SVD, reconstruction, agreement with `eigendecomposition` |
| 4 | `compute fingerprint`: selection, checks, statistics in a group package, readings at read | planted, null, parity goldens, refusals |
| 5 | `docs/reference.md` section and the command list | a verb is not done until it is listed |

## 10. Open questions for the maintainers

1. **Naming the reference.** The method rule's references are nxr, Rheome, the Studio, Brainstorm, MNE-Python and scipy. Is the earlier
   implementation accepted as the default's reference, with the public notebooks as the papers' reference (and both goldens kept)?
2. **The default region summary.** `mean` reproduces the earlier implementation; `pca` reproduces the papers' scouts. Keep `mean` as the default?
   A plain mean of vertex *power* does not cancel opposite orientations (it is taken after the spectrum); confirm the earlier implementation applied
   it after the spectrum, not to the time series.
3. **The scale and centring of `pca`.** Brainstorm's PCA scout has changed between versions (scaling of the component, centring of the time series,
   and in recent versions one component across files). Which version's behaviour is the option's reference: a unit-norm eigenvector, or the
   component scaled to the region's power?
4. **Accumulating per-bin power:** sensor-side (M² · F, ~350 MB for 1–150 Hz) or source-side (N · F, ~36 MB, one N × M product a segment), or a
   threshold between them?
5. **Epoch rejection.** The earlier implementation built its halves from kept 4-s epochs after a peak-to-peak / gradient rejection (median + 3 MAD
   over the session). nxr-neuro only leaves out the recording's bad segments. Should that rejection become a detector that writes a bad-segment
   Selection (so the halves match), or is the caller's own bad-segment marking enough?
6. **The default kernel.** Make `--kernel` required for every `sourcepsd` run (it is a choice of data), not only with `--resolve bins`?
7. **Members without both measurements:** refuse (proposed), or an explicit `--skip-incomplete` that records each skipped member and why, as the
   earlier implementation listed them?
8. **Controls as comparisons:** add an empty-room comparison (the noise recording through each person's kernel, as the papers did) and the shortened
   30-second challenges to the reference section's recipes, or leave them to the caller?
9. **Other features:** the papers' amplitude-envelope connectome (68 × 68 per band) and band-limited spectra as further `--features` values, later?
10. **Uncertainty:** store bootstrap or permutation intervals for accuracy and self-identifiability (as statistics over resampled members), or leave
    inference outside the tool?
11. **The PCA's circularity:** add a cross-validated option (fit the components without the member scored), as a named option beside the papers'?
