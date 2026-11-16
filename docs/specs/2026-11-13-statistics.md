# The statistic node: whole-recording results computed from partitions (spec, 2026-11-13)

## Why

Context (2026-11-13): computations over a whole recording that do not fit the selection/measurement system are statistics.

Today such results are stored as whatever node fits, and what they mean lives in free-form `producer_json`:
- a sensor PSD is a Field on channels × a frequency line;
- source band power is a Field per band;
- a cross-spectrum is an Operator;
- a distribution has no home (`nxr::histogram` is parked);
- the burst thresholds sit in a Selection's `params_json`.

Nothing in the model says "this is a statistic of recording R over span S, from these n members of this partition, by this method".

**The model to start from is Brainstorm's `timefreq` file** (`db_template('timefreqmat')`). One file holds:
- the values `TF` [rows × time × freq];
- the axes (`RowNames`, `Time` — one window for a PSD —, `TimeBands`, `Freqs` or bands);
- what it is (`Measure`, `Method`, `Options`, `DataType`);
- what it rests on (`nAvg`, `Leff`, `Std`);
- its parent (`DataFile`).

## The three layers

| layer | what | stored as |
|---|---|---|
| Field | the samples, materialized (a recording, a source series, wavelet coefficients) | `field` |
| Selection / measurement | a reduction per MEMBER of a partition (a tile, a window, a parcel): a table of sampled measurements | `selection` + `selection_measurement` |
| **Statistic** | computed FROM the partition's members, over a span (the whole recording, a run of chunks, an event set): a PSD, a cross-spectrum, a distribution, a percentile, a group mean | **`statistic`** (new) |

A statistic always names the partition it was computed from and how many of its members it rests on.
- **Welch's PSD:** the mean over the segments (the partition: the segment tiling) of each segment's periodogram.
- **The analytic cross-spectrum's span estimate:** the sum over the tiles.
- **A histogram:** the counts over the tiles.
- **A percentile:** read off a histogram, a statistic of a statistic.

Per-member values (a spectrogram, a cross-spectrum per window) are the same statistic with the partition as an axis. Collapsing that axis is the
statistic over the span. Storing per-member values is the caller's choice (an option); by default only the reduction is stored.

## The node

### Row: table `statistic` (subject-scoped, `status` like every node, a zarr node at `path`)

| column | meaning |
|---|---|
| `id`, `subject_id`, `status`, `name`, `path`, `acquisition_id` | as every node |
| `measure` | what the values ARE, from the statistics registry: `power_density`, `cross_spectral_density`, `amplitude`, `count`, `probability`, `percentile`, … |
| `method` | how they were estimated: `welch`, `analytic`, `histogram`, `percentile`, `mean`, … (registry) |
| `of_field_id` | the parent Field (the recording, the source series); null for a statistic of a statistic |
| `of_statistic_id` | the parent statistic (a percentile of a histogram); null otherwise |
| `partition_selection_id` | the partition it was computed from (a segment tiling, a constant-Q level, the chunk grid, an event set) |
| `span_json` | the span of the parent it summarises: `{start_s, stop_s, start_sample, stop_sample, what, tile_level}` (io's `span_record`) |
| `n_members` | members it rests on (`nAvg`: segments averaged, tiles counted); `members_skipped` (bad, partial) |
| `dof` | degrees of freedom where the method defines them (Welch: from n and the overlap; the analytic estimate: `independent_estimates`) |
| `axes_json` | the ordered axes of the array (below) |
| `components_json` | the components on the last axis when there are several: `["mean", "std"]`, `["re", "im"]`, `["sum", "count"]` |
| `unit`, `scaling_json` | the unit, and the spectral scaling record (#109) where it applies |
| `data_type` | float32 · float64 |
| `options_json` | every parameter of the method (the method rule: every option is in the provenance) |
| `producer_json` | the provenance and the CACHE KEY, as every node's |
| `description`, `created_*`, `modified_*` | as every node |

### Axes (`axes_json`): the array's dimensions, in order

Each axis is one of:
- `{"name": "channel", "manifold_id": …}` or `{"name": "channel", "selection_id": …}`: rows on a manifold, or the members of a Selection, in its order
  (a kernel's channels, the sources, the tiles of a partition when per-member values are stored);
- `{"name": "frequency", "manifold_id": <a frequency line>}`: an existing closed-form line (`<rec>_freq`, as `psd compute` writes today);
- `{"name": "bin", "edges": {…closed form…}}` or `{"name": "lag", "values_path": …}`: an axis of the statistic's own, closed form when it can be (the
  histogram's log bins: `{lo, hi, bins, scale: "log"}`), else an array beside the values;
- `{"name": "band", "selection_id": <bands on the frequency manifold>}`: the members of a band set (Brainstorm's `TimeBands`/`Freqs` as bands);
- the components axis, last, when `components_json` is set.

The array's shape is the axes' sizes. Brainstorm's `TF` [rows × time × freq] is three axes:
- channel;
- the partition (one member for a PSD: the span);
- frequency.

### Zarr

- **The values:** an array at `path`, under the parent's folder: `<parent path>_statistics/<name>`. It is `<name>` directly when there is no parent.
- **`zarr.json` attributes:** the row, written by a `statistic_attributes` view, as every node's are.
- **An axis of its own** that is not closed form is a sibling array `<path>_<axis>`.

### Without migration

- **New datasets:** `model.sql` gains the table and its view. `user_version` is unchanged: the change is additive, as `allow_value_type` is.
- **Old datasets:** the table and view are created IN PLACE by the first write of a statistic (`ensure_statistics`, recorded in `meta`).
- **Readers:** they treat a missing table as no statistics.
- **Recovery** (pending nodes) skips a table the dataset lacks.

## The system

- **io writes and reads it:**
  - `SubjectWriter::statistic(StatisticSpec)` writes the row and the array in one composition;
  - `read_statistic(db, id | name)` returns the row, the axes resolved and the array;
  - `find_cached` takes the table `statistic`.
- **A registry** (`io/schema/statistics.json`) lists every measure and method with its meaning, unit rule and component names, as `measures.json` does
  for measurements.
- **Producers** compute the partition's measurements in their fold (the chunk service, the measurement functions) and reduce them to the statistic.
- **Pipelines** name a statistic as an input: read if stored, computed and stored if not (decision 1 of the event spec).

## Order (each a PR)

1. **io:** the table, the view, `ensure_statistics`, the writer and reader, the registry, `find_cached`; tests on a synthetic subject and on a database
   made without the table (an old dataset), where the first write adds it.
2. **`psd compute`:**
   - The PSD becomes a statistic: measure `power_density`, method `welch`, the partition the segment tiling (a closed-form spans Selection), `n_members`
     the segments, axes channel × frequency.
   - Its readers (`measure export`, `sourcepsd`'s comparisons, the parity tests) read the statistic.
   - An existing PSD Field stays readable (no migration).
3. **`sourcepsd`, `sourcetensor`, `crossspectrum`** (Welch and analytic: the per-window series is the statistic with the partition as an axis),
   **`quadrature`, `hodgepower`**.
4. **The histogram** (`nxr::histogram`, parked on `feat/histogram-measure`): measure `count`, method `histogram`, a bin axis. Percentiles become a
   statistic of it, and the burst thresholds read them.
5. **The Brainstorm importer:** sensor-level `timefreq` files (PSDs, TF maps) as statistics; today they are skipped.

## What stays as it is

- **Fields** for pointwise and fully resolved results (a filtered recording, a source series, wavelet coefficients, the optical flow per frame).
- **Measurements** for per-member reductions (tile band power, parcel means).
- **Result maps** that are per vertex and band (a band power map, a phase gradient). Whether they become statistics is decided with step 3, case by
  case. They are statistics of a span, but they are also maps that the measurement system reduces over parcels.

## Checklist (the architecture policy)

- **Compute:** unchanged; statistics are made from the existing pure functions.
- **io:** owns the table, the writer and reader, and the arrays. Pipelines only declare what they store.
- **No hard-wired source:** a statistic names its parent, its partition and its span. Every choice is the caller's and recorded.
