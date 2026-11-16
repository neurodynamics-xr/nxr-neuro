# Audit, 2026-11-13: the Studio's atlas and Rheome's joint ladder, for cortical space × time cells (read-only)

**Purpose.** The plan `docs/specs/2026-11-13-cortical-events.md` (branch `docs/cortical-events`) builds space × time cells (a spatial tile × a time
tile) with windowed band features per cell, detection on the cell graph, and a per-event roll-up. The the maintainers decided (2026-11-13): a spatial level IS its
spatial band (the Weyl band λ = 4π·2^l/A), as a constant-Q time level IS its band (D84). Our rule: port the references' designs, do not redesign.
This audit records what the references do. It is not a design, and it does not resolve the differences: they go to the maintainers.
- Repos read: the Studio `nxr-cortical-flow/backend/src`, Rheome `rheome/+rheome`, nxr `nxr/src/nxr`, nxr-neuro (main, 139ef0c).
- No dataset content was read.

## 1. The Studio's atlas

### 1.1 What an atlas is
- **Measurements on tile partitions, array-backed** (`db/atlas.ts:1-22`): no atlas table and no second array route (D68, D96).
- **Space:** the tile partition at ONE depth D per tree: `<surface> tiles L<D>` (the subject's own ladder) and `<surface> group tiles L<D>`
  (`atlas.ts:7-9, 149`). The leaves are `2·2^D` (both hemispheres), codes `h·2^D + t` (`db/rollup.ts:5`).
- **Space × time:** a JOINT partition (`type 'set'`, cell `'0,1'`) on the product manifold cortex × the recording's time Line
  (`atlas.ts:12-14, 192-195, 213-215`).

### 1.2 The joint cell code (exact)
- **`code = (tile << B) | frame`**, B = `bitsFor(F)` = max(1, ceil(log2(max(2, F)))), F the number of frames (`atlas.ts:50, 176, 214`).
  The description string says so: "code = (tile << B) | frame" (`atlas.ts:214`).
- The code is a LABEL only. Nothing is stored per code: the joint set is "in closed form" (`atlas.ts:12`), `n_members = tiles × F`, and its
  `params_json.joint = {space, time, bits, tiles, frames, layout: ['band','tile','frame']}` (`atlas.ts:215`).
- **The data are arrays [band × tile × frame]**, read by index, not by code: `readAtlas` slices `[[k,k+1],[0,T],[0,F]]` (`entities/atlas.ts:78-79`).
  `atlasInfo` returns `bits`, `frames`, `tiles`, `depth`, `code0` (`entities/atlas.ts:11-26, 34-44`).

### 1.3 What a frame is
- **A fixed window of the day-anchored TOWER**, not a sample and not a ladder level of the constant-Q bank: "Frames are cycles of the day-anchored
  tower (D135, TOWER): frame k is tower code `frame_code0 + k`" (`atlas.ts:178-179`; `db/rollup.ts:7-9`).
- Stored as a closed-form `spans` Selection on the recording's own time Line, `params_json.tiling = {origin, width: frameS, hop: frameS, count: F}`,
  named `<rec> frames <frameS> s` (`atlas.ts:185-190`). Defaults: `frame_level` 0 and 0.25 s for groups from before the convention; the
  tower's `frame_level` −18 ≈ 0.33 s is the current convention (`db/rollup.ts:7`; nxr-neuro `tiles.h:43, 51`).
- A level-m time tile is `2^m` consecutive frames aligned on the tower, parent = code >> 1; the first and last may be partial (`rollup.ts:7-9, 48-52`).
- Frames count good samples: `samples` is a row on the frame tiling (`atlas.ts:14, 206-207`).

### 1.4 Which levels
- **Space: ONE stored depth D** (default 8; `atlas.ts:234`; `entities/atlas.ts:12`); every coarser level is a roll-up read, never stored
  (`rollup.ts:1-3`). Finer than D is refused (`rollup.ts:42`).
- **Time: ONE stored frame level**; coarser time tiles are a roll-up (`rollup.ts:50-52`).
- So the atlas is a single (D, frame) grid with arithmetic views at (ℓ ≤ D, m ≥ 0). It is NOT a ladder of joint levels each with its own band.

### 1.5 How bands are stored
- **Bands are TEMPORAL frequency bands**: `atlas bands`, a `spans` Selection on the dataset's default subject's Frequency manifold, a span [lo, hi) Hz
  per band, code = the band's index, octaves 1–64 Hz by default (`atlas.ts:68-89, 140`; nxr `pipelines/atlas/build.py:233-252`; nxr-neuro `tiles.h:44`).
- **A row's band is the `within` pair:** `within_selection_id` = the bands Selection, `within_code` = the band index; one row per band, all sharing the
  one joint partition and its one spatial tiling (`atlas.ts:117-121, 217`). `within` is "restricted to: the part of ANOTHER manifold this value is over"
  (`db/database.ts:162-170`).
- **There is no spatial band.** Many (temporal) bands share one spatial depth D; the spatial level is a pure tiling, never paired with its Weyl band
  in the atlas. (The Weyl pairing exists separately in the ladder: `db/surface-ladder.ts:19-37, 62-63`, a level of m tiles ↔ λ(m) = 4π·m/A.)
- Group atlases collapse time ("subjects share no clock"): band arrays [band × tile] (`atlas.ts:269`).

### 1.6 Measures per cell and roll-up
- **Stored arrays** per band, mapped by `ATLAS_MEASURES` (`atlas.ts:33-41`): `power` → `power_sum`, `env` → `envelope_sum`, `envmax` → `envelope_max`,
  `samples` → `count`, plus the map sums `weight, sum, sum_square, count, min, max`, and `gradient_*`, `tensor_*` families.
- **Merge rule** (`rollup.ts:60-63`): `max`/`*_max` → max, `min` → min, else sum. Only mergeable sums roll up; means and variances are DERIVED after
  (`rollup.ts:11-12, 69-78`).
- **Space:** `code >> (D − ℓ)`; a node's leaves are one contiguous block, so the roll-up is a reshape and a reduction (`rollup.ts:5-6, 38-46, 58`).
- **Time:** `rollTime` groups `2^m` frames on the tower, neutral element for missing frames (0, −∞, +∞) (`rollup.ts:48-52`).
- **Space and time commute** (nxr `pipelines/atlas/joint.py:5-7`). The two are applied in turn in `readAtlas` (`entities/atlas.ts:80`).
- **The density of a cell** is `power_sum / (area × good samples)`, normalised at read (`entities/atlas.ts:171-179`; nxr `joint.py:52-58`).
- **Path-dependent arrays** (peak trajectory, Levi-Civita transport, orientation drift) are explicit per level, not rolled up (`atlas.ts:20, 153`;
  nxr `joint.py:9-20`).

### 1.7 `searchAtlas` (`entities/atlas.ts:95-218`)
- **Input** `AtlasQuery` (`:98-112`): `measure`, `band`, space `level`, `timeLevel`, `normalize` (default on: a sum divided by weight, or by area × good
  samples), `above`, `below`, `top`, `nodes` (restrict to these).
- **Steps:** read and roll the array to (level, timeLevel) (`:164`); normalise a sum measure (`:171-180`); keep every cell passing `above`/`below`
  (NaN dropped; `:187-196`); sort by value, take `top` (`:197-198`).
- **Output** `AtlasSearch` (`:114-123`): `hits` {node, tile, value, start, stop}; `spans` (the hit cells' time tiles merged where they touch, `:199-201`);
  a TRANSIENT `spans` Selection on the recording's time Line, params `search` (`:205-215`); the stored chunks the spans touch (`chunksUnder`, `:126-133, 216`).
- **It is a cell-by-cell threshold.** No hysteresis, no neighbours, no connected regions, no seeds and growth, no minimum duration or area, no merge
  across space. Two adjacent hit cells are two hits; only their TIMES merge. Searched over one band at a time.
- `tilePath` (nxr `measure/tiles.py:103-130`, the Studio's `engine/tilePath.ts`) is the only region-like logic: a Viterbi best walk of ONE feature
  through [tiles × frames] on the tile graph (`tile_adjacency`, `:87-94`). It returns a path, not a set of cells. It is tracking (family C), not detection.
- **Tile graph:** `tile_adjacency` (two tiles adjacent where a mesh edge joins their vertices; nxr `measure/tiles.py:87-94`, `joint.py:33-45`).

### 1.8 Row-form measurements and `within` (the non-atlas route)
- A measurement is "what a FIELD is over one MEMBER of a selection"; stored once at the finest level, coarser views by `GROUP BY code >> shift`
  under the measure's reduction; `weight` makes a mean exact (`db/measurements.ts:15-33`).
- **`levels` vs `time`** (`measurements.ts:36-47`): a measure that stays true across levels is stored once at the finest; one where "the level IS the
  band (D84)" is stored per level, because a band's power at one level is not derivable from another's.
- `within_selection_id`/`within_code`: "the part of ANOTHER manifold this value is over" and what it is grouped BY; null = the whole
  (`db/database.ts:162-170`).

## 2. Rheome's joint ladder

### 2.1 The joint cell (`+geom/jointladder.m`)
- **A row is a SELECTION, not a measurement** (`:7`): one observation window of the joint eigenmode-frequency plane, "written in the spatial and
  temporal units it is easier to reason in". Output: one row per **depth × octave × frame level** (`:61-64, 107-109`).
- **Three coordinates:** a tile depth (diameter D, "standing for a band of Laplace-Beltrami eigenvalues", `:9-10`), a temporal octave [fLo, fHi]
  with its window W (a constant-Q support, dyadic, anchored at 1 s: `W = 2^ceil(log2(k/fLo))`, `:38-41, 90`), and a frame level (frame rate Fs/2^level,
  down to one frame per window, `:94-95`).
- **Space-time scale pairing = speed:** speedMax = D/dt (one tile per frame), speedTrack = D/(MinDwell·dt) with MinDwell 2.5, speedMin = D/W
  (`:17-23, 102`). Cells on one diagonal share a speed.
- **The depth's band is stated, not tied:** the octave of eigenvalues a depth stands for is σ ∈ [σmin, 2σmin], λ = 2/σ², with σmin = D/TileSigma
  (`:25-36, 99-101`); "two tree depths are one spatial octave, because area halves per depth" (`:35-36`). `jointcell` picks the coarsest depth whose
  tile still holds the feature (`+geom/jointcell.m:12-14, 68-75`).
- **Stored where:** the row says what CAN be measured; what IS measured is "stored at the row's address"; power rolls up exactly, a trajectory or speed
  does not (`jointladder.m:11-15`).
- **Instrument columns** (`nodeAdmissible`, `R50mm`) are the instrument's bound, not the ladder's (`:45-47`). Not for our tools.

### 2.2 The spatial ladder and the "forbidden" rule (`+geom/ladder.m`)
- **One spatial octave = TWO depths** (area halves, diameter ÷√2; `:13-17`). Level-for-level between time and space is "off by a factor of two per
  octave" (`:16-17`). Spatial octaves are anchored on the hemisphere as an equivalent disc, wavelength halving per octave (`:103-109`).
- **The invariant** (`:19-25`): by Weyl N(k) = A k²/4π, so an octave holds ×4 modes; a depth step ×½ area; so modes per cell are constant along
  "the diagonal" D increasing by 2 per octave. "Every (tile, octave) cell holds the same number of modes, just as every (tile, band) cell holds the
  same number of cycles."
- **`.forbidden`** (`:27-31, 121`): TRUE when the node diameter is below `Cycles` × the octave's centre wavelength (`diam(i) < Cycles*wlC(j)`, default
  `Cycles` 2 node diameters per wavelength). Stated reason: "BELOW it (node smaller than the wavelength) the cell is not a measurement at all: a mode
  whose wavelength exceeds the patch does not complete a cycle inside it, so its 'energy in this patch' is an artefact of where the patch was drawn."
  Above the diagonal a cell is "merely coarse... still a valid measurement" (`:27-29`).
- **What it is NOT:** it is a rule between a spatial TILE and a spatial BAND (wavelength), not between space and time, and not a speed rule. The
  plan's gloss ("a pattern smaller in space than its speed allows in time", cortical-events spec, step 1) is a misreading; the time side of
  Rheome's joint ladder has no forbidden flag (`grep` of `+geom`, `+detect`, `+scale`, `+select` finds `forbidden` only in `ladder.m`).
- **What the ladder says may be recorded** (`:33-44`): band-limited energy (filter the current to the octave, linear and exact, then integrate |J|²
  over the node; rolls up over nodes within an octave). NOT "eigenmode bandpower of this node" (a global mode has no per-patch share); NOT a size.
- **Band and tile are independent in Rheome:** a table row per spatial octave, `naturalDepth` the finest tile holding `Cycles` wavelengths
  (`:115-118`); many octaves can sit on one depth, and below the diagonal is forbidden. The Studio's atlas has no spatial band at all (1.5).

### 2.3 Events (`+detect/eventtable.m`, `+scale/measure_events.m`)
- **Where events come from** (`measure_events.m:9-14, 225-233`): the tracker's detections (a Viterbi path per hemisphere and 2 s window beating a
  mode-shift surrogate threshold, depth-8 tiles, 18.75 frames/s). One row per event: hemisphere, tiles visited, first and last time. "The same code
  reads atlas-cell events once they exist" (`:10-11`): the table is generic.
- **Merge** (`:14-15, 235-248`): detections of one hemisphere less than `MergeGapS` apart whose tiles overlap or touch (the tile adjacency with an
  identity, `Ga`, `:106`, `G.A > 0 | speye`) are one event; tiles = the union, t1 = the max.
- **Event area and tile count** (`eventtable.m:47, 58`): `k = unique(tiles)` over the event's space tiles; `nTiles = numel(k)`;
  **`areaMM2 = 1e6 · Σ A.(h)(k)`** (tile areas in m², summed over the UNIQUE tiles the event visits, in one hemisphere; the union of its spatial
  tiles, with no time weighting).
- **Event envelope** (`eventtable.m:8-9, 47-48`): the area-weighted mean of the tile envelopes over the event's tiles, `w = A(k)/ΣA(k)`, at the
  envelope's native rate; the shape (risefall) is fitted to it.
- **Duration/boundaries:** from the shape fit, not from time tiles (`eventtable.m:20-26`).

### 2.4 Occupancy (`+detect/occupancy.m`)
- A dyadic pyramid of on/off over time, per channel: level L's tile is ON when the fraction of its 2^L finest tiles that are on is ≥ Θ (0.75)
  (`:4-8, 53`). Occupancy is a mean, so it rolls up exactly (`:51`). `.runs{L}` maximal runs of on-tiles per level; `.depth` the longest enclosing
  on-tile (`:11-18`).
- Time only, per channel: no spatial adjacency, no joint cell. It is the time analogue of "hysteresis by scale".

## 3. nxr (Python), briefly
- `geom/ladder.py`: the Studio's `surfaceLadder`, `Ladder` (`:132`), `surface_ladder` (`:146`); the same equal-mass halving, `onePiece`, and codes.
- `measure/rollup.py`: `space` (`:21-31`) and `time` (`:34-46`, with `code0`), the same arithmetic as `rollup.ts`.
- `measure/tiles.py`: the spatial scalogram, `tile_adjacency` (`:87`), `tile_path` (`:103`, Viterbi), and `cell_readouts` (`:148`): "a vector field over
  a tile of space level j × a segment of time", every readout an area- and frame-weighted average, `[T × S]` (tiles × segments).
- `pipelines/atlas/joint.py`: the joint cell as (node s at level ℓ, time tile k at level m) = `2^(D−ℓ)` leaves × `2^m` frames (`:3-5`);
  `cell_density` [bands, nodes, time tiles] (`:52-58`); `peak_trajectory` (`:60-63`). Same layout as the Studio.
- `pipelines/atlas/build.py:16`: "code = (tile << B) | frame, rows are the band arrays, within the atlas bands". Same as the Studio.
- Nothing in nxr detects events on cells: its tile code is the Studio's, ported.

## 4. Porting conclusions against the decided rule
The rule: a spatial level IS its spatial band; a joint cell's Selection is the pair (spatial level, time level) of nxr-neuro's surface ladder and
constant-Q time ladder; nothing is added to a row's key.

### 4.1 Port as is (with the reference's names)
| piece | reference | port |
|---|---|---|
| joint cell code | `code = (tile << B) \| frame`, `B = bitsFor(F)` (`atlas.ts:50, 214`) | a label for the cell; the data are arrays [tile × frame], read by index |
| closed form | the joint set has no rows; `params_json.joint {space, time, bits, tiles, frames, layout}` (`atlas.ts:215`) | the same `joint` record, `layout` ['tile','frame'] (the band axis goes: 4.2 a) |
| space roll-up | `code >> (D − ℓ)`, contiguous blocks (`rollup.ts:38-46`) | already the lookup's `code >> shift` (`io/src/measures.cpp:70, 100`) and `Tree::at` (`ladder.h:100`) |
| time roll-up | `rollTime`, 2^m frames, neutral for missing (`rollup.ts:48-52`) | nxr-neuro's time ladder rolls up by the tile tree (`code >> shift`); its tile edges are the tiling's own (see 4.2 c) |
| merge rule | sum / max / min; derive means after (`rollup.ts:60-78`) | `nxr::partition::accumulate` Sums and the registry's derived measures (`partition.h:29-40`; `measures.cpp`) |
| normalised cell value | power / (area × good samples) (`entities/atlas.ts:177-178`) | the same, at read |
| tile graph | `tile_adjacency` (nxr `tiles.py:87`; Studio) | `tileAdjacency` (`ladder.h:113`) |
| the band-pass measures | `power_sum`, `envelope_sum`, `envelope_max`, `samples` (`atlas.ts:36`) | the same names and merge rule |
| event area | `areaMM2 = Σ area of the UNIQUE spatial tiles` (`eventtable.m:58`) | the union of the event's tiles (plan, step 4); `tileAreas` (`ladder.h:109`) |
| event envelope | area-weighted mean over the event's tiles (`eventtable.m:47-48`) | `partition::accumulate` with the tile areas as weights |
| merge of events | detections touching in tile adjacency within a time gap join (`measure_events.m:235-248`) | the plan's merge in step 3 (gap in time, adjacency of tiles) |

### 4.2 Where the references differ from the decided rule (stated, NOT resolved: for the maintainers)
- **(a) The band.** The Studio's atlas puts the band in the row's `within` pair (temporal Hz bands, many bands on ONE spatial tiling, `atlas.ts:117-121`)
  and never pairs a spatial level with a spatial band. The rule says the spatial level is its band, so a row's key is (spatial level, time level).
  Under the rule there is no band axis in the joint array. The atlas's [band × tile × frame] becomes one array per (time level, spatial level) pair.
  This is the plan's decision 1, already decided; the audit confirms the atlas has nothing to port for the spatial band.
- **(b) One stored depth vs a ladder of joint levels.** The atlas stores ONE depth D and ONE frame level; coarser views are arithmetic roll-ups
  of the same array (`rollup.ts:1-3`). The rule says a spatial band is a level, so the band feature at level l is a different filtered quantity (a graph
  wavelet at its Weyl band), not the roll-up of the finest. Band power at one level is not derivable from another's: this is the Studio's own `time`
  measure span ("the level IS the band", `measurements.ts:44-47`). So per-cell band features are stored per (spatial level, time level),
  and a roll-up applies only to measures of one fixed pair (e.g. the cell's amplitude to a coarser tile within the same band), not across pairs.
- **(c) Frames vs the constant-Q time ladder.** The atlas's frame is a fixed `frame_s` window of the day-anchored tower, `2^m` consecutive frames
  per coarser tile (`atlas.ts:178-190`; `rollup.ts:7-9`). nxr-neuro's constant-Q ladder has one level per distinct window of its bands
  (`io/include/nxr/io/constant_q.h:3-9`), each level a closed-form tiling `{origin, width, hop, count}`, with a level the band's own. The cell's time
  index is then a tile of that level, and the code's low bits are a tile index of that level, not a base frame. Whether level widths halve exactly
  (so a coarser tile is a union of finer ones) is as the tiling says; the atlas's tower alignment (`code0`, partial first and last) is a separate
  convention that the constant-Q ladder may or may not carry.
- **(d) Space × time pairing.** The atlas pairs nothing: any (ℓ, m) is read. Rheome pairs them by SPEED (D/dt, `jointladder.m:17-23`) and by cycles per
  tile (22.6 in time, ~2 in space, `ladder.m:71-74`), and gives a spatial octave two depths (`ladder.m:13-17`). Under the rule (a level is half a
  spatial octave: `ladder.h` `lambdaMin/lambdaMax`, `tiles.h`) a joint level is a free PAIR, as plan step 1 says. Nothing in the references picks which pairs
  to compute; the speed of a cell pair (D/dt) is derivable, not a key.
- **(e) The "below the diagonal" rule is spatial (tile vs wavelength), not space-vs-time.** It forbids a cell whose tile is smaller than `Cycles`
  wavelengths of its band (`ladder.m:27-31, 121`). Under the rule a level's band λ and tile area are tied (λ = 4π·2^l/A), so each level's
  tile diameter in wavelengths is fixed by the tiling alone (`SurfaceLevel::diameterInWavelengths`, `ladder.h:70`); the rule then forbids no level
  by itself and only removes the Rheome case "band finer than its tile". The plan's description of this rule (decision 3) should be corrected.
  This audit proposes no limit: it is the maintainers' call whether the rule is a named option at all.
- **(f) Detection.** The Studio's `searchAtlas` is a per-cell threshold (above, below, top) with merge of times only, no connectedness
  (`entities/atlas.ts:187-201`). Rheome's events are tracker detections merged by touching tiles (`measure_events.m:235-248`); `occupancy.m` is time-only
  hysteresis by scale. Neither has seeds and growth over a cell graph, a minimum area, or connected components in space × time. The plan's
  `detect_cells` is therefore new, from `detect_spans` (nxr-neuro) extended with `tileAdjacency`, not a port. State it as such.
- **(g) Event area.** Rheome's event area is the union of the event's space tiles over its whole life (`eventtable.m:58`): no time weighting, one hemisphere.
  The plan also lists "the mean area per time tile". The latter has no reference (a new derived measure).
- **(h) The result.** The atlas's search result is a `spans` Selection in time only (`entities/atlas.ts:205-215`); a labelling of cells by event has
  no counterpart in the Studio or Rheome (an event is a row with a list of tile indices, `measure_events.m:226-232`). The plan's `set`
  labelling (code = event ordinal, −1 elsewhere) is not a port. The time projection (`searchAtlas.spans`) is a port.

### 4.3 What nxr-neuro already has
| need | where |
|---|---|
| the surface ladder, Weyl pairing, areas, adjacency, tree codes | `compute/include/nxr/ladder.h:36-37` (weyl), `:64-84` (`SurfaceLevel`: `lambdaMin/Max`, `wavelength`, `diameterInWavelengths`), `:100` (`Tree::at`), `:109-113` (`tileAreas`, `tileAdjacency`) |
| the ladder stored | `pipelines/include/nxr/pipelines/tiles.h:3-13` (`<surface> tiles` ladder; level Selections `tiles L<l>`; areas `w`; `lc/.../adjacency`), group tree `:40-52` (frame_level −18, default bands 1–64 Hz) |
| the spatial band filter at the ladder's levels | `nxr::filterbank::weyl(area, levels)` (`compute/include/nxr/filterbank.h:93`); the joint time-vertex bank `compute/include/nxr/jointbank.h` |
| the constant-Q time ladder | `io/include/nxr/io/constant_q.h:1-40`, closed-form tilings per level |
| reduction over a partition, any labelling | `nxr::partition::accumulate` (`compute/include/nxr/partition.h:40`): negative code = no member (`:36`), `medians` (`:45`) |
| the lookup's roll-up and `within` | `stored_query` with `shift` (`code >> shift`, `io/src/measures.cpp:70-100`) and `withinSelectionId`/`withinCode` (`:88-92`); the rows' schema carries `within_selection_id`, `within_code` (`:199-207`) |
| time-tile band features and detection | `band_bursts` (`pipelines/test/test_band_bursts.cpp`); `nxr::intervals` for `detect_spans` |
| cell tile sums | `reduceTiles` (`partition.h:63-66`): a segmented reduce over tiles of `width` samples, the Studio's `reduceTiles` |

Not in nxr-neuro: the array-backed [tile × frame] roll-up arithmetic of `rollup.ts` (nxr-neuro's `shift` is over per-member ROWS, not arrays), a joint
Selection (product manifold, cell `'0,1'`), a connected-component routine on a cell graph, and a spatial band measure per tile.

## 5. For the maintainers (the differences, not resolved)
1. The atlas has no spatial band; the rule is new relative to it (4.2 a, b). Band features are stored per (spatial level, time level).
2. The atlas's frame is a tower window, our time level is a constant-Q ladder level (4.2 c).
3. Rheome's "below the diagonal" rule is tile vs wavelength, not space vs time (4.2 e); the plan's decision 3 should be reworded before it is answered.
4. Detection on a cell graph and the event labelling have no reference (4.2 f, h): they are new in nxr-neuro.
5. Event area: the union of the event's spatial tiles (Rheome, `eventtable.m:58`) is ported; the mean area per time tile is not in any reference (4.2 g).
