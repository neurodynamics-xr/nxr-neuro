# The tile grids as definitions: `tiles`, `group_tree`, `group_tiles` (spec, 2026-11-17)

Status: **implemented** (written, built in the compile round). They replace the hand-wired `tiles compute`, `group_tree_compute` and `group_tiles_compute` on the selection model
(`2026-11-17-selection-module.md`): the dyadic spatial tiles are a **grid** with a closed-form hierarchy, made by compute's dyadic builder and stored by io's `save_grid`.

## What a tile grid is

A partition of the surface defined before any data. Equal-area bisection (each tile halved by equal mass across its geodesic diameter, every tile one piece); level l has 2^l tiles per
connected piece, paired with the band it adds by Weyl's law, N(lambda) = A lambda / 4 pi. The member number is the finest tile code `h 2^D + t` (h the piece slot, t the tile), level l's member is
`code >> (D - l)`: nothing per level is held but its label l and its band. Empty tiles are empty members (counts are `slots << l`). Roll-ups are along the hierarchy only.

`params.grid` is the rule: `{kind: dyadic, depth D, slots, pieces}`. `slots` (at least 2) is what the codes reserve; `pieces` is the number of connected pieces (a one-piece surface has `slots` 2, `pieces` 1:
this was written as `pieces` where `slots` was meant, and is fixed).

## The three graphs

```
tiles        surface ─ tile_grid {depth} ─▶ tiles
group_tree   surface, frames? ─ tile_grid {tree: group, resolution, atlas_depth, frame_level, lc_levels} ─▶ tiles
group_tiles  surface (+ the default subject's group tree, bound by the runner) ─ group_tile_grid ─▶ tiles
```

| piece | where | rule |
|---|---|---|
| the grid | `tile_grid` calls `nxr::selection::dyadic(surface, {depth})`; `group_tile_grid` calls `nxr::ladder::groupTreeFor` (the transfer through the registered spheres) | compute: arrays in, a grid out |
| the levels' facts | area, tile centres, adjacency, the Weyl band per level (`lad::tileAreas / tileCentres / tileAdjacency`, `wavelengthOf`) | pure functions of the arrays; no loop over data |
| the Levi-Civita connection | `lad::leviCivita`, `antisymmetric` on the stored canonical frames (the optional input `frames`) | a value over a PAIR of tiles: a measurement over a selection of pairs (an `indices` selection, pair number a * tiles + b), not rows on the grid and not a reduction |
| the store | output type `grid`: io's `save_grid` (a `ladder` row, one `set` row per level, the tile codes as the member numbers, the level node's `w`, centres, adjacency); dictionaries as the Studio's | io: load and store only |

The runner reads the surface, the stored frames and (group tiles) the default subject through `Bound` (`coordinate_unit`, `subject_kind`, `group_template`); the steps hold no handle.

## Against the Studio

`backend/src/db/surface-ladder.ts` and `studio/src/engine/ladder.ts` keep their names here: `surfaceLadder` (the bisection), the levels with `lambda_min`, `lambda_max`, `wavelength_m`, `tau`, the ladder
dictionary (code = level, `measurement` = the level's wavelength, attributes = the level record), the level rows `<ladder name> L<l>` (type `set`, `member_ordinal` = code = l), the ladder row type `ladder`,
`tileAreas`, `tileCentres`. Differences, all older than this change: a tile dictionary names a tile `piece p tile t` (nxr's layout), up to 1024 tiles; the level record has no `finest_tiles`, `diameter_m` or area
spread (the Studio's dictionary carries them); positions are read as stored (float64), as the Studio's backend does.

## What it cannot resolve

A tile is as equal in area as the geodesic-diameter cut allows. The group tree is the registered sphere's correspondence (nearest vertex; the median and maximum chord are recorded). The Levi-Civita
connection between distant tiles depends on the path (recorded).

## Open points (for the maintainers)

- `finest` (the deepest level every piece reaches) is recorded as the depth written: compute's `dyadic` does not return it. With the default depth they are equal; with `--depth` smaller than the finest, `finest`
  is the depth. Exposing `finest` from `dyadic` (compute) would restore it.
- The default surface (the primary one) is not selectable from the definition: `--surface` names the surface (the library's own callers pass it).
- `--replace` of a grid moves the old rows' arrays and the level nodes aside (`_replaced-<date>/`, as every other output), it never deletes them.
