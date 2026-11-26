# Time is a mesh: lines without roles (amendment to the type-system spec, 2026-11-20)

Amends [2026-11-17-type-system.md](2026-11-17-type-system.md), sections 3.1 (manifolds as terms), 4 (generics) and 8 (what the runner checks).

## 1. The rule

A line is a 1-D mesh, with the same status as a surface mesh. A recording's time is a line. So are a frequency axis, a frame line and a mode axis.
A line's type says what the mesh is. It never says how a node uses it:

- **Topology.** A path (open: a recording's samples, a one-sided spectrum) or a cycle (periodic: a phase, a circular time window). The dataset model
  already stores this: `topology.kind` is `lattice`, `path` or `circle` at rank 1 (`io/schema/model.sql`). A recording's time line is written as a
  rank-1 lattice, a regular path.
- **Coordinate.** Origin, spacing and unit, in a coordinate system (`{time, clock, s}`, `{frequency, hz, Hz}`). An index line has no unit.
- **Provenance.** How it was made from another line, when it was: a frame line is tiled from a time line, and a frequency line is made by a transform of a time
  line. This is a relation, as `derived` relates the glued surface to the cortex.

Space-time is the product of meshes. A field on the cortex over time lies on cortex × time, and operators on the product come from its factors (on
cortex × time, the Laplacian is L_cortex ⊗ I + I ⊗ L_time). This is why time must be a mesh, not a tag: the product with time is what allows
space-time analysis.

## 2. What replaces each role

`Role` (`types.h`: none, time, frames, time_tiles, frequency, modes, levels) is removed from `Manifold`.

| role | it was | it becomes |
|---|---|---|
| `time` | the line a stream walks | a line whose coordinate is in seconds (`clock`, s) |
| `frequency` | a spectrum's axis | a line in Hz, made from a time line by a transform (relation recorded) |
| `frames` | a time line or a line of windows | a line in seconds made from a time line by tiling (relation `tiled`, its window and step); "time or frames" is "a line in seconds" |
| `modes` | an eigenmode axis | points with an eigenvalue coordinate (`spectrum:<operator>`), no topology of rank 1 |
| `time_tiles`, `levels` | the dyadic tile pyramid | not manifold ROWS: a level is a partition, held by the selection system (its levels and members, as today). **Amended 2026-11-26** ([2026-11-26-levels-of-one-manifold.md](2026-11-26-levels-of-one-manifold.md)): computed from that partition, each level is also a cell complex in its own right (neighbours, boundary, d0, ⋆0, ⋆1, halfedges where valid), addressed (manifold, level, code), never a manifold row or a name of its own |
| the factor a stream walks | the last factor when its role was time or frames | the run's: the chunk service walks the factor the field's port declares `streamed`; the checker requires that factor to be a line in seconds |

## 3. Signatures

- `time_var("T")` becomes a variable constrained to a line whose coordinate is in seconds.
  `frames_var("T")` is the same constraint: a frame line is a line in seconds. A node that needs raw samples rather than frames declares it with
  `samples: true`: the line must not be tiled from another line.
- A constraint on topology, where a node's mathematics needs it, is declared the same way. An FFT that assumes periodicity reads a cycle, or a path that
  it pads. A filter with zero-phase sections reads a path. A node never assumes a topology silently.
- External signatures (`type_json.h`) carry `topology: path | cycle | any` and `unit` instead of `role`. The JSON form of a line names its topology,
  coordinate system, origin, spacing and `made_from` relation.

## 4. The runner

At plan time the line's topology kind and coordinate system are read from the stored records (`topology.kind`, `geometry.coordinate_system_id`). Today
only the coordinate system is read, and `time` is inferred from it. A `lattice` of rank 1 is a regular path.

## 5. What it gives the other faces

- **nxr-charts.** `Line` draws a field on M × (a line). Its x axis is the line's coordinate (origin, spacing, unit), its trace labels come from M's names, and
  a cycle can be drawn wrapped. No role is needed.
- **The Studio.** `LinePort.role` (`time`, `spectral`, `parameter`) goes the same way when its ports are generated from this language.

## 6. Plan (one PR; scoped tests) - DONE 2026-11-20 (PR feat/time-as-a-mesh)

1. `types.h` and `types.cpp`: drop `Role`; add `topology` (path, cycle, none) and `made_from` (relation, of) to a line; equality includes topology.
2. `signatures.cpp` and `type_json.cpp`: time and frames variables become unit constraints; JSON in and out with `topology` and `unit`.
3. `checker.cpp` and `compute.cpp`: the streamed factor comes from the port; the made lines (Welch, transforms, tiling) record `made_from`.
4. The runner reads the stored topology kind.
5. Tests:
   - the existing `test_graph*` and `test_definitions_*` updated, with no change in what they accept or refuse;
   - new: a cycle refused by a path-only node;
   - new: a frame line accepted where a line in seconds is asked;
   - new: a tiled line refused where raw samples are asked.

## Checklist (the architecture policy)

- Types only: no node's mathematics changes, and no data is read beyond the records the runner already reads.
- Reductions stay with the selection system: the tile levels move out of the manifold types into it.
