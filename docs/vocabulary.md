# Vocabulary

One meaning per word, for nxr-neuro and the programs that read it. **Status: the target vocabulary.** Code, schema keys and documents adopt it
during the consolidation; until a part is converted it may still use a retired word (the table at the end says what each one becomes).

## Spaces

| word | means |
|---|---|
| **manifold** | A space as a chain complex: its cells of every rank, and the boundary map between consecutive ranks. A surface, a volume, a line (time, frequency, eigenvalues), a set of points (channels), or a **product** of these (channels x time). |
| **cell** | One element of a manifold: a vertex, an edge, a face or a tetrahedron. |
| **rank** | A cell's dimension: 0, 1, 2, 3. On a product, one rank per factor. A **dual cell** stands for a cell of the complementary rank (a vertex's Voronoi area). |
| **vertex, edge, face** | The cells of rank 0, 1 and 2. On a line, a vertex is a sample (a time sample, a frequency bin, an eigenmode) and an edge the interval between two consecutive ones. |
| **index** | A cell's address: its number among the cells of its rank, at its level. |
| **level** | A resolution of a manifold, itself a manifold. **Level 0** is the manifold as sampled or meshed, its finest discretization. A coarser level comes with its **reduction** operator from the finer level and its **expansion** operator back. |
| **width** | How much of the finer level each cell of a coarser level takes in (a kernel's width). Width divided by spacing is how much neighbouring cells overlap. |

The cells of a cell are its boundary, rank by rank: a face is bounded by edges, an edge by two vertices. Each rank's cells are stored as their
vertex lists, which are also the boundary map; the discrete derivatives are its transpose (d0: vertices to edges, d1: edges to faces).

## Coordinates

| word | means |
|---|---|
| **coordinates** | Each vertex's position. **Stored per vertex** where irregular (a surface's positions, the eigenvalues). **Origin and spacing** where regular: vertex i is at origin + i x spacing. A level's resolution is its spacing: 2 Hz on a linear frequency level, 1/8 octave on a logarithmic one. |
| **coordinate system** | What coordinates are measured in: its axes, linear or logarithmic, and its unit (metres, seconds, hertz). |
| **tangent frame** | The two tangent directions at each vertex of a surface. |

## The four classes

| word | means | its type |
|---|---|---|
| **manifold** | As above. | Its cells of each rank, its coordinates, its level (and the manifold it is a level of). |
| **field** | Values on the cells of a manifold. | **Manifold** (with its factors), **rank** (one per factor), **level**; then the **number type**, the **components** per cell, what geometric components are expressed in (**a tangent frame or a coordinate system**), the **unit** and the **quantity**. |
| **operator** | A linear map from the cells of one (manifold, rank, level) to the cells of another: across manifolds (an inverse kernel, the eigenvectors), across ranks (d0, d1), across levels (**reduction**, **expansion**). | **From:** manifold, rank, level. **To:** manifold, rank, level. Its number type, its components per cell at each end, its unit in and out. |
| **selection** | A set of cells: picking only. | The manifold, rank and level of the cells it holds. |

## Level operators and measurements

| word | means |
|---|---|
| **reduction** | The operator from a finer level to a coarser one: each row a coarse cell, holding its finer cells' weights. **Sharp** when each finer cell is in one coarse cell (weights 0 or 1, or 1/n for a mean); **soft** when rows overlap (kernel weights: frequency kernels, wavelets). |
| **expansion** | The operator from a coarser level back to a finer one: the synthesis. It undoes the reduction when the kernels form a partition of unity; otherwise it is the reduction's recorded pair. |
| **measurement** | A **field on a coarser level**, made by reducing a field with a named measure: the reduction operator applied to the field (sum, mean), or to a pointwise function of it (power: the mean of the squares), or, for order statistics (maximum, median), computed over each coarse cell's finer cells from the reduction's pattern. Its provenance names the field, the reduction and the measure. A ratio of measurements is derived when read. |

## Values

| word | means |
|---|---|
| **number type** | Real, complex or quaternion. |
| **components** | How many numbers each cell holds: 1 (a scalar), 2 (a tangent vector), 3 (a vector in a coordinate system). |
| **quantity** | What the values are: power, current, phase current. |
| **unit** | The values' unit: T, nA·m, m/s. |

## Storage, not scale

| word | means |
|---|---|
| **chunk** | The unit a recording is loaded in: 2^15 samples (by construction, the cells of level 15 of the time line). |
| **shard** | The unit a recording is stored in: 2^17 samples. |
| **overlap** | The samples a chunk's transform takes in beyond the chunk, on each side, from the neighbouring chunks (mirrored at the recording's two ends), so that a kernel applied to the chunk is correct inside it; the overlapping samples are dropped from the chunk's results. Stated in samples. Neighbouring chunks' transforms then share samples, as overlapping segments do in Welch's estimate: with an overlap of half a chunk on each side the transform is two chunks long (a power of two) and every sample is taken in exactly twice. |

## Recordings

| word | means |
|---|---|
| **channel** | One recorded signal: a row of a recording (MEG, reference, EOG, ECG, stimulus). Data. |
| **sensor** | A physical detector with a position and an orientation (an MEG gradiometer or magnetometer, an EEG electrode). Geometry. A sensor surface's vertices are sensors; a selection says which channel records each. |

## Retired words and what they become

| retired | becomes |
|---|---|
| domain, target (in types) | manifold, rank, level; from / to for an operator's ends |
| frame (on its own) | tangent frame; coordinate system; or a level of the time line |
| tile | a cell of a coarser level |
| band, sub-band | a cell of a level of a spectral line (frequency, eigenvalues). Names of particular cells stay ("the alpha band") |
| partition, kernel set | a coarser level and its reduction operator (sharp, soft) |
| separable / joint level | a product of levels / a level of a product manifold |
| voices per octave, cells per octave | a level's spacing on a logarithmic axis |
| member, atom, leaf, rung | a cell of a level, or a level |
| pyramid, ladder (of levels), roll-up | the levels of a manifold and their reductions |
| bank, filterbank, joint bank | the kernels of a soft reduction operator |
| segment, window (as time units) | a cell of a level of the time line; overlapping ones (Welch's) are a soft level |
| scale (as a noun for a level) | level |
| halo | overlap |
| gather | a measurement |
| cell (anything other than a manifold's cell) | the word above that fits |
