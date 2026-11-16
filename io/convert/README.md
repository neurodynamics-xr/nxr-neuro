# Importing and converting data

nxr-neuro imports other formats **natively**, with no MATLAB, Brainstorm or other external program at run time: the readers are C++ in
`io/` and run through the command line (`nxr-neuro convert …`).

| Source | Reader | What is imported |
|---|---|---|
| Brainstorm protocols (`.mat`, read with matio) | `io/src/bst_*.cpp` (`nxr/bst/*.h`) | anatomy and surfaces (winding made canonical, `io/src/surface.cpp`), recordings, inverse kernels, maps, events, fibers (`io/src/bst_fibers.cpp`) |
| meshes (OBJ, PLY, STL) | `io/src/mesh*.cpp` | surfaces |

Planned, also native: BIDS (MEG/EEG), FIF, FreeSurfer subjects, GIFTI/CIFTI; and exports to standard formats (see
`docs/specs/2026-11-16-distribution-and-interoperability.md`, sections 3 and 5).

Exporters that run inside another program (for example a Brainstorm plug-in that writes nxr datasets) belong to that program, not to this
repository.
