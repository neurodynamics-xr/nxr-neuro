# Importing and converting data

nxr-neuro imports other formats **natively**, with no MATLAB, Brainstorm or other external program at run time: the readers are C++ in
`io/` and run through the command line (`nxr-neuro convert …`).

| Source | Reader | What is imported |
|---|---|---|
| Brainstorm protocols (`.mat`, read with matio) | `io/src/bst_*.cpp` (`nxr/bst/*.h`) | anatomy and surfaces (winding made canonical, `io/src/surface.cpp`), recordings, inverse kernels, maps, events, fibers (`io/src/bst_fibers.cpp`) |
| meshes (OBJ, PLY, STL) | `io/src/mesh*.cpp` | surfaces |
| FreeSurfer subject (`lh.pial`, `sphere.reg`, `.annot`, `.label`) | `io/src/mesh.cpp`, `io/convert/templates.cpp` | the surfaces and registered sphere of a template (fsaverage), the parcellations named by the caller, the cortex mask |
| Brainstorm default anatomy (ICBM152) | `io/convert/templates.cpp` over `io/src/bst_subject.cpp` | the cortex surfaces with their registered spheres, the parcellations named by the caller, the MNI-to-anatomy frame |
| DSI Studio / TrackVis `.trk` | `io/convert/trk.cpp`, `io/convert/templates.cpp` | the streamlines of HCP-1065 in the anatomy's coordinates and their ends on a surface (distance gate) |

Planned, also native: BIDS (MEG/EEG), FIF, FreeSurfer subjects, GIFTI/CIFTI; and exports to standard formats (see
`docs/specs/2026-11-16-distribution-and-interoperability.md`, sections 3 and 5).

Exporters that run inside another program (for example a Brainstorm plug-in that writes nxr datasets) belong to that program, not to this
repository.

## Template packages

`nxr-neuro create template --recipe fsaverage|icbm152 --sources DIR --package P ...` builds a template package from a folder of the source files you downloaded
(`nxr/convert/templates.h`), and `release template` freezes it. Rules every importer follows:

- **Every item it writes carries an `origin` record** (source, licence, attribution, whether it may be redistributed); the streamlines also carry the `nxr-origin/1`
  document in their producer record (citations, licence, terms, processing steps, the checksums of the files read).
- **Parcellations only when named** (`--parcellation NAME`, repeatable); a parcellation whose terms are not confirmed is written as not redistributable, and the atlases
  that must never be redistributed (Hammers, LPBA40) are never kept in a redistributable edition.
- **The source files are checked against `io/convert/manifest/template_sources.json`** (file names, SHA-256, where to download, licence); a file whose checksum differs
  or that is not listed is refused unless `--accept-unlisted-sources` is given, and the checksum actually read is recorded in the origin.
- The `.trk` reader follows the TrackVis header and reads points into RAS mm as nibabel's streamline reader does; the streamlines are resampled to equal arc length
  (both ends kept exactly), shifted to MNI by the half-voxel shift validated for this file family, and moved into the anatomy by its own MNI and SCS frames.
