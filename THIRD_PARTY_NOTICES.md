# Third-party notices

nxr-neuro is licensed under BSD-3-Clause (`LICENSE`). It uses the following third-party software. Each keeps its own licence and copyright; the
licence texts are in the source trees named below or at the project URLs. Versions are those pinned in the CMake files.

| Component | Version | How it comes in | Licence | Copyright |
|---|---|---|---|---|
| Eigen | 3.4.0 (SHA256 pinned in `cmake/eigen.cmake`) | fetched at configure time (a system Eigen can be named) | MPL-2.0 | the Eigen authors, https://eigen.tuxfamily.org |
| Spectra | v1.2.0 | fetched (`compute/CMakeLists.txt`) | MPL-2.0 | Yixuan Qiu and the Spectra contributors, https://github.com/yixuan/spectra |
| geometry-central | git submodule `compute/deps/geometry-central` | submodule | MIT | Copyright (c) 2017-2019 Nicholas Sharp and the geometry-central contributors |
| pocketfft (C++ header-only) | commit `4d865cba24359b05b37da4e75aee24498c3b78e7` | vendored, `compute/deps/pocketfft/` (licence: `LICENSE.md` there) | BSD-3-Clause | Copyright (C) 2010-2018 Max-Planck-Society |
| nlohmann/json | 3.11.3 | vendored, `io/extern/nlohmann/json.hpp` | MIT | Copyright (c) 2013-2023 Niels Lohmann |
| rapidyaml (ryml, single header) | 0.15.2 | vendored, `io/extern/ryml/` (licence: `LICENSE.txt` there) | MIT | Copyright (c) 2018, Joao Paulo Magalhaes |
| CLI11 (single header) | 2.4.2 | vendored, `cli/extern/CLI11/` (licence: `LICENSE` there) | BSD-3-Clause | Copyright (c) 2017-2024 University of Cincinnati, developed by Henry Schreiner under NSF award 1414736 |
| c-blosc | v1.21.6 | fetched (`io/CMakeLists.txt`) | BSD-3-Clause | Copyright (c) Francesc Alted and the Blosc Development Team |
| zstd | the copy carried by c-blosc (always in the release binaries), or a system libzstd | linked | BSD-3-Clause (dual-licensed with GPL-2.0; used under BSD-3-Clause) | Copyright (c) Meta Platforms, Inc. and affiliates |
| zlib | system zlib (Emscripten's port in the WebAssembly build; zlib 1.3.1 built from source for the Linux release binary) | linked | zlib licence | Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler |
| SQLite | system library; the 3.51 amalgamation in the WebAssembly build and the release binaries (`NXR_NEURO_PORTABLE`) | linked / fetched | public domain | the SQLite authors, https://sqlite.org |
| HDF5 | hdf5_1.14.6 (built only when MATLAB v7.3 files are read) | fetched | BSD-3-Clause-style (HDF Group licence) | Copyright (c) The HDF Group; University of Illinois Board of Trustees |
| matio | v1.5.28 | fetched | BSD-2-Clause | Copyright (c) Christopher C. Hulbert and the matio contributors |

## Code bundled inside the components above

- **geometry-central** carries, in its own `deps/`, `happly` (MIT, Nick Sharp), `nanoflann` (BSD-2-Clause, Massimo Santoro-Fernandes, Jose Luis Blanco and
  contributors) and `nanort` (MIT, Syoyo Fujita). Their licence texts are in `compute/deps/geometry-central/deps/` and the geometry-central `LICENSE`.
- **rapidyaml** bundles c4core (MIT, Joao Paulo Magalhaes) and `debugbreak.h` (BSD-2-Clause, Copyright (c) 2011-2021 Scott Tsai); the notice is kept in
  the amalgamated header.
- **nlohmann/json** bundles small MIT-licensed parts (e.g. Grisu2 number printing by Florian Loitsch); see the header.

## Build and test tools (not distributed)

CMake and Ninja; Node.js with `cmake-js` and `node-addon-api` (MIT) for the Node binding; Emscripten (MIT/NCSA) for WebAssembly; pytest, NumPy and SciPy
(BSD-3-Clause) for the parity tests. Golden test data were generated with Brainstorm (GPL-3.0), Rheome and nxr used only as external reference programs: no
code of those programs is in this repository (see `CONTRIBUTING.md`, the method rule).

The licence texts of the MPL-2.0 components (Eigen, Spectra) require that their own source files stay available under the MPL-2.0; nxr-neuro does not modify them.
