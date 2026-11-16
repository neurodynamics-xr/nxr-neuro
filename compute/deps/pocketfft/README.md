# pocketfft (vendored)

The C++ header-only pocketfft (`pocketfft_hdronly.h`, BSD-3-Clause, Max-Planck-Society; the FFT inside NumPy >= 2 and SciPy), unmodified.

- source: https://github.com/mreineck/pocketfft, branch `cpp`, commit `4d865cba24359b05b37da4e75aee24498c3b78e7`
- `pocketfft_hdronly.h` sha256 `518a95c552aeaee352aa30122137cd9e135d23c91e2863d9272aaa758744c86d`; license: `LICENSE.md`
- used only through `src/fft.cpp` (`nxr::fft`, `include/nxr/fft.h`), compiled with `POCKETFFT_NO_MULTITHREADING` so no pthreads are needed
  (wasm, N-API workers). The header is not part of nxr_compute's public include path.
- to update: replace the header, update the commit and hash here, run `test_fft`.
