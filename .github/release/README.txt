nxr-neuro: the command line toolkit for measuring the spatiotemporal patterns of cortical activity from MEG/EEG.

This folder holds one file to run:

  nxr-neuro            the command (a single self-contained binary)
  LICENSE              BSD-3-Clause
  THIRD_PARTY_NOTICES.md   the licences of the software it contains

Install: put `nxr-neuro` on your PATH, for example

  mkdir -p ~/.local/bin && cp nxr-neuro ~/.local/bin/

Check it:

  nxr-neuro --version
  nxr-neuro --help
  nxr-neuro compute --help

Linux: x86_64, glibc 2.28 or newer (any distribution from 2018 on). It needs no other library.
macOS: Apple silicon (arm64), macOS 12 or newer. The binary is not signed or notarised yet; if macOS refuses to open it, run
`xattr -d com.apple.quarantine nxr-neuro` once (see the release notes in the repository's README).

Source, documentation and issues: https://github.com/neurodynamics-xr/nxr-neuro
