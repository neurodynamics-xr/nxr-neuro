"""Synthetic Brainstorm protocols for the importer's parity tests — built with nxr's own test helpers (`tests/convert/eeg_fixture.py`,
`atlas_meshes.py`) and scipy.io.savemat. Nothing real: octahedra and icospheres.

Importing this needs nxr's environment (`uv run --project $NXR`) and `NXR` set (the helpers live in `$NXR/tests/convert`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

NXR = Path(os.environ["NXR"])
sys.path.insert(0, str(NXR / "tests" / "convert"))
from atlas_meshes import icosphere, two_hemispheres  # noqa: E402
from eeg_fixture import _surface, make_eeg_protocol  # noqa: E402


def scouts(rows):
    """Brainstorm scouts: (vertices 1-based, label, colour 0-1)"""
    return np.array([(np.asarray(v, float), lab, np.asarray(c, float)) for v, lab, c in rows],
                    dtype=[("Vertices", "O"), ("Label", "O"), ("Color", "O")])


def atlas(entries):
    """entries: [(name, scouts rows)]"""
    return np.array([(n, scouts(r)) for n, r in entries], dtype=[("Name", "O"), ("Scouts", "O")])


def write_tess(path: Path, vertices, faces0, *, comment="", atlas_=None, sphere=None, compress=True, extra=None):
    """A tess_*.mat: Vertices (nV x 3), Faces (1-based), Comment, Atlas, and Reg.Sphere.Vertices when `sphere` is given.
    compress=True is MATLAB's v7 (compressed v5 file), False is v6 (uncompressed)."""
    d = {"Vertices": np.asarray(vertices, float), "Faces": np.asarray(faces0, float) + 1, "Comment": comment,
         "Atlas": atlas_ if atlas_ is not None else np.array([], dtype=object)}
    if sphere is not None:
        d["Reg"] = {"Sphere": {"Vertices": np.asarray(sphere, float)}}
    d.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(path), d, do_compression=compress)
    return path


def hemispheres_mesh(level=2, flip_second=False):
    """two icosphere hemispheres as one cortex: (vertices, 0-based faces, per-hemisphere vertex runs)"""
    pos, faces, _ = two_hemispheres(level)
    n = len(pos) // 2
    faces = np.array(faces)
    if flip_second:
        faces[len(faces) // 2:] = faces[len(faces) // 2:][:, ::-1]
    return pos, faces, n


def cortex_atlases(n_per_hemi, *, order=0):
    """Desikan-Killiany (3 regions, each a vertex range), a one-region atlas (not taken), and the L/R Structures"""
    n = n_per_hemi
    third = n // 3
    dk = [(list(range(1, third + 1)), "frontal L", [1, 0, 0]),
          (list(range(third + 1, 2 * third + 1)), "parietal L", [0, 1, 0]),
          (list(range(n + 1, n + third + 1)), "occipital R", [0, 0, 0.5])]
    if order:
        dk = dk[::-1]
    structures = [(list(range(1, n + 1)), "Cortex L", [0.2, 0.4, 0.6]), (list(range(n + 1, 2 * n + 1)), "Cortex R", [0.6, 0.4, 0.2])]
    one = [(list(range(1, 6)), "only", [0.1, 0.1, 0.1])]
    return atlas([("Desikan-Killiany", dk), ("Lonely", one), ("Structures", structures)])


def sphere_for(vertices):
    v = np.asarray(vertices, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True) * 0.1


# ── the protocol's index (data/protocol.mat), as Brainstorm writes it ──────────────────────────────────────────────

SURF = [("FileName", "O"), ("Comment", "O"), ("SurfaceType", "O")]


def surface_list(rows):
    return np.array(rows, dtype=SURF)


def write_index(root: Path, subjects: list[dict], studies: list[dict], default_surfaces=None, comment="synthetic", compress=False):
    """subjects: dict(name, surfaces=[(file, comment, type)], i_cortex, i_scalp, use_default)
       studies: dict(subject, condition, raw(bool), channel=(file, n, modalities) | None, data=[(file, comment)], kernels=[(file, comment)], heads=[(file, comment)])"""
    sub_dt = [("Name", "O"), ("FileName", "O"), ("Surface", "O"), ("iCortex", "O"), ("iScalp", "O"), ("UseDefaultAnat", "O")]
    rows = []
    for s in subjects:
        rows.append((s["name"], f"{s['name']}/brainstormsubject.mat", surface_list(s["surfaces"]) if s["surfaces"] else np.array([]),
                     np.array([[s["i_cortex"]]], float) if s.get("i_cortex") else np.array([]),
                     np.array([[s["i_scalp"]]], float) if s.get("i_scalp") else np.array([]),
                     np.array([[1.0 if s.get("use_default") else 0.0]])))
    default = np.array([("@default_subject", "@default_subject/brainstormsubject.mat", surface_list(default_surfaces or []) if default_surfaces else np.array([]),
                         np.array([[1.0]]), np.array([]), np.array([[0.0]]))], dtype=sub_dt)
    subjects_arr = np.array(rows, dtype=sub_dt)

    st_dt = [("Name", "O"), ("FileName", "O"), ("BrainStormSubject", "O"), ("Condition", "O"), ("Channel", "O"), ("Data", "O"), ("HeadModel", "O"), ("Result", "O")]
    ch_dt = [("FileName", "O"), ("Comment", "O"), ("nbChannels", "O"), ("Modalities", "O")]
    d_dt = [("FileName", "O"), ("Comment", "O"), ("DataType", "O"), ("BadTrial", "O")]
    f_dt = [("FileName", "O"), ("Comment", "O")]
    srows = []
    for st in studies:
        cond = f"@raw{st['condition']}" if st.get("raw") else st["condition"]
        ch = np.array([(st["channel"][0], "channels", np.array([[st["channel"][1]]], float), np.array(st["channel"][2], dtype=object))], dtype=ch_dt) \
            if st.get("channel") else np.array([])
        data = np.array([(f, c, "recordings", np.array([[0.0]])) for f, c in st.get("data", [])], dtype=d_dt) if st.get("data") else np.array([])
        heads = np.array(st.get("heads", []), dtype=f_dt) if st.get("heads") else np.array([])
        res = np.array(st.get("kernels", []), dtype=f_dt) if st.get("kernels") else np.array([])
        srows.append((cond, f"{st['subject']}/{cond}/brainstormstudy.mat", f"{st['subject']}/brainstormsubject.mat",
                      np.array([cond], dtype=object), ch, data, heads, res))
    srows.append(("@default_study", "@default_study/brainstormstudy.mat", "@default_subject/brainstormsubject.mat", np.array(["@default_study"], dtype=object),
                  np.array([]), np.array([]), np.array([]), np.array([])))
    studies_arr = np.array(srows, dtype=st_dt)
    savemat(str(root / "data" / "protocol.mat"), {
        "ProtocolInfo": {"Comment": comment, "STUDIES": "x", "SUBJECTS": "y"},
        "ProtocolSubjects": {"Subject": subjects_arr, "DefaultSubject": default},
        "ProtocolStudies": {"Study": studies_arr},
        "DbVersion": 5.0,
    }, do_compression=compress)


# ── whole protocols ────────────────────────────────────────────────────────────────────────────────────────────────

def build_protocol(root: Path, *, compress=True, index=True, kernels=True) -> dict:
    """S01: two EEG conditions (the kernels sit on tess_cortex_mid_low), whose cortex is two hemispheres with atlases and a registration sphere, plus a
    pial octahedron and a head mask; S02 borrows the default anatomy and has one condition; the default anatomy holds a two-hemisphere cortex."""
    root = Path(root)
    run1, run2 = "sub-S01_task-rest_run-01", "sub-S01_task-rest_run-02"
    make_eeg_protocol(root, subject="S01", condition=run1, n_chan=8)
    make_eeg_protocol(root, subject="S01", condition=run2, n_chan=6)
    make_eeg_protocol(root, subject="S02", condition="C", n_chan=8, use_default_anat=True)
    anat = root / "anat" / "S01"
    pos, faces, n = hemispheres_mesh(2)
    write_tess(anat / "tess_cortex_mid_low.mat", pos, faces, comment="mid_%dV" % len(pos), atlas_=cortex_atlases(n), sphere=sphere_for(pos), compress=compress)
    hp, hf = icosphere(2, 0.09)
    write_tess(anat / "tess_head_mask.mat", hp, hf, comment="head mask", compress=compress)
    # the default anatomy: a cortex of two hemispheres, no registration
    d = root / "anat" / "@default_subject"
    d.mkdir(parents=True, exist_ok=True)
    write_tess(d / "tess_cortex_pial_low.mat", pos, faces, comment="default pial", atlas_=cortex_atlases(n, order=1), compress=compress)
    savemat(str(d / "brainstormsubject.mat"), {"UseDefaultAnat": np.array([[0.0]])})
    info = {"root": root, "run1": run1, "run2": run2}
    if not kernels:     # sessions without kernels and head models: what nxr-io's session import reproduces exactly
        for f in list((root / "data").rglob("results_*KERNEL*.mat")) + list((root / "data").rglob("headmodel_*.mat")):
            f.unlink()
    else:               # the kernels and head models come sized for the 6-vertex octahedron: resize them to the cortex they sit on (nxr checks rows against vertices)
        for f in (root / "data").rglob("headmodel_*.mat"):
            m = loadmat(str(f), squeeze_me=False)
            n_chan = m["Gain"].shape[0]
            m.update(GridLoc=pos, GridOrient=pos, Gain=np.zeros((n_chan, 3 * len(pos))))
            savemat(str(f), {k: v for k, v in m.items() if not k.startswith("__")})
        for f in (root / "data").rglob("results_*KERNEL*.mat"):
            m = loadmat(str(f), squeeze_me=False)
            n_chan = m["ImagingKernel"].shape[1]
            m["ImagingKernel"] = np.zeros((3 * len(pos), n_chan))
            savemat(str(f), {k: v for k, v in m.items() if not k.startswith("__")})
    if index:
        eeg = lambda c, s="S01": [(f"{s}/@raw{c}/results_MN_EEG_KERNEL_000000_0000.mat", "MN: EEG(Unconstr) 2018")]
        write_index(root, [
            {"name": "S01", "surfaces": [("S01/tess_cortex_mid_low.mat", "cortex_mid", "Cortex"), ("S01/tess_cortex_white_high.mat", "white", "Cortex"),
                                         ("S01/tess_cortex_pial_low.mat", "pial", "Cortex"), ("S01/tess_head_mask.mat", "head", "Scalp"),
                                         ("S01/tess_missing.mat", "gone", "Other")], "i_cortex": 1, "i_scalp": 4},
            {"name": "S02", "surfaces": [], "use_default": True},
        ], [
            {"subject": "S01", "condition": run1, "channel": (f"S01/@raw{run1}/channel.mat", 8, ["EEG"]), "data": []},
            {"subject": "S01", "condition": run1, "raw": True, "channel": (f"S01/@raw{run1}/channel.mat", 8, ["EEG"]),
             "data": [(f"S01/@raw{run1}/data_0raw_{run1}.mat", "Raw")], "kernels": eeg(run1), "heads": [(f"S01/@raw{run1}/headmodel_surf_openmeeg.mat", "OpenMEEG BEM")]},
            {"subject": "S01", "condition": run2, "raw": True, "channel": (f"S01/@raw{run2}/channel.mat", 6, ["EEG", "ECG"]),
             "data": [(f"S01/@raw{run2}/data_0raw_{run2}.mat", "Raw")], "kernels": eeg(run2), "heads": [(f"S01/@raw{run2}/headmodel_surf_openmeeg.mat", "OpenMEEG BEM")]},
            {"subject": "S02", "condition": "C", "raw": True, "channel": ("S02/@rawC/channel.mat", 8, ["EEG"]), "data": [("S02/@rawC/data_0raw_C.mat", "Raw")],
             "kernels": eeg("C", "S02"), "heads": []},
        ], default_surfaces=[("@default_subject/tess_cortex_pial_low.mat", "default pial", "Cortex")], compress=compress)
    return info


def build_anatomy_only(root: Path, *, compress=True) -> dict:
    """A subject with an anat folder and a condition-less data folder: nothing for nxr's `convert subject` to anchor on (it refuses); the anatomy alone for nxr-io"""
    root = Path(root)
    (root / "data" / "A01").mkdir(parents=True)
    pos, faces, n = hemispheres_mesh(1)
    anat = root / "anat" / "A01"
    write_tess(anat / "tess_cortex_pial_low.mat", pos, faces, comment="pial", atlas_=cortex_atlases(n), compress=compress)
    savemat(str(anat / "brainstormsubject.mat"), {"UseDefaultAnat": np.array([[0.0]])})
    return {"root": root}


def to_v73(src: Path, dst: Path) -> Path:
    """`src` (a v5/v7 .mat) re-saved by MATLAB itself as a genuine v7.3 file: the same variables, `save -v7.3`. Skips the test when MATLAB is not
    found (set NXR_MATLAB, or have `matlab` on PATH or at ~/.local/bin/matlab). One MATLAB process, a few seconds."""
    import shutil
    import subprocess
    import pytest
    exe = os.environ.get("NXR_MATLAB") or shutil.which("matlab") or str(Path.home() / ".local/bin/matlab")
    if not Path(exe).exists():
        pytest.skip("MATLAB not found (NXR_MATLAB): a genuine v7.3 file needs it")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = f"s = load('{src}'); save('{dst}', '-struct', 's', '-v7.3');"
    p = subprocess.run([exe, "-batch", cmd], capture_output=True, text=True)
    assert p.returncode == 0 and dst.exists(), p.stdout + p.stderr
    return dst


def write_v73(path: Path) -> Path:
    """an HDF5 file with MATLAB's v7.3 header (512-byte user block holding the 128-byte header) but none of MATLAB's variable attributes
    (no MATLAB_class): no MAT-file reader makes anything of it. A genuine v7.3 file is `to_v73`."""
    import h5py
    with h5py.File(path, "w", userblock_size=512) as f:
        f.create_dataset("Vertices", data=np.zeros((3, 3)))
        f.create_dataset("Faces", data=np.array([[1, 2, 3.0]]))
    head = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Mon Nov  9 10:00:00 2026 HDF5 schema 1.00 ."
    head = head.ljust(124, b" ") + b"\x00\x02IM"
    with open(path, "r+b") as fh:
        fh.write(head)
    return path


# ── sessions: a raw link + a .bst, imported excerpts, events — SYNTHETIC ───────────────────────────────────────────

def cellstr(*items):
    c = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        c[i] = x
    return c


def make_events(rows, channels=None):
    """rows: [(label, colour, times)] with times a (1 or 2) x n matrix; channels: {label: [names per occurrence]} -> a Brainstorm events struct array"""
    dt = [("label", "O"), ("color", "O"), ("epochs", "O"), ("times", "O"), ("channels", "O"), ("notes", "O")]
    ev = np.zeros((len(rows),), dtype=dt)
    for i, (lab, col, times) in enumerate(rows):
        t = np.atleast_2d(np.asarray(times, float))
        per = (channels or {}).get(lab)
        ch = cellstr(*[np.array(p, dtype=object) if isinstance(p, (list, tuple)) else p for p in per]) if per else np.empty(0, object)
        ev[i] = (lab, np.asarray(col, float), np.ones(t.shape[1]), t, ch, np.empty(0, object))
    return ev


def write_bst(path: Path, data: np.ndarray, epoch: int, hdr: int = 8, truncate_last: int = 0) -> None:
    """a BST-BIN: hdr bytes, then epochs of [nchannels x epoch] float32; the last epoch is stored full-size (padded with zeros)"""
    n_chan, n_t = data.shape
    n_ep = -(-n_t // epoch)
    with open(path, "wb") as fh:
        fh.write(b"\0" * hdr)
        for e in range(n_ep):
            blk = np.zeros((n_chan, epoch), "<f4")
            part = data[:, e * epoch:(e + 1) * epoch]
            blk[:, :part.shape[1]] = part
            fh.write(blk.tobytes())
    if truncate_last:
        with open(path, "r+b") as fh:
            fh.truncate(path.stat().st_size - truncate_last)


def session_data(n_chan: int, n_t: int, seed: int = 7) -> np.ndarray:
    """float32 noise with a dead stretch (an all-zero inner chunk is not written) on channel 1 and a NaN-free ramp on channel 0"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_chan, n_t)).astype(np.float32) * np.float32(1e-12)
    x[0] = np.linspace(-1, 1, n_t, dtype=np.float32)
    if n_chan > 1 and n_t > 40000:
        x[1, :40000] = 0
    return x


def build_session_protocol(root: Path, *, subject="S01", condition="rest", n_chan=4, n_t=1000, sfreq=100.0, epoch=300, names=None, events=None,
                           flags=None, channel_mat=None, imported=None, raw=True, original_format=None, relink=False, kernels=False, hdr=8, seed=7,
                           truncate_last=0) -> dict:
    """S01 with one condition: the anatomy (an octahedron cortex), the channel file, and either a raw link with its .bst (`raw`) or imported excerpts
    (`imported`: [(file stem, F matrix, extra variables)]); `events` goes into the link's F.events; `relink`: the link records a path that is gone"""
    root = Path(root)
    names = names or [f"EEG{i:03d}" for i in range(n_chan)]
    anat = root / "anat" / subject
    anat.mkdir(parents=True, exist_ok=True)
    savemat(str(anat / "brainstormsubject.mat"), {"UseDefaultAnat": np.array([[0.0]])})
    v, f = _surface()
    write_tess(anat / "tess_cortex_pial_low.mat", v, np.asarray(f) - 1, comment="pial_6V")
    cond = root / "data" / subject / (f"@raw{condition}" if raw else condition)
    cond.mkdir(parents=True, exist_ok=True)
    chan = channel_mat or {"Comment": "EEG channels", "Channel": np.array([(n, "EEG", np.zeros((3, 1)), np.array([]), np.array([])) for n in names],
                                                                          dtype=[("Name", "O"), ("Type", "O"), ("Loc", "O"), ("Orient", "O"), ("Weight", "O")])}
    savemat(str(cond / "channel.mat"), chan)
    flag = np.ones((len(names), 1)) if flags is None else np.asarray(flags, float).reshape(-1, 1)
    info = {"root": root, "condition": condition, "dir": cond, "names": names}
    if raw:
        data = session_data(n_chan, n_t, seed)
        bst = cond / f"{condition}.bst"
        write_bst(bst, data, epoch, hdr, truncate_last)
        info.update(data=data, bst=bst)
        link = {"format": original_format or "BST-BIN", "filename": str(bst), "device": "CTF",
                "prop": {"sfreq": sfreq, "times": np.array([0.0, (n_t - 1) / sfreq]), "currCtfComp": np.array([]), "destCtfComp": np.array([])},
                "header": {"nchannels": n_chan, "nsamples": n_t, "epochsize": epoch, "hdrsize": hdr}}
        if events is not None:
            link["events"] = events
        savemat(str(cond / f"data_0raw_{condition}.mat"), {"F": link, "ChannelFlag": flag, "Comment": "Raw | notch(60Hz)"})
        if relink:
            link["filename"] = "/gone/away/" + bst.name
            savemat(str(cond / f"data_0raw_{condition}.mat"), {"F": link, "ChannelFlag": flag, "Comment": "Raw | notch(60Hz)"})
    for stem, F, extra in imported or []:
        t = np.arange(F.shape[1]) / sfreq
        savemat(str(cond / f"data_{stem}.mat"), {"F": F, "Time": t.reshape(1, -1), "ChannelFlag": flag, "Comment": stem, "DataType": "recordings", **extra})
    if kernels:
        n_src = v.shape[0]
        savemat(str(cond / "results_MN_EEG_KERNEL_000000_0000.mat"), {"Comment": "MN: EEG(Unconstr) 2018", "ImagingKernel": np.zeros((3 * n_src, n_chan)),
                                                                      "SurfaceFile": f"{subject}/tess_cortex_pial_low.mat"})
    return info
