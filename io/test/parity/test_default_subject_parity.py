"""The default subject: `nxr-neuro dataset create --templates` writes the FreeSurfer surfaces as nibabel reads them, and the parcellations as nxr writes them.

Needs the nxr-neuro build and nxr's environment:

    NXR=<path to nxr> NXR_NEURO=<build>/cli/nxr-neuro \\
      uv run --project $NXR python -m pytest -q --basetemp=<a scratch folder> test/parity/test_default_subject_parity.py

Synthetic data only: two octahedra (the hemispheres).

(b) `dataset create --templates …` writes @default_subject with the surfaces nibabel reads (metres; the sphere as stored);
(d) its parcellations are nxr's.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import writer_scenario as ws  # noqa: E402
from spellings import to_io  # noqa: E402

NXR = Path(os.environ["NXR"]) if "NXR" in os.environ else None
IO = Path(os.environ["NXR_NEURO"]) if "NXR_NEURO" in os.environ else None
pytestmark = pytest.mark.skipif(NXR is None or IO is None, reason="set NXR and NXR_NEURO (the nxr-neuro build)")
NXR_MAIN = "import sys; from nxr.cli.main import main; sys.exit(main(sys.argv[1:]))"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
STAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z")
TABLES = ("manifold", "field", "operator", "selection")


def run(cmd, *args):
    return subprocess.run([*cmd, *map(str, args)], capture_output=True, text=True)


def io(*args):
    return run([str(IO)], *to_io(args))


def nxr(*args):
    return run([sys.executable, "-c", NXR_MAIN], *args)


def mask(x, roots=()):
    s = json.dumps(x, sort_keys=True)
    for r in roots:
        s = s.replace(str(Path(r).resolve()), "<ROOT>").replace(str(r), "<ROOT>")
    s = UUID.sub("<ID>", s)
    s = STAMP.sub("<T>", s)
    s = re.sub(r"nxr-(convert|io|neuro) [0-9.]+ / (Python [0-9.]+|C\+\+17)", "<STAMP>", s)
    return json.loads(s)


def octahedron(cx, r):
    V = np.array([[cx + r, 0, 0], [cx - r, 0, 0], [cx, r, 0], [cx, -r, 0], [cx, 0, r], [cx, 0, -r]], float)
    F = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], np.int32)
    return V, F


def write_fs(path, V, F):
    with open(path, "wb") as f:
        f.write(b"\xff\xff\xfe" + b"created by the parity test\n\n")
        f.write(np.array([len(V), len(F)], ">i4").tobytes())
        f.write(V.astype(">f4").tobytes())
        f.write(F.astype(">i4").tobytes())


def hemispheres_mm():
    (lv, f), (rv, _) = octahedron(-30.0, 10.0), octahedron(30.0, 10.0)
    return lv, rv, f


def write_annots(label: Path):
    """aparc and aparc.a2009s for both octahedra: an `unknown` and a `Medial_wall` vertex, a region with no vertex, two colours per family"""
    import nibabel.freesurfer as nfs
    label.mkdir(parents=True, exist_ok=True)
    for annot, names in (("aparc", ["unknown", "frontal", "parietal", "Medial_wall", "unused"]),
                         ("aparc.a2009s", ["Unknown", "G_front", "S_central", "Medial_wall", "G_unused"])):
        ctab = np.array([[25, 5, 25, 0, 0], [220, 20, 10, 0, 0], [20, 220, 60, 0, 0], [80, 80, 80, 0, 0], [5, 5, 200, 0, 0]], np.int32)
        for h, labels in (("lh", [0, 1, 1, 2, 3, 2]), ("rh", [1, 2, 2, 1, 3, 0])):
            nfs.write_annot(str(label / f"{h}.{annot}.annot"), np.array(labels, np.int32), ctab, names, fill_ctab=True)


@pytest.fixture(scope="module", autouse=True, params=[pytest.param(51, id="f51"), pytest.param(52, id="f52")])
def store_format(request):
    """the store format every dataset of the module is made at (nxr's writer then follows the dataset's own)"""
    return request.param


@pytest.fixture(scope="module")
def base(tmp_path_factory, store_format):
    """a FreeSurfer subjects folder (fsaverage5 = two octahedra), a dataset whose subject S1 holds that pial as its primary surface"""
    root = tmp_path_factory.mktemp("default_subject")
    lv, rv, f = hemispheres_mm()
    surf = root / "templates" / "fsaverage5" / "surf"
    surf.mkdir(parents=True)
    for name, s in (("pial", 1.0), ("white", 0.9), ("inflated", 1.2)):
        for h, V, cx in (("lh", lv, -30.0), ("rh", rv, 30.0)):
            W = V.copy()
            W[:, 0] = cx + (W[:, 0] - cx) * s
            W[:, 1:] *= s
            write_fs(surf / f"{h}.{name}", W, f)
    sph, _ = octahedron(0.0, 100.0)
    write_fs(surf / "lh.sphere.reg", sph, f)
    write_fs(surf / "rh.sphere.reg", sph, f)
    write_annots(root / "templates" / "fsaverage5" / "label")
    pial_m = np.vstack([lv, rv]).astype(np.float32).astype(np.float64) / 1000.0
    faces = np.vstack([f, f + 6])
    (root / "pial.obj").write_text("".join(f"v {float(x)!r} {float(y)!r} {float(z)!r}\n" for x, y, z in pial_m) + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces))
    store = root / "store"
    assert io("dataset", "create", "--store-format", store_format, "--datastore", store, "--name", "d").returncode == 0
    assert io("subject", "create", "--dataset", store / "d", "--name", "S1").returncode == 0
    p = io("surface", "add", "--dataset", store / "d", "--subject", "S1", root / "pial.obj", "--name", "cortex")
    assert p.returncode == 0, p.stdout + p.stderr
    return root


def copy_of(base, name):
    shutil.copytree(base / "store", base / name)
    return base / name / "d"


def nodes(ds: Path, subject: str):
    """every manifold, field, operator and selection row of the subject, by path"""
    c = sqlite3.connect(f"file:{ds / 'dataset.sqlite'}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    sid = c.execute("SELECT id FROM subject WHERE name = ?", (subject,)).fetchone()["id"]
    out = {}
    for t in TABLES:
        for r in c.execute(f"SELECT * FROM {t} WHERE subject_id = ?", (sid,)):
            out[(t, r["path"])] = dict(r)
    return out


def test_b_dataset_create_writes_the_default_subject(base, store_format):
    import nibabel.freesurfer as nfs
    from nxr.io import small_files
    store = base / "created"
    p = io("dataset", "create", "--store-format", store_format, "--datastore", store, "--name", "d", "--templates", base / "templates", "--atlas-depth", "1", "--lc-levels", "1")
    assert p.returncode == 0, p.stdout + p.stderr
    r = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")][-1]
    st = store / "d" / "@default_subject.nxr.zarr"
    surf = base / "templates" / "fsaverage5" / "surf"
    for name in ("pial", "white", "inflated"):
        (lv, _), (rv, _) = nfs.read_geometry(str(surf / f"lh.{name}")), nfs.read_geometry(str(surf / f"rh.{name}"))
        got = np.asarray(small_files.open_array(st / f"source/cortex_{name}/vertices", mode="r")[...])
        assert np.array_equal(got, np.vstack([lv, rv]) / 1000.0), name
    sl, _ = nfs.read_geometry(str(surf / "lh.sphere.reg"))
    sph = np.asarray(small_files.open_array(st / "source/cortex_pial_sphere/vertices", mode="r")[...])
    assert np.array_equal(sph[:6], sl)
    c = sqlite3.connect(f"file:{store / 'd' / 'dataset.sqlite'}?mode=ro", uri=True)
    assert c.execute("SELECT kind, status FROM subject WHERE name = '@default_subject'").fetchone() == ("template", "complete")
    assert c.execute("SELECT is_primary FROM manifold WHERE name = 'cortex_pial'").fetchone() == (1,)
    # nxr reads it: the dataset's one template subject is the default subject
    import nxr.io as nio
    assert [t["name"] for t in nio.open(store / "d").subjects(kind="template")] == ["@default_subject"]


def test_d_the_parcellations_are_nxrs(base, store_format):
    """the pial's parcellations as nxr's DefaultSubjectWriter.parcellations writes them from the same annotation files: the set, each member's row,
    codes and dictionary (names, codes, colours)"""
    import nibabel.freesurfer as nfs
    from nxr.io import small_files
    from nxr.io.writer import SubjectWriter, open_writer
    from nxr.pipelines.atlas.default_subject import DefaultSubjectWriter
    a = base / "parc-io"
    p = io("dataset", "create", "--store-format", store_format, "--datastore", a, "--name", "d", "--templates", base / "templates", "--atlas-depth", "1", "--lc-levels", "1")
    assert p.returncode == 0, p.stdout + p.stderr
    b = base / "parc-nxr"
    assert io("dataset", "create", "--store-format", store_format, "--datastore", b, "--name", "d").returncode == 0
    fs0 = base / "templates" / "fsaverage5"
    (lv, lf), (rv, rf) = nfs.read_geometry(str(fs0 / "surf" / "lh.pial")), nfs.read_geometry(str(fs0 / "surf" / "rh.pial"))
    pos, faces, n_lh = np.vstack([lv, rv]) / 1000.0, np.vstack([lf, rf + len(lv)]), len(lv)
    ds = open_writer(b / "d")
    try:
        sub = SubjectWriter.create(ds, "@default_subject", kind="template", folder="@default_subject.nxr.zarr", source_format="freesurfer", source_path=str(fs0),
                                   template_id=None, created_by="parity")
        with sub.composition():
            W = DefaultSubjectWriter(sub, lambda *_: None)
            _, sid = W.surface("cortex_pial", pos, faces, None, n_lh, {"source": "fsaverage5 pial"}, primary=True)
            assert W.parcellations(fs0, "source/cortex_pial", sid, n_lh, len(pos)) == ["Structures", "Desikan-Killiany", "Destrieux"]
    finally:
        ds.close()

    def atlases(ds):
        c = sqlite3.connect(f"file:{ds / 'dataset.sqlite'}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        rows = {r["path"]: dict(r) for r in c.execute("SELECT * FROM selection WHERE path LIKE 'source/cortex_pial_atlases%'")}
        dicts = {}
        for path, r in rows.items():
            if r["dictionary_id"]:
                d = c.execute("SELECT name, scope FROM dictionary WHERE id = ?", (r["dictionary_id"],)).fetchone()
                ent = [tuple(e) for e in c.execute("SELECT code, name, color_json, ordinal FROM dictionary_entry WHERE dictionary_id = ? ORDER BY code", (r["dictionary_id"],))]
                dicts[path] = (tuple(d), ent)
        arrays = {path: np.asarray(small_files.open_array(ds / "@default_subject.nxr.zarr" / r["array_path"], mode="r")[...])
                  for path, r in rows.items() if r["array_path"]}
        return rows, dicts, arrays

    ra, da, xa = atlases(a / "d")
    rb, db, xb = atlases(b / "d")
    assert sorted(ra) == sorted(rb) == ["source/cortex_pial_atlases", "source/cortex_pial_atlases/Desikan-Killiany", "source/cortex_pial_atlases/Destrieux",
                                        "source/cortex_pial_atlases/Structures"]
    tool = ("created_by", "modified_by")                       # the writing tool's own stamp
    for path in ra:
        assert mask({k: v for k, v in ra[path].items() if k not in tool}, (a,)) == mask({k: v for k, v in rb[path].items() if k not in tool}, (b,)), path
    assert da == db
    assert sorted(xa) == sorted(xb)
    for path in xa:
        assert xa[path].dtype == xb[path].dtype and np.array_equal(xa[path], xb[path]), path
    # the exclusions: unknown and Medial_wall unassigned, the region with no vertex not a level
    dk = [e[1] for e in da["source/cortex_pial_atlases/Desikan-Killiany"][1]]
    assert dk == ["unassigned", "frontal L", "parietal L", "frontal R", "parietal R"]


def icosphere(levels):
    t = (1 + 5 ** 0.5) / 2
    V = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6],
         [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    V = [list(np.array(v) / np.linalg.norm(v)) for v in V]
    for _ in range(levels):
        mid, F2 = {}, []
        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = (np.array(V[a]) + np.array(V[b])) / 2
                V.append(list(p / np.linalg.norm(p)))
                mid[k] = len(V) - 1
            return mid[k]
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            F2 += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        F = F2
    return np.array(V), np.array(F, np.int32)


def test_e_the_group_tree_is_nxrs_ladder(tmp_path, store_format):
    """the default subject's group tree: a `ladder` Selection (D85) on the pial whose every level's codes are nxr's surface_ladder per hemisphere
    (h·2^l + t; nxr's ladder is the Studio's, bit for bit), `tree: "group"`, and the atlas definitions nxr's default subject declares"""
    import nibabel.freesurfer as nfs
    from nxr.io import small_files
    from nxr.geom.ladder import surface_ladder
    from nxr.pipelines.atlas.atlas_store import DEFAULT_BANDS
    from nxr.pipelines.atlas.tower import FRAME_LEVEL, period_s
    V, F = icosphere(3)                                         # 642 vertices a hemisphere
    rng = np.random.default_rng(3)
    surf = tmp_path / "templates" / "fsaverage5" / "surf"
    surf.mkdir(parents=True)
    for h, cx in (("lh", -40.0), ("rh", 40.0)):
        P = V * (60 + rng.uniform(-6, 6, (len(V), 1))) + [cx, 0, 0]   # a bumpy sphere (mm): the cuts are not ties
        for name in ("pial", "white", "inflated"):
            write_fs(surf / f"{h}.{name}", P, F)
        write_fs(surf / f"{h}.sphere.reg", V * 100, F)
    store = tmp_path / "store"
    p = io("dataset", "create", "--store-format", store_format, "--datastore", store, "--name", "d", "--templates", tmp_path / "templates", "--atlas-depth", "4")
    assert p.returncode == 0, p.stdout + p.stderr
    c = sqlite3.connect(f"file:{store / 'd' / 'dataset.sqlite'}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    lad = c.execute("SELECT * FROM selection WHERE type = 'ladder'").fetchall()
    assert len(lad) == 1 and lad[0]["name"] == "cortex_pial tiles" and lad[0]["member_of_id"] is None
    params = json.loads(lad[0]["params_json"])
    assert params["tree"] == "group"
    a = params["atlas"]
    assert a["default"] is True and a["tree"] == "group" and a["depth"] == 4 and a["resolution"] == "fsaverage5"
    fid = c.execute("SELECT id FROM field WHERE name = 'cortex_pial_canonical_frames'").fetchone()
    assert fid is not None and a["frames"] == fid[0]                     # the definitions name the default pial's canonical frames, as nxr's do
    assert a["frame_level"] == FRAME_LEVEL and a["frame_s"] == period_s(FRAME_LEVEL)
    assert a["bands_hz"] == [list(b) for b in DEFAULT_BANDS] and a["lc_levels"] == [4, 6] and a["trajectory"] == {"space_level": 4, "time_level": 2}
    levels = c.execute("SELECT * FROM selection WHERE member_of_id = ? ORDER BY member_ordinal", (lad[0]["id"],)).fetchall()
    ref = []
    for h in ("lh", "rh"):
        P, Fh = nfs.read_geometry(str(surf / f"{h}.pial"))
        ref.append(surface_ladder(P.astype(np.float64) / 1000.0, Fh))
    finest = min(len(r.labels) for r in ref) - 1
    assert len(levels) == finest + 1 and finest >= 4
    st = store / "d" / "@default_subject.nxr.zarr"
    for l, row in enumerate(levels):
        assert row["type"] == "set" and row["member_ordinal"] == l and row["name"] == f"cortex_pial tiles L{l}"
        got = np.asarray(small_files.open_array(st / row["array_path"], mode="r")[...])
        want = np.r_[ref[0].labels[l], 2 ** l + ref[1].labels[l]]
        assert np.array_equal(got, want), f"level {l}: {int((got != want).sum())} vertices differ"


def test_f_the_frames_and_the_levi_civita_connection(tmp_path, store_format):
    """the default subject's frames and the group tree's Levi-Civita connection, on two EXACT spheres (the pial IS the registered sphere, scaled), where both are known
    in closed form: north is the meridian towards +z; the parallel transport from tile centre a to centre b along the great circle is the rotation about a x b.
    - canonical frames: e1 the meridian, n the sphere's normal; the poles (two a hemisphere) singular;
    - the anatomical frame from the default subject's own axes (FreeSurfer's RAS);
    - the tile centres are nxr's `node_centres` (the member nearest the area-weighted centroid);
    - Omega[a, b] (a measurement on the level's Selection, code a, within_code b) is the exact transport's angle, and Omega[b, a] = -Omega[a, b].
    Proved able to fail: with nxr's `antisymmetric` (angle(e^{i Omega_ab} e^{-i Omega_ba}) / 2) every pair with |Omega| > 90 degrees is off by 180 (L4: 120 of 512 pairs)."""
    from nxr.io import small_files
    from nxr.geom.ladder import vertex_areas
    from nxr.geom.trees import Tree
    from nxr.measure.vectors import node_centres
    V, F = icosphere(3)
    n, R = len(V), 60.0
    C = {"lh": np.array([-40.0, 0, 0]), "rh": np.array([40.0, 0, 0])}
    surf = tmp_path / "templates" / "fsaverage5" / "surf"
    surf.mkdir(parents=True)
    for h in ("lh", "rh"):
        for name in ("pial", "white", "inflated"):
            write_fs(surf / f"{h}.{name}", V * R + C[h], F)
        write_fs(surf / f"{h}.sphere.reg", V * 100, F)
    store = tmp_path / "store"
    p = io("dataset", "create", "--store-format", store_format, "--datastore", store, "--name", "d", "--templates", tmp_path / "templates", "--atlas-depth", "4")
    assert p.returncode == 0, p.stdout + p.stderr
    st = store / "d" / "@default_subject.nxr.zarr"
    c = sqlite3.connect(f"file:{store / 'd' / 'dataset.sqlite'}?mode=ro", uri=True)
    arr = lambda path: np.asarray(small_files.open_array(st / path, mode="r")[...])
    fr = arr(c.execute("SELECT path FROM field WHERE name = 'cortex_pial_canonical_frames'").fetchone()[0]).astype(np.float64)
    e1, e2, nn = fr[:, 0], fr[:, 1], fr[:, 2]
    u = np.r_[V, V]
    sing = np.sort(arr(c.execute("SELECT array_path FROM selection WHERE name = 'cortex_pial_frame_singularities'").fetchone()[0]))
    poles = sorted([int(np.argmax(V[:, 2])), int(np.argmin(V[:, 2])), n + int(np.argmax(V[:, 2])), n + int(np.argmin(V[:, 2]))])
    assert sing.tolist() == poles
    ok = np.isfinite(e1[:, 0])
    assert ok.sum() == 2 * n - 4
    north = np.array([0, 0, 1.0]) - u[:, 2:3] * u
    north = north[ok] / np.linalg.norm(north[ok], axis=1, keepdims=True)
    assert np.degrees(np.arccos(np.clip((e1[ok] * north).sum(1), -1, 1))).max() < 1.0      # the meridian (the push-forward's least squares on a P1 mesh)
    assert (nn * u).sum(1).min() > 0.9999
    assert np.allclose(e2[ok], np.cross(nn[ok], e1[ok]), atol=1e-6)
    an = json.loads(c.execute("SELECT producer_json FROM field WHERE name = 'cortex_pial_anatomical_angle'").fetchone()[0])
    assert an["definition"] == "ras" and an["reference"] == "canonical frames"
    lad = c.execute("SELECT id, params_json FROM selection WHERE type = 'ladder'").fetchone()
    lc_levels = json.loads(lad[1])["atlas"]["lc_levels"]
    pos = np.r_[V * R + C["lh"], V * R + C["rh"]] / 1000.0
    area = vertex_areas(pos, np.r_[F, F + n])
    for l in lc_levels:
        sid, codes_path = c.execute("SELECT id, array_path FROM selection WHERE member_of_id = ? AND member_ordinal = ?", (lad[0], l)).fetchone()
        codes = arr(codes_path)
        cen = arr(c.execute("SELECT array_path FROM selection WHERE name = ?", (f"cortex_pial_lc_centres_L{l}",)).fetchone()[0])
        want = node_centres(Tree("group", l, codes, l), pos, area, l, ~ok)
        assert np.array_equal(cen, want), f"L{l}: centres differ from nxr's node_centres"
        M = np.full((len(cen), len(cen)), np.nan)
        for a_, b_, v, unit, w in c.execute("SELECT code, within_code, value, unit, within_selection_id FROM selection_measurement WHERE selection_id = ? AND measure = 'levi-civita connection'", (sid,)):
            assert unit == "rad" and w == sid
            M[a_, b_] = v
        same = (cen[:, None] < n) == (cen[None, :] < n)
        assert np.array_equal(np.isfinite(M), same), "a value for every pair of centres in one hemisphere, none across"
        assert np.allclose(M[same], -M.T[same], atol=1e-12)
        E = np.full_like(M, np.nan)
        for a_ in range(len(cen)):
            for b_ in range(len(cen)):
                pa, pb = u[cen[a_]], u[cen[b_]]
                ax = np.cross(pa, pb)
                s_ = np.linalg.norm(ax)
                if not same[a_, b_] or s_ < 1e-9 or pa @ pb < -0.5:        # (nearly) the same point, or past 120 degrees: the great circle is far from unique
                    continue
                k = ax / s_
                th = np.arctan2(s_, pa @ pb)
                v = e1[cen[a_]]
                Rv = v * np.cos(th) + np.cross(k, v) * np.sin(th) + k * (k @ v) * (1 - np.cos(th))
                E[a_, b_] = np.arctan2(Rv @ e2[cen[b_]], Rv @ e1[cen[b_]])
        m = np.isfinite(E)
        d = np.degrees(np.abs(np.angle(np.exp(1j * (M[m] - E[m])))))
        assert m.sum() > 100 and np.median(d) < 0.15 and d.max() < 1.0, f"L{l}: median {np.median(d):.3f}, max {d.max():.3f} degrees from the exact transport"
