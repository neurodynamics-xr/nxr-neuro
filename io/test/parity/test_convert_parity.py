"""Parity of the Brainstorm importer: `nxr-neuro convert …` does what `nxr convert …` does, on SYNTHETIC protocols (octahedra and icospheres written with
scipy.io.savemat by nxr's own test helpers — nothing real).

Run with nxr's environment and the built tool:

    NXR=<path to nxr> NXR_NEURO=<build>/cli/nxr-neuro uv run --project $NXR python -m pytest -q test/parity/test_convert_parity.py

For each scenario the same command runs through nxr's Python and through nxr-neuro, with ids a counter and a fixed clock on both sides, into two copies of one
starting dataset; then every row of every table, every zarr.json (as text) and every chunk file (bytes) are compared. `list` and `scan` outputs are compared
as text (a scan's `scanned_at` is the one thing masked: it is the clock).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from spellings import to_io  # noqa: E402

NXR = Path(os.environ["NXR"]) if "NXR" in os.environ else None
IO = Path(os.environ["NXR_NEURO"]) if "NXR_NEURO" in os.environ else None
pytestmark = pytest.mark.skipif(NXR is None or IO is None, reason="set NXR and NXR_NEURO")

if NXR is not None:
    import convert_fixtures as cf  # noqa: E402
    import writer_scenario as ws  # noqa: E402

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
STAMP = re.compile(r"nxr-(convert|io|neuro) [0-9.]+ / (Python [0-9.]+|C\+\+17)")
SCANNED = re.compile(r'"scanned_at": "[^"]*"')


def run_io(*args, start=None, ok=None):
    env = {**os.environ, **({"NXR_IO_DETERMINISTIC": str(start)} if start is not None else {})}
    p = subprocess.run([str(IO), *to_io(args)], capture_output=True, text=True, env=env)
    if ok is True:
        assert p.returncode == 0, p.stdout + p.stderr
    return p


def run_nxr(*args, start=None, ok=None):
    code = ("import sys; sys.path.insert(0, %r); import writer_scenario as ws; %s from nxr.cli.main import main; sys.exit(main(sys.argv[1:]))"
            % (str(HERE), f"ws.deterministic({start});" if start is not None else ""))
    p = subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True)
    if ok is True:
        assert p.returncode == 0, p.stdout + p.stderr
    return p


def lines(p) -> list[dict]:
    return [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]


def masked_text(s: str) -> str:
    """the producing tool's name and a scan's clock hidden — nothing else: the ids are a counter on both sides and compare as they are"""
    return STAMP.sub("<STAMP>", SCANNED.sub('"scanned_at": "<T>"', s))


# ── datasets, and their comparison ─────────────────────────────────────────────────────────────────────────────────

def starting_pair(tmp: Path, subjects=("S01",)) -> tuple[Path, Path]:
    """one dataset (`demo`, subjects made empty), copied: the same state for both tools"""
    from nxr.io.writer import SubjectWriter, create_dataset
    ws.deterministic(0)
    ds = create_dataset(tmp / "py", "demo", source_tool="parity", created_by="parity")
    for s in subjects:
        SubjectWriter.create(ds, s, created_by="parity").complete()
    ds.close()
    shutil.copytree(tmp / "py", tmp / "io")
    return tmp / "py" / "demo", tmp / "io" / "demo"


def dump(folder: Path) -> tuple[dict, dict]:
    rows = ws.dump_rows(folder)
    files = ws.dump_files(folder.parent)
    return rows, files


def assert_same_dataset(a: Path, b: Path, expect_rows: int = 10) -> None:
    ra, fa = dump(a)
    rb, fb = dump(b)
    assert ra["_user_version"] == rb["_user_version"] == 51
    for t in sorted(set(ra) | set(rb)):
        x, y = json.loads(masked_text(json.dumps(ra[t], sort_keys=True))), json.loads(masked_text(json.dumps(rb[t], sort_keys=True)))
        if isinstance(x, list):                       # sorted again, the producing tool's name hidden (rows written by the two tools sort alike)
            x, y = (sorted(v, key=lambda d: json.dumps(d, sort_keys=True)) for v in (x, y))
        assert x == y, f"table {t} differs:\n{x}\n{y}"
    assert sum(len(v) for k, v in ra.items() if isinstance(v, list)) >= expect_rows
    assert sorted(fa) == sorted(fb), set(fa) ^ set(fb)
    for k in fa:
        if fa[k][0] == "text":
            assert masked_text(fa[k][1]) == masked_text(fb[k][1]), f"{k} differs as text"
        else:
            assert fa[k] == fb[k], f"chunk bytes of {k} differ"


# ── list and scan ──────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def protocol(tmp_path_factory):
    root = tmp_path_factory.mktemp("proto") / "bst"
    cf.build_protocol(root)
    return root


@pytest.fixture(scope="module")
def protocol_walk(tmp_path_factory):
    """no protocol.mat; a subject with only a kernel, one with a plain condition of excerpts, a stray @-folder, v7 (compressed) tess files"""
    root = tmp_path_factory.mktemp("walk") / "bst"
    cf.build_protocol(root, compress=True, index=False)
    s3 = root / "data" / "S03"
    (s3 / "K").mkdir(parents=True)
    from scipy.io import savemat
    savemat(str(s3 / "K" / "results_MN_MEG_KERNEL_1.mat"), {"SurfaceFile": "S03/tess_cortex_x.mat"})
    savemat(str(s3 / "K" / "channel_ctf.mat"), {"Comment": "c"})
    (s3 / "E").mkdir()
    savemat(str(s3 / "E" / "data_block001.mat"), {"F": np.zeros((2, 2))})
    (s3 / "@intra").mkdir()
    (root / "data" / "Group_analysis" / "@intra").mkdir(parents=True)
    (root / "data" / "empty_subject" / "C").mkdir(parents=True)
    return root


def both(*args, **kw):
    return run_nxr(*args, **kw), run_io(*args, **kw)


@pytest.mark.parametrize("which", ["protocol", "protocol_walk"])
def test_list_prints_what_nxr_prints(which, request):
    root = request.getfixturevalue(which)
    a, b = both("convert", "list", root)
    assert a.returncode == b.returncode == 0
    assert a.stdout == b.stdout
    assert len(json.loads(a.stdout)["subjects"]) >= 2


@pytest.mark.parametrize("which", ["protocol", "protocol_walk"])
@pytest.mark.parametrize("flags", [(), ("--prefer", "walk"), ("--prefer", "protocol.mat"), ("--summary",)])
def test_scan_prints_what_nxr_prints(which, flags, request):
    root = request.getfixturevalue(which)
    if which == "protocol_walk" and "protocol.mat" in flags:
        pytest.skip("no protocol.mat")
    a, b = both("convert", "scan", root, *flags)
    assert a.returncode == b.returncode == 0, a.stderr + b.stderr
    assert masked_text(a.stdout) == masked_text(b.stdout)
    if not flags:
        view = json.loads(a.stdout)
        assert view["schema"] == "nxr.source@1.0"


def test_scan_of_the_index_has_the_index_facts(protocol):
    """what only the index (not the walk) gives: comments, channels, modalities, the cortex iCortex names, the kernels' own surface, a listed-but-absent surface"""
    v = json.loads(run_io("convert", "scan", protocol, ok=True).stdout)
    assert v["kind"] == "protocol.mat" and v["db_version"] == 5.0 and v["protocol_comment"] == "synthetic"
    s01 = next(s for s in v["subjects"] if s["name"] == "S01")
    assert s01["cortex"] == "tess_cortex_mid_low.mat" and s01["scalp"] == "tess_head_mask.mat"
    assert "tess_missing.mat" in s01["discrepancy"] and [x["present"] for x in s01["surfaces"]].count(False) == 1
    c = s01["conditions"][1]
    assert c["n_channels"] == 6 and c["modalities"] == ["ECG", "EEG"] and c["primary_surface"] == "tess_cortex_mid_low.mat"
    assert c["recordings"][0]["raw_format"] == "BST-BIN" and c["recordings"][0]["sfreq"] == 100.0 and c["recordings"][0]["n_samples"] == 400
    s02 = next(s for s in v["subjects"] if s["name"] == "S02")
    assert s02["uses_default_anat"] and s02["surfaces"][0]["file"] == "tess_cortex_pial_low.mat"


@pytest.mark.parametrize("compress", [True, False], ids=["index_v7", "index_v6"])
def test_scan_of_a_compressed_and_an_uncompressed_index(compress, tmp_path):
    root = tmp_path / "bst"
    cf.build_protocol(root, compress=compress)
    a, b = both("convert", "scan", root)
    assert a.returncode == b.returncode == 0 and masked_text(a.stdout) == masked_text(b.stdout)
    assert json.loads(b.stdout)["kind"] == "protocol.mat"


def test_scan_out_writes_the_view_and_prints_the_counts(protocol, tmp_path):
    a, b = tmp_path / "py" / "view.json", tmp_path / "io" / "view.json"
    pa, pb = run_nxr("convert", "scan", protocol, "--out", a), run_io("convert", "scan", protocol, "--out", b, ok=True)
    assert lines(pa)[0]["wrote"] == str(a) and lines(pb)[0]["wrote"] == str(b)
    assert {k: v for k, v in lines(pa)[0].items() if k != "wrote"} == {k: v for k, v in lines(pb)[0].items() if k != "wrote"}
    assert masked_text(a.read_text()) == masked_text(b.read_text())
    pa, pb = run_nxr("convert", "scan", protocol, "--out", a, "--summary"), run_io("convert", "scan", protocol, "--out", b, "--summary")
    assert pa.stdout.replace(str(a), "F") == pb.stdout.replace(str(b), "F")


def test_list_and_scan_of_a_non_protocol_say_the_same(tmp_path):
    """nxr lets the exception go (a traceback on stderr, its last line the message); nxr-neuro prints the same message as a `stage: error` line"""
    for cmd in ("list", "scan"):
        a, b = both("convert", cmd, tmp_path)
        assert a.returncode == b.returncode == 1
        assert lines(b) == [{"stage": "error", "message": a.stderr.strip().splitlines()[-1]}]


# ── surfaces: `convert surface` ────────────────────────────────────────────────────────────────────────────────────

def scenario_files(root: Path, compress: bool = True) -> dict[str, Path]:
    """tess files of every kind: cortex (two hemispheres, atlases, sphere), the same on v6, a plain octahedron, a mis-wound hemisphere, a head surface,
    an unhealthy mesh, a second cortex sharing the atlas dictionary (its scouts in another order), unicode labels"""
    a = root / "anat" / "A"
    out = {}
    pos, faces, n = cf.hemispheres_mesh(2)
    out["cortex"] = cf.write_tess(a / "tess_cortex_pial_low.mat", pos, faces, comment="pial", atlas_=cf.cortex_atlases(n), sphere=cf.sphere_for(pos), compress=compress)
    out["cortex_v6"] = cf.write_tess(a / "tess_cortex_v6.mat", pos, faces, comment="v6", atlas_=cf.cortex_atlases(n, order=1), sphere=cf.sphere_for(pos), compress=False)
    v, f = cf._surface()
    out["octahedron"] = cf.write_tess(a / "tess_cortex_octa.mat", v * 0.05, f.astype(int) - 1, comment="octahedron", compress=compress)
    pos2, faces2, n2 = cf.hemispheres_mesh(2, flip_second=True)
    out["miswound"] = cf.write_tess(a / "tess_cortex_miswound.mat", pos2, faces2, comment="second hemisphere inward", atlas_=cf.cortex_atlases(n2), compress=compress)
    hp, hf = cf.icosphere(2, 0.09)
    out["head"] = cf.write_tess(a / "tess_head_mask.mat", hp, hf, comment="head", compress=compress)
    bad = np.vstack([faces, faces[:1]])                                       # a duplicate face: warned, imported anyway
    out["unhealthy"] = cf.write_tess(a / "tess_cortex_unhealthy.mat", pos, bad, comment="duplicate face", compress=compress)
    uni = cf.atlas([("Désikan", [(list(range(1, 20)), "précentral L", [1, 0.25, 0]), (list(range(20, 40)), "pariétal R", [0, 0.5, 0.75])]),
                    ("Structures", [(list(range(1, n + 1)), "Cortex Left", [0.2, 0.4, 0.6]), (list(range(n + 1, 2 * n + 1)), "Cortex Right", [0.6, 0.4, 0.2])])])
    out["unicode"] = cf.write_tess(a / "tess_cortex_unicode.mat", pos, faces, comment="région été", atlas_=uni, compress=compress)
    # one hemisphere only in the Structures atlas (not two runs): no components; scouts out of order
    one = cf.atlas([("Structures", [(list(range(1, n + 1)), "Cortex L", [0.2, 0.4, 0.6])]), ("Parc", [(list(range(1, 11)), "a", [1, 1, 1]), (list(range(11, 21)), "b", [0, 0, 0])])])
    out["one_hemisphere"] = cf.write_tess(a / "tess_cortex_one.mat", pos, faces, comment="one", atlas_=one, sphere=cf.sphere_for(pos)[:-1], compress=compress)
    return out


def import_surfaces(tmp: Path, root: Path, names: list[str], files: dict, *, bst_root=True):
    a, b = starting_pair(tmp)
    args = ["--subject", "S01", "--file", *[files[n] for n in names], *(["--bst-root", root] if bst_root else [])]
    pa = run_nxr("convert", "surface", "--dataset", a, *args, start=1000)
    pb = run_io("convert", "surface", "--dataset", b, *args, start=1000)
    return a, b, pa, pb


SETS = {
    "cortex_with_atlases_sphere_hemispheres": ["cortex"],
    "v6_uncompressed": ["cortex_v6"],
    "plain_octahedron": ["octahedron"],
    "miswound_flipped": ["miswound"],
    "head_surface_goes_to_mri": ["head"],
    "unhealthy_is_warned_not_refused": ["unhealthy"],
    "unicode_labels": ["unicode"],
    "structures_with_one_hemisphere": ["one_hemisphere"],
    "two_cortices_share_the_atlas_dictionary": ["cortex", "cortex_v6"],
    "everything_at_once": ["cortex", "octahedron", "miswound", "head", "unicode", "cortex_v6"],
}


@pytest.mark.parametrize("scenario", list(SETS))
def test_convert_surface_is_nxrs(scenario, tmp_path):
    root = tmp_path / "proto"
    files = scenario_files(root)
    a, b, pa, pb = import_surfaces(tmp_path, root, SETS[scenario], files)
    assert pa.returncode == pb.returncode == 0, pa.stdout + pa.stderr + pb.stdout + pb.stderr
    la, lb = lines(pa), lines(pb)
    assert la == lb, (la, lb)
    assert_same_dataset(a, b)
    from nxr.io.database import open_database, text_parity
    from nxr.io.writer import open_writer
    with open_writer(b, lock=False) as ds:
        assert text_parity(ds.db, ds.root.parent if False else ds.root)["mismatched"] == 0


def test_the_scenarios_hold_what_they_say(tmp_path):
    """the comparison is only worth something if the scenarios have the features: components, a sphere, atlases, a flipped component, a head surface"""
    root = tmp_path / "proto"
    files = scenario_files(root)
    a, b, pa, pb = import_surfaces(tmp_path, root, SETS["everything_at_once"], files)
    out = {x["name"]: x for x in lines(pb) if x["stage"] == "surface"}
    assert out["cortex_pial_low"]["hemispheres"] == 2 and out["cortex_pial_low"]["registration"] and out["cortex_pial_low"]["atlases"] == 2
    assert out["cortex_miswound"]["flipped"] == [1] and out["cortex_octa"]["atlases"] == 0 and not out["cortex_octa"]["registration"]
    assert out["head_mask"]["folder"] == "mri" and out["cortex_unicode"]["atlases"] == 2
    import sqlite3
    con = sqlite3.connect(str(b / "dataset.sqlite"))
    assert con.execute("SELECT count(*) FROM manifold WHERE name LIKE '%_sphere'").fetchone()[0] >= 2
    assert con.execute("SELECT count(*) FROM selection WHERE path LIKE '%_atlases/%'").fetchone()[0] >= 4
    assert con.execute("SELECT count(*) FROM dictionary WHERE scope = 'app'").fetchone()[0] >= 3


def test_convert_surface_without_a_bst_root_has_no_source(tmp_path):
    root = tmp_path / "proto"
    files = scenario_files(root)
    a, b, pa, pb = import_surfaces(tmp_path, root, ["cortex", "head"], files, bst_root=False)
    assert pa.returncode == pb.returncode == 0
    assert_same_dataset(a, b)


def test_convert_surface_refuses_a_missing_file_like_nxr(tmp_path):
    a, b = starting_pair(tmp_path)
    args = ["--subject", "S01", "--file", tmp_path / "nope.mat"]
    pa, pb = run_nxr("convert", "surface", "--dataset", a, *args, start=1000), run_io("convert", "surface", "--dataset", b, *args, start=1000)
    assert pa.returncode == pb.returncode == 1 and lines(pa) == lines(pb)
    assert "not a file" in lines(pb)[-1]["message"]


def test_a_failed_import_leaves_nothing(tmp_path):
    """the second file is bad: the composition is rolled back whole, as in nxr (the first surface is not kept)"""
    root = tmp_path / "proto"
    files = scenario_files(root)
    bad = root / "anat" / "A" / "tess_cortex_bad.mat"
    from scipy.io import savemat
    savemat(str(bad), {"Vertices": np.zeros((4, 3)), "Comment": "no faces"})
    a, b = starting_pair(tmp_path)
    args = ["--subject", "S01", "--file", files["cortex"], bad]
    pa, pb = run_nxr("convert", "surface", "--dataset", a, *args, start=1000), run_io("convert", "surface", "--dataset", b, *args, start=1000)
    assert pa.returncode == pb.returncode == 1
    assert lines(pa)[-1] == lines(pb)[-1] and lines(pb)[-1]["message"] == "KeyError: 'Faces'"
    import sqlite3
    for d in (a, b):
        assert sqlite3.connect(str(d / "dataset.sqlite")).execute("SELECT count(*) FROM manifold").fetchone()[0] == 0
    ra, rb = ws.dump_rows(a), ws.dump_rows(b)
    assert masked_text(json.dumps(ra, sort_keys=True)) == masked_text(json.dumps(rb, sort_keys=True))


def test_a_fibres_file_is_not_a_surface(tmp_path):
    from scipy.io import savemat
    f = tmp_path / "tess_fibers_x.mat"
    savemat(str(f), {"Points": np.zeros((2, 3, 3))})
    a, b = starting_pair(tmp_path)
    pa = run_nxr("convert", "surface", "--dataset", a, "--subject", "S01", "--file", f)
    pb = run_io("convert", "surface", "--dataset", b, "--subject", "S01", "--file", f)
    assert pa.returncode == pb.returncode == 1
    assert "fibres are curves, not a surface" in lines(pa)[-1]["message"] and "fibres are curves, not a surface" in lines(pb)[-1]["message"]


def test_an_empty_comment_is_no_comment_where_nxr_fails(tmp_path):
    """A DIFFERENCE, on purpose. pymatreader returns an empty char as an empty array, and nxr's `if surf.get("Comment")` then raises ("truth value of an empty
    array is ambiguous"); matio returns the empty char as stored, so nxr-neuro imports the surface with no comment."""
    v, f = cf._surface()
    t = cf.write_tess(tmp_path / "proto" / "anat" / "A" / "tess_cortex_c.mat", v * 0.05, f.astype(int) - 1, comment="")
    a, b = starting_pair(tmp_path)
    args = ["--subject", "S01", "--file", t]
    pa, pb = run_nxr("convert", "surface", "--dataset", a, *args, start=1000), run_io("convert", "surface", "--dataset", b, *args, start=1000)
    assert pa.returncode == 1 and "truth value of an empty array is ambiguous" in lines(pa)[-1]["message"]
    assert pb.returncode == 0
    import sqlite3
    assert sqlite3.connect(str(b / "dataset.sqlite")).execute("SELECT comment FROM manifold WHERE name = 'cortex_c'").fetchone() == (None,)


# ── a subject's anatomy: `convert subject` ─────────────────────────────────────────────────────────────────────────


def nxr_anatomy_only(dataset: Path, root: Path, subject: str) -> list[dict]:
    """What `nxr convert subject` does for the anatomy, through nxr's own functions: the subject row (source = the anatomy folder), then the primary surface
    the first importable condition's plan names (its kernels' SurfaceFile, else iCortex, else the pial literal), as one composition. (The sessions, images,
    maps and fibres are what nxr-neuro reports as not ported yet.)"""
    from nxr.convert.subject import _conditions, anat_dir, index_cortex, kernel_surface, stamp
    from nxr.convert.surface import export_surface
    from nxr.io.writer import SubjectWriter, open_writer
    ws.deterministic(1000)
    root = Path(root)
    ds = open_writer(dataset)
    anat, _ = anat_dir(root, subject)
    sub = SubjectWriter.create(ds, subject, source_format="brainstorm", source_path=str(anat), created_by=stamp())
    convertible, _ = _conditions(root, subject)
    first = convertible[0] if convertible else None
    name = (kernel_surface(root / "data" / subject / first, root / "data" / subject / f"@raw{first}", anat) if first else None) \
        or index_cortex(root, subject) or "tess_cortex_pial_low.mat"
    with sub.composition():
        info = export_surface(sub, anat / name, bst_root=str(root), primary=True)
    ds.close()
    return [info]



def nxr_subject(dataset: Path, root: Path, subject: str) -> list[dict]:
    """`nxr convert subject` itself (ids a counter, a fixed clock)"""
    p = run_nxr("convert", "subject", root, "--subject", subject, "--dataset", dataset, start=1000)
    assert p.returncode == 0, p.stdout + p.stderr
    return lines(p)


@pytest.mark.parametrize("compress", [True, False], ids=["v7", "v6"])
@pytest.mark.parametrize("subject", ["S01", "S02"])
def test_convert_subject_is_nxrs(subject, compress, tmp_path):
    """the whole subject: the anatomy, every session (channels, recording with its grid and envelope, flags, events), in sorted order; a protocol without kernels"""
    root = tmp_path / "proto"
    cf.build_protocol(root, compress=compress, kernels=False)
    a, b = starting_pair(tmp_path, subjects=())
    out_a = nxr_subject(a, root, subject)
    pb = run_io("convert", "subject", root, "--subject", subject, "--dataset", b, start=1000)
    assert pb.returncode == 0, pb.stdout + pb.stderr
    assert_same_dataset(a, b)
    out = lines(pb)
    strip = lambda L: [{k: v for k, v in x.items() if k not in ("store", "id")} for x in L if x["stage"] not in ("result",)]
    assert strip(out_a) == strip(out)
    assert out[-1]["sessions"] == (["sub-S01_task-rest_run-01", "sub-S01_task-rest_run-02"] if subject == "S01" else ["C"])
    surf = next(x for x in out if x["stage"] == "surface")
    assert surf["name"] == ("cortex_mid_low" if subject == "S01" else "cortex_pial_low") and surf["hemispheres"] == 2


def test_convert_subject_reports_the_parts_it_does_not_port(tmp_path):
    root = tmp_path / "proto"
    cf.build_protocol(root)       # with kernels and head models
    (root / "anat" / "S01" / "subjectimage_MRI_T1.mat").write_bytes(b"x")
    from scipy.io import savemat
    (root / "data" / "S01" / "PET").mkdir()
    savemat(str(root / "data" / "S01" / "PET" / "results_surface_PET.mat"), {"ImageGridAmp": np.zeros((6, 1))})
    _, b = starting_pair(tmp_path, subjects=())
    pb = run_io("convert", "subject", root, "--subject", "S01", "--dataset", b, ok=True)
    skipped = {x["what"]: x["reason"] for x in lines(pb) if x["stage"] == "skipped" and "what" in x}
    # integrated importer: sessions (kernels and head models included), maps and fibres are imported, not reported; the unreadable images are reported as nxr reports them
    assert "subject images" in skipped and skipped["subject images"].startswith("no subject image imported from ") and "subjectimage_MRI_T1.mat" in skipped["subject images"]
    assert [k for k in skipped if k.startswith("results_")] == ["results_surface_PET.mat"] and "no SurfaceFile" in skipped["results_surface_PET.mat"]   # a map with no surface, reported as nxr reports it
    res = lines(pb)[-1]
    assert res["stage"] == "result" and res["sessions"] == ["sub-S01_task-rest_run-01", "sub-S01_task-rest_run-02"]
    # and a fibres file is not taken for a surface by `convert surface`
    savemat(str(root / "anat" / "S01" / "tess_fibers_s.mat"), {"Points": np.zeros((1, 2, 3))})
    p = run_io("convert", "surface", "--dataset", b, "--subject", "S01", "--file", root / "anat" / "S01" / "tess_fibers_s.mat")
    assert p.returncode == 1


def test_convert_subject_with_kernels_and_head_models_is_nxrs(tmp_path):
    """the kernels, forward models and their provenance rows (created_utc/modified_utc, producer_json) are nxr's: rows, zarr.json text, chunk bytes"""
    root = tmp_path / "proto"
    cf.build_protocol(root)       # kernels and head models, sized to the cortex
    a, b = starting_pair(tmp_path, subjects=())
    pa = run_nxr("convert", "subject", root, "--subject", "S01", "--dataset", a, start=1000)
    pb = run_io("convert", "subject", root, "--subject", "S01", "--dataset", b, start=1000)
    assert pa.returncode == pb.returncode == 0, (pa.stdout[-600:], pb.stdout[-600:])
    assert any(x["stage"] == "inverse" for x in lines(pb))
    assert_same_dataset(a, b)


def test_convert_subject_without_a_condition_is_refused_like_nxr(tmp_path):
    """nxr refuses a subject it cannot anchor in a session; so does the integrated importer (the same message)"""
    root = tmp_path / "proto"
    cf.build_anatomy_only(root)
    a, b = starting_pair(tmp_path, subjects=())
    pn = run_nxr("convert", "subject", root, "--subject", "A01", "--dataset", a)
    pb = run_io("convert", "subject", root, "--subject", "A01", "--dataset", b, start=1000)
    assert pn.returncode == pb.returncode == 1
    assert "nothing to anchor a subject" in lines(pn)[-1]["message"] and "nothing to anchor a subject" in lines(pb)[-1]["message"]


def test_convert_subject_refuses_an_absent_anatomy_like_nxr(tmp_path):
    root = tmp_path / "proto"
    cf.build_protocol(root)
    (root / "anat" / "S01" / "tess_cortex_mid_low.mat").unlink()
    a, b = starting_pair(tmp_path, subjects=())
    pb = run_io("convert", "subject", root, "--subject", "S01", "--dataset", b)
    assert pb.returncode == 1 and "subject has no anatomy (tess_cortex_mid_low.mat not found under anat/)" in lines(pb)[-1]["message"]
    import sqlite3
    assert sqlite3.connect(str(b / "dataset.sqlite")).execute("SELECT count(*) FROM subject").fetchone()[0] == 0     # the composition was rolled back whole


# ── v7.3 (HDF5) files are read (was: refused, naming the file) ──────────────────────────────────────────────────────

def test_a_v73_surface_imports_as_its_v7_twin(tmp_path):
    """changed meaning: this was `test_a_v73_file_is_refused_with_its_name`. The same cortex, saved by MATLAB as -v7.3, imports to the same dataset
    (same rows, same arrays) as the v7 file; nxr's Python (pymatreader over h5py) reads the v7.3 file too and is compared with it."""
    root = tmp_path / "proto"
    files = scenario_files(root)
    v73 = cf.to_v73(files["cortex"], tmp_path / "v73" / "tess_cortex_pial_low.mat")
    a, b_v7 = starting_pair(tmp_path / "a")
    _, b_v73 = starting_pair(tmp_path / "b")
    pa = run_io("convert", "surface", "--dataset", b_v7, "--subject", "S01", "--file", files["cortex"], start=1000)
    pb = run_io("convert", "surface", "--dataset", b_v73, "--subject", "S01", "--file", v73, start=1000)
    assert pa.returncode == pb.returncode == 0, pa.stdout + pa.stderr + pb.stdout + pb.stderr
    assert_same_dataset(b_v7, b_v73)
    pn = run_nxr("convert", "surface", "--dataset", a, "--subject", "S01", "--file", v73, start=1000)
    assert pn.returncode == 0, pn.stdout + pn.stderr
    assert_same_dataset(a, b_v73)


def test_an_hdf5_file_that_is_not_a_matlab_file_is_refused_with_its_name(tmp_path):
    """changed meaning: was `..._without_the_matlab_header_is_refused_too` ("v7.3" in the message). A plain HDF5 file is no v7.3 file; matio says so."""
    import h5py
    f = tmp_path / "tess_plain_hdf5.mat"
    with h5py.File(f, "w") as h:
        h.create_dataset("Vertices", data=np.zeros((3, 3)))
    _, b = starting_pair(tmp_path)
    p = run_io("convert", "surface", "--dataset", b, "--subject", "S01", "--file", f)
    assert p.returncode == 1 and str(f) in lines(p)[-1]["message"] and "MAT-file" in lines(p)[-1]["message"]
    import sqlite3
    assert sqlite3.connect(str(b / "dataset.sqlite")).execute("SELECT count(*) FROM manifold").fetchone()[0] == 0


def test_a_v73_kernel_is_read_by_the_scan_and_the_subject(tmp_path):
    """changed meaning: was `test_a_v73_kernel_stops_the_subject_but_not_the_scan`. The kernel, saved by MATLAB as -v7.3, names its SurfaceFile to the scan
    and lets the subject import, as its v7 twin does (nxr's Python reads it too)."""
    root = tmp_path / "proto"
    cf.build_protocol(root)
    kernel = root / "data" / "S01" / "@rawsub-S01_task-rest_run-01" / "results_MN_EEG_KERNEL_000000_0000.mat"
    twin = tmp_path / "kernel_v7.mat"
    shutil.copy(kernel, twin)
    cf.to_v73(twin, kernel)
    v = json.loads(run_io("convert", "scan", root, ok=True).stdout)
    s01 = next(s for s in v["subjects"] if s["name"] == "S01")
    assert s01["conditions"][0]["primary_surface"] == "tess_cortex_mid_low.mat" and s01["conditions"][1]["primary_surface"] == "tess_cortex_mid_low.mat"
    vn = json.loads(run_nxr("convert", "scan", root, ok=True).stdout)
    assert [c["primary_surface"] for c in next(s for s in vn["subjects"] if s["name"] == "S01")["conditions"]] == [c["primary_surface"] for c in s01["conditions"]]
    _, b = starting_pair(tmp_path, subjects=())
    p = run_io("convert", "subject", root, "--subject", "S01", "--dataset", b)
    assert p.returncode == 0, p.stdout + p.stderr


def test_an_unreadable_hdf5_raw_link_lists_without_a_format(tmp_path):
    root = tmp_path / "proto"
    cf.build_protocol(root)
    link = root / "data" / "S02" / "@rawC" / "data_0raw_C.mat"
    link.unlink()
    cf.write_v73(link)
    c = json.loads(run_io("convert", "list", root, ok=True).stdout)["subjects"][1]["conditions"][0]
    assert c["raw_links"] == ["data_0raw_C.mat"] and c["raw_format"] is None


def test_the_importers_not_ported_say_so(tmp_path):
    for cmd in ("qc", "migrate", "mne-meg"):            
        p = run_io("convert", cmd, "x")
        assert p.returncode == 1 and "is not ported yet" in lines(p)[-1]["message"] and lines(p)[-1]["message"].startswith("NotImplementedError: ")
