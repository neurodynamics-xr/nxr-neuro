"""A catalogue member's content hash follows every write into its package, and `query measurements` reads and rolls up the measures their writers declare
(docs/reference.md, "The dataset module" and "Declared measures"). Synthetic data only: a dataset made here, an icosahedron, nxr-neuro's own synthetic recording."""

import numpy as np
import pytest

import nxr_neuro as nn


def icosahedron(radius=0.05):
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6],
                  [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], np.int32)
    return v * radius / np.linalg.norm(v[0]), f


def entry_hash(ds, name):
    return next(s["content_hash"] for s in nn.list.subjects(dataset=str(ds))["subjects"] if s["name"] == name)


def package_hash(ds, name):
    return nn.show.package(package=str(ds / f"{name}.nxr.zarr"))["record"]["content_hash"]


@pytest.fixture
def member(tmp_path):
    """a catalogue with the subject S1: the icosahedron `cortex` and a 4 s recording `rest` at 200 Hz"""
    ds = tmp_path / "study"
    nn.create.dataset(dataset=str(ds))
    nn.create.subject(dataset=str(ds), name="S1")
    v, f = icosahedron()
    nn.add.surface(dataset=str(ds), subject="S1", name="cortex", vertices=v, faces=f, primary=True)
    nn.create.recording(package=str(ds / "S1.nxr.zarr"), synthetic=True, name="rest", channels=4, sfreq=200.0, duration=4.0, seed=0)
    return ds


def test_the_entry_hash_follows_every_kind_of_write_and_stays_when_nothing_is_written(member, tmp_path):
    ds, d = member, str(member)
    seen = []

    def wrote():
        h = entry_hash(ds, "S1")
        assert h == package_hash(ds, "S1"), "the catalogue's entry holds the package's own hash"
        assert h not in seen, "a write changes the hash"
        seen.append(h)

    def unchanged():
        assert entry_hash(ds, "S1") == seen[-1] == package_hash(ds, "S1")

    nn.add.line(dataset=d, subject="S1", name="freq", n=4, spacing=1.0, origin=1.0, unit="Hz", dimension="frequency")   # the first write through the catalogue after the package's own (the recording)
    wrote()
    # reads, a list, a refused write: nothing written, the hash stays
    nn.read.subject(dataset=d, subject="S1")
    nn.list.fields(dataset=d, subject="S1")
    with pytest.raises(nn.AlreadyExists):
        nn.add.line(dataset=d, subject="S1", name="freq", n=4, spacing=1.0, origin=1.0, unit="Hz", dimension="frequency")
    unchanged()
    nn.add.field(dataset=d, subject="S1", name="fx", manifold="cortex", values=np.arange(12.0))
    wrote()
    nn.add.selection(dataset=d, subject="S1", name="halves", manifold="cortex", type="partition", members=np.array([0] * 6 + [1] * 6, np.int32))
    wrote()
    nn.add.measurements(dataset=d, subject="S1", selection="halves", field="fx", measure={"total": np.array([15.0, 51.0])},
                        declare=[{"name": "total", "merge": "sum"}])
    wrote()
    nn.add.statistic(dataset=d, subject="S1", name="psd", measure="power_density", method="welch", axes=["channel:cortex"], values=np.ones(12))
    wrote()
    nn.add.operator(dataset=d, subject="S1", name="mix", from_manifold="cortex", to_manifold="cortex", values=np.eye(12))
    wrote()
    nn.run("select_time_grid", dataset=d, subject="S1", recording="rest", rule="tiles", duration="1 s", all=True)   # a run's outputs
    wrote()
    nn.run("select_time_grid", dataset=d, subject="S1", recording="rest", rule="tiles", duration="1 s", all=True)   # a cache hit writes nothing
    unchanged()
    # copied nodes: a surface of another package of the same subject
    src = tmp_path / "earlier"
    nn.create.dataset(dataset=str(src))
    nn.create.subject(dataset=str(src), name="S1")
    v, f = icosahedron(0.07)
    nn.add.surface(dataset=str(src), subject="S1", name="outer", vertices=v, faces=f, no_primary=True)
    nn.copy.nodes(dataset=d, subject="S1", from_=str(src / "S1.nxr.zarr"), manifold=["outer"])
    wrote()
    # the catalogue is current: a selection of the member is not refused as stale
    assert nn.list.subjects(dataset=d, subjects=["S1"])["subjects"][0]["content_hash"] == seen[-1]
    assert nn.verify.package(package=str(ds / "S1.nxr.zarr"))["ok"] is True


def test_query_reads_and_rolls_up_declared_measures_by_their_rule(member):
    """per-member measures written with `add measurements --declare` (one value a member) and `--rows-on` (an array a member) over the 1 s tiles of `rest`, read at the
    stored level and rolled up to 2 s by their declared rule, against numpy"""
    d = str(member)
    nn.add.field(dataset=d, subject="S1", name="fx", manifold="cortex", values=np.arange(12.0))
    r = nn.run("select_time_grid", dataset=d, subject="S1", recording="rest", rule="tiles", duration="1 s", all=True)
    grid = "rest time grid tiles 1 s"
    assert any(s["name"] == grid for s in r["stored"])
    members = nn.list.selections(dataset=d, subject="S1", name=grid)["rows"][0]["members"]
    rng = np.random.default_rng(7)
    energy = rng.uniform(1, 2, members)                       # one value a member, merge sum
    level = rng.uniform(-1, 1, members)                       # one value a member, merge mean (no weight declared: each member weighs one)
    top = rng.normal(size=(members, 12))                     # an array a member, its rows on the cortex's vertices, merge max
    flux = rng.normal(size=(members, 12))                     # the same, merge sum
    odd = rng.normal(size=members)                            # merge none: never rolled up
    nn.add.measurements(dataset=d, subject="S1", selection=grid, measure={"energy": energy, "level": level, "odd": odd},
                        declare=[{"name": "energy", "merge": "sum"}, {"name": "level", "merge": "mean"}, {"name": "odd", "merge": "none"}])
    nn.add.measurements(dataset=d, subject="S1", selection=grid, field="fx", rows_on="cortex", measure={"top": top, "flux": flux},
                        declare=[{"name": "top", "merge": "max"}, {"name": "flux", "merge": "sum"}])

    def q(measure, **kw):
        return nn.query.measurements(dataset=d, subject="S1", selection=grid, measure=measure, **kw)["result"]

    pairs = [list(range(i, min(i + 2, members))) for i in range(0, members, 2)]   # 1.28 s tiles to 2.56 s: two by two
    # one value a member: as stored, and rolled up in memory and in io's SQL
    assert [row["value"] for row in q("energy")["rows"]] == pytest.approx(energy.tolist())
    up = q("energy", tile="2 s")
    assert up["merge"] == "sum" and up["values"]["shape"] == [1, len(pairs)] and up["level"]["snapped"]["seconds"] == pytest.approx(2.56)
    np.testing.assert_allclose(up["values"]["data"], [energy[p].sum() for p in pairs])
    io = {row["code"]: row["value"] for row in q("energy", tile="2 s", where="io")["rows"]}
    np.testing.assert_allclose([io[k] for k in range(len(pairs))], [energy[p].sum() for p in pairs])
    np.testing.assert_allclose(q("level", tile="2 s")["values"]["data"], [level[p].mean() for p in pairs])
    np.testing.assert_allclose(q("energy", tile="whole")["values"]["data"], [energy.sum()])
    with pytest.raises(nn.Error, match="none"):
        q("odd", tile="2 s")
    # an array a member, its rows on the cortex: [rows x members] as stored, and rolled up
    stored = q("flux", rows_on="cortex", field="fx")
    assert stored["values"]["shape"] == [12, members]
    np.testing.assert_allclose(np.reshape(stored["values"]["data"], (12, members)), flux.T)
    up = q("flux", rows_on="cortex", tile="2 s")              # the field found as the one key its arrays are stored under
    np.testing.assert_allclose(np.reshape(up["values"]["data"], (12, len(pairs))), np.stack([flux[p].sum(axis=0) for p in pairs], axis=1))
    up = q("top", rows_on="cortex", field="fx", tile="2 s")
    np.testing.assert_allclose(np.reshape(up["values"]["data"], (12, len(pairs))), np.stack([top[p].max(axis=0) for p in pairs], axis=1))
    with pytest.raises(nn.Error, match="compute"):
        q("flux", rows_on="cortex", tile="2 s", where="io")   # io's SQL reads no array a member: said, never an empty answer

