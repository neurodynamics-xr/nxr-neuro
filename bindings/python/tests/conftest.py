"""Fixtures. Everything is synthetic: generated signals here, and the synthetic files and goldens that `io/formats/test` holds (made by
MNE-Python, nibabel and pandas from generated data, each next to its generator)."""

import json
from pathlib import Path

import numpy as np
import pytest

FORMATS_TEST = Path(__file__).resolve().parents[3] / "io" / "formats" / "test"


@pytest.fixture(scope="session")
def formats_dir():
    if not (FORMATS_TEST / "golden").is_dir():
        pytest.skip("the repository's io/formats/test folder is not next to the tests")
    return FORMATS_TEST


@pytest.fixture(scope="session")
def golden(formats_dir):
    def load(name):
        return json.loads((formats_dir / "golden" / f"{name}_golden.json").read_text())

    return load


@pytest.fixture(scope="session")
def f32(formats_dir):
    def load(name):
        return np.fromfile(formats_dir / "golden" / name, dtype="<f4")

    return load


@pytest.fixture
def rng():
    return np.random.default_rng(20261117)


@pytest.fixture
def signal(rng):
    """Four channels of coloured noise plus a 12 Hz and a 40 Hz sinusoid, 1000 Hz, 8 s."""
    sfreq, n = 1000.0, 8000
    t = np.arange(n) / sfreq
    x = rng.normal(size=(4, n)).cumsum(axis=1) * 0.05 + rng.normal(size=(4, n))
    x += 2.0 * np.sin(2 * np.pi * 12 * t) + 1.0 * np.sin(2 * np.pi * 40 * t + 0.3)
    return x, sfreq
