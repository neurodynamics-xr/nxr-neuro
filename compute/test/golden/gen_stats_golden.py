"""Writes stats_golden.txt: the references of nxr::stats on synthetic samples (numpy and scipy): mean, sd (ddof 1, MATLAB's std), median, the robust SD
1.4826 * median|x - median| (Rheome detect.spindle), percentiles 10, 50, 75, 98 (numpy linear), peak max|x|, kurtosis and skewness (scipy.stats,
fisher=False, bias=True: MATLAB's kurtosis and skewness), min, max. Synthetic data only.
Run:  uv run --project <nxr checkout> python compute/test/golden/gen_stats_golden.py compute/test/golden/stats_golden.txt"""
import sys

import numpy as np
from scipy import stats

rng = np.random.default_rng(20261113)
samples = {"lognormal": rng.lognormal(0, 1, 501), "normal_even": rng.normal(2, 3, 200), "small": np.array([3.0, -1.0, 7.0, 2.0])}
with open(sys.argv[1], "w") as f:
    for name, x in samples.items():
        med = float(np.median(x))
        vals = {"mean": np.mean(x), "sd": np.std(x, ddof=1), "median": med, "mad": 1.4826 * np.median(np.abs(x - med)),
                "p10": np.percentile(x, 10), "p50": np.percentile(x, 50), "p75": np.percentile(x, 75), "p98": np.percentile(x, 98),
                "peak": np.max(np.abs(x)), "kurtosis": stats.kurtosis(x, fisher=False, bias=True), "skewness": stats.skew(x, bias=True),
                "min": np.min(x), "max": np.max(x)}
        f.write(f"sample {name} {len(x)}\n" + " ".join(repr(float(v)) for v in x) + "\n")
        f.write(" ".join(f"{k}={float(v)!r}" for k, v in vals.items()) + "\n")
