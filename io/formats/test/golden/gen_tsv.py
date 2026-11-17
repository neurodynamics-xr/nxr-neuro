#!/usr/bin/env python3
"""Synthetic TSV files and their goldens, made with pandas (a reference for parity only).

Writes test/data/tsv_*.tsv and test/golden/tsv_golden.json (the cells as pandas reads them with dtype=str, keep_default_na=False) and
test/golden/tsv_written_golden.tsv (the bytes pandas writes for the table the C++ writer test builds). Run:
  uv run --project <nxr> python test/golden/gen_tsv.py
"""
import json
import pathlib
import pandas as pd

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
data_dir.mkdir(exist_ok=True)

df = pd.DataFrame({
    "participant_id": ["sub-01", "sub-02", "sub-03"],
    "age": ["34", "n/a", "27.5"],
    "group": ["control", "patient", "control"],
    "note": ["", "has, comma", "two words"],
})
tsv = df.to_csv(sep="\t", index=False, na_rep="n/a", lineterminator="\n")
(data_dir / "tsv_lf.tsv").write_text(tsv, encoding="utf-8")
(data_dir / "tsv_crlf.tsv").write_bytes(tsv.replace("\n", "\r\n").encode())
(data_dir / "tsv_bom.tsv").write_bytes(b"\xef\xbb\xbf" + tsv.encode())
(data_dir / "tsv_ragged.tsv").write_text("a\tb\tc\n1\t2\t3\n4\t5\n", encoding="utf-8")

golden = {}
for name in ["lf", "crlf", "bom"]:
    t = pd.read_csv(data_dir / f"tsv_{name}.tsv", sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    golden[name] = {"columns": list(t.columns), "rows": t.values.tolist()}
(here / "tsv_golden.json").write_text(json.dumps(golden, indent=1))

# what the C++ writer test writes: empty cell -> n/a
w = pd.DataFrame({"onset": ["0", "1.5"], "duration": ["0.5", "n/a"], "label": ["a", "n/a"]})
(here / "tsv_written_golden.tsv").write_text(w.to_csv(sep="\t", index=False, na_rep="n/a", lineterminator="\n"), encoding="utf-8")
print("ok")
