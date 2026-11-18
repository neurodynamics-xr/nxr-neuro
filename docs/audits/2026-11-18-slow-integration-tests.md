# Integration tests over 5 s, to shrink later (2026-11-18)

An integration test is small: one synthetic subject, one chunk, one band, seconds (CONTRIBUTING.md, Test kinds). These are over 5 s in the last recorded run and
should be cut down to that size, the maths they also check moving to unit or golden tests. Each carries `slow` in `tests/kinds.tsv` until it is shrunk.

| test | seconds (last recorded run) |
|---|---|
| test_psd_pipeline | 6.1 |
| cli_import | 5.5 |

Also labelled `slow` and to be reviewed the same way (their own comments say minutes): test_spatial_events_planted, test_spatial_threshold.
Near the limit (4 to 5 s): cli_help (4.3).
