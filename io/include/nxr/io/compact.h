// nxr-io — COMPACT a dataset's arrays from format 50 to format 51 (docs/specs/2026-11-14-dataset-wide-runs.md, section 1).
//
// Format 51 writes the chunk key separator "." (c.0.1 in the array's folder: no c/ folders) and shards an array over 4 MiB by the default policy
// (~64 MiB shards of the 2 MiB inner chunks: a few files, not thousands). `compact` rewrites each array of an older dataset into that layout, ONE ARRAY AT
// A TIME, through the store interface:
//
//   1. the new layout is written BESIDE the old one (`<array>/.compact/`), slab by slab (never the whole array in memory);
//   2. every slab of the new layout is read back and compared with the old one BYTE FOR BYTE (the decoded values; a mismatch refuses the array);
//   3. a marker `.compact/.verified` says the new layout is whole and checked;
//   4. the swap: the new chunk files move up beside the old `c/` folder (different keys: nothing is overwritten), then the array's `zarr.json` is written (the
//      COMMIT: from then on the array is format 51), then the old `c/` folder and `.compact/` are removed.
//
// A crash before the commit leaves the old layout intact (the next run starts the array again, or finishes the swap when the marker is there); a crash
// after it leaves a format-51 array with debris the next run removes. An array already in format 51 is skipped (idempotent). A row of the array's node
// (field, operator, statistic) has its `codecs_json` brought to the file's, and its zarr.json is written with the view's text. The dataset is opened
// by the caller with the WRITER LOCK (a dataset another writer holds is refused by `open_writer`); `meta.store_format` is set when no array of the dataset
// is left in format 50, and 52 when, besides, no subject has a small key left as a file.
//
// FORMAT 52 (/15: about 100 files a subject): after a subject's arrays, `compact` moves every small key of its folder tree (every `zarr.json`, every
// chunk or shard under 1 MiB) into the subject's `small_files.sqlite`, a batch at a time: the batch is written, read back and compared with its files BYTE FOR BYTE,
// and only then are its files removed; the empty folders go last. A crash leaves a key in both places (the same bytes; the table is read first), and the next
// run finishes. `small_files = false` stops at format 51.
#pragma once
#include <functional>
#include <optional>
#include <string>

#include "nxr/io/json_text.h"
#include "nxr/io/writer.h"

namespace nxr::io {

struct CompactOptions {
  std::optional<std::string> subject;   // only this subject (default: every subject of the dataset)
  bool dry_run = false;                 // report what would change, change nothing
  bool small_files = true;              // format 52: also move the small keys of each subject into its small_files.sqlite (verified byte for byte, the files removed)
  bool shard = true;                    // false: only the separator changes (an array over 4 MiB stays unsharded, as `write_region` needs)
  // Called at the stages of an array's compaction ("written", "verified", "moved", "committed") with the array's key: a test throws from it to stop there.
  std::function<void(const std::string& stage, const std::string& array)> on_stage;
};

// The report: per subject and in total, the arrays seen, compacted, already in format 51 and skipped (links), and the files and folders under the arrays before
// and after (a dry run: the files after are at most the grid's cells).
Json compact_dataset(DatasetWriter& ds, const CompactOptions& opt = {});

}  // namespace nxr::io
