// nxr-io — JOBS (docs/specs/2026-11-20-data-layer-queries.md, 3.8): what `run` and `import` leave in the dataset's `job` table, and `list jobs` reads back.
//
// One row for each run or import: what ran (`kind`: `run:<pipeline>`, `import:<what>`), its parameters, the nodes it read, when it started and ended, the program (`requested_by`) and the outcome
// (`status` done or failed, `error` the message of a failed one, `result` what a finished one stored). The row is written after the work, in its own short unit, so a failed run is recorded
// too (the work itself stored nothing). The rows are the schema's `job` table, as it stands; they are not content (the package hash does not cover them).
#pragma once
#include <functional>
#include <optional>
#include <string>

#include "nxr/io/task/commands.h"
#include "nxr/io/writer.h"

namespace nxr::io::task {

struct JobRecord {
  std::string kind;                        // `run:psd`, `import:recording`
  std::string status = "done";             // done | failed
  Json params = Json::object();            // as given
  Json inputs;                             // the nodes it read (null: none named)
  Json result;                             // what it stored (null: nothing, or not summarised)
  std::optional<std::string> error;        // a failed job's message
  std::string requested_by;                // the program (`nxr-neuro 0.1 / C++17`)
  std::string started_utc, finished_utc;
};

/** write the row into the open dataset (or package) `ds`, for `subject` (by name; none: the dataset's); the unit of the work has ended */
void record_job(DatasetWriter& ds, const std::optional<std::string>& subject, const JobRecord& job);
void record_job(SubjectWriter& sub, const JobRecord& job);

/** where an import's job row goes: a subject package (a file importer's), or a dataset (a catalogue: its `subject`'s package) */
struct JobTarget {
  std::string dataset, package;
  std::optional<std::string> subject;
};

/** run `body` and record it: a done job with what it returned, or a failed one with its message (the error goes on). When the row cannot be written, the work's own outcome stands: a finished
 *  command's result then carries `job_error`, a failed one's error is the work's. */
CommandResult recorded(const std::string& kind, const JobTarget& target, const Json& params, const std::string& program, const std::function<CommandResult()>& body);

}  // namespace nxr::io::task
