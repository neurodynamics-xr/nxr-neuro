// Jobs (nxr/io/task/jobs.h): the rows `run` and `import` leave in the `job` table.
#include "nxr/io/task/jobs.h"

#include "commands_open.h"
#include "nxr/io/errors.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/task/dataset.h"

namespace nxr::io::task {

using namespace commands_detail;

namespace {
// what a finished job keeps of its result: the object, unless it is large (a scan's whole view): then its keys
Json kept(const Json& result) {
  if (result.is_null()) return result;
  if (result.dump().size() <= 16384) return result;
  Json keys = Json::array();
  if (result.is_object()) for (auto it = result.begin(); it != result.end(); ++it) keys.push_back(it.key());
  return Json{{"truncated", true}, {"keys", keys}};
}

void write_row(Database& db, const std::string& dataset_id, const std::optional<std::string>& subject_id, const JobRecord& j) {
  Row r{{"id", V(uuid7())}, {"kind", V(j.kind)}, {"host", V("main")}, {"status", V(j.status)}, {"dataset_id", V(dataset_id)}, {"params_json", V(j.params.dump())},
        {"requested_by", V(j.requested_by)}, {"started_utc", V(j.started_utc)}, {"finished_utc", V(j.finished_utc)}};
  if (subject_id) r["subject_id"] = V(*subject_id);
  if (!j.inputs.is_null()) r["inputs_json"] = V(j.inputs.dump());
  if (!j.result.is_null()) r["result_json"] = V(kept(j.result).dump());
  if (j.error) r["error"] = V(*j.error);
  if (j.status == "done") r["progress"] = V(1.0);
  if (!j.started_utc.empty()) r["created_utc"] = V(j.started_utc);
  db.insert("job", r);
}
}  // namespace

void record_job(DatasetWriter& ds, const std::optional<std::string>& subject, const JobRecord& job) {
  std::optional<std::string> sid;
  if (subject) {
    if (const auto s = ds.subject(*subject)) sid = text(*s, "id");
  } else if (ds.is_package()) {
    sid = Dataset::of(ds).package_subject_id();
  }
  write_row(ds.db(), ds.id(), sid, job);
}

void record_job(SubjectWriter& sub, const JobRecord& job) { write_row(sub.db(), sub.dataset().id(), sub.id(), job); }

CommandResult recorded(const std::string& kind, const JobTarget& target, const Json& params, const std::string& program, const std::function<CommandResult()>& body) {
  JobRecord j;
  j.kind = kind;
  j.params = params;
  j.requested_by = program;
  j.started_utc = now_utc();
  auto write = [&]() {
    j.finished_utc = now_utc();
    if (!target.package.empty()) {
      DatasetWriter w = open_package_writer(target.package);
      record_job(w, std::nullopt, j);
      w.close();
    } else {
      DatasetWriter w = is_catalogue_dataset(target.dataset) ? open_dataset(target.dataset, target.subject, OpenOptions{}, {}) : open_writer(target.dataset);
      record_job(w, target.subject, j);
      w.close();
    }
  };
  CommandResult r;
  try {
    r = body();
  } catch (const Refused&) {
    throw;   // a request refused as given (an option's value) did not run: no row
  } catch (const std::exception& e) {
    j.status = "failed";
    j.error = e.what();
    try { write(); } catch (...) {}   // the work's own error is the one to report
    throw;
  }
  j.result = r.result;
  try {
    write();
  } catch (const std::exception& e) {
    r.result["job_error"] = e.what();
  }
  return r;
}

}  // namespace nxr::io::task
