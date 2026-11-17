#include "nxr/pipelines/listing.h"

#include "nxr/pipelines/task.h"

#include "nxr/io/errors.h"

namespace nxr::pipelines {

using namespace nxr::io;

Json list_subject_nodes(DatasetWriter& dataset, SubjectWriter& subject, const std::string& what) {
  if (what != "recordings" && what != "kernels" && what != "results")
    throw Error(code::invalid_argument, "list: '" + what + "' is not something a subject holds by kind (recordings, kernels or results)");
  (void)dataset;
  const task::Subject subj = task::Subject::of(subject);
  Json out = Json::array();
  if (what == "recordings" || what == "kernels") {
    const bool rec = what == "recordings";
    for (const task::Node& n : subj.find_all({.cls = rec ? task::Cls::field : task::Cls::operator_, .kind = rec ? "recording" : "inverse kernel"}))
      out.push_back(Json{{"name", n.name}, {"kind", n.kind.value_or("")}, {"table", rec ? "field" : "operator"}});
    return out;
  }
  for (const task::Node& n : subj.find_all({.cls = task::Cls::field}))
    if (n.kind.value_or("") != "recording") out.push_back(Json{{"name", n.name}, {"kind", n.kind.value_or("")}, {"table", "field"}});
  for (const task::Node& n : subj.find_all({.cls = task::Cls::statistic}))
    out.push_back(Json{{"name", n.name}, {"kind", "statistic"}, {"table", "statistic"}});
  return out;
}

}  // namespace nxr::pipelines
