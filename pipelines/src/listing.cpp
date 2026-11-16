#include "nxr/pipelines/listing.h"

#include "nxr/io/errors.h"

namespace nxr::pipelines {

using namespace nxr::io;

Json list_subject_nodes(DatasetWriter& dataset, SubjectWriter& subject, const std::string& what) {
  if (what != "recordings" && what != "kernels" && what != "results")
    throw Error(code::invalid_argument, "list: '" + what + "' is not something a subject holds by kind (recordings, kernels or results)");
  const std::string sid = subject.id();
  std::vector<Row> rows;
  const char* table = "field";
  if (what == "recordings") {
    rows = dataset.db().all("SELECT name, kind FROM field WHERE subject_id = ? AND kind = 'recording' AND status = 'complete' ORDER BY name", {V(sid)});
  } else if (what == "kernels") {
    table = "operator";
    rows = dataset.db().all("SELECT name, kind FROM operator WHERE subject_id = ? AND kind = 'inverse kernel' AND status = 'complete' ORDER BY name", {V(sid)});
  } else {
    Json out = Json::array();
    for (const Row& r : dataset.db().all("SELECT name, kind FROM field WHERE subject_id = ? AND kind != 'recording' AND status = 'complete' ORDER BY name", {V(sid)}))
      out.push_back(Json{{"name", text(r, "name")}, {"kind", text(r, "kind")}, {"table", "field"}});
    for (const Row& r : dataset.db().all("SELECT name FROM statistic WHERE subject_id = ? AND status = 'complete' ORDER BY name", {V(sid)}))
      out.push_back(Json{{"name", text(r, "name")}, {"kind", "statistic"}, {"table", "statistic"}});
    return out;
  }
  Json out = Json::array();
  for (const Row& r : rows) out.push_back(Json{{"name", text(r, "name")}, {"kind", text(r, "kind")}, {"table", table}});
  return out;
}

}  // namespace nxr::pipelines
