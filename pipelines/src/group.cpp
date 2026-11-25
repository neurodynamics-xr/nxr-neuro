// nxr-pipelines — what the GROUP TREE of a subject is carried from (group_template.h): the default subject's primary cortex, its registered sphere and its group grid, read for the runner's
// Bound::group_template (the node `select.dyadic_surface`, rule transfer). The group reduce is the definition `roll_up` (scope group).
#include "nxr/pipelines/group_template.h"

#include <algorithm>
#include <cstring>
#include <memory>

#include "nxr/io/catalogue.h"
#include "nxr/io/layout.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/selection_reads.h"
#include "nxr/io/subject_reads.h"
#include "nxr/pipelines/input.h"

namespace nxr::pipelines {

using namespace nxr::io;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(code::no_such_row, m); }

std::vector<std::int32_t> read_i32(const Store& store, const std::string& key) {
  ArrayReader r = ArrayReader::open(store, key);
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<std::int32_t> out(raw.size() / 4);
  if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * 4);
  return out;
}
Json parse_obj(const Row& r, const char* col = "params_json") {
  const Json j = Json::parse(opt_text(r, col).value_or("{}"), nullptr, false);
  return j.is_object() ? j : Json::object();
}
std::optional<SurfaceInput> sphere_of(SubjectWriter& sub, const std::string& surface) {
  try { return read_surface(sub, surface + "_sphere", true); } catch (const Error& e) { if (e.code() == code::no_such_row) return std::nullopt; throw; }
}
}  // namespace

// the default subject's group tree as the subject `sub` is carried onto it (the node `select.dyadic_surface` rule transfer reads it through the runner: Bound::group_template)
GroupTemplate read_group_template(DatasetWriter& ds, SubjectWriter& sub, const std::string& sname) {
  // format 56: a subject package carries nothing of the template; the dataset's template is opened READ ONLY through the catalogue (the package's folder lies in the dataset's)
  std::unique_ptr<DatasetWriter> template_package;
  DatasetWriter* tds = &ds;
  if (ds.is_package()) {
    OpenOptions read_only;
    read_only.lock = false;
    template_package = std::make_unique<DatasetWriter>(nxr::io::open_template_of(ds, read_only));
    tds = template_package.get();
  }
  const auto tpl = tds->template_();
  if (!tpl) missing("the dataset has no template subject: the group tree is its ladder");
  const SurfaceInput S = read_surface(sub, sname, true);
  const int nV = static_cast<int>(S.n_vertices());
  SubjectWriter tsub = SubjectWriter::open(*tds, text(*tpl, "name"));
  const SurfaceInput T = read_surface(tsub, std::nullopt, true);
  const auto tsphere = sphere_of(tsub, T.name), ssphere = sphere_of(sub, sname);
  if (!tsphere) missing(text(*tpl, "name") + ": no registered sphere (" + T.name + "_sphere)");
  if (!ssphere) missing(sub.name() + ": no registered sphere (" + sname + "_sphere): the group tree is assigned through it");
  if (ssphere->n_vertices() != nV) invalid(sname + "_sphere has " + std::to_string(ssphere->n_vertices()) + " vertices, the surface " + std::to_string(nV));
  if (tsphere->n_vertices() != T.n_vertices()) invalid(T.name + "_sphere has another vertex count than " + T.name);
  const auto tl = nxr::io::selection::named_row(tsub.db(), text(*tpl, "id"), T.name + " tiles", "ladder");
  if (!tl) missing(text(*tpl, "name") + ": no ladder `" + T.name + " tiles` (the group tree)");
  if (parse_obj(*tl).value("tree", std::string()) != "group") invalid(text(*tpl, "name") + ": `" + T.name + " tiles` is not a group tree (tree = " + parse_obj(*tl).value("tree", std::string("subject")) + ")");
  const std::vector<Row> tlevels = nxr::io::selection::member_rows(tsub.db(), text(*tl, "id"));
  if (tlevels.empty()) invalid(text(*tpl, "name") + ": its group ladder has no levels");
  GroupTemplate g;
  g.subject = text(*tpl, "name");
  g.surface = T.name;
  g.n_vertices = T.n_vertices();
  g.faces = T.faces;
  g.sphere = tsphere->vertices;
  g.subject_sphere = ssphere->vertices;
  g.finest = static_cast<int>(tlevels.size()) - 1;
  g.codes = read_i32(tsub.store(), text(tlevels.back(), "array_path"));
  if (static_cast<std::int64_t>(g.codes.size()) != T.n_vertices()) invalid(g.subject + ": its finest level does not fit its surface");
  g.slots = static_cast<int>(std::max<std::int64_t>(2, integer(tlevels.back(), "n_members") >> g.finest));   // the piece slots the codes reserve
  g.ladder_dictionary_id = text(*tl, "dictionary_id");
  for (const Row& r : tlevels) g.level_dictionary_ids.push_back(text(r, "dictionary_id"));
  return g;
}

}  // namespace nxr::pipelines
