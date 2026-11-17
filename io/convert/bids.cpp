// The importer of a BIDS dataset: see nxr/convert/bids.h.
#include "nxr/convert/bids.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>

#include "nxr/convert/recording.h"
#include "nxr/formats/bids.h"
#include "nxr/io/participants.h"
#include "nxr/io/subject_package.h"

namespace nxr::convert {

namespace fs = std::filesystem;
namespace code = nxr::io::code;
namespace bids = nxr::formats::bids;
namespace tsv = nxr::formats::tsv;
using nxr::io::Json;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}
std::string upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return s;
}
bool starts_with(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }

// a cell that says something: not empty, not "n/a"
bool given(const tsv::Table& t, std::size_t row, const std::string& column) {
  if (t.column(column) < 0) return false;
  const std::string& c = t.cell(row, column);
  return !c.empty() && !tsv::Table::is_na(c);
}

// the channel kind a BIDS channel type states ("" when the reader's own is kept: MEG channels, whose gradiometer or magnetometer the file's unit says)
std::string kind_of(const std::string& bids_type) {
  const std::string t = upper(bids_type);
  if (starts_with(t, "MEG")) return "";
  if (t == "TRIG") return "stim";
  return lower(t);
}

double metres_per(const std::string& unit) {
  if (unit == "m") return 1.0;
  if (unit == "cm") return 1e-2;
  if (unit == "mm") return 1e-3;
  return 0.0;
}

std::string base_name(const std::string& path) { return path.substr(path.rfind('/') == std::string::npos ? 0 : path.rfind('/') + 1); }

// the `space` entity of a file ("" when it has none)
std::string space_of(const std::string& path) {
  try { return bids::parse_name(base_name(path)).get("space").value_or(""); } catch (const std::exception&) { return ""; }
}

std::string coordinate_unit_key(const std::string& datatype) { return datatype == "ieeg" ? "iEEGCoordinateUnits" : upper(datatype) + "CoordinateUnits"; }

const std::set<std::string>& identifying_sidecar_keys() {
  static const std::set<std::string> k = {"InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName", "DeviceSerialNumber", "SubjectArtefactDescription"};
  return k;
}

}  // namespace

nxr::formats::RecordingHeader bids_header(const nxr::formats::RecordingHeader& base, const tsv::Table* channels, const tsv::Table* events, const tsv::Table* electrodes,
                                          const std::string& position_unit) {
  nxr::formats::RecordingHeader h = base;
  if (channels) {
    if (channels->column("name") < 0) refuse("a channels table has no `name` column");
    std::map<std::string, std::size_t> row_of;
    for (std::size_t r = 0; r < channels->rows.size(); ++r) row_of[channels->cell(r, "name")] = r;
    for (auto& c : h.channels) {
      const auto it = row_of.find(c.name);
      if (it == row_of.end()) continue;
      const std::size_t r = it->second;
      if (given(*channels, r, "type")) {
        const std::string k = kind_of(channels->cell(r, "type"));
        if (!k.empty()) c.kind = k;
      }
      if (given(*channels, r, "status") && lower(channels->cell(r, "status")) == "bad") {
        c.bad = true;
        if (std::find(h.bad_channels.begin(), h.bad_channels.end(), c.name) == h.bad_channels.end()) h.bad_channels.push_back(c.name);
      }
    }
  }
  if (electrodes) {
    const double to_m = metres_per(position_unit);
    if (to_m == 0.0) refuse("electrode positions need a coordinate unit of m, cm or mm (the coordinate system says '" + position_unit + "')");
    for (const char* col : {"name", "x", "y", "z"}) if (electrodes->column(col) < 0) refuse(std::string("an electrodes table has no `") + col + "` column");
    std::map<std::string, std::size_t> row_of;
    for (std::size_t r = 0; r < electrodes->rows.size(); ++r) row_of[electrodes->cell(r, "name")] = r;
    // the positions are stored in the unit the header already states (the reader's own positions stay as they are), else in the coordinate system's
    const std::string unit = h.position_unit.empty() ? position_unit : h.position_unit;
    const double scale = to_m / metres_per(unit);
    for (auto& c : h.channels) {
      if (c.location) continue;
      const auto it = row_of.find(c.name);
      if (it == row_of.end()) continue;
      const std::size_t r = it->second;
      if (!given(*electrodes, r, "x") || !given(*electrodes, r, "y") || !given(*electrodes, r, "z")) continue;
      c.location = std::array<double, 3>{electrodes->number(r, "x") * scale, electrodes->number(r, "y") * scale, electrodes->number(r, "z") * scale};
    }
    h.position_unit = unit;
  }
  if (events) {
    for (const char* col : {"onset", "duration"}) if (events->column(col) < 0) refuse(std::string("an events table has no `") + col + "` column");
    std::vector<nxr::formats::Annotation> kept;
    for (const auto& a : h.annotations) if (starts_with(upper(a.label), "BAD")) kept.push_back(a);   // bad segments of the data file stay
    for (std::size_t r = 0; r < events->rows.size(); ++r) {
      if (!given(*events, r, "onset")) refuse("an events table has a row with no onset");
      nxr::formats::Annotation a;
      a.onset = events->number(r, "onset");
      a.duration = given(*events, r, "duration") ? events->number(r, "duration") : 0.0;
      if (given(*events, r, "trial_type")) a.label = events->cell(r, "trial_type");
      else if (given(*events, r, "value")) a.label = events->cell(r, "value");
      else a.label = "event";
      kept.push_back(a);
    }
    h.annotations = std::move(kept);
  }
  return h;
}

Json import_bids(const fs::path& bids_root, const fs::path& dataset_folder, const BidsImport& opt) {
  const bids::DirectorySource src(bids_root.string());
  bids::ReadOptions ro;
  ro.subjects = opt.subjects;
  const bids::Dataset ds = bids::read_dataset(src, ro);
  if (ds.subjects.empty()) refuse("the BIDS folder holds no subject" + std::string(opt.subjects.empty() ? "" : " among those named"));
  if (ds.description.dataset_type.value_or("raw") != "raw")
    refuse("this is a BIDS derivatives dataset (DatasetType '" + *ds.description.dataset_type + "'); import raw data");
  const std::string licence = ds.description.license.value_or("unknown");
  std::string attribution;
  for (const auto& a : ds.description.authors) attribution += (attribution.empty() ? "" : ", ") + a;
  if (attribution.empty()) attribution = ds.description.how_to_acknowledge.value_or("unknown");

  // the dataset: a catalogue in a new folder
  nxr::io::CreatePackageOptions pkg_opt;
  {
    nxr::io::CreateCatalogueOptions co;
    co.name = dataset_folder.filename().empty() ? dataset_folder.parent_path().filename().string() : dataset_folder.filename().string();
    co.source_tool = "bids";
    co.source_protocol = ds.description.bids_version;
    co.created_by = opt.registered_by;
    co.built_by = opt.registered_by;
    nxr::io::Catalogue cat = nxr::io::create_catalogue(dataset_folder, co);
    pkg_opt = cat.package_options();
  }
  pkg_opt.subject.created_by = opt.registered_by;
  pkg_opt.built_by = opt.registered_by;

  Json recordings = Json::array(), skipped = Json::array();
  std::vector<fs::path> packages;
  std::vector<std::string> names;
  for (const auto& subject : ds.subjects) {
    const std::string sname = "sub-" + subject.label;
    const fs::path pkg = fs::absolute(dataset_folder) / (sname + ".nxr.zarr");
    nxr::io::DatasetWriter w = nxr::io::create_package(pkg, sname, pkg_opt);
    nxr::io::SubjectWriter sub = nxr::io::SubjectWriter::open(w, sname);
    for (const auto& session : subject.sessions) {
      for (const auto& rec : session.recordings) {
        // the recording's name: its entities but the subject's; its acquisition: the session
        std::string rname;
        for (const auto& [k, v] : rec.entities.items) if (k != "sub") rname += (rname.empty() ? "" : "_") + k + "-" + v;
        if (rname.empty()) rname = rec.datatype;
        const std::string file = rec.data_files.empty() ? "" : rec.data_files.front();
        auto skip = [&](const std::string& why) { skipped.push_back(Json{{"subject", sname}, {"recording", rname}, {"file", file}, {"reason", why}}); };
        if (file.empty()) { skip("no data file"); continue; }
        if (rec.data_files.size() > 1 && lower(file).find(".fif") == std::string::npos) { skip("a recording split into several files (only FIF's are followed)"); continue; }
        std::string format;
        try { format = format_of(file); } catch (const nxr::io::Error& e) { skip(e.what()); continue; }
        const std::string abs_file = (fs::absolute(bids_root) / file).string();
        std::unique_ptr<nxr::formats::RecordingReader> reader;
        try { reader = open_recording(format, abs_file); } catch (const std::exception& e) { skip(std::string("the file is not read: ") + e.what()); continue; }

        // the tables: channels, events, and the electrodes with their coordinate system (matched by `space`)
        Json notes = Json::array();
        const tsv::Table* electrodes = nullptr;
        const Json* coordsystem = nullptr;
        std::string position_unit, electrodes_path, space;
        if (!rec.electrodes.empty()) {
          electrodes = &rec.electrodes.front().table;
          electrodes_path = rec.electrodes.front().path;
          space = space_of(electrodes_path);
          for (const auto& cs : rec.coordsystems) if (space_of(cs.path) == space) coordsystem = &cs.content;
          if (coordsystem && coordsystem->contains(coordinate_unit_key(rec.datatype)) && (*coordsystem)[coordinate_unit_key(rec.datatype)].is_string())
            position_unit = (*coordsystem)[coordinate_unit_key(rec.datatype)].get<std::string>();
          if (metres_per(position_unit) == 0.0) {
            notes.push_back("the electrode positions are not imported: the coordinate system states no unit (m, cm or mm)");
            electrodes = nullptr;
          }
        }
        auto header = std::make_shared<nxr::formats::RecordingHeader>(
            bids_header(reader->header(), rec.channels ? &rec.channels->table : nullptr, rec.events ? &rec.events->table : nullptr, electrodes, position_unit));

        Json bj = Json::object();
        bj["subject"] = sname;
        Json ents = Json::object();
        for (const auto& [k, v] : rec.entities.items) ents[k] = v;
        bj["entities"] = ents;
        Json files = Json::object();
        files["data"] = file;
        if (rec.channels) files["channels"] = rec.channels->path;
        if (rec.events) files["events"] = rec.events->path;
        if (electrodes) files["electrodes"] = electrodes_path;
        if (coordsystem) {
          for (const auto& cs : rec.coordsystems) if (&cs.content == coordsystem) files["coordsystem"] = cs.path;
        }
        files["sidecars"] = rec.sidecar_files;
        bj["files"] = files;
        Json sidecar = Json::object();
        if (rec.sidecar.is_object())
          for (auto it = rec.sidecar.begin(); it != rec.sidecar.end(); ++it) if (!(opt.anonymize && identifying_sidecar_keys().count(it.key()))) sidecar[it.key()] = it.value();
        bj["sidecar"] = sidecar;
        if (coordsystem) bj["coordinate_system"] = *coordsystem;   // verbatim: this importer does not interpret a frame
        if (!space.empty()) bj["electrodes_space"] = space;
        if (rec.channels && rec.channels->table.column("status_description") >= 0) {
          Json why = Json::object();
          for (std::size_t r = 0; r < rec.channels->table.rows.size(); ++r)
            if (given(rec.channels->table, r, "status_description")) why[rec.channels->table.cell(r, "name")] = rec.channels->table.cell(r, "status_description");
          if (!why.empty()) bj["channel_status_description"] = why;
        }

        RecordingImport ri;
        ri.name = rname;
        ri.acquisition = session.label.empty() ? "main" : "ses-" + session.label;
        ri.format = format;
        ri.source = file;
        ri.source_file = fs::is_directory(abs_file) ? "" : abs_file;
        ri.licence = licence;
        ri.attribution = attribution;
        ri.anonymize = opt.anonymize;
        ri.in_place = opt.in_place;
        ri.header = header;
        ri.extra = Json{{"bids", bj}};
        ri.progress = opt.progress;
        Json out = import_recording(w, sub, *reader, ri);
        out["subject"] = sname;
        out["acquisition"] = ri.acquisition;
        out["notes"] = notes;
        recordings.push_back(out);
      }
    }
    w.close();
    packages.push_back(pkg);
    names.push_back(sname);
  }

  nxr::io::RegisterOptions reg;
  reg.registered_by = opt.registered_by;
  Json registered = Json::array();
  for (const auto& r : nxr::io::register_packages(dataset_folder, packages, reg)) {
    registered.push_back(Json{{"package", r.package.filename().string()}, {"name", r.name}, {"outcome", r.outcome}, {"reason", r.reason}});
    if (r.outcome == "refused") throw nxr::io::Error(code::invalid_argument, "the catalogue refused " + r.package.filename().string() + ": " + r.reason);
  }

  // participants.tsv: the rows of the subjects imported (a participant with no folder is left out), with the sidecar as it is
  Json part = Json();
  if (opt.participants && ds.participants) {
    tsv::Table t = ds.participants->table;
    if (t.column("participant_id") < 0) refuse("participants.tsv has no `participant_id` column");
    const int id_col = t.column("participant_id");
    std::vector<std::vector<std::string>> rows;
    for (auto& row : t.rows) if (std::find(names.begin(), names.end(), row[static_cast<std::size_t>(id_col)]) != names.end()) rows.push_back(row);
    const std::int64_t dropped = static_cast<std::int64_t>(t.rows.size() - rows.size());
    t.rows = std::move(rows);
    const fs::path tmp_tsv = fs::absolute(dataset_folder) / ".participants_import.tsv", tmp_json = fs::absolute(dataset_folder) / ".participants_import.json";
    tsv::write_file(tmp_tsv.string(), t);
    std::optional<fs::path> sidecar;
    if (ds.participants_sidecar) {
      std::ofstream(tmp_json) << ds.participants_sidecar->content.dump(2) << "\n";
      sidecar = tmp_json;
    }
    try {
      const auto rep = nxr::io::participants_import(dataset_folder, tmp_tsv, sidecar);
      part = Json{{"variables", rep.variables}, {"n_subjects", rep.n_subjects}, {"n_values", rep.n_values}, {"left_out", dropped}};
    } catch (...) {
      std::error_code ec;
      fs::remove(tmp_tsv, ec);
      fs::remove(tmp_json, ec);
      throw;
    }
    std::error_code ec;
    fs::remove(tmp_tsv, ec);
    fs::remove(tmp_json, ec);
  }
  return Json{{"dataset", dataset_folder.string()}, {"name", ds.description.name}, {"bids_version", ds.description.bids_version}, {"licence", licence},
              {"subjects", names}, {"recordings", recordings}, {"skipped", skipped}, {"participants", part}, {"packages", registered},
              {"issues", static_cast<std::int64_t>(ds.issues.size())}};
}

}  // namespace nxr::convert
