// The commands' shared openers (io/src only; nxr/io/task/commands.h): how every command opens the dataset it names, and its notes.
#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/catalogue.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/writer.h"

namespace nxr::io::task::commands_detail {

namespace fs = std::filesystem;


inline void note(const Progress& progress, const std::string& text) {
  if (progress) progress(Json{{"stage", "note"}, {"note", text}});
}

inline TemplateSearch search_of(const std::vector<std::string>& templates) {
  TemplateSearch s;
  for (const auto& p : templates) s.command_paths.push_back(p);
  return s;
}

// THE DATASET A COMMAND WORKS ON: a catalogue's subject package (found through the catalogue), or a one-database dataset (store formats 50-52), as every command opens it.
inline DatasetWriter open_dataset(const std::string& folder, const std::optional<std::string>& subject, const OpenOptions& o, const std::vector<std::string>& templates,
                           const Progress& progress) {
  if (is_catalogue_dataset(folder)) {
    if (!subject) throw Refused(folder + " is a catalogue of subject packages: name the subject to work on with --subject", true);
    return open_subject_package(folder, *subject, o, search_of(templates));
  }
  if (fs::is_regular_file(fs::path(folder) / "dataset.sqlite"))
    note(progress, folder + " is a one-database dataset; `nxr-neuro split dataset` makes it a catalogue of subject packages (the default form).");
  return open_writer(folder, o);
}

// The commands that make or change a one-database dataset itself refuse a catalogue, naming what to use.
inline DatasetWriter open_one_database(const std::string& folder, const std::string& what, const OpenOptions& o = {}) {
  if (is_catalogue_dataset(folder))
    throw Refused(what + " works on a one-database dataset; " + folder + " is a catalogue of subject packages (`list subjects`, `show subject` and the `compute` verbs open its packages)", true);
  return open_writer(folder, o);
}

inline OpenOptions reader() {
  OpenOptions o;
  o.lock = false;
  return o;
}

}  // namespace nxr::io::task::commands_detail
