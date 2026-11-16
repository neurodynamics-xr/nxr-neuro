#include "output.h"

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "cli.h"

namespace cli {

OutputState& output() {
  static OutputState s;
  return s;
}

void configure_output(bool json_flag, bool human_flag, bool quiet, bool no_color_flag, const std::string& command) {
  OutputState& o = output();
  o.json_asked = json_flag;
  o.human = json_flag ? false : human_flag ? true : (isatty(STDOUT_FILENO) != 0);
  o.quiet = quiet;
  const char* nc = std::getenv("NO_COLOR");
  o.color = o.human && !no_color_flag && !(nc && *nc) && isatty(STDOUT_FILENO) != 0 && isatty(STDERR_FILENO) != 0;
  o.command = command;
}

namespace {

std::string paint(const OutputState& o, const char* code, const std::string& t) { return o.color ? std::string("\033[") + code + "m" + t + "\033[0m" : t; }

std::string compact(const Json& v, std::size_t limit = 110) {
  std::string t = v.is_string() ? v.get<std::string>() : pyjson::dumps(v);
  if (t.size() > limit) t = t.substr(0, limit - 3) + "...";
  return t;
}

// a list of flat objects as an aligned table
void table(std::ostream& out, const OutputState& o, const Json& rows, const std::string& indent) {
  std::vector<std::string> cols;
  for (const Json& r : rows)
    if (r.is_object())
      for (auto it = r.begin(); it != r.end(); ++it)
        if (std::find(cols.begin(), cols.end(), it.key()) == cols.end()) cols.push_back(it.key());
  std::vector<std::size_t> w(cols.size());
  for (std::size_t c = 0; c < cols.size(); ++c) {
    w[c] = cols[c].size();
    for (const Json& r : rows) w[c] = std::max(w[c], r.contains(cols[c]) ? compact(r[cols[c]], 60).size() : 0);
  }
  std::string head = indent;
  for (std::size_t c = 0; c < cols.size(); ++c) { std::ostringstream cell; cell << std::left << std::setw(static_cast<int>(w[c]) + 2) << cols[c]; head += cell.str(); }
  out << paint(o, "2", head) << '\n';
  for (const Json& r : rows) {
    out << indent;
    for (std::size_t c = 0; c < cols.size(); ++c) out << std::left << std::setw(static_cast<int>(w[c]) + 2) << (r.contains(cols[c]) ? compact(r[cols[c]], 60) : std::string());
    out << '\n';
  }
}

bool is_table(const Json& v) {
  if (!v.is_array() || v.empty()) return false;
  for (const Json& r : v) {
    if (!r.is_object()) return false;
    for (auto it = r.begin(); it != r.end(); ++it) if (it.value().is_array() || it.value().is_object()) return false;
  }
  return true;
}

// the range a computation used (after snapping to the method's boundaries): the producer's span record, or its text
std::string range_text(const Json& span) {
  if (span.is_string()) return span.get<std::string>();
  if (span.is_object() && span.contains("start_s") && span.contains("stop_s")) {
    std::ostringstream t;
    t << "[" << span["start_s"].get<double>() << ", " << span["stop_s"].get<double>() << ") s";
    if (span.contains("chunks") && span["chunks"].is_string()) t << " (" << span["chunks"].get<std::string>() << ")";
    return t.str();
  }
  return {};
}

std::int64_t last_bar_width = 0;

void progress_line(const OutputState& o, const Json& j) {
  const std::int64_t done = j.value("done", 0), count = j.value("count", 0);
  std::ostringstream t;
  const double sec = j.contains("seconds") && j["seconds"].is_number() ? j["seconds"].get<double>() : 0.0;
  if (isatty(STDERR_FILENO)) {
    const int width = 24;
    const int filled = count > 0 ? static_cast<int>(width * done / count) : 0;
    t << "\r  [" << std::string(static_cast<std::size_t>(filled), '#') << std::string(static_cast<std::size_t>(width - filled), '-') << "] " << done << "/" << count << " chunks";
    if (sec > 0) t << "  " << std::fixed << std::setprecision(1) << sec << " s";
    if (done >= count) t << "\n";
    std::cerr << t.str() << std::flush;
    last_bar_width = static_cast<std::int64_t>(t.str().size());
  } else {
    std::cerr << "  chunk " << done << "/" << count << "\n";
  }
  (void)o;
}

void render(const Json& j) {
  const OutputState& o = output();
  const std::string stage = j.is_object() && j.contains("stage") && j["stage"].is_string() ? j["stage"].get<std::string>() : "";
  if (stage == "chunk") { if (!o.quiet) progress_line(o, j); return; }
  if (stage == "error") { report_error(j.value("message", std::string("failed")), j.value("code", std::string())); return; }
  if (stage != "result") {   // a stage of the run: one dim line on stderr
    if (o.quiet) return;
    std::string t = "  " + stage;
    for (auto it = j.begin(); it != j.end(); ++it) if (it.key() != "stage") t += "  " + it.key() + "=" + compact(it.value(), 60);
    std::cerr << paint(o, "2", t) << "\n";
    return;
  }
  // the result line: what was done, computed or cache hit, how long, the range used
  std::string line = o.command.empty() ? "done" : o.command + ": " + paint(o, "32", "done");
  std::vector<std::string> notes;
  if (j.contains("cache") && j["cache"].is_string()) notes.push_back(j["cache"] == "hit" ? "cache hit" : "computed");
  if (j.contains("seconds") && j["seconds"].is_number()) { std::ostringstream t; t << std::fixed << std::setprecision(2) << j["seconds"].get<double>() << " s"; notes.push_back(t.str()); }
  const std::string range = j.contains("span") ? range_text(j["span"]) : std::string();
  if (!range.empty()) notes.push_back("range " + range);
  for (std::size_t i = 0; i < notes.size(); ++i) line += (i ? ", " : " (") + notes[i];
  if (!notes.empty()) line += ")";
  std::cout << line << "\n";
  if (o.quiet) return;
  std::size_t kw = 0;
  for (auto it = j.begin(); it != j.end(); ++it) kw = std::max(kw, it.key().size());
  for (auto it = j.begin(); it != j.end(); ++it) {
    const std::string& k = it.key();
    if (k == "stage" || k == "seconds" || k == "span" || (k == "cache" && it.value().is_string())) continue;
    if (is_table(it.value())) {
      std::cout << "  " << k << ":\n";
      table(std::cout, o, it.value(), "    ");
    } else {
      std::cout << "  " << std::left << std::setw(static_cast<int>(kw) + 2) << (k + ":") << compact(it.value()) << "\n";
    }
  }
}

}  // namespace

void emit(const Json& obj) {
  if (!output().human) { std::cout << pyjson::dumps(obj) << std::endl; return; }
  render(obj);
}

void report_error(const std::string& message, const std::string& code) {
  const OutputState& o = output();
  if (!o.human) {
    Json e{{"stage", "error"}, {"message", message}};
    if (o.json_asked && !code.empty()) e["code"] = code;
    std::cout << pyjson::dumps(e) << std::endl;
    return;
  }
  std::cerr << paint(o, "1;31", "nxr-neuro: error:") << " " << message << (code.empty() ? "" : "  [" + code + "]") << "\n";
}

}  // namespace cli
