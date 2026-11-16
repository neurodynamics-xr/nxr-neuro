// nxr-neuro's output: the one place that decides how a command's lines reach the person or the script.
//
//   JSON   (`--json`, or stdout is not a terminal): today's JSON lines on stdout, byte for byte; a failure is a `stage: error` line.
//   human  (stdout is a terminal): a progress bar on stderr, then on stdout the result line (what was done, cache hit or computed, the range used) and what was stored.
// Progress and diagnostics go to stderr, results to stdout. Colour only in a terminal and never when NO_COLOR is set. `--quiet` prints only the result line.
#pragma once
#include <string>

namespace cli {

struct OutputState {
  bool human = false;          // the readable form
  bool json_asked = false;     // `--json` was given (errors then carry their code)
  bool quiet = false;
  bool color = false;
  std::string command;         // "compute psd": named in the result line
};
OutputState& output();

/** decide the form: `--json` / `--human` win; else a terminal on stdout is human. NO_COLOR and a terminal decide colour. */
void configure_output(bool json_flag, bool human_flag, bool quiet, bool no_color_flag, const std::string& command);

/** a failure: the legacy `stage: error` line (JSON), or `nxr-neuro: error: ...` on stderr (human); `code` is added to the JSON object when `--json` was asked for */
void report_error(const std::string& message, const std::string& code);

}  // namespace cli
