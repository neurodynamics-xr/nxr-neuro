// nxr-neuro's commands, declared once (cli/command_table.cpp): every command's verb and noun, its old spelling, its options with meaning, default and unit, an example.
// The command line (CLI11), `--help`, the man pages, the shell completion and the alias table are all made from this one table, so they cannot drift apart; a test
// (`nxr-neuro __check-options`) compares each command's options with what its parser accepts.
#pragma once
#include <string>
#include <vector>

namespace cli {

struct OptionDoc {
  std::string name, value, meaning, def, unit;   // value: the placeholder in help (empty: a flag)
  bool show_default = false;                     // print the default even when it reads "none" (a pipeline parameter whose default is the word none)
};
struct OptionUse {
  std::string name;                      // the name on the new command line
  std::string legacy;                    // the name the parser knows it by, when different (`--h` is `--exceedances`)
  bool required = false, multi = false;
  bool param = false;                    // a parameter of a pipeline definition: `--<parameter> VALUE` (dashes for underscores); the parser reads it as `--set parameter=VALUE`
};
struct CommandDoc {
  std::string verb, noun;                // "run", "psd"
  std::vector<std::string> old;          // the old spelling's words ("psd", "compute"); empty: a new command
  std::string group;                     // the topic a computation is listed under
  std::string summary, example;
  std::string positional;                // the placeholder of the one positional argument, or empty
  bool positional_optional = false;      // `pos: FILE?`: the positional argument may be left out (`add surface` from --vertices and --faces)
  bool span = false;                     // a computation over a recording: needs --from/--to or --all
  bool seconds = true;                   // the span may be given in seconds (else only --all)
  bool alias = true;                     // the old spelling is a hidden alias (false: the new command shares its words)
  bool refused = false;                  // the old spelling is not kept: it is refused, naming this command (`alias: refused`; spec 2026-11-17-compute-and-run.md, 6.3)
  std::string definition;                // a pipeline definition's command: the definition (a shipped name or a path) it runs; its parameters are the options
  bool unprobed = false;                 // its parser is not parse() (cannot be checked against the table)
  std::vector<OptionUse> options;        // the parser's own options (old spelling), required ones marked
  std::string path() const { return noun.empty() ? verb : verb + " " + noun; }
};

const std::vector<CommandDoc>& commands();
/** every pipeline `run` takes: the hand-wired ones (the table's `run <what>`) and the shipped definitions; a name is in one table only (`nxr-neuro __check-pipelines`) */
std::vector<std::string> pipeline_names();
/** an option's help; `path` ("run psd") picks a command's own wording when the table has one (`--set@run psd`) */
const OptionDoc& option_doc(const std::string& name, const std::string& path = "");
bool is_flag(const std::string& name);
/** the verbs in the order they are listed, with a sentence each */
const std::vector<std::pair<std::string, std::string>>& verbs();

std::string completion_script(const std::string& shell);   // bash | zsh | fish
std::string man_page(const std::string& verb, const std::string& noun);   // roff; both empty: the command itself
/** seconds from "90", "1:30" or "1:01:30"; Usage error otherwise */
double parse_time(const std::string& option, const std::string& text);

int run_command_line(int argc, char** argv);

}  // namespace cli
