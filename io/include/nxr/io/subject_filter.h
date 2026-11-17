// nxr-io — the SUBJECT FILTER LANGUAGE, version 1 (docs/specs/2026-11-16-subject-centred-datasets.md section 3.1): `--where FILTER`, a filter over a dataset's
// PARTICIPANT VARIABLES only. It is parsed and type-checked here and evaluated in memory over the participants table; the text is never passed to SQL and cannot
// reach any other table.
//
//   filter      = disjunction ;  disjunction = conjunction { "or" conjunction } ;  conjunction = negation { "and" negation } ;
//   negation    = "not" negation | primary ;  primary = "(" disjunction ")" | test ;
//   test        = variable comparator value | variable "in" "(" value { "," value } ")" | variable "is" "missing" ;
//   comparator  = "==" | "!=" | "<" | "<=" | ">" | ">=" ;
//   variable    = name | "`" any character except "`" "`" ;      name = letter { letter | digit | "_" }
//   value       = number | string | "true" | "false" ;           string = "'" ... "'" with '' for a quote
//
// `not` binds tightest, then `and`, then `or`. Checks, before anything is read (each refusal says what is wrong): an unknown variable (naming those that exist, or
// that there is no participants table); a type mismatch (a `number` variable takes numbers and every comparator; a `string` variable takes strings with ==, != and
// in; a `boolean` variable takes true/false with == and !=); a value outside a categorical variable's levels (naming them); a syntax error (its position and what
// was expected; `=` alone is refused with a hint to write `==`).
// MISSING VALUES follow SQL's three-valued logic: a comparison or `in` on a missing value is unknown; `not unknown` is unknown; `unknown and false` is false,
// `unknown or true` is true; a subject is selected only when the whole filter is TRUE. `is missing` is the one test true on a missing value.
// A new construct is a new language version, recorded in provenance; a version-1 filter keeps its meaning.
#pragma once
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "nxr/io/errors.h"
#include "nxr/io/json_text.h"

namespace nxr::io {

constexpr int kFilterLanguageVersion = 1;

enum class VariableType { Number, String, Boolean };
const char* variable_type_name(VariableType t);

// What is recorded about a variable (a row of the participants table's variable_dictionary).
struct VariableInfo {
  std::string name;
  VariableType type = VariableType::String;
  std::optional<std::string> unit, description;
  std::optional<Json> levels;   // a JSON list of the values a categorical variable may take
};

// One subject's value of one variable: absent from the map is a missing value.
using ParticipantValue = std::variant<double, std::string, bool>;
using ParticipantValues = std::map<std::string, ParticipantValue>;

enum class Tri { False, True, Unknown };

class SubjectFilter {
 public:
  // Parse and type-check `text` against the variables the participants table has (an empty list: there is no participants table, and any variable is refused).
  static SubjectFilter parse(const std::string& text, const std::vector<VariableInfo>& variables);
  // The value of the whole filter for one subject.
  Tri evaluate(const ParticipantValues& values) const;
  bool selects(const ParticipantValues& values) const { return evaluate(values) == Tri::True; }
  const std::string& text() const { return text_; }
  struct Node;

 private:
  std::string text_;
  std::shared_ptr<const Node> root_;
};

}  // namespace nxr::io
