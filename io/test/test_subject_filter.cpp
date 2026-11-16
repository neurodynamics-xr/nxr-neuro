// The subject filter language, version 1: parsing, the checks (unknown variable, types, levels, syntax), precedence and three-valued logic on missing values.
#include "check.h"
#include "nxr/io/subject_filter.h"

using namespace nxr::io;

namespace {
std::string refusal(const std::string& text, const std::vector<VariableInfo>& vars) {
  try { SubjectFilter::parse(text, vars); } catch (const Error& e) { return e.what(); }
  return "";
}
bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }
}  // namespace

int main() {
  VariableInfo age{"age", VariableType::Number, std::string("years"), std::nullopt, std::nullopt};
  VariableInfo group{"group", VariableType::String, std::nullopt, std::nullopt, Json::array({"control", "patient"})};
  VariableInfo site{"site", VariableType::String, std::nullopt, std::nullopt, std::nullopt};
  VariableInfo handed{"handedness score", VariableType::Number, std::nullopt, std::nullopt, std::nullopt};
  VariableInfo smoker{"smoker", VariableType::Boolean, std::nullopt, std::nullopt, std::nullopt};
  const std::vector<VariableInfo> vars = {age, group, site, handed, smoker};

  const auto run = [&](const std::string& f, const ParticipantValues& v) { return SubjectFilter::parse(f, vars).evaluate(v); };
  const ParticipantValues old_control = {{"age", 70.0}, {"group", std::string("control")}, {"site", std::string("a")}, {"handedness score", 50.0}, {"smoker", false}};
  const ParticipantValues young_patient = {{"age", 30.0}, {"group", std::string("patient")}, {"site", std::string("c")}, {"handedness score", 10.0}, {"smoker", true}};

  // the spec's examples
  CHECK(run("group == 'control' and age >= 60", old_control) == Tri::True);
  CHECK(run("group == 'control' and age >= 60", young_patient) == Tri::False);
  CHECK(run("site in ('a', 'b') and not (group is missing)", old_control) == Tri::True);
  CHECK(run("site in ('a', 'b') and not (group is missing)", young_patient) == Tri::False);
  CHECK(run("`handedness score` > 40 or group != 'patient'", old_control) == Tri::True);
  CHECK(run("`handedness score` > 40 or group != 'patient'", young_patient) == Tri::False);
  for (const char* cmp : {"<", "<=", ">", ">=", "==", "!="}) CHECK(!refusal(std::string("age ") + cmp + " 5", vars).size());
  CHECK(run("age < -1.5e1", old_control) == Tri::False);
  CHECK(run("smoker == true", young_patient) == Tri::True);
  CHECK(run("smoker != true", young_patient) == Tri::False);
  const std::vector<VariableInfo> quoted = {{"o'brien score", VariableType::String, std::nullopt, std::nullopt, std::nullopt}};
  CHECK(SubjectFilter::parse("`o'brien score` == 'it''s'", quoted).evaluate({{"o'brien score", std::string("it's")}}) == Tri::True);

  // precedence: not > and > or, parentheses group
  CHECK(run("age > 50 or age < 40 and group == 'control'", old_control) == Tri::True);               // and first: true or (false and true)
  CHECK(run("(age > 50 or age < 40) and group == 'patient'", old_control) == Tri::False);
  CHECK(run("not age > 50 or group == 'control'", old_control) == Tri::True);                          // (not (age > 50)) or true
  CHECK(run("not (age > 50 or group == 'control')", old_control) == Tri::False);

  // missing values: SQL's three-valued logic; a subject is selected only when the whole filter is TRUE
  const ParticipantValues no_age = {{"group", std::string("control")}};
  CHECK(run("age >= 60", no_age) == Tri::Unknown);
  CHECK(run("age in (1, 2)", no_age) == Tri::Unknown);
  CHECK(run("not age >= 60", no_age) == Tri::Unknown);
  CHECK(run("age >= 60 and group == 'patient'", no_age) == Tri::False);          // unknown and false
  CHECK(run("age >= 60 or group == 'control'", no_age) == Tri::True);            // unknown or true
  CHECK(run("age >= 60 or group == 'patient'", no_age) == Tri::Unknown);
  CHECK(run("age is missing", no_age) == Tri::True);
  CHECK(run("age is missing", old_control) == Tri::False);
  CHECK(!SubjectFilter::parse("age >= 60", vars).selects(no_age));
  CHECK(!SubjectFilter::parse("not age >= 60", vars).selects(no_age));           // not unknown is unknown: never selected by accident
  CHECK(SubjectFilter::parse("age is missing", vars).selects(no_age));

  // refusals, each saying what is wrong
  std::string r = refusal("agee > 5", vars);
  CHECK(has(r, "unknown variable 'agee'") && has(r, "age, group, site, handedness score, smoker"));
  r = refusal("age > 5", {});
  CHECK(has(r, "no participants table"));
  CHECK(has(refusal("age == 'old'", vars), "is a number"));
  CHECK(has(refusal("group > 'a'", vars), "no order"));
  CHECK(has(refusal("group == 3", vars), "string variable"));
  CHECK(has(refusal("smoker > true", vars), "== and != only"));
  CHECK(has(refusal("smoker in (true)", vars), "== and != only"));
  CHECK(has(refusal("group == 'other'", vars), "one of its levels: 'control', 'patient'"));
  CHECK(has(refusal("group in ('control', 'x')", vars), "levels"));
  CHECK(has(refusal("age = 5", vars), "write '=='"));
  CHECK(has(refusal("age >= ", vars), "expected a value"));
  CHECK(has(refusal("age >= 5 and", vars), "position"));
  CHECK(has(refusal("(age >= 5", vars), "expected ')'"));
  CHECK(has(refusal("age >= 5 )", vars), "position 10"));
  CHECK(has(refusal("age is", vars), "expected 'missing'"));
  CHECK(has(refusal("age >= 5 age", vars), "expected 'and', 'or'"));
  CHECK(has(refusal("", vars), "expected a variable name"));
  CHECK(has(refusal("'a' == age", vars), "expected a variable name"));
  CHECK(has(refusal("age >= 5; DROP TABLE subject", vars), "unexpected character ';'"));
  CHECK(has(refusal("`age >= 5", vars), "never closed"));
  CHECK(has(refusal("group == 'control", vars), "never closed"));
  CHECK(has(refusal("in == 3", vars), "keyword"));
  CHECK(SubjectFilter::parse("age>=5", vars).text() == "age>=5");
  CHECK_EQ(kFilterLanguageVersion, 1);
  return nxrtest::finish("subject_filter");
}
