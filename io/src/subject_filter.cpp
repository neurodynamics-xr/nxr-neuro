#include "nxr/io/subject_filter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>

namespace nxr::io {

const char* variable_type_name(VariableType t) {
  switch (t) {
    case VariableType::Number: return "number";
    case VariableType::String: return "string";
    case VariableType::Boolean: return "boolean";
  }
  return "";
}

struct SubjectFilter::Node {
  enum class Kind { Or, And, Not, Compare, In, IsMissing } kind = Kind::Compare;
  std::vector<std::shared_ptr<const Node>> kids;   // Or / And: two or more; Not: one
  std::string variable;
  std::string comparator;                          // Compare
  std::vector<ParticipantValue> values;            // Compare: one; In: the list
};

namespace {
using Node = SubjectFilter::Node;
using NodePtr = std::shared_ptr<const Node>;

Error refuse(const std::string& m) { return Error(code::invalid_argument, "filter: " + m); }

struct Token {
  enum class Kind { Word, Variable, Number, String, Op, End } kind = Kind::End;
  std::string text;       // Word / Variable / Op; String: its value
  double number = 0;
  std::size_t pos = 0;    // 1-based column
};

std::vector<Token> tokenize(const std::string& s) {
  std::vector<Token> out;
  std::size_t i = 0;
  const auto at = [&](std::size_t k) { return k < s.size() ? s[k] : '\0'; };
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (std::isspace(c)) { ++i; continue; }
    Token t;
    t.pos = i + 1;
    if (c == '(' || c == ')' || c == ',') { t.kind = Token::Kind::Op; t.text = std::string(1, s[i]); ++i; }
    else if (c == '=' ) {
      if (at(i + 1) != '=') throw refuse("position " + std::to_string(i + 1) + ": '=' alone is not a comparison; write '==' to test equality");
      t.kind = Token::Kind::Op; t.text = "=="; i += 2;
    } else if (c == '!') {
      if (at(i + 1) != '=') throw refuse("position " + std::to_string(i + 1) + ": '!' must be followed by '=' ('!='); write `not` to negate a test");
      t.kind = Token::Kind::Op; t.text = "!="; i += 2;
    } else if (c == '<' || c == '>') {
      t.kind = Token::Kind::Op; t.text = std::string(1, s[i]); ++i;
      if (at(i) == '=') { t.text += '='; ++i; }
    } else if (c == '`') {
      const auto end = s.find('`', i + 1);
      if (end == std::string::npos) throw refuse("position " + std::to_string(i + 1) + ": a variable name opened with ` is never closed");
      t.kind = Token::Kind::Variable; t.text = s.substr(i + 1, end - i - 1);
      if (t.text.empty()) throw refuse("position " + std::to_string(i + 1) + ": an empty variable name");
      i = end + 1;
    } else if (c == '\'') {
      std::string v;
      std::size_t k = i + 1;
      bool closed = false;
      while (k < s.size()) {
        if (s[k] == '\'') {
          if (at(k + 1) == '\'') { v += '\''; k += 2; continue; }
          closed = true; ++k; break;
        }
        v += s[k++];
      }
      if (!closed) throw refuse("position " + std::to_string(i + 1) + ": a string opened with ' is never closed");
      t.kind = Token::Kind::String; t.text = v; i = k;
    } else if (std::isdigit(c) || (c == '-' && std::isdigit(static_cast<unsigned char>(at(i + 1))))) {
      std::size_t k = i + 1;
      while (std::isdigit(static_cast<unsigned char>(at(k)))) ++k;
      if (at(k) == '.' && std::isdigit(static_cast<unsigned char>(at(k + 1)))) { ++k; while (std::isdigit(static_cast<unsigned char>(at(k)))) ++k; }
      if (at(k) == 'e' || at(k) == 'E') {
        std::size_t e = k + 1;
        if (at(e) == '+' || at(e) == '-') ++e;
        if (std::isdigit(static_cast<unsigned char>(at(e)))) { while (std::isdigit(static_cast<unsigned char>(at(e)))) ++e; k = e; }
      }
      t.kind = Token::Kind::Number; t.text = s.substr(i, k - i); t.number = std::strtod(t.text.c_str(), nullptr); i = k;
    } else if (std::isalpha(c)) {
      std::size_t k = i + 1;
      while (std::isalnum(static_cast<unsigned char>(at(k))) || at(k) == '_') ++k;
      t.kind = Token::Kind::Word; t.text = s.substr(i, k - i); i = k;
    } else {
      throw refuse("position " + std::to_string(i + 1) + ": unexpected character '" + std::string(1, s[i]) + "'");
    }
    out.push_back(std::move(t));
  }
  Token end;
  end.pos = s.size() + 1;
  out.push_back(end);
  return out;
}

std::string names_of(const std::vector<VariableInfo>& vars) {
  std::string out;
  for (const auto& v : vars) out += (out.empty() ? "" : ", ") + v.name;
  return out;
}

class Parser {
 public:
  Parser(std::vector<Token> toks, const std::vector<VariableInfo>& vars) : t_(std::move(toks)), vars_(vars) {}
  NodePtr run() {
    NodePtr n = disjunction();
    if (cur().kind != Token::Kind::End) throw refuse(where(cur()) + ": expected 'and', 'or' or the end of the filter, found " + describe(cur()));
    return n;
  }

 private:
  const Token& cur() const { return t_[i_]; }
  static std::string where(const Token& t) { return "position " + std::to_string(t.pos); }
  static std::string describe(const Token& t) {
    switch (t.kind) {
      case Token::Kind::End: return "the end of the filter";
      case Token::Kind::String: return "the string '" + t.text + "'";
      case Token::Kind::Variable: return "the variable `" + t.text + "`";
      default: return "'" + t.text + "'";
    }
  }
  bool word(const char* w) const { return cur().kind == Token::Kind::Word && cur().text == w; }
  bool op(const char* o) const { return cur().kind == Token::Kind::Op && cur().text == o; }

  NodePtr disjunction() {
    std::vector<NodePtr> kids{conjunction()};
    while (word("or")) { ++i_; kids.push_back(conjunction()); }
    return kids.size() == 1 ? kids[0] : make(Node::Kind::Or, std::move(kids));
  }
  NodePtr conjunction() {
    std::vector<NodePtr> kids{negation()};
    while (word("and")) { ++i_; kids.push_back(negation()); }
    return kids.size() == 1 ? kids[0] : make(Node::Kind::And, std::move(kids));
  }
  NodePtr negation() {
    if (word("not")) { ++i_; return make(Node::Kind::Not, {negation()}); }
    return primary();
  }
  NodePtr primary() {
    if (op("(")) {
      ++i_;
      NodePtr n = disjunction();
      if (!op(")")) throw refuse(where(cur()) + ": expected ')', found " + describe(cur()));
      ++i_;
      return n;
    }
    return test();
  }
  static NodePtr make(Node::Kind k, std::vector<NodePtr> kids) {
    auto n = std::make_shared<Node>();
    n->kind = k;
    n->kids = std::move(kids);
    return n;
  }

  const VariableInfo& variable(const Token& t) const {
    for (const auto& v : vars_) if (v.name == t.text) return v;
    if (vars_.empty()) throw refuse(where(t) + ": unknown variable '" + t.text + "': this dataset has no participants table");
    throw refuse(where(t) + ": unknown variable '" + t.text + "'; the participants table has: " + names_of(vars_));
  }

  ParticipantValue value(const VariableInfo& v) {
    const Token t = cur();
    ParticipantValue out;
    if (t.kind == Token::Kind::Number) {
      if (v.type != VariableType::Number) throw refuse(where(t) + ": '" + v.name + "' is a " + variable_type_name(v.type) + " variable; " + t.text + " is a number");
      out = t.number;
    } else if (t.kind == Token::Kind::String) {
      if (v.type != VariableType::String) throw refuse(where(t) + ": '" + v.name + "' is a " + variable_type_name(v.type) + " variable; '" + t.text + "' is a string");
      out = t.text;
    } else if (t.kind == Token::Kind::Word && (t.text == "true" || t.text == "false")) {
      if (v.type != VariableType::Boolean) throw refuse(where(t) + ": '" + v.name + "' is a " + variable_type_name(v.type) + " variable; " + t.text + " is a boolean");
      out = (t.text == "true");
    } else {
      throw refuse(where(t) + ": expected a value (a number, a 'string', true or false), found " + describe(t));
    }
    ++i_;
    if (v.levels && v.levels->is_array()) {
      bool ok = false;
      for (const auto& lv : *v.levels) {
        if (std::holds_alternative<double>(out) && lv.is_number() && lv.get<double>() == std::get<double>(out)) ok = true;
        if (std::holds_alternative<std::string>(out) && lv.is_string() && lv.get<std::string>() == std::get<std::string>(out)) ok = true;
        if (std::holds_alternative<bool>(out) && lv.is_boolean() && lv.get<bool>() == std::get<bool>(out)) ok = true;
      }
      if (!ok) {
        std::string levels;
        for (const auto& lv : *v.levels) levels += (levels.empty() ? "" : ", ") + (lv.is_string() ? "'" + lv.get<std::string>() + "'" : lv.dump());
        throw refuse(where(t) + ": '" + v.name + "' takes one of its levels: " + levels);
      }
    }
    return out;
  }

  NodePtr test() {
    const Token vt = cur();
    if (vt.kind != Token::Kind::Word && vt.kind != Token::Kind::Variable)
      throw refuse(where(vt) + ": expected a variable name, '(' or 'not', found " + describe(vt));
    if (vt.kind == Token::Kind::Word && (vt.text == "and" || vt.text == "or" || vt.text == "not" || vt.text == "in" || vt.text == "is" || vt.text == "missing" || vt.text == "true" || vt.text == "false"))
      throw refuse(where(vt) + ": expected a variable name, found the keyword '" + vt.text + "' (write a variable with that name between backticks)");
    const VariableInfo& var = variable(vt);
    ++i_;
    auto n = std::make_shared<Node>();
    n->variable = var.name;
    if (word("is")) {
      ++i_;
      if (!word("missing")) throw refuse(where(cur()) + ": expected 'missing' after 'is', found " + describe(cur()));
      ++i_;
      n->kind = Node::Kind::IsMissing;
      return n;
    }
    if (word("in")) {
      const Token in_tok = cur();
      ++i_;
      if (var.type == VariableType::Boolean) throw refuse(where(in_tok) + ": '" + var.name + "' is a boolean variable: it takes == and != only");
      if (!op("(")) throw refuse(where(cur()) + ": expected '(' after 'in', found " + describe(cur()));
      ++i_;
      n->kind = Node::Kind::In;
      n->values.push_back(value(var));
      while (op(",")) { ++i_; n->values.push_back(value(var)); }
      if (!op(")")) throw refuse(where(cur()) + ": expected ',' or ')', found " + describe(cur()));
      ++i_;
      return n;
    }
    if (cur().kind == Token::Kind::Op && (cur().text == "==" || cur().text == "!=" || cur().text == "<" || cur().text == "<=" || cur().text == ">" || cur().text == ">=")) {
      const Token ct = cur();
      ++i_;
      const bool ordering = ct.text != "==" && ct.text != "!=";
      if (ordering && var.type == VariableType::String) throw refuse(where(ct) + ": '" + var.name + "' is a string variable: text has no order, use == != or in");
      if (ordering && var.type == VariableType::Boolean) throw refuse(where(ct) + ": '" + var.name + "' is a boolean variable: it takes == and != only");
      n->kind = Node::Kind::Compare;
      n->comparator = ct.text;
      n->values.push_back(value(var));
      return n;
    }
    throw refuse(where(cur()) + ": expected a comparison (== != < <= > >=), 'in' or 'is missing' after '" + var.name + "', found " + describe(cur()));
  }

  std::vector<Token> t_;
  const std::vector<VariableInfo>& vars_;
  std::size_t i_ = 0;
};

bool same(const ParticipantValue& a, const ParticipantValue& b) { return a == b; }

Tri tri_not(Tri a) { return a == Tri::Unknown ? Tri::Unknown : (a == Tri::True ? Tri::False : Tri::True); }

Tri run(const Node& n, const ParticipantValues& vals) {
  switch (n.kind) {
    case Node::Kind::Or: {
      Tri r = Tri::False;
      for (const auto& k : n.kids) {
        const Tri v = run(*k, vals);
        if (v == Tri::True) return Tri::True;
        if (v == Tri::Unknown) r = Tri::Unknown;
      }
      return r;
    }
    case Node::Kind::And: {
      Tri r = Tri::True;
      for (const auto& k : n.kids) {
        const Tri v = run(*k, vals);
        if (v == Tri::False) return Tri::False;
        if (v == Tri::Unknown) r = Tri::Unknown;
      }
      return r;
    }
    case Node::Kind::Not: return tri_not(run(*n.kids[0], vals));
    case Node::Kind::IsMissing: return vals.count(n.variable) ? Tri::False : Tri::True;
    case Node::Kind::In: {
      const auto it = vals.find(n.variable);
      if (it == vals.end()) return Tri::Unknown;
      for (const auto& v : n.values) if (same(it->second, v)) return Tri::True;
      return Tri::False;
    }
    case Node::Kind::Compare: {
      const auto it = vals.find(n.variable);
      if (it == vals.end()) return Tri::Unknown;
      const ParticipantValue& want = n.values[0];
      if (n.comparator == "==") return same(it->second, want) ? Tri::True : Tri::False;
      if (n.comparator == "!=") return same(it->second, want) ? Tri::False : Tri::True;
      const double a = std::holds_alternative<double>(it->second) ? std::get<double>(it->second) : std::nan("");
      const double b = std::holds_alternative<double>(want) ? std::get<double>(want) : std::nan("");
      if (std::isnan(a) || std::isnan(b)) return Tri::Unknown;
      if (n.comparator == "<") return a < b ? Tri::True : Tri::False;
      if (n.comparator == "<=") return a <= b ? Tri::True : Tri::False;
      if (n.comparator == ">") return a > b ? Tri::True : Tri::False;
      return a >= b ? Tri::True : Tri::False;
    }
  }
  return Tri::Unknown;
}
}  // namespace

SubjectFilter SubjectFilter::parse(const std::string& text, const std::vector<VariableInfo>& variables) {
  SubjectFilter f;
  f.text_ = text;
  Parser p(tokenize(text), variables);
  f.root_ = p.run();
  return f;
}

Tri SubjectFilter::evaluate(const ParticipantValues& values) const { return run(*root_, values); }

}  // namespace nxr::io
