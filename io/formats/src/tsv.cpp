#include "nxr/formats/tsv.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "nxr/formats/errors.h"

namespace nxr::formats::tsv {

int Table::column(const std::string& name) const {
  for (std::size_t i = 0; i < columns.size(); ++i) if (columns[i] == name) return static_cast<int>(i);
  return -1;
}

const std::string& Table::cell(std::size_t row, const std::string& col) const {
  const int c = column(col);
  if (c < 0) throw range_error("TSV: no column \"" + col + "\"");
  if (row >= rows.size()) throw range_error("TSV: row " + std::to_string(row) + " of " + std::to_string(rows.size()));
  return rows[row][static_cast<std::size_t>(c)];
}

double Table::number(std::size_t row, const std::string& col) const {
  const std::string& s = cell(row, col);
  if (is_na(s)) return std::nan("");
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (s.empty() || *end != '\0') throw format_error("TSV: \"" + s + "\" in column \"" + col + "\" is not a number");
  return v;
}

Table read(const ByteSource& src) {
  std::string text = src.string(0, static_cast<std::size_t>(src.size()));
  std::size_t pos = 0;
  if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF && static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF) pos = 3;
  Table t;
  std::size_t line_no = 0;
  bool have_header = false;
  while (pos < text.size()) {
    std::size_t end = text.find('\n', pos);
    if (end == std::string::npos) end = text.size();
    std::string line = text.substr(pos, end - pos);
    pos = end + 1;
    ++line_no;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;   // blank lines (a trailing one) carry nothing
    std::vector<std::string> cells;
    std::size_t s = 0;
    while (true) {
      const std::size_t tab = line.find('\t', s);
      if (tab == std::string::npos) { cells.push_back(line.substr(s)); break; }
      cells.push_back(line.substr(s, tab - s));
      s = tab + 1;
    }
    if (!have_header) { t.columns = std::move(cells); have_header = true; continue; }
    if (cells.size() != t.columns.size())
      throw format_error("TSV: line " + std::to_string(line_no) + " has " + std::to_string(cells.size()) + " cells, the header has " + std::to_string(t.columns.size()));
    t.rows.push_back(std::move(cells));
  }
  if (!have_header) throw format_error("TSV: no header row");
  return t;
}

Table read_file(const std::string& path) {
  FileSource f(path);
  return read(f);
}

void write(ByteSink& sink, const Table& t) {
  auto check = [](const std::string& c) {
    if (c.find_first_of("\t\n\r") != std::string::npos) throw range_error("TSV: a cell holds a tab or a line break: \"" + c + "\"");
  };
  std::string out;
  for (std::size_t i = 0; i < t.columns.size(); ++i) {
    check(t.columns[i]);
    if (t.columns[i].empty()) throw range_error("TSV: an empty column name");
    if (i) out += '\t';
    out += t.columns[i];
  }
  out += '\n';
  for (const auto& row : t.rows) {
    if (row.size() != t.columns.size()) throw range_error("TSV: a row of " + std::to_string(row.size()) + " cells under " + std::to_string(t.columns.size()) + " columns");
    for (std::size_t i = 0; i < row.size(); ++i) {
      check(row[i]);
      if (i) out += '\t';
      out += row[i].empty() ? std::string("n/a") : row[i];
    }
    out += '\n';
  }
  sink.write(out);
}

void write_file(const std::string& path, const Table& t) {
  FileSink f(path);
  write(f, t);
  f.close();
}

std::string format_number(double v) {
  if (std::isnan(v)) return "n/a";
  if (std::isinf(v)) throw range_error("TSV: infinity has no BIDS spelling");
  char b[40];
  for (int prec = 1; prec <= 17; ++prec) {
    std::snprintf(b, sizeof b, "%.*g", prec, v);
    if (std::strtod(b, nullptr) == v) break;
  }
  return b;
}

nlohmann::json read_sidecar(const ByteSource& src) {
  try {
    return nlohmann::json::parse(src.string(0, static_cast<std::size_t>(src.size())));
  } catch (const nlohmann::json::exception& e) {
    throw format_error(std::string("JSON sidecar: ") + e.what());
  }
}

void write_sidecar(ByteSink& sink, const nlohmann::json& j) {
  sink.write(j.dump(4) + "\n");
}

}  // namespace nxr::formats::tsv
