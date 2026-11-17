// BIDS-style TSV tables (participants, channels, events, measurements) and their JSON sidecars.
//
// A table is a header row and rows of text cells. The reader takes UTF-8 with LF or CRLF line ends and an optional byte-order mark, and refuses a
// row whose cell count differs from the header's (with its line number). The writer writes UTF-8, LF, tab-separated, a header row first, and
// `n/a` for an empty cell or a NaN (BIDS has no empty cells); it refuses a cell holding a tab or a line break. No quoting, as BIDS has none.
#pragma once
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "nxr/formats/byte_source.h"

namespace nxr::formats::tsv {

struct Table {
  std::vector<std::string> columns;
  std::vector<std::vector<std::string>> rows;

  int column(const std::string& name) const;               // index, or -1
  const std::string& cell(std::size_t row, const std::string& column) const;   // throws code::out_of_range for a missing column or row
  static bool is_na(const std::string& cell) { return cell == "n/a"; }
  double number(std::size_t row, const std::string& column) const;             // "n/a" is NaN; other non-numbers throw code::format
};

Table read(const ByteSource& src);
Table read_file(const std::string& path);
void write(ByteSink& sink, const Table& table);
void write_file(const std::string& path, const Table& table);

// A number as a cell: the shortest decimal that reads back to the same double; "n/a" for NaN; "inf"/"-inf" are refused (BIDS has none).
std::string format_number(double v);

// The JSON sidecar: read, and write (indent 4, keys sorted, a final newline).
nlohmann::json read_sidecar(const ByteSource& src);
void write_sidecar(ByteSink& sink, const nlohmann::json& j);

}  // namespace nxr::formats::tsv
