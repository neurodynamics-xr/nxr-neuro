// TSV tables against pandas goldens (test/golden/gen_tsv.py), the writer's bytes, number formatting, sidecars, and refusals.
#include <cmath>
#include <fstream>
#include <functional>
#include <iterator>

#include "check.h"
#include "nxr/formats/errors.h"
#include "nxr/formats/tsv.h"

using namespace nxr::formats;
using nlohmann::json;

static const std::string kData = std::string(NXR_FORMATS_TEST_DIR) + "/data/";

static bool throws_code(const std::function<void()>& f, const char* code) {
  try { f(); } catch (const nxr::io::Error& e) { return e.code() == code; }
  return false;
}

int main() {
  std::ifstream gf(std::string(NXR_FORMATS_TEST_DIR) + "/golden/tsv_golden.json");
  const json g = json::parse(gf);
  for (const auto& [name, c] : g.items()) {
    const tsv::Table t = tsv::read_file(kData + "tsv_" + name + ".tsv");
    CHECK_EQ(t.columns.size(), c["columns"].size());
    for (std::size_t i = 0; i < t.columns.size() && i < c["columns"].size(); ++i) CHECK_EQ(t.columns[i], c["columns"][i].get<std::string>());
    CHECK_EQ(t.rows.size(), c["rows"].size());
    for (std::size_t r = 0; r < t.rows.size() && r < c["rows"].size(); ++r)
      for (std::size_t k = 0; k < t.rows[r].size(); ++k) CHECK_EQ(t.rows[r][k], c["rows"][r][k].get<std::string>());
  }
  {
    const tsv::Table t = tsv::read_file(kData + "tsv_lf.tsv");
    CHECK(tsv::Table::is_na(t.cell(1, "age")));
    CHECK(std::isnan(t.number(1, "age")));
    CHECK_EQ(t.number(2, "age"), 27.5);
    CHECK(throws_code([&] { t.number(0, "group"); }, code::format));
    CHECK(throws_code([&] { t.cell(0, "nope"); }, code::out_of_range));
    CHECK(throws_code([&] { t.cell(9, "age"); }, code::out_of_range));
    // read, write, read: the same table; the empty cell becomes n/a
    MemorySink out;
    tsv::write(out, t);
    MemorySource back(out.data());
    const tsv::Table t2 = tsv::read(back);
    CHECK(t2.columns == t.columns);
    CHECK_EQ(t2.rows[0][3], std::string("n/a"));
    CHECK_EQ(t2.rows[1][3], t.rows[1][3]);
  }
  {
    tsv::Table w;
    w.columns = {"onset", "duration", "label"};
    w.rows = {{"0", "0.5", "a"}, {"1.5", "", ""}};
    MemorySink out;
    tsv::write(out, w);
    std::ifstream gi(std::string(NXR_FORMATS_TEST_DIR) + "/golden/tsv_written_golden.tsv", std::ios::binary);
    const std::string want((std::istreambuf_iterator<char>(gi)), std::istreambuf_iterator<char>());
    CHECK_EQ(out.str(), want);   // the bytes pandas writes
    w.rows[0][2] = "tab\there";
    MemorySink bad;
    CHECK(throws_code([&] { tsv::write(bad, w); }, code::out_of_range));
  }
  // numbers
  CHECK_EQ(tsv::format_number(0.1), std::string("0.1"));
  CHECK_EQ(tsv::format_number(1.0), std::string("1"));
  CHECK_EQ(tsv::format_number(-2.5e-7), std::string("-2.5e-07"));
  CHECK_EQ(tsv::format_number(std::nan("")), std::string("n/a"));
  CHECK_EQ(std::stod(tsv::format_number(1.0 / 3.0)), 1.0 / 3.0);
  CHECK(throws_code([&] { tsv::format_number(INFINITY); }, code::out_of_range));
  // refusals
  CHECK(throws_code([&] { tsv::read_file(kData + "tsv_ragged.tsv"); }, code::format));
  CHECK(throws_code([&] { MemorySource m(std::string("")); tsv::read(m); }, code::format));
  // sidecars
  {
    MemorySink out;
    tsv::write_sidecar(out, json{{"SamplingFrequency", 250}, {"Description", "x"}});
    CHECK_EQ(out.str(), std::string("{\n    \"Description\": \"x\",\n    \"SamplingFrequency\": 250\n}\n"));
    MemorySource m(out.data());
    CHECK_EQ(tsv::read_sidecar(m)["SamplingFrequency"].get<int>(), 250);
    MemorySource badj(std::string("{not json"));
    CHECK(throws_code([&] { tsv::read_sidecar(badj); }, code::format));
  }
  return nxrtest::finish("formats_tsv");
}
