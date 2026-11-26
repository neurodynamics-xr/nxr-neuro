// nxr-neuro read — read an nxr dataset: list its subjects and every array its rows name (shape · dtype · CRC-32 of the
// decoded little-endian bytes), or read one array / region. (Step 1's `nxr-read`, folded into the one command.) It parses, calls the library's `read dataset`
// (nxr/io/task/commands.h `read_dataset`) and prints its result as tab-separated lines.
//
//   nxr-neuro read <dataset folder>
//   nxr-neuro read <dataset folder> --array <subject> <path> [--region a:b,c:d,...] [--out FILE]
#include <iostream>
#include <sstream>

#include "nxr/io/task/commands.h"

using namespace nxr::io;

namespace {

std::string shape_text(const Json& s) {
  std::string t = "[";
  for (std::size_t i = 0; i < s.size(); ++i) t += (i ? "," : "") + std::to_string(s[i].get<std::int64_t>());
  return t + "]";
}

}  // namespace

int read_command(int argc, char** argv) {
  try {
    if (argc < 2) { std::cerr << "usage: nxr-neuro read <dataset folder> [--array <subject> <path> [--region a:b,c:d] [--out FILE]]\n"; return 2; }
    task::ReadDatasetRequest rq;
    rq.folder = argv[1];
    std::string region;
    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--array" && i + 2 < argc) { rq.array = std::array<std::string, 2>{argv[i + 1], argv[i + 2]}; i += 2; }
      else if (a == "--region" && i + 1 < argc) region = argv[++i];
      else if (a == "--out" && i + 1 < argc) rq.out = argv[++i];
      else { std::cerr << "unknown argument: " << a << '\n'; return 2; }
    }
    if (rq.out && rq.out->empty()) rq.out.reset();
    if (rq.array && !region.empty()) {
      std::stringstream ss(region);
      for (std::string part; std::getline(ss, part, ',');) {
        const auto c = part.find(':');
        rq.region.push_back({std::stoll(part.substr(0, c)), std::stoll(part.substr(c + 1))});
      }
    }
    const task::CommandResult r = task::read_dataset(rq);
    if (rq.array) {
      std::cout << shape_text(r.result["shape"]) << '\t' << r.result["dtype"].get<std::string>() << '\t' << r.result["checksum"].get<std::string>() << '\n';
      return 0;
    }
    for (const auto& s : r.result["subjects"])
      std::cout << "subject\t" << s["name"].get<std::string>() << '\t' << s["path"].get<std::string>() << '\t' << s["kind"].get<std::string>() << '\n';
    for (const auto& x : r.result["arrays"])
      std::cout << x["subject"].get<std::string>() << '\t' << x["record"].get<std::string>() << '\t' << x["path"].get<std::string>() << '\t' << shape_text(x["shape"]) << '\t'
                << x["dtype"].get<std::string>() << '\t' << x["checksum"].get<std::string>() << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
