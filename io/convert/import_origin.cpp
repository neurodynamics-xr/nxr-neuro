#include "nxr/convert/import_origin.h"

#include <filesystem>
#include <fstream>

#include "nxr/io/hash.h"
#include "nxr/io/subject_package.h"

namespace nxr::convert {

using nxr::io::Json;

Json origin_record(const ImportOrigin& o, const std::string& path) {
  Json r = Json{{"schema", "nxr-origin/1"}, {"source", o.format + ":" + (o.source.empty() ? std::filesystem::path(path).filename().string() : o.source)}, {"licence", o.licence},
                {"attribution", o.attribution}, {"redistributable", o.redistributable}};
  if (!o.source_file.empty() && std::filesystem::is_regular_file(o.source_file)) {
    std::ifstream in(o.source_file, std::ios::binary);
    nxr::io::Sha256 h;
    std::vector<char> buf(1 << 20);
    while (in) {
      in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
      h.update(reinterpret_cast<const std::uint8_t*>(buf.data()), static_cast<std::size_t>(in.gcount()));
    }
    r["source"] = r["source"].get<std::string>() + "#sha256=" + h.hex();
  }
  return r;
}

Json stamp_origin(nxr::io::DatasetWriter& w, const ImportOrigin& o, const std::string& path) {
  const Json rec = origin_record(o, path);
  int n = 0;
  for (const auto& it : nxr::io::items_without_origin(w.db())) {
    nxr::io::set_origin(w.db(), it.table, it.id, {rec["source"].get<std::string>(), o.licence, o.attribution, o.redistributable});
    ++n;
  }
  return Json{{"origin", rec}, {"origin_items", n}};
}

}  // namespace nxr::convert
