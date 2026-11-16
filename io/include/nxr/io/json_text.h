// nxr-io — the canonical JSON text (D145): JSON as JavaScript's JSON.stringify writes it.
//
// A node's zarr.json is its row: the document's own keys, then `"attributes":` and the text of the `<table>_attributes` view,
// byte for byte. Every `*_json` column is written the same way (`jdump`). nxr.io.database.js_number / js_json / zarr_json_text.
#pragma once
#include <string>
#include <nlohmann/json.hpp>

namespace nxr::io {

using Json = nlohmann::ordered_json;   // keys keep the order they were written in

// A number as ECMAScript Number::toString writes it: 0 -> "0", 1e-7 -> "1e-7", 1e21 -> "1e+21"; NaN and +-inf -> "null".
std::string js_number(double x);
// JSON.stringify of a value: compact, keys in order, non-ASCII as is, numbers JavaScript's way (an integer as an integer).
std::string js_json(const Json& v);
// A `*_json` column's text.
inline std::string jdump(const Json& v) { return js_json(v); }
// The canonical zarr.json: the document's keys as they are (but "attributes"), then "attributes": and `attributes` verbatim.
std::string zarr_json_text(const Json& doc, const std::string& attributes);
// The plain layout group (a folder without a row), as the Studio's writer creates it.
extern const char* const kEmptyGroup;

}  // namespace nxr::io
