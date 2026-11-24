// The layering guard's fixture (cmake/check_layering.cmake, test layering_recordings_read_by_chunk_service_brackets): a comment with an unbalanced bracket,
// a half-open interval [start, stop), must not merge the lines after it; the word SELECT below is in a comment, not SQL text. The guard must PASS.
auto s = "x";
int b = 1;  // so a node can SELECT a row
