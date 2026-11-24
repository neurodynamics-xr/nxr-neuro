// The layering guard's fixture (test layering_recordings_read_by_chunk_service_brackets_catches): after a comment with an unbalanced bracket,
// a half-open interval [start, stop), a real SQL string must still be caught, at its own line. The guard must FAIL.
int a = 0;
auto q = "SELECT id FROM field";
