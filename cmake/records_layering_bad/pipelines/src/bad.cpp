// A tree the guard must refuse (test layering_sql_only_in_io_catches): SQL and the handle in a pipeline.
#include "nxr/io/database.h"
void f(Sub& sub) { sub.db().all("SELECT * FROM field WHERE id = ?"); }
