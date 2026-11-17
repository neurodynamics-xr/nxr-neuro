// a fixture the guard must refuse (test layering_nodes_without_io_catches): a node that includes io and reaches the database
#include "nxr/io/statistic.h"
void node(nxr::io::SubjectWriter& sub) { sub.db().read("manifold", "x"); }
