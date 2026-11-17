// nxr-pipelines — the task-level io API under its short name: `task::Subject`, `task::Want`, `task::Node` (nxr/io/task/task.h). Pipelines find their inputs,
// ask what is measured and commit results through it, and never see a row, a table or a database.
#pragma once
#include "nxr/io/task/task.h"

namespace nxr::pipelines {
namespace task = nxr::io::task;
}  // namespace nxr::pipelines
