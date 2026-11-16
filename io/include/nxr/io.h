// nxr-io — the nxr dataset format, read and written: include this for the whole API (namespace nxr::io).
//
//   reading   Dataset (the rows), ArrayReader (the arrays)                                 dataset.h · array_reader.h
//   writing   create_dataset · open_writer · DatasetWriter · SubjectWriter · remove_node    writer.h
//             array_meta · layout_array · populate_array · view(...)                        array_writer.h
//             Database · drain · text_parity · js_json                                      database.h · json_text.h
//             acquire_writer_lock                                                           writer_lock.h
//             folder_of · folder_for · safe_name · atlas_node                               layout.h
//             read_mesh · add_surface                                                       surface.h
//   errors    Error (a stable code() and nxr's wording)                                     errors.h
#pragma once
#include "nxr/io/array_reader.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/database.h"
#include "nxr/io/dataset.h"
#include "nxr/io/errors.h"
#include "nxr/io/chunk_loop.h"
#include "nxr/io/json_text.h"
#include "nxr/io/layout.h"
#include "nxr/io/store.h"
#include "nxr/io/surface.h"
#include "nxr/io/writer.h"
#include "nxr/io/statistic.h"
#include "nxr/io/writer_lock.h"
