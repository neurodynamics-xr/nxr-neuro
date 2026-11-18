# Test audit (2026-11-18)

Every test was reviewed against four rules (CONTRIBUTING.md, *Test kinds*): a refactor gate is removed with its refactor; tests check behaviour (values,
shapes, refusals with their reason), never the layout of a provenance record, log text or private names; a binding test checks the binding; an
integration test is small. Decisions: KEEP, REMOVE, MERGE, REWRITE, PORT (kept out of the build until ported).

## Counts (ctest, from `tests/kinds.tsv`)

| kind | before | after |
|---|---|---|
| unit | 129 | 125 |
| golden | 77 | 73 |
| integration | 62 | 62 |
| guard | 13 | 13 |
| **total** | **281** | **273** |

The totals include the three PORT tests, which are listed in the table but not built by default (`NXR_BUILD_UNPORTED_TESTS`): the default
configure registers 278 tests before and 270 after.

C++: 8 removed, 0 merged, 9 rewritten, 3 left to port. Bindings: 2 scripts removed, 1 Python test rewritten.

## REMOVE

| test | kind | reason |
|---|---|---|
| test_refactor_6b | unit | refactor gate: old functions copied into the test, compared with the merged ones |
| test_row31_operators | unit | refactor gate: operators module against the construction it replaced |
| test_frames_gc | unit | refactor gate: frames against geometry-central, bit for bit, for the rows that now take them from it |
| test_factor_solvers | unit | refactor gate for replacing the factor cache by geometry-central's solvers |
| test_pcg | unit | refactor gate: pcg against the loops it replaced; pcg's behaviour is covered by test_opticalflow_methods (default method, equal to direct) |
| test_frames_wt | golden | golden recorded from the old frames path (taps + sinc); the exact behaviour is test_wt_at's |
| test_signal_banks_golden | golden | hashes recorded from the transforms before their refactor; the transforms keep their reference goldens (test_filterbank, test_timebank, test_jointbank, test_signal_grid) |
| test_welch_bits | golden | hashes recorded from the Welch classes before their composition; parity with scipy, nxr and Brainstorm and chunk invariance stay in test_welch |

Removed with them: their old-code goldens and generators (`frames_wt_golden.txt`, `signal_banks_golden.txt`, `welch_bits_golden.txt`,
`psd_linear_map_golden.txt` and the four `gen_*` files), `golden_hash.h`, `signal_banks_inputs.h`, `welch_inputs.h`.

## REWRITE

| test | kind | change |
|---|---|---|
| test_psd_linear_map | golden → unit | the bitwise comparison with the old `sourcePower` dropped; the behaviour kept (diag(K S Kᵀ) is the power of the mapped series, orientations summed, the refusal) |
| test_filter_pipeline | integration | the assertions on the provenance record's layout dropped (the derivation is checked by `derived_from_id`, the values by the PSD) |
| test_headmodel | integration | provenance and option echo dropped; the gain is checked against the direct computation |
| test_meeg_pipelines | integration | option echo (depth, SNR, regularisation) and provenance dropped; the kernel's kind (`measure`, `orientation`), which definitions look kernels up by, kept |
| test_registration | integration | the provenance's method dropped; the stored matrix equals the compute function's for that method |
| test_runner_execute | integration | provenance layout dropped; the values against the direct computation kept |
| test_band_bursts | integration | the thresholds' record layout dropped; the spans against the planted bursts kept |
| test_wt | unit (slow) | the tiny-chunk case uses chunks of 5 samples instead of 1 (each chunk is mostly halo, so 1 multiplied the work); the bound is per chunk, independent of size |
| test_isoline_methods | unit | a 10 242-vertex icosphere instead of 163 842, and no timing (performance belongs in `bench/`); the assertion (the same segments) does not depend on size |
| bindings/python/tests/test_run.py | Python | the assertion on `meta["producer"]["bad_segments"]` dropped; the bad span is checked by the values (the PSD equals the pure function with the span left out, and differs from the plain one) |

## PORT (behind `NXR_BUILD_UNPORTED_TESTS`; written for the hand-wired verbs the definitions replaced)

| test | reason it stays |
|---|---|
| test_patches_pipeline | the `patches` definition is live and has no other end-to-end test |
| test_cellquadrature | the `cellquadrature` definition is live; its node has no other test |
| test_scale_selection | the `scaleselection` definition is live; its compute functions are unit-tested (test_scale, test_scalespace), the pipeline is not |

Each should be ported to the runner as a small integration test (one synthetic subject, one chunk, one band); the flag goes when all three are done.

## Slow tests left as they are

Their assertions depend on size (false-alarm rates, event counts, statistics over many frames), so shrinking them is not a change of constants:
test_spatial_events_planted, test_spatial_threshold (about a minute each), test_psd_pipeline and cli_import (labelled `slow`). Over 5 s in an older local
record, to review the same way: test_spatial_analytic, test_vortex_cores_pipeline, test_scale_step, test_burst_extent, test_global_bursts.

## Bindings

| file | decision | reason |
|---|---|---|
| bindings/scripts/_smoke-matlab-manifold.m | REMOVE | duplicates the MEX parity tests (bindings/mex/test) and asserts retired behaviour (functions that are now wired refused as not wired) |
| bindings/scripts/_smoke-time-varying.mjs | REMOVE | a one-off smoke test (its own header says so) of two generator bindings, loading the addon file directly rather than through `bindings/node/index.mjs`; never run by CI |
| bindings/python/tests (test_filter, test_io, test_manifold, test_pca, test_welch) | KEEP | binding calls against independent references (scipy, numpy, MNE, nibabel, pandas), names and error classes |
| bindings/mex/test (first_slice, test_*.m, the smoke, check_symbols.sh) | KEEP | the MATLAB face: calls, shapes, errors, parity with the other faces, the symbol check |
| bindings/node/test, bindings/wasm/test, bindings/scripts/_smoke-*.mjs (the rest) | KEEP | one smoke per entry family of the Node and wasm faces, and the cross-binding parity |
| bindings/wasm-io/test | KEEP | wasm client parity with the native command and the Studio |
| bindings/mex/test/explore_cortex.m, viz_cortex_demo.m | KEEP (not tests) | demos the `nxr.viz` package is documented against; not run by CI |

## KEEP (unchanged)

Guards, reference goldens and unit tests of public functions; integration tests that check the wiring.

**unit** (122): meeg_test_forward, meeg_test_inverse, meeg_test_sources, test_analytic, test_array_io, test_band_moments, test_bst, test_cancellation, test_catalogue, test_chart, test_chunk_loop, test_chunk_separator, test_codec, test_codec_pipeline, test_compact, test_compute_types, test_connection_laplacian, test_connectome, test_connectome_gate, test_connectome_partition, test_connectome_registration, test_covariant_differential, test_cross_spectrum, test_crud, test_default_subject, test_dirac_face_operator, test_dirac_operator, test_dtype, test_eigen, test_eigen_complete, test_eigen_consistency, test_eigenspectrum, test_export_bids, test_export_tsv, test_extent, test_face_gradient, test_factorization, test_fibers, test_field_generators, test_field_registry, test_format52, test_formats_byte_source, test_formats_recording, test_formats_trx, test_formats_zip, test_frame_tiling, test_gauge_workflow, test_geometry_bundle, test_geometry_central_extras, test_graph, test_graph_agnostic, test_graph_eigen, test_graph_frames, test_graph_moves, test_graph_types, test_groups, test_hodge_dirac, test_import_bids, test_import_cifti, test_import_fif_objects, test_import_surfaces, test_intervals, test_intrinsic_delaunay, test_intrinsic_forms, test_intrinsic_surface, test_io_eigen, test_lag, test_mass_variants, test_measurement_query, test_measures, test_mesh_bundles, test_migrate, test_migrate_measurement, test_monogenic, test_node_traits, test_normalize, test_operator_dec, test_operator_eigensolve, test_operator_registry, test_operators_facet, test_opticalflow_methods, test_parallel_allocation, test_passthrough_accessors, test_patterns, test_progress, test_quantity, test_query_measure, test_read_existing, test_records, test_regions, test_scale, test_scalespace, test_scaling, test_schema, test_schema51, test_schema53, test_schema54, test_small_file_store, test_sparse, test_spectral_scaling, test_speedspectrum, test_sphere_registration, test_spin_dirac, test_split_groups, test_statistic, test_store, test_subject_filter, test_subject_package, test_surface_arrays, test_synthetic, test_task, test_task_dataset, test_template_import, test_templates, test_tensor, test_transport, test_value_types, test_vector_covariant_operators, test_visualization_primitives, test_wave, test_writer, test_wt_at

**golden** (73): bst_v73_tess_etc_vs_v7, bst_v73_utf8_vs_v7, meeg_test_forward_parity, meeg_test_inverse_parity, store_identity, store_identity_v50_compacts, store_identity_v50_layout, test_burst_stages, test_cf_parity, test_connectome_golden, test_constant_q, test_cores, test_covariance, test_crossings, test_crossspectrum_bins, test_export_fif, test_export_gifti, test_fft, test_filter, test_filterbank, test_form_columns, test_formats_bids, test_formats_brainvision, test_formats_cifti, test_formats_ctf, test_formats_edf, test_formats_eeglab, test_formats_fif, test_formats_fif_objects, test_formats_gifti, test_formats_kit, test_formats_nifti, test_formats_stc, test_formats_tsv, test_forms, test_frames, test_glue, test_glued_form, test_group_tree, test_hat, test_histogram, test_hodge_weights, test_import_nifti, test_import_recording, test_integrals, test_jointbank, test_ladder, test_magnetic, test_measurement, test_montage, test_morlet, test_opticalflow, test_opticalflow_ambient, test_opticalflow_connection, test_pca, test_phase, test_psd_golden, test_pyramid, test_quadrature, test_risefall, test_rollup, test_rollup_parity, test_signal_design, test_signal_grid, test_source_psd, test_spacetime_kernels, test_stats, test_surrogate, test_taps, test_time_tiles, test_timebank, test_tracks, test_welch

**integration** (53): cli_aliases, cli_generated, cli_help, cli_helptext, cli_import (slow), cli_json, cli_output, cli_packages, cli_pipelines, cli_range, cli_refused, cli_table, cli_verbs, test_analytic_frames_step, test_apply_kernel_form, test_apply_kernel_levels, test_burst_extent, test_connectome_definition, test_connectome_nodes, test_crossspectrum, test_dataset_with_template, test_definitions_load, test_definitions_typed, test_face, test_frames_pipeline, test_global_bursts, test_hodge_power, test_import_in_place, test_measure_pipeline, test_montage_create, test_montage_nodes, test_package_pipelines, test_phasegradient, test_pipelines, test_pointspread, test_psd_pipeline (slow), test_quadrature_pipeline, test_runner_outputs, test_runner_plan, test_scale_step, test_select_makers, test_source_tensor, test_spatial_analytic, test_spatial_band, test_spatial_bands, test_spatial_events_pipeline, test_spatial_events_planted (slow), test_spatial_hat, test_spatial_threshold (slow), test_speedspectrum_pipeline, test_vortex_cores_pipeline, test_welch_halo, test_wishart_null

**guard** (13): dispatch_is_the_only_call_path, formats_layering, formats_layering_check_catches, layering_nodes_without_io, layering_nodes_without_io_catches, layering_recordings_read_by_chunk_service, layering_sql_only_in_io, layering_sql_only_in_io_catches, layering_toolboxes_catches, layering_toolboxes_follow_the_dag, records_rows_current, test_names_across_layers, test_names_across_layers_catches
