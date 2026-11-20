// GENERATED from cli/command_table.cpp by tools/command_table.mjs (node tools/command_table.mjs): do not edit.
import type { ArrayInput, Result } from './index.js';
export interface CreateDatasetOptions {
  atlas_finest?: string | number;
  dataset?: string | number;
  datastore?: string | number;
  default?: string | number;
  lc_scales?: string | number;
  modes?: string | number;
  name?: string | number;
  source_path?: string | number;
  source_protocol?: string | number;
  source_tool?: string | number;
  store_format?: string | number;
  templates?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListDatasetsOptions {
  datastore: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ShowDatasetOptions {
  dataset: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface DeleteDatasetOptions {
  datastore: string | number;
  name: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface RecoverDatasetOptions {
  dataset: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface CompactDatasetOptions {
  dataset: string | number;
  dry_run?: boolean;
  no_shard?: boolean;
  no_small_files?: boolean;
  subject?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface CreateRecordingOptions {
  acquisition?: string | number;
  amplitude?: string | number;
  channels?: string | number;
  duration?: string | number;
  frequency?: string[] | string;
  name?: string | number;
  noise?: string | number;
  package: string | number;
  seed?: string | number;
  sfreq?: string | number;
  synthetic?: boolean;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface CreatePackageOptions {
  home_dataset: string | number;
  kind?: string | number;
  name: string | number;
  package: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface CreateIndexOptions {
  dataset: string | number;
  force?: boolean;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface RegisterPackageOptions {
  dataset: string | number;
  package?: string[] | string;
  registered_by: string | number;
  replace?: boolean;
  scan?: boolean;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListMembersOptions {
  dataset: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportParticipantsOptions {
  dataset: string | number;
  sidecar?: string | number;
  tsv: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ExportParticipantsOptions {
  dataset: string | number;
  sidecar?: string | number;
  tsv: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface SplitDatasetOptions {
  dataset: string | number;
  dry_run?: boolean;
  registered_by: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface MigrateMeasurementsOptions {
  dataset: string | number;
  dry_run?: boolean;
  remove_statistics?: boolean;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface JoinDatasetOptions {
  dataset: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface VerifyPackageOptions {
  package: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ShowPackageOptions {
  package: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ReleaseTemplateOptions {
  into: string | number;
  name: string | number;
  package: string | number;
  released_by: string | number;
  template_version: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ReleaseRedistributableOptions {
  into: string | number;
  package: string | number;
  released_by: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface CreateTemplateOptions {
  accept_unlisted_sources?: boolean;
  cortex_mask?: boolean;
  hcp_end_surface?: string | number;
  hcp_points?: string | number;
  hcp_trk?: string | number;
  home_dataset?: string | number;
  max_end_distance?: string | number;
  name?: string | number;
  package: string | number;
  parcellation?: string[] | string;
  recipe: string | number;
  scalp?: boolean;
  sources: string | number;
  surface?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface CreateRegistrationOptions {
  attribution?: string | number;
  from: string | number;
  from_sphere: string | number;
  home_dataset?: string | number;
  licence?: string | number;
  method?: string | number;
  name: string | number;
  neighbours?: string | number;
  one_direction?: boolean;
  package: string | number;
  redistributable?: string | number;
  source?: string | number;
  templates?: string[] | string;
  to: string | number;
  to_sphere: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListTemplatesOptions {
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface CreateSubjectOptions {
  dataset: string | number;
  kind?: string | number;
  name: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListSubjectsOptions {
  dataset: string | number;
  subjects?: string | number;
  where?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ShowSubjectOptions {
  dataset: string | number;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface DeleteSubjectOptions {
  dataset: string | number;
  node?: string | number;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface AddSurfaceOptions {
  /** the positional argument (`nxr-neuro add surface FILE`) */
  file?: string;
  dataset: string | number;
  faces?: ArrayInput | string;
  name?: string | number;
  no_primary?: boolean;
  primary?: boolean;
  scale?: string | number;
  subject: string | number;
  templates?: string[] | string;
  vertices?: ArrayInput | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListRecordingsOptions {
  dataset: string | number;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListKernelsOptions {
  dataset: string | number;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListResultsOptions {
  dataset: string | number;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListPipelinesOptions {
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportListOptions {
  /** the positional argument (`nxr-neuro import list PROTOCOL`) */
  protocol: string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportScanOptions {
  /** the positional argument (`nxr-neuro import scan PROTOCOL`) */
  protocol: string;
  out?: string | number;
  prefer?: string | number;
  summary?: boolean;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportSubjectOptions {
  /** the positional argument (`nxr-neuro import subject PROTOCOL`) */
  protocol: string;
  dataset: string | number;
  resample?: string | number;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportTemplateOptions {
  /** the positional argument (`nxr-neuro import template PROTOCOL`) */
  protocol: string;
  dataset: string | number;
  surface?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportGroupOptions {
  /** the positional argument (`nxr-neuro import group PROTOCOL`) */
  protocol: string;
  dataset: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportTemplatesOptions {
  /** the positional argument (`nxr-neuro import templates TEMPLATES_DIR`) */
  templates_dir: string;
  dataset: string | number;
  subject?: string | number;
  surface?: string | number;
  trk?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportSurfaceOptions {
  bst_root?: string | number;
  dataset: string | number;
  file?: string[] | string;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportMriOptions {
  anat: string | number;
  dataset: string | number;
  only?: string[] | string;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportBidsOptions {
  anonymize?: boolean;
  dataset: string | number;
  from: string | number;
  in_place?: boolean;
  no_participants?: boolean;
  registered_by?: string | number;
  subject?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportFifOptions {
  acquisition?: string | number;
  anonymize?: boolean;
  attribution?: string | number;
  channels?: string | number;
  from: string | number;
  kind?: string | number;
  licence?: string | number;
  name?: string | number;
  package: string | number;
  recording?: string | number;
  source_root?: string | number;
  surface?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportNiftiOptions {
  anonymize?: boolean;
  attribution?: string | number;
  field_kind?: string | number;
  from: string | number;
  kind?: string | number;
  label_names?: string | number;
  licence?: string | number;
  name?: string | number;
  package: string | number;
  parcellation?: string | number;
  source_root?: string | number;
  unit?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportFreesurferOptions {
  annotations?: string[] | string;
  anonymize?: boolean;
  attribution?: string | number;
  field_kind?: string | number;
  from: string | number;
  licence?: string | number;
  name?: string | number;
  orig?: string | number;
  overlays?: string[] | string;
  package: string | number;
  source_root?: string | number;
  surfaces?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportGiftiOptions {
  anonymize?: boolean;
  attribution?: string | number;
  field_kind?: string | number;
  from: string | number;
  kind?: string | number;
  licence?: string | number;
  name?: string | number;
  package: string | number;
  parcellation?: string | number;
  source_root?: string | number;
  surface?: string | number;
  unit?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportCiftiOptions {
  anonymize?: boolean;
  attribution?: string | number;
  field_kind?: string | number;
  from: string | number;
  left?: string | number;
  licence?: string | number;
  name?: string | number;
  package: string | number;
  parcellation?: string | number;
  right?: string | number;
  source_root?: string | number;
  unit?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportTransformOptions {
  channels?: string | number;
  from: string | number;
  name?: string | number;
  package: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportRecordingOptions {
  acquisition?: string | number;
  anonymize?: boolean;
  attribution?: string | number;
  format?: string | number;
  from: string | number;
  in_place?: boolean;
  licence?: string | number;
  montage?: string | number;
  name?: string | number;
  package: string | number;
  source_root?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ExportGiftiOptions {
  all?: boolean;
  field?: string | number;
  from?: string | number;
  hemispheres?: boolean;
  kind?: string | number;
  labelling?: string | number;
  out: string | number;
  package: string | number;
  surface_unit?: string | number;
  to?: string | number;
  with_surface?: boolean;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ExportTsvOptions {
  field?: string | number;
  measures?: string | number;
  out: string | number;
  package: string | number;
  partition?: string | number;
  statistic?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ExportFifOptions {
  all?: boolean;
  covariance?: string | number;
  field?: string | number;
  forward?: string | number;
  from?: string | number;
  inverse?: string | number;
  out: string | number;
  package: string | number;
  to?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ExportBidsDerivativesOptions {
  all?: boolean;
  from?: string | number;
  measurements?: string | number;
  out: string | number;
  package: string | number;
  results: string | number;
  run?: string | number;
  session?: string | number;
  space?: string | number;
  stc?: boolean;
  task?: string | number;
  to?: string | number;
  with_surface?: boolean;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportConditionOptions {
  /** the positional argument (`nxr-neuro import condition PROTOCOL`) */
  protocol: string;
  all_surfaces?: boolean;
  channel_file?: string | number;
  condition: string | number;
  dataset: string | number;
  extra_surface?: string[] | string;
  include_imported?: boolean;
  no_gain?: boolean;
  no_raw?: boolean;
  recording?: string | number;
  resample?: string | number;
  subject: string | number;
  surface?: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportInverseOptions {
  acquisition: string | number;
  bst_root: string | number;
  channel_file?: string | number;
  channels_id?: string | number;
  dataset: string | number;
  file: string | number;
  forward_id?: string | number;
  name?: string | number;
  subject: string | number;
  surface_id?: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportMapOptions {
  acquisition?: string | number;
  bst_root: string | number;
  dataset: string | number;
  file: string[] | string;
  provenance?: string | number;
  replace?: boolean;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportResampleOptions {
  dataset: string | number;
  name?: string | number;
  recording: string | number;
  subject: string | number;
  templates?: string[] | string;
  to: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportBadsegmentsOptions {
  dataset: string | number;
  from: string | number;
  replace?: boolean;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ImportEventsOptions {
  dataset: string | number;
  from: string | number;
  replace?: boolean;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ReadDatasetOptions {
  /** the positional argument (`nxr-neuro read dataset FOLDER`) */
  folder: string;
  array?: string[] | string;
  out?: string | number;
  region?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ReadCrossspectrumOptions {
  dataset: string | number;
  name: string | number;
  pair?: string | number;
  reading?: string | number;
  seed?: string | number;
  subject: string | number;
  templates?: string[] | string;
  weights?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface QueryMeasurementsOptions {
  band?: string | number;
  dataset: string | number;
  factor?: string | number;
  field?: string | number;
  limit?: string | number;
  max_value?: string | number;
  measure: string | number;
  min_value?: string | number;
  order?: string | number;
  selection: string | number;
  spatial_scale?: string | number;
  subject: string | number;
  subjects?: string | number;
  templates?: string[] | string;
  tile?: string | number;
  where?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ExportMeasurementsOptions {
  dataset: string | number;
  exclude?: string | number;
  fields: string | number;
  keep_all?: boolean;
  labelling: string | number;
  layout?: string | number;
  out: string | number;
  recipe?: string | number;
  scaling?: string | number;
  statistics?: string | number;
  subjects?: string | number;
  table: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface CompileOptions {
  /** the positional argument (`nxr-neuro compile  DEFINITION`) */
  definition: string;
  dataset?: string | number;
  inputs?: string | number;
  set?: string[] | string;
  signatures?: string | number;
  subject?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ReadFieldOptions {
  all?: boolean;
  dataset: string | number;
  from?: string | number;
  name: string | number;
  out?: string | number;
  projectors?: string | number;
  rows?: string | number;
  shape_only?: boolean;
  subject: string | number;
  templates?: string[] | string;
  to?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ReadStatisticOptions {
  dataset: string | number;
  name: string | number;
  out?: string | number;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ReadOperatorOptions {
  dataset: string | number;
  name: string | number;
  out?: string | number;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ReadSelectionOptions {
  dataset: string | number;
  field?: string | number;
  measure?: string | number;
  name: string | number;
  out?: string | number;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ReadManifoldOptions {
  dataset: string | number;
  family?: string | number;
  name: string | number;
  out?: string | number;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface AddFieldOptions {
  by_operator?: string | number;
  cell?: string | number;
  contribution?: string[] | string;
  dataset: string | number;
  derived_from?: string | number;
  description?: string | number;
  kind?: string | number;
  manifold: string | number;
  name: string | number;
  producer?: Record<string, unknown> | unknown[] | string;
  replace?: boolean;
  subject: string | number;
  templates?: string[] | string;
  time?: string | number;
  unit?: string | number;
  value_type?: string | number;
  values: ArrayInput | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface AddSelectionOptions {
  dataset: string | number;
  description?: string | number;
  entries?: string | number;
  indices?: ArrayInput | string;
  manifold: string | number;
  members?: ArrayInput | string;
  name: string | number;
  of_field?: string | number;
  producer?: Record<string, unknown> | unknown[] | string;
  replace?: boolean;
  spans?: ArrayInput | string;
  subject: string | number;
  templates?: string[] | string;
  type: string | number;
  weights?: ArrayInput | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface AddStatisticOptions {
  axes: string[] | string;
  dataset: string | number;
  description?: string | number;
  measure: Record<string, ArrayInput | string> | string[];
  method: string | number;
  name: string | number;
  of_field?: string | number;
  partition?: string | number;
  replace?: boolean;
  subject: string | number;
  templates?: string[] | string;
  unit?: string | number;
  value_type?: string | number;
  values: ArrayInput | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface AddMeasurementsOptions {
  dataset: string | number;
  field?: string | number;
  measure: Record<string, ArrayInput | string> | string[];
  replace?: boolean;
  selection: string | number;
  subject: string | number;
  templates?: string[] | string;
  unit?: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface AddOperatorOptions {
  data?: ArrayInput | string;
  dataset: string | number;
  description?: string | number;
  from_manifold: string | number;
  from_selection?: string | number;
  indices?: ArrayInput | string;
  indptr?: ArrayInput | string;
  kind?: string | number;
  layout?: string | number;
  method?: string | number;
  name: string | number;
  params?: Record<string, unknown> | unknown[] | string;
  replace?: boolean;
  subject: string | number;
  templates?: string[] | string;
  to_manifold: string | number;
  values?: ArrayInput | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface AddPointsOptions {
  coordinate_system?: string | number;
  dataset: string | number;
  description?: string | number;
  name: string | number;
  positions: ArrayInput | string;
  replace?: boolean;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface AddLineOptions {
  coordinates?: ArrayInput | string;
  dataset: string | number;
  dimension?: string | number;
  kinds?: ArrayInput | string;
  n?: string | number;
  name: string | number;
  origin?: string | number;
  replace?: boolean;
  spacing?: string | number;
  subject: string | number;
  templates?: string[] | string;
  unit: string | number;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListManifoldsOptions {
  after?: string | number;
  columns?: string | number;
  dataset: string | number;
  kind?: string | number;
  limit?: string | number;
  name?: string | number;
  on?: string | number;
  subject?: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListFieldsOptions {
  after?: string | number;
  columns?: string | number;
  dataset: string | number;
  kind?: string | number;
  limit?: string | number;
  name?: string | number;
  on?: string | number;
  subject?: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListSelectionsOptions {
  after?: string | number;
  columns?: string | number;
  dataset: string | number;
  kind?: string | number;
  limit?: string | number;
  name?: string | number;
  on?: string | number;
  subject?: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListOperatorsOptions {
  after?: string | number;
  columns?: string | number;
  dataset: string | number;
  kind?: string | number;
  limit?: string | number;
  name?: string | number;
  on?: string | number;
  subject?: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListStatisticsOptions {
  after?: string | number;
  columns?: string | number;
  dataset: string | number;
  kind?: string | number;
  limit?: string | number;
  name?: string | number;
  on?: string | number;
  subject?: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListDictionariesOptions {
  after?: string | number;
  columns?: string | number;
  dataset: string | number;
  kind?: string | number;
  limit?: string | number;
  name?: string | number;
  subject?: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListSpacesOptions {
  after?: string | number;
  columns?: string | number;
  dataset: string | number;
  kind?: string | number;
  limit?: string | number;
  name?: string | number;
  subject?: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListAcquisitionsOptions {
  after?: string | number;
  columns?: string | number;
  dataset: string | number;
  limit?: string | number;
  name?: string | number;
  subject?: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ListJobsOptions {
  after?: string | number;
  columns?: string | number;
  dataset: string | number;
  kind?: string | number;
  limit?: string | number;
  subject?: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface ReadSubjectOptions {
  classes?: string | number;
  dataset: string | number;
  related?: boolean;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface AddProductOptions {
  dataset: string | number;
  factors: string | number;
  name: string | number;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface AddDictionaryOptions {
  dataset: string | number;
  description?: string | number;
  entries: Record<string, unknown> | unknown[] | string;
  name: string | number;
  scope: string | number;
  subject?: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface EditManifoldOptions {
  dataset: string | number;
  description?: string | number;
  label?: string | number;
  name: string | number;
  primary?: string | number;
  subject: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}
export interface EditDictionaryOptions {
  colour?: string | number;
  dataset: string | number;
  description?: string | number;
  entries?: Record<string, unknown> | unknown[] | string;
  entry?: string | number;
  label?: string | number;
  name: string | number;
  name_of?: string | number;
  params?: Record<string, unknown> | unknown[] | string;
  replace?: boolean;
  subject?: string | number;
  templates?: string[] | string;
  /** receives each progress line the command prints, as an object */
  progress?: (line: Record<string, unknown>) => void;
}

export interface Verbs {
  create: {
    dataset: (options?: CreateDatasetOptions) => Promise<Result>;
    recording: (options: CreateRecordingOptions) => Promise<Result>;
    package: (options: CreatePackageOptions) => Promise<Result>;
    index: (options: CreateIndexOptions) => Promise<Result>;
    template: (options: CreateTemplateOptions) => Promise<Result>;
    registration: (options: CreateRegistrationOptions) => Promise<Result>;
    subject: (options: CreateSubjectOptions) => Promise<Result>;
  };
  list: {
    datasets: (options: ListDatasetsOptions) => Promise<Result>;
    members: (options: ListMembersOptions) => Promise<Result>;
    templates: (options?: ListTemplatesOptions) => Promise<Result>;
    subjects: (options: ListSubjectsOptions) => Promise<Result>;
    recordings: (options: ListRecordingsOptions) => Promise<Result>;
    kernels: (options: ListKernelsOptions) => Promise<Result>;
    results: (options: ListResultsOptions) => Promise<Result>;
    pipelines: (options?: { progress?: (line: Record<string, unknown>) => void }) => Promise<Result>;
    manifolds: (options: ListManifoldsOptions) => Promise<Result>;
    fields: (options: ListFieldsOptions) => Promise<Result>;
    selections: (options: ListSelectionsOptions) => Promise<Result>;
    operators: (options: ListOperatorsOptions) => Promise<Result>;
    statistics: (options: ListStatisticsOptions) => Promise<Result>;
    dictionaries: (options: ListDictionariesOptions) => Promise<Result>;
    spaces: (options: ListSpacesOptions) => Promise<Result>;
    acquisitions: (options: ListAcquisitionsOptions) => Promise<Result>;
    jobs: (options: ListJobsOptions) => Promise<Result>;
  };
  show: {
    dataset: (options: ShowDatasetOptions) => Promise<Result>;
    package: (options: ShowPackageOptions) => Promise<Result>;
    subject: (options: ShowSubjectOptions) => Promise<Result>;
  };
  delete: {
    dataset: (options: DeleteDatasetOptions) => Promise<Result>;
    subject: (options: DeleteSubjectOptions) => Promise<Result>;
  };
  recover: {
    dataset: (options: RecoverDatasetOptions) => Promise<Result>;
  };
  compact: {
    dataset: (options: CompactDatasetOptions) => Promise<Result>;
  };
  register: {
    package: (options: RegisterPackageOptions) => Promise<Result>;
  };
  import: {
    participants: (options: ImportParticipantsOptions) => Promise<Result>;
    list: { (protocol: string, options?: Omit<ImportListOptions, 'protocol'>): Promise<Result>; (options: ImportListOptions): Promise<Result> };
    scan: { (protocol: string, options?: Omit<ImportScanOptions, 'protocol'>): Promise<Result>; (options: ImportScanOptions): Promise<Result> };
    subject: { (protocol: string, options: Omit<ImportSubjectOptions, 'protocol'>): Promise<Result>; (options: ImportSubjectOptions): Promise<Result> };
    template: { (protocol: string, options: Omit<ImportTemplateOptions, 'protocol'>): Promise<Result>; (options: ImportTemplateOptions): Promise<Result> };
    group: { (protocol: string, options: Omit<ImportGroupOptions, 'protocol'>): Promise<Result>; (options: ImportGroupOptions): Promise<Result> };
    templates: { (templates_dir: string, options: Omit<ImportTemplatesOptions, 'templates_dir'>): Promise<Result>; (options: ImportTemplatesOptions): Promise<Result> };
    surface: (options: ImportSurfaceOptions) => Promise<Result>;
    mri: (options: ImportMriOptions) => Promise<Result>;
    bids: (options: ImportBidsOptions) => Promise<Result>;
    fif: (options: ImportFifOptions) => Promise<Result>;
    nifti: (options: ImportNiftiOptions) => Promise<Result>;
    freesurfer: (options: ImportFreesurferOptions) => Promise<Result>;
    gifti: (options: ImportGiftiOptions) => Promise<Result>;
    cifti: (options: ImportCiftiOptions) => Promise<Result>;
    transform: (options: ImportTransformOptions) => Promise<Result>;
    recording: (options: ImportRecordingOptions) => Promise<Result>;
    condition: { (protocol: string, options: Omit<ImportConditionOptions, 'protocol'>): Promise<Result>; (options: ImportConditionOptions): Promise<Result> };
    inverse: (options: ImportInverseOptions) => Promise<Result>;
    map: (options: ImportMapOptions) => Promise<Result>;
    resample: (options: ImportResampleOptions) => Promise<Result>;
    badsegments: (options: ImportBadsegmentsOptions) => Promise<Result>;
    events: (options: ImportEventsOptions) => Promise<Result>;
  };
  export: {
    participants: (options: ExportParticipantsOptions) => Promise<Result>;
    gifti: (options: ExportGiftiOptions) => Promise<Result>;
    tsv: (options: ExportTsvOptions) => Promise<Result>;
    fif: (options: ExportFifOptions) => Promise<Result>;
    bids_derivatives: (options: ExportBidsDerivativesOptions) => Promise<Result>;
    measurements: (options: ExportMeasurementsOptions) => Promise<Result>;
  };
  split: {
    dataset: (options: SplitDatasetOptions) => Promise<Result>;
  };
  migrate: {
    measurements: (options: MigrateMeasurementsOptions) => Promise<Result>;
  };
  join: {
    dataset: (options: JoinDatasetOptions) => Promise<Result>;
  };
  verify: {
    package: (options: VerifyPackageOptions) => Promise<Result>;
  };
  release: {
    template: (options: ReleaseTemplateOptions) => Promise<Result>;
    redistributable: (options: ReleaseRedistributableOptions) => Promise<Result>;
  };
  add: {
    surface: { (file: string, options: Omit<AddSurfaceOptions, 'file'>): Promise<Result>; (options: AddSurfaceOptions): Promise<Result> };
    field: (options: AddFieldOptions) => Promise<Result>;
    selection: (options: AddSelectionOptions) => Promise<Result>;
    statistic: (options: AddStatisticOptions) => Promise<Result>;
    measurements: (options: AddMeasurementsOptions) => Promise<Result>;
    operator: (options: AddOperatorOptions) => Promise<Result>;
    points: (options: AddPointsOptions) => Promise<Result>;
    line: (options: AddLineOptions) => Promise<Result>;
    product: (options: AddProductOptions) => Promise<Result>;
    dictionary: (options: AddDictionaryOptions) => Promise<Result>;
  };
  read: {
    dataset: { (folder: string, options?: Omit<ReadDatasetOptions, 'folder'>): Promise<Result>; (options: ReadDatasetOptions): Promise<Result> };
    crossspectrum: (options: ReadCrossspectrumOptions) => Promise<Result>;
    field: (options: ReadFieldOptions) => Promise<Result>;
    statistic: (options: ReadStatisticOptions) => Promise<Result>;
    operator: (options: ReadOperatorOptions) => Promise<Result>;
    selection: (options: ReadSelectionOptions) => Promise<Result>;
    manifold: (options: ReadManifoldOptions) => Promise<Result>;
    subject: (options: ReadSubjectOptions) => Promise<Result>;
  };
  query: {
    measurements: (options: QueryMeasurementsOptions) => Promise<Result>;
  };
  edit: {
    manifold: (options: EditManifoldOptions) => Promise<Result>;
    dictionary: (options: EditDictionaryOptions) => Promise<Result>;
  };
  compile: { (definition: string, options?: Omit<CompileOptions, 'definition'>): Promise<Result>; (options: CompileOptions): Promise<Result> };
}
