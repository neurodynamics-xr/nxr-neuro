// The command table: every verb, every command (its old spelling, topic, summary, options) and every option's meaning, default and unit, declared once.
// cli/registry.cpp reads it to build the command line, `--help`, the man pages, the shell completion and the old-spelling aliases; `nxr-neuro __check-options`
// compares each command's options with what its parser accepts, so the table cannot drift from the code.
//
// Syntax (one piece per command; blank lines and `#` lines are ignored):
//   verb NAME<TAB>sentence                     a verb of the grammar
//   --option<TAB>VALUE<TAB>meaning<TAB>default<TAB>unit     an option's help (VALUE empty: a flag); the same text everywhere the option is used
//   @ VERB NOUN                                a command (`@ run`: the verb is the command)
//     old: the old spelling's words           group: the topic of a computation       summary: one sentence
//     pos: the positional argument's name     span: yes  a computation over a recording (needs --from/--to or --all); seconds: no  only --all
//     opts: the parser's options; `+` several values, `*` required     example: a full example (else made from the required options)
#include <vector>

std::vector<const char*> command_table_text() {
  return {
R"TBL(verb run	Run a pipeline on a subject: a built-in (psd, crossspectrum, phasegradient, measure, ...), a built-in definition, or a definition file
verb compute	Compute a node or a graph on arrays or files (reserved: not available yet; pipelines are run with `run`)
verb list	List datasets, subjects, recordings, kernels, results or pipelines
verb show	Show a dataset or a subject
verb read	Read stored arrays and derived readings
verb query	Query stored measurements
verb export	Export stored results: measurements as a table, cortical maps as GIFTI
verb import	Import data from Brainstorm into a dataset
verb create	Create a dataset or a subject
verb delete	Delete a dataset or a subject
verb compact	Compact a dataset
verb recover	Recover a dataset after an interrupted write
verb reduce	Reduce stored results across subjects
verb add	Add a surface to a subject
verb register	Register subject packages in a dataset's catalogue
verb split	Split a one-database dataset into subject packages with a catalogue
verb join	Join subject packages back into a one-database dataset
verb release	Release a template or registration as a frozen, versioned package
verb verify	Verify a subject package against the hashes it records
verb migrate	Migrate a dataset's stored layout in place
)TBL",
R"TBL(--atlas-finest	QUANTITY	The finest tile of the default subject's group tree (the dataset's atlas leaves), as a quantity: parts per hemisphere ("parts 256"), a tile diameter ("14 mm") or a tile area ("area 150 mm2"); needs --templates.	parts 256 (2^8 tiles a hemisphere)	parts
--lc-scales	QUANTITY,...	The tile scales, as quantities, at which the Levi-Civita connection between neighbouring tiles is measured (a measurement over a level's product with itself); needs --templates.	parts 16, parts 64	parts
--atlas-finest@create dataset	QUANTITY	The finest tile of the default subject's group tree (the dataset's atlas leaves), as a quantity: parts per hemisphere ("parts 256"), a tile diameter ("14 mm") or a tile area ("area 150 mm2"); needs --templates.	parts 256 (2^8 tiles a hemisphere)	parts
--lc-scales@create dataset	QUANTITY,...	The tile scales, as quantities, at which the Levi-Civita connection between neighbouring tiles is measured (a measurement over a level's product with itself); needs --templates.	parts 16, parts 64	parts
--acquisition	NAME	The Brainstorm acquisition (condition) the kernel or map belongs to.	none	-
--acquisition@import inverse	NAME	The Brainstorm acquisition (condition) the kernel belongs to; it names the session the kernel is stored under.	required	-
--acquisition@import map	NAME	The Brainstorm acquisition (condition) the map belongs to, used to find a map's surface or atlas when its file does not name one.	none	-
--all		Do the time line of every recording in the subject (alternative to --recording and --line)	off	-
--all-surfaces		Import every surface of the subject, not only the ones the condition's head model uses.	off	-
--all-surfaces@import condition		Import every surface of the subject's anatomy, not only the surface of the condition's head model and the extra surfaces.	off	-
--alpha	X	Horn-Schunck smoothness weight of the optical flow; vary it and report the range.	0.01 for the connection regulariser, 1 for ambient	-
--anat	FILE	The anatomical MRI file to import.	none	-
--anat@import mri	FOLDER	The subject's anatomical folder of the source protocol, whose MRI images are imported into the subject.	required	-
--anchor	HZ	Itersine and similar time filterbanks: the anchor frequency the member spacing is built from.	1.0	Hz
--array	SUBJECT PATH	Read one array: the subject and the array's path in the dataset (prints its bytes, or writes them with --out).	none	-
--array@read dataset	SUBJECT PATH	Read one array: the subject's name and the array's path inside the subject; prints its shape, data type and checksum. Without --array every array of the dataset is listed.	none (list all arrays)	-
--band	LO,HI	The frequency band in hertz (select_time_grid rule frames: its centre sets the frame length; surrogate_lagfree_pairs: the band of the stored windows; measure query: INDEX picks one band by its index, a whole number).	none	Hz
--band@query measurements	QUANTITY	The temporal band to read, as a quantity in Hz ("8-12 Hz"); the band set's member it falls in is read, and its snapped span is printed.	none (all bands, or the only one)	Hz
--bands	NAME:LO-HI,...	The frequency bands to compute, each a name and its low and high edge (the mean over the bins from LO to HI, inclusive); a band SET for `run measure` is a quantity (select_bands rule frequency).	Brainstorm's bands (delta 2-4, theta 5-7, alpha 8-12, beta 15-29, gamma1 30-59, gamma2 60-90)	Hz
--bandwidth	HZ	Notch width: the 3 dB bandwidth of each notch	1	Hz
--batch	N	Number of surrogate draws made at a time before the early-stopping test (Besag-Clifford).	20	draws
--batch@run surrogate_lagfree_vortex_compare	N	Draws are made in batches of this size	20	draws
--batch@run surrogate_lagfree_spatial_events_compare	N	Draws made between checks of the stopping rule.	20	draws
--berg-dipoles	J	Number of equivalent dipoles in the Berg-Scherg approximation of the three-shell EEG sphere.	3	-
--berg-starts	N	Number of simplex starts when fitting the three-dipole Berg-Scherg approximation of the EEG sphere model; the best fit is kept	12	-
--berg-terms	N	Number of series terms fitted when approximating the multi-shell EEG model by the Berg-Scherg parameters.	200	terms
--block-frames	N	How many frames the steps after a frame tiling process at once; limits memory only, the result does not change (0 = the whole chunk's frames).	256	frames
--block-frames@run	N	How many frames the steps after a frame tiling process at once; limits memory only, the result does not change (0 = the whole chunk's frames).	256	frames
--block-samples	N	Samples processed at a time, which sets memory use only and does not change the result (run: width for the steps after the last filter, 0 = whole chunk)	run: 1024	samples
--block-samples@run	N	Width in samples of the blocks that flow through the steps after the last filter; limits memory only, the result does not change (0 = the whole chunk).	1024	samples
--block-seconds	SECONDS	Length of the blocks whose own mean is removed before the covariance is formed (noisecov: noise window; inverse: when computing the noise covariance from a recording).	one block (a single mean for the whole window)	s
--bst-root	PATH	Root folder of the Brainstorm protocol, used to resolve the surface files named by --file.	none (empty)	-
--bst-root@import inverse	PATH	Root folder of the Brainstorm protocol, used to resolve the kernel's channel, surface and head-model links.	required	-
--bst-root@import map	DIR	The Brainstorm protocol folder the map files are read from, with the anatomy folders the maps' surfaces and atlases are found in.	required	-
--bst-root@import surface	PATH	Root folder of the Brainstorm protocol, used to find the surface's registration and atlas files next to each --file.	none (empty)	-
--bursts	NAME	Name of a stored set of burst time spans (surrogate_lagfree_pairs: required, each burst is tested; cellquadrature: optional, also computed per burst)	none	-
--centre	HZ	Centre frequency of the band from which the frame length is derived (the coarsest tower level whose frame is no longer than 1/(centre x frames-per-cycle)).	none	Hz
--centres	HZ,...	Centre frequencies of the filterbank members (Morlet, Morse, Amor and Bump families); for Morlet, empty means the single member at --freq.	none	Hz
--channel-file	FILE	The channel file to use instead of the condition's own.	the condition's channel file	-
--channel-file@import condition	FILE	Channel file (in the condition folder) to use for the condition's recordings instead of the one the condition names.	none (the condition's own channel file)	-
--channel-file@import inverse	FILE	The channel file to take the kernel's channels from instead of the one found for the kernel.	found from the kernel's condition	-
--channels	NAME	Name of the sensor channel set the head model is computed for	the subject's only channel set with a type labelling	-
--channels-id	ID	The id of the channels the kernel was made for.	found from the kernel file	-
--channels-id@import inverse	ID	Id of the stored channels the kernel was made for (a row of the subject).	found from the subject's session channels	-
--channels-of	KERNEL	Use the channels of this stored inverse kernel, in its order, for the cross-spectrum; an alternative to --types.	none (all good channels of the chosen types)	-
--chunks	K|A-B|all	Which part of the recording to read: one chunk K, a range of chunks A-B, or all chunks looped one after another (a chunk is 2^15 samples of every channel). Exclusive with --start/--stop.	0 (one chunk)	chunks
--combine	each|pool|mean	How values are combined before summarising: each row separately, all rows pooled, or the mean over rows per member	each	-
--condition	NAME	The Brainstorm condition to import.	none	-
--condition@import condition	NAME	The condition (acquisition) of the protocol to import; it is stored as a session of the subject.	required	-
)TBL",
R"TBL(--convention	scs|ras	Axes in which the surface's own coordinates are given: scs (anterior +x, superior +z, left +y) or ras (anterior +y, superior +z, left -x).	scs	-
--cross-spectrum	NAME	Name of the stored sensor cross-spectrum (from crossspectrum compute) to read.	none (required)	-
--cycles	C	Cycles of the band's lowest frequency in each window (select_time_grid rule constant_q) or wavelet width as centre frequency / C (speedspectrum)	constant_q: set by its method (about 6.8 at 1200 Hz); speedspectrum: 6	cycles
--datafit	auto|gauge|triangle	Data term of the Horn-Schunck flow: auto picks the regulariser's own (gauge for connection, triangle for ambient); gauge is connection only; triangle uses per-triangle finite elements.	auto	-
--dataset	PATH	The dataset to work on: the path of its folder (a dataset is addressed by its own path; the folder of a one-database dataset, or of a catalogue of subject packages).	required	-
--datastore	DIR	The folder that holds the dataset folders (their parent); the dataset is then DIR/NAME.	none (required)	-
--dataset@create dataset	PATH	The folder of the new dataset (made if absent): it holds the catalogue and, by default, each subject's package.	required (unless --datastore)	-
--datastore@create dataset	DIR	Deprecated: makes the one-database form, the dataset DIR/NAME (needs --name); `split dataset` turns it into a catalogue. Use --dataset.	none	-
--datastore@delete dataset	PATH	The folder that holds the folder of the dataset to delete (its parent).	required	-
--datastore@list datasets	DIR	The folder that holds the dataset folders to list.	required	-
--ddof	N	Delta degrees of freedom: the standard deviation divides by n minus this number (1 matches MATLAB's std).	1	-
--default	NAME	The template resolution used for the dataset's default subject (needs --templates).	fsaverage5	-
--default@create dataset	NAME	The subject of the --templates folder (the template resolution) that becomes the dataset's default subject; needs --templates.	fsaverage5	-
--defects	keep|exclude	keep: use every edge of the mesh in the weighted fit; exclude: give zero weight to folded or very short edges (needs --spatial-band)	keep	-
--depth-channels	all|one-type	Which channels the depth-weighting norm is computed on: all channels, or one channel type (gradiometers if present, else magnetometers, else EEG).	all	-
--depth-exponent	P	Exponent of the depth weighting that counteracts the bias of the minimum-norm solution towards superficial sources	0.5	-
--depth-limit	X	Upper limit of the depth weighting (Brainstorm's WeightLimit): the largest-to-smallest weight ratio is limited to this value squared in variance.	10	-
--depth-limit-on	weight|norm	Where the depth limit is applied: on the weights (Brainstorm's way) or on the inverse norms (MNE-Python's way).	weight	-
--depth-norm	frobenius|spectral	Matrix norm used for each source's gain when depth weighting	frobenius	-
--detrend	none|constant|linear	Trend removed from each Welch segment before windowing.	constant (the Brainstorm option set)	-
--divisor	samples-minus-blocks|samples|samples-minus-one	What the covariance sum is divided by: samples minus the number of blocks, the samples, or samples minus one.	samples-minus-blocks	-
--draws	N	Number of lag-free Wishart draws a burst (surrogate_lagfree_pairs)	1000	draws
--dry-run		Report what would change (compact: files and folders before and after; run: inputs found and arrows checked) without changing or computing anything.	off	-
--dry-run@compact dataset		Report the files and folders before and after compacting, without changing anything.	off	-
--dry-run@run		Print the plan (inputs found, every connection checked, every output's place) without reading data or writing anything.	off	-
--edge-lag	frame|frame-normal|frame-tangent|axial|ambient	How each edge's phase lag is read: in the surface's own frame (frame), its normal or tangent part alone, modulo pi (axial), or from the x y z components as they are (ambient).	frame	-
--eeg-conductivities	S1,S2,S3	Conductivities of the EEG sphere shells, innermost to outermost (brain, skull, scalp)	0.33,0.0042,0.33	S/m
--eeg-radii	R1,R2,R3	Relative radii of the EEG head model's shells, innermost to outermost (brain, skull, scalp).	0.88,0.93,1.0	relative
--eeg-series		Use the multi-shell series itself for the EEG sphere instead of the Berg-Scherg equivalent dipoles (slower, exact up to truncation).	off	-
--eeg-series-terms	N	Number of terms of the multi-shell series when it is used instead of the three-dipole fit (with --eeg-series)	200	terms
--eeg-surface	NAME	Name of the surface the EEG sphere is fitted to.	the surface called scalp (or head)	-
--enter-sd	SD	A patch is kept when a tile reaches the median plus this many robust standard deviations of the map's tile means.	1.5	SD
--estimator	welch|analytic	How the sensor cross-spectrum is estimated: Welch segments, or the band's analytic signal summed per time tile (a series of windows)	welch	-
--events	NAME	Name of the spatial-events pipeline whose stored statistics are compared with lag-free surrogate draws.	spatial_events	-
--events@run surrogate_lagfree_spatial_events_compare	NAME	The pipeline that named the data's stored events statistics (<NAME>.home.<recording>, <NAME>.summary.<recording>).	spatial_events	-
--exceedances	N	Besag-Clifford stopping count: a tile or vertex is decided once it has this many exceedances of the data.	10	exceedances
--exceedances@run surrogate_lagfree_vortex_compare	H	A tile is decided once this many surrogate draws reach or exceed its observed value (Besag-Clifford)	10	draws
--exceedances@run surrogate_lagfree_spatial_events_compare	N	The number of exceedances (draws at least as extreme as the data) at which a vertex's p-value is settled; sets the sequential stopping rule.	10	draws
--exclude	NAMES	Comma-separated member names left out of the exported table.	unassigned	-
--exclude@export measurements	NAMES	Comma-separated members (by dictionary name) left out of the table	unassigned	-
--exit-sd	SD	A patch extends over tiles at or above the median plus this many robust standard deviations	1.0	robust SD
)TBL",
R"TBL(--extra-surface	FILE	An extra surface file to import with the condition (repeatable).	none	-
--extra-surface@import condition	FILE	An additional surface file of the subject's anatomy to import with the condition; repeat the option for several.	none	-
--family	NAME	Family of the time filter bank: itersine, meyer, morlet, morse, amor or bump (see the command's description for the accepted names).	none (required)	-
--factor	NAME	measure query: the factor of a product selection the roll-up runs along (its name); none: the only grid factor.	none	-
--field	NAME	Name of a stored field to read (measure query: restrict the query to this field; phasegradient: the phase 1-form to build, only gradient exists).	measure query: none; phasegradient: gradient	-
--factor@query measurements	NAME	For a product selection: the factor the roll-up runs along, by its name; with none, the only grid factor.	the only grid factor	-
--field@query measurements	NAME	The stored field the measurement was made of, by name or id.	the only field measured	-
--fields	PATTERN	Regular expression matched against the whole name of each stored field to measure (export: the first capture group names the condition)	none (required)	-
--fields@export measurements	REGEX	Regular expression over Field names selecting the Fields to export; its first capture group names the condition (e.g. the band)	required	-
--file	FILE	A Brainstorm anatomy file (tess_*.mat) to convert; repeat or give several.	none (required)	-
--file@import inverse	FILE	The Brainstorm inverse kernel file (results_*KERNEL*.mat) to import; one kernel per call, never overwritten.	required	-
--file@import map	FILE	A map file to import: a source map (results_*.mat, one value per vertex) or a region map (matrix_*.mat, one value per atlas region); repeatable.	required	-
--file@import surface	FILE	A Brainstorm cortex surface file (tess_*.mat) to convert and store; repeat or give several.	required	-
--fit	inner|mesh|modes	How the timing map is fitted to the edge lags: by inner products with the modes, by a weighted fit on the whole surface complex, or by a weighted fit within the exact modes.	inner	-
--flow-tolerance	X	Relative residual at which the conjugate-gradient flow solve stops (pcg and per-scale methods).	1e-8	-
--fold-edges	keep|axial|all	How edges across a fold of the surface are read: keep all as they are, axial (lag read modulo pi where the ends' normals are opposed), or all (every edge modulo pi).	keep	-
--forward	NAME	Name of the stored head model (forward operator) to use.	the subject's only one	-
--forward-id	ID	The id of the head model the kernel was made with.	found from the kernel file	-
--forward-id@import inverse	ID	Id of the stored head model the kernel was made with.	found from the kernel's HeadModelFile link	-
--fpass	HZ|LO,HI	Passband edge: one frequency for lowpass or highpass, LO,HI for bandpass	none (required)	Hz
--fraction	X	Fraction of the peak power at the reference level above which the connected region counts as the source's patch.	0.5	-
--frames-per-cycle	N	Frames per cycle of the band's centre frequency, which sets the frame level (the coarsest whose frame is at or under 1/(centre x N))	36	frames
--freq	HZ	Frequency whose nearest filter-bank member is analysed.	none (required)	Hz
--frequencies	F,...	Centre frequencies to remove (notch), e.g. a line frequency and its harmonics.	none (give this or --line)	Hz
--from	FILE	The file to read the bad segments or events from (import badsegments, import events).	none	-
--from@import badsegments	FILE	JSON file listing the bad segments (start and stop times) to store as the subject's bad-segment selection.	required	-
--from@import events	FOLDER	The source protocol's data folder to read the events from.	required	-
--fstop	LO,HI	Stopband edges of a bandstop filter	none (required)	Hz
--gain	X	Peak in-band response of each filter-bank member (2 makes the coefficient's magnitude the physical amplitude).	none (required)	-
--gamma	G	Morse family: the gamma parameter of the wavelet.	3	-
--gradient		Also measure the gradient of each scalar field (area-weighted face gradients projected on the vertex tangent plane)	off	-
--graph-family	mexhat|heat|itersine|logitersine	Family of the spatial (graph) filter bank over the eigenvalue.	none (required)	-
--graph-filters	N	Number of members of the graph (eigenvalue) filterbank; 0 means derived from --graph-voices.	0	filters
--graph-voices	N	Members per octave of the spatial (graph) filterbank	3	members/octave
--half-width	OCT	Meyer family: half-width of each transition between members, in octaves.	0.3333 (1/3)	octaves
--halo-seconds	SECONDS	Context read on each side of a chunk so the band filter's edge effects fall inside it	8	s
--harmonics	N	With --line, the number of harmonics of the line frequency to notch.	every multiple below the Nyquist frequency	-
)TBL",
R"TBL(--include-imported		Also import conditions that were themselves imported.	off	-
--include-imported@import condition		When the condition has a raw recording, also import its data files that were themselves imported (by default only the raw recording is).	off	-
--input	amplitude|signed	What the flow is computed on: the source amplitude |J| of the kernel applied to the complex band coefficients, or the signed band signal itself (fixed-orientation kernels only).	amplitude	-
--joint	NAME	Name of the stored product selection (space x time cells) to use	none (required: a run with several candidates is refused naming them)	-
--joint-types		Whiten the channel types of a modality (e.g. magnetometers with gradiometers) together, keeping their cross-covariance, instead of one block per type.	off	-
--keep-all		Keep every member in the table, including those excluded by default.	off	-
--keep-all@export measurements		Keep every member in the table, including the default excluded one (clears --exclude)	off	-
--keep-bad-segments		Keep the recording's bad segments in the computation instead of leaving them out	off	-
--kernel	NAME	Name of the stored inverse kernel (the map from channels to sources) to apply.	the subject's only kernel	-
--kernels	SPEC,...	Which kernels' pyramids are summed: all, constrained, free, latest, a method, stamp, name or glob (kinds combine with AND, tokens of one kind with OR).	all	-
--kind	subject|template|group	Kind of subject to create	subject	-
--kind@create subject	subject|template|group	Kind of the new subject: an ordinary subject, a template subject or a group subject.	subject	-
--labelling	NAME	The partition (a stored Selection with one code per vertex, e.g. a parcellation or a tile level) whose members are measured.	none (required)	-
--labelling@export measurements	NAME	The partition (labelling) whose members are the table's columns, named the same in every subject	required	-
--layout	split|wide	Table layout: split gives a table per condition and statistic with a column per member; wide gives one table with a column per condition, member and statistic.	split	-
--layout@export measurements	split|wide	split writes one table per condition and statistic with a column per member; wide writes one table with a column per condition, member and statistic	split	-
--levels	L,...	Spatial tile levels of the scale space the blur is measured over.	2,3,4,5,6,7,8	levels
--limit	N	Maximum number of rows returned	none	rows
--limit@query measurements	N	The most rows to return.	none (all rows)	rows
--limits	LO-HI	Frequency limits of the filter bank for the itersine and meyer families.	see the command's description	Hz
--list	V,V,...	run: list the available pipeline definitions and their steps (flag); pointspread: native vertex indices to use as the source sample instead of --vertices	run: off; pointspread: none	-
--map	NAME	The stored cortical map (a scalar per vertex of a surface) in which patches are sought.	none (required)	-
--max-area	M2	Largest patch area kept; larger patches are dropped.	none	m^2
--max-condition	C	Canonical frames: a vertex whose push-forward Jacobian has a condition number above C is treated as singular	50	-
--max-draws	N	Largest number of surrogate draws.	100	draws
--max-draws@run surrogate_lagfree_vortex_compare	N	Most surrogate draws made; a rerun with a larger value continues where the last stopped	100	draws
--max-draws@run surrogate_lagfree_spatial_events_compare	N	The most draws made.	100	draws
--max-iterations	N	Iteration limit of the iterative flow solver (pcg and perscale methods).	500	iterations
--max-value	X	Keep only rows with a value at or below X	none	measure units
--max-value@query measurements	X	Keep only rows whose rolled-up value is at most this.	none	measure's unit
--measure	NAME	The measure to summarise or read (e.g. mean, max, power_sum).	none (required)	-
--measure@query measurements	NAME	The measure to read, e.g. a stored measure of a band's cells or one derived from its ingredients.	required	-
--measures	NAME,...	Which measures to compute per member: mean, max, min, mean_square, abs_max, sum.	mean, max, min, mean_square, abs_max	-
--meg	overlapping|single|none	MEG head model: per-sensor overlapping spheres, one single sphere, or no MEG channels	overlapping	-
--meg-surface	NAME	Name of the surface the MEG spheres are fitted to.	the surface called inner skull	-
--members-own		The measure is the members' own (area, number of elements or tiles); --of then names its parent.	off	-
--method	NAME	select_time_grid constant_q: hertz (octave edges on powers of two in Hz) or cycles (fixed cycle count)	hertz	-
)TBL",
R"TBL(--min-area	M2	Smallest patch kept.	none	m^2
--min-crossings	N	Minimum number of crossings a tile needs (in both runs for repeat) to be tested.	10	crossings
--min-crossings@run surrogate_lagfree_vortex_compare	N	A tile counts only with at least this many crossings	10	crossings
--min-draws	N	Smallest number of surrogate draws made before early stopping may decide a tile or vertex	20	draws
--min-draws@run surrogate_lagfree_vortex_compare	N	Fewest surrogate draws made	20	draws
--min-draws@run surrogate_lagfree_spatial_events_compare	N	The fewest draws made before stopping, even if every vertex has reached --exceedances.	20	draws
--min-link	N	Two events of different scale pairs count as one event when at least this many pairs of their cells nest in space and overlap in time.	1	pairs
--min-projection	P	Reliability limit of the anatomical frame: the anterior axis must project on the tangent plane with at least this length.	0.34	-
--min-value	X	Keep only rows with a value at or above X	none	measure units
--min-value@query measurements	X	Keep only rows whose rolled-up value is at least this.	none	measure's unit
--mne		Use MNE-Python's conventions for the kernel: joint types, ridge over the rank, PCA whitener, depth exponent 0.8 by the spectral norm.	off	-
--mne-eeg-layers		Use MNE-Python's EEG sphere layers and conductivities instead of the default three-shell layers.	off	-
--modes	K	Number of Hodge-Dirac eigenmodes (run eigenmodes, dataset create: how many to solve; pointspread, cellquadrature, patches: size of the package when it must be computed, 0 = standard; speedspectrum, hodgepower: use the first K of the stored package)	eigen solve: 3000; dataset create: none; speedspectrum, hodgepower: all; pointspread, cellquadrature, patches: 0	modes
--modes@create dataset	K	Solve the default subject's pial Hodge-Dirac eigenmode basis with about this many modes and store it; needs --templates.	none (no basis is solved)	modes
--name	NAME	Name of what is stored: the dataset, subject or surface, or the stem of the stored result (filters: <recording>_<verb>).	derived from the inputs	-
--name@add surface	NAME	Name the surface is stored under for the subject.	the mesh file's name without its extension, made safe as a node name	-
--name@create dataset	NAME	Name of the new dataset.	the folder's name (required with --datastore)	-
--name@create subject	NAME	Name of the new subject in the dataset.	required	-
--name@delete dataset	NAME	Name of the dataset's folder to delete, inside the folder given by --datastore	required	-
--name@import inverse	NAME	Name of the stored kernel.	Brainstorm's file stem, session-qualified if already taken	-
--name@import resample	NAME	Name of the resampled recording that is created	<recording>_<rate>Hz (e.g. meg_250Hz)	-
--name@read crossspectrum	NAME	The stored band-resolved cross-spectrum to read, named <stem>_<band>_bins (made by run crossspectrum --resolve bins).	required	-
--no-average-reference		Do not re-reference the EEG gain rows and covariance to the average	off	-
--no-bad-segments		Keep the recording's marked bad segments in the estimate instead of leaving out the segments that touch them.	off (bad segments are left out)	-
--no-depth		Turn depth weighting off.	off (depth weighting on)	-
--no-detect-diagonal		Do not whiten a diagonal noise covariance by its diagonal alone	off	-
--no-eeg		Do not compute the EEG channels of the head model.	off (EEG is computed)	-
--no-gain		Do not store the head model's gain matrix.	off (the gain is stored)	-
--no-gain@import condition		Do not store the head model's gain matrix.	off (the gain is stored)	-
--no-primary		Do not mark the added mesh as the subject's primary surface	off	-
--no-primary@add surface		Store the surface without marking it as the subject's primary surface.	off (the surface is made primary)	-
--no-project-electrodes		Do not move the electrodes radially onto the fitted EEG sphere.	off (electrodes are projected)	-
--no-raw		Do not import the raw recording, only the condition's other data.	off	-
--no-raw@import condition		Do not import the condition's raw recording; import only its other data files.	off	-
--no-remove-mean		Do not remove the mean (DC offset) of each block before computing or filtering.	off (mean removed)	-
--no-shard		Do not group chunks into shards when compacting	off	-
--no-shard@compact dataset		Do not group chunks into shards; only the key separator changes, so an array over 4 MiB stays unsharded.	off (arrays are sharded)	-
)TBL",
R"TBL(--no-small-files		Stop at the sharded layout; do not also move each subject's small files into its small-files database.	off	-
--no-small-files@compact dataset		Stop at the sharded layout; do not also move each subject's small keys into its small-files database.	off (small files are moved)	-
--node	PATH	Delete only this stored node of the subject, by store path, instead of the whole subject.	none (whole subject)	-
--node@delete subject	NAME	Delete only this node (manifold, field, operator or selection) of the subject instead of the whole subject	none (the whole subject is deleted)	-
--noise-cov	NAME	Name of a stored noise covariance to use (alternative to --recording)	none	-
--normalisation	none|relative	Normalisation of the band power maps: none, or relative (each map divided by the source's total power over all bins up to Nyquist).	none	-
--of	NAME	The stored field, or band of a map statistic, that was measured	none (required)	-
--only	NAME	Import only this named anatomical image (repeatable).	all images	-
--only@import mri	NAME	Import only this named anatomical image; repeat the option for several.	all images	-
--order	N	Filters: Butterworth prototype order (the filter has 2N poles). Spatial band commands: degree of the Chebyshev series (0 chooses the smallest of 40, 80, ... 5120 within tolerance). measure query: sort order asc or desc of the rows.	filters 4; spatial 0 (automatic); measure query see the command's description	-
--order@query measurements	asc|desc	Order the rows by value, ascending or descending.	by code	-
--out	PATH	Output file (convert scan: the scan result written as a file) or output folder (measure export: where the table is written, made if absent).	none (measure export: required)	-
--out@export measurements	PATH	Folder the tables, the provenance file and the dictionary are written to (created if absent)	required	-
--out@import scan	PATH	File to write the scanned protocol view to as JSON (writing it prints a one-line summary instead of the whole view).	none (the view is printed)	-
--out@read dataset	FILE	Write the bytes of the array (or of its --region) read with --array to this file, as little-endian values of its data type.	none	-
--over	NAME	Name of the partition (stored selection) measured over: statistic: the partition whose members were measured; surrogate_lagfree_vortex_compare, repeat: the tile-level selection of the stored crossings; measure compute: the time tiling	statistic, measure compute: none (required); null, repeat: the one the recordings share	-
--over@run surrogate_lagfree_vortex_compare	NAME	The tile level Selection the crossings were measured over	the one the recordings have	-
--overlap	FRACTION	Fraction by which successive Welch segments overlap.	0.5 (the Brainstorm option set)	fraction
--oversample	X	Oversampling of each filterbank member's own sample rate.	2.0	-
--padding	mean|zeros	How the filter's edge transient is padded: with the mean of the signal's transient or with zeros.	mean	-
--pair	A,B	One pair of channels, by channel index (row A, column B), to read from the band-resolved cross-spectrum.	none	-
--pair@read crossspectrum	A,B	One pair of channels to read, by recording channel index (row A, column B); an alternative to --seed.	none (one of --pair or --seed)	-
--pairs	K	Number of strongest quadrature pairs stored per band	3	pairs
--periodic		Use the periodic (DFT-even) form of the taper instead of the symmetric form.	off for the default and brainstorm sets (symmetric); on for scipy	-
--phase-of	current|stream	Signal whose phase is taken: the band's current vector field, or its scalar stream function	current	-
--preconditioner	ldlt|ic	Preconditioner of the flow's conjugate-gradient solve: ldlt (exact factor, built once) or ic (incomplete Cholesky).	ldlt	-
--prefer	protocol.mat|walk	Which reader to prefer for the Brainstorm protocol: its protocol.mat file or a walk of the folders.	automatic	-
--prefer@import scan	protocol.mat|walk	Which reader to use for the protocol: its protocol.mat file or a walk of the folders.	automatic (protocol.mat when present)	-
--primary		Mark the added mesh as the subject's primary surface	on	-
--primary@add surface		Mark the added surface as the subject's primary surface (the one other commands use when no --surface is named).	on	-
--provenance	FILE	A JSON file of provenance to store with the imported maps.	none	-
--provenance@import map	FILE	A JSON file of provenance to store with each imported map.	none	-
--psd	NAME	The recording's stored Welch spectrum, whose shape gives the band's effective bandwidth.	none (required)	-
--rank-tolerance	single|double	Precision used to decide the rank of the noise covariance: single or double precision.	single	-
--reading	coherency|imag|phase|delay|psi	Quantity read from the stored bin-by-bin cross-spectrum (positive delay and psi: the row channel leads)	coherency	-
--reading@read crossspectrum	coherency|imag|phase|delay|psi	What to derive per pair from the bins: coherency, its imaginary part, phase, the phase-slope delay (ms) or the phase-slope index; positive delay and psi mean the row channel leads.	coherency	-
)TBL",
R"TBL(--recipe	NAME	Name of the recipe (the recorded set of choices) the measurements are made or read by.	measure compute: partition-<weight>; measure export: any	-
--recipe@export measurements	NAME	Use only measurements made by this recipe (name or name@version); a subject with rows from another recipe is refused	none (any recipe)	-
--recording	NAME	Name of the stored recording to read.	none (required)	-
--recording-a	NAME	First recording, whose stored tile crossings are used (repeat: one of the two compared)	none (required)	-
--recording-a@run surrogate_lagfree_vortex_compare	NAME	A recording whose stored tile crossings are compared with lag-free surrogate draws	required	-
--recording-b	NAME	Second recording whose crossings are compared with the first (run-to-run repeat).	none	-
--recording-b@run surrogate_lagfree_vortex_compare	NAME	A second recording, to test the run-to-run repeat as well (draws made in step)	none	-
--recording@run surrogate_lagfree_spatial_events_compare	NAME	The recording whose stored spatial events are compared with lag-free surrogate draws.	required	-
--recording@import condition	FILE	Import only this data file of the condition (e.g. data_0raw_x.mat) instead of every data file the condition has.	none (all the condition's recordings)	-
--recording@import resample	NAME	The recording to resample	required	-
--reference-level	L	Spatial tile level whose power the blur patch is cut from.	6	level
--region	a:b,c:d	Read only this region of the array: a start:stop range per dimension.	the whole array	samples
--region@read dataset	a:b,c:d	Read only this region of the array: one start:stop index range per dimension, stop excluded; used with --array.	the whole array	samples
--regularisation	ridge|diagonal|none	Regularisation of the noise covariance when whitening	ridge	-
--regulariser	connection|ambient	Smoothness term of the flow: connection (connection Laplacian on the signpost gauge) or ambient (Dirichlet energy of the ambient 3-vector field, Lefevre and Baillet).	connection	-
--replace		Replace an existing result of the same name instead of refusing; the old one is moved aside, not deleted.	off	-
--resample	HZ	Resample the converted recordings to this sampling rate	none (keep the original rate)	Hz
--resample@import condition	HZ	Resample the condition's recordings to this sampling rate (polyphase resampling) when they are imported.	none (the original rate is kept)	Hz
--resample@import subject	HZ	Resample every recording of the subject to this sampling rate (polyphase resampling) when it is imported.	none (the original rate is kept)	Hz
--resolution	HZ	Frequency resolution on which the filter bank is designed.	0.5	Hz
--resolve	band|bins	Whether to also keep each band's cross-spectrum bin by bin (bins, Welch estimator only) or only the band mean (band).	band	-
--response	peak|mean	Which response of an event is compared across scale pairs: the largest standardized cell amplitude or their mean	peak	-
--ridge-fraction	X	Ridge added in whitening, as a fraction of the mean eigenvalue of the noise covariance block (Brainstorm's noise regularisation, MNE-Python's reg).	0.1	fraction
--ridge-over-rank		Take the ridge from the trace divided by the rank (MNE-Python) instead of by the number of channels (Brainstorm).	off	-
--ripple	R	Passband and stopband deviation of the Kaiser-window FIR filter, as a linear value (0.001 is 60 dB attenuation)	0.001	-
--scale	X	Factor applied to the mesh coordinates to bring them to metres.	1	-
--scale-family	FAMILY	perscale flow method: the graph filterbank whose band-pass members give one flow per spatial scale (mexhat, itersine or logitersine).	none (required for perscale)	-
--scale-filters	N	Number of members of the per-scale graph filterbank (method perscale); 0 derives it from the voices	0	members
--scale-voices	N	Per-scale flow method: voices per octave of the graph filter bank.	3	voices/octave
--scale@add surface	X	Factor multiplied into every vertex coordinate of the mesh file to bring it to metres.	1	-
--scaling	density|spectrum|chain	How the spectrum is scaled: power per Hz, power per bin, or the chain scaling (measure export: stored|density|power|amplitude|amplitude-density|db|db-density, applied to the stored values).	density (measure export: stored)	-
--scaling@export measurements	NAME	Scaling applied to the values at read: stored, density, power, amplitude, amplitude-density, db or db-density (needs the Field's recorded spectral scaling)	stored	-
--seed	N	crossspectrum read: the channel index to read against every channel (alternative to --pair); surrogate_lagfree_pairs: seed of the random draws	surrogate_lagfree_pairs: 0; crossspectrum read: none	-
--seed@read crossspectrum	A	Read channel A (a recording channel index) against every channel of the statistic; an alternative to --pair.	none (one of --pair or --seed)	-
--segment-level	L	Welch segment of 2^L samples (an alternative to the segment length in seconds or samples).	none (the segment length option is used)	level
)TBL",
R"TBL(--segment-samples	N	Welch segment length in samples; wins over --segment-seconds.	derived from the 4 s segment	samples
--segment-seconds	SECONDS	Length of each Welch segment (under the tiles setting it is snapped to a power-of-two number of samples)	4	s
--selection	NAME	The stored product selection (space x time cells) whose measurement is read.	none (required)	-
--selection@query measurements	NAME	The selection (a grid, a product or a set of events) whose members' measurements are read, by name or id; it must be complete.	required	-
--sensor-weight-exponent	P	Exponent of the distance weighting used to fit each sensor's overlapping sphere.	4	-
--set	NAME=VALUE	Set a pipeline parameter, NAME=VALUE with VALUE read as YAML (8, [15, 29], dSPM); repeatable.	none	-
--set@run surrogate_lagfree_vortex_compare	NAME=VALUE	Parameter of every null draw (repeatable); {recording} stands for each recording's name	none	-
--set@run surrogate_lagfree_spatial_events_compare	NAME=VALUE	A parameter of every surrogate draw's spatial-events run (levels, threshold, band, ...); repeatable.	none (the data's parameters)	-
--set@run	NAME=VALUE	Override a parameter of the pipeline definition; the value is read as YAML (8, [15, 29], dSPM). Repeatable.	none (the definition's own values)	-
--shape	morlet|meyer|power_partition	Band filter shape: morlet (Gaussian centred on the band, FWHM equal to its width) or meyer (surrogate_lagfree_pairs).	morlet	-
--shift	absolute|relative	How the eigenvalue solve is shifted: at one absolute shift (reference method) or by a scale-aware relative ladder.	absolute	-
--shift@run eigenmodes	absolute|relative	Shift of the shift-invert solve: absolute (one solve per ladder at --sigma, the reference method) or relative (a ladder scaled to the spectrum).	absolute	-
--side	auto|left|right	Hemisphere of a single-piece surface	auto	-
--sides	analytic|two-sided	Whether the time bank is one-sided (analytic, positive frequencies) or two-sided.	none (required)	-
--sign	positive|negative|magnitude	Which tile means form patches: high values, low values, or absolute value	positive	-
--signal	normal|ambient	Signal whose phase gradient is taken: ambient (the full current, its x y z components) or normal (the component along the surface normal).	ambient	-
--snr	S	Assumed amplitude signal-to-noise ratio of the minimum-norm kernel	3	-
--snr-normalisation	all-channels|whitener-rank	Whether the signal-to-noise ratio is normalised over all channels or over the rank of the whitener.	all-channels	-
--source	NAME	The pipeline whose stored cells and events are read.	cortical_event_pairs	-
--source-path	PATH	Records the path of the source data in the dataset's provenance	none	-
--source-path@create dataset	PATH	Records the path of the source data the dataset was made from, in the dataset's provenance.	none	-
--source-protocol	NAME	Record the name of the source protocol the dataset was made from.	none	-
--source-protocol@create dataset	NAME	Records the name of the source protocol the dataset was made from, in the dataset's provenance.	none	-
--source-surface	NAME	Surface whose vertices are the sources of the head model.	the primary (cortex) surface	-
--source-tool	NAME	Records the tool that produced the source data in the dataset's provenance	none	-
--source-tool@create dataset	NAME	Records the tool that produced the source data, in the dataset's provenance.	none	-
--spatial-scale	QUANTITY	The spatial scale to read, as a quantity in a chart of the surface's space ("20 mm", "wavelength 40 mm", "parts 16", "area 400 mm2"); coarser than the stored level it is rolled up at read, through the grid's closed-form hierarchy and the measure's merge rule; the level is printed with the snapped quantity.	the stored level	mm
--spatial-scale@query measurements	QUANTITY	The spatial scale to read, as a quantity in a chart of the surface's space ("20 mm", "wavelength 40 mm", "parts 16", "area 400 mm2"); coarser than the stored level it is rolled up at read, through the grid's closed-form hierarchy and the measure's merge rule; the level is printed with the snapped quantity.	the stored level	mm
--space-levels	L,...	Spatial tile levels of the surface to select over (all combinations with --time-levels).	none (required)	levels
--spatial-band@query measurements	N	The spatial band (eigenmode level) of the cells to read.	none	-
--spatial-shape	NAME	Shape of the spatial scale-space filter at each level (mexhat is the Mexican hat).	mexhat	-
--speed-bins	N	Number of log-spaced phase-speed bins of the speed spectrum.	40	bins
--speed-max	C	Largest phase speed of the speed histogram bins	from the spectrum (the largest 2 pi f / mu over modes and bins)	length/s
--speed-min	X	Smallest phase speed of the log-spaced speed bins.	0 (from the spectrum)	length/s
--sphere-cost	geometric|algebraic	Cost the head sphere is fitted with: distance to the sphere (geometric) or the linear closed-form fit (algebraic).	geometric	-
)TBL",
R"TBL(--stage	NAME|INDEX	Stage of the stored crossings to use, by name (burst, onset, within, offset) or by index	0 (the first stage, burst)	-
--stage@run surrogate_lagfree_vortex_compare	NAME|INDEX	The stage of the stored crossings to test, by name (burst, onset, within, offset) or index	0	-
--star1-floor	X	Floor on the cotangent weights of the Hodge star; values at or below 0 mean no floor.	none	-
--star1-floor@run eigenmodes	F	Raise the edge cotan weights (the 1-form Hodge star) that fall below F up to F; unset means no floor.	none	-
--start	SEC	Start of the span read, in seconds on the recording's clock; give together with --stop. Exclusive with --chunks.	none (one chunk)	s
--start-s	SECONDS	Time of the recording's first sample as seconds from local midnight, anchoring the time-tile levels	the anchor is the recording	s
--statistics	LIST	Comma-separated statistics written per subject in the exported table (e.g. mean,sd).	see the command's description	-
--statistics@export measurements	NAMES	Comma-separated statistics to export: mean, sd, variance, rms, mean_square, max, min, abs_max, sum, area, n_elements	mean	-
--stop	SEC	End of the span read, in seconds on the recording's clock (exclusive); give together with --start.	none (one chunk)	s
--store-format	51|52	Array storage layout of the new dataset (52 keeps small keys in a table)	52	-
--store-format@create dataset	51|52	Array storage layout of the new dataset: 52 keeps each subject's small keys in a table, 51 keeps only the folder tree.	52	-
--subject	NAME	The subject of the dataset to work on.	required	-
--subjects	A,B,...	Subjects to include, by name.	every subject of the dataset (run roll_up, group scope: required)	-
--subjects@export measurements	NAMES	Comma-separated subjects to include	every subject of the dataset	-
--summaries	LIST	Comma-separated summaries across the members, such as median, mad, percentile:75, mean, sd, peak, kurtosis, skewness, min, max	statistic: none (required); patches: count,sum,median,mean,sd,percentile:25,percentile:75,percentile:90	-
--summary		Print a one-line summary of the scan.	off	-
--summary@import scan		Print a one-line summary (kind, root, counts) instead of the whole view.	off	-
--surface	NAME	Surface to work on, by name.	none (required when the subject has several surfaces: the run names them and is refused)	-
--surface-id	ID	The id of the surface the kernel was made on.	found from the kernel file	-
--surface-id@import inverse	ID	Id of the stored cortex surface the kernel was made on.	found from the kernel's SurfaceFile link	-
--surface@import condition	FILE	The subject's cortex surface file (tess_*.mat) the condition is imported on, instead of the one its head model names.	none (the condition's own surface)	-
--surface@import template	NAME	Convert only this surface (a tess_*.mat file of the template anatomy); repeat for several.	every surface of the template's default anatomy	-
--surface@import templates	NAME	The surface the template maps are placed on	the subject's surface with each map's vertex count	-
--symmetric		Use the symmetric form of the taper window instead of the periodic one	off (the window form follows --set)	-
--table	NAME	Name of the exported table.	none (required)	-
--table@export measurements	NAME	Name of the table; it names the data_table and the output files	required	-
--template	SUBJECT	Template subject whose anatomical frame is carried over through the registered spheres.	none (the surface's own axes)	-
--template-surface	NAME	Surface of the template subject whose frame is carried over	the template's primary surface	-
--templates	DIR	A folder of released templates to look in (repeatable), before NXR_TEMPLATE_PATH and the user setting template_path; read only when a template's arrays are used.	none (the environment variable and the user setting only)	-
--templates@create dataset	DIR	A FreeSurfer subjects folder; the dataset is created together with its default subject (surfaces, parcellations, frames, group tree) read from it.	none (no default subject is written)	-
--threads	N	Workers for the kernel products; the result is identical for every count (0 = the machine's performance cores, 1 = serial).	0	threads
--through	NAME	Name of a stored 2-row operator (a rotation plane) over the channels: compute the two-sided rotary spectrum of its complex signal	none	-
--tile-pairs	on|off	Also compute the blocks between touching tiles (waves crossing from one tile into the next)	on	-
--tiles		Also write the maps' tile summaries per level of the tile ladder.	off	-
--time-bandwidth	B	Morse family: the time-bandwidth product of the wavelet.	60	-
--tile	QUANTITY	The time tile to read, as a duration ("2 s") or a band ("8-12 Hz"); coarser than the stored tile it is rolled up at read; the tile is printed with its snapped length in seconds.	the stored tile	s
--tile@query measurements	QUANTITY	The time tile to read, as a duration ("2 s") or a band ("8-12 Hz"); coarser than the stored tile it is rolled up at read; the tile is printed with its snapped length in seconds.	the stored tile	s
--time-levels	M1,M2,..	Constant-Q time levels (each one's band is a pair's rhythm) crossed with the space levels.	none (required)	levels
--to	HZ	The sampling rate to resample to (import resample).	none	Hz
--to@import resample	HZ	The new sampling rate; the recording is resampled by a polyphase filter (scipy resample_poly)	required	Hz
--tolerance	T	lowpass, notch: IIR look-behind tail left out, relative to the response's L1 norm; pointspread, cellquadrature, patches: largest gain error accepted for the Chebyshev series	filters: 1e-15; pointspread, cellquadrature, patches: 0.01	-
)TBL",
R"TBL(--transform	auto|eigen|chebyshev|exact	How the spatial band is applied: auto (stored eigenmodes; else computed, else Chebyshev announced), eigen, chebyshev (series in the pencil) or exact (dense modes, at most 4096 vertices).	auto	-
--transient-samples	K	Number of samples at each edge treated as the filter transient (instead of the shortest span holding 99 percent of the impulse response's energy).	from the energy rule	samples
--transition	HZ	Width of each transition band of the FIR filter	the reference's rule (Brainstorm)	Hz
--home-dataset	NAME	The name of the dataset a package is made for (provenance, not membership).	none (required)	-
--registered-by	WHO	Who registers: recorded with each catalogue entry.	none (required)	-
--released-by	WHO	Who releases: the program and the releaser's stated name, recorded in the release.	none (required)	-
--scan		Register every subject package directly inside the dataset folder.	off	-
--force		Rebuild even when the index is current.	off	-
--tsv	FILE	A BIDS participants.tsv file (tab-separated, a participant_id column, n/a for missing).	none (required)	-
--sidecar	FILE	The BIDS participants.json sidecar describing the variables.	none	-
--into	DIR	The templates folder a released package is made in.	none (required)	-
--template-version	VERSION	The version string the releaser gives (e.g. 1.0); name@version must be new.	none (required)	-
--where	FILTER	A filter over the participant variables, in the filter language version 1 (== != < <= > >=, in (...), is missing, and, or, not, parentheses).	none	-
--templates@register package	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this command uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@list templates	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this listing uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@show subject	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this command uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@delete subject	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this command uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@add surface	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this command uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@list recordings	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this listing uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@list kernels	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this listing uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@list results	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this listing uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@import subject	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this import uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@import surface	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this import uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@import mri	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this import uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@import condition	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this import uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@import inverse	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this import uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@import map	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this import uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@import resample	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this import uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@import badsegments	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this import uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@import events	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this import uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@read crossspectrum	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this reading uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@query measurements	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this query uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run surrogate_lagfree_spatial_events_compare	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run surrogate_lagfree_vortex_compare	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@register package	DIR	A folder of released templates to look in (repeatable); when given, each package's template must be found there by name, version and hash and its anchor rows must equal the template's.	none (not checked)	-
--templates@list templates	DIR	A folder to look in for released templates (repeatable), before NXR_TEMPLATE_PATH and the user setting template_path.	none (the environment variable and the user setting only)	-
--package@create package	PATH	The folder of the subject package to make (<subject>.nxr.zarr); it must not exist yet.	required	-
--name@create package	NAME	Name of the subject the package holds.	required	-
--home-dataset@create package	NAME	The dataset the package is made for, by its name: provenance, not membership.	required	-
--kind@create package	subject|template|group	Kind of the subject the package holds: an ordinary subject, a template or a group.	subject	-
--package@register package	PATH	A subject package's folder (repeatable); an alternative to --scan.	none	-
--scan@register package		Register every *.nxr.zarr package directly inside the dataset folder; an alternative to --package.	off	-
--registered-by@register package	WHO	Who registers: recorded with each catalogue entry.	required	-
--replace@register package		Another subject id under a name the catalogue holds replaces it; the old package is moved aside, never deleted.	off	-
--where@query measurements	io|compute	Where the query runs: io (the SQL query, for value filters, order, limit and group queries) or compute (in memory, from the finest ingredients and the grid's closed form). A value filter, order or limit runs in io and refuses compute.	compute (io when a value filter, order or limit is given)	-
--where@list subjects	FILTER	List the subjects whose participant variables satisfy this filter (version 1 of the filter language); the dataset must be a catalogue with a participants table.	none (all subjects are listed)	-
--subjects@query measurements	A,B,...	Across these subjects (the same members in each): the ingredients are summed code for code before the roll-up.	the one subject	-
--subjects@list subjects	A,B,...	List these subjects, by name or id (a catalogue dataset); not together with --where.	none (all subjects are listed)	-
--force@create index		Rebuild the index even when it is current.	off	-
--tsv@import participants	FILE	The BIDS participants.tsv to import: tab-separated, a participant_id column matched to subject names, n/a for missing.	required	-
--sidecar@import participants	FILE	The BIDS participants.json sidecar: Description, Units, Levels and Type of each variable.	none (types are inferred)	-
--tsv@export participants	FILE	The file to write the participants table to as BIDS participants.tsv; nothing else is written.	required	-
--sidecar@export participants	FILE	Also write the participants.json sidecar to this file.	none	-
--registered-by@split dataset	WHO	Who runs the split: recorded with each catalogue entry.	required	-
--dry-run@migrate measurements		Report, per subject, the statistics that would become measurements, the products and tilings that would be rewritten, and the conflicts, without changing anything.	off	-
--remove-statistics@migrate measurements		The explicit second step, taken only when asked for explicitly: remove each migrated statistic whose measurement is verified (and its array). Never the default: the old statistics are kept.	off	-
--dry-run@split dataset		Report the subjects and the format without changing anything (not even bringing the dataset to format 52).	off	-
--package@verify package	PATH	The folder of the subject package to verify.	required	-
--package@show package	PATH	The folder of the subject package to show.	required	-
--package@release template	PATH	The folder of the template or registration package to release; it is left as it was.	required	-
--name@release template	NAME	Name the template is released under.	required	-
--template-version@release template	VERSION	Version string of the release (e.g. 1.0); name@version must be new, a released version is frozen.	required	-
--released-by@release template	WHO	Who releases: the program and the releaser's stated name, recorded in the release.	required	-
--into@release template	DIR	The templates folder the frozen copy <name>@<version>.nxr.zarr is made in.	required	-
--package@release redistributable	PATH	The folder of the released full edition to make the redistributable edition from.	required	-
--into@release redistributable	DIR	The templates folder the redistributable edition is made in.	required	-
--released-by@release redistributable	WHO	Who releases: recorded in the release.	required	-
--from-sphere	NAME	The surface of the first template that carries its registered sphere (e.g. cortex_pial_sphere).	none (required)	-
--to-sphere	NAME	The surface of the second template that carries its registered sphere (e.g. cortex_pial_sphere).	none (required)	-
--neighbours	K	Modified Shepard weights: the number of nearest source vertices looked at.	8	vertices
--one-direction		Make only the operator from the first template to the second.	off (both directions are made)	-
--surfaces	NAME...	The surfaces of a FreeSurfer subject to import (surf/lh.<name> and rh.<name>), e.g. white pial.	white	-
--annotations	NAME...	The FreeSurfer annotations to import as labelling Selections (label/lh.<name>.annot and rh.<name>.annot): named explicitly, none is imported unless named.	none	-
--overlays	NAME...	The FreeSurfer per-vertex files to import as Fields (surf/lh.<name> and rh.<name>, e.g. thickness curv sulc).	none	-
--orig	FILE	The volume (orig.mgz) that holds c_ras, the offset from FreeSurfer's surface RAS to scanner RAS.	<subject>/mri/orig.mgz, else the surface file's own volume geometry	-
--left	NAME	The package's surface for the left cortex (CIFTI_STRUCTURE_CORTEX_LEFT); its vertex count must be the file's.	required when the file has the left cortex	-
--right	NAME	The package's surface for the right cortex (CIFTI_STRUCTURE_CORTEX_RIGHT); its vertex count must be the file's.	required when the file has the right cortex	-
--field-kind	TEXT	The kind of the Field written (what its values are).	the command's own	-
--label-names	FILE	A text file of `code name` lines naming the entries of a parcellation.	entries are named by their code	-
--unit	TEXT	The unit of the values written, when known.	none	-
--licence	TEXT	The licence recorded in the origin of every item.	derived from the two spheres' origin records	-
--attribution	TEXT	The attribution recorded in the origin of every item.	derived from the two spheres' origin records	-
--redistributable	yes|no	Whether the registration may be redistributed.	yes only when both spheres' origin records say so	-
--package@create registration	PATH	The folder of the registration package to make (a working copy; `release template` freezes it); it must not exist yet.	required	-
--name@create registration	NAME	Name of the registration, the subject the package holds.	required	-
--from@create registration	NAME@VERSION	The template the operators carry fields FROM, as released (NAME@VERSION or NAME@VERSION:redistributable); found on the template search path.	required	-
--to@create registration	NAME@VERSION	The template the operators carry fields TO, as released; it must differ from --from.	required	-
--from-sphere@create registration	NAME	The surface of the --from template that carries its registered sphere (e.g. cortex_pial_sphere).	required	-
--to-sphere@create registration	NAME	The surface of the --to template that carries its registered sphere (e.g. cortex_pial_sphere).	required	-
--method@create registration	shepard|nearest|barycentric	How a destination vertex takes its value from the source vertices: modified Shepard weights over the nearest ones (the reference's), the nearest vertex, or barycentric coordinates in the source triangle that holds it.	shepard	-
--neighbours@create registration	K	With shepard: the number of nearest source vertices looked at; the K-th sets the radius and takes no weight.	8	vertices
--one-direction@create registration		Make only the operator from --from to --to; by default the operator back is made too.	off (both directions are made)	-
--templates@create registration	DIR	A folder of released templates to look in (repeatable), before NXR_TEMPLATE_PATH and the user setting template_path.	none (the environment variable and the user setting only)	-
--home-dataset@create registration	NAME	The dataset the package is made for, by its name: provenance, not membership.	registrations	-
--source@create registration	TEXT	The source recorded in the origin of every item.	derived: computed from the two templates' spheres	-
--licence@create registration	TEXT	The licence recorded in the origin of every item.	derived from the two spheres' origin records	-
--attribution@create registration	TEXT	The attribution recorded in the origin of every item.	derived from the two spheres' origin records	-
--redistributable@create registration	yes|no	Whether the registration may be redistributed (decides what the redistributable edition keeps).	yes only when both spheres' origin records say so	-
--recipe	fsaverage|icbm152|icbm152_2023b	Which template to build from its source files.	none (required)	-
--sources	DIR	The folder of the downloaded source files the recipe reads.	none (required)	-
--parcellation	NAME	A parcellation to write, named explicitly (repeatable); none is written unless named.	none (no parcellation)	-
--cortex-mask		Also write the cortex mask (the vertices that are not the medial wall).	off	-
--scalp		Also write the scalp and the landmarks nasion, LPA, RPA, for placing the standard electrode montages (not a boundary-element model).	off	-
--surface	NAME	A surface to write (repeatable).	the recipe's own	-
--hcp-trk	FILE	A DSI Studio .trk of streamlines on the HCP-1065 template, to import into the ICBM152 template.	none	-
--hcp-end-surface	NAME	The surface of the package the streamline ends are placed on (a choice of data).	none (required with --hcp-trk)	-
--max-end-distance	MM	An end farther than this from the end surface is left out of the ends operator; every end's distance is stored.	5	mm
--hcp-points	N	Points per streamline after resampling at equal arc length.	64	points
--accept-unlisted-sources		Read a source file the manifest does not list or whose checksum differs; the checksum read is recorded in the origin.	off (refused)	-
--recipe@create template	fsaverage|icbm152|icbm152_2023b	fsaverage: FreeSurfer's fsaverage; icbm152_2023b: Brainstorm's default anatomy ICBM152_2023b; icbm152: its current default (ICBM152 2026). With --hcp-trk, also HCP-1065 streamlines.	required	-
--sources@create template	DIR	The folder of the downloaded source files (for fsaverage a FreeSurfer subject folder holding surf/ and label/; for icbm152 Brainstorm's default-anatomy folder). Each file is checked against the manifest.	required	-
--package@create template	PATH	The folder of the template package to make (a working copy; `release template` freezes it); it must not exist yet.	required	-
--name@create template	NAME	Name of the template, the subject the package holds.	the recipe's template name	-
--surface@create template	NAME	fsaverage: a FreeSurfer surface name (pial, white, inflated; the first is primary); icbm152: a Brainstorm tess_*.mat file name (the first is primary).	pial white inflated (fsaverage); tess_cortex_pial_high.mat (icbm152)	-
--parcellation@create template	NAME	A parcellation to write, named explicitly (repeatable): fsaverage an annotation (aparc, aparc.a2009s); icbm152 an atlas named in the surface file (Desikan-Killiany, Schaefer_100_7net, ...). None is written unless named; an atlas whose terms are not confirmed is written as not redistributable.	none (no parcellation)	-
--cortex-mask@create template		Also write the cortex mask (fsaverage only): the vertices that are not the medial wall.	off	-
--scalp@create template		Also write the scalp (a closed head surface, `head`) and the landmarks nasion, LPA, RPA (`head_landmarks`), for placing the standard electrode montages: ICBM152 tess_head.mat and subjectimage_T1.mat; fsaverage bem/outer_skin.surf and bem/fsaverage-fiducials.fif (MNE-Python's). Not a boundary-element model.	off	-
--hcp-trk@create template	FILE	A DSI Studio .trk of streamlines (icbm152 only); imported by its MNI coordinates with its origin (CC BY-SA 4.0 and the HCP terms).	none	-
--hcp-end-surface@create template	NAME	The surface of the package the streamline ends are placed on (e.g. cortex_pial_high); a choice of data.	required with --hcp-trk	-
--max-end-distance@create template	MM	An end farther than this from the end surface is left out of the ends operator; every end's distance is stored either way.	5	mm
--hcp-points@create template	N	Points per streamline after resampling at equal arc length; the two ends are kept exactly.	64	points
--accept-unlisted-sources@create template		Read a source file the manifest does not list or whose checksum differs; the checksum read is recorded in the origin.	off (refused)	-
--home-dataset@create template	NAME	The dataset the package is made for, by its name: provenance, not membership.	templates	-
--tree	subject|group	The tree the leaves belong to: the surface's own ladder, or the dataset's group tree through the registered sphere (the same leaves in every subject).	subject	-
--trk	FILE	A fibre-tract file (.trk) to import with the templates.	none	-
--trk@import templates	FILE	A tractography streamline file (.trk) whose streamlines and their end points are added to the templates	none	-
--types	TYPE,...	Channel types to use, e.g. "MEG MAG,MEG GRAD,EEG" (only good channels of these types).	MEG, MEG MAG, MEG GRAD, EEG, SEEG, ECOG	-
--vertices	N	Number of vertices, spread evenly over the surface, where unit currents are placed	20	vertices
--weight	none|barycentric|voronoi	How the vertices of each member are weighted (measure compute: none is the plain vertex mean, barycentric or voronoi weight by area); phasegradient: power|coherence, the edge weight of a weighted fit.	measure compute: none, must be given; phasegradient: power	-
--weights	area|none	cellquadrature: weight vertices by their area or equally; crossspectrum read: weight the delay and phase-slope fits by coherence or not	cellquadrature: area; crossspectrum read: coherence	-
--weights@read crossspectrum	coherence|none	For --reading delay: weight each frequency bin of the phase-slope fit by its squared coherency, or count every bin equally.	coherence	-
--whitener-form	symmetric|pca	Form of the noise whitener: symmetric or PCA.	symmetric	-
--window	hann|hamming|blackman|rectangular	Taper applied to each Welch segment.	hamming (default and brainstorm sets); hann (scipy set)	-
--windows	NAME	The stored analytic cross-spectrum windows (named <stem>_<band>_windows) the bursts are read from.	none (required)	-
--acquisition@import recording	NAME	The session of the subject the recording is stored under.	rest	-
--anonymize@import recording		Drop the identifying header fields (the file's start time, subject, experimenter, patient and recording identification) instead of keeping them as the file has them.	off (the header is kept as the file has it)	-
--montage@import recording	NAME	A montage stored in the subject (a points manifold such as `eeg_1005`, made by `create montage`): each channel's name is matched to the montage's labels EXACTLY (case as written; the montage's recorded aliases, e.g. T3 = T7, count and are marked as such); the montage and the match are recorded with the recording, and the channels no label matches are listed. Nothing is guessed: no prefix or suffix is stripped, no channel is placed that no label names.	none (no montage recorded)	-
--format@import recording	FORMAT	The file's format: fif, edf, bdf, brainvision, eeglab, ctf or kit.	from the file's extension	-
--anonymize@import bids		Drop the identifying header fields of every recording and the sidecar's institution and device-serial keys.	off (kept as the files have them)	-
--dataset@import bids	PATH	The folder of the new dataset (made by the import; refused if it holds a catalogue): one subject package per BIDS subject, registered in its catalogue.	required	-
--from@import bids	DIR	The BIDS dataset's folder (MEG, EEG or iEEG recordings).	required	-
--amplitude	VOLTS	The amplitude of the first sine of a synthetic recording (the second has half of it, the third a third, ...).	2e-05	V
--amplitude@create recording	VOLTS	The amplitude of the first sine of the synthetic recording (the second has half of it, the third a third, ...).	2e-05	V
--channels@create recording	N	The number of channels of the synthetic recording (1 to 4096).	8	-
--duration	SECONDS	The duration of a synthetic recording.	60	s
--duration@create recording	SECONDS	The duration of the synthetic recording.	60	s
--frequency	HZ	The frequency of a sine of a synthetic recording (repeatable); each is below half the sampling rate.	10	Hz
--frequency@create recording	HZ	The frequency of a sine of the synthetic recording (repeatable); each is below half the sampling rate.	10	Hz
--noise	VOLTS	The standard deviation of the white noise of a synthetic recording (0 for none).	5e-06	V
--noise@create recording	VOLTS	The standard deviation of the white noise of the synthetic recording (0 for none).	5e-06	V
--seed@create recording	N	The seed of the synthetic recording: the same seed and options give the same samples.	1	-
--sfreq	HZ	The sampling rate of a synthetic recording.	250	Hz
--sfreq@create recording	HZ	The sampling rate of the synthetic recording.	250	Hz
--package@create recording	PATH	The subject package (made by `create package`) the recording is written into.	required	-
--name@create recording	NAME	Name the recording is stored under.	synthetic	-
--acquisition@create recording	NAME	The session of the subject the recording is stored under.	rest	-
--synthetic		Make a synthetic recording: sine waves plus noise from a seed (create recording).	off	-
--synthetic@create recording		Make a synthetic recording: sine waves plus noise from a seed.	off	-
--in-place@import bids		Link every recording's file instead of copying its samples (the packages record each file's path and SHA-256).	off (the samples are copied)	-
--no-participants		Do not import participants.tsv (import bids).	off	-
--no-participants@import bids		Do not import participants.tsv as the dataset's participant variables.	off (participants are imported)	-
--registered-by@import bids	WHO	Who imports: recorded with each catalogue entry.	nxr-neuro and its version	-
--subject@import bids	LABEL	Import only this BIDS subject (its label, with or without `sub-`); repeat the option for several.	all subjects	-
--in-place@import recording		Link the recording's file instead of copying its samples: the package records the file's path and SHA-256, a run streams the samples from the original file, and a changed file is refused. One file only (a CTF folder is copied).	off (the samples are copied)	-
--licence@import recording	TEXT	The licence recorded in the origin of every item the import makes.	unknown	-
--attribution@import recording	TEXT	Who is to be credited, recorded in the origin of every item the import makes.	unknown	-
--anonymize		Drop the identifying fields of a file's header (import recording).	off	-
--format	FORMAT	A file's format (import recording).	from the file's extension	-
--in-place		Link a recording's file instead of copying its samples (import recording).	off	-
--source-root	DIR	The folder a file's recorded path is relative to (import recording).	the file's own folder	-
--acquisition	NAME	The session of the subject the result is stored under (import recording).	rest	-
--from@import fif	FILE	The FIF object file to import: a source space (-src.fif), covariance (-cov.fif), forward solution (-fwd.fif), inverse operator (-inv.fif), event list (-eve.fif), annotation file (-annot.fif) or head-to-MRI transform (-trans.fif).	required	-
--package@import fif	PATH	The subject package (made by `create package`) the object is imported into.	required	-
--kind@import fif	src|cov|fwd|inv|eve|annot|trans	What the file holds, when it is not to be found from its contents.	from the file's contents	-
--name@import fif	NAME	The base name of what is written (items are <name>_source_space, <name>_noise_cov, ...).	the file's name without its suffix (-fwd.fif and the like)	-
--acquisition@import fif	NAME	The session of the subject the object is stored under.	rest	-
--channels@import fif	NAME	The package's sensor manifold the file's channels are matched in, by name (a channel it lacks is refused).	the package's only sensor manifold	-
--recording@import fif	NAME	The package's recording an event or annotation file belongs to (its events are added to it; existing events are never overwritten).	required for -eve.fif and -annot.fif	-
--surface@import fif	NAME	An existing surface manifold of the package to put a surface source space's Selection on; its vertex count must be the source space's.	the surface is made from the file	-
--source-root@import fif	DIR	The folder the file's recorded path is relative to (an absolute path is never recorded).	the file's own folder	-
--licence@import fif	TEXT	The licence recorded in the origin of every item the import makes.	unknown	-
--attribution@import fif	TEXT	Who is to be credited, recorded in the origin of every item the import makes.	unknown	-
--anonymize@import fif		Drop the file names, working folder and command line the file records.	off (kept as the file has them)	-
--from@import nifti	FILE	The NIfTI-1 or NIfTI-2 volume to import (.nii, .nii.gz, or a .hdr/.img pair).	required	-
--package@import nifti	PATH	The subject package (made by `create package`) the volume is imported into.	required	-
--name@import nifti	NAME	The base name of what is written (the manifold is <name>_volume, the field or Selection <name>).	the file's name without .nii / .nii.gz	-
--kind@import nifti	volume|labels	An intensity volume (a Field) or a label volume (a codes Selection); labels need --parcellation.	labels when --parcellation is given, else volume	-
--parcellation@import nifti	NAME	The name of the parcellation a label volume is. A label volume is imported only when it is named.	none (the volume is intensities)	-
--label-names@import nifti	FILE	A text file of `code name` lines naming the entries of the parcellation.	entries are named by their code	-
--field-kind@import nifti	TEXT	The kind of the intensity Field.	intensity	-
--unit@import nifti	TEXT	The unit of the values, when known.	none	-
--source-root@import nifti	DIR	The folder the file's recorded path is relative to (an absolute path is never recorded).	the file's own folder	-
--licence@import nifti	TEXT	The licence recorded in the origin of every item the import makes.	unknown	-
--attribution@import nifti	TEXT	Who is to be credited, recorded in the origin of every item the import makes.	unknown	-
--anonymize@import nifti		Drop the file's free-text description and auxiliary file name.	off (kept as the file has them)	-
--from@import freesurfer	DIR	The FreeSurfer subject folder (with surf/, label/, mri/) to import.	required	-
--package@import freesurfer	PATH	The subject package the surfaces are imported into.	required	-
--name@import freesurfer	NAME	The prefix of what is written: <name>_<surface>, <name>_<annotation>, <name>_<overlay>.	cortex	-
--surfaces@import freesurfer	NAME...	The surfaces to import (lh.<name> and rh.<name> are read as one surface in freesurfer-tkr, metres).	white	-
--annotations@import freesurfer	NAME...	Annotations to import as labelling Selections; named explicitly, none otherwise.	none	-
--overlays@import freesurfer	NAME...	Per-vertex files to import as Fields (thickness, curv, sulc, ...).	none	-
--orig@import freesurfer	FILE	The volume that holds c_ras (the transition from FreeSurfer's surface RAS to scanner RAS).	<subject>/mri/orig.mgz, else the surface file's own volume geometry	-
--field-kind@import freesurfer	TEXT	The kind of the overlay Fields.	overlay	-
--source-root@import freesurfer	DIR	The folder the folder's recorded path is relative to (an absolute path is never recorded).	the folder's parent	-
--licence@import freesurfer	TEXT	The licence recorded in the origin of every item the import makes.	unknown	-
--attribution@import freesurfer	TEXT	Who is to be credited, recorded in the origin of every item the import makes.	unknown	-
--anonymize@import freesurfer		Drop the "created by" line and the volume file name the surfaces record.	off (kept as the files have them)	-
--from@import gifti	FILE	The GIFTI file (.surf.gii, .func.gii, .shape.gii, .label.gii) to import.	required	-
--package@import gifti	PATH	The subject package the file is imported into.	required	-
--name@import gifti	NAME	The base name of what is written.	the file's name without .surf.gii / .gii	-
--kind@import gifti	surface|overlay|labels	What the arrays are: a surface, maps (Fields), or labels (a codes Selection).	from the arrays	-
--surface@import gifti	NAME	The surface of the package that maps or labels are on (the vertex counts must agree).	required for maps and labels	-
--parcellation@import gifti	NAME	The name of the parcellation a label file is. A label file is imported only when it is named.	none	-
--field-kind@import gifti	TEXT	The kind of the Fields written.	overlay	-
--unit@import gifti	TEXT	The unit of the values, when known.	none	-
--source-root@import gifti	DIR	The folder the file's recorded path is relative to (an absolute path is never recorded).	the file's own folder	-
--licence@import gifti	TEXT	The licence recorded in the origin of every item the import makes.	unknown	-
--attribution@import gifti	TEXT	Who is to be credited, recorded in the origin of every item the import makes.	unknown	-
--anonymize@import gifti		Drop the file's free-text metadata.	off (kept as the file has them)	-
--from@import cifti	FILE	The dense CIFTI-2 map (.dscalar.nii, .dlabel.nii) to import.	required	-
--package@import cifti	PATH	The subject package the map is imported into.	required	-
--name@import cifti	NAME	The base name of what is written.	the file's name without .dscalar.nii / .nii	-
--left@import cifti	NAME	The package's surface for the left cortex.	required when the file has the left cortex	-
--right@import cifti	NAME	The package's surface for the right cortex.	required when the file has the right cortex	-
--parcellation@import cifti	NAME	The name of the parcellation a dlabel is. A dlabel is imported only when it is named.	none	-
--field-kind@import cifti	TEXT	The kind of the Fields written.	overlay	-
--unit@import cifti	TEXT	The unit of the values, when known.	none	-
--source-root@import cifti	DIR	The folder the file's recorded path is relative to (an absolute path is never recorded).	the file's own folder	-
--licence@import cifti	TEXT	The licence recorded in the origin of every item the import makes.	unknown	-
--attribution@import cifti	TEXT	Who is to be credited, recorded in the origin of every item the import makes.	unknown	-
--anonymize@import cifti		Accepted for uniformity: nothing identifying is copied.	off	-
--from@import transform	FILE	The -trans.fif file (a head-to-MRI transform, either direction) to import.	required	-
--package@import transform	PATH	The subject package the transform is imported into.	required	-
--name@import transform	NAME	The name of the transition operator.	the file's name without its suffix	-
--channels@import transform	NAME	The sensor manifold of the package that holds the operator.	the package's only sensor manifold, else its first manifold	-
--from@import recording	FILE	The recording file to copy into the package (a CTF recording is its .ds folder).	required	-
--name@import recording	NAME	Name the recording is stored under.	the file's name without its extension	-
--package@import recording	PATH	The subject package (made by `create package`) the recording is copied into.	required	-
--source-root@import recording	DIR	The folder the file's recorded path is relative to (an absolute path is never recorded).	the file's own folder	-
--package@export gifti	PATH	The folder of the subject package (<subject>.nxr.zarr, or a member of a catalogue) whose field or labelling is exported; it is only read.	required	-
--field@export gifti	NAME	The Field to export, by name: a map on a surface (one data array) or a series on a surface and a time line (one data array per frame of the time range). Not with --labelling.	none (give this or --labelling)	-
--labelling@export gifti	NAME	The partition or set Selection on a surface to export as a label file, with the names and colours of its dictionary. Not with --field.	none (give this or --field)	-
--out@export gifti	STEM	The path stem the files are written under (its folder must exist): <STEM>.func.gii or .shape.gii or .label.gii, <STEM>.surf.gii with --with-surface, and the provenance sidecar <first file>.json; with --hemispheres, <STEM>_hemi-L and <STEM>_hemi-R.	required	-
--kind@export gifti	func|shape	The file type of a Field: func (.func.gii, functional data; a series is always func) or shape (.shape.gii, static data on the surface such as curvature).	func	-
--with-surface@export gifti		Also write the surface itself (.surf.gii): the pointset and the triangles, as stored.	off	-
--hemispheres@export gifti		Write one file of each kind per hemisphere (the glued cortex split into the hemispheres it stores as components, left and right), each marked CortexLeft or CortexRight.	off (one file for the whole surface)	-
--surface-unit@export gifti	m|mm	Write the surface's coordinates in this unit (a scale by a power of ten; the stored unit must be stated); GIFTI readers such as Workbench expect mm.	as stored	-
--from@export gifti	SEC	Start of the time range of a series, in seconds on the recording's clock (or m:ss); give together with --to. Required for a series, refused for a map.	required for a series	s
--to@export gifti	SEC	End of the time range of a series, in seconds (or m:ss), exclusive; give together with --from.	required for a series	s
--all@export gifti		The whole series, every frame (the recording is still read one chunk at a time).	off	-
--with-surface		Also write the surface itself (export gifti).	off	-
--hemispheres		Write one file per hemisphere (export gifti).	off	-
--package@export tsv	PATH	The folder of the subject package whose measurements or statistic are exported; it is only read.	required	-
--out@export tsv	STEM	The path stem the table and its sidecar are written under (its folder must exist): <STEM>.tsv and the BIDS sidecar <STEM>.json, whose GeneratedBy holds the provenance.	required	-
--statistic@export tsv	NAME	The stored statistic to write as a long-form table (one row per value, a column per axis, `component` when it has components). Not with --partition.	none (give this or --partition)	-
--partition@export tsv	NAME	The labelling (partition) whose members are the rows of a measurements table; needs --field and --measures.	none (give this or --statistic)	-
--field@export tsv	NAME	The Field the measurements are of, by name (with --partition).	none (required with --partition)	-
--measures@export tsv	NAME,...	The measures to write, one column each: stored ones as stored, derived ones (variance, sd, ...) from their ingredients and named derived in the sidecar (with --partition).	none (required with --partition)	-
--partition	NAME	The labelling whose members are the rows of an exported measurements table (export tsv).	none	-
--statistic	NAME	A stored statistic to export (export tsv).	none	-
--surface-unit	m|mm	The unit the surface's coordinates are written in (export gifti).	as stored	-
--package@export fif	PATH	The folder of the subject package whose field is exported; it is only read.	required	-
--covariance@export fif	NAME	A covariance statistic to write as MNE's noise covariance, `<STEM>-cov.fif` (read by `mne.read_cov`), with the channel names of its channel Selection and its degrees of freedom. Give it instead of --field; it has no time range.	none (a field is written)	-
--forward@export fif	NAME	Refused: a forward solution is not written (an MNE forward needs each channel's FIF info, kind, coil type and position, which a package does not keep).	-	-
--inverse@export fif	NAME	Refused: an inverse operator is not written (an MNE inverse operator is the eigen-decomposition of the whitened forward, which a stored kernel does not hold).	-	-
--field@export fif	NAME	The Field to write as a source estimate, by name: a map on a glued cortex (one time point) or a series on the cortex and a time line (one time point per sample of the time range). The cortex must store its left and right hemispheres as components.	required	-
--out@export fif	STEM	The path stem the files are written under (its folder must exist): <STEM>-lh.stc and <STEM>-rh.stc, read by MNE-Python's read_source_estimate, and the provenance sidecar <STEM>-lh.stc.json.	required	-
--from@export fif	SEC	Start of the time range of a series, in seconds on the recording's clock (or m:ss); give together with --to. Required for a series, refused for a map.	required for a series	s
--to@export fif	SEC	End of the time range of a series, in seconds (or m:ss), exclusive; give together with --from.	required for a series	s
--all@export fif		The whole series, every time point (the recording is still read one chunk at a time and written one time point at a time).	off	-
--package@export bids-derivatives	PATH	The folder of the subject package whose results are exported; it is only read.	required	-
--out@export bids-derivatives	FOLDER	The derivatives folder (made if absent, e.g. derivatives/nxr-neuro): dataset_description.json and sub-<label>/[ses-<label>/] with the files named by BIDS entities. A second run into it merges the description.	required	-
--results@export bids-derivatives	REGEX	A regular expression matched against the WHOLE name of each field on a cortical surface (a map or a series), labelling and statistic to export; the name becomes the desc- label (letters and digits only).	required	-
--space@export bids-derivatives	NAME	The space- entity (the template the maps lie on).	the template the subject is registered to (refused when there is none)	-
--session@export bids-derivatives	LABEL	The ses- entity (and folder).	none (no session)	-
--task@export bids-derivatives	LABEL	The task- entity.	none	-
--run@export bids-derivatives	INDEX	The run- entity.	none	-
--from@export bids-derivatives	SEC	Start of the time range of every series matched, in seconds (or m:ss); give together with --to. Required when a series matches.	required for a series	s
--to@export bids-derivatives	SEC	End of the time range of every series matched, in seconds (or m:ss), exclusive; give together with --from.	required for a series	s
--all@export bids-derivatives		The whole of every series matched.	off	-
--with-surface@export bids-derivatives		Also write the cortical surface (.surf.gii) beside each map and series.	off	-
--stc@export bids-derivatives		Also write each map and series as MNE source estimates (-lh.stc and -rh.stc); the cortex must store its two hemispheres.	off	-
--measurements@export bids-derivatives	PARTITION:FIELD:MEASURES	Also write the measurements table of a partition's members for a field: the partition, the field and comma-separated measures, joined by colons.	none	-
--results	REGEX	The names of the results to export (export bids-derivatives).	none	-
--space	NAME	The space- entity of an exported file (export bids-derivatives).	none	-
--session	LABEL	The ses- entity of an exported file (export bids-derivatives).	none	-
--task	LABEL	The task- entity of an exported file (export bids-derivatives).	none	-
--run	INDEX	The run- entity of an exported file (export bids-derivatives).	none	-
--stc		Also write source estimates (export bids-derivatives).	off	-
--measurements	PARTITION:FIELD:MEASURES	A measurements table to write (export bids-derivatives).	none	-
)TBL",
R"TBL(@ create dataset
old: dataset create
alias: refused
summary: Create a dataset: a folder holding a catalogue of subject packages (the default form). --datastore makes the old one-database form.
opts: --atlas-finest --dataset --datastore --default --lc-scales --modes --name --source-path --source-protocol --source-tool --store-format --templates
)TBL",
R"TBL(@ list datasets
old: dataset list
alias: refused
summary: List the dataset folders found in a folder.
opts: --datastore*
)TBL",
R"TBL(@ show dataset
old: dataset show
alias: refused
summary: Show a dataset: its subjects by kind, its source and its creation date.
opts: --dataset*
)TBL",
R"TBL(@ delete dataset
old: dataset delete
alias: refused
summary: Delete a dataset's folder.
opts: --datastore* --name*
)TBL",
R"TBL(@ recover dataset
old: dataset recover
alias: refused
summary: Finish or roll back an interrupted write in a dataset.
opts: --dataset*
)TBL",
R"TBL(@ compact dataset
old: dataset compact
alias: refused
summary: Pack a dataset's small files and shards to save space (a dry run shows what would change).
opts: --dataset* --dry-run --no-shard --no-small-files --subject
)TBL",
R"TBL(@ create recording
old: __import synthetic
alias: no
summary: Make a small synthetic recording (sine waves plus noise from a seed; channels, sampling rate and duration as options) in a subject package, as an imported recording is stored.
opts: --acquisition --amplitude --channels --duration --frequency+ --name --noise --package* --seed --sfreq --synthetic
example: nxr-neuro create recording --synthetic --package work/s1.nxr.zarr --name rest --channels 8 --sfreq 250 --duration 60 --seed 1
)TBL",
R"TBL(@ create package
old: package create
alias: no
group: Subject packages
summary: Create a subject package: one folder holding one subject's rows, small keys and shards, openable on its own.
opts: --home-dataset* --kind --name* --package*
)TBL",
R"TBL(@ create index
old: package index
alias: no
summary: Build the cross-subject index of a catalogue dataset (a cache made from its packages; deletable).
opts: --dataset* --force
)TBL",
R"TBL(@ register package
old: package register
alias: no
summary: Register subject packages in a dataset's catalogue (idempotent; each package is checked, none is moved or read).
opts: --dataset* --package+ --registered-by* --replace --scan --templates+
)TBL",
R"TBL(@ list members
old: package status
alias: no
summary: List a catalogue's subject packages with their registered state (registered, missing or stale).
opts: --dataset*
)TBL",
R"TBL(@ import participants
old: package participants-import
alias: no
summary: Import a BIDS participants.tsv (and its participants.json) into the dataset's own participants file.
opts: --dataset* --sidecar --tsv*
)TBL",
R"TBL(@ export participants
old: package participants-export
alias: no
summary: Write the dataset's participants table as BIDS participants.tsv, only where the command names.
opts: --dataset* --sidecar --tsv*
)TBL",
R"TBL(@ split dataset
old: package split
alias: no
summary: Split a one-database dataset into subject packages with a catalogue, in place, verified row for row and byte for byte, resumable; the old database is kept.
opts: --dataset* --dry-run --registered-by*
)TBL",
R"TBL(@ migrate measurements
old: package migrate-measurements
alias: no
summary: Make every value over a selection's members one measurement (schema 53): per-member statistics become array-backed measurements, verified byte for byte; a product that only coupled a level names the grid and the level; day-anchored tower tilings are marked read-only. In place, subject by subject, resumable; the old statistics are kept.
opts: --dataset* --dry-run --remove-statistics
)TBL",
R"TBL(@ join dataset
old: package join
alias: no
summary: Join a catalogue's subject packages back into a one-database dataset (the reverse of split); everything replaced is kept aside.
opts: --dataset*
)TBL",
R"TBL(@ verify package
old: package verify
alias: no
summary: Recompute every key's digest and the content hash of a subject package and report any difference.
opts: --package*
)TBL",
R"TBL(@ show package
old: package show
alias: no
summary: Show a subject package's record: its subject, versions, template, release and content hash.
opts: --package*
)TBL",
R"TBL(@ release template
old: package release
alias: no
summary: Release a template (or registration) package as a frozen, versioned copy; refuses an item with no origin record.
opts: --into* --name* --package* --released-by* --template-version*
)TBL",
R"TBL(@ release redistributable
old: package release-redistributable
alias: no
summary: Make the redistributable edition of a released template, keeping only the items whose origin allows it.
opts: --into* --package* --released-by*
)TBL",
R"TBL(@ create template
old: package template-create
alias: no
summary: Build a template package (fsaverage, or ICBM152 with HCP-1065 streamlines) from a folder of the source files you downloaded, each checked against the manifest and given an origin record.
opts: --accept-unlisted-sources --cortex-mask --scalp --hcp-end-surface --hcp-points --hcp-trk --home-dataset --max-end-distance --name --package* --parcellation+ --recipe* --sources* --surface+
example: nxr-neuro create template --recipe fsaverage --sources ~/downloads/fsaverage --package work/fsaverage.nxr.zarr --parcellation aparc --parcellation aparc.a2009s
)TBL",
R"TBL(@ create registration
old: package registration-create
alias: no
summary: Make a registration between two released templates through their registered spheres: the operators that carry a field from one to the other.
opts: --attribution --from* --from-sphere* --home-dataset --licence --method --name* --neighbours --one-direction --package* --redistributable --source --templates+ --to* --to-sphere*
example: nxr-neuro create registration --package A-to-B.nxr.zarr --name A-to-B --from A@1.0 --to B@1.0 --from-sphere cortex_pial_sphere --to-sphere cortex_pial_sphere --method shepard
)TBL",
R"TBL(@ list templates
old: package templates
alias: no
summary: List the released templates and registrations found on the template search path.
opts: --templates+
)TBL",
R"TBL(@ create subject
old: subject create
alias: refused
summary: Create a subject (or a template or group subject) in a dataset.
opts: --dataset* --kind --name*
)TBL",
R"TBL(@ list subjects
old: subject list
alias: refused
summary: List the subjects of a dataset; of a catalogue dataset, those named by --subjects or satisfying --where over the participant variables.
opts: --dataset* --subjects --where
)TBL",
R"TBL(@ show subject
old: subject show
alias: refused
summary: Show how many manifolds, fields, operators and selections a subject holds.
opts: --dataset* --subject* --templates+
)TBL",
R"TBL(@ delete subject
old: subject delete
alias: refused
summary: Delete a subject, or one of its nodes, from a dataset.
opts: --dataset* --node --subject* --templates+
)TBL",
R"TBL(@ add surface
old: surface add
alias: refused
summary: Add a mesh file (OBJ, PLY or STL) to a subject as a surface.
pos: FILE
opts: --dataset* --name --no-primary --primary --scale --subject* --templates+
)TBL",
R"TBL(@ list recordings
old: __list recordings
summary: List the recordings of a subject.
opts: --dataset* --subject* --templates+
)TBL",
R"TBL(@ list kernels
old: __list kernels
summary: List the inverse kernels of a subject.
opts: --dataset* --subject* --templates+
)TBL",
R"TBL(@ list results
old: __list results
summary: List the fields and statistics the computations stored for a subject.
opts: --dataset* --subject* --templates+
)TBL",
R"TBL(@ list pipelines
old: run --list
alias: refused
summary: List the pipeline definitions and the steps they are made of.
unprobed: yes
opts: 
)TBL",
R"TBL(@ import list
old: convert list
alias: refused
summary: List the subjects of a Brainstorm protocol.
pos: PROTOCOL
opts: 
)TBL",
R"TBL(@ import scan
old: convert scan
alias: refused
summary: Scan a Brainstorm protocol and describe what it holds.
pos: PROTOCOL
opts: --out --prefer --summary
)TBL",
R"TBL(@ import subject
old: convert subject
alias: refused
summary: Import a Brainstorm subject's anatomy into a dataset.
pos: PROTOCOL
opts: --dataset* --resample --subject* --templates+
)TBL",
R"TBL(@ import template
old: convert template
alias: refused
summary: Import a Brainstorm protocol's template subject into a dataset.
pos: PROTOCOL
opts: --dataset* --surface+
)TBL",
R"TBL(@ import group
old: convert group
alias: refused
summary: Import a Brainstorm protocol's group subject into a dataset.
pos: PROTOCOL
opts: --dataset*
)TBL",
R"TBL(@ import templates
old: convert templates
alias: refused
summary: Import the fsaverage reference templates (and optional fibre tracts) into a dataset.
pos: TEMPLATES_DIR
opts: --dataset* --subject --surface --trk
)TBL",
R"TBL(@ import surface
old: convert surface
alias: refused
summary: Import Brainstorm tessellation files as surfaces of a subject.
opts: --bst-root --dataset* --file+ --subject* --templates+
)TBL",
R"TBL(@ import mri
old: convert mri
alias: refused
summary: Import an anatomical MRI of a subject.
opts: --anat* --dataset* --only+ --subject* --templates+
)TBL",
R"TBL(@ import bids
old: __import bids
alias: no
summary: Import a BIDS dataset (MEG, EEG, iEEG) as a dataset of subject packages, with its participant variables, channel status, events and electrode positions.
opts: --anonymize --dataset* --from* --in-place --no-participants --registered-by --subject+
)TBL",
R"TBL(@ import fif
old: __import fif
alias: no
summary: Import an MNE FIF object (source space, covariance, forward solution, inverse operator, events, annotations or head-to-MRI transform) into a subject package.
opts: --acquisition --anonymize --attribution --channels --from* --kind --licence --name --package* --recording --source-root --surface
)TBL",
R"TBL(@ import nifti
old: __import nifti
alias: no
summary: Import a NIfTI volume into a subject package: a lattice in its voxel chart (voxel size as spacing in mm), its intensities as a Field (or a named parcellation as a codes Selection), and the voxel-to-world transform as a transition operator in millimetres.
opts: --anonymize --attribution --field-kind --from* --kind --label-names --licence --name --package* --parcellation --source-root --unit
)TBL",
R"TBL(@ import freesurfer
old: __import freesurfer
alias: no
summary: Import a FreeSurfer subject into a subject package: surfaces in freesurfer-tkr (metres) with the transition to scanner RAS from c_ras, named annotations as labelling Selections, overlays as Fields.
opts: --annotations+ --anonymize --attribution --field-kind --from* --licence --name --orig --overlays+ --package* --source-root --surfaces+
)TBL",
R"TBL(@ import gifti
old: __import gifti
alias: no
summary: Import a GIFTI file into a subject package: a surface in the system its transform matrix names, maps as Fields, or a named parcellation as a codes Selection.
opts: --anonymize --attribution --field-kind --from* --kind --licence --name --package* --parcellation --source-root --surface --unit
)TBL",
R"TBL(@ import cifti
old: __import cifti
alias: no
summary: Import a dense CIFTI-2 map into a subject package: the cortical brain models on named surfaces through an indices Selection, maps as Fields, a named parcellation as codes Selections.
opts: --anonymize --attribution --field-kind --from* --left --licence --name --package* --parcellation --right --source-root --unit
)TBL",
R"TBL(@ import transform
old: __import transform
alias: no
summary: Import a head-to-MRI transform (an MNE -trans.fif) into a subject package as a transition operator from the head frame to FreeSurfer's surface RAS.
opts: --channels --from* --name --package*
)TBL",
R"TBL(@ import recording
old: __import recording
alias: no
summary: Copy a recording file (FIF, EDF, BDF, BrainVision, EEGLAB, CTF, KIT) into a subject package, with its channels, events, bad channels and an origin record.
opts: --acquisition --anonymize --attribution --format --from* --in-place --licence --montage --name --package* --source-root
)TBL",
R"TBL(@ export gifti
old: __export gifti
alias: no
summary: Write a cortical Field (a map, or a series over a stated time range) or a labelling as GIFTI (.func.gii, .shape.gii, .label.gii), with the surface on request and a provenance sidecar.
opts: --all --field --from --hemispheres --kind --labelling --out* --package* --surface-unit --to --with-surface
)TBL",
R"TBL(@ export tsv
old: __export tsv
alias: no
summary: Write stored measurements (a row per member of a partition, a column per measure) or a statistic (long form) as a BIDS-style TSV with a JSON sidecar of units, levels and provenance.
opts: --field --measures --out* --package* --partition --statistic
)TBL",
R"TBL(@ export fif
old: __export fif
alias: no
summary: Write a source map or a source series on the two hemispheres of a cortex as MNE source estimates (-lh.stc and -rh.stc), or a covariance statistic as MNE's -cov.fif, with a provenance sidecar.
opts: --all --covariance --field --forward --from --inverse --out* --package* --to
)TBL",
R"TBL(@ export bids-derivatives
old: __export bids-derivatives
alias: no
summary: Write results (maps, series, labellings, statistics, measurements, source estimates) as a BIDS derivatives folder: files named with BIDS entities, sidecars with Sources and provenance, and a dataset_description.json listing nxr-neuro's suffixes.
opts: --all --from --measurements --out* --package* --results* --run --session --space --stc --task --to --with-surface
)TBL",
R"TBL(@ import condition
old: convert convert
alias: refused
summary: Import a Brainstorm condition (its recordings, channels and head models) into a dataset.
pos: PROTOCOL
opts: --all-surfaces --channel-file --condition* --dataset* --extra-surface+ --include-imported --no-gain --no-raw --recording --resample --subject* --surface --templates+
)TBL",
R"TBL(@ import inverse
old: convert inverse
alias: refused
summary: Import a Brainstorm inverse kernel as an operator of a subject.
opts: --acquisition* --bst-root* --channel-file --channels-id --dataset* --file* --forward-id --name --subject* --surface-id --templates+
)TBL",
R"TBL(@ import map
old: convert map
alias: refused
summary: Import Brainstorm source or matrix maps as fields of a subject.
opts: --acquisition --bst-root* --dataset* --file+* --provenance --replace --subject* --templates+
)TBL",
R"TBL(@ import resample
old: convert resample
alias: refused
summary: Resample a recording to a new sampling rate, as a new recording.
opts: --dataset* --name --recording* --subject* --to* --templates+
)TBL",
R"TBL(@ import badsegments
old: convert badsegments
alias: refused
summary: Import a recording's bad segments from a file.
opts: --dataset* --from* --replace --subject* --templates+
)TBL",
R"TBL(@ import events
old: convert events
alias: refused
summary: Import a recording's events from a file.
opts: --dataset* --from* --replace --subject* --templates+
)TBL",
R"TBL(@ read dataset
old: read
summary: Read an nxr dataset folder: list its arrays, or read one array or region.
pos: FOLDER
unprobed: yes
opts: --array+ --out --region
)TBL",
R"TBL(@ read crossspectrum
old: crossspectrum read
alias: refused
summary: Read coherency, phase, delay or phase-slope index from a stored band-resolved cross-spectrum.
opts: --dataset* --name* --pair --reading --seed --subject* --weights --templates+
)TBL",
R"TBL(@ query measurements
old: measure query
alias: refused
summary: Query stored measurements over a selection's members at any spatial and temporal scale, stated as quantities (rolled up at read).
opts: --band --dataset* --factor --field --limit --max-value --measure* --min-value --order --selection* --spatial-scale --subject* --subjects --tile --templates+ --where
)TBL",
R"TBL(@ export measurements
old: measure export
alias: refused
summary: Export stored measurements as a table for statistics (a row per subject), with its dictionary and provenance.
opts: --dataset* --exclude --fields* --keep-all --labelling* --layout --out* --recipe --scaling --statistics --subjects --table*
)TBL",
R"TBL(@ run surrogate_lagfree_spatial_events_compare
old: __surrogate_lagfree_spatial_events_compare compute
group: Surrogates
summary: The spatial events against lag-free surrogate draws (draws with early stopping): per-vertex p and q, the patch sizes.
span: yes
opts: --batch --chunks --dataset* --events --exceedances=--h --max-draws --min-draws --recording* --replace --set+ --start --stop --subject* --templates+
)TBL",
R"TBL(@ run surrogate_lagfree_vortex_compare
old: __surrogate_lagfree_vortex_compare compute
group: Surrogates
summary: The vortex cores' tile crossings against lag-free surrogate draws (draws with early stopping): per-tile p and q, the run repeat.
span: yes
opts: --batch --chunks --dataset* --exceedances=--h --max-draws --min-crossings --min-draws --over --recording-a* --recording-b --replace --set+ --stage --start --stop --subject* --templates+
)TBL",
R"TBL(--inion	X,Y,Z	The inion in the head's coordinates (metres), when you know it; create montage.	the farthest crossing of the line from the nasion through the midpoint of the preauricular points with the scalp (recorded in the montage)	m
--landmarks	NAME	The points manifold of the nasion, LPA and RPA of the template (create montage).	head_landmarks	-
--package@create montage	PATH	The template (or any subject) package that holds the scalp and its landmarks; the montage is added to it (a working copy, not a frozen release).	required	-
--subject@create montage	NAME	The subject of the package the montage is added to.	the package's only subject	-
--scalp@create montage	NAME	The closed scalp surface the positions are placed on.	head	-
--landmarks@create montage	NAME	The points manifold of the nasion, LPA and RPA, in the scalp's coordinate system.	head_landmarks	-
--name@create montage	NAME	Name of the stored points manifold (its names, system partition and aliases are stored beside it).	eeg_1005	-
--inion@create montage	X,Y,Z	The inion in the head's coordinates (metres), when you know it; the convention otherwise: the farthest crossing of the line from the nasion through the midpoint of the preauricular points with the scalp.	the convention (recorded in the montage)	m
--tolerance@create montage	T	Cz stops moving when it moves less than this fraction of the nasion-inion distance.	1e-7	-
--max-iterations@create montage	N	The most Cz iterations before refusing.	100	iterations
)TBL",
R"TBL(@ create montage
old: __montage create
alias: no
summary: Place the standard 10-05 electrode montage (345 positions; the 10-10 and 10-20 are subsets of its names) on a template's own scalp, stored in its package as a points manifold with its names, for `import recording --montage`.
opts: --inion --landmarks --max-iterations --name --package* --scalp --subject --tolerance
example: nxr-neuro create montage --package work/icbm152.nxr.zarr --name eeg_1005
)TBL",
  };
}
