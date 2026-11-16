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
R"TBL(verb compute	Compute something from a recording or from stored results (psd, crossspectrum, phasegradient, pyramid, ...)
verb run	Run a pipeline definition on a subject
verb list	List datasets, subjects, recordings, kernels, results or pipelines
verb show	Show a dataset or a subject
verb read	Read stored arrays and derived readings
verb query	Query stored measurements
verb export	Export stored measurements as a table
verb import	Import data from Brainstorm into a dataset
verb create	Create a dataset or a subject
verb delete	Delete a dataset or a subject
verb compact	Compact a dataset
verb recover	Recover a dataset after an interrupted write
verb reduce	Reduce stored results across subjects
verb add	Add a surface to a subject
)TBL",
R"TBL(--acquisition	NAME	The Brainstorm acquisition (condition) the kernel or map belongs to.	none	-
--all		Do the time line of every recording in the subject (alternative to --recording and --line)	off	-
--all-surfaces		Import every surface of the subject, not only the ones the condition's head model uses.	off	-
--alpha	X	Horn-Schunck smoothness weight of the optical flow; vary it and report the range.	0.01 for the connection regulariser, 1 for ambient	-
--anat	FILE	The anatomical MRI file to import.	none	-
--anchor	HZ	Itersine and similar time filterbanks: the anchor frequency the member spacing is built from.	1.0	Hz
--array	SUBJECT PATH	Read one array: the subject and the array's path in the dataset (prints its bytes, or writes them with --out).	none	-
--atlas-depth	LEVEL	Leaf level of the group tree that serves as the dataset's default atlas (needs --templates)	8	level
--band	LO,HI	The frequency band in hertz (frametiling: its centre sets the frame length; wishart: the band of the stored windows; measure query: INDEX picks one band by its index, a whole number).	none	Hz
--bands	NAME:LO-HI,...	The frequency bands to compute, each a name and its low and high edge (the mean over the bins from LO to HI, inclusive); pyramid build takes LO-HI,... octave bands without names.	Brainstorm's bands (delta 2-4, theta 5-7, alpha 8-12, beta 15-29, gamma1 30-59, gamma2 60-90); pyramid build: octaves 1-2 to 32-64	Hz
--bandwidth	HZ	Notch width: the 3 dB bandwidth of each notch	1	Hz
--batch	N	Number of null draws made at a time before the early-stopping test (Besag-Clifford).	20	draws
--berg-dipoles	J	Number of equivalent dipoles in the Berg-Scherg approximation of the three-shell EEG sphere.	3	-
--berg-starts	N	Number of simplex starts when fitting the three-dipole Berg-Scherg approximation of the EEG sphere model; the best fit is kept	12	-
--berg-terms	N	Number of series terms fitted when approximating the multi-shell EEG model by the Berg-Scherg parameters.	200	terms
--block-frames	N	How many frames the steps after a frame tiling process at once; limits memory only, the result does not change (0 = the whole chunk's frames).	256	frames
--block-samples	N	Samples processed at a time, which sets memory use only and does not change the result (run: width for the steps after the last filter, 0 = whole chunk; pyramid build: samples projected at a time)	run: 1024; pyramid build: 512 (fixed-orientation kernel) or 128 (free)	samples
--block-seconds	SECONDS	Length of the blocks whose own mean is removed before the covariance is formed (noisecov: noise window; inverse: when computing the noise covariance from a recording).	one block (a single mean for the whole window)	s
--bst-root	PATH	Root folder of the Brainstorm protocol, used to resolve the surface files named by --file.	none (empty)	-
--bursts	NAME	Name of a stored set of burst time spans (wishart: required, each burst is tested; cellquadrature: optional, also computed per burst)	none	-
--centre	HZ	Centre frequency of the band from which the frame length is derived (the coarsest tower level whose frame is no longer than 1/(centre x frames-per-cycle)).	none	Hz
--centres	HZ,...	Centre frequencies of the filterbank members (Morlet, Morse, Amor and Bump families); for Morlet, empty means the single member at --freq (opticalflow).	none	Hz
--channel-file	FILE	The channel file to use instead of the condition's own.	the condition's channel file	-
--channels	NAME	Name of the sensor channel set the head model is computed for	the subject's only channel set with a type labelling	-
--channels-id	ID	The id of the channels the kernel was made for.	found from the kernel file	-
--channels-of	KERNEL	Use the channels of this stored inverse kernel, in its order, for the cross-spectrum; an alternative to --types.	none (all good channels of the chosen types)	-
--chunks	K|A-B|all	Which part of the recording to read: one chunk K, a range of chunks A-B, or all chunks looped one after another (a chunk is 2^15 samples of every channel). Exclusive with --start/--stop.	0 (one chunk)	chunks
--combine	each|pool|mean	How values are combined before summarising: each row separately, all rows pooled, or the mean over rows per member	each	-
--condition	NAME	The Brainstorm condition to import.	none	-
--convention	scs|ras	Axes in which the surface's own coordinates are given: scs (anterior +x, superior +z, left +y) or ras (anterior +y, superior +z, left -x).	scs	-
--cross-spectrum	NAME	Name of the stored sensor cross-spectrum (from crossspectrum compute) to read.	none (required)	-
--cycles	C	Cycles of the band's lowest frequency in each window (constantq) or wavelet width as centre frequency / C (speedspectrum, opticalflow)	constantq: set by --method (about 6.8 at 1200 Hz); speedspectrum, opticalflow: 6	cycles
--datafit	auto|gauge|triangle	Data term of the Horn-Schunck flow: auto picks the regulariser's own (gauge for connection, triangle for ambient); gauge is connection only; triangle uses per-triangle finite elements.	auto	-
--dataset	DATASTORE/NAME	The dataset to work on, as datastore/dataset (show, recover and compact take the dataset folder).	none (required)	-
--datastore	DIR	Folder that holds the datasets	none (required)	-
--ddof	N	Delta degrees of freedom: the standard deviation divides by n minus this number (1 matches MATLAB's std).	1	-
--default	NAME	The template resolution used for the dataset's default subject (needs --templates).	fsaverage5	-
--defects	keep|exclude	keep: use every edge of the mesh in the weighted fit; exclude: give zero weight to folded or very short edges (needs --spatial-band)	keep	-
--depth	LEVEL	Level of the surface's tile ladder that gives the leaf tiles on which sums are stored (2^LEVEL tiles per piece); for tiles compute, the finest level written.	the finest level every piece of the surface reaches (pyramid reduce: the stored depth)	level
--depth-channels	all|one-type	Which channels the depth-weighting norm is computed on: all channels, or one channel type (gradiometers if present, else magnetometers, else EEG).	all	-
--depth-exponent	P	Exponent of the depth weighting that counteracts the bias of the minimum-norm solution towards superficial sources	0.5	-
--depth-limit	X	Upper limit of the depth weighting (Brainstorm's WeightLimit): the largest-to-smallest weight ratio is limited to this value squared in variance.	10	-
--depth-limit-on	weight|norm	Where the depth limit is applied: on the weights (Brainstorm's way) or on the inverse norms (MNE-Python's way).	weight	-
--depth-norm	frobenius|spectral	Matrix norm used for each source's gain when depth weighting	frobenius	-
--detrend	none|constant|linear	Trend removed from each Welch segment before windowing.	constant (the Brainstorm option set)	-
--divisor	samples-minus-blocks|samples|samples-minus-one	What the covariance sum is divided by: samples minus the number of blocks, the samples, or samples minus one.	samples-minus-blocks	-
--draws	N	Number of complex Wishart random draws used for the null distribution	1000	draws
--dry-run		Report what would change (compact: files and folders before and after; run: inputs found and arrows checked) without changing or computing anything.	off	-
--edge-lag	frame|frame-normal|frame-tangent|axial|ambient	How each edge's phase lag is read: in the surface's own frame (frame), its normal or tangent part alone, modulo pi (axial), or from the x y z components as they are (ambient).	frame	-
--eeg-conductivities	S1,S2,S3	Conductivities of the EEG sphere shells, innermost to outermost (brain, skull, scalp)	0.33,0.0042,0.33	S/m
--eeg-radii	R1,R2,R3	Relative radii of the EEG head model's shells, innermost to outermost (brain, skull, scalp).	0.88,0.93,1.0	relative
--eeg-series		Use the multi-shell series itself for the EEG sphere instead of the Berg-Scherg equivalent dipoles (slower, exact up to truncation).	off	-
--eeg-series-terms	N	Number of terms of the multi-shell series when it is used instead of the three-dipole fit (with --eeg-series)	200	terms
--eeg-surface	NAME	Name of the surface the EEG sphere is fitted to.	the surface called scalp (or head)	-
--enter-sd	SD	A patch is kept when a tile reaches the median plus this many robust standard deviations of the map's tile means.	1.5	SD
--estimator	welch|analytic	How the sensor cross-spectrum is estimated: Welch segments, or the band's analytic signal summed per time tile (a series of windows)	welch	-
--events	NAME	Name of the spatial-events pipeline whose stored statistics are tested against the null.	spatial_events	-
--exceedances	N	Besag-Clifford stopping count: a tile or vertex is decided once it has this many exceedances of the data.	10	exceedances
--exclude	NAMES	Comma-separated member names left out of the exported table.	unassigned	-
--exit-sd	SD	A patch extends over tiles at or above the median plus this many robust standard deviations	1.0	robust SD
--extra-surface	FILE	An extra surface file to import with the condition (repeatable).	none	-
--family	NAME	Family of the time filter bank: itersine, meyer, morlet, morse, amor or bump (opticalflow: see the command's description for the accepted names).	none (required)	-
--field	NAME	Name of a stored field to read (measure query: restrict the query to this field; phasegradient: the phase 1-form to build, only gradient exists).	measure query: none; phasegradient: gradient	-
--fields	PATTERN	Regular expression matched against the whole name of each stored field to measure (export: the first capture group names the condition)	none (required)	-
--file	FILE	A Brainstorm anatomy file (tess_*.mat) to convert; repeat or give several.	none (required)	-
--fit	inner|mesh|modes	How the timing map is fitted to the edge lags: by inner products with the modes, by a weighted fit on the whole surface complex, or by a weighted fit within the exact modes.	inner	-
--flip-eps	E	Tolerance of the intrinsic Delaunay edge flips: flip while the cotangent weight is below -E	1e-6	-
--flow-tolerance	X	Relative residual at which the conjugate-gradient flow solve stops (pcg and per-scale methods).	1e-8	-
--fmax	HZ	Upper frequency limit (constantq: ceiling of the time ladder, at most Nyquist; speedspectrum: the highest frequency bin used).	constantq: 64 Hz cap (method default); speedspectrum: 40	Hz
--fmin	HZ	Lowest frequency (constantq: lowest band edge; speedspectrum: lowest frequency bin used)	constantq: 0.5 (hertz method) or 0 (cycles method); speedspectrum: 1	Hz
--fold-edges	keep|axial|all	How edges across a fold of the surface are read: keep all as they are, axial (lag read modulo pi where the ends' normals are opposed), or all (every edge modulo pi).	keep	-
--forward	NAME	Name of the stored head model (forward operator) to use.	the subject's only one	-
--forward-id	ID	The id of the head model the kernel was made with.	found from the kernel file	-
--fpass	HZ|LO,HI	Passband edge: one frequency for lowpass or highpass, LO,HI for bandpass	none (required)	Hz
--fraction	X	Fraction of the peak power at the reference level above which the connected region counts as the source's patch.	0.5	-
--frame-level	L	Tower level of a time frame, a whole number: a frame lasts 86400 x 2^L seconds (-18 = 0.33 s, -25 = 2.57 ms). (frametiling: alternative to --centre or --band)	-18 (frametiling: none, give a level, centre or band)	-
--frames-per-cycle	N	Frames per cycle of the band's centre frequency, which sets the frame level (the coarsest whose frame is at or under 1/(centre x N))	36	frames
--freq	HZ	Frequency whose nearest filter-bank member is analysed.	none (required)	Hz
--frequencies	F,...	Centre frequencies to remove (notch), e.g. a line frequency and its harmonics.	none (give this or --line)	Hz
--from	FILE	The file to read the bad segments or events from (import badsegments, import events).	none	-
--fstop	LO,HI	Stopband edges of a bandstop filter	none (required)	Hz
--gain	X	Peak in-band response of each filter-bank member (2 makes the coefficient's magnitude the physical amplitude).	none (required)	-
--gamma	G	Morse family: the gamma parameter of the wavelet.	3	-
--gradient		Also measure the gradient of each scalar field (area-weighted face gradients projected on the vertex tangent plane)	off	-
)TBL",
R"TBL(--graph-family	mexhat|heat|itersine|logitersine	Family of the spatial (graph) filter bank over the eigenvalue.	none (required)	-
--graph-filters	N	Number of members of the graph (eigenvalue) filterbank; 0 means derived from --graph-voices.	0	filters
--graph-voices	N	Members per octave of the spatial (graph) filterbank	3	members/octave
--half-width	OCT	Meyer family: half-width of each transition between members, in octaves.	0.3333 (1/3)	octaves
--halo-seconds	SECONDS	Context read on each side of a chunk so the band filter's edge effects fall inside it	8	s
--harmonics	N	With --line, the number of harmonics of the line frequency to notch.	every multiple below the Nyquist frequency	-
--include-imported		Also import conditions that were themselves imported.	off	-
--input	amplitude|signed	What the flow is computed on: the source amplitude |J| of the kernel applied to the complex band coefficients, or the signed band signal itself (fixed-orientation kernels only).	amplitude	-
--joint	NAME	Name of the stored joint space-time partition to use	the one for the cross-spectrum's recording, the space level and its tile level	-
--joint-types		Whiten the channel types of a modality (e.g. magnetometers with gradiometers) together, keeping their cross-covariance, instead of one block per type.	off	-
--keep-all		Keep every member in the table, including those excluded by default.	off	-
--keep-bad-segments		Keep the recording's bad segments in the computation instead of leaving them out	off	-
--kernel	NAME	Name of the stored inverse kernel (the map from channels to sources) to apply.	the subject's only kernel	-
--kernels	SPEC,...	Which kernels' pyramids are summed: all, constrained, free, latest, a method, stamp, name or glob (kinds combine with AND, tokens of one kind with OR).	all	-
--kind	subject|template|group	Kind of subject to create	subject	-
--labelling	NAME	The partition (a stored Selection with one code per vertex, e.g. a parcellation or a tile level) whose members are measured.	none (required)	-
--layout	split|wide	Table layout: split gives a table per condition and statistic with a column per member; wide gives one table with a column per condition, member and statistic.	split	-
--lc-levels	L,L,...	Tile-ladder levels of the Levi-Civita connection stored with the default subject (needs --templates)	4,6	level
--level	L	A level (patches: the tile-ladder level whose spatial band is used; crossspectrum, cyclepattern: the window tile level, windows of 2^L samples; measure compute: the level of a constant-Q time ladder).	none (required for patches; otherwise see the command's description)	level
--levels	L,...	Spatial tile levels of the scale space the blur is measured over.	2,3,4,5,6,7,8	levels
--limit	N	Maximum number of rows returned	none	rows
--limits	LO-HI	Frequency limits of the filter bank for the itersine and meyer families.	see the command's description	Hz
--line	NAME or F	Time line to work on (constantq: a time line by name, alternative to --recording or --all); for lowpass/notch: the mains frequency F, removed with its harmonics.	none	constantq: -; notch: Hz
--list	V,V,...	run: list the available pipeline definitions and their steps (flag); pointspread: native vertex indices to use as the source sample instead of --vertices	run: off; pointspread: none	-
--map	NAME	The stored cortical map (a scalar per vertex of a surface) in which patches are sought.	none (required)	-
--max-area	M2	Largest patch area kept; larger patches are dropped.	none	m^2
--max-condition	C	Canonical frames: a vertex whose push-forward Jacobian has a condition number above C is treated as singular	50	-
--max-draws	N	Largest number of null draws.	100	draws
--max-iterations	N	Iteration limit of the iterative flow solver (pcg and perscale methods).	500	iterations
--max-value	X	Keep only rows with a value at or below X	none	measure units
--measure	NAME	The measure to summarise or read (e.g. mean, max, power_sum).	none (required)	-
--measures	NAME,...	Which measures to compute per member: mean, max, min, mean_square, abs_max, sum.	mean, max, min, mean_square, abs_max	-
--meg	overlapping|single|none	MEG head model: per-sensor overlapping spheres, one single sphere, or no MEG channels	overlapping	-
--meg-surface	NAME	Name of the surface the MEG spheres are fitted to.	the surface called inner skull	-
--members-own		The measure is the members' own (area, number of elements or tiles); --of then names its parent.	off	-
--method	NAME	constantq: hertz (octave edges on powers of two in Hz) or cycles (fixed cycle count); opticalflow: pcg, direct or perscale solver	constantq: hertz; opticalflow: pcg	-
--min-area	M2	Smallest patch kept.	none	m^2
--min-crossings	N	Minimum number of crossings a tile needs (in both runs for repeat) to be tested.	10	crossings
--min-draws	N	Smallest number of null draws made before early stopping may decide a tile or vertex	20	draws
--min-link	N	Two events of different scale pairs count as one event when at least this many pairs of their cells nest in space and overlap in time.	1	pairs
--min-projection	P	Reliability limit of the anatomical frame: the anterior axis must project on the tangent plane with at least this length.	0.34	-
--min-value	X	Keep only rows with a value at or above X	none	measure units
--mne		Use MNE-Python's conventions for the kernel: joint types, ridge over the rank, PCA whitener, depth exponent 0.8 by the spectral norm.	off	-
--mne-eeg-layers		Use MNE-Python's EEG sphere layers and conductivities instead of the default three-shell layers.	off	-
--modes	K	Number of Hodge-Dirac eigenmodes (eigen solve, dataset create: how many to solve; pointspread, cellquadrature, patches: size of the package when it must be computed, 0 = standard; speedspectrum, hodgepower: use the first K of the stored package)	eigen solve: 3000; dataset create: none; speedspectrum, hodgepower: all; pointspread, cellquadrature, patches: 0	modes
--mollify	X	Mollification margin of the intrinsic Delaunay construction (Sharp and Crane), relative to the mean edge length; 0 means no margin.	1e-6	relative
--name	NAME	Name of what is stored: the dataset, subject or surface, or the stem of the stored result (filters: <recording>_<verb>).	derived from the inputs	-
--no-average-reference		Do not re-reference the EEG gain rows and covariance to the average	off	-
--no-bad-segments		Keep the recording's marked bad segments in the estimate instead of leaving out the segments that touch them.	off (bad segments are left out)	-
--no-depth		Turn depth weighting off.	off (depth weighting on)	-
--no-detect-diagonal		Do not whiten a diagonal noise covariance by its diagonal alone	off	-
--no-eeg		Do not compute the EEG channels of the head model.	off (EEG is computed)	-
--no-gain		Do not store the head model's gain matrix.	off (the gain is stored)	-
--no-levi-civita		Do not write the Levi-Civita (transported) form of the vector tile summaries.	off (written)	-
--no-primary		Do not mark the added mesh as the subject's primary surface	off	-
--no-project-electrodes		Do not move the electrodes radially onto the fitted EEG sphere.	off (electrodes are projected)	-
--no-raw		Do not import the raw recording, only the condition's other data.	off	-
--no-remove-mean		Do not remove the mean (DC offset) of each block before computing or filtering.	off (mean removed)	-
--no-shard		Do not group chunks into shards when compacting	off	-
--no-small-files		Stop at the sharded layout; do not also move each subject's small files into its small-files database.	off	-
--node	PATH	Delete only this stored node of the subject, by store path, instead of the whole subject.	none (whole subject)	-
--noise-cov	NAME	Name of a stored noise covariance to use (alternative to --recording)	none	-
--normalisation	none|relative	Normalisation of the band power maps: none, or relative (each map divided by the source's total power over all bins up to Nyquist).	none	-
--normalize		Re-orthonormalise the eigenmodes afterwards (orthonormal in the mass matrix).	off	-
--of	NAME	The stored field, or band of a map statistic, that was measured	none (required)	-
--only	NAME	Import only this named anatomical image (repeatable).	all images	-
--order	N	Filters: Butterworth prototype order (the filter has 2N poles). Spatial band commands: degree of the Chebyshev series (0 chooses the smallest of 40, 80, ... 5120 within tolerance). measure query: sort order asc or desc of the rows.	filters 4; spatial 0 (automatic); measure query see the command's description	-
--order	N	Butterworth prototype order of the band-stop filter (the filter has 2N poles).	4	-
--order	N	Butterworth prototype order of the notch filter.	4	-
--order	asc|desc	Sort order of the rows by value.	see the command's description	-
--out	PATH	Output file (convert scan: the scan result written as a file) or output folder (measure export: where the table is written, made if absent).	none (measure export: required)	-
--over	NAME	Name of the partition (stored selection) measured over: statistic: the partition whose members were measured; null, repeat: the tile-level selection of the stored crossings; measure compute: the time tiling	statistic, measure compute: none (required); null, repeat: the one the recordings share	-
--overlap	FRACTION	Fraction by which successive Welch segments overlap.	0.5 (the Brainstorm option set)	fraction
--oversample	X	Oversampling of each filterbank member's own sample rate.	2.0	-
--package	STEM	Stem of the stored Hodge-Dirac eigenmode package to use	the one of the kernel's surface	-
--padding	mean|zeros	How the filter's edge transient is padded: with the mean of the signal's transient or with zeros.	mean	-
--pair	A,B	One pair of channels, by channel index (row A, column B), to read from the band-resolved cross-spectrum.	none	-
--pairs	K	Number of strongest quadrature pairs stored per band	3	pairs
--patterns	Q	Number of leading cycle patterns kept per spatial band.	3	-
--periodic		Use the periodic (DFT-even) form of the taper instead of the symmetric form.	off for the default and brainstorm sets (symmetric); on for scipy	-
--phase-of	current|stream	Signal whose phase is taken: the band's current vector field, or its scalar stream function	current	-
--preconditioner	ldlt|ic	Preconditioner of the flow's conjugate-gradient solve: ldlt (exact factor, built once) or ic (incomplete Cholesky).	ldlt	-
--prefer	protocol.mat|walk	Which reader to prefer for the Brainstorm protocol: its protocol.mat file or a walk of the folders.	automatic	-
--primary		Mark the added mesh as the subject's primary surface	on	-
--provenance	FILE	A JSON file of provenance to store with the imported maps.	none	-
--psd	NAME	The recording's stored Welch spectrum, whose shape gives the band's effective bandwidth.	none (required)	-
--rank-tolerance	single|double	Precision used to decide the rank of the noise covariance: single or double precision.	single	-
--reading	coherency|imag|phase|delay|psi	Quantity read from the stored bin-by-bin cross-spectrum (positive delay and psi: the row channel leads)	coherency	-
--recipe	NAME	Name of the recipe (the recorded set of choices) the measurements are made or read by.	measure compute: partition-<weight>; measure export: any	-
--recording	NAME	Name of the stored recording to read.	none (required)	-
--recording-a	NAME	First recording, whose stored tile crossings are used (repeat: one of the two compared)	none (required)	-
--recording-b	NAME	Second recording whose crossings are compared with the first (run-to-run repeat).	none	-
--reference-level	L	Spatial tile level whose power the blur patch is cut from.	6	level
--region	a:b,c:d	Read only this region of the array: a start:stop range per dimension.	the whole array	samples
--regularisation	ridge|diagonal|none	Regularisation of the noise covariance when whitening	ridge	-
--regulariser	connection|ambient	Smoothness term of the flow: connection (connection Laplacian on the signpost gauge) or ambient (Dirichlet energy of the ambient 3-vector field, Lefevre and Baillet).	connection	-
--replace		Replace an existing result of the same name instead of refusing; the old one is moved aside, not deleted.	off	-
)TBL",
R"TBL(--resample	HZ	Resample the converted recordings to this sampling rate	none (keep the original rate)	Hz
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
--scaling	density|spectrum|chain	How the spectrum is scaled: power per Hz, power per bin, or the chain scaling (measure export: stored|density|power|amplitude|amplitude-density|db|db-density, applied to the stored values).	density (measure export: stored)	-
--scaling	stored|density|power|amplitude|amplitude-density|db|db-density	How the stored values are scaled in the table.	stored	-
--seed	N	crossspectrum read: the channel index to read against every channel (alternative to --pair); wishart: seed of the random draws	wishart: 1; crossspectrum read: none	-
--segment-level	L	Welch segment of 2^L samples (an alternative to the segment length in seconds or samples).	none (the segment length option is used)	level
--segment-samples	N	Welch segment length in samples; wins over --segment-seconds.	derived from the 4 s segment	samples
--segment-seconds	SECONDS	Length of each Welch segment (under the tiles setting it is snapped to a power-of-two number of samples)	4	s
--selection	NAME	The stored joint Selection (space x time cells) whose measurement is read.	none (required)	-
--sensor-weight-exponent	P	Exponent of the distance weighting used to fit each sensor's overlapping sphere.	4	-
--set	NAME=VALUE	Set a pipeline parameter, NAME=VALUE with VALUE read as YAML (8, [15, 29], dSPM); repeatable.	none	-
--set	NAME=VALUE	Choose the reference whose settings are the starting point: tiles, brainstorm, scipy or nxr-chain; the options after it override.	tiles	-
--set	NAME=VALUE	Choose the reference whose settings are the starting point: tiles, brainstorm, scipy or nxr-chain; the options after it override.	tiles	-
--set	NAME=VALUE	Choose the reference whose settings are the starting point: tiles, brainstorm, scipy or nxr-chain; the options after it override.	tiles	-
--set	NAME=VALUE	Choose the reference whose settings are the starting point: tiles, brainstorm, scipy or nxr-chain; the options after it override.	tiles	-
--set	NAME=VALUE	Choose the reference whose settings are the starting point: tiles, brainstorm, scipy or nxr-chain; the options after it override.	tiles	-
--set	NAME=VALUE	Choose the reference whose settings are the starting point: tiles, brainstorm, scipy or nxr-chain; the options after it override.	tiles	-
--set	NAME=VALUE	Choose the reference whose settings are the starting point: tiles, brainstorm, scipy or nxr-chain; the options after it override.	tiles	-
--shape	morlet|meyer|power_partition	Band filter shape: morlet (Gaussian centred on the band, FWHM equal to its width) or meyer (wishart); pyramid build also accepts power_partition (band powers add to broadband).	morlet	-
--shift	absolute|relative	How the eigenvalue solve is shifted: at one absolute shift (reference method) or by a scale-aware relative ladder.	absolute	-
--side	auto|left|right	Hemisphere of a single-piece surface	auto	-
--sides	analytic|two-sided	Whether the time bank is one-sided (analytic, positive frequencies) or two-sided.	none (required)	-
--sigma	S	Absolute shift of the eigenvalue solve (with --shift absolute).	-0.001	1/m^2
--sign	positive|negative|magnitude	Which tile means form patches: high values, low values, or absolute value	positive	-
--signal	normal|ambient	Signal whose phase gradient is taken: ambient (the full current, its x y z components) or normal (the component along the surface normal).	ambient	-
--slice	N	Use spectrum slicing with windows of N modes; 0 means one solve.	0 (none)	modes
--snr	S	Assumed amplitude signal-to-noise ratio of the minimum-norm kernel	3	-
--snr-normalisation	all-channels|whitener-rank	Whether the signal-to-noise ratio is normalised over all channels or over the rank of the whitener.	all-channels	-
--source	NAME	The pipeline whose stored cells and events are read.	cortical_event_pairs	-
--source-path	PATH	Records the path of the source data in the dataset's provenance	none	-
--source-protocol	NAME	Record the name of the source protocol the dataset was made from.	none	-
--source-surface	NAME	Surface whose vertices are the sources of the head model.	the primary (cortex) surface	-
--source-tool	NAME	Records the tool that produced the source data in the dataset's provenance	none	-
--space-level	L	Level of the surface's tile ladder that gives the spatial tiles (-1 on measure query: the whole cortex).	measure query: the stored depth; others none (required)	level
--space-levels	L,...	Spatial tile levels of the surface to select over (all combinations with --time-levels).	none (required)	levels
--spatial-band	BAND	Spatial band: modes, weyl:J or weyl:all for the phase and cycle patterns (all modes, one Weyl octave J, each member of the Weyl bank); for measure query the whole-number index of a stored spatial band	phasegradient: none (the kernel as it is); cyclepattern: weyl:all; measure query: none	-
--spatial-shape	NAME	Shape of the spatial scale-space filter at each level (mexhat is the Mexican hat).	mexhat	-
--speed-bins	N	Number of log-spaced phase-speed bins of the speed spectrum.	40	bins
--speed-max	C	Largest phase speed of the speed histogram bins	from the spectrum (the largest 2 pi f / mu over modes and bins)	length/s
--speed-min	X	Smallest phase speed of the log-spaced speed bins.	0 (from the spectrum)	length/s
--sphere-cost	geometric|algebraic	Cost the head sphere is fitted with: distance to the sphere (geometric) or the linear closed-form fit (algebraic).	geometric	-
--stage	NAME|INDEX	Stage of the stored crossings to use, by name (burst, onset, within, offset) or by index	0 (the first stage, burst)	-
--star1-floor	X	Floor on the cotangent weights of the Hodge star; values at or below 0 mean no floor.	none	-
--start	SEC	Start of the span read, in seconds on the recording's clock; give together with --stop. Exclusive with --chunks.	none (one chunk)	s
--start-s	SECONDS	Time of the recording's first sample as seconds from local midnight, anchoring the time-tile levels	the anchor is the recording	s
--statistics	LIST	Comma-separated statistics written per subject in the exported table (e.g. mean,sd).	see the command's description	-
--stop	SEC	End of the span read, in seconds on the recording's clock (exclusive); give together with --start.	none (one chunk)	s
--store-format	51|52	Array storage layout of the new dataset (52 keeps small keys in a table)	52	-
--subject	NAME	The subject of the dataset to work on.	none (required by most commands; dataset compact: all subjects)	-
--subjects	A,B,...	Subjects to include, by name.	every subject of the dataset (pyramid reduce: required)	-
--summaries	LIST	Comma-separated summaries across the members, such as median, mad, percentile:75, mean, sd, peak, kurtosis, skewness, min, max	statistic: none (required); patches: count,sum,median,mean,sd,percentile:25,percentile:75,percentile:90	-
--summary		Print a one-line summary of the scan.	off	-
--surface	NAME	Surface to work on, by name.	the primary (cortex) surface (joint, scaleselection: the one the joint partitions use)	-
--surface-id	ID	The id of the surface the kernel was made on.	found from the kernel file	-
--symmetric		Use the symmetric form of the taper window instead of the periodic one	off (the window form follows --set)	-
--table	NAME	Name of the exported table.	none (required)	-
--template	SUBJECT	Template subject whose anatomical frame is carried over through the registered spheres.	none (the surface's own axes)	-
--template-surface	NAME	Surface of the template subject whose frame is carried over	the template's primary surface	-
--templates	DIR	A FreeSurfer subjects folder; writes the default subject (surfaces, parcellations, frames, group tree) with the dataset.	none	-
--threads	N	Workers for the kernel products; the result is identical for every count (0 = the machine's performance cores, 1 = serial).	0	threads
--through	NAME	Name of a stored 2-row operator (a rotation plane) over the channels: compute the two-sided rotary spectrum of its complex signal	none	-
--tile-levels	L1,L2,..	Levels of the tile ladder at which tile summaries of the maps are written (needs --tiles).	1 up to --tile-max-level (6)	levels
--tile-max-level	N	Write tile summaries for levels 1 to N (with --tiles).	6	level
--tile-pairs	on|off	Also compute the blocks between touching tiles (waves crossing from one tile into the next)	on	-
--tiles		Also write the maps' tile summaries per level of the tile ladder.	off	-
--time-bandwidth	B	Morse family: the time-bandwidth product of the wavelet.	60	-
--time-level	M	Level of the recording's constant-Q time ladder (joint compute: the time tiles of the cells; measure query: time rolled up to tiles of 2^M frames)	measure query: 0; joint compute: none (required)	level
--time-levels	M1,M2,..	Constant-Q time levels (each one's band is a pair's rhythm) crossed with the space levels.	none (required)	levels
--timing-of	FIELD,...	Also store the timing map (lags) of these fields of each pattern: stream, potential, normal.	none	-
--to	HZ	The sampling rate to resample to (import resample).	none	Hz
--tolerance	T	lowpass, notch: IIR look-behind tail left out, relative to the response's L1 norm; pointspread, cellquadrature, patches: largest gain error accepted for the Chebyshev series; opticalflow: tail of the filters' impulse response the margin may leave out	filters: 1e-15; pointspread, cellquadrature, patches: 0.01; opticalflow: from the recording	-
--transform	auto|eigen|chebyshev|exact	How the spatial band is applied: auto (stored eigenmodes; else computed, else Chebyshev announced), eigen, chebyshev (series in the pencil) or exact (dense modes, at most 4096 vertices).	auto	-
--transient-samples	K	Number of samples at each edge treated as the filter transient (instead of the shortest span holding 99 percent of the impulse response's energy).	from the energy rule	samples
--transition	HZ	Width of each transition band of the FIR filter	the reference's rule (Brainstorm)	Hz
--tree	subject|group	The tree the leaves belong to: the surface's own ladder, or the dataset's group tree through the registered sphere (the same leaves in every subject).	subject	-
--trk	FILE	A fibre-tract file (.trk) to import with the templates.	none	-
--types	TYPE,...	Channel types to use, e.g. "MEG MAG,MEG GRAD,EEG" (only good channels of these types).	MEG, MEG MAG, MEG GRAD, EEG, SEEG, ECOG	-
--vertices	N	Number of vertices, spread evenly over the surface, where unit currents are placed	20	vertices
--voices	N	Filter-bank members per octave (constantq: time-ladder voices; opticalflow, speedspectrum: itersine voices).	4 for filter banks; constantq see the command's description	voices/octave
--weight	none|barycentric|voronoi	How the vertices of each member are weighted (measure compute: none is the plain vertex mean, barycentric or voronoi weight by area); phasegradient: power|coherence, the edge weight of a weighted fit.	measure compute: none, must be given; phasegradient: power	-
)TBL",
R"TBL(--weights	area|none	cellquadrature: weight vertices by their area or equally; crossspectrum read: weight the delay and phase-slope fits by coherence or not	cellquadrature: area; crossspectrum read: coherence	-
--whitener-form	symmetric|pca	Form of the noise whitener: symmetric or PCA.	symmetric	-
--window	hann|hamming|blackman|rectangular	Taper applied to each Welch segment.	hamming (default and brainstorm sets); hann (scipy set)	-
--window-seconds	SECONDS	Window length for the analytic estimator or the cycle patterns, snapped to the nearest time-tile level (alternative to --level)	crossspectrum: the finest of the bands' windows; cyclepattern: 4	s
--windows	NAME	The stored analytic cross-spectrum windows (named <stem>_<band>_windows) the bursts are read from.	none (required)	-
)TBL",
R"TBL(@ create dataset
old: dataset create
summary: Create an empty dataset in a datastore, optionally with its default subject (surfaces, frames, group tree).
opts: --atlas-depth --datastore* --default --lc-levels --modes --name* --source-path --source-protocol --source-tool --store-format --templates
)TBL",
R"TBL(@ list datasets
old: dataset list
summary: List the datasets of a datastore.
opts: --datastore*
)TBL",
R"TBL(@ show dataset
old: dataset show
summary: Show a dataset: its subjects by kind, its source and its creation date.
opts: --dataset*
)TBL",
R"TBL(@ delete dataset
old: dataset delete
summary: Delete a dataset from its datastore.
opts: --datastore* --name*
)TBL",
R"TBL(@ recover dataset
old: dataset recover
summary: Finish or roll back an interrupted write in a dataset.
opts: --dataset*
)TBL",
R"TBL(@ compact dataset
old: dataset compact
summary: Pack a dataset's small files and shards to save space (a dry run shows what would change).
opts: --dataset* --dry-run --no-shard --no-small-files --subject
)TBL",
R"TBL(@ create subject
old: subject create
summary: Create a subject (or a template or group subject) in a dataset.
opts: --dataset* --kind --name*
)TBL",
R"TBL(@ list subjects
old: subject list
summary: List the subjects of a dataset.
opts: --dataset*
)TBL",
R"TBL(@ show subject
old: subject show
summary: Show how many manifolds, fields, operators and selections a subject holds.
opts: --dataset* --subject*
)TBL",
R"TBL(@ delete subject
old: subject delete
summary: Delete a subject, or one of its nodes, from a dataset.
opts: --dataset* --node --subject*
)TBL",
R"TBL(@ add surface
old: surface add
summary: Add a mesh file (OBJ, PLY or STL) to a subject as a surface.
pos: FILE
opts: --dataset* --name --no-primary --primary --scale --subject*
)TBL",
R"TBL(@ list recordings
old: __list recordings
summary: List the recordings of a subject.
opts: --dataset* --subject*
)TBL",
R"TBL(@ list kernels
old: __list kernels
summary: List the inverse kernels of a subject.
opts: --dataset* --subject*
)TBL",
R"TBL(@ list results
old: __list results
summary: List the fields and statistics the computations stored for a subject.
opts: --dataset* --subject*
)TBL",
R"TBL(@ list pipelines
old: run --list
summary: List the pipeline definitions and the steps they are made of.
unprobed: yes
opts: 
)TBL",
R"TBL(@ list spatial-bands
old: pyramid spatial-bands
summary: List a surface's spatial bands (one per level of its tile ladder).
opts: --dataset* --subject* --surface
)TBL",
R"TBL(@ import list
old: convert list
summary: List the subjects of a Brainstorm protocol.
pos: PROTOCOL
opts: 
)TBL",
R"TBL(@ import scan
old: convert scan
summary: Scan a Brainstorm protocol and describe what it holds.
pos: PROTOCOL
opts: --out --prefer --summary
)TBL",
R"TBL(@ import subject
old: convert subject
summary: Import a Brainstorm subject's anatomy into a dataset.
pos: PROTOCOL
opts: --dataset* --resample --subject*
)TBL",
R"TBL(@ import template
old: convert template
summary: Import a Brainstorm protocol's template subject into a dataset.
pos: PROTOCOL
opts: --dataset* --surface+
)TBL",
R"TBL(@ import group
old: convert group
summary: Import a Brainstorm protocol's group subject into a dataset.
pos: PROTOCOL
opts: --dataset*
)TBL",
R"TBL(@ import templates
old: convert templates
summary: Import the fsaverage reference templates (and optional fibre tracts) into a dataset.
pos: TEMPLATES_DIR
opts: --dataset* --subject --surface --trk
)TBL",
R"TBL(@ import surface
old: convert surface
summary: Import Brainstorm tessellation files as surfaces of a subject.
opts: --bst-root --dataset* --file+ --subject*
)TBL",
R"TBL(@ import mri
old: convert mri
summary: Import an anatomical MRI of a subject.
opts: --anat* --dataset* --only+ --subject*
)TBL",
R"TBL(@ import condition
old: convert convert
summary: Import a Brainstorm condition (its recordings, channels and head models) into a dataset.
pos: PROTOCOL
opts: --all-surfaces --channel-file --condition* --dataset* --extra-surface+ --include-imported --no-gain --no-raw --recording --resample --subject* --surface
)TBL",
R"TBL(@ import inverse
old: convert inverse
summary: Import a Brainstorm inverse kernel as an operator of a subject.
opts: --acquisition* --bst-root* --channel-file --channels-id --dataset* --file* --forward-id --name --subject* --surface-id
)TBL",
R"TBL(@ import map
old: convert map
summary: Import Brainstorm source or matrix maps as fields of a subject.
opts: --acquisition --bst-root* --dataset* --file+* --provenance --replace --subject*
)TBL",
R"TBL(@ import resample
old: convert resample
summary: Resample a recording to a new sampling rate, as a new recording.
opts: --dataset* --name --recording* --subject* --to*
)TBL",
R"TBL(@ import badsegments
old: convert badsegments
summary: Import a recording's bad segments from a file.
opts: --dataset* --from* --replace --subject*
)TBL",
R"TBL(@ import events
old: convert events
summary: Import a recording's events from a file.
opts: --dataset* --from* --replace --subject*
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
summary: Read coherency, phase, delay or phase-slope index from a stored band-resolved cross-spectrum.
opts: --dataset* --name* --pair --reading --seed --subject* --weights
)TBL",
R"TBL(@ query measurements
old: measure query
summary: Query a stored measurement pyramid at any spatial and temporal scale.
opts: --band --dataset* --field --limit --max-value --measure* --min-value --order --selection* --space-level --spatial-band --subject* --time-level
)TBL",
R"TBL(@ export measurements
old: measure export
summary: Export stored measurements as a table for statistics (a row per subject), with its dictionary and provenance.
opts: --dataset* --exclude --fields* --keep-all --labelling* --layout --out* --recipe --scaling --statistics --subjects --table*
)TBL",
R"TBL(@ reduce pyramid
old: pyramid reduce
summary: Reduce several subjects' group-tree pyramids into the group's pyramid.
opts: --dataset* --depth --kernels --subjects*
)TBL",
R"TBL(@ run
alias: no
old: run
summary: Run a pipeline definition (a YAML file or one of the built-in definitions) on a subject.
pos: PIPELINE
span: yes
opts: --block-frames --block-samples --chunks --dataset --dry-run --replace --set+ --start --stop --subject
)TBL",
R"TBL(@ compute noisecov
old: noisecov compute
group: Source imaging
summary: The noise covariance of a recording's good channels.
span: yes
opts: --block-seconds --chunks --dataset* --divisor --name --no-remove-mean --recording* --replace --start --stop --subject* --types
)TBL",
R"TBL(@ compute inverse
old: inverse compute
group: Source imaging
summary: The depth-weighted minimum-norm inverse kernel, from a noise covariance and a head model.
span: yes
opts: --block-seconds --chunks --dataset* --depth-channels --depth-exponent --depth-limit --depth-limit-on --depth-norm --divisor --forward --joint-types --mne --name --no-average-reference --no-depth --no-detect-diagonal --no-remove-mean --noise-cov --rank-tolerance --recording --regularisation --replace --ridge-fraction --ridge-over-rank --snr --snr-normalisation --start --stop --subject* --types --whitener-form
)TBL",
R"TBL(@ compute headmodel
old: headmodel compute
group: Source imaging
summary: The sphere head model (gain matrix) for a subject's cortex, MEG and EEG.
opts: --berg-dipoles --berg-starts --berg-terms --channels --dataset* --eeg-conductivities --eeg-radii --eeg-series --eeg-series-terms --eeg-surface --meg --meg-surface --mne-eeg-layers --name --no-eeg --no-project-electrodes --replace --sensor-weight-exponent --source-surface --sphere-cost --subject*
)TBL",
R"TBL(@ compute pointspread
old: pointspread compute
group: Source imaging
summary: The instrument's blur: unit currents at sample vertices through the head model and inverse kernel.
opts: --dataset* --forward --fraction --kernel --levels --list --modes --name --order --recording* --reference-level --replace --spatial-shape --subject* --tolerance --transform --vertices
)TBL",
R"TBL(@ compute eigenmodes
old: eigen solve
group: Geometry
summary: The Hodge-Dirac eigenmodes of a stored surface (cached).
opts: --dataset* --flip-eps --modes --mollify --normalize --replace --shift --sigma --slice --star1-floor --subject* --surface
)TBL",
R"TBL(@ compute frames
old: frames compute
group: Geometry
summary: The canonical (north, west) and anatomical frames of a surface.
opts: --convention --dataset* --max-condition --min-projection --replace --side --subject* --surface --template --template-surface
)TBL",
R"TBL(@ compute tiles
old: tiles compute
group: Geometry
summary: The dyadic equal-area tile ladder of a surface.
opts: --dataset* --depth --replace --subject* --surface
)TBL",
R"TBL(@ compute constantq
old: constantq compute
group: Time
summary: The constant-Q tiling of a time line (a ladder of dyadic windows).
opts: --all --cycles --dataset* --fmax --fmin --line --method --recording --subject* --voices
)TBL",
R"TBL(@ compute frametiling
old: frametiling compute
group: Time
summary: The frame tiling on the time tower: the cycles of one tower level over a recording.
opts: --band --centre --dataset* --frame-level --frames-per-cycle --recording* --replace --start-s --subject*
)TBL",
R"TBL(@ compute joint
old: joint compute
group: Time
summary: The joint partition of space and time: cells of a spatial tile level by a time tile level.
opts: --dataset* --recording* --replace --space-level* --subject* --surface --time-level*
)TBL",
R"TBL(@ compute lowpass
old: lowpass
group: Time
summary: A zero-phase low-pass filter, written as a new recording.
opts: --bandwidth --dataset* --fpass --frequencies --fstop --harmonics --line --name --no-remove-mean --order --padding --recording* --ripple --subject* --tolerance --transient-samples --transition
)TBL",
R"TBL(@ compute highpass
old: highpass
group: Time
summary: A zero-phase high-pass filter, written as a new recording.
opts: --bandwidth --dataset* --fpass --frequencies --fstop --harmonics --line --name --no-remove-mean --order --padding --recording* --ripple --subject* --tolerance --transient-samples --transition
)TBL",
R"TBL(@ compute bandpass
old: bandpass
group: Time
summary: A zero-phase band-pass filter, written as a new recording.
opts: --bandwidth --dataset* --fpass --frequencies --fstop --harmonics --line --name --no-remove-mean --order --padding --recording* --ripple --subject* --tolerance --transient-samples --transition
)TBL",
R"TBL(@ compute bandstop
old: bandstop
group: Time
summary: A zero-phase band-stop filter, written as a new recording.
opts: --bandwidth --dataset* --fpass --frequencies --fstop --harmonics --line --name --no-remove-mean --order --padding --recording* --ripple --subject* --tolerance --transient-samples --transition
)TBL",
R"TBL(@ compute notch
old: notch
group: Time
summary: A zero-phase notch filter (line noise and harmonics), written as a new recording.
opts: --bandwidth --dataset* --fpass --frequencies --fstop --harmonics --line --name --no-remove-mean --order --padding --recording* --ripple --subject* --tolerance --transient-samples --transition
)TBL",
R"TBL(@ compute psd
old: psd compute
group: Spectra
summary: The Welch power spectral density of a recording, bad segments left out.
span: yes
opts: --chunks --dataset* --detrend --name --no-bad-segments --overlap --periodic --recording* --replace --scaling --segment-level --segment-samples --segment-seconds --set --start --stop --subject* --symmetric --through --window
)TBL",
R"TBL(@ compute sourcepsd
old: sourcepsd compute
group: Spectra
summary: The band power maps of the sources (the Welch PSD of the sources, grouped in bands).
span: yes
opts: --bands --chunks --dataset* --detrend --kernel --name --no-bad-segments --normalisation --overlap --periodic --recording* --replace --scaling --segment-level --segment-samples --segment-seconds --set --start --stop --subject* --symmetric --window
)TBL",
R"TBL(@ compute sourcetensor
old: sourcetensor compute
group: Spectra
summary: The band cross-spectral tensor of the current, its Stokes quaternion, normal power and share.
span: yes
opts: --bands --chunks --dataset* --detrend --kernel --name --no-bad-segments --normalisation --overlap --periodic --recording* --replace --scaling --segment-level --segment-samples --segment-seconds --set --start --stop --subject* --symmetric --window
)TBL",
R"TBL(@ compute crossspectrum
old: crossspectrum compute
group: Spectra
summary: The sensor cross-spectral density per band (Welch, or per time window with the analytic estimator).
span: yes
opts: --bands --channels-of --chunks --dataset* --detrend --estimator --level --name --no-bad-segments --overlap --periodic --recording* --replace --resolve --scaling --segment-level --segment-samples --segment-seconds --set --start --stop --subject* --symmetric --types --window --window-seconds
)TBL",
R"TBL(@ compute quadrature
old: quadrature compute
group: Spectra
summary: The quadrature spectrum of the sources per band: fraction, energy map and leading pairs.
span: yes
opts: --bands --chunks --dataset* --detrend --kernel --name --no-bad-segments --no-levi-civita --overlap --pairs --periodic --recording* --replace --scaling --segment-level --segment-samples --segment-seconds --set --start --stop --subject* --symmetric --tile-levels --tile-max-level --tiles --window
)TBL",
R"TBL(@ compute hodgepower
old: hodgepower compute
group: Spectra
summary: The band power of the Hodge-Dirac eigenmode coefficients of a free-orientation kernel.
span: yes
opts: --bands --chunks --dataset* --detrend --kernel --modes --name --no-bad-segments --overlap --package --periodic --recording* --replace --scaling --segment-level --segment-samples --segment-seconds --set --start --stop --subject* --symmetric --window
)TBL",
R"TBL(@ compute speedspectrum
old: speedspectrum compute
group: Spectra
summary: The speed spectrum: mode energy per phase speed from the joint eigenvalue and frequency bank.
span: yes
opts: --anchor --centres --chunks --cycles --dataset* --detrend --family --fmax --fmin --gain --gamma --graph-family --graph-filters --graph-voices --half-width --kernel --limits --modes --name --no-bad-segments --overlap --package --periodic --recording* --replace --scaling --segment-level --segment-samples --segment-seconds --set --sides --speed-bins --speed-max --speed-min --start --stop --subject* --symmetric --time-bandwidth --voices --window
)TBL",
R"TBL(@ compute phasegradient
old: phasegradient compute
group: Cortical flow and patterns
summary: The phase gradient of a band from a stored cross-spectrum: phase current, power, speed and travelling index.
opts: --cross-spectrum* --dataset* --defects --depth --edge-lag --field --fit --fold-edges --kernel --name --package --phase-of --replace --signal --spatial-band --subject* --weight
)TBL",
R"TBL(@ compute cyclepattern
old: cyclepattern compute
group: Cortical flow and patterns
summary: The most consistent cycle patterns of the current from a cross-spectrum per window.
opts: --cross-spectrum* --dataset* --kernel --level --name --package --patterns --replace --spatial-band --subject* --timing-of --window-seconds
)TBL",
R"TBL(@ compute opticalflow
old: opticalflow compute
group: Cortical flow and patterns
summary: Optical flow (Horn-Schunck) of the source estimate: where and how fast the activity moves.
span: yes
opts: --alpha --anchor --centres --chunks --cycles --datafit --dataset* --depth --family --flow-tolerance --frame-level --freq --gain --gamma --half-width --input --kernel --limits --max-iterations --method --name --no-levi-civita --oversample --package --preconditioner --recording* --regulariser --replace --resolution --scale-family --scale-filters --scale-voices --start --start-s --stop --subject* --tile-levels --tile-max-level --tiles --time-bandwidth --tolerance --voices
)TBL",
R"TBL(@ compute cellquadrature
old: cellquadrature compute
group: Cortical flow and patterns
summary: Quadrature (travelling vs standing) per joint cell and per burst, from the stored cross-spectrum.
opts: --bursts --cross-spectrum* --dataset* --joint --kernel --modes --order --replace --space-level* --subject* --tile-pairs --tolerance --transform --weights
)TBL",
R"TBL(@ compute patches
old: patches compute
group: Cortical flow and patterns
summary: Patches on a static cortical map: where it stands out from its surroundings, at the size of its tiles.
opts: --dataset* --enter-sd --exit-sd --level* --map* --max-area --min-area --modes --order --replace --sign --subject* --summaries --tolerance --transform
)TBL",
R"TBL(@ compute scaleselection
old: scaleselection compute
group: Cortical flow and patterns
summary: Each event's own size and rhythm: the best spatial and temporal level by standardised response.
opts: --dataset* --min-link --name --recording* --replace --response --source --space-levels* --subject* --surface --time-levels*
)TBL",
R"TBL(@ compute pyramid
old: pyramid build
group: Measurements
summary: The measurement pyramid of a recording's cortical current, from space and time tiles.
span: yes
opts: --bands --block-samples --chunks --dataset* --depth --frame-level --halo-seconds --keep-bad-segments --kernel --recording* --replace --shape --start --start-s --stop --subject* --threads --tree
)TBL",
R"TBL(@ compute measure
old: measure compute
group: Measurements
summary: Stored fields measured over a partition's members, with their recipe.
span: yes
opts: --chunks --dataset* --fields* --gradient --labelling --level --measures --over --recipe --replace --start --stop --subject* --weight
)TBL",
R"TBL(@ compute statistic
old: statistic compute
group: Measurements
summary: A stored measurement summarised over its partition's members.
opts: --combine --dataset* --ddof --measure* --members-own --name --of* --over* --replace --subject* --summaries*
)TBL",
R"TBL(@ compute wishart
old: wishart compute
group: Nulls
summary: The Wishart null of the second-moment measures of bursts.
opts: --band* --bursts* --dataset* --draws --psd* --recording* --replace --seed --shape --subject* --windows*
)TBL",
R"TBL(@ compute spatialnull
old: spatialnull compute
group: Nulls
summary: The lag-free null of the spatial events: draws with early stopping, per-vertex p and q.
span: yes
seconds: no
opts: --batch --chunks --dataset* --events --exceedances=--h --max-draws --min-draws --recording* --replace --set+ --subject*
)TBL",
R"TBL(@ compute null
old: null compute
group: Nulls
summary: The lag-free null's tests of the vortex cores' crossings, per tile.
span: yes
seconds: no
opts: --batch --chunks --dataset* --exceedances=--h --max-draws --min-crossings --min-draws --over --recording-a* --recording-b --replace --set+ --stage --subject*
)TBL",
R"TBL(@ compute repeat
old: repeat compute
group: Nulls
summary: The vortex cores' run-to-run repeat: two recordings' stored tile crossings compared.
opts: --dataset* --min-crossings --name --over --recording-a* --recording-b* --replace --stage --subject*
)TBL",
  };
}
