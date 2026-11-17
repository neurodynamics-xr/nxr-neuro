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
R"TBL(verb run	Run a pipeline on a subject: a built-in (psd, crossspectrum, phasegradient, pyramid, ...), a built-in definition, or a definition file
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
)TBL",
R"TBL(--acquisition	NAME	The Brainstorm acquisition (condition) the kernel or map belongs to.	none	-
--acquisition@import inverse	NAME	The Brainstorm acquisition (condition) the kernel belongs to; it names the session the kernel is stored under.	required	-
--acquisition@import map	NAME	The Brainstorm acquisition (condition) the map belongs to, used to find a map's surface or atlas when its file does not name one.	none	-
--all		Do the time line of every recording in the subject (alternative to --recording and --line)	off	-
--all-surfaces		Import every surface of the subject, not only the ones the condition's head model uses.	off	-
--all-surfaces@import condition		Import every surface of the subject's anatomy, not only the surface of the condition's head model and the extra surfaces.	off	-
--all@run constantq		Make the ladder for the time line of every recording in the subject (alternative to --recording and --line).	off	-
--alpha	X	Horn-Schunck smoothness weight of the optical flow; vary it and report the range.	0.01 for the connection regulariser, 1 for ambient	-
--alpha@run opticalflow	X	Horn-Schunck smoothness weight of the flow; vary it and report the range, since a larger weight smooths the flow.	0.01 for the connection regulariser, 1 for ambient	-
--anat	FILE	The anatomical MRI file to import.	none	-
--anat@import mri	FOLDER	The subject's anatomical folder of the source protocol, whose MRI images are imported into the subject.	required	-
--anchor	HZ	Itersine and similar time filterbanks: the anchor frequency the member spacing is built from.	1.0	Hz
--anchor@run opticalflow	HZ	Itersine family only: the anchor frequency from which the members' log-frequency spacing is built.	1	Hz
--anchor@run speedspectrum	HZ	Itersine time bank: the frequency the voice grid is anchored at	1	Hz
--array	SUBJECT PATH	Read one array: the subject and the array's path in the dataset (prints its bytes, or writes them with --out).	none	-
--array@read dataset	SUBJECT PATH	Read one array: the subject's name and the array's path inside the subject; prints its shape, data type and checksum. Without --array every array of the dataset is listed.	none (list all arrays)	-
--atlas-depth	LEVEL	Leaf level of the group tree that serves as the dataset's default atlas (needs --templates)	8	level
--atlas-depth@create dataset	LEVEL	Leaf level of the group tree that serves as the dataset's default atlas; written with the default subject, so it needs --templates.	8	level
--band	LO,HI	The frequency band in hertz (frametiling: its centre sets the frame length; wishart: the band of the stored windows; measure query: INDEX picks one band by its index, a whole number).	none	Hz
--band@run frametiling	LO,HI	The band the frames are made for: its centre sets the frame level, and the band is recorded with the tiling.	none	Hz
--band@run wishart	LO,HI	The frequency band the stored windows are of.	required	Hz
--band@query measurements	N	The band (the row's within-code) to read when the pyramid's joint selection holds several.	none (all bands, or the only one)	-
--bands	NAME:LO-HI,...	The frequency bands to compute, each a name and its low and high edge (the mean over the bins from LO to HI, inclusive); pyramid build takes LO-HI,... octave bands without names.	Brainstorm's bands (delta 2-4, theta 5-7, alpha 8-12, beta 15-29, gamma1 30-59, gamma2 60-90); pyramid build: octaves 1-2 to 32-64	Hz
--bands@run crossspectrum	NAME:LO-HI,...	Frequency bands to estimate, each a name and its low and high edge in hertz; each band's cross-spectrum is the mean over its bins (LO to HI inclusive).	delta 2-4, theta 5-7, alpha 8-12, beta 15-29, gamma1 30-59, gamma2 60-90	Hz
--bands@run hodgepower	NAME:LO-HI,...	The frequency bands to compute power in, as name:lo-hi in Hz, e.g. alpha:8-12.	delta 2-4, theta 5-7, alpha 8-12, beta 15-29, gamma1 30-59, gamma2 60-90	Hz
--bands@run pyramid	LO-HI,...	Frequency bands to build, as comma-separated LO-HI edges (half-open [LO, HI)).	octaves 1-2 to 32-64 Hz	Hz
--bands@run quadrature	NAME:LO-HI,...	Frequency bands to analyse, each a name and edges; the band's cross-spectrum is the mean of the bins from LO to HI inclusive.	Brainstorm's bands (delta 2-4, theta 5-7, alpha 8-12, beta 15-29, gamma1 30-59, gamma2 60-90)	Hz
--bands@run sourcepsd	NAME:LO-HI,...	Frequency bands to map, each a name with its low and high edge (the mean over the bins from LO to HI, inclusive).	Brainstorm's bands: delta 2-4, theta 5-7, alpha 8-12, beta 15-29, gamma1 30-59, gamma2 60-90	Hz
--bands@run sourcetensor	NAME:LO-HI,...	The frequency bands to compute, each a name and its low and high edge in Hz (the mean over the bins from LO to HI, inclusive).	Brainstorm's bands (delta 2-4, theta 5-7, alpha 8-12, beta 15-29, gamma1 30-59, gamma2 60-90)	Hz
--bandwidth	HZ	Notch width: the 3 dB bandwidth of each notch	1	Hz
--bandwidth@run bandpass	HZ	Notch width; it belongs to the notch filter and has no effect on a bandpass.	1	Hz
--bandwidth@run bandstop	HZ	Notch width; used by notch only and not by bandstop (its edges are --fstop).	1 (notch only)	Hz
--bandwidth@run highpass	HZ	Not used by highpass: it is a FIR design; the notch's 3 dB bandwidth is for the notch command.	1	Hz
--bandwidth@run lowpass	HZ	Notch width; not used by lowpass (notch only).	1 (notch only)	Hz
--bandwidth@run notch	HZ	The 3-dB width of each notch.	1	Hz
--batch	N	Number of null draws made at a time before the early-stopping test (Besag-Clifford).	20	draws
--batch@run null	N	Draws are made in batches of this size	20	draws
--batch@run spatialnull	N	Draws made between checks of the stopping rule.	20	draws
--berg-dipoles	J	Number of equivalent dipoles in the Berg-Scherg approximation of the three-shell EEG sphere.	3	-
--berg-dipoles@run headmodel	N	The number of equivalent dipoles J in the Berg-Scherg fit of the EEG sphere.	3	-
--berg-starts	N	Number of simplex starts when fitting the three-dipole Berg-Scherg approximation of the EEG sphere model; the best fit is kept	12	-
--berg-starts@run headmodel	N	The number of simplex starting points in the Berg-Scherg fit; the best minimum is kept.	12	-
--berg-terms	N	Number of series terms fitted when approximating the multi-shell EEG model by the Berg-Scherg parameters.	200	terms
--berg-terms@run headmodel	N	The number of series terms the Berg-Scherg fit is made to.	200	-
--block-frames	N	How many frames the steps after a frame tiling process at once; limits memory only, the result does not change (0 = the whole chunk's frames).	256	frames
--block-frames@run	N	How many frames the steps after a frame tiling process at once; limits memory only, the result does not change (0 = the whole chunk's frames).	256	frames
--block-samples	N	Samples processed at a time, which sets memory use only and does not change the result (run: width for the steps after the last filter, 0 = whole chunk; pyramid build: samples projected at a time)	run: 1024; pyramid build: 512 (fixed-orientation kernel) or 128 (free)	samples
--block-samples@run pyramid	N	Samples projected through the kernel at a time; sets memory use only, not the result.	512 (fixed-orientation kernel) or 128 (free orientation)	samples
--block-samples@run	N	Width in samples of the blocks that flow through the steps after the last filter; limits memory only, the result does not change (0 = the whole chunk).	1024	samples
--block-seconds	SECONDS	Length of the blocks whose own mean is removed before the covariance is formed (noisecov: noise window; inverse: when computing the noise covariance from a recording).	one block (a single mean for the whole window)	s
--block-seconds@run inverse	SECONDS	With --recording: split the noise-covariance window into blocks of this length, each with its own mean removed.	one block for the whole window	s
--block-seconds@run noisecov	SECONDS	Split the chosen span into blocks of this length; each block has its own mean removed before the covariance is summed.	none (the span is one block)	s
--bst-root	PATH	Root folder of the Brainstorm protocol, used to resolve the surface files named by --file.	none (empty)	-
--bst-root@import inverse	PATH	Root folder of the Brainstorm protocol, used to resolve the kernel's channel, surface and head-model links.	required	-
--bst-root@import map	DIR	The Brainstorm protocol folder the map files are read from, with the anatomy folders the maps' surfaces and atlases are found in.	required	-
--bst-root@import surface	PATH	Root folder of the Brainstorm protocol, used to find the surface's registration and atlas files next to each --file.	none (empty)	-
--bursts	NAME	Name of a stored set of burst time spans (wishart: required, each burst is tested; cellquadrature: optional, also computed per burst)	none	-
--bursts@run cellquadrature	NAME	Stored set of burst time spans on the recording's line; when given, quadrature is also computed per burst.	none (cells only)	-
--bursts@run wishart	NAME	Stored set of burst time spans on the recording's line (tile-aligned); each burst is tested against the null.	required	-
--centre	HZ	Centre frequency of the band from which the frame length is derived (the coarsest tower level whose frame is no longer than 1/(centre x frames-per-cycle)).	none	Hz
--centre@run frametiling	HZ	The centre frequency the frame level is derived from (the middle of --band when that is given).	none	Hz
--centres	HZ,...	Centre frequencies of the filterbank members (Morlet, Morse, Amor and Bump families); for Morlet, empty means the single member at --freq (opticalflow).	none	Hz
--centres@run opticalflow	HZ,...	Centre frequencies of the filter-bank members (Morlet, Morse, Amor and Bump); for Morlet, empty means the single member at --freq.	none	Hz
--centres@run speedspectrum	HZ,HZ,...	Morlet, Morse, Amor and Bump time banks: the member centre frequencies	none	Hz
--channel-file	FILE	The channel file to use instead of the condition's own.	the condition's channel file	-
--channel-file@import condition	FILE	Channel file (in the condition folder) to use for the condition's recordings instead of the one the condition names.	none (the condition's own channel file)	-
--channel-file@import inverse	FILE	The channel file to take the kernel's channels from instead of the one found for the kernel.	found from the kernel's condition	-
--channels	NAME	Name of the sensor channel set the head model is computed for	the subject's only channel set with a type labelling	-
--channels-id	ID	The id of the channels the kernel was made for.	found from the kernel file	-
--channels-id@import inverse	ID	Id of the stored channels the kernel was made for (a row of the subject).	found from the subject's session channels	-
--channels-of	KERNEL	Use the channels of this stored inverse kernel, in its order, for the cross-spectrum; an alternative to --types.	none (all good channels of the chosen types)	-
--channels-of@run crossspectrum	KERNEL	Use the channels of this stored inverse kernel, in its order, so the cross-spectrum matches the kernel; an alternative to --types.	none (the good channels of the chosen types)	-
--channels@run headmodel	NAME	The channels manifold (sensor positions and types) the gain is computed for.	the subject's only points manifold with a type labelling	-
--chunks	K|A-B|all	Which part of the recording to read: one chunk K, a range of chunks A-B, or all chunks looped one after another (a chunk is 2^15 samples of every channel). Exclusive with --start/--stop.	0 (one chunk)	chunks
--combine	each|pool|mean	How values are combined before summarising: each row separately, all rows pooled, or the mean over rows per member	each	-
--combine@run statistic	each|pool|mean	How values are combined before summarising: each row (channel or vertex) separately, every value of every row pooled, or the mean over rows per member.	each	-
--condition	NAME	The Brainstorm condition to import.	none	-
--condition@import condition	NAME	The condition (acquisition) of the protocol to import; it is stored as a session of the subject.	required	-
)TBL",
R"TBL(--convention	scs|ras	Axes in which the surface's own coordinates are given: scs (anterior +x, superior +z, left +y) or ras (anterior +y, superior +z, left -x).	scs	-
--convention@run frames	scs|ras	Axes in which the surface's own coordinates are given: scs (anterior +x, superior +z, left +y) or ras (anterior +y, superior +z, left -x); used when no template is given.	scs	-
--cross-spectrum	NAME	Name of the stored sensor cross-spectrum (from crossspectrum compute) to read.	none (required)	-
--cross-spectrum@run cellquadrature	NAME	Stored analytic cross-spectrum windows (<stem>_<band>_windows from crossspectrum compute --estimator analytic) to read; its tile level must match the joint's time level.	required	-
--cross-spectrum@run cyclepattern	NAME	Name of the stored per-window sensor cross-spectrum series (<stem>_<band>_windows, from crossspectrum compute --estimator analytic) to read.	required	-
--cross-spectrum@run phasegradient	NAME	Name of the band's stored cross-spectrum whose phase is read.	required	-
--cycles	C	Cycles of the band's lowest frequency in each window (constantq) or wavelet width as centre frequency / C (speedspectrum, opticalflow)	constantq: set by --method (about 6.8 at 1200 Hz); speedspectrum, opticalflow: 6	cycles
--cycles@run constantq	C	Cycles of a band's lowest frequency held by its window (window rounded up to a power of two of samples).	set by --method (hertz: about 6.83 at 1200 Hz; cycles: 8)	cycles
--cycles@run opticalflow	C	Morlet family only: the wavelet's width is its centre frequency divided by this number of cycles.	6	cycles
--cycles@run speedspectrum	C	Morlet time bank: cycles per member (width = centre / cycles)	6	cycles
--datafit	auto|gauge|triangle	Data term of the Horn-Schunck flow: auto picks the regulariser's own (gauge for connection, triangle for ambient); gauge is connection only; triangle uses per-triangle finite elements.	auto	-
--datafit@run opticalflow	auto|gauge|triangle	Data term of the Horn-Schunck equation: auto (gauge for the connection regulariser, triangle for ambient), gauge (connection only), or per-triangle finite elements.	auto	-
--dataset	PATH	The dataset to work on: the path of its folder (a dataset is addressed by its own path; the folder of a one-database dataset, or of a catalogue of subject packages).	required	-
--datastore	DIR	The folder that holds the dataset folders (their parent); the dataset is then DIR/NAME.	none (required)	-
--dataset@create dataset	PATH	The folder of the new dataset (made if absent): it holds the catalogue and, by default, each subject's package.	required (unless --datastore)	-
--datastore@create dataset	DIR	Deprecated: makes the one-database form, the dataset DIR/NAME (needs --name); `split dataset` turns it into a catalogue. Use --dataset.	none	-
--datastore@delete dataset	PATH	The folder that holds the folder of the dataset to delete (its parent).	required	-
--datastore@list datasets	DIR	The folder that holds the dataset folders to list.	required	-
--ddof	N	Delta degrees of freedom: the standard deviation divides by n minus this number (1 matches MATLAB's std).	1	-
--ddof@run statistic	N	Delta degrees of freedom: the standard deviation divides by n minus this number (1 is MATLAB's std).	1	-
--default	NAME	The template resolution used for the dataset's default subject (needs --templates).	fsaverage5	-
--default@create dataset	NAME	The subject of the --templates folder (the template resolution) that becomes the dataset's default subject; needs --templates.	fsaverage5	-
--defects	keep|exclude	keep: use every edge of the mesh in the weighted fit; exclude: give zero weight to folded or very short edges (needs --spatial-band)	keep	-
--defects@run phasegradient	keep|exclude	With --spatial-band: exclude gives weight 0 to edges at a fold-over of the mesh or shorter than 0.1 of the median edge; keep leaves them in.	keep	-
--depth	LEVEL	Level of the surface's tile ladder that gives the leaf tiles on which sums are stored (2^LEVEL tiles per piece); for tiles compute, the finest level written.	the finest level every piece of the surface reaches (pyramid reduce: the stored depth)	level
--depth-channels	all|one-type	Which channels the depth-weighting norm is computed on: all channels, or one channel type (gradiometers if present, else magnetometers, else EEG).	all	-
--depth-channels@run inverse	all|one-type	Channels the depth-weighting norm uses: all channels (Brainstorm) or one type only (MEG GRAD, else MEG MAG, else EEG; MNE-Python).	all	-
--depth-exponent	P	Exponent of the depth weighting that counteracts the bias of the minimum-norm solution towards superficial sources	0.5	-
--depth-exponent@run inverse	P	Depth-weighting exponent p of the source norm.	0.5 (0.8 with --mne)	-
--depth-limit	X	Upper limit of the depth weighting (Brainstorm's WeightLimit): the largest-to-smallest weight ratio is limited to this value squared in variance.	10	-
--depth-limit-on	weight|norm	Where the depth limit is applied: on the weights (Brainstorm's way) or on the inverse norms (MNE-Python's way).	weight	-
--depth-limit-on@run inverse	weight|norm	Where the depth limit is applied: on the weight (Brainstorm) or on the norm (MNE-Python).	weight	-
--depth-limit@run inverse	L	Depth-weighting limit: the largest weight relative to the smallest is bounded by this factor.	10	-
--depth-norm	frobenius|spectral	Matrix norm used for each source's gain when depth weighting	frobenius	-
--depth-norm@run inverse	frobenius|spectral	Norm of each source's gain block: Frobenius (Brainstorm) or spectral, the largest eigenvalue (MNE-Python).	frobenius	-
--depth@run opticalflow	LEVEL	Level of the surface's tile ladder whose leaf tiles hold the stored sums of divergence, curl, speed and velocity; the ladder is made if absent.	the finest level every piece of the surface reaches	level
--depth@run phasegradient	LEVEL	Level of the surface's tile ladder that gives the leaf tiles on which the sums are stored (plain route only, not with --spatial-band).	the finest level every piece of the surface reaches	level
--depth@run pyramid	LEVEL	Level of the leaf tiles (2^LEVEL tiles per surface piece) on which the sums are stored.	the finest level every piece reaches (group tree: the dataset's atlas depth)	level
--depth@run tiles	LEVEL	Finest level of the tile ladder written; level l has 2^l tiles per connected piece of the surface.	the finest level every piece of the surface reaches	level
--depth@reduce pyramid	LEVEL	Level of the group tree at which the members' pyramids are summed, cell for cell.	the atlas depth of the group tree definitions	level
--detrend	none|constant|linear	Trend removed from each Welch segment before windowing.	constant (the Brainstorm option set)	-
--detrend@run crossspectrum	none|constant|linear	Trend removed from each Welch segment before the taper is applied; used by the Welch estimator.	constant	-
--detrend@run hodgepower	none|constant|linear	What is removed from each Welch segment before the window.	constant	-
--detrend@run psd	none|constant|linear	What is removed from each segment before windowing	constant	-
--detrend@run quadrature	none|constant|linear	Trend removed from each Welch segment before windowing.	constant (set tiles, brainstorm, scipy); none (nxr-chain)	-
--detrend@run sourcepsd	none|constant|linear	Trend removed from each Welch segment before the window is applied.	constant	-
--detrend@run sourcetensor	none|constant|linear	Trend removed from each Welch segment before windowing.	constant	-
--detrend@run speedspectrum	none|constant|linear	What is removed from each segment before windowing	constant	-
--divisor	samples-minus-blocks|samples|samples-minus-one	What the covariance sum is divided by: samples minus the number of blocks, the samples, or samples minus one.	samples-minus-blocks	-
--divisor@run inverse	samples-minus-blocks|samples|samples-minus-one	With --recording: what the noise covariance's sum of squares is divided by (N-B, N or N-1).	samples-minus-blocks	-
--divisor@run noisecov	samples-minus-blocks|samples|samples-minus-one	What the sum of squares is divided by: samples minus the number of blocks, the number of samples, or samples minus one.	samples-minus-blocks	-
--draws	N	Number of complex Wishart random draws used for the null distribution	1000	draws
--draws@run wishart	N	Number of complex Wishart draws of each burst's own zero-lag covariance that form the null.	1000	draws
--dry-run		Report what would change (compact: files and folders before and after; run: inputs found and arrows checked) without changing or computing anything.	off	-
--dry-run@compact dataset		Report the files and folders before and after compacting, without changing anything.	off	-
--dry-run@run		Print the plan (inputs found, every connection checked, every output's place) without reading data or writing anything.	off	-
--edge-lag	frame|frame-normal|frame-tangent|axial|ambient	How each edge's phase lag is read: in the surface's own frame (frame), its normal or tangent part alone, modulo pi (axial), or from the x y z components as they are (ambient).	frame	-
--edge-lag@run phasegradient	frame|frame-normal|frame-tangent|axial|ambient	How the phase lag along each mesh edge is compared (with --spatial-band): in the surface frame, its normal or tangent part alone, modulo pi, or as raw x y z components.	frame	-
--eeg-conductivities	S1,S2,S3	Conductivities of the EEG sphere shells, innermost to outermost (brain, skull, scalp)	0.33,0.0042,0.33	S/m
--eeg-conductivities@run headmodel	C1,C2,...	Conductivities of the EEG sphere's shells, innermost to outermost.	0.33,0.0042,0.33	S/m
--eeg-radii	R1,R2,R3	Relative radii of the EEG head model's shells, innermost to outermost (brain, skull, scalp).	0.88,0.93,1.0	relative
--eeg-radii@run headmodel	R1,R2,...	Relative radii of the EEG sphere's shells, innermost to outermost (brain, skull, scalp).	0.88,0.93,1.0	fraction
--eeg-series		Use the multi-shell series itself for the EEG sphere instead of the Berg-Scherg equivalent dipoles (slower, exact up to truncation).	off	-
--eeg-series-terms	N	Number of terms of the multi-shell series when it is used instead of the three-dipole fit (with --eeg-series)	200	terms
--eeg-series-terms@run headmodel	N	The number of terms of the multi-shell series, with --eeg-series.	200	-
--eeg-series@run headmodel		Compute the EEG gain with the multi-shell series itself instead of the Berg-Scherg equivalent dipoles (exact up to truncation, slower).	off	-
--eeg-surface	NAME	Name of the surface the EEG sphere is fitted to.	the surface called scalp (or head)	-
--eeg-surface@run headmodel	NAME	The surface the EEG sphere is fitted to.	the surface called scalp (or head)	-
--enter-sd	SD	A patch is kept when a tile reaches the median plus this many robust standard deviations of the map's tile means.	1.5	SD
--enter-sd@run patches	K	A patch must reach the median plus K robust standard deviations of the tile means.	1.5	SD
--estimator	welch|analytic	How the sensor cross-spectrum is estimated: Welch segments, or the band's analytic signal summed per time tile (a series of windows)	welch	-
--estimator@run crossspectrum	welch|analytic	How the cross-spectrum is estimated: by Welch segments, or from each band's analytic signal summed per time tile (a series of windows).	welch	-
--events	NAME	Name of the spatial-events pipeline whose stored statistics are tested against the null.	spatial_events	-
--events@run spatialnull	NAME	The pipeline that named the data's stored events statistics (<NAME>.home.<recording>, <NAME>.summary.<recording>).	spatial_events	-
--exceedances	N	Besag-Clifford stopping count: a tile or vertex is decided once it has this many exceedances of the data.	10	exceedances
--exceedances@run null	H	A tile is decided once this many null draws reach or exceed its observed value (Besag-Clifford)	10	draws
--exceedances@run spatialnull	N	The number of exceedances (draws at least as extreme as the data) at which a vertex's p-value is settled; sets the sequential stopping rule.	10	draws
--exclude	NAMES	Comma-separated member names left out of the exported table.	unassigned	-
--exclude@export measurements	NAMES	Comma-separated members (by dictionary name) left out of the table	unassigned	-
--exit-sd	SD	A patch extends over tiles at or above the median plus this many robust standard deviations	1.0	robust SD
)TBL",
R"TBL(--exit-sd@run patches	K	A patch extends over neighbouring tiles at or above the median plus K robust standard deviations.	1.0	SD
--extra-surface	FILE	An extra surface file to import with the condition (repeatable).	none	-
--extra-surface@import condition	FILE	An additional surface file of the subject's anatomy to import with the condition; repeat the option for several.	none	-
--family	NAME	Family of the time filter bank: itersine, meyer, morlet, morse, amor or bump (opticalflow: see the command's description for the accepted names).	none (required)	-
--family@run opticalflow	itersine|meyer|morlet|morse|amor|bump	Family of the time filter bank that band-passes the recording to the analysed band.	required	-
--family@run speedspectrum	itersine|meyer|morlet|morse|amor|bump	Family of the time (frequency) filter bank	required	-
--field	NAME	Name of a stored field to read (measure query: restrict the query to this field; phasegradient: the phase 1-form to build, only gradient exists).	measure query: none; phasegradient: gradient	-
--field@run phasegradient	gradient	The phase 1-form that is built: the lag along each edge (gradient is the only one).	gradient	-
--field@query measurements	NAME	The stored field the measurement was made of, by name or id.	the only field measured	-
--fields	PATTERN	Regular expression matched against the whole name of each stored field to measure (export: the first capture group names the condition)	none (required)	-
--fields@run measure	PATTERN	Regular expression matched against the whole name of each stored field to measure (a scalar per vertex for a surface partition; a time-resolved field with --over).	required	-
--fields@export measurements	REGEX	Regular expression over Field names selecting the Fields to export; its first capture group names the condition (e.g. the band)	required	-
--file	FILE	A Brainstorm anatomy file (tess_*.mat) to convert; repeat or give several.	none (required)	-
--file@import inverse	FILE	The Brainstorm inverse kernel file (results_*KERNEL*.mat) to import; one kernel per call, never overwritten.	required	-
--file@import map	FILE	A map file to import: a source map (results_*.mat, one value per vertex) or a region map (matrix_*.mat, one value per atlas region); repeatable.	required	-
--file@import surface	FILE	A Brainstorm cortex surface file (tess_*.mat) to convert and store; repeat or give several.	required	-
--fit	inner|mesh|modes	How the timing map is fitted to the edge lags: by inner products with the modes, by a weighted fit on the whole surface complex, or by a weighted fit within the exact modes.	inner	-
--fit@run phasegradient	inner|mesh|modes	How the timing map is fitted (with --spatial-band): inner products with the modes' 1-forms, a weighted fit on the whole mesh, or a weighted fit within the exact modes.	inner	-
--flip-eps	E	Tolerance of the intrinsic Delaunay edge flips: flip while the cotangent weight is below -E	1e-6	-
--flip-eps@run eigenmodes	E	Edge-flip tolerance of the intrinsic Delaunay construction: an edge is flipped while its cotan weight is below -E.	1e-6	-
--flow-tolerance	X	Relative residual at which the conjugate-gradient flow solve stops (pcg and per-scale methods).	1e-8	-
--flow-tolerance@run opticalflow	X	Relative residual at which the iterative solver (pcg, perscale) stops.	1e-8	-
--fmax	HZ	Upper frequency limit (constantq: ceiling of the time ladder, at most Nyquist; speedspectrum: the highest frequency bin used).	constantq: 64 Hz cap (method default); speedspectrum: 40	Hz
--fmax@run constantq	HZ	Highest frequency of the ladder; overrides the method's top (never above Nyquist).	hertz: largest power of two at or below Nyquist, at most 64; cycles: Nyquist	Hz
--fmax@run speedspectrum	HZ	Highest frequency bin used; the time bank should cover it	40	Hz
--fmin	HZ	Lowest frequency (constantq: lowest band edge; speedspectrum: lowest frequency bin used)	constantq: 0.5 (hertz method) or 0 (cycles method); speedspectrum: 1	Hz
--fmin@run constantq	HZ	Lowest frequency of the ladder.	hertz: 0.5; cycles: 0	Hz
--fmin@run speedspectrum	HZ	Lowest frequency bin used; the time bank should cover it	1	Hz
--fold-edges	keep|axial|all	How edges across a fold of the surface are read: keep all as they are, axial (lag read modulo pi where the ends' normals are opposed), or all (every edge modulo pi).	keep	-
--fold-edges@run phasegradient	keep|axial|all	With --spatial-band: read the lag modulo pi on edges whose end normals are opposed (axial) or on every edge (all), so a fold is not read as phase.	keep	-
--forward	NAME	Name of the stored head model (forward operator) to use.	the subject's only one	-
--forward-id	ID	The id of the head model the kernel was made with.	found from the kernel file	-
--forward-id@import inverse	ID	Id of the stored head model the kernel was made with.	found from the kernel's HeadModelFile link	-
--forward@run inverse	NAME	Name of the stored head model (forward operator) whose gain the kernel inverts.	the subject's only head model	-
--forward@run pointspread	NAME	The head model (forward operator) the unit currents are planted through	the subject's only forward model	-
--fpass	HZ|LO,HI	Passband edge: one frequency for lowpass or highpass, LO,HI for bandpass	none (required)	Hz
--fpass@run bandpass	LO,HI	Lower and upper passband edge of the band-pass filter, both below the Nyquist frequency.	required	Hz
--fpass@run bandstop	HZ	Passband edges of lowpass, highpass and bandpass; not used by bandstop (give --fstop).	none (not used)	Hz
--fpass@run highpass	HZ	The highpass cut-off: frequencies above it pass (the lower passband edge).	required	Hz
--fpass@run lowpass	HZ	Upper edge of the passband: frequencies above it are attenuated.	required	Hz
--fpass@run notch	HZ	Not used by notch; the pass edge of the low-pass, high-pass and band-pass commands.	none	Hz
--fraction	X	Fraction of the peak power at the reference level above which the connected region counts as the source's patch.	0.5	-
--fraction@run pointspread	FRACTION	The patch is the connected region at the reference level whose power is at least this fraction of its peak	0.5	fraction
--frame-level	L	Tower level of a time frame, a whole number: a frame lasts 86400 x 2^L seconds (-18 = 0.33 s, -25 = 2.57 ms). (frametiling: alternative to --centre or --band)	-18 (frametiling: none, give a level, centre or band)	-
--frame-level@run frametiling	LEVEL	The tower level of one frame (0 is the day, -18 is 0.33 s, -25 is 2.57 ms); alternative to --centre and --band.	none (derived from the centre)	level
--frame-level@run opticalflow	L	Tower level of a stored time frame, a whole number: a frame lasts 86400 x 2^L seconds (-18 is 0.33 s).	-18	-
--frame-level@run pyramid	L	Tower level of a time frame, a whole number: a frame lasts 86400 x 2^L seconds.	-18 (0.33 s)	level
--frames-per-cycle	N	Frames per cycle of the band's centre frequency, which sets the frame level (the coarsest whose frame is at or under 1/(centre x N))	36	frames
--frames-per-cycle@run frametiling	N	Frames to one cycle of the centre frequency; the level is the coarsest whose frame is no longer than 1/(centre x N).	36	frames
--freq	HZ	Frequency whose nearest filter-bank member is analysed.	none (required)	Hz
--freq@run opticalflow	HZ	Frequency whose nearest filter-bank member is analysed.	required	Hz
--frequencies	F,...	Centre frequencies to remove (notch), e.g. a line frequency and its harmonics.	none (give this or --line)	Hz
--frequencies@run bandpass	HZ,...	Notch frequencies; they belong to the notch filter and have no effect on a bandpass.	none	Hz
--frequencies@run bandstop	F1,F2,...	Notch frequencies; used by notch only and not by bandstop.	none (notch only)	Hz
--frequencies@run highpass	F,...	Not used by highpass: it is a FIR design; the notch centre frequencies are for the notch command.	none	Hz
--frequencies@run lowpass	F1,F2,...	Notch frequencies; not used by lowpass (notch only).	none	Hz
--frequencies@run notch	F1,F2,...	The frequencies to remove; an alternative to --line.	none (give this or --line)	Hz
--from	FILE	The file to read the bad segments or events from (import badsegments, import events).	none	-
--from@import badsegments	FILE	JSON file listing the bad segments (start and stop times) to store as the subject's bad-segment selection.	required	-
--from@import events	FOLDER	The source protocol's data folder to read the events from.	required	-
--fstop	LO,HI	Stopband edges of a bandstop filter	none (required)	Hz
--fstop@run bandpass	LO,HI	Stopband edges of the band-stop filter; they belong to bandstop and have no effect on a bandpass.	none	Hz
--fstop@run bandstop	LO,HI	Lower and upper edge of the stopband that is removed (zero-phase Butterworth band-stop, forward-backward).	required	Hz
--fstop@run highpass	LO,HI	Not used by highpass: it is a FIR design; the stopband edges are for the bandstop command.	none	Hz
--fstop@run lowpass	LO,HI	Lower and upper edges of the stopband; not used by lowpass (bandstop only).	none	Hz
--fstop@run notch	LO,HI	Not used by notch; the stop band of the band-stop command.	none	Hz
--gain	X	Peak in-band response of each filter-bank member (2 makes the coefficient's magnitude the physical amplitude).	none (required)	-
--gain@run opticalflow	X	Peak in-band response of each filter-bank member; 2 makes the coefficient's magnitude the physical amplitude of a real recording.	required	-
--gain@run speedspectrum	G	Peak in-band response of every time bank member	required	-
--gamma	G	Morse family: the gamma parameter of the wavelet.	3	-
--gamma@run opticalflow	X	Morse family only: the gamma parameter of the wavelet.	3	-
--gamma@run speedspectrum	G	Morse time bank: the gamma parameter	3	-
--gradient		Also measure the gradient of each scalar field (area-weighted face gradients projected on the vertex tangent plane)	off	-
--gradient@run measure		Also measure the gradient of each scalar field (area-weighted face gradients projected on the vertex tangent plane); needs the canonical frames from frames compute. Surface partitions only.	off	-
--graph-family	mexhat|heat|itersine|logitersine	Family of the spatial (graph) filter bank over the eigenvalue.	none (required)	-
--graph-family@run speedspectrum	mexhat|heat|itersine|logitersine	Family of the graph (spatial) filter bank over the Laplacian eigenvalues	required	-
--graph-filters	N	Number of members of the graph (eigenvalue) filterbank; 0 means derived from --graph-voices.	0	filters
--graph-filters@run speedspectrum	N	Number of graph bank members; 0 derives it from --graph-voices	0	-
--graph-voices	N	Members per octave of the spatial (graph) filterbank	3	members/octave
--graph-voices@run speedspectrum	V	Graph bank members per octave	3	-
--half-width	OCT	Meyer family: half-width of each transition between members, in octaves.	0.3333 (1/3)	octaves
--half-width@run opticalflow	OCTAVES	Meyer family only: half-width of the transition between neighbouring octave bands.	0.3333	octaves
--half-width@run speedspectrum	OCTAVES	Meyer time bank: half-width of each transition	0.3333	octaves
--halo-seconds	SECONDS	Context read on each side of a chunk so the band filter's edge effects fall inside it	8	s
--halo-seconds@run pyramid	SECONDS	Context read on each side of a chunk so the band filter's edge effects fall inside it.	8	s
--harmonics	N	With --line, the number of harmonics of the line frequency to notch.	every multiple below the Nyquist frequency	-
--harmonics@run bandpass	N	Number of harmonics of a notch --line; it belongs to the notch filter and has no effect on a bandpass.	none	-
--harmonics@run bandstop	N	Number of line harmonics to notch; used by notch only and not by bandstop.	none (notch only)	-
--harmonics@run highpass	N	Not used by highpass: it is a FIR design; it counts line harmonics for the notch command.	none	-
--harmonics@run lowpass	N	Number of line-frequency multiples to notch; not used by lowpass (notch with --line only).	every multiple below Nyquist (notch only)	-
)TBL",
R"TBL(--harmonics@run notch	N	With --line, remove the first N multiples of the line frequency.	every multiple below Nyquist	-
--include-imported		Also import conditions that were themselves imported.	off	-
--include-imported@import condition		When the condition has a raw recording, also import its data files that were themselves imported (by default only the raw recording is).	off	-
--input	amplitude|signed	What the flow is computed on: the source amplitude |J| of the kernel applied to the complex band coefficients, or the signed band signal itself (fixed-orientation kernels only).	amplitude	-
--input@run opticalflow	amplitude|signed	Map the flow is computed on: the source amplitude |J| of the kernel applied to the band's complex coefficients, or the signed band signal itself (fixed-orientation kernels only).	amplitude	-
--joint	NAME	Name of the stored joint space-time partition to use	the one for the cross-spectrum's recording, the space level and its tile level	-
--joint-types		Whiten the channel types of a modality (e.g. magnetometers with gradiometers) together, keeping their cross-covariance, instead of one block per type.	off	-
--joint-types@run inverse		Whiten the channel types of a modality (magnetometers with gradiometers) together, keeping their cross-covariance, instead of one block per type.	off	-
--joint@run cellquadrature	NAME	Stored joint space-time partition whose cells are measured.	the one for the cross-spectrum's recording, this space level and its tile level	-
--keep-all		Keep every member in the table, including those excluded by default.	off	-
--keep-all@export measurements		Keep every member in the table, including the default excluded one (clears --exclude)	off	-
--keep-bad-segments		Keep the recording's bad segments in the computation instead of leaving them out	off	-
--keep-bad-segments@run pyramid		Keep the recording's bad segments in the sums instead of leaving them out.	off (bad segments are left out)	-
--kernel	NAME	Name of the stored inverse kernel (the map from channels to sources) to apply.	the subject's only kernel	-
--kernel@run cellquadrature	NAME	Name of the stored inverse kernel (free or fixed orientation) with the spatial band folded in.	the subject's only kernel	-
--kernel@run cyclepattern	NAME	Name of the stored free-orientation inverse kernel that maps the cross-spectrum to sources.	the subject's only kernel	-
--kernel@run hodgepower	NAME	The free-orientation inverse kernel the recording is read through.	the subject's only kernel	-
--kernel@run opticalflow	NAME	Name of the stored inverse kernel that maps the channels to sources on the cortex.	the subject's only kernel	-
--kernel@run phasegradient	NAME	Name of the free-orientation inverse kernel whose sources carry the phase.	the subject's only kernel	-
--kernel@run pointspread	NAME	The inverse kernel the planted currents are read back through	the subject's only kernel of the forward model's sources	-
--kernel@run pyramid	NAME	Name of the stored inverse kernel that maps the recording's channels to sources.	the subject's only kernel	-
--kernel@run quadrature	NAME	Name of the stored inverse kernel that maps the sensors to the sources.	the subject's only kernel	-
--kernel@run sourcepsd	NAME	Name of the stored inverse kernel that maps the recording's channels to sources.	the subject's only kernel	-
--kernel@run sourcetensor	NAME	Name of the stored free-orientation inverse kernel (channels to sources) whose current's tensor is computed; a fixed-orientation kernel is refused.	the subject's only kernel	-
--kernel@run speedspectrum	NAME	The free-orientation inverse kernel the recording is mapped to the cortex with	the subject's only kernel	-
--kernels	SPEC,...	Which kernels' pyramids are summed: all, constrained, free, latest, a method, stamp, name or glob (kinds combine with AND, tokens of one kind with OR).	all	-
--kernels@reduce pyramid	SPEC,...	Which source estimates of each member are summed: all, constrained, free, latest, or a kernel's method, stamp, name or glob (kinds combine with AND, tokens of one kind with OR).	all	-
--kind	subject|template|group	Kind of subject to create	subject	-
--kind@create subject	subject|template|group	Kind of the new subject: an ordinary subject, a template subject or a group subject.	subject	-
--labelling	NAME	The partition (a stored Selection with one code per vertex, e.g. a parcellation or a tile level) whose members are measured.	none (required)	-
--labelling@run measure	NAME	The partition (a stored Selection with one code per vertex, e.g. a parcellation or a tile level) whose members are measured; not used with --over.	required unless --over is given	-
--labelling@export measurements	NAME	The partition (labelling) whose members are the table's columns, named the same in every subject	required	-
--layout	split|wide	Table layout: split gives a table per condition and statistic with a column per member; wide gives one table with a column per condition, member and statistic.	split	-
--layout@export measurements	split|wide	split writes one table per condition and statistic with a column per member; wide writes one table with a column per condition, member and statistic	split	-
--lc-levels	L,L,...	Tile-ladder levels of the Levi-Civita connection stored with the default subject (needs --templates)	4,6	level
--lc-levels@create dataset	L,L,...	Comma-separated tile levels at which the Levi-Civita connection of the default subject is stored; needs --templates.	4,6	level
--level	L	A level (patches: the tile-ladder level whose spatial band is used; crossspectrum, cyclepattern: the window tile level, windows of 2^L samples; measure compute: the level of a constant-Q time ladder).	none (required for patches; otherwise see the command's description)	level
--level@run crossspectrum	L	Analytic estimator only: tile level of the windows, each 2^L samples long (0 to 15); an alternative to --window-seconds.	the finest of the bands' constant-Q windows (about 6.8 cycles of the band's low edge)	level
--level@run cyclepattern	L	Time-tile level of the windows, 2^L samples (at least the stored level); the stored tiles are rolled up to it. Alternative to --window-seconds.	none (--window-seconds applies)	level
--level@run measure	L	Level of the constant-Q time ladder named by --over; used only with --over.	none (required for a ladder)	level
--level@run patches	LEVEL	The ladder level (tiles of 2^LEVEL per piece) whose Weyl band is the map's filter; the ladder must exist.	required	level
--levels	L,...	Spatial tile levels of the scale space the blur is measured over.	2,3,4,5,6,7,8	levels
--levels@run pointspread	LEVELS	Comma-separated spatial scale levels searched for each patch's characteristic size	2,3,4,5,6,7,8	level
--limit	N	Maximum number of rows returned	none	rows
--limit@query measurements	N	The most rows to return.	none (all rows)	rows
--limits	LO-HI	Frequency limits of the filter bank for the itersine and meyer families.	see the command's description	Hz
--limits@run opticalflow	LO-HI	Itersine and Meyer families: the frequency limits of the filter bank, in hertz (e.g. 2-40).	none (set by the family)	Hz
--limits@run speedspectrum	LO-HI	Itersine and Meyer time banks: the bank's frequency limits	none	Hz
--line	NAME or F	Time line to work on (constantq: a time line by name, alternative to --recording or --all); for lowpass/notch: the mains frequency F, removed with its harmonics.	none	constantq: -; notch: Hz
--line@run bandpass	HZ	Mains frequency of the notch filter; it has no effect on a bandpass.	none	Hz
--line@run bandstop	F	Mains frequency to notch with its harmonics; used by notch only and not by bandstop.	none (notch only)	Hz
--line@run constantq	NAME	A time line of the subject, by name, to make the ladder for (alternative to --recording and --all).	none	-
--line@run highpass	F	Not used by highpass: it is a FIR design; the mains frequency is for the notch command.	none	Hz
--line@run lowpass	HZ	Mains line frequency to notch; not used by lowpass (notch only).	none	Hz
--line@run notch	HZ	The mains line frequency, removed together with its harmonics; an alternative to --frequencies.	none (give this or --frequencies)	Hz
--list	V,V,...	run: list the available pipeline definitions and their steps (flag); pointspread: native vertex indices to use as the source sample instead of --vertices	run: off; pointspread: none	-
--list@run pointspread	VERTICES	Comma-separated native vertices to plant the currents at; wins over --vertices	none	-
--map	NAME	The stored cortical map (a scalar per vertex of a surface) in which patches are sought.	none (required)	-
--map@run patches	NAME	The stored field to find patches in: one scalar per vertex of a surface, all finite.	required	-
--max-area	M2	Largest patch area kept; larger patches are dropped.	none	m^2
--max-area@run patches	AREA	The largest patch kept.	none	m2
--max-condition	C	Canonical frames: a vertex whose push-forward Jacobian has a condition number above C is treated as singular	50	-
--max-condition@run frames	C	Canonical frames: a vertex whose push-forward Jacobian has a condition number above C is treated as singular.	50	-
--max-draws	N	Largest number of null draws.	100	draws
--max-draws@run null	N	Most null draws made; a rerun with a larger value continues where the last stopped	100	draws
--max-draws@run spatialnull	N	The most draws made.	100	draws
--max-iterations	N	Iteration limit of the iterative flow solver (pcg and perscale methods).	500	iterations
--max-iterations@run opticalflow	N	Most iterations the iterative solver (pcg, perscale) may take for one frame.	500	-
--max-value	X	Keep only rows with a value at or below X	none	measure units
--max-value@query measurements	X	Keep only rows whose rolled-up value is at most this.	none	measure's unit
--measure	NAME	The measure to summarise or read (e.g. mean, max, power_sum).	none (required)	-
--measure@run statistic	NAME	The stored measure to summarise across the members (e.g. mean, mean_square, amplitude); a derived measure is refused.	required	-
--measure@query measurements	NAME	The measure to read, e.g. a stored measure of the pyramid or one derived from its sums.	required	-
--measures	NAME,...	Which measures to compute per member: mean, max, min, mean_square, abs_max, sum.	mean, max, min, mean_square, abs_max	-
--measures@run measure	NAME,...	Which measures to compute per member: mean, max, min, mean_square, abs_max, sum (with --over: mean, mean_square, max, min, abs_max; the sample count is always stored).	mean, max, min, mean_square, abs_max (with --over: mean, mean_square, max, min)	-
--meg	overlapping|single|none	MEG head model: per-sensor overlapping spheres, one single sphere, or no MEG channels	overlapping	-
--meg-surface	NAME	Name of the surface the MEG spheres are fitted to.	the surface called inner skull	-
--meg-surface@run headmodel	NAME	The surface the MEG spheres are fitted to.	the surface called inner skull	-
--meg@run headmodel	overlapping|single|none	The MEG head model: one sphere per sensor, one sphere for all sensors, or no MEG channels.	overlapping	-
--members-own		The measure is the members' own (area, number of elements or tiles); --of then names its parent.	off	-
--members-own@run statistic		The measure is the members' own (area, number of elements or tiles), a property of the partition and not of a field; --of then names its parent.	off	-
--method	NAME	constantq: hertz (octave edges on powers of two in Hz) or cycles (fixed cycle count); opticalflow: pcg, direct or perscale solver	constantq: hertz; opticalflow: pcg	-
--method@run constantq	hertz|cycles	How the windows are anchored: octave edges on powers of two in Hz (hertz, capped at 64 Hz), or a fixed number of cycles per window (cycles).	hertz	-
--method@run opticalflow	pcg|direct|perscale	Solver: pcg (preconditioned conjugate gradients), direct (exact solve, for parity with the reference) or perscale (one flow per spatial scale).	pcg	-
)TBL",
R"TBL(--min-area	M2	Smallest patch kept.	none	m^2
--min-area@run patches	AREA	The smallest patch kept.	none	m2
--min-crossings	N	Minimum number of crossings a tile needs (in both runs for repeat) to be tested.	10	crossings
--min-crossings@run null	N	A tile counts only with at least this many crossings	10	crossings
--min-crossings@run repeat	N	Minimum number of crossings a tile needs in both recordings to count in the repeat statistic.	10	crossings
--min-draws	N	Smallest number of null draws made before early stopping may decide a tile or vertex	20	draws
--min-draws@run null	N	Fewest null draws made	20	draws
--min-draws@run spatialnull	N	The fewest draws made before stopping, even if every vertex has reached --exceedances.	20	draws
--min-link	N	Two events of different scale pairs count as one event when at least this many pairs of their cells nest in space and overlap in time.	1	pairs
--min-link@run scaleselection	N	Events of two pairs are one cross-pair event when at least this many of their cells nest in space and overlap in time	1	pairs
--min-projection	P	Reliability limit of the anatomical frame: the anterior axis must project on the tangent plane with at least this length.	0.34	-
--min-projection@run frames	P	Reliability limit of the anatomical frame: the anterior axis must project on the tangent plane with at least this length (sin 20 degrees).	0.34	-
--min-value	X	Keep only rows with a value at or above X	none	measure units
--min-value@query measurements	X	Keep only rows whose rolled-up value is at least this.	none	measure's unit
--mne		Use MNE-Python's conventions for the kernel: joint types, ridge over the rank, PCA whitener, depth exponent 0.8 by the spectral norm.	off	-
--mne-eeg-layers		Use MNE-Python's EEG sphere layers and conductivities instead of the default three-shell layers.	off	-
--mne-eeg-layers@run headmodel		Use MNE-Python's four EEG layers (radii 0.90, 0.92, 0.97, 1.00; conductivities 0.33, 1.0, 0.004, 0.33 S/m) instead of three.	off (three layers)	-
--mne@run inverse		Start from MNE-Python's conventions (joint types, ridge over rank, PCA whitener, depth 0.8 spectral, Lambda by whitener rank); the other options still apply after it.	off (Brainstorm's defaults)	-
--modes	K	Number of Hodge-Dirac eigenmodes (eigen solve, dataset create: how many to solve; pointspread, cellquadrature, patches: size of the package when it must be computed, 0 = standard; speedspectrum, hodgepower: use the first K of the stored package)	eigen solve: 3000; dataset create: none; speedspectrum, hodgepower: all; pointspread, cellquadrature, patches: 0	modes
--modes@run cellquadrature	K	Size of the Hodge-Dirac eigenmode package when it must be computed for the spatial band (0 = standard size).	0	modes
--modes@run eigenmodes	K	About this many positive Hodge-Dirac modes are kept, half exact and half coexact.	3000	modes
--modes@run hodgepower	N	Use the first N modes of the Hodge-Dirac package.	all modes	-
--modes@run patches	N	When the eigenmodes are computed, how many (more if the band needs it); 0 is the standard 3000.	0	-
--modes@run pointspread	N	Number of eigenmodes to compute when the spatial filter package is built; 0 uses the standard size, or more when the band needs it	0	-
--modes@run speedspectrum	N	Use only the first N modes of the Hodge-Dirac package	all modes	-
--modes@create dataset	K	Solve the default subject's pial Hodge-Dirac eigenmode basis with about this many modes and store it; needs --templates.	none (no basis is solved)	modes
--mollify	X	Mollification margin of the intrinsic Delaunay construction (Sharp and Crane), relative to the mean edge length; 0 means no margin.	1e-6	relative
--mollify@run eigenmodes	D	Mollification margin of the intrinsic Delaunay construction, as a fraction of the mean edge length (0 = no margin).	1e-6	fraction
--name	NAME	Name of what is stored: the dataset, subject or surface, or the stem of the stored result (filters: <recording>_<verb>).	derived from the inputs	-
--name@add surface	NAME	Name the surface is stored under for the subject.	the mesh file's name without its extension, made safe as a node name	-
--name@run bandpass	NAME	Name of the filtered recording that is stored.	<recording>_bandpass	-
--name@run bandstop	NAME	Name of the new filtered recording.	<recording>_bandstop	-
--name@run crossspectrum	NAME	Stem of the names of the stored cross-spectrum operators (one per band).	<recording>_crossspectrum	-
--name@run cyclepattern	NAME	Stem of the stored cycle-pattern results.	<cross-spectrum without _windows>_cycle	-
--name@run headmodel	NAME	The name the head model is stored under.	headmodel	-
--name@run highpass	NAME	Name of the new filtered recording.	<recording>_highpass	-
--name@run hodgepower	NAME	The stem of the names of the stored fields (one current and one flow field per band).	<recording>_<kernel>_hodge	-
--name@run inverse	NAME	Name of the stored inverse kernel.	<forward>_minnorm	-
--name@run lowpass	NAME	Name the filtered recording is stored under.	<recording>_lowpass	-
--name@run noisecov	NAME	Name of the stored noise covariance.	<recording>_noise_cov	-
--name@run notch	NAME	The name of the filtered recording.	<recording>_notch	-
--name@run opticalflow	NAME	Stem of the names of the stored flow results.	<recording>_<kernel>_opticalflow	-
--name@run phasegradient	NAME	Stem of the names of the stored results.	<recording>_<kernel>_phasegradient	-
--name@run pointspread	NAME	Name of the stored point-spread statistic	pointspread.<kernel>	-
--name@run psd	NAME	Name of the stored PSD Field	<recording>_psd_welch	-
--name@run quadrature	NAME	Stem of the names of the stored fields.	<recording>_<kernel>_quadrature	-
--name@run repeat	NAME	Name the repeat result is stored under.	derived from the two recording names (plus the stage when not the first)	-
--name@run scaleselection	NAME	Name of the cross-pair events' Selection	<recording>_scale_selection	-
--name@run sourcepsd	NAME	Stem of the names of the stored band power maps.	<recording>_<kernel>_psd	-
--name@run sourcetensor	NAME	Stem of the stored fields (tensor, Stokes quaternion, normal power and share, each per band).	<recording>_<kernel>_tensor	-
--name@run speedspectrum	STEM	Stem of the stored result names	<recording>_<kernel>_speedspectrum	-
--name@run statistic	NAME	Name of the stored statistic.	<of>_<measure>_<over>	-
--name@create dataset	NAME	Name of the new dataset.	the folder's name (required with --datastore)	-
--name@create subject	NAME	Name of the new subject in the dataset.	required	-
--name@delete dataset	NAME	Name of the dataset's folder to delete, inside the folder given by --datastore	required	-
--name@import inverse	NAME	Name of the stored kernel.	Brainstorm's file stem, session-qualified if already taken	-
--name@import resample	NAME	Name of the resampled recording that is created	<recording>_<rate>Hz (e.g. meg_250Hz)	-
--name@read crossspectrum	NAME	The stored band-resolved cross-spectrum to read, named <stem>_<band>_bins (made by run crossspectrum --resolve bins).	required	-
--no-average-reference		Do not re-reference the EEG gain rows and covariance to the average	off	-
--no-average-reference@run inverse		Do not average-reference the EEG, sEEG and ECoG rows of the gain and the covariance.	off (average reference applied)	-
--no-bad-segments		Keep the recording's marked bad segments in the estimate instead of leaving out the segments that touch them.	off (bad segments are left out)	-
--no-bad-segments@run crossspectrum		Keep the recording's marked bad segments in the estimate instead of leaving out the Welch segments that touch a bad sample.	off (segments with a bad sample are left out)	-
--no-bad-segments@run hodgepower		Keep Welch segments that touch the recording's bad segments.	off (they are left out)	-
--no-bad-segments@run psd		Keep segments that touch the recording's bad segments instead of leaving them out	off	-
--no-bad-segments@run quadrature		Also use segments that overlap the recording's marked bad segments.	off (bad segments excluded)	-
--no-bad-segments@run sourcepsd		Keep the recording's bad segments in the estimate instead of leaving out the segments that touch them.	off (bad segments are left out)	-
--no-bad-segments@run sourcetensor		Keep the recording's marked bad segments in the estimate instead of leaving out the Welch segments that touch them.	off (bad segments are left out)	-
--no-bad-segments@run speedspectrum		Keep segments that touch the recording's bad segments instead of leaving them out	off	-
--no-depth		Turn depth weighting off.	off (depth weighting on)	-
--no-depth@run inverse		Turn depth weighting off (every source weighted equally).	off (depth weighting on)	-
--no-detect-diagonal		Do not whiten a diagonal noise covariance by its diagonal alone	off	-
--no-detect-diagonal@run inverse		Do not whiten a diagonal noise covariance by its diagonal alone; treat it like any other covariance.	off (a diagonal covariance is detected)	-
--no-eeg		Do not compute the EEG channels of the head model.	off (EEG is computed)	-
--no-eeg@run headmodel		Do not compute the EEG channels' gain.	off (EEG is computed)	-
--no-gain		Do not store the head model's gain matrix.	off (the gain is stored)	-
--no-gain@import condition		Do not store the head model's gain matrix.	off (the gain is stored)	-
--no-levi-civita		Do not write the Levi-Civita (transported) form of the vector tile summaries.	off (written)	-
--no-levi-civita@run opticalflow		Do not write the Levi-Civita (transported) form of the vector tile summaries; needs --tiles.	off (written)	-
--no-levi-civita@run quadrature		With --tiles: skip the Levi-Civita (transported) form of the tile summaries; the canonical-frame form is still written.	off	-
--no-primary		Do not mark the added mesh as the subject's primary surface	off	-
--no-primary@add surface		Store the surface without marking it as the subject's primary surface.	off (the surface is made primary)	-
--no-project-electrodes		Do not move the electrodes radially onto the fitted EEG sphere.	off (electrodes are projected)	-
--no-project-electrodes@run headmodel		Keep the electrodes where they are instead of moving them radially onto the fitted EEG sphere.	off (electrodes are projected)	-
--no-raw		Do not import the raw recording, only the condition's other data.	off	-
--no-raw@import condition		Do not import the condition's raw recording; import only its other data files.	off	-
--no-remove-mean		Do not remove the mean (DC offset) of each block before computing or filtering.	off (mean removed)	-
--no-remove-mean@run bandpass		Do not subtract each channel's mean before filtering. By default the mean is removed and added back only when the filter has no high-pass part.	off (the mean is removed)	-
--no-remove-mean@run bandstop		Do not subtract the record's mean before filtering (the mean is added back afterwards for a filter without a high-pass part).	off (mean removed)	-
--no-remove-mean@run highpass		Do not subtract the record's channel means before filtering (they are otherwise removed and not added back, as the filter has a high-pass part).	off (mean removed)	-
--no-remove-mean@run inverse		With --recording: keep each block's mean in the noise covariance (raw second moment) instead of removing it.	off (mean removed)	-
--no-remove-mean@run lowpass		Do not subtract each channel's mean before filtering.	off (the mean is removed, and added back since a lowpass has no high-pass part)	-
--no-remove-mean@run noisecov		Do not remove each block's mean (DC offset) before summing: the covariance is then the raw second moment.	off (each block's mean is removed)	-
--no-remove-mean@run notch		Do not subtract each channel's mean before filtering.	off (the mean is removed, and added back)	-
--no-shard		Do not group chunks into shards when compacting	off	-
--no-shard@compact dataset		Do not group chunks into shards; only the key separator changes, so an array over 4 MiB stays unsharded.	off (arrays are sharded)	-
)TBL",
R"TBL(--no-small-files		Stop at the sharded layout; do not also move each subject's small files into its small-files database.	off	-
--no-small-files@compact dataset		Stop at the sharded layout; do not also move each subject's small keys into its small-files database.	off (small files are moved)	-
--node	PATH	Delete only this stored node of the subject, by store path, instead of the whole subject.	none (whole subject)	-
--node@delete subject	NAME	Delete only this node (manifold, field, operator or selection) of the subject instead of the whole subject	none (the whole subject is deleted)	-
--noise-cov	NAME	Name of a stored noise covariance to use (alternative to --recording)	none	-
--noise-cov@run inverse	NAME	Name of a stored noise covariance to whiten with; an alternative to --recording.	none (give this or --recording)	-
--normalisation	none|relative	Normalisation of the band power maps: none, or relative (each map divided by the source's total power over all bins up to Nyquist).	none	-
--normalisation@run sourcepsd	none|relative	Divide each source's band power map by its total power over all bins up to Nyquist (relative), or leave it as power.	none	-
--normalisation@run sourcetensor	none	Only none is accepted: a tensor is not normalised, since the shares are read from it.	none	-
--normalize		Re-orthonormalise the eigenmodes afterwards (orthonormal in the mass matrix).	off	-
--normalize@run eigenmodes		Make the modes orthonormal under the mass matrix after the solve (Cholesky whitening); the reference method does not.	off	-
--of	NAME	The stored field, or band of a map statistic, that was measured	none (required)	-
--of@run statistic	NAME	What was measured: a stored field by name, or a band of a map statistic (<statistic>_<band>).	required	-
--only	NAME	Import only this named anatomical image (repeatable).	all images	-
--only@import mri	NAME	Import only this named anatomical image; repeat the option for several.	all images	-
--order	N	Filters: Butterworth prototype order (the filter has 2N poles). Spatial band commands: degree of the Chebyshev series (0 chooses the smallest of 40, 80, ... 5120 within tolerance). measure query: sort order asc or desc of the rows.	filters 4; spatial 0 (automatic); measure query see the command's description	-
--order@run bandpass	N	Butterworth order of the band-stop filter; it has no effect on the band-pass, which is a linear-phase FIR filter whose order follows from --ripple and --transition.	4	-
--order@run bandstop	N	Order N of the Butterworth prototype (the band-stop has 2N poles per stage).	4	-
--order@run cellquadrature	N	Degree of the Chebyshev series for the spatial band (0 picks the smallest of 40, 80, ... 5120 within --tolerance).	0	-
--order@run highpass	N	Not used by highpass: it is a FIR design; the Butterworth order is for the bandstop and notch commands.	4	-
--order@run lowpass	N	Butterworth order of the bandstop prototype; not used by lowpass (FIR design).	4 (bandstop only)	-
--order@run notch	N	Not used by notch, which is a 2nd-order notch applied forward and backward; the band-stop command's Butterworth order.	4	-
--order@run patches	N	Chebyshev: the series degree; 0 is the smallest of 40, 80, ... 5120 that follows the band's gain within --tolerance.	0	-
--order@run pointspread	N	Degree of the Chebyshev series when that transform is used; 0 picks the smallest of 40, 80, ... 5120 that meets --tolerance	0	-
--order@query measurements	asc|desc	Order the rows by value, ascending or descending.	by code	-
--out	PATH	Output file (convert scan: the scan result written as a file) or output folder (measure export: where the table is written, made if absent).	none (measure export: required)	-
--out@export measurements	PATH	Folder the tables, the provenance file and the dictionary are written to (created if absent)	required	-
--out@import scan	PATH	File to write the scanned protocol view to as JSON (writing it prints a one-line summary instead of the whole view).	none (the view is printed)	-
--out@read dataset	FILE	Write the bytes of the array (or of its --region) read with --array to this file, as little-endian values of its data type.	none	-
--over	NAME	Name of the partition (stored selection) measured over: statistic: the partition whose members were measured; null, repeat: the tile-level selection of the stored crossings; measure compute: the time tiling	statistic, measure compute: none (required); null, repeat: the one the recordings share	-
--over@run measure	NAME	Measure over a time partition instead of a surface one: a constant-Q ladder (with --level) or a spans tiling such as the chunk grid; the result is a time-resolved field.	none (a surface partition is measured)	-
--over@run null	NAME	The tile level Selection the crossings were measured over	the one the recordings have	-
--over@run repeat	NAME	The tile-level partition both recordings were measured over.	the one both recordings have (several is refused)	-
--over@run statistic	NAME	The partition measured over, by name: a stored selection whose members were measured (a parcellation, a level of a time ladder).	required	-
--overlap	FRACTION	Fraction by which successive Welch segments overlap.	0.5 (the Brainstorm option set)	fraction
--overlap@run crossspectrum	FRACTION	Fraction of a Welch segment shared with the next, in [0, 1); under the default tiles set it must keep the hop a power of two (0, 0.5, 0.75, ...).	0.5	fraction
--overlap@run hodgepower	FRACTION	The fraction of a Welch segment shared with the next, in [0, 1).	0.5	fraction
--overlap@run psd	FRACTION	Fraction of a segment shared with the next, in [0, 1)	0.5	fraction
--overlap@run quadrature	F	Fraction of a Welch segment shared with the next; under --set tiles the hop must also be a tile (0, 0.5, 0.75, ...).	0.5	fraction
--overlap@run sourcepsd	FRACTION	Fraction of a Welch segment shared with the next, in [0, 1); under the default tile set the hop must stay a power of two samples.	0.5	fraction
--overlap@run sourcetensor	FRACTION	Fraction by which successive Welch segments overlap.	0.5	fraction
--overlap@run speedspectrum	FRACTION	Fraction of a segment shared with the next, in [0, 1)	0.5	fraction
--oversample	X	Oversampling of each filterbank member's own sample rate.	2.0	-
--oversample@run opticalflow	X	Factor by which the filter bank's own sampling rate exceeds the number of frequency bins it keeps.	2	-
--package	STEM	Stem of the stored Hodge-Dirac eigenmode package to use	the one of the kernel's surface	-
--package@run cyclepattern	STEM	Stem of the stored Hodge-Dirac eigenmode package (with full forms) on the kernel's surface to use.	the one of the kernel's surface	-
--package@run hodgepower	NAME	The Hodge-Dirac package's name stem (hodgeDirac or hodgeDirac__<surface>).	the package of the kernel's surface	-
--package@run opticalflow	STEM	Stem of the stored Hodge-Dirac eigenmode package used for the flow.	the one of the kernel's surface	-
--package@run phasegradient	NAME	The stored Hodge-Dirac mode package that defines the spatial band (needs --spatial-band).	the kernel surface's own package	-
--package@run speedspectrum	STEM	Stem of the stored Hodge-Dirac mode package to use	the package of the kernel's surface	-
--padding	mean|zeros	How the filter's edge transient is padded: with the mean of the signal's transient or with zeros.	mean	-
--padding@run bandpass	mean|zeros	How the two ends of the recording are padded before filtering: with the mean of the first and last samples, or with zeros.	mean	-
--padding@run bandstop	mean|zeros	FIR end padding; not used by bandstop, which is an IIR filter.	mean (not used)	-
--padding@run highpass	mean|zeros	How the filter's edge transient is padded: with the mean of the transient or with zeros.	mean	-
--padding@run lowpass	mean|zeros	How the two ends of the record are padded for the filter: with the mean of the first and last transient samples, or with zeros.	mean	-
--padding@run notch	mean|zeros	Not used by notch (an IIR filter, whose ends use odd reflection); the end padding of the FIR filters.	mean	-
--pair	A,B	One pair of channels, by channel index (row A, column B), to read from the band-resolved cross-spectrum.	none	-
--pair@read crossspectrum	A,B	One pair of channels to read, by recording channel index (row A, column B); an alternative to --seed.	none (one of --pair or --seed)	-
--pairs	K	Number of strongest quadrature pairs stored per band	3	pairs
--pairs@run quadrature	K	Number of strongest quadrature pairs (travelling or standing components) kept per band.	3	-
--patterns	Q	Number of leading cycle patterns kept per spatial band.	3	-
--patterns@run cyclepattern	Q	Number of leading cycle patterns kept per spatial band (a whole number, at least 1).	3	-
--periodic		Use the periodic (DFT-even) form of the taper instead of the symmetric form.	off for the default and brainstorm sets (symmetric); on for scipy	-
--periodic@run crossspectrum		Use the periodic (DFT-even) form of the Welch taper instead of the symmetric form.	off (symmetric, as the default tiles and brainstorm sets)	-
--periodic@run hodgepower		Use the periodic (DFT-even) form of the window.	off (set tiles: symmetric)	-
--periodic@run psd		Use the periodic (DFT-even) form of the window	off (the default set uses the symmetric form)	-
--periodic@run quadrature		Use the periodic (DFT-even) form of the window; an alternative to --symmetric.	set-dependent (symmetric in tiles and brainstorm; periodic in scipy, nxr-chain)	-
--periodic@run sourcepsd		Use the periodic (DFT-even) form of the window instead of the symmetric form.	off (symmetric, as the default set)	-
--periodic@run sourcetensor		Use the periodic (DFT-even) form of the taper instead of the symmetric form.	off (symmetric, except the scipy set)	-
--periodic@run speedspectrum		Use the periodic (DFT-even) form of the window	off (the default set uses the symmetric form)	-
--phase-of	current|stream	Signal whose phase is taken: the band's current vector field, or its scalar stream function	current	-
--phase-of@run phasegradient	current|stream	Which signal's phase is taken (with --spatial-band): the band's current vector, or its stream function.	current	-
--preconditioner	ldlt|ic	Preconditioner of the flow's conjugate-gradient solve: ldlt (exact factor, built once) or ic (incomplete Cholesky).	ldlt	-
--preconditioner@run opticalflow	ldlt|ic	Preconditioner of the iterative solver (pcg, perscale): the exact factorisation ldlt, or incomplete Cholesky ic.	ldlt	-
--prefer	protocol.mat|walk	Which reader to prefer for the Brainstorm protocol: its protocol.mat file or a walk of the folders.	automatic	-
--prefer@import scan	protocol.mat|walk	Which reader to use for the protocol: its protocol.mat file or a walk of the folders.	automatic (protocol.mat when present)	-
--primary		Mark the added mesh as the subject's primary surface	on	-
--primary@add surface		Mark the added surface as the subject's primary surface (the one other commands use when no --surface is named).	on	-
--provenance	FILE	A JSON file of provenance to store with the imported maps.	none	-
--provenance@import map	FILE	A JSON file of provenance to store with each imported map.	none	-
--psd	NAME	The recording's stored Welch spectrum, whose shape gives the band's effective bandwidth.	none (required)	-
--psd@run wishart	NAME	The recording's stored Welch spectrum; it gives the band's effective bandwidth, hence the degrees of freedom.	required	-
--rank-tolerance	single|double	Precision used to decide the rank of the noise covariance: single or double precision.	single	-
--rank-tolerance@run inverse	single|double	Threshold for counting non-zero covariance eigenvalues: n x ulp of the largest, in single or double precision.	single	-
--reading	coherency|imag|phase|delay|psi	Quantity read from the stored bin-by-bin cross-spectrum (positive delay and psi: the row channel leads)	coherency	-
--reading@read crossspectrum	coherency|imag|phase|delay|psi	What to derive per pair from the bins: coherency, its imaginary part, phase, the phase-slope delay (ms) or the phase-slope index; positive delay and psi mean the row channel leads.	coherency	-
)TBL",
R"TBL(--recipe	NAME	Name of the recipe (the recorded set of choices) the measurements are made or read by.	measure compute: partition-<weight>; measure export: any	-
--recipe@run measure	NAME	Name of the recipe (the recorded set of choices) the measurements are stored under; surface partitions only.	partition-<weight>	-
--recipe@export measurements	NAME	Use only measurements made by this recipe (name or name@version); a subject with rows from another recipe is refused	none (any recipe)	-
--recording	NAME	Name of the stored recording to read.	none (required)	-
--recording-a	NAME	First recording, whose stored tile crossings are used (repeat: one of the two compared)	none (required)	-
--recording-a@run null	NAME	A recording whose stored tile crossings are tested against the null	required	-
--recording-a@run repeat	NAME	First recording, whose stored tile crossings are compared.	required	-
--recording-b	NAME	Second recording whose crossings are compared with the first (run-to-run repeat).	none	-
--recording-b@run null	NAME	A second recording, to test the run-to-run repeat as well (draws made in step)	none	-
--recording-b@run repeat	NAME	Second recording, whose crossings are compared with the first; the order does not change the result.	required	-
--recording@run bandpass	NAME	The stored recording to filter; all its channels and samples are kept.	required	-
--recording@run bandstop	NAME	Name of the stored recording to filter; the result is a new recording on the same channels and time line.	required	-
--recording@run constantq	NAME	Recording whose time line gets the ladder (alternative to --line and --all).	none	-
--recording@run crossspectrum	NAME	The stored sensor recording whose cross-spectrum is computed (its span is set by the time-range options).	required	-
--recording@run frametiling	NAME	The recording the tiling is made for.	required	-
--recording@run highpass	NAME	Name of the stored recording to filter.	required	-
--recording@run hodgepower	NAME	The sensor recording whose cross-spectrum is read through the kernel.	required	-
--recording@run inverse	NAME	Recording to compute (and store) the noise covariance from; an alternative to --noise-cov.	none (give this or --noise-cov)	-
--recording@run joint	NAME	The recording whose time line's constant-Q ladder gives the time level	required	-
--recording@run lowpass	NAME	Name of the stored recording to filter.	required	-
--recording@run noisecov	NAME	The stored recording from which the noise covariance is computed (its span is set by the time-range options).	required	-
--recording@run notch	NAME	The recording to filter.	required	-
--recording@run opticalflow	NAME	The stored recording whose band is analysed (its span is set by the time-range options).	required	-
--recording@run pointspread	NAME	The recording whose band chain the statistic is of	required	-
--recording@run psd	NAME	The recording whose channels' power spectra are estimated	required	-
--recording@run pyramid	NAME	Name of the stored recording to build the pyramid from.	required	-
--recording@run quadrature	NAME	Name of the stored recording to analyse.	required	-
--recording@run scaleselection	NAME	The recording whose scale pairs are selected over	required	-
--recording@run sourcepsd	NAME	Name of the stored recording to read.	required	-
--recording@run sourcetensor	NAME	Name of the stored recording to read.	required	-
--recording@run spatialnull	NAME	The recording whose stored spatial events are tested against the null.	required	-
--recording@run speedspectrum	NAME	The recording whose cortical activity is analysed	required	-
--recording@run wishart	NAME	The recording the windows are of.	required	-
--recording@import condition	FILE	Import only this data file of the condition (e.g. data_0raw_x.mat) instead of every data file the condition has.	none (all the condition's recordings)	-
--recording@import resample	NAME	The recording to resample	required	-
--reference-level	L	Spatial tile level whose power the blur patch is cut from.	6	level
--reference-level@run pointspread	LEVEL	The scale level whose power the patch is cut from	6	level
--region	a:b,c:d	Read only this region of the array: a start:stop range per dimension.	the whole array	samples
--region@read dataset	a:b,c:d	Read only this region of the array: one start:stop index range per dimension, stop excluded; used with --array.	the whole array	samples
--regularisation	ridge|diagonal|none	Regularisation of the noise covariance when whitening	ridge	-
--regularisation@run inverse	ridge|diagonal|none	How the whitener is regularised: ridge, diagonal (Brainstorm 'diag') or none.	ridge	-
--regulariser	connection|ambient	Smoothness term of the flow: connection (connection Laplacian on the signpost gauge) or ambient (Dirichlet energy of the ambient 3-vector field, Lefevre and Baillet).	connection	-
--regulariser@run opticalflow	connection|ambient	Smoothness term: connection (the connection Laplacian on the signpost gauge) or ambient (the Dirichlet energy of the ambient 3-vector field, Lefevre and Baillet 2008).	connection	-
--replace		Replace an existing result of the same name instead of refusing; the old one is moved aside, not deleted.	off	-
--resample	HZ	Resample the converted recordings to this sampling rate	none (keep the original rate)	Hz
--resample@import condition	HZ	Resample the condition's recordings to this sampling rate (polyphase resampling) when they are imported.	none (the original rate is kept)	Hz
--resample@import subject	HZ	Resample every recording of the subject to this sampling rate (polyphase resampling) when it is imported.	none (the original rate is kept)	Hz
--resolution	HZ	Frequency resolution on which the filter bank is designed.	0.5	Hz
--resolution@run opticalflow	HZ	Frequency resolution of the grid on which the filter bank is designed.	0.5	Hz
--resolve	band|bins	Whether to also keep each band's cross-spectrum bin by bin (bins, Welch estimator only) or only the band mean (band).	band	-
--resolve@run crossspectrum	band|bins	With bins, also store each band's cross-spectrum bin by bin (Welch estimator only); with band, store only the band mean.	band	-
--response	peak|mean	Which response of an event is compared across scale pairs: the largest standardized cell amplitude or their mean	peak	-
--response@run scaleselection	peak|mean	Event response used to pick the scale: the largest or the mean standardized cell amplitude	peak	-
--ridge-fraction	X	Ridge added in whitening, as a fraction of the mean eigenvalue of the noise covariance block (Brainstorm's noise regularisation, MNE-Python's reg).	0.1	fraction
--ridge-fraction@run inverse	E	Ridge size as a fraction of the covariance block's mean eigenvalue (Brainstorm's noise regularisation).	0.1	fraction
--ridge-over-rank		Take the ridge from the trace divided by the rank (MNE-Python) instead of by the number of channels (Brainstorm).	off	-
--ridge-over-rank@run inverse		Take the ridge's mean eigenvalue as trace over rank (MNE-Python) instead of trace over channel count.	off	-
--ripple	R	Passband and stopband deviation of the Kaiser-window FIR filter, as a linear value (0.001 is 60 dB attenuation)	0.001	-
--ripple@run bandpass	X	Passband and stopband deviation of the Kaiser-window FIR design (linear); 1e-3 is a 60 dB stopband attenuation.	0.001	-
--ripple@run bandstop	R	FIR (Kaiser) passband and stopband deviation; not used by bandstop, which is an IIR filter.	1e-3 (not used)	-
--ripple@run highpass	R	Passband and stopband deviation of the Kaiser-window FIR filter, as a linear value (0.001 is 60 dB attenuation).	0.001	-
--ripple@run lowpass	R	Passband and stopband deviation of the Kaiser-window FIR design (linear); 1e-3 gives 60 dB stopband attenuation.	1e-3	-
--ripple@run notch	X	Not used by notch; the passband and stopband ripple of the FIR filters.	0.001	fraction
--scale	X	Factor applied to the mesh coordinates to bring them to metres.	1	-
--scale-family	FAMILY	perscale flow method: the graph filterbank whose band-pass members give one flow per spatial scale (mexhat, itersine or logitersine).	none (required for perscale)	-
--scale-family@run opticalflow	mexhat|itersine|logitersine	perscale only: the graph filter bank whose band-pass members each give one flow per spatial scale.	required for perscale	-
--scale-filters	N	Number of members of the per-scale graph filterbank (method perscale); 0 derives it from the voices	0	members
--scale-filters@run opticalflow	N	perscale only: number of members of the spatial filter bank; 0 derives it from --scale-voices.	0	-
--scale-voices	N	Per-scale flow method: voices per octave of the graph filter bank.	3	voices/octave
--scale-voices@run opticalflow	N	perscale only: members of the spatial filter bank per octave.	3	voices/octave
--scale@add surface	X	Factor multiplied into every vertex coordinate of the mesh file to bring it to metres.	1	-
--scaling	density|spectrum|chain	How the spectrum is scaled: power per Hz, power per bin, or the chain scaling (measure export: stored|density|power|amplitude|amplitude-density|db|db-density, applied to the stored values).	density (measure export: stored)	-
--scaling@run crossspectrum	density|spectrum|chain	Scaling of the Welch estimate: power per hertz, power per bin, or the nxr chain's scaling.	density	-
--scaling@run hodgepower	density|spectrum|chain	How the Welch power is scaled: per Hz, as a power spectrum, or as nxr's chain.	density	-
--scaling@run psd	density|spectrum|chain	Scaling of the estimate: power per Hz, power per bin, or the nxr spectral chain's convention	density	-
--scaling@run quadrature	density|spectrum|chain	Scaling of the Welch spectrum: power per Hz, power per bin, or nxr's chain scaling.	density	-
--scaling@run sourcepsd	density|spectrum|chain	Scaling of the spectrum: power per Hz, power per bin, or the chain scaling.	density	-
--scaling@run sourcetensor	density|spectrum|chain	How the spectrum is scaled: power per Hz (density), power per bin (spectrum), or the chain scaling.	density	-
--scaling@run speedspectrum	density|spectrum|chain	Scaling of the spectral estimate: power per Hz, power per bin, or the nxr spectral chain's convention	density	-
--scaling@export measurements	NAME	Scaling applied to the values at read: stored, density, power, amplitude, amplitude-density, db or db-density (needs the Field's recorded spectral scaling)	stored	-
--seed	N	crossspectrum read: the channel index to read against every channel (alternative to --pair); wishart: seed of the random draws	wishart: 1; crossspectrum read: none	-
--seed@run wishart	N	Seed of the random draws; the same seed reproduces the same null.	1	-
--seed@read crossspectrum	A	Read channel A (a recording channel index) against every channel of the statistic; an alternative to --pair.	none (one of --pair or --seed)	-
--segment-level	L	Welch segment of 2^L samples (an alternative to the segment length in seconds or samples).	none (the segment length option is used)	level
--segment-level@run crossspectrum	L	Welch segment of 2^L samples (tiles set only); an alternative to --segment-seconds and --segment-samples.	none (the segment length in seconds is used)	level
--segment-level@run hodgepower	K	The Welch segment is a tile of 2^K samples (set tiles only); alternative to --segment-seconds and --segment-samples.	round(log2(4 s x sampling rate))	level
--segment-level@run psd	LEVEL	Segment length as a tile level: 2^level samples; an alternative to --segment-seconds and --segment-samples	none	level
--segment-level@run quadrature	K	Welch segment as a tile of 2^K samples; only under --set tiles; an alternative to --segment-seconds and --segment-samples.	the level nearest 4 s	level
--segment-level@run sourcepsd	L	Welch segment length as a tile level: 2^L samples (alternative to --segment-seconds and --segment-samples; tiles set only).	none (the 4 s segment is used)	level
--segment-level@run sourcetensor	L	Welch segment of 2^L samples (an alternative to the segment length in seconds or samples).	none (the 4 s segment is used)	level
)TBL",
R"TBL(--segment-level@run speedspectrum	LEVEL	Segment length as a tile level: 2^level samples; an alternative to --segment-seconds and --segment-samples	none	level
--segment-samples	N	Welch segment length in samples; wins over --segment-seconds.	derived from the 4 s segment	samples
--segment-samples@run crossspectrum	N	Welch segment length in samples; under the tiles set it must be a power of two. Alternative to --segment-seconds and --segment-level.	none (derived from the 4 s segment)	samples
--segment-samples@run hodgepower	N	The Welch segment length in samples; alternative to --segment-seconds and --segment-level.	none	samples
--segment-samples@run psd	N	Segment length in samples; an alternative to --segment-seconds and --segment-level	none	samples
--segment-samples@run quadrature	N	Welch segment length in samples (a power of two under --set tiles); an alternative to --segment-seconds and --segment-level.	none	samples
--segment-samples@run sourcepsd	N	Welch segment length in samples; under the tiles set it must be a power of two.	none (from --segment-seconds)	samples
--segment-samples@run sourcetensor	N	Welch segment length in samples; an alternative to the other segment options.	derived from the 4 s segment	samples
--segment-samples@run speedspectrum	N	Segment length in samples; an alternative to --segment-seconds and --segment-level	none	samples
--segment-seconds	SECONDS	Length of each Welch segment (under the tiles setting it is snapped to a power-of-two number of samples)	4	s
--segment-seconds@run crossspectrum	SECONDS	Length of each Welch segment; under the tiles set it is snapped to a power-of-two number of samples.	4	s
--segment-seconds@run hodgepower	SECONDS	The Welch segment length in seconds; alternative to --segment-samples and --segment-level.	4	s
--segment-seconds@run psd	SECONDS	Segment length in seconds; under the tiles set it is rounded to a 2^k-sample segment	4	s
--segment-seconds@run quadrature	SECONDS	Welch segment length; under --set tiles it snaps to the nearest 2^k-sample tile.	4	s
--segment-seconds@run sourcepsd	SECONDS	Welch segment length; under the tiles set it snaps to the nearest power-of-two number of samples.	4	s
--segment-seconds@run sourcetensor	SECONDS	Length of each Welch segment; under the tiles set it is snapped to a power-of-two number of samples.	4	s
--segment-seconds@run speedspectrum	SECONDS	Segment length in seconds; under the tiles set it is rounded to a 2^k-sample segment	4	s
--selection	NAME	The stored joint Selection (space x time cells) whose measurement is read.	none (required)	-
--selection@query measurements	NAME	The pyramid's joint selection or spatial partition to read, by name or id; it must be complete.	required	-
--sensor-weight-exponent	P	Exponent of the distance weighting used to fit each sensor's overlapping sphere.	4	-
--sensor-weight-exponent@run headmodel	P	The exponent p in the overlapping-sphere fit's weights, which fall with the distance d from the sensor as d^-p.	4	-
--set	NAME=VALUE	Set a pipeline parameter, NAME=VALUE with VALUE read as YAML (8, [15, 29], dSPM); repeatable.	none	-
--set@run crossspectrum	tiles|brainstorm|scipy|nxr-chain	Reference whose Welch settings are the starting point (Hamming symmetric taper, 50 % overlap, constant detrend, density for tiles and brainstorm).	tiles	-
--set@run hodgepower	tiles|brainstorm|scipy|nxr-chain	The named set of Welch options to start from: tiles (Brainstorm's taper, detrend and scaling on 2^k-sample segments), brainstorm, scipy or nxr-chain.	tiles	-
--set@run null	NAME=VALUE	Parameter of every null draw (repeatable); {recording} stands for each recording's name	none	-
--set@run psd	tiles|brainstorm|scipy|nxr-chain	Named set of Welch options to start from (Brainstorm's taper, detrend and density on 2^k-sample segments; Brainstorm's exact 4 s; scipy.signal.welch; or the nxr chain)	tiles	-
--set@run quadrature	tiles|brainstorm|scipy|nxr-chain	Option set for the Welch estimate: tiles (Brainstorm's taper and detrend on 2^k-sample segments), brainstorm, scipy or nxr-chain.	tiles	-
--set@run sourcepsd	tiles|brainstorm|scipy|nxr-chain	Reference whose Welch settings (window, overlap, detrend, scaling, segment rule) are the starting point; other options then override them.	tiles	-
--set@run sourcetensor	tiles|brainstorm|scipy|nxr-chain	The reference set of Welch options to start from (taper, overlap, detrend, scaling); the other Welch options then override it.	tiles	-
--set@run spatialnull	NAME=VALUE	A parameter of every null draw's spatial-events run (levels, threshold, band, ...); repeatable.	none (the data's parameters)	-
--set@run speedspectrum	tiles|brainstorm|scipy|nxr-chain	Named set of Welch options to start from (Brainstorm's taper, detrend and density on 2^k-sample segments; Brainstorm's exact 4 s; scipy.signal.welch; or the nxr chain)	tiles	-
--set@run	NAME=VALUE	Override a parameter of the pipeline definition; the value is read as YAML (8, [15, 29], dSPM). Repeatable.	none (the definition's own values)	-
--shape	morlet|meyer|power_partition	Band filter shape: morlet (Gaussian centred on the band, FWHM equal to its width) or meyer (wishart); pyramid build also accepts power_partition (band powers add to broadband).	morlet	-
--shape@run pyramid	morlet|power_partition	Band filter shape: morlet (Gaussian centred on the band, FWHM equal to its width) or power_partition (band powers add up to broadband).	morlet	-
--shape@run wishart	morlet|meyer	The band's analytic gain used for its effective bandwidth: morlet or meyer.	morlet	-
--shift	absolute|relative	How the eigenvalue solve is shifted: at one absolute shift (reference method) or by a scale-aware relative ladder.	absolute	-
--shift@run eigenmodes	absolute|relative	Shift of the shift-invert solve: absolute (one solve per ladder at --sigma, the reference method) or relative (a ladder scaled to the spectrum).	absolute	-
--side	auto|left|right	Hemisphere of a single-piece surface	auto	-
--side@run frames	auto|left|right	Hemisphere of a single-piece surface: auto decides each piece by position.	auto	-
--sides	analytic|two-sided	Whether the time bank is one-sided (analytic, positive frequencies) or two-sided.	none (required)	-
--sides@run speedspectrum	analytic|two-sided	Whether the time bank has positive-frequency members only (one-sided cross-spectrum) or members at both +f and -f	required	-
--sigma	S	Absolute shift of the eigenvalue solve (with --shift absolute).	-0.001	1/m^2
--sigma@run eigenmodes	S	The absolute shift of the shift-invert solve (only with --shift absolute).	-1e-3	-
--sign	positive|negative|magnitude	Which tile means form patches: high values, low values, or absolute value	positive	-
--sign@run patches	positive|negative|magnitude	Which tile means the thresholds apply to: high values (bumps), low values (dips) or absolute values.	positive	-
--signal	normal|ambient	Signal whose phase gradient is taken: ambient (the full current, its x y z components) or normal (the component along the surface normal).	ambient	-
--signal@run phasegradient	normal|ambient	The current whose phase is measured: ambient (its x y z components) or normal (the component normal to the cortex).	ambient	-
--slice	N	Use spectrum slicing with windows of N modes; 0 means one solve.	0 (none)	modes
--slice@run eigenmodes	N	Spectrum slicing: solve in windows of N modes, each with its own shift; 0 means one solve per ladder.	0 (no slicing)	modes
--snr	S	Assumed amplitude signal-to-noise ratio of the minimum-norm kernel	3	-
--snr-normalisation	all-channels|whitener-rank	Whether the signal-to-noise ratio is normalised over all channels or over the rank of the whitener.	all-channels	-
--snr-normalisation@run inverse	all-channels|whitener-rank	Mean eigenvalue used to scale Lambda: over all channels (Brainstorm) or over the whitener's rank (MNE-Python).	all-channels	-
--snr@run inverse	S	Assumed amplitude signal-to-noise ratio that sets the regularisation Lambda (Lambda^2 = 1 / snr^2 in MNE-Python's terms).	3	-
--source	NAME	The pipeline whose stored cells and events are read.	cortical_event_pairs	-
--source-path	PATH	Records the path of the source data in the dataset's provenance	none	-
--source-path@create dataset	PATH	Records the path of the source data the dataset was made from, in the dataset's provenance.	none	-
--source-protocol	NAME	Record the name of the source protocol the dataset was made from.	none	-
--source-protocol@create dataset	NAME	Records the name of the source protocol the dataset was made from, in the dataset's provenance.	none	-
--source-surface	NAME	Surface whose vertices are the sources of the head model.	the primary (cortex) surface	-
--source-surface@run headmodel	NAME	The surface whose vertices are the sources.	the primary (cortex) surface	-
--source-tool	NAME	Records the tool that produced the source data in the dataset's provenance	none	-
--source-tool@create dataset	NAME	Records the tool that produced the source data, in the dataset's provenance.	none	-
--source@run scaleselection	NAME	The pipeline whose stored labellings, cells and baselines are read	cortical_event_pairs	-
--space-level	L	Level of the surface's tile ladder that gives the spatial tiles (-1 on measure query: the whole cortex).	measure query: the stored depth; others none (required)	level
--space-level@run cellquadrature	L	Level of the surface's tile ladder; its Weyl band is the spatial band and its tiles the spatial cells.	required	level
--space-level@run joint	LEVEL	The surface tile ladder's level (2^level tiles per piece)	required	level
--space-level@query measurements	LEVEL	The spatial level the cells are rolled up to (at most the stored depth; -1 is the whole cortex).	the stored depth	level
--space-levels	L,...	Spatial tile levels of the surface to select over (all combinations with --time-levels).	none (required)	levels
--space-levels@run scaleselection	LEVELS	Comma-separated surface tile levels; the pairs are all space levels times time levels	required	level
--spatial-band	BAND	Spatial band: modes, weyl:J or weyl:all for the phase and cycle patterns (all modes, one Weyl octave J, each member of the Weyl bank); for measure query the whole-number index of a stored spatial band	phasegradient: none (the kernel as it is); cyclepattern: weyl:all; measure query: none	-
--spatial-band@run cyclepattern	BAND	Spatial band: modes (all modes), weyl:J (one Weyl octave J) or weyl:all (every member of the Weyl bank).	weyl:all	-
--spatial-band@run phasegradient	modes|weyl:J|weyl:all	Measure at a spatial scale: every mode, one Meyer octave J of the Laplacian spectrum, or each octave; unset uses the kernel as it is.	none (the kernel as it is)	-
--spatial-band@query measurements	N	The spatial band (eigenmode level) of the cells to read.	none	-
--spatial-shape	NAME	Shape of the spatial scale-space filter at each level (mexhat is the Mexican hat).	mexhat	-
--spatial-shape@run pointspread	NAME	Gain of each spatial level: mexhat (Mexican hat) or meyer (the level is its own Weyl band)	mexhat	-
--speed-bins	N	Number of log-spaced phase-speed bins of the speed spectrum.	40	bins
--speed-bins@run speedspectrum	N	Number of log-spaced phase speed bins of the histogram	40	-
--speed-max	C	Largest phase speed of the speed histogram bins	from the spectrum (the largest 2 pi f / mu over modes and bins)	length/s
--speed-max@run speedspectrum	C	Largest phase speed of the histogram; 0 takes it from the spectrum	0	m/s
--speed-min	X	Smallest phase speed of the log-spaced speed bins.	0 (from the spectrum)	length/s
--speed-min@run speedspectrum	C	Smallest phase speed of the histogram; 0 takes it from the spectrum	0	m/s
--sphere-cost	geometric|algebraic	Cost the head sphere is fitted with: distance to the sphere (geometric) or the linear closed-form fit (algebraic).	geometric	-
--sphere-cost@run headmodel	geometric|algebraic	The cost minimised when fitting the spheres: distances to the sphere, or the closed-form algebraic cost.	geometric	-
)TBL",
R"TBL(--stage	NAME|INDEX	Stage of the stored crossings to use, by name (burst, onset, within, offset) or by index	0 (the first stage, burst)	-
--stage@run null	NAME|INDEX	The stage of the stored crossings to test, by name (burst, onset, within, offset) or index	0	-
--stage@run repeat	NAME|INDEX	Stage of the stored crossings to compare, by name (burst, onset, within, offset) or by index.	0 (the first stage, burst)	-
--star1-floor	X	Floor on the cotangent weights of the Hodge star; values at or below 0 mean no floor.	none	-
--star1-floor@run eigenmodes	F	Raise the edge cotan weights (the 1-form Hodge star) that fall below F up to F; unset means no floor.	none	-
--start	SEC	Start of the span read, in seconds on the recording's clock; give together with --stop. Exclusive with --chunks.	none (one chunk)	s
--start-s	SECONDS	Time of the recording's first sample as seconds from local midnight, anchoring the time-tile levels	the anchor is the recording	s
--start-s@run frametiling	SECONDS	The recording's first sample as seconds from local midnight, which anchors the tiling to the day.	none (anchored to the recording)	s
--start-s@run opticalflow	SECONDS	Time of the recording's first sample as seconds from local midnight, anchoring the frame tiling.	none (the anchor is the recording)	s
--start-s@run pyramid	SECONDS	Time of the recording's first sample as seconds from local midnight, which anchors the time frames.	the anchor is the recording	s
--statistics	LIST	Comma-separated statistics written per subject in the exported table (e.g. mean,sd).	see the command's description	-
--statistics@export measurements	NAMES	Comma-separated statistics to export: mean, sd, variance, rms, mean_square, max, min, abs_max, sum, area, n_elements	mean	-
--stop	SEC	End of the span read, in seconds on the recording's clock (exclusive); give together with --start.	none (one chunk)	s
--store-format	51|52	Array storage layout of the new dataset (52 keeps small keys in a table)	52	-
--store-format@create dataset	51|52	Array storage layout of the new dataset: 52 keeps each subject's small keys in a table, 51 keeps only the folder tree.	52	-
--subject	NAME	The subject of the dataset to work on.	required	-
--subjects	A,B,...	Subjects to include, by name.	every subject of the dataset (pyramid reduce: required)	-
--subjects@export measurements	NAMES	Comma-separated subjects to include	every subject of the dataset	-
--subjects@reduce pyramid	A,B,...	The member subjects whose group-tree pyramids are summed into the group result, by name.	required	-
--summaries	LIST	Comma-separated summaries across the members, such as median, mad, percentile:75, mean, sd, peak, kurtosis, skewness, min, max	statistic: none (required); patches: count,sum,median,mean,sd,percentile:25,percentile:75,percentile:90	-
--summaries@run patches	A,B,...	The statistics taken across the patches, e.g. count, sum, median, mean, sd, percentile:25.	count,sum,median,mean,sd,percentile:25,percentile:75,percentile:90	-
--summaries@run statistic	LIST	Comma-separated summaries across the members: median, mad, percentile:P, mean, sd, peak, kurtosis, skewness, min, max.	required	-
--summary		Print a one-line summary of the scan.	off	-
--summary@import scan		Print a one-line summary (kind, root, counts) instead of the whole view.	off	-
--surface	NAME	Surface to work on, by name.	the primary (cortex) surface (joint, scaleselection: the one the joint partitions use)	-
--surface-id	ID	The id of the surface the kernel was made on.	found from the kernel file	-
--surface-id@import inverse	ID	Id of the stored cortex surface the kernel was made on.	found from the kernel's SurfaceFile link	-
--surface@run eigenmodes	NAME	Name of the subject's stored surface to solve on.	the subject's primary surface	-
--surface@run frames	NAME	The subject's surface to compute the frames on.	the primary (cortex) surface	-
--surface@run joint	NAME	The subject's surface whose tile ladder gives the spatial level	the subject's primary surface	-
--surface@run scaleselection	NAME	The subject's surface	the surface of the joint partitions	-
--surface@run tiles	NAME	The subject's surface whose tile ladder is computed, by name.	the primary (cortex) surface	-
--surface@import condition	FILE	The subject's cortex surface file (tess_*.mat) the condition is imported on, instead of the one its head model names.	none (the condition's own surface)	-
--surface@import template	NAME	Convert only this surface (a tess_*.mat file of the template anatomy); repeat for several.	every surface of the template's default anatomy	-
--surface@import templates	NAME	The surface the template maps are placed on	the subject's surface with each map's vertex count	-
--surface@list spatial-bands	NAME	The subject's surface whose spatial band bank is listed	the subject's primary surface	-
--symmetric		Use the symmetric form of the taper window instead of the periodic one	off (the window form follows --set)	-
--symmetric@run crossspectrum		Use the symmetric form of the Welch taper instead of the periodic form.	off (the form follows --set)	-
--symmetric@run hodgepower		Use the symmetric form of the window.	off (set tiles: symmetric)	-
--symmetric@run psd		Use the symmetric form of the window (MATLAB-style)	on under the default set	-
--symmetric@run quadrature		Use the symmetric form of the window (as MATLAB's hamming(n)); an alternative to --periodic.	set-dependent (symmetric in tiles and brainstorm)	-
--symmetric@run sourcepsd		Use the symmetric form of the window instead of the periodic form.	on for tiles and brainstorm sets (periodic for scipy)	-
--symmetric@run sourcetensor		Use the symmetric form of the taper window instead of the periodic one.	off (the form follows --set)	-
--symmetric@run speedspectrum		Use the symmetric form of the window (MATLAB-style)	on under the default set	-
--table	NAME	Name of the exported table.	none (required)	-
--table@export measurements	NAME	Name of the table; it names the data_table and the output files	required	-
--template	SUBJECT	Template subject whose anatomical frame is carried over through the registered spheres.	none (the surface's own axes)	-
--template-surface	NAME	Surface of the template subject whose frame is carried over	the template's primary surface	-
--template-surface@run frames	NAME	The template subject's surface whose frame is carried over.	the template's primary surface	-
--template@run frames	SUBJECT	Template subject of the same dataset whose anatomical frame is carried over through the registered spheres.	none (the surface's own axes)	-
--templates	DIR	A folder of released templates to look in (repeatable), before NXR_TEMPLATE_PATH and the user setting template_path; read only when a template's arrays are used.	none (the environment variable and the user setting only)	-
--templates@create dataset	DIR	A FreeSurfer subjects folder; the dataset is created together with its default subject (surfaces, parcellations, frames, group tree) read from it.	none (no default subject is written)	-
--threads	N	Workers for the kernel products; the result is identical for every count (0 = the machine's performance cores, 1 = serial).	0	threads
--threads@run pyramid	N	Workers for the kernel products; the result is identical for every count (0 = the machine's performance cores, 1 = serial).	0	threads
--through	NAME	Name of a stored 2-row operator (a rotation plane) over the channels: compute the two-sided rotary spectrum of its complex signal	none	-
--through@run psd	NAME	Name of a stored 2-row rotary operator over the channels; the two-sided PSD of the complex series it forms is computed instead	none	-
--tile-levels	L1,L2,..	Levels of the tile ladder at which tile summaries of the maps are written (needs --tiles).	1 up to --tile-max-level (6)	levels
--tile-levels@run opticalflow	L1,L2,...	Levels of the tile ladder at which tile summaries of the maps are written; needs --tiles.	1 up to --tile-max-level	level
--tile-levels@run quadrature	L1,L2,...	With --tiles: the tile-ladder levels to summarise.	1 up to --tile-max-level	level
--tile-max-level	N	Write tile summaries for levels 1 to N (with --tiles).	6	level
--tile-max-level@run opticalflow	L	Highest tile level of the summaries when --tile-levels is not given; needs --tiles.	6	level
--tile-max-level@run quadrature	L	With --tiles: the finest level summarised when --tile-levels is not given.	6	level
--tile-pairs	on|off	Also compute the blocks between touching tiles (waves crossing from one tile into the next)	on	-
--tile-pairs@run cellquadrature	on|off	Also compute the blocks between touching tiles, so a wave running from one tile into the next is captured.	on	-
--tiles		Also write the maps' tile summaries per level of the tile ladder.	off	-
--tiles@run opticalflow		Also write the maps' tile summaries per level of the tile ladder (scalars area-weighted, vectors in the frame and Levi-Civita forms).	off	-
--tiles@run quadrature		Also write the maps' tile summaries per ladder level (scalars area-weighted, vectors in the canonical frame and the Levi-Civita form).	off	-
--time-bandwidth	B	Morse family: the time-bandwidth product of the wavelet.	60	-
--time-bandwidth@run opticalflow	X	Morse family only: the time-bandwidth product of the wavelet.	60	-
--time-bandwidth@run speedspectrum	B	Morse time bank: the time-bandwidth product	60	-
--time-level	M	Level of the recording's constant-Q time ladder (joint compute: the time tiles of the cells; measure query: time rolled up to tiles of 2^M frames)	measure query: 0; joint compute: none (required)	level
--time-level@run joint	LEVEL	The recording's constant-Q ladder level (its member ordinal; the band it resolves)	required	level
--time-level@query measurements	LEVEL	The time level the cells are rolled up to: tiles of 2^LEVEL frames on the tower (0 is single frames).	0	level
--time-levels	M1,M2,..	Constant-Q time levels (each one's band is a pair's rhythm) crossed with the space levels.	none (required)	levels
--time-levels@run scaleselection	LEVELS	Comma-separated constant-Q time levels, each giving the pair's rhythm band	required	level
--timing-of	FIELD,...	Also store the timing map (lags) of these fields of each pattern: stream, potential, normal.	none	-
--timing-of@run cyclepattern	FIELD,...	Also store the timing map (lags) of these fields of each pattern: stream, potential, normal.	none	-
--to	HZ	The sampling rate to resample to (import resample).	none	Hz
--to@import resample	HZ	The new sampling rate; the recording is resampled by a polyphase filter (scipy resample_poly)	required	Hz
--tolerance	T	lowpass, notch: IIR look-behind tail left out, relative to the response's L1 norm; pointspread, cellquadrature, patches: largest gain error accepted for the Chebyshev series; opticalflow: tail of the filters' impulse response the margin may leave out	filters: 1e-15; pointspread, cellquadrature, patches: 0.01; opticalflow: from the recording	-
--tolerance@run bandpass	X	Impulse-response tail that the chunked IIR margin may leave out; it applies to IIR filters (notch, band-stop) and has no effect on the FIR band-pass.	1e-15	-
--tolerance@run bandstop	T	Chunked IIR: the impulse-response tail, relative to its L1 norm, that the look-behind and look-ahead margin may leave out.	1e-15	-
--tolerance@run cellquadrature	T	Largest gain error accepted for the Chebyshev series of the spatial band.	0.01	-
--tolerance@run highpass	T	Not used by highpass: it is a FIR design; the IIR look-behind tail tolerance is for the notch and bandstop commands.	1e-15	-
--tolerance@run lowpass	T	Tail of the impulse response, relative to its L1 norm, that the chunk look-behind/ahead of an IIR filter may leave out; not used by the FIR lowpass.	1e-15	-
--tolerance@run notch	X	How far the notch's impulse-response tail may be cut when filtering chunk by chunk, relative to its L1 norm.	1e-15	fraction
--tolerance@run opticalflow	X	Impulse-response tail the filter bank's margin may leave out, relative to its norm.	derived from the recording's stored precision (1e-7 float32, 1e-12 float64)	-
)TBL",
R"TBL(--tolerance@run patches	X	Chebyshev: the largest difference between the band's gain and the series that is accepted; a worse fit is refused.	0.01	fraction
--tolerance@run pointspread	TOL	Largest gain error accepted for a Chebyshev series; a band the series cannot follow is refused	0.01	-
--transform	auto|eigen|chebyshev|exact	How the spatial band is applied: auto (stored eigenmodes; else computed, else Chebyshev announced), eigen, chebyshev (series in the pencil) or exact (dense modes, at most 4096 vertices).	auto	-
--transform@run cellquadrature	auto|eigen|chebyshev|exact	How the spatial band is applied: stored eigenmodes first (auto), eigen, a Chebyshev series, or exact dense modes.	auto	-
--transform@run patches	auto|eigen|chebyshev|exact	The spatial transform: stored eigenmodes (computed if absent), eigenmodes, a Chebyshev series, or dense modes of the mesh (up to 4096 vertices).	auto	-
--transform@run pointspread	auto|eigen|chebyshev|exact	How the spatial filter is applied: auto uses stored eigenmodes, else computes them, else an announced Chebyshev series; exact uses dense modes (at most 4096 vertices)	auto	-
--transient-samples	K	Number of samples at each edge treated as the filter transient (instead of the shortest span holding 99 percent of the impulse response's energy).	from the energy rule	samples
--transient-samples@run bandpass	N	Number of end samples whose mean pads the recording's ends; sets the length of the end padding instead of the 99 % energy rule.	the duration holding 99 % of the impulse response's energy	samples
--transient-samples@run bandstop	K	FIR end padding length; not used by bandstop, which is an IIR filter.	none (not used)	samples
--transient-samples@run highpass	K	Number of samples at each edge treated as the filter transient, instead of the shortest span holding 99 percent of the impulse response's energy.	from the energy rule	samples
--transient-samples@run lowpass	N	Length K of the end padding, stated in samples instead of the shortest impulse response holding 99 % of its energy.	none (energy rule)	samples
--transient-samples@run notch	N	Not used by notch; the end-padding length of the FIR filters.	none	samples
--transition	HZ	Width of each transition band of the FIR filter	the reference's rule (Brainstorm)	Hz
--transition@run bandpass	HZ	Width of each transition band of the FIR design.	0.5 (1 when the lower edge is above 5 Hz)	Hz
--transition@run bandstop	HZ	FIR transition bandwidth; not used by bandstop, which is an IIR filter.	none (not used)	Hz
--transition@run highpass	HZ	Width of the transition band of the FIR filter (the high-pass edge's).	the reference's rule (Brainstorm)	Hz
--transition@run lowpass	HZ	Width of the transition band between passband and stopband.	1 for a lowpass alone (the reference's rule)	Hz
--transition@run notch	HZ	Not used by notch; the transition bandwidth of the FIR filters.	none	Hz
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
--templates@list spatial-bands	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this listing uses an array of the template.	none (the environment variable and the user setting only)	-
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
--templates@run noisecov	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run inverse	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run headmodel	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run pointspread	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run eigenmodes	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run frames	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run tiles	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run constantq	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run frametiling	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run joint	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run lowpass	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run highpass	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run bandpass	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run bandstop	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run notch	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run psd	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run sourcepsd	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run sourcetensor	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run crossspectrum	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run quadrature	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run hodgepower	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run speedspectrum	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run phasegradient	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run cyclepattern	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run opticalflow	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run cellquadrature	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run patches	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run scaleselection	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run pyramid	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run measure	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run statistic	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run wishart	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run spatialnull	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run null	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
--templates@run repeat	DIR	A folder of released templates to look in (repeatable) for the template the subject's package is registered to, before NXR_TEMPLATE_PATH and the user setting template_path; read only when this computation uses an array of the template.	none (the environment variable and the user setting only)	-
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
--where@list subjects	FILTER	List the subjects whose participant variables satisfy this filter (version 1 of the filter language); the dataset must be a catalogue with a participants table.	none (all subjects are listed)	-
--subjects@list subjects	A,B,...	List these subjects, by name or id (a catalogue dataset); not together with --where.	none (all subjects are listed)	-
--force@create index		Rebuild the index even when it is current.	off	-
--tsv@import participants	FILE	The BIDS participants.tsv to import: tab-separated, a participant_id column matched to subject names, n/a for missing.	required	-
--sidecar@import participants	FILE	The BIDS participants.json sidecar: Description, Units, Levels and Type of each variable.	none (types are inferred)	-
--tsv@export participants	FILE	The file to write the participants table to as BIDS participants.tsv; nothing else is written.	required	-
--sidecar@export participants	FILE	Also write the participants.json sidecar to this file.	none	-
--registered-by@split dataset	WHO	Who runs the split: recorded with each catalogue entry.	required	-
--dry-run@split dataset		Report the subjects and the format without changing anything (not even bringing the dataset to format 52).	off	-
--package@verify package	PATH	The folder of the subject package to verify.	required	-
--package@show package	PATH	The folder of the subject package to show.	required	-
--package@release template	PATH	The folder of the template or registration package to release; it is left as it was.	required	-
--name@release template	NAME	Name the template is released under.	required	-
--template-version@release template	VERSION	Version string of the release (e.g. 1.0); name@version must be new, a released version is frozen.	required	-
--released-by@release template	WHO	Who releases: the program and the releaser's stated name, recorded in the release.	required	-
--into@reduce pyramid	NAME	On a catalogue dataset: the name of the group package the result is written to (made if absent, else its group fields are replaced); one per analysis, never a default one. Not for a one-database dataset, whose group lives in its default subject.	required for a catalogue	-
--templates@reduce pyramid	DIR	A folder of released templates to look in (repeatable) for the default subject the members' packages are registered to, before NXR_TEMPLATE_PATH and the user setting template_path; the dataset's own template member is found first.	none (the environment variable and the user setting only)	-
--into@release template	DIR	The templates folder the frozen copy <name>@<version>.nxr.zarr is made in.	required	-
--package@release redistributable	PATH	The folder of the released full edition to make the redistributable edition from.	required	-
--into@release redistributable	DIR	The templates folder the redistributable edition is made in.	required	-
--released-by@release redistributable	WHO	Who releases: recorded in the release.	required	-
--from-sphere	NAME	The surface of the first template that carries its registered sphere (e.g. cortex_pial_sphere).	none (required)	-
--to-sphere	NAME	The surface of the second template that carries its registered sphere (e.g. cortex_pial_sphere).	none (required)	-
--neighbours	K	Modified Shepard weights: the number of nearest source vertices looked at.	8	vertices
--one-direction		Make only the operator from the first template to the second.	off (both directions are made)	-
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
--hcp-trk@create template	FILE	A DSI Studio .trk of streamlines (icbm152 only); imported by its MNI coordinates with its origin (CC BY-SA 4.0 and the HCP terms).	none	-
--hcp-end-surface@create template	NAME	The surface of the package the streamline ends are placed on (e.g. cortex_pial_high); a choice of data.	required with --hcp-trk	-
--max-end-distance@create template	MM	An end farther than this from the end surface is left out of the ends operator; every end's distance is stored either way.	5	mm
--hcp-points@create template	N	Points per streamline after resampling at equal arc length; the two ends are kept exactly.	64	points
--accept-unlisted-sources@create template		Read a source file the manifest does not list or whose checksum differs; the checksum read is recorded in the origin.	off (refused)	-
--home-dataset@create template	NAME	The dataset the package is made for, by its name: provenance, not membership.	templates	-
--tree	subject|group	The tree the leaves belong to: the surface's own ladder, or the dataset's group tree through the registered sphere (the same leaves in every subject).	subject	-
--tree@run pyramid	subject|group	Tree the leaf tiles belong to: the surface's own ladder, or the dataset's group tree (the same leaves in every subject).	subject	-
--trk	FILE	A fibre-tract file (.trk) to import with the templates.	none	-
--trk@import templates	FILE	A tractography streamline file (.trk) whose streamlines and their end points are added to the templates	none	-
--types	TYPE,...	Channel types to use, e.g. "MEG MAG,MEG GRAD,EEG" (only good channels of these types).	MEG, MEG MAG, MEG GRAD, EEG, SEEG, ECOG	-
--types@run crossspectrum	TYPE,...	Channel types to include (only the good channels of these types), e.g. "MEG MAG,MEG GRAD,EEG"; an alternative to --channels-of.	MEG, MEG MAG, MEG GRAD, EEG, SEEG, ECOG	-
--types@run inverse	TYPE,...	With --recording: channel types for the noise covariance, e.g. "MEG MAG,MEG GRAD,EEG" (good channels only).	MEG, MEG MAG, MEG GRAD, EEG, SEEG, ECOG	-
--types@run noisecov	TYPE,...	Channel types the covariance covers (only the good channels of these types), e.g. "MEG MAG,MEG GRAD,EEG".	MEG, MEG MAG, MEG GRAD, EEG, SEEG, ECOG	-
--vertices	N	Number of vertices, spread evenly over the surface, where unit currents are placed	20	vertices
--vertices@run pointspread	N	Number of native vertices planted with a unit current, spread evenly over the surface (lowest index first)	20	-
--voices	N	Filter-bank members per octave (constantq: time-ladder voices; opticalflow, speedspectrum: itersine voices).	4 for filter banks; constantq see the command's description	voices/octave
--voices@run constantq	N	Number of bands per octave, counted from the top down.	6	-
--voices@run opticalflow	N	Itersine family only: filter-bank members per octave.	4	voices/octave
--voices@run speedspectrum	V	Itersine time bank: members per octave	4	-
--weight	none|barycentric|voronoi	How the vertices of each member are weighted (measure compute: none is the plain vertex mean, barycentric or voronoi weight by area); phasegradient: power|coherence, the edge weight of a weighted fit.	measure compute: none, must be given; phasegradient: power	-
--weight@run measure	none|barycentric|voronoi	How the vertices of each member are weighted: none is the plain vertex mean, barycentric or voronoi weight by area; not used with --over.	required unless --over is given	-
--weight@run phasegradient	power|coherence	Edge weight of a weighted fit (with --spatial-band): the edge's cross-power or its coherence.	power	-
--weights	area|none	cellquadrature: weight vertices by their area or equally; crossspectrum read: weight the delay and phase-slope fits by coherence or not	cellquadrature: area; crossspectrum read: coherence	-
--weights@run cellquadrature	area|none	Weight each vertex of a tile by the area it stands for, or count all vertices equally.	area	-
--weights@read crossspectrum	coherence|none	For --reading delay: weight each frequency bin of the phase-slope fit by its squared coherency, or count every bin equally.	coherence	-
--whitener-form	symmetric|pca	Form of the noise whitener: symmetric or PCA.	symmetric	-
--whitener-form@run inverse	symmetric|pca	Form of the whitener: symmetric [M x M] (Brainstorm) or PCA [rank x M] (MNE-Python); the kernel is the same.	symmetric	-
--window	hann|hamming|blackman|rectangular	Taper applied to each Welch segment.	hamming (default and brainstorm sets); hann (scipy set)	-
--window-seconds	SECONDS	Window length for the analytic estimator or the cycle patterns, snapped to the nearest time-tile level (alternative to --level)	crossspectrum: the finest of the bands' windows; cyclepattern: 4	s
--window-seconds@run crossspectrum	SECONDS	Analytic estimator only: window length, snapped to the nearest time-tile level; an alternative to --level.	the finest of the bands' constant-Q windows	s
--window-seconds@run cyclepattern	SECONDS	Window length, snapped to the nearest time-tile level; ignored when --level is given.	4	s
--window@run crossspectrum	hann|hamming|blackman|rectangular	Taper applied to each Welch segment.	hamming	-
--window@run hodgepower	hann|hamming|blackman|rectangular	The taper applied to each Welch segment.	hamming	-
--window@run psd	hann|hamming|blackman|rectangular	Taper applied to each segment	hamming	-
--window@run quadrature	hann|hamming|blackman|rectangular	Taper applied to each Welch segment.	hamming (set tiles, brainstorm); hann (scipy, nxr-chain)	-
--window@run sourcepsd	hann|hamming|blackman|rectangular	Taper applied to each Welch segment.	hamming (hann for the scipy set)	-
--window@run sourcetensor	hann|hamming|blackman|rectangular	Taper applied to each Welch segment.	hamming (hann under the scipy set)	-
--window@run speedspectrum	hann|hamming|blackman|rectangular	Taper applied to each segment	hamming	-
--windows	NAME	The stored analytic cross-spectrum windows (named <stem>_<band>_windows) the bursts are read from.	none (required)	-
--windows@run wishart	NAME	Stored analytic cross-spectrum windows (<stem>_<band>_windows) whose burst tiles are summed and tested.	required	-
--acquisition@import recording	NAME	The session of the subject the recording is stored under.	rest	-
--anonymize@import recording		Drop the identifying header fields (the file's start time, subject, experimenter, patient and recording identification) instead of keeping them as the file has them.	off (the header is kept as the file has it)	-
--format@import recording	FORMAT	The file's format: fif, edf, bdf, brainvision, eeglab, ctf or kit.	from the file's extension	-
--licence@import recording	TEXT	The licence recorded in the origin of every item the import makes.	unknown	-
--attribution@import recording	TEXT	Who is to be credited, recorded in the origin of every item the import makes.	unknown	-
--anonymize		Drop the identifying fields of a file's header (import recording).	off	-
--format	FORMAT	A file's format (import recording).	from the file's extension	-
--source-root	DIR	The folder a file's recorded path is relative to (import recording).	the file's own folder	-
--acquisition	NAME	The session of the subject the result is stored under (import recording).	rest	-
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
opts: --atlas-depth --dataset --datastore --default --lc-levels --modes --name --source-path --source-protocol --source-tool --store-format --templates
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
opts: --accept-unlisted-sources --cortex-mask --hcp-end-surface --hcp-points --hcp-trk --home-dataset --max-end-distance --name --package* --parcellation+ --recipe* --sources* --surface+
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
R"TBL(@ list spatial-bands
old: pyramid spatial-bands
alias: refused
summary: List a surface's spatial bands (one per level of its tile ladder).
opts: --dataset* --subject* --surface --templates+
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
R"TBL(@ import recording
old: __import recording
alias: no
summary: Copy a recording file (FIF, EDF, BDF, BrainVision, EEGLAB, CTF, KIT) into a subject package, with its channels, events, bad channels and an origin record.
opts: --acquisition --anonymize --attribution --format --from* --licence --name --package* --source-root
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
summary: Write a source map or a source series on the two hemispheres of a cortex as MNE source estimates (-lh.stc and -rh.stc), with a provenance sidecar.
opts: --all --field* --from --out* --package* --to
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
summary: Query a stored measurement pyramid at any spatial and temporal scale.
opts: --band --dataset* --field --limit --max-value --measure* --min-value --order --selection* --space-level --spatial-band --subject* --time-level --templates+
)TBL",
R"TBL(@ export measurements
old: measure export
alias: refused
summary: Export stored measurements as a table for statistics (a row per subject), with its dictionary and provenance.
opts: --dataset* --exclude --fields* --keep-all --labelling* --layout --out* --recipe --scaling --statistics --subjects --table*
)TBL",
R"TBL(@ reduce pyramid
old: pyramid reduce
alias: refused
summary: Reduce several subjects' group-tree pyramids into the group's pyramid; on a catalogue dataset the result is a group package (--into).
opts: --dataset* --depth --into --kernels --subjects* --templates+
)TBL",
R"TBL(@ run noisecov
old: noisecov compute
alias: refused
group: Source imaging
summary: The noise covariance of a recording's good channels.
span: yes
opts: --block-seconds --chunks --dataset* --divisor --name --no-remove-mean --recording* --replace --start --stop --subject* --types --templates+
)TBL",
R"TBL(@ run inverse
old: inverse compute
alias: refused
group: Source imaging
summary: The depth-weighted minimum-norm inverse kernel, from a noise covariance and a head model.
span: yes
opts: --block-seconds --chunks --dataset* --depth-channels --depth-exponent --depth-limit --depth-limit-on --depth-norm --divisor --forward --joint-types --mne --name --no-average-reference --no-depth --no-detect-diagonal --no-remove-mean --noise-cov --rank-tolerance --recording --regularisation --replace --ridge-fraction --ridge-over-rank --snr --snr-normalisation --start --stop --subject* --types --whitener-form --templates+
)TBL",
R"TBL(@ run headmodel
old: headmodel compute
alias: refused
group: Source imaging
summary: The sphere head model (gain matrix) for a subject's cortex, MEG and EEG.
opts: --berg-dipoles --berg-starts --berg-terms --channels --dataset* --eeg-conductivities --eeg-radii --eeg-series --eeg-series-terms --eeg-surface --meg --meg-surface --mne-eeg-layers --name --no-eeg --no-project-electrodes --replace --sensor-weight-exponent --source-surface --sphere-cost --subject* --templates+
)TBL",
R"TBL(@ run pointspread
old: pointspread compute
alias: refused
group: Source imaging
summary: The instrument's blur: unit currents at sample vertices through the head model and inverse kernel.
opts: --dataset* --forward --fraction --kernel --levels --list --modes --name --order --recording* --reference-level --replace --spatial-shape --subject* --tolerance --transform --vertices --templates+
)TBL",
R"TBL(@ run eigenmodes
old: eigen solve
alias: refused
group: Geometry
summary: The Hodge-Dirac eigenmodes of a stored surface (cached).
opts: --dataset* --flip-eps --modes --mollify --normalize --replace --shift --sigma --slice --star1-floor --subject* --surface --templates+
)TBL",
R"TBL(@ run frames
old: frames compute
alias: refused
group: Geometry
summary: The canonical (north, west) and anatomical frames of a surface.
opts: --convention --dataset* --max-condition --min-projection --replace --side --subject* --surface --template --template-surface --templates+
)TBL",
R"TBL(@ run tiles
old: tiles compute
alias: refused
group: Geometry
summary: The dyadic equal-area tile ladder of a surface.
opts: --dataset* --depth --replace --subject* --surface --templates+
)TBL",
R"TBL(@ run constantq
old: constantq compute
alias: refused
group: Time
summary: The constant-Q tiling of a time line (a ladder of dyadic windows).
opts: --all --cycles --dataset* --fmax --fmin --line --method --recording --subject* --voices --templates+
)TBL",
R"TBL(@ run frametiling
old: frametiling compute
alias: refused
group: Time
summary: The frame tiling on the time tower: the cycles of one tower level over a recording.
opts: --band --centre --dataset* --frame-level --frames-per-cycle --recording* --replace --start-s --subject* --templates+
)TBL",
R"TBL(@ run joint
old: joint compute
alias: refused
group: Time
summary: The joint partition of space and time: cells of a spatial tile level by a time tile level.
opts: --dataset* --recording* --replace --space-level* --subject* --surface --time-level* --templates+
)TBL",
R"TBL(@ run lowpass
old: lowpass
alias: refused
group: Time
summary: A zero-phase low-pass filter, written as a new recording.
opts: --bandwidth --dataset* --fpass --frequencies --fstop --harmonics --line --name --no-remove-mean --order --padding --recording* --ripple --subject* --tolerance --transient-samples --transition --templates+
)TBL",
R"TBL(@ run highpass
old: highpass
alias: refused
group: Time
summary: A zero-phase high-pass filter, written as a new recording.
opts: --bandwidth --dataset* --fpass --frequencies --fstop --harmonics --line --name --no-remove-mean --order --padding --recording* --ripple --subject* --tolerance --transient-samples --transition --templates+
)TBL",
R"TBL(@ run bandpass
old: bandpass
alias: refused
group: Time
summary: A zero-phase band-pass filter, written as a new recording.
opts: --bandwidth --dataset* --fpass --frequencies --fstop --harmonics --line --name --no-remove-mean --order --padding --recording* --ripple --subject* --tolerance --transient-samples --transition --templates+
)TBL",
R"TBL(@ run bandstop
old: bandstop
alias: refused
group: Time
summary: A zero-phase band-stop filter, written as a new recording.
opts: --bandwidth --dataset* --fpass --frequencies --fstop --harmonics --line --name --no-remove-mean --order --padding --recording* --ripple --subject* --tolerance --transient-samples --transition --templates+
)TBL",
R"TBL(@ run notch
old: notch
alias: refused
group: Time
summary: A zero-phase notch filter (line noise and harmonics), written as a new recording.
opts: --bandwidth --dataset* --fpass --frequencies --fstop --harmonics --line --name --no-remove-mean --order --padding --recording* --ripple --subject* --tolerance --transient-samples --transition --templates+
)TBL",
R"TBL(@ run psd
old: psd compute
alias: refused
group: Spectra
summary: The Welch power spectral density of a recording, bad segments left out.
span: yes
opts: --chunks --dataset* --detrend --name --no-bad-segments --overlap --periodic --recording* --replace --scaling --segment-level --segment-samples --segment-seconds --set --start --stop --subject* --symmetric --through --window --templates+
)TBL",
R"TBL(@ run sourcepsd
old: sourcepsd compute
alias: refused
group: Spectra
summary: The band power maps of the sources (the Welch PSD of the sources, grouped in bands).
span: yes
opts: --bands --chunks --dataset* --detrend --kernel --name --no-bad-segments --normalisation --overlap --periodic --recording* --replace --scaling --segment-level --segment-samples --segment-seconds --set --start --stop --subject* --symmetric --window --templates+
)TBL",
R"TBL(@ run sourcetensor
old: sourcetensor compute
alias: refused
group: Spectra
summary: The band cross-spectral tensor of the current, its Stokes quaternion, normal power and share.
span: yes
opts: --bands --chunks --dataset* --detrend --kernel --name --no-bad-segments --normalisation --overlap --periodic --recording* --replace --scaling --segment-level --segment-samples --segment-seconds --set --start --stop --subject* --symmetric --window --templates+
)TBL",
R"TBL(@ run crossspectrum
old: crossspectrum compute
alias: refused
group: Spectra
summary: The sensor cross-spectral density per band (Welch, or per time window with the analytic estimator).
span: yes
opts: --bands --channels-of --chunks --dataset* --detrend --estimator --level --name --no-bad-segments --overlap --periodic --recording* --replace --resolve --scaling --segment-level --segment-samples --segment-seconds --set --start --stop --subject* --symmetric --types --window --window-seconds --templates+
)TBL",
R"TBL(@ run quadrature
old: quadrature compute
alias: refused
group: Spectra
summary: The quadrature spectrum of the sources per band: fraction, energy map and leading pairs.
span: yes
opts: --bands --chunks --dataset* --detrend --kernel --name --no-bad-segments --no-levi-civita --overlap --pairs --periodic --recording* --replace --scaling --segment-level --segment-samples --segment-seconds --set --start --stop --subject* --symmetric --tile-levels --tile-max-level --tiles --window --templates+
)TBL",
R"TBL(@ run hodgepower
old: hodgepower compute
alias: refused
group: Spectra
summary: The band power of the Hodge-Dirac eigenmode coefficients of a free-orientation kernel.
span: yes
opts: --bands --chunks --dataset* --detrend --kernel --modes --name --no-bad-segments --overlap --package --periodic --recording* --replace --scaling --segment-level --segment-samples --segment-seconds --set --start --stop --subject* --symmetric --window --templates+
)TBL",
R"TBL(@ run speedspectrum
old: speedspectrum compute
alias: refused
group: Spectra
summary: The speed spectrum: mode energy per phase speed from the joint eigenvalue and frequency bank.
span: yes
opts: --anchor --centres --chunks --cycles --dataset* --detrend --family --fmax --fmin --gain --gamma --graph-family --graph-filters --graph-voices --half-width --kernel --limits --modes --name --no-bad-segments --overlap --package --periodic --recording* --replace --scaling --segment-level --segment-samples --segment-seconds --set --sides --speed-bins --speed-max --speed-min --start --stop --subject* --symmetric --time-bandwidth --voices --window --templates+
)TBL",
R"TBL(@ run phasegradient
old: phasegradient compute
alias: refused
group: Cortical flow and patterns
summary: The phase gradient of a band from a stored cross-spectrum: phase current, power, speed and travelling index.
opts: --cross-spectrum* --dataset* --defects --depth --edge-lag --field --fit --fold-edges --kernel --name --package --phase-of --replace --signal --spatial-band --subject* --weight --templates+
)TBL",
R"TBL(@ run cyclepattern
old: cyclepattern compute
alias: refused
group: Cortical flow and patterns
summary: The most consistent cycle patterns of the current from a cross-spectrum per window.
opts: --cross-spectrum* --dataset* --kernel --level --name --package --patterns --replace --spatial-band --subject* --timing-of --window-seconds --templates+
)TBL",
R"TBL(@ run opticalflow
old: opticalflow compute
alias: refused
group: Cortical flow and patterns
summary: Optical flow (Horn-Schunck) of the source estimate: where and how fast the activity moves.
span: yes
opts: --alpha --anchor --centres --chunks --cycles --datafit --dataset* --depth --family --flow-tolerance --frame-level --freq --gain --gamma --half-width --input --kernel --limits --max-iterations --method --name --no-levi-civita --oversample --package --preconditioner --recording* --regulariser --replace --resolution --scale-family --scale-filters --scale-voices --start --start-s --stop --subject* --tile-levels --tile-max-level --tiles --time-bandwidth --tolerance --voices --templates+
)TBL",
R"TBL(@ run cellquadrature
old: cellquadrature compute
alias: refused
group: Cortical flow and patterns
summary: Quadrature (travelling vs standing) per joint cell and per burst, from the stored cross-spectrum.
opts: --bursts --cross-spectrum* --dataset* --joint --kernel --modes --order --replace --space-level* --subject* --tile-pairs --tolerance --transform --weights --templates+
)TBL",
R"TBL(@ run patches
old: patches compute
alias: refused
group: Cortical flow and patterns
summary: Patches on a static cortical map: where it stands out from its surroundings, at the size of its tiles.
opts: --dataset* --enter-sd --exit-sd --level* --map* --max-area --min-area --modes --order --replace --sign --subject* --summaries --tolerance --transform --templates+
)TBL",
R"TBL(@ run scaleselection
old: scaleselection compute
alias: refused
group: Cortical flow and patterns
summary: Each event's own size and rhythm: the best spatial and temporal level by standardised response.
opts: --dataset* --min-link --name --recording* --replace --response --source --space-levels* --subject* --surface --time-levels* --templates+
)TBL",
R"TBL(@ run pyramid
old: pyramid build
alias: refused
group: Measurements
summary: The measurement pyramid of a recording's cortical current, from space and time tiles.
span: yes
opts: --bands --block-samples --chunks --dataset* --depth --frame-level --halo-seconds --keep-bad-segments --kernel --recording* --replace --shape --start --start-s --stop --subject* --threads --tree --templates+
)TBL",
R"TBL(@ run measure
old: measure compute
alias: refused
group: Measurements
summary: Stored fields measured over a partition's members, with their recipe.
span: yes
opts: --chunks --dataset* --fields* --gradient --labelling --level --measures --over --recipe --replace --start --stop --subject* --weight --templates+
)TBL",
R"TBL(@ run statistic
old: statistic compute
alias: refused
group: Measurements
summary: A stored measurement summarised over its partition's members.
opts: --combine --dataset* --ddof --measure* --members-own --name --of* --over* --replace --subject* --summaries* --templates+
)TBL",
R"TBL(@ run wishart
old: wishart compute
alias: refused
group: Nulls
summary: The Wishart null of the second-moment measures of bursts.
opts: --band* --bursts* --dataset* --draws --psd* --recording* --replace --seed --shape --subject* --windows* --templates+
)TBL",
R"TBL(@ run spatialnull
old: spatialnull compute
alias: refused
group: Nulls
summary: The lag-free null of the spatial events: draws with early stopping, per-vertex p and q.
span: yes
opts: --batch --chunks --dataset* --events --exceedances=--h --max-draws --min-draws --recording* --replace --set+ --start --stop --subject* --templates+
)TBL",
R"TBL(@ run null
old: null compute
alias: refused
group: Nulls
summary: The lag-free null's tests of the vortex cores' crossings, per tile.
span: yes
opts: --batch --chunks --dataset* --exceedances=--h --max-draws --min-crossings --min-draws --over --recording-a* --recording-b --replace --set+ --stage --start --stop --subject* --templates+
)TBL",
R"TBL(@ run repeat
old: repeat compute
alias: refused
group: Nulls
summary: The vortex cores' run-to-run repeat: two recordings' stored tile crossings compared.
opts: --dataset* --min-crossings --name --over --recording-a* --recording-b* --replace --stage --subject* --templates+
)TBL",
  };
}
