# Spindles and slow waves: the design (PARKED, 2026-11-17)

**Status:** written down and parked by the maintainers. Not to be implemented until they lift the hold. The audit it builds on is
`docs/audits/2026-11-17-sleep-events-and-montages.md`.

## Why

Sleep spindles and slow waves are transient patterns whose onset, offset and spread differ across channels. The question the lab asks of them is
the same as for every pattern: **when** does it happen, **where** does it start, and **how** does it spread. A detector that reports only one
onset per channel from a threshold crossing makes the timing a function of the threshold and of the amplitude (the reference's own notes say the
reported duration grows with amplitude near the gate). This design measures the delays from the shape of the envelopes instead.

## Decided

- **Reference.** The spindle detector's default reproduces Rheome's `detect.spindle` (its thresholds and gates): an event must reach
  median + 3 robust SD of the channel's envelope, is extended to where it falls below median + 2.5 robust SD, events closer than 0.1 s merge, and
  events are kept between 0.5 and 3 s. The baseline is the whole recording (or a stated baseline mask), never per segment. Slow waves follow
  Massimini et al. 2004 (negative peak, peak-to-peak and the multi-channel criterion). The detectors are ported directly; no reference goldens
  are required.
- **Windows first.** A **global window** is defined wherever at least one channel shows an event (the union of overlapping per-channel events).
  Each channel's own event is kept inside its window, not merged away. Later analyses (cross-spectra, cortical patterns) use the global windows.
- **Bands from the chunk FFT, not filter + Hilbert.** Each chunk is transformed once; the complex band coefficients of the dyadic constant-Q
  tiles give the amplitude (`|c|`) and the phase (`arg c`). No recursive filter, no Hilbert transform, no resampling; pointwise per tile and so
  GPU-friendly. Rheome's own route (an IIR band-pass and the Hilbert envelope at 100 Hz) may be kept as a named option for comparison.
- **Delays from envelope cross-spectra.** The cross-spectrum of the channels' amplitude envelopes, accumulated over the global windows (a sum
  over the windows partition: a measurement, as the cardinal rule requires), gives a **lag map**: each channel's delay from the slope of the
  cross-spectral phase across envelope frequencies, and its consistency; the imaginary part separates true delays from zero-lag mixing.
  Per-event lags are kept with their uncertainty (a single 0.5–3 s event holds few envelope cycles). Threshold onsets may be kept as a secondary
  measure.

## The pipeline (when built)

1. Chunk FFT → complex band coefficients (the spindle band; the slow-wave band).
2. Per-channel detection on the band amplitude (the reference's thresholds, merging and gates).
3. Global windows: a spans Selection on the time line, one member per window; the channel events as spans linked to their window.
4. Envelope cross-spectra accumulated over the windows → the lag map; per-event lags with uncertainty.
5. Per-event measures over the windows: peak frequency (from the coefficients), amplitude, duration, participating channels.

It is built as registered nodes and YAML definitions on the pipeline type system, in the burst-detection family.
