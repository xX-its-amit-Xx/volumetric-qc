# Metric reference

This document explains, one metric at a time, **what the metric computes**, **what artifact it catches**, **how the threshold is interpreted**, and **the relevant references**. Defaults are tuned for the `generic` preset; modality presets override individual values — see `volumetric_qc/pipeline/config.py` for the full list.

---

## `intensity` — per-slice statistics and drift

**What it computes.** For each sampled z-slice, the mean, standard deviation, and 1st/50th/99th percentiles of intensity. From the sequence of per-slice means it fits a linear regression and reports the slope normalized by the overall mean (`drift_slope`) and the coefficient of variation of the per-slice mean (`cv`).

**What it catches.** Photobleaching across the volume (negative `drift_slope`), depth-dependent attenuation from refractive-index mismatch (gradual decay), or stage-illumination drift over a multi-day acquisition (high `cv`).

**Threshold.** `intensity_drift_max` (default 0.30) — the absolute normalized slope must stay below this fraction. A slope of 0.30 means the mean intensity changes by 30% from one end of the volume to the other.

**References.** Bleaching dynamics in light-sheet microscopy: Reynaud et al., HFSP J 2008. SHIELD imaging at depth: Park et al., 2018.

---

## `sharpness` — per-slice focus quality

**What it computes.** Variance of the Laplacian (`L = ∇² I`) per z-slice. The output is normalized by the maximum across the volume so a value of 1.0 marks the sharpest slice. Slices below `outlier_fraction` of the peak are flagged as out-of-focus.

**What it catches.** Focus drift during a long acquisition, mechanical settling on a fresh sample, and (less commonly) genuine deep-tissue scattering.

**Threshold.** `sharpness_min_relative` (default 0.05) — the worst slice must retain at least 5% of the peak sharpness. Tighten for in-focus stack-only analyses; loosen for whole-organ scans where the surface slices are expected to be soft.

**References.** Pertuz et al., *Pattern Recognition* 2013 (focus measure operators). Forster et al., *Microscopy Research and Technique* 2004 (variance-of-Laplacian for fluorescence stacks).

---

## `background` — uniformity and autofluorescence

**What it computes.** Two things at once:
1. **Background uniformity** — per-slice background level (low-percentile patch mean), with the coefficient of variation across all low-percentile patches reported as `background_cv`.
2. **Autofluorescence ratio** — overall background mean divided by overall signal mean (`autofluor_ratio`).

**What it catches.** Vignetting and uneven illumination push up `background_cv`. Residual lipid pockets or incomplete clearing push up `autofluor_ratio` because background pixels become bright relative to true stain.

**Thresholds.** `background_cv_max` (default 0.40), `autofluorescence_ratio_max` (default 0.30).

**References.** Susaki et al., *Nature Protocols* 2015 (autofluorescence in cleared tissue). CIDRE field-illumination correction (Smith et al., *Nat Methods* 2015) for context on what good background uniformity looks like.

---

## `channel_bleed` — cross-channel spectral leakage

**What it computes.** For each ordered pair of channels (A, B), restrict to pixels where channel A is above its 90th percentile (signal pixels), and compute the Pearson correlation of A and B intensity in that mask. High positive correlation = bleed-through from A into B.

**What it catches.** Emission-spectrum overlap that wasn't fully separated by the dichroics; autofluorescent lipids contaminating a long-wavelength channel.

**Threshold.** `channel_bleed_corr_max` (default 0.40) — applied per ordered pair.

**References.** Zimmermann, *Adv Biochem Eng Biotechnol* 2005 (spectral unmixing in fluorescence). Renier et al., 2016 (iDISCO+ channel design).

---

## `registration` — cross-channel pixel shift

**What it computes.** Sub-pixel phase correlation (`skimage.registration.phase_cross_correlation`) between channels on a stratified sample of z-slices. Reports the per-axis (dy, dx) median shift in voxels for each ordered channel pair, plus the per-slice distribution for spread analysis.

**What it catches.** Chromatic aberration across wavelengths, asymmetric optical paths, lateral stage drift between sequential wavelength acquisitions.

**Threshold.** `registration_shift_max_voxels` (default 2.0) — the maximum absolute shift on any axis. Convert to micrometers using the `voxel_size_um` config field for physical reporting.

**References.** Guizar-Sicairos et al., *Optics Letters* 2008 (efficient sub-pixel image registration). Preibisch et al., *Bioinformatics* 2009 (Fiji stitching for context).

---

## `artifacts.stripes` — light-sheet stripe shadows

**What it computes.** Per sampled z-slice, the 2D power spectrum is masked to a narrow wedge along the kx=0 and ky=0 axes (excluding DC). The fraction of total energy in that mask is `energy_ratio`. Stripes show up as a bright line in FFT magnitude orthogonal to the stripe direction.

**What it catches.** Horizontal/vertical banding caused by absorbing features in the illumination path (debris, the sample chamber edge, beam aperture).

**Threshold.** `stripe_energy_ratio_max` (default 0.20).

**References.** Mayer et al., *Optics Express* 2018 (stripe correction in light-sheet). VSNR / wavelet-FFT despriping in cleared tissue (Liang et al., 2016).

---

## `artifacts.bubbles` — air pockets and debris

**What it computes.** Laplacian-of-Gaussian blob detection (`skimage.feature.blob_log`) on each sampled z-slice, at multiple scales between `min_sigma` and `max_sigma`, with a contrast threshold. Reports the count per slice plus the (z, y, x, sigma) of every detected blob.

**What it catches.** Air bubbles introduced during sample mounting, dust on the lens, particulate contamination in the immersion medium.

**Threshold.** `bubbles_per_slice_max` (default 5).

**References.** Lindeberg, *Int J Computer Vision* 1998 (scale-space blob detection).

---

## `artifacts.folding` — tissue folds and tears

**What it computes.** Sobel gradient magnitude per pixel; report the fraction of pixels whose magnitude exceeds `median + z_score · 1.4826 · MAD`.

**What it catches.** Sample handling damage that produces sharp gradient discontinuities — folds where two layers of tissue stack on top of each other, or tears that leave straight bright edges.

**Threshold.** `folding_outlier_fraction_max` (default 0.02).

**References.** Wickramasinghe et al., *Sci Rep* 2019 (tissue clearing artifact taxonomy).

---

## `clearing` — residual lipid and RI-mismatch speckle

**What it computes.** Fraction of FFT energy above a normalized spatial-frequency cutoff (default 40% of Nyquist) on stratified z-slices.

**What it catches.** Incomplete clearing leaves bright high-frequency speckle that does not correspond to genuine staining structure. Refractive-index mismatch between sample and immersion medium produces similar high-frequency texture, especially in superficial slices.

**Threshold.** `clearing_residue_max` (default 0.15).

**References.** Susaki & Ueda, *Chem Biol* 2016 (chemical principles of tissue clearing). Park et al., 2018 (SHIELD; clearing-quality metrics).

---

## Tuning thresholds for your own data

The defaults are deliberately conservative — they ship with the assumption that a published-grade sample should easily pass. To calibrate against your own pipeline:

1. Run `volumetric-qc` on a small curated set of known-good and known-bad samples.
2. Load the JSON summaries (`qc_summary.json`) and read off the metric values for each.
3. Set thresholds to the worst pass-grade value plus a small margin.
4. Re-run on the bad samples; verify they fail.
5. Record the chosen thresholds in `qc_config.yaml` and commit it alongside your analysis code.

See [cookbook.md](cookbook.md) §3 for a fully worked example.
