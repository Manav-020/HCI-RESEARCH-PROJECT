# Subject-Specific Calibration and EEG Emotion Explanations (DEAP)

HCI research project. We test whether calibrating an EEG emotion model to a new person makes its explanations **more faithful** and **more stable**, while **keeping predictive performance**.

> **Research question:** Does subject-specific calibration make EEG emotion explanations more faithful and stable to the model's actual decision process while maintaining predictive performance?

## Summary of findings

All results are from 32-fold leave-one-subject-out (LOSO) evaluation on DEAP, with 20 calibration and 20 test trials per target subject. This is an exploratory analysis, and the main result is mostly negative.

| Finding | Evidence |
|---|---|
| **Prediction is at chance for every arm.** | Balanced accuracy (BA) 0.47-0.52. L0 general 0.495; final L4 model 0.483. No paired BA difference is significant. |
| **An earlier notebook result (personalized BA ~0.56) did not replicate.** | The pipeline gives 0.489 for the same configuration. See `replication_diff.md` in a run folder. |
| **Faithfulness gains come from sparsity, not calibration.** | L4 vs L0 compact faithfulness +0.10 (Holm p = 0.006), but L4 equals the L3 general model. Random-k feature selection gives 0.739, MI selection 0.770, and correlation-only 0.532. |
| **Shrinkage personalization never adapted.** | Brier-based λ selection chose λ = ∞ in 32/32 folds (BA-based: 19/32). The predictive-preservation check (margin −0.03) was not met for L4 vs L0 (lower bound −0.048). |
| **Calibration affects stability, depending on normalization.** | Without normalization, target-only explanations are less stable (refit cosine −0.23 at L0, −0.20 at L1, Holm p < 0.01). With per-subject normalization (L2), target-only is more stable (top-5 overlap +0.078, Holm p = 0.001) and more faithful (+0.10, Holm p = 0.012). |
| **Per-subject normalization lowers general-model faithfulness.** | −0.138 (Holm p = 0.002). |
| **Window length has no meaningful effect.** | BA 0.497-0.514 across 2 s / 4 s / 8 s / full (not tested for significance). |
| **Exploratory sequential model did not help.** | GRU mean BA 0.513 (descriptive only, outside the main claim). |

Because predictions are near chance, none of these explanations can be shown to reflect real emotion-related signal. Treat the faithfulness numbers as properties of the models, not evidence of physiological validity.

## Method ladder

Each rung adds one change, so its effect can be isolated.

| Arm | Description |
|---|---|
| **L0** | Full-stimulus (seconds 5-60) log band power: 32 channels × 4 bands (theta, alpha, beta, gamma) = 128 features. Correlation filter (|r| > 0.90), scaling, grouped-CV logistic regression. |
| **L1** | Replace the full-stimulus feature with robust windowed features (2/4/8 s windows, artifact rejection, median across clean windows). The 3 s baseline is not used. |
| **L2** | L1 features z-scored per subject, using only that subject's first 20 presentation-order trials (no labels). This counts as target adaptation. |
| **L3** | Mutual-information feature selection (top-k, k chosen by grouped inner CV, recomputed inside each training fold). Controls: random-k and correlation-only. |
| **L4** | Shrinkage personalization: fit the target model with a penalty λ‖w − w_source‖² toward the source model. λ is chosen on pseudo-target source subjects, never on the real target's test trials. |

Comparators at each rung: general model, 10× weighted source+calibration mix (`mix10`), shrinkage, intercept-only recalibration, target-only, and majority class.

**Exploratory (not part of the main claim):** window statistics (mean/std/slope) with the same pipeline, and a small GRU over window sequences with head-only fine-tuning explained by integrated gradients.

## Explanation metrics

- **Explanation:** exact linear SHAP in standardized feature space, summed per channel (additivity is asserted).
- **Compact faithfulness:** top-5 channels by |attribution|; random channel masking (p = 0.25, 100 repeats); score = 1 − MSE(actual change, compact prediction) / mean(actual change²). A shuffled-attribution control is included.
- **Deletion curves** (top / least / random channel order) and deletion-curve AUC.
- **Stability:** 20 paired stratified bootstrap refits; cosine similarity, top-5 overlap and Spearman rank correlation of channel attributions against the original model.
- **Caveat:** compact faithfulness partly reflects how concentrated the model weights are. It is not physiological causality.

## Statistics

Subject-level paired analysis (values are averaged within subject first): mean difference, paired bootstrap 95% CI (20,000 resamples), sign-flip test, Wilcoxon signed-rank test, improved/tied/degraded counts, Holm correction across primary endpoints (compact faithfulness, refit cosine stability, balanced accuracy). The BA predictive-preservation margin (−0.03) is illustrative and pre-chosen, not tuned.

## Leakage controls

- LOSO: the target subject is never in any fit set (asserted in tests).
- Target trials 1-20 (presentation order) calibrate; trials 21-40 are test-only.
- Feature filter, scaler, MI selection, C and λ are all fit on source subjects only, with grouped (by subject) inner CV.
- MI is a pipeline transformer, so it is refit within each inner training fold.
- Per-subject normalization uses only the subject's own first 20 trials' features.
- Known small optimism: preprocessing is fit once per outer fold and reused for pseudo-target λ selection.

## Repository structure

```
.
├── run_pipeline.py        # CLI entry point
├── src/                   # data loading, features, selection, personalization,
│                          # explanation, stats, report generation, exploratory arm
├── tests/                 # pytest suite (leakage, SHAP additivity, optimizer, windowing, ...)
├── requirements.txt
├── results/latest/        # final report copies (report.md, report.html, paired_summary.csv)
└── dataset/               # NOT included, see below
```

Full timestamped run folders (`results/<timestamp>/`, with all CSVs, figures and logs) are not committed.

## Dataset (not included)

This repository does not contain DEAP data. DEAP requires a signed EULA from its authors: <https://www.eecs.qmul.ac.uk/mmv/datasets/deap/>

Download the preprocessed Python version and place it as:

```
dataset/
├── data_preprocessed_python/   # s01.dat ... s32.dat
└── Metadata/participant_ratings.xls
```

`participant_ratings.xls` provides reference valence labels and presentation order (trials in the `.dat` files are in video order). The pipeline checks stored dominance and liking against the workbook for every trial before using it.

## Install and run (PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Optional, for the exploratory GRU (CPU build):
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m pytest
python run_pipeline.py --quick      # smoke test: 6 subjects, few resamples
python run_pipeline.py              # full run
```

Linux/macOS: use `python3 -m venv .venv` and `source .venv/bin/activate`.

| Flag | Purpose |
|---|---|
| `--quick` | Small smoke-test run (6 subjects, 5 bootstrap repeats, 1,000 stats resamples) |
| `--subjects N` | Limit the number of LOSO subjects |
| `--resume` | Reuse completed per-subject checkpoints (use the same `--out`) |
| `--skip-exploratory` | Skip window-statistics and GRU arms |
| `--gru-subjects N` | Cap GRU target folds |
| `--out PATH` | Name the results folder |

Extracted features are cached under `cache/` (keyed by subject file hash), so extraction runs once.

## Reading the outputs

Each run writes to `results/<timestamp>/`:

- `report.md` / `report.html`: full report (the HTML is self-contained and opens offline)
- `paired_summary.csv`: all paired comparisons with CIs, p-values and Holm correction
- `subject_metrics.csv`, `lambda_sweep.csv`, `window_sensitivity.csv`, `artifact_rejection.csv`, `stability_refits.csv`, `deletion_curves.csv`, `feature_selection_controls.csv`, `failures.csv`, `config.json`, `run.log`
- `replication_diff.md`: audit comparing this pipeline with the earlier notebook

## Limitations

- Near-chance prediction limits what the explanations can mean.
- Only 20 test trials per subject, so per-subject estimates are noisy (this is the likely reason the earlier notebook result did not replicate).
- LOSO models share source subjects, so folds are not independent.
- Compact faithfulness depends on the masking protocol and on weight concentration.
- MI is computed on pooled source subjects, which mixes between- and within-subject variation.
- The unofficial/unverified provenance of the supplied EEG files relative to the official DEAP release was not checked.
- The GRU may overfit with about 1,280 trials, and exploratory results are not part of the main claim.

## Citation

DEAP dataset: Koelstra, S. et al. "DEAP: A Database for Emotion Analysis Using Physiological Signals." *IEEE Transactions on Affective Computing* 3(1), 2012.

## Authors

Manav Patel,Nishcha Mittal, Kirish Kachadiya, Shreya Ashar, Zeel Ramoliya

SVNIT, Human computer interaction, Dr Sudhakar  Mishra

## License

<Add a license for the code, e.g. MIT. DEAP data is covered by its own EULA and is not redistributed here.>
