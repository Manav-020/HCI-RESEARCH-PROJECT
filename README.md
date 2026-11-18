# EEG emotion calibration experiment

Reproducible DEAP LOSO pipeline for the question: does subject-specific calibration make EEG emotion explanations more faithful and stable while preserving predictive performance? The workbook provides reference valence and presentation order; source files remain read-only.

## Install and run (PowerShell)

```powershell
cd 'D:\coding\python\HCI RESEARCH PROJECT'
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Optional exploratory GRU, CPU build:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest
python run_pipeline.py --quick
python run_pipeline.py
```

Use `--subjects N` to cap LOSO subjects, `--gru-subjects N` to cap GRU target folds, `--skip-exploratory` to skip step 4a and GRU, and `--out PATH` to name the results folder. Resume a partially completed run with the same `--out PATH` and `--resume`. The feature cache under `cache/` stores extracted 2, 4, 8, full-stimulus and statistics features by subject and source-file SHA256; missing subject entries are extracted once and added.

## Data and leakage controls

The workbook is matched case-insensitively and aligned by video id; stored dominance/liking labels are checked. The signal, label, shape, finite-value, flat-channel, presentation-trial, and split checks run before model fitting. The first 20 trials in presentation order calibrate; the final 20 remain test-only. All source-only choices use grouped source CV. Target normalization uses only its unlabeled first 20 feature rows. MI is a pipeline transformer, so it refits within each inner training fold. Assertions prevent target IDs entering source fit sets.

## Method ladder

- **L0:** full stimulus 128 log band powers, source correlation filter/scaler and grouped logistic selection.
- **L1:** robustly aggregated stimulus-window features with artifact rejection; no baseline is used. After auditing the stored signal amplitude, rejection uses per-trial RMS robust z ≤ 8 and peak-to-peak ≤ 2000 input units. The scale audit, threshold sweep, rejection rates, and fallback counts are saved per run.
- **L2:** L1 features z-scored separately per subject from its first 20 trials, before correlation filtering and scaling. Earlier code accidentally used the normalized L2 feature matrix for L1 too; the current runner separates the raw and normalized matrices, and tests cover their difference and calibration statistics.
- **L3:** source-only MI selection, k selected from 10/20/30/40/all inside grouped CV. Random-k (10 draws) and correlation-filter-only controls are included.
- **L4:** calibration shrinkage toward the source logistic model. Pseudo-target simulations compare Brier/log-loss selection, balanced-accuracy selection (ties choose larger lambda), and finite-lambda-only Brier selection. An intercept-only calibration comparator freezes source coefficients. The actual target test sweep is descriptive only. Preprocessing is reused within each outer fold for pseudo-target fitting, as allowed for runtime, and this small optimism is documented.

Each applicable rung reports general, 10x weighted mix, shrinkage, and target-only models; the protocol defines source-derived class factors and normalized total weights for mix10. Explanation outputs include exact linear SHAP additivity checks, top-five compact faithfulness, shuffled-channel control, deletion curves/AUC, and stratified paired refits. Window sensitivity covers 2/4/8 seconds and full stimulus. Paired inference uses subject means, bootstrap CIs, sign-flip and Wilcoxon tests, and Holm adjustment. The BA preservation margin is −0.03, pre-chosen and illustrative, not tuned.

## Exploratory - not part of the main claim

Step 4a uses window-feature mean/std/slope statistics with per-subject calibration normalization and grouped MI/logistic/shrinkage. If PyTorch CPU is installed, a small GRU uses 26×128 sequences, hidden size 16, dropout 0.3, source-group early stopping, and head-only calibration fine-tuning. Integrated gradients are aggregated by channel; completeness error, masking faithfulness and deletion curves are reported. Use `--gru-subjects N` to cap runtime.

## Outputs

Each run writes CSVs, config hashes, logs, checkpoints, 150-dpi figures, `report.md`, and offline self-contained `report.html` to `results/<timestamp>/`. `latest/` contains copies of the final report artifacts. Every displayed statistic is generated from run CSVs. The report lists skipped or failed analyses with reasons; inspect `failures.csv`, `exploratory_failures.csv`, and `run.log` before interpretation. Near-chance prediction, small per-subject test sets, shared LOSO sources, masking assumptions, and non-causal interpretation are limitations.

<!-- AUTO RUN SUMMARY START -->
## Latest generated run

Output folder: `results\20261115_140431`. Completed folds: 32; failed folds: 0. Replication status: does not replicate all approximate reference metrics within the stated ±0.05 diagnostic tolerance. L0 general BA: 0.4948; L4 shrinkage_brier BA: 0.4832; compact faithfulness (L0 general/L4 shrinkage_brier): 0.6697/0.7702; L4 shrinkage_brier refit cosine: 0.3939.

All metrics above are read from the run CSVs. See the run's `report.md` and `replication_diff.md` for paired statistics, the implementation audit, limitations, and analyses that remain unavailable.
<!-- AUTO RUN SUMMARY END -->
