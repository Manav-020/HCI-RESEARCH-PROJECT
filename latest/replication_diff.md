# Notebook replication audit

**Status:** does not replicate all approximate reference metrics within the stated ±0.05 diagnostic tolerance. The ±0.05 threshold is a declared diagnostic convention, not a statistical test.

| item | notebook_and_current_check | status |
| --- | --- | --- |
| Feature interval | Notebook uses stimulus sample 1024 through 8063 (Python slice 1024:8064); current L0 uses the same interval. | MATCH |
| Welch settings | fs=128 Hz, nperseg=256, noverlap=128; current L0 uses the same values. | MATCH |
| Band power | Both use theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz and log integrated PSD (sum times df); current code was changed from mean PSD to the notebook formula. | MATCH AFTER FIX |
| Feature ordering | Channel-major, then theta/alpha/beta/gamma; current flattening preserves that ordering. | MATCH |
| Correlation filter | Source-fold training only; variance <1e-12 dropped; greedy original-column order; discard later feature when \|r\|>0.90. | MATCH AFTER FIX |
| Preprocessing CV | Correlation filter then StandardScaler inside each GroupKFold training fold; source-only 3-group CV. | MATCH |
| C selection | [0.01, 0.1, 1], balanced-accuracy scoring, GroupKFold(3), first grid value wins exact ties. | MATCH |
| General class handling | LogisticRegression(class_weight=balanced); explicit source class factors are algebraically equivalent and normalized to source count. | MATCH |
| mix10 calibration weighting | Source class factors fixed; calibration sample gets 10x factor; all weights normalized to source sample count. | MATCH |
| Solver | lbfgs, max_iter=5000; current additionally retries at 10000 only on convergence warning. | MATCH |
| Explanation masks | Same exact SHAP, p=.25 independent channel masks, 100 repetitions, top-five channel sum, and reconstruction score; current now uses notebook vectorized seeded mask order and stable top-five tie order. | MATCH AFTER FIX |
| Reference labels/order | Notebook REFERENCE_ROWS and workbook have equal 1280 rows and zero numeric difference; source files are reordered by presentation_trial for target calibration/test. | MATCH |
| Notebook result provenance | The notebook states that no DEAP results are embedded and that it was tested on synthetic data. The ~0.56/~0.60 values are approximate prior-run references, not outputs reproducible from notebook cells in this file. | REFERENCE LIMITATION |

## Result check (from subject_metrics.csv)

| metric | observed | approximate_reference | absolute_difference | tolerance | status |
| --- | --- | --- | --- | --- | --- |
| L0 general BA | 0.49483 | 0.49000 | 0.00483 | 0.05000 | within tolerance |
| L0 mix10 BA | 0.48907 | 0.56000 | 0.07093 | 0.05000 | outside tolerance |
| L0 general compact faithfulness | 0.66967 | 0.68000 | 0.01033 | 0.05000 | within tolerance |
| L0 mix10 compact faithfulness | 0.68729 | 0.60000 | 0.08729 | 0.05000 | outside tolerance |