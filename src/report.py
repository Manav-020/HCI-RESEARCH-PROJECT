from pathlib import Path
import base64, html, re, json
import numpy as np, pandas as pd

def _table(df,limit=None):
    if df is None or df.empty:return '_No rows were produced._'
    d=df if limit is None else df.head(limit)
    cols=list(d.columns);lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,r in d.iterrows():
        vals=[]
        for v in r:
            if pd.isna(v):vals.append('-')
            elif isinstance(v,(float,np.floating)):vals.append(f'{float(v):.5f}')
            else:vals.append(str(v).replace('|','\\|'))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def write_report(out,config,exploratory=None):
    p=Path(out)
    def read(name):
        f=p/name
        try:return pd.read_csv(f) if f.exists() and f.stat().st_size>2 else pd.DataFrame()
        except Exception:return pd.DataFrame()
    metrics=read('subject_metrics.csv'); paired=read('paired_summary.csv'); stability=read('stability_refits.csv');controls=read('feature_selection_controls.csv');sweep=read('lambda_sweep.csv');selection=read('lambda_selection.csv');deletion=read('deletion_curves.csv');window=read('window_sensitivity.csv');explore=read('exploratory_metrics.csv');fail=read('failures.csv');pres=read('predictive_preservation.csv');artifact=read('artifact_rejection.csv');shuffled=read('shuffled_controls.csv');shuffle_cal=read('shuffled_calibration.csv');explore_fail=read('exploratory_failures.csv');scaleaudit=read('signal_scale_audit.csv');thresholdaudit=read('artifact_threshold_sweep.csv')
    lambdaselect=read('lambda_selection.csv')
    if 'deletion_auc' not in metrics:metrics['deletion_auc']=np.nan
    if 'compact_faithfulness' not in controls:controls['compact_faithfulness']=np.nan
    nfold=metrics.subject.nunique() if 'subject' in metrics else 0;nfailed=fail.subject.nunique() if 'subject' in fail else 0
    def avg(arm,model,col):
        if metrics.empty or col not in metrics:return np.nan
        q=metrics[(metrics.arm==arm)&(metrics.model==model)][col]
        return float(q.mean()) if q.notna().any() else np.nan
    ba0=avg('L0','general','balanced_accuracy');baf=avg('L4','shrinkage_brier','balanced_accuracy');fa0=avg('L0','general','compact_faithfulness');faf=avg('L4','shrinkage_brier','compact_faithfulness')
    l0mixba=avg('L0','mix10','balanced_accuracy');l0mixfa=avg('L0','mix10','compact_faithfulness')
    refchecks=[('general balanced accuracy',ba0,.49),('mix10 balanced accuracy',l0mixba,.56),('general compact faithfulness',fa0,.68),('mix10 compact faithfulness',l0mixfa,.60)]
    refwarn=[f'{name}: observed {value:.4f}, approximate reference {reference:.2f}' for name,value,reference in refchecks if np.isfinite(value) and abs(value-reference)>.05]
    replication_note=('Notebook replication check (warning threshold: absolute difference > 0.05): '+('; '.join(refwarn) if refwarn else 'all four L0 metrics are within 0.05 of their approximate reference values.') )
    brier_sel=lambdaselect[lambdaselect.selection_variant=='shrinkage_brier'] if not lambdaselect.empty and 'selection_variant' in lambdaselect else pd.DataFrame()
    inf_selected=int(brier_sel.selected_lambda.astype(str).str.lower().isin(['inf','infinity']).sum()) if not brier_sel.empty else 0
    variants=', '.join(f"{name}={int((lambdaselect[lambdaselect.selection_variant==name].selected_lambda.astype(str).str.lower().isin(['inf','infinity'])).sum())}/{int((lambdaselect.selection_variant==name).sum())} selected infinity" for name in ('shrinkage_brier','shrinkage_ba','shrinkage_finite')) if not lambdaselect.empty and 'selection_variant' in lambdaselect else 'no lambda selections recorded'
    lambda_interpretation=(f"Pseudo-target selection results: {variants}. Infinity returns the source model, so those selections do not adapt coefficients on the target." if len(lambdaselect) else "No pseudo-target lambda selections were recorded.")
    cosf=stability[(stability.arm=='L4')&(stability.model=='shrinkage_brier')].cosine.mean() if not stability.empty and {'arm','model','cosine'}.issubset(stability.columns) else np.nan
    diff=baf-ba0 if np.isfinite(baf) and np.isfinite(ba0) else np.nan
    presline='Not estimable from these results.'
    if not pres.empty and pd.notna(pres.iloc[0].get('one_sided_95_lower')):
        pr=pres.iloc[0];presline=f"The one-sided 95% lower bound for the L4 shrinkage minus L0 general BA difference is {pr.one_sided_95_lower:.4f}; the pre-chosen illustrative margin is {pr.margin:.2f}; preserved={bool(pr.preserved)}."
    def paired_line(comp,endpoint,label):
        q=paired[(paired.comparison==comp)&(paired.endpoint==endpoint)] if not paired.empty and {'comparison','endpoint'}.issubset(paired.columns) else pd.DataFrame()
        if q.empty:return f'{label}: not assessable because the paired endpoint has no rows.'
        r=q.iloc[0]
        if pd.isna(r.ci_low):return f'{label}: not assessable (no paired observations).'
        if r.ci_low>0:result='the paired 95% interval is above zero'
        elif r.ci_high<0:result='the paired 95% interval is below zero'
        else:result='the paired 95% interval includes zero'
        hp=r.get('holm_primary_endpoints',r.holm_p)
        return f"{label}: mean delta {r.mean_delta:+.4f} (95% CI {r.ci_low:+.4f}, {r.ci_high:+.4f}; sign-flip p={r.signflip_p:.4f}; Holm across primary endpoints p={hp:.4f}); {result}."
    evidence='\n'.join(['- '+paired_line('L4 shrinkage_brier - L0 general','balanced_accuracy','Final vs baseline balanced accuracy'),'- '+paired_line('L4 shrinkage_brier - L0 general','compact_faithfulness','Final vs baseline compact faithfulness'),'- '+paired_line('L4 shrinkage_brier - L4 general','refit_cosine','Personalized vs general refit stability')])
    status=[]
    def statusrow(name,done,reason=''):
        status.append({'analysis':name,'status':'computed' if done else 'not computed','reason':reason if not done else ''})
    statusrow('LOSO prediction',nfold>0,'No completed subject metric rows.')
    statusrow('Compact faithfulness',metrics.compact_faithfulness.notna().any() if 'compact_faithfulness' in metrics else False,'No valid faithfulness rows.')
    statusrow('Stability refits',len(stability)>0,'No bootstrap refit rows.')
    st_arms=set(stability.arm) if 'arm' in stability else set();statusrow('Stability refits across all rungs',set(['L0','L1','L2','L3','L4']).issubset(st_arms),'Bootstrap stability rows are limited to arms represented in stability_refits.csv.')
    statusrow('Deletion curves',len(deletion)>0,'No deletion curve rows.')
    statusrow('MI/random-k/correlation controls',len(controls)>0,'No selection-control rows.')
    statusrow('Window sensitivity',len(window)>0,'No window sensitivity rows.')
    statusrow('Exploratory 4a',bool((explore.features=='mean_std_slope').any()) if 'features' in explore else False,'The 4a model did not produce metric rows.')
    statusrow('Exploratory GRU',bool(explore.model.astype(str).str.contains('gru|head',case=False).any()) if 'model' in explore else False,'The GRU did not produce metric rows; see exploratory_failures.csv.')
    statusrow('Shuffled-channel attribution control',len(read('shuffled_controls.csv'))>0,'No shuffled-channel control rows.')
    statusrow('Shuffled calibration-label control',len(read('shuffled_calibration.csv'))>0,'No shuffled calibration-label rows.')
    statusrow('Predictive-preservation bound',not pres.empty and pd.notna(pres.iloc[0].get('one_sided_95_lower')),'No assessable paired BA differences.')
    statusrow('Pseudo-target lambda selection',len(lambdaselect)>0,'No pseudo-target selection rows.')
    statusrow('Exact SHAP additivity assertion',('shap_additivity_max_error' in metrics and metrics.shap_additivity_max_error.dropna().max()<1e-7),'No additivity errors were saved or the maximum exceeded 1e-7.')
    statusrow('Paired inferential statistics',not paired.empty,'No paired summary rows.')
    statusrow('Per-subject correlation matrices',(p/'correlation_example.csv').exists(),'The pipeline did not export before/after-filter correlation matrices.')
    statusrow('Subject fold failures',nfailed==0,'See failures.csv for fold errors.')
    statusdf=pd.DataFrame(status);statusdf.to_csv(p/'analysis_status.csv',index=False)
    metric_summary=(metrics.groupby(['arm','model'],dropna=False).agg(n=('subject','nunique'),balanced_accuracy=('balanced_accuracy','mean'),compact_faithfulness=('compact_faithfulness','mean'),macro_f1=('macro_f1','mean'),roc_auc=('roc_auc','mean'),brier=('brier','mean'),log_loss=('log_loss','mean'),deletion_auc=('deletion_auc','mean')).reset_index() if not metrics.empty else pd.DataFrame())
    lambda_display=[]
    for _,r in lambdaselect.iterrows():
        key=str(r.get('selected_lambda','inf'))
        row={'subject':int(r.subject),'selection_variant':r.get('selection_variant','shrinkage_brier'),'selected_lambda':float('inf') if key.lower() in ('inf','infinity') else float(key),'pseudo_subjects':r.get('pseudo_subjects',np.nan)}
        for src,dst in [('selection_brier','selected_cv_brier'),('selection_log_loss','selected_cv_log_loss'),('selection_balanced_accuracy','selected_cv_balanced_accuracy')]:
            try:row[dst]=json.loads(r[src]).get(key,np.nan)
            except Exception:row[dst]=np.nan
        lambda_display.append(row)
    lambda_display=pd.DataFrame(lambda_display)
    gru=explore[(explore.model.astype(str).str.contains('gru|head',case=False))] if not explore.empty and 'model' in explore else pd.DataFrame()
    gru_ba=float(gru.balanced_accuracy.mean()) if not gru.empty else np.nan
    gru_sentence=(f"GRU mean held-out balanced accuracy was {gru_ba:.4f}, {'not above' if gru_ba<=.5 else 'above'} the 0.5000 chance reference descriptively; no inferential test against chance was run, and this comparison is exploratory." if np.isfinite(gru_ba) else 'GRU performance was not assessable because no GRU metric rows were recorded.')
    replication_items=[
      ('Feature interval','Notebook uses stimulus sample 1024 through 8063 (Python slice 1024:8064); current L0 uses the same interval.','MATCH'),
      ('Welch settings','fs=128 Hz, nperseg=256, noverlap=128; current L0 uses the same values.','MATCH'),
      ('Band power','Both use theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz and log integrated PSD (sum times df); current code was changed from mean PSD to the notebook formula.','MATCH AFTER FIX'),
      ('Feature ordering','Channel-major, then theta/alpha/beta/gamma; current flattening preserves that ordering.','MATCH'),
      ('Correlation filter','Source-fold training only; variance <1e-12 dropped; greedy original-column order; discard later feature when |r|>0.90.','MATCH AFTER FIX'),
      ('Preprocessing CV','Correlation filter then StandardScaler inside each GroupKFold training fold; source-only 3-group CV.','MATCH'),
      ('C selection','[0.01, 0.1, 1], balanced-accuracy scoring, GroupKFold(3), first grid value wins exact ties.','MATCH'),
      ('General class handling','LogisticRegression(class_weight=balanced); explicit source class factors are algebraically equivalent and normalized to source count.','MATCH'),
      ('mix10 calibration weighting','Source class factors fixed; calibration sample gets 10x factor; all weights normalized to source sample count.','MATCH'),
      ('Solver','lbfgs, max_iter=5000; current additionally retries at 10000 only on convergence warning.','MATCH'),
      ('Explanation masks','Same exact SHAP, p=.25 independent channel masks, 100 repetitions, top-five channel sum, and reconstruction score; current now uses notebook vectorized seeded mask order and stable top-five tie order.','MATCH AFTER FIX'),
      ('Reference labels/order','Notebook REFERENCE_ROWS and workbook have equal 1280 rows and zero numeric difference; source files are reordered by presentation_trial for target calibration/test.','MATCH'),
      ('Notebook result provenance','The notebook states that no DEAP results are embedded and that it was tested on synthetic data. The ~0.56/~0.60 values are approximate prior-run references, not outputs reproducible from notebook cells in this file.','REFERENCE LIMITATION')]
    repchecks=[('L0 general BA',ba0,.49),('L0 mix10 BA',l0mixba,.56),('L0 general compact faithfulness',fa0,.68),('L0 mix10 compact faithfulness',l0mixfa,.60)]
    repdiff=[{'metric':n,'observed':v,'approximate_reference':ref,'absolute_difference':abs(v-ref),'tolerance':.05,'status':'within tolerance' if np.isfinite(v) and abs(v-ref)<=.05 else 'outside tolerance'} for n,v,ref in repchecks]
    rep_status='replicates within the stated ±0.05 diagnostic tolerance' if all(q['status']=='within tolerance' for q in repdiff) else 'does not replicate all approximate reference metrics within the stated ±0.05 diagnostic tolerance'
    repdoc='# Notebook replication audit\n\n'+f'**Status:** {rep_status}. The ±0.05 threshold is a declared diagnostic convention, not a statistical test.\n\n'+'_table_placeholder_'
    repdoc=repdoc.replace('_table_placeholder_',_table(pd.DataFrame(replication_items,columns=['item','notebook_and_current_check','status']))+'\n\n## Result check (from subject_metrics.csv)\n\n'+_table(pd.DataFrame(repdiff)))
    (p/'replication_diff.md').write_text(repdoc,encoding='utf8')
    figures=['paired_forest.png','ladder_deltas.png','faithfulness_forest.png','stability_forest.png','faithfulness_accuracy_lambda.png','lambda_faithfulness.png','window_sensitivity.png','selection_controls.png','paired_slopes.png','deletion_curves.png','channel_attribution.png','correlation_matrices.png','exploratory_comparison.png','exploratory_ig_attribution.png','predictive_performance.png']
    report=f'''# EEG emotion calibration and explanation experiment

## Abstract

This automatically generated report summarizes the completed artifacts for {nfold} LOSO target folds, with {nfailed} failed folds. It evaluates the specified DEAP calibration protocol and reports unavailable analyses explicitly. A result is not treated as evidence when its corresponding CSV contains no measured rows.

## Research question

Does subject-specific calibration make EEG emotion explanations more faithful and stable to the model's actual decision process while maintaining predictive performance?

> **Headline findings (descriptive):** folds completed {nfold}/{nfold+nfailed}; L0 general BA {ba0:.4f}; L4 shrinkage_brier BA {baf:.4f}; paired mean BA change {diff:+.4f}; compact faithfulness L0/L4 {fa0:.4f}/{faf:.4f}; L4 refit cosine {cosf:.4f}. Replication: {rep_status}. Interpret only alongside the paired intervals and availability table below.

## Run and data protocol

The input workbook was `{config.get('reference_file','unknown')}`. Reference rows were joined to video-order signal trials, checked against stored dominance and liking, and sorted into presentation order. Per target, trials 1–20 are calibration and 21–40 are held-out test. The test labels are used only for evaluation and descriptive test-only lambda curves. LOSO source subjects and each subject’s first 20 unlabeled features fit preprocessing. Seed is {config.get('seed',42)}; n_jobs is {config.get('n_jobs',1)}.

L1/L2 bug investigation: the previous runner passed the already normalized 4-second feature dictionary to both rungs, which made L2 a no-op relative to L1. The current runner retains raw L1 features and separately derives L2 by per-subject z-scoring from trials 1–20 before source correlation filtering and scaling. Tests assert the L2 features differ and that calibration means and standard deviations are approximately 0 and 1. The feature cache is at `{config.get('feature_cache','cache')}`; entries are keyed by subject signal-file hashes. Available cached subjects are reused, and only missing cache entries are extracted.

## Methods

- **L0:** full stimulus log band powers, correlation filtering, source scaling and grouped logistic model selection.
- **L1:** 2/4/8-second stimulus windows, artifact rejection, and robust median aggregation; no baseline samples.
- **L2:** L1 features normalized independently for each subject from its first 20 presentation-order trials.
- **L3:** MI ranking within an sklearn pipeline, recomputed inside GroupKFold training folds, with k selected from the configured grid. Random-k and correlation-only controls are reported.
- **L4:** L3 features with source-to-calibration coefficient shrinkage. Lambda is selected using pseudo-target calibration/validation within source subjects. The test-only lambda sweep is descriptive and is not used for selection. Per-fold preprocessing is reused for pseudo-target models, which introduces the documented small optimism in the pseudo-target score.

The mixing comparator applies 10x calibration weights relative to source weights, fixes class factors from source labels, and normalizes total weight to the source sample count. The intercept-only comparator freezes source coefficients and fits only the intercept on balanced calibration log-loss. Exact additive linear explanations use source-background means in transformed feature space. All models are evaluated on the same high-valence log-odds quantity. Faithfulness uses independent channel masks (p=.25) and top-five compact predictions. Stability uses stratified source/calibration bootstrap refits. The exploratory section is excluded from the main claim.

## Results

L0 general mean balanced accuracy: **{ba0:.4f}**. L4 shrinkage_brier mean balanced accuracy: **{baf:.4f}**. Difference: **{diff:+.4f}**. L0 general compact faithfulness: **{fa0:.4f}**; L4 shrinkage_brier compact faithfulness: **{faf:.4f}**. L4 shrinkage_brier refit cosine stability: **{cosf:.4f}**. Missing or NaN values are not substituted. See `replication_diff.md` for the line-item audit and post-fix metric check.

{replication_note}

{lambda_interpretation}

{presline}

### Metrics by rung and model

{_table(metric_summary)}

### Paired statistics

The table contains subject-averaged paired mean changes, paired bootstrap intervals, sign-flip tests, Wilcoxon checks, counts, and endpoint-wise Holm-adjusted p-values. Quick runs use the configured quick resample count; full runs use 20,000 resamples.

{evidence}

{_table(paired)}

### Feature-selection controls

Random-k results are averaged over the ten draws within subject before paired analyses. The L2 correlation-only control is shown alongside L3 MI and random-k.

{_table(controls.groupby(['arm','method'],dropna=False).agg(n=('subject','nunique'),balanced_accuracy=('balanced_accuracy','mean'),compact_faithfulness=('compact_faithfulness','mean')).reset_index() if not controls.empty else controls)}

### Shuffled controls

The `score` column is shuffled-channel compact faithfulness; values are summarized by arm and model.

{_table(shuffled.assign(metric='shuffled_channel_compact_faithfulness').groupby(['arm','model','metric']).score.agg(['mean','std','count']).reset_index() if not shuffled.empty else shuffled)}

The table below reports held-out balanced accuracy and Brier score after shuffled calibration labels.

{_table(shuffle_cal.groupby('arm').agg(n_subjects=('subject','nunique'),balanced_accuracy=('balanced_accuracy','mean'),brier=('brier','mean')).reset_index() if not shuffle_cal.empty else shuffle_cal)}

### Lambda selection and descriptive sweep

Pseudo-target selection is compared for shrinkage_brier, shrinkage_ba, and shrinkage_finite. The selected-CV columns below are numeric summaries at the selected lambda. Actual target test-set lambda results below are descriptive only and were not used to select lambda.

{_table(lambda_display)}

{_table(sweep.groupby('lambda',dropna=False).agg(n=('subject','nunique'),balanced_accuracy=('balanced_accuracy','mean'),compact_faithfulness=('compact_faithfulness','mean'),brier=('brier','mean')).reset_index() if not sweep.empty else sweep)}

### Window sensitivity and artifact rejection

Artifact thresholds were set to robust RMS z ≤ 8 and peak-to-peak ≤ 2000 input units after auditing the supplied signal amplitude. The values are in the dataset-provided signal scale; no conversion to microvolts is assumed. The threshold sweep includes the original 4/200 rule and the selected rule.

{_table(scaleaudit)}

{_table(thresholdaudit.groupby(['window_s','rms_robust_z_max','peak_to_peak_max']).agg(fallback_trials=('fallback_trials','sum'),fallback_rate=('fallback_rate','mean'),rejected_window_rate=('rejected_window_rate','mean')).reset_index() if not thresholdaudit.empty else thresholdaudit)}

{_table(window.groupby('window_s').agg(n=('subject','nunique'),balanced_accuracy=('balanced_accuracy','mean'),compact_faithfulness=('compact_faithfulness','mean')).reset_index() if not window.empty else window)}

{_table(artifact.groupby('window_s').agg(n_subjects=('subject','nunique'),mean_rejection_rate=('rejection_rate','mean'),fallback_trials=('fallback_trials','sum')).reset_index() if not artifact.empty else artifact)}

## Figures

{''.join(f'![{x}]({x})\n\n' for x in figures if (p/x).exists())}

## Exploratory - not part of the main claim

This section reports the window-statistics (mean/std/slope) classifier and optional GRU only. It is separated from the main ladder and does not enter the main inference.

{gru_sentence}

{_table(explore)}

## Failures and analyses not computed

Completed subject folds: {nfold}; failed subject folds: {nfailed}. A failed-fold count changes the analysis N and is not imputed. See failures.csv and exploratory_failures.csv.

{_table(statusdf)}

The officially released DEAP provenance was not independently verified. Each target has only 20 test trials; LOSO folds share source subjects. Faithfulness depends on the channel-masking assumption and weight concentration and is not physiological causality. MI is pooled across source subjects. Near-chance predictive performance limits explanation value. The −0.03 predictive-preservation margin is illustrative and pre-chosen, not tuned.

The workbook provenance was not independently verified. The exploratory GRU can overfit the small trial set; its results are explicitly outside the main claim.

### Failed folds

{_table(fail)}

Exploratory failures:

{_table(explore_fail)}

## Reviewer questions

**Why MI rather than differential entropy?** For a band-limited Gaussian, differential entropy is monotonic in log power; MI gives a distinct label-relevance criterion.

**Why windows rather than baseline correction?** Short baselines can include movement artifacts; window aggregation can reduce the impact of isolated noisy intervals.

**Why shrinkage?** It limits changes to the source decision rule when target calibration labels are few.

**Why top five?** It is the pre-specified compact explanation size; full and compact faithfulness are reported separately.

**Why is subject normalization adaptation?** It uses target calibration-period features, even without their labels.

## Reproducibility

See config.json for seeds, subjects, package versions, feature-cache references, and hashes. `run.log` records runtime and warnings; the CSV files are the source of every reported statistic.
'''
    (p/'report.md').write_text(report,encoding='utf8')
    # Convert the same Markdown content to an offline, self-contained HTML document.
    ids=[];body=[];lines=report.splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('# '):body.append(f'<h1>{html.escape(line[2:])}</h1>')
        elif line.startswith('## '):
            title=line[3:];anchor=re.sub('[^a-z0-9]+','-',title.lower()).strip('-');ids.append((title,anchor));body.append(f'<h2 id="{anchor}">{html.escape(title)}</h2>')
        elif line.startswith('|'):
            tbl=[]
            while i<len(lines) and lines[i].startswith('|'):tbl.append(lines[i]);i+=1
            i-=1;rr=[[c.strip() for c in row.strip('|').split('|')] for row in tbl if not all(set(c.strip())<=set('-: ') for c in row.strip('|').split('|'))]
            if rr:
                table='<table><thead><tr>'+''.join(f'<th>{html.escape(c)}</th>' for c in rr[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(c)}</td>' for c in r)+'</tr>' for r in rr[1:])+'</tbody></table>'
                body.append('<details><summary>Expand table</summary>'+table+'</details>' if len(rr)>22 else table)
        elif line.startswith('!['):
            m=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            if m and (p/m.group(2)).exists():
                uri='data:image/png;base64,'+base64.b64encode((p/m.group(2)).read_bytes()).decode('ascii');body.append(f'<img alt="{html.escape(m.group(1))}" src="{uri}">')
        elif line.startswith('- '):body.append(f'<li>{html.escape(line[2:])}</li>')
        elif line.strip():
            para=html.escape(line).replace('**','<b>',1) if False else html.escape(line)
            para=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',para);para=re.sub(r'`(.+?)`',r'<code>\1</code>',para);body.append(f'<p>{para}</p>')
        i+=1
    toc='<nav><b>Contents</b><ul>'+''.join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for t,a in ids)+'</ul></nav>'
    page='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>body{font:16px system-ui;max-width:1150px;margin:2em auto;padding:0 1em;color:#1c2430;background:#fff}h1,h2{color:#183b56}table{border-collapse:collapse;width:100%;margin:1em 0}th,td{border:1px solid #b8c4d0;padding:.45em;text-align:left}th{background:#e8eff7}nav,aside{padding:1em;background:#eef4fa;border-left:4px solid #4472c4}img{max-width:100%;height:auto;margin:1em 0}code{background:#f1f3f5;padding:.1em .25em}details{margin:1em 0}summary{cursor:pointer;font-weight:600}</style></head><body>'+toc+''.join(body)+'</body></html>'
    (p/'report.html').write_text(page,encoding='utf8')
    root=p.parent.parent
    readme=root/'README.md'
    if readme.exists():
        old=readme.read_text(encoding='utf8');start='<!-- AUTO RUN SUMMARY START -->';end='<!-- AUTO RUN SUMMARY END -->'
        generated=f'''{start}
## Latest generated run

Output folder: `{p.relative_to(root)}`. Completed folds: {nfold}; failed folds: {nfailed}. Replication status: {rep_status}. L0 general BA: {ba0:.4f}; L4 shrinkage_brier BA: {baf:.4f}; compact faithfulness (L0 general/L4 shrinkage_brier): {fa0:.4f}/{faf:.4f}; L4 shrinkage_brier refit cosine: {cosf:.4f}.

All metrics above are read from the run CSVs. See the run's `report.md` and `replication_diff.md` for paired statistics, the implementation audit, limitations, and analyses that remain unavailable.
{end}'''
        if start in old and end in old:old=old[:old.index(start)]+generated+old[old.index(end)+len(end):]
        else:old=old.rstrip()+'\n\n'+generated+'\n'
        readme.write_text(old,encoding='utf8')
