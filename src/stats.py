"""Paired subject-level inference helpers."""
import numpy as np
from scipy.stats import wilcoxon
import pandas as pd

def paired_test(a,b,resamples=20000,seed=42):
    a=np.asarray(a,float); b=np.asarray(b,float); ok=np.isfinite(a)&np.isfinite(b); d=(b-a)[ok]
    if not len(d): return {'n':0,'mean_delta':np.nan,'ci_low':np.nan,'ci_high':np.nan,'signflip_p':np.nan,'wilcoxon_p':np.nan,'improved':0,'tied':0,'degraded':0}
    rng=np.random.default_rng(seed); boot=d[rng.integers(0,len(d),(resamples,len(d)))].mean(1)
    flips=np.where(rng.integers(0,2,(resamples,len(d))),1,-1)*d
    p=(1+np.sum(np.abs(flips.mean(1))>=abs(d.mean())))/(resamples+1)
    try: wp=1.0 if np.allclose(d,0) else float(wilcoxon(d).pvalue)
    except ValueError: wp=1.0
    return {'n':len(d),'mean_delta':float(d.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),'signflip_p':float(p),'wilcoxon_p':wp,'improved':int((d>0).sum()),'tied':int((d==0).sum()),'degraded':int((d<0).sum())}

def holm(pvalues):
    p=np.asarray(pvalues,float); good=np.flatnonzero(np.isfinite(p)); out=np.full(len(p),np.nan)
    if len(good):
        ix=good[np.argsort(p[good])]; m=len(ix); vals=np.maximum.accumulate([(m-i)*p[j] for i,j in enumerate(ix)]); out[ix]=np.minimum(vals,1)
    return out

def build_paired_summary(metrics,stability,controls,n_resamples=20000):
    rows=[]
    endpoints={'balanced_accuracy':(metrics,'balanced_accuracy'),'compact_faithfulness':(metrics,'compact_faithfulness'),'refit_cosine':(stability,'cosine'),'top5_overlap':(stability,'top5_overlap'),'spearman':(stability,'spearman')}
    pairs=[]
    for arm in ['L1','L2','L3','L4']:
        pairs.append((f'{arm} general - L0 general',('L0','general'),(arm,'general'),'metrics'))
    personalization=['mix10','target_only','intercept_only','shrinkage_brier','shrinkage_ba','shrinkage_finite']
    for arm in ['L0','L1','L2','L3','L4']:
        for person in personalization:
            pairs.append((f'{arm} {person} - {arm} general',(arm,'general'),(arm,person),'metrics'))
    pairs.append(('L4 shrinkage_brier - L0 general',('L0','general'),('L4','shrinkage_brier'),'metrics'))
    pairs.append(('L4 shrinkage_brier - L4 mix10',('L4','mix10'),('L4','shrinkage_brier'),'metrics'))
    for variant in ['shrinkage_ba','shrinkage_finite']:
        pairs.append((f'L4 {variant} - L4 shrinkage_brier',('L4','shrinkage_brier'),('L4',variant),'metrics'))
    for ep,(df,col) in endpoints.items():
        for name,a,b,kind in pairs:
            if kind=='metrics':
                if col not in metrics:continue
                av=metrics[(metrics.arm==a[0])&(metrics.model==a[1])].set_index('subject')[col]
                bv=metrics[(metrics.arm==b[0])&(metrics.model==b[1])].set_index('subject')[col]
                common=av.index.intersection(bv.index);av=av.reindex(common);bv=bv.reindex(common)
                if not len(common):continue
            else: continue
            r=paired_test(av,bv,n_resamples);rows.append({'comparison':name,'endpoint':ep,**r})
        if ep in ('refit_cosine','top5_overlap','spearman'):
            # Compare subject-averaged stability of each personalization to the general refit.
            for arm in stability.arm.unique() if not stability.empty else []:
              for person in personalization:
                av=stability[(stability.arm==arm)&(stability.model=='general')].set_index('subject')[col]
                bv=stability[(stability.arm==arm)&(stability.model==person)].set_index('subject')[col]
                common=av.index.intersection(bv.index);av=av.reindex(common);bv=bv.reindex(common)
                if not len(common):continue
                r=paired_test(av,bv,n_resamples);rows.append({'comparison':f'{arm} {person} - {arm} general','endpoint':ep,**r})
    # Feature selection controls: average random draws within subject before paired tests.
    if not controls.empty:
        for metric in ['balanced_accuracy','compact_faithfulness']:
            if metric not in controls:continue
            c=controls[controls.arm=='L3'].groupby(['subject','method'])[metric].mean().unstack('method')
            for lhs,rhs in [('random-k','MI'),('corr-only','MI'),('random-k','corr-only')]:
                if lhs in c and rhs in c:
                    r=paired_test(c[rhs],c[lhs],n_resamples);rows.append({'comparison':f'L3 {lhs} - {rhs}','endpoint':metric,**r})
    out=pd.DataFrame(rows)
    if not out.empty:
        out['holm_p']=np.nan
        for ep,ix in out.groupby('endpoint').groups.items():out.loc[ix,'holm_p']=holm(out.loc[ix,'signflip_p'].to_numpy())
        out['holm_primary_endpoints']=np.nan
        primary=[('L4 shrinkage_brier - L0 general','compact_faithfulness'),('L4 shrinkage_brier - L4 general','refit_cosine'),('L4 shrinkage_brier - L0 general','balanced_accuracy')]
        ids=[]
        for comp,ep in primary:
            ix=out.index[(out.comparison==comp)&(out.endpoint==ep)]
            if len(ix):ids.append(ix[0])
        if ids:out.loc[ids,'holm_primary_endpoints']=holm(out.loc[ids,'signflip_p'].to_numpy())
    return out

def predictive_preservation(metrics,baseline=('L0','general'),final=('L4','shrinkage_brier'),margin=-.03,n_resamples=20000,seed=42):
    a=metrics[(metrics.arm==baseline[0])&(metrics.model==baseline[1])].set_index('subject').balanced_accuracy
    b=metrics[(metrics.arm==final[0])&(metrics.model==final[1])].set_index('subject').balanced_accuracy
    d=(b-a).dropna().to_numpy();
    if not len(d):return {'n':0,'mean_delta':np.nan,'one_sided_95_lower':np.nan,'margin':margin,'preserved':False}
    rng=np.random.default_rng(seed);boot=d[rng.integers(0,len(d),(n_resamples,len(d)))].mean(1);lo=float(np.quantile(boot,.05))
    return {'n':len(d),'mean_delta':float(d.mean()),'one_sided_95_lower':lo,'margin':margin,'preserved':bool(lo>margin),'margin_prechosen':True}
