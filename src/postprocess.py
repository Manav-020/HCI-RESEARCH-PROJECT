"""Generate static figures directly from run CSVs."""
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from .stats import build_paired_summary,predictive_preservation

def _save(fig,p,name):fig.tight_layout();fig.savefig(p/name,dpi=150,bbox_inches='tight');plt.close(fig)
def _read(p,name):
 f=p/name
 return pd.read_csv(f) if f.exists() and f.stat().st_size>2 else pd.DataFrame()

def generate(out,quick=False):
 p=Path(out);m=_read(p,'subject_metrics.csv');s=_read(p,'stability_refits.csv');c=_read(p,'feature_selection_controls.csv');d=_read(p,'deletion_curves.csv');w=_read(p,'window_sensitivity.csv');lam=_read(p,'lambda_sweep.csv');attr=_read(p,'channel_attributions.csv');ex=_read(p,'exploratory_metrics.csv');corr=_read(p,'correlation_example.csv');ig=_read(p,'exploratory_ig_attributions.csv')
 paired=build_paired_summary(m,s,c,1000 if quick else 20000);paired.to_csv(p/'paired_summary.csv',index=False)
 pd.DataFrame([predictive_preservation(m,n_resamples=1000 if quick else 20000)]).to_csv(p/'predictive_preservation.csv',index=False)
 if not m.empty:
  g=m.groupby(['arm','model']).balanced_accuracy.mean().dropna();fig,ax=plt.subplots(figsize=(10,4));g.plot.bar(ax=ax,color='#4472c4');ax.set_ylabel('Mean held-out balanced accuracy');ax.set_title('Predictive performance');_save(fig,p,'predictive_performance.png')
 if not paired.empty:
  for ep,name in [('balanced_accuracy','ladder_deltas.png'),('compact_faithfulness','faithfulness_forest.png'),('refit_cosine','stability_forest.png')]:
   q=paired[(paired.endpoint==ep)&paired.ci_low.notna()];fig,ax=plt.subplots(figsize=(9,4))
   if len(q):
    y=np.arange(len(q));ax.errorbar(q.mean_delta,y,xerr=[q.mean_delta-q.ci_low,q.ci_high-q.mean_delta],fmt='o',capsize=3);ax.set_yticks(y,q.comparison);ax.axvline(0,color='k',lw=.8)
   else:ax.text(.5,.5,f'{ep}: no paired rows',ha='center',va='center',transform=ax.transAxes)
   ax.set_xlabel(f'Paired mean difference: {ep} (95% bootstrap CI)');ax.set_title(f'Paired {ep} comparisons');_save(fig,p,name)
  q=paired[paired.endpoint.isin(['balanced_accuracy','compact_faithfulness','refit_cosine'])];fig,ax=plt.subplots(figsize=(9,4))
  for j,ep in enumerate(['balanced_accuracy','compact_faithfulness','refit_cosine']):
   z=q[q.endpoint==ep]
   if len(z):ax.scatter(z.mean_delta,np.full(len(z),j),label=ep,alpha=.7)
  ax.axvline(0,color='k',lw=.8);ax.set_yticks(range(3),['BA','Faithfulness','Refit cosine']);ax.set_xlabel('Paired mean delta');ax.set_title('Ladder paired deltas');_save(fig,p,'paired_forest.png')
 if not lam.empty and {'lambda','balanced_accuracy','compact_faithfulness'}.issubset(lam.columns):
  lam=lam.copy();lam['lam_num']=pd.to_numeric(lam['lambda'].replace({'inf':np.inf}),errors='coerce');q=lam.groupby('lam_num').agg(ba=('balanced_accuracy','mean'),ba_sd=('balanced_accuracy','std'),faith=('compact_faithfulness','mean'),faith_sd=('compact_faithfulness','std'),n=('subject','nunique')).reset_index().sort_values('lam_num');sef=q.faith_sd.fillna(0)/np.sqrt(q.n.clip(lower=1));seb=q.ba_sd.fillna(0)/np.sqrt(q.n.clip(lower=1))
  fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(q.faith,q.ba,xerr=1.96*sef,yerr=1.96*seb,fmt='o-',capsize=3);ax.set_xlabel('Compact faithfulness');ax.set_ylabel('Balanced accuracy');ax.set_title('Descriptive test-only lambda trade-off');_save(fig,p,'faithfulness_accuracy_lambda.png')
  x=np.arange(len(q));fig,ax=plt.subplots(figsize=(7,4));ax.plot(x,q.faith,marker='o');ax.fill_between(x,q.faith-1.96*sef,q.faith+1.96*sef,alpha=.2);ax.set_xticks(x,[str(v) for v in q.lam_num],rotation=45);ax.set_xlabel('Lambda');ax.set_ylabel('Compact faithfulness');ax.set_title('Test-only lambda faithfulness');_save(fig,p,'lambda_faithfulness.png')
 if not w.empty:
  q=w.groupby('window_s').agg(ba=('balanced_accuracy','mean'),sd=('balanced_accuracy','std'),n=('subject','nunique')).reset_index();q.window_s=q.window_s.astype(str);se=1.96*q.sd.fillna(0)/np.sqrt(q.n.clip(lower=1));fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(q.window_s,q.ba,yerr=se,marker='o',capsize=3);ax.set_xlabel('Window length');ax.set_ylabel('Balanced accuracy');ax.set_title('Window-length sensitivity');_save(fig,p,'window_sensitivity.png')
 if not c.empty:
  if 'compact_faithfulness' not in c:c['compact_faithfulness']=np.nan
  q=c.groupby('method').agg(ba=('balanced_accuracy','mean'),faith=('compact_faithfulness','mean')).reset_index();fig,axes=plt.subplots(1,2,figsize=(9,4));axes[0].bar(q.method,q.ba,color='#70ad47');axes[0].set_ylabel('Balanced accuracy');axes[0].tick_params(axis='x',rotation=30);axes[1].bar(q.method,q.faith,color='#ed7d31');axes[1].set_ylabel('Faithfulness');axes[1].tick_params(axis='x',rotation=30);fig.suptitle('MI vs random-k vs correlation-only');_save(fig,p,'selection_controls.png')
 if not m.empty:
  fig,axes=plt.subplots(1,3,figsize=(12,4))
  for ax,metric in zip(axes,['balanced_accuracy','compact_faithfulness','refit_cosine']):
   if metric=='refit_cosine':
    if s.empty:ix=[]
    else:a=s[(s.arm=='L4')&(s.model=='general')].set_index('subject').cosine;b=s[(s.arm=='L4')&(s.model=='shrinkage_brier')].set_index('subject').cosine;ix=a.index.intersection(b.index)
   else:a=m[(m.arm=='L4')&(m.model=='general')].set_index('subject')[metric];b=m[(m.arm=='L4')&(m.model=='shrinkage_brier')].set_index('subject')[metric];ix=a.index.intersection(b.index)
   for sid in ix:ax.plot([0,1],[a[sid],b[sid]],marker='o',alpha=.55)
   ax.set_xticks([0,1],['L4 general','L4 shrinkage_brier']);ax.set_ylabel(metric.replace('_',' ').title())
  fig.suptitle('Paired subject slopes');_save(fig,p,'paired_slopes.png')
 if not d.empty:
  q=d.groupby(['order','k'])[['log_odds_change','probability_change']].mean().reset_index();fig,axes=plt.subplots(1,2,figsize=(10,4))
  for order,z in q.groupby('order'):axes[0].plot(z.k,z.log_odds_change,marker='o',label=order);axes[1].plot(z.k,z.probability_change,marker='o',label=order)
  axes[0].set_xlabel('Channels masked');axes[0].set_ylabel('Log-odds change');axes[1].set_xlabel('Channels masked');axes[1].set_ylabel('Probability change');axes[0].legend();axes[1].legend();fig.suptitle('Deletion curves');_save(fig,p,'deletion_curves.png')
 if not attr.empty:
  q=attr[(attr.arm=='L4')&attr.model.isin(['general','mix10','shrinkage_brier','shrinkage_ba','shrinkage_finite','intercept_only'])].groupby(['model','channel']).mean_abs_attribution.mean().reset_index();fig,ax=plt.subplots(figsize=(11,4))
  for model,z in q.groupby('model'):ax.plot(z.channel,z.mean_abs_attribution,marker='.',label=model)
  ax.set_xlabel('EEG channel index');ax.set_ylabel('Mean absolute attribution');ax.set_title('L4 channel attributions');ax.legend();_save(fig,p,'channel_attribution.png')
 if not corr.empty:
  fig,axes=plt.subplots(1,2,figsize=(10,4))
  for ax,stage in zip(axes,('before','after')):
   q=corr[corr.stage==stage];mat=q.pivot(index='feature_i',columns='feature_j',values='correlation').to_numpy();im=ax.imshow(mat,vmin=-1,vmax=1,cmap='coolwarm');ax.set_title(stage)
  fig.colorbar(im,ax=axes.ravel().tolist());fig.suptitle('Correlation matrices before/after filter');_save(fig,p,'correlation_matrices.png')
 if not ex.empty:
  q=ex.groupby(['features','model']).balanced_accuracy.mean();fig,ax=plt.subplots(figsize=(9,4));q.plot.bar(ax=ax,color='#8064a2');ax.set_ylabel('Balanced accuracy');ax.set_title('EXPLORATORY - not part of the main claim');_save(fig,p,'exploratory_comparison.png')
 if not ig.empty:
  q=ig.groupby(['model','channel']).mean_abs_ig.mean().reset_index();fig,ax=plt.subplots(figsize=(10,4))
  for model,z in q.groupby('model'):ax.plot(z.channel,z.mean_abs_ig,label=model)
  ax.set_xlabel('EEG channel index');ax.set_ylabel('Mean absolute integrated gradient');ax.set_title('EXPLORATORY - GRU integrated gradients');ax.legend();_save(fig,p,'exploratory_ig_attribution.png')
 return paired
