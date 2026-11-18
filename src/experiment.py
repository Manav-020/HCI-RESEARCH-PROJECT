"""Full LOSO experiment using cached features and leakage-aware model fitting."""
from pathlib import Path
import copy, json, time, logging, traceback, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .pipeline import (ROOT,DATA,SEED,load_all,cached_features,norm_subject,metrics,assert_no_target_leak,extract_trial,artifact_scale_audit,ARTIFACT_RMS_Z_MAX,ARTIFACT_P2P_MAX)
from .models import (grouped_select,build_pipeline,transform,params,predict_logit,predict_prob,explain,faithfulness,shuffled_faithfulness,deletion,selected_original_indices)
from .stats import paired_test,holm

LAMBDAS=[0,.01,.1,1,10,100,1000,np.inf]
CKPT_NAMES=['allrows','sweeps','lambdaselect','ctrl','stab','deletions','shuffle','sens','attrib_subject','explore','gru_deletions','gru_attributions','corr_example','shuffle_cal','fails']

def load_fold_checkpoints(out):
    d=Path(out)/'checkpoints';d.mkdir(exist_ok=True);loaded={k:[] for k in CKPT_NAMES};done=set()
    for marker in d.glob('subject_*_complete.txt'):
        sid=int(marker.stem.split('_')[1]);done.add(sid)
    for name in CKPT_NAMES:
        for f in d.glob(f'subject_*_{name}.csv'):
            try:loaded[name].extend(pd.read_csv(f).to_dict('records'))
            except pd.errors.EmptyDataError:pass
    return loaded,done

def save_fold_checkpoint(out,sid,lists):
    d=Path(out)/'checkpoints';d.mkdir(exist_ok=True)
    for name in CKPT_NAMES:
        rows=[r for r in lists[name] if int(r.get('subject',-1))==sid]
        if rows:pd.DataFrame(rows).to_csv(d/f'subject_{sid:02d}_{name}.csv',index=False)
    (d/f'subject_{sid:02d}_complete.txt').write_text('complete',encoding='utf8')

def fixed_fit(X,y,groups,C,k,selection='mi',seed=42):
    p=build_pipeline(C,k,selection,seed)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        try:p.fit(X,y)
        except ConvergenceWarning:p.set_params(model__max_iter=10000);p.fit(X,y)
    return {'pipeline':p,'C':C,'k':k,'selection':selection}

def weighted_bundle(srcfit,Xsrc,ysrc,Xcal,ycal):
    b=copy.deepcopy(srcfit); c=srcfit['pipeline'].named_steps['model'].C
    sw=weighted_mix_sample_weights(ysrc,ycal)
    z=np.vstack([transform(srcfit,Xsrc),transform(srcfit,Xcal)]); y=np.r_[ysrc,ycal]
    m=LogisticRegression(C=c,solver='lbfgs',max_iter=5000,class_weight=None,random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        try:m.fit(z,y,sample_weight=sw)
        except ConvergenceWarning:m.set_params(max_iter=10000);m.fit(z,y,sample_weight=sw)
    b['coef']=m.coef_.ravel();b['intercept']=float(m.intercept_[0]);return b

def weighted_mix_sample_weights(ysrc,ycal):
    n=len(ysrc);counts=np.bincount(ysrc,minlength=2);factors=np.array([n/(2*counts[k]) if counts[k] else 0 for k in range(2)])
    w=np.r_[factors[ysrc],10*factors[ycal]];return w*n/w.sum()

def target_bundle(srcfit,Xcal,ycal):
    if len(np.unique(ycal))<2:return None
    b=copy.deepcopy(srcfit); old=srcfit['pipeline'].named_steps['model'];m=LogisticRegression(C=old.C,solver='lbfgs',max_iter=5000,class_weight='balanced',random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        try:m.fit(transform(srcfit,Xcal),ycal)
        except ConvergenceWarning:m.set_params(max_iter=10000);m.fit(transform(srcfit,Xcal),ycal)
    b['coef']=m.coef_.ravel();b['intercept']=float(m.intercept_[0]);return b

def shrink_objective(t,X,y,sw,ws,bs,lam):
    w,b=t[:-1],t[-1];p=np.clip(expit(X@w+b),1e-12,1-1e-12);loss=-np.sum(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+.5*lam*np.sum((w-ws)**2)+.5*lam*(b-bs)**2
    return loss,np.r_[X.T@(sw*(p-y))+lam*(w-ws),np.sum(sw*(p-y))+lam*(b-bs)]

def shrink_bundle(srcfit,Xcal,ycal,lam):
    ws,bs=params(srcfit)
    if np.isinf(lam):
        out=copy.deepcopy(srcfit);out['coef']=ws.copy();out['intercept']=float(bs);return out
    X=transform(srcfit,Xcal); y=np.asarray(ycal);cnt=np.bincount(y,minlength=2);sw=np.array([len(y)/(2*cnt[v]) if cnt[v] else 0 for v in y]);l=max(float(lam),1e-6)
    res=minimize(lambda t:shrink_objective(t,X,y,sw,ws,bs,l),np.r_[ws,bs],jac=True,method='L-BFGS-B')
    if not res.success and not np.isfinite(res.fun):raise RuntimeError('Shrinkage optimizer failed: '+res.message)
    out=copy.deepcopy(srcfit);out['coef']=res.x[:-1];out['intercept']=float(res.x[-1]);return out

def intercept_bundle(srcfit,Xcal,ycal):
    """Personalize only the intercept; source coefficients remain frozen."""
    coef,b0=params(srcfit);z=transform(srcfit,Xcal);y=np.asarray(ycal,dtype=float)
    if len(y)==0:raise ValueError('Intercept personalization needs calibration trials')
    counts=np.bincount(y.astype(int),minlength=2);sw=np.array([len(y)/(2*counts[int(v)]) if counts[int(v)] else 0.0 for v in y])
    base=z@coef
    def objective(v):
        b=float(v[0]);p=np.clip(expit(base+b),1e-12,1-1e-12)
        loss=-np.sum(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+0.5e-6*(b-b0)**2
        grad=np.sum(sw*(p-y))+1e-6*(b-b0)
        return float(loss),np.asarray([grad])
    res=minimize(objective,np.asarray([b0]),jac=True,method='L-BFGS-B',bounds=[(b0-30,b0+30)])
    if not np.isfinite(res.fun):raise RuntimeError('Intercept-only optimizer returned nonfinite loss')
    out=copy.deepcopy(srcfit);out['coef']=coef.copy();out['intercept']=float(res.x[0]);return out

def source_arrays(features,data,sids):
    return np.concatenate([features[s] for s in sids]),np.concatenate([data[s]['y'] for s in sids]),np.repeat(np.asarray(sids),40)

def pseudo_lambda(features,data,source_sids,selection,k,C,kgrid,cgrid,basefit=None):
    vals={str(v):[] for v in LAMBDAS}
    Xall,yall,gall=source_arrays(features,data,source_sids)
    for pseudo in source_sids:
        fit_sids=[s for s in source_sids if s!=pseudo];assert pseudo not in fit_sids
        if basefit is None:
            X,y,g=source_arrays(features,data,fit_sids); fit=grouped_select(X,y,g,selection,kgrid,cgrid,SEED)
        else:
            # Per-fold preprocessing (including scaler and MI ranking) is reused as permitted for runtime;
            # the pseudo-target labels are excluded from the pseudo source classifier fit.
            keep=gall!=pseudo;fit=copy.deepcopy(basefit);Zall=transform(basefit,Xall);cv=GroupKFold(3);best=None
            for cc in cgrid:
                sc=[]
                for it,iv in cv.split(Zall[keep],yall[keep],gall[keep]):
                    mm=LogisticRegression(C=cc,solver='lbfgs',max_iter=5000,class_weight='balanced',random_state=SEED)
                    with warnings.catch_warnings():
                        warnings.simplefilter('error',ConvergenceWarning)
                        try:mm.fit(Zall[keep][it],yall[keep][it])
                        except ConvergenceWarning:mm.set_params(max_iter=10000);mm.fit(Zall[keep][it],yall[keep][it])
                    sc.append(balanced_accuracy_score(yall[keep][iv],mm.predict(Zall[keep][iv])))
                score=float(np.mean(sc))
                if best is None or score>best[0]:best=(score,cc)
            m=LogisticRegression(C=best[1],solver='lbfgs',max_iter=5000,class_weight='balanced',random_state=SEED)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                try:m.fit(Zall[keep],yall[keep])
                except ConvergenceWarning:m.set_params(max_iter=10000);m.fit(Zall[keep],yall[keep])
            fit['pipeline'].named_steps['model']=m;X,y=Xall[keep],yall[keep]
        xcal=features[pseudo][:20];ycal=data[pseudo]['y'][:20];xval=features[pseudo][20:];yval=data[pseudo]['y'][20:]
        for lam in LAMBDAS:
            sh=shrink_bundle(fit,xcal,ycal,lam);p=np.clip(predict_prob(sh,xval),1e-7,1-1e-7)
            vals[str(lam)].append({'brier':brier_score_loss(yval,p),'log_loss':log_loss(yval,p,labels=[0,1]),
                                   'balanced_accuracy':balanced_accuracy_score(yval,(p>=.5).astype(int)) if len(np.unique(yval))==2 else np.nan})
    mean={key:{metric:float(np.nanmean([r[metric] for r in runs])) for metric in ('brier','log_loss','balanced_accuracy')} for key,runs in vals.items() if runs}
    return select_lambda_variants(mean),mean

def select_lambda_variants(mean):
    def lam_value(k):return np.inf if str(k).lower() in ('inf','infinity') else float(k)
    brier=min(mean,key=lambda q:(mean[q]['brier'],mean[q]['log_loss']))
    ba=max(mean,key=lambda q:(-np.inf if not np.isfinite(mean[q]['balanced_accuracy']) else mean[q]['balanced_accuracy'],lam_value(q)))
    finite=[key for key in mean if str(key).lower() not in ('inf','infinity')]
    finite_brier=min(finite,key=lambda q:(mean[q]['brier'],mean[q]['log_loss'])) if finite else brier
    return {'shrinkage_brier':lam_value(brier),'shrinkage_ba':lam_value(ba),'shrinkage_finite':lam_value(finite_brier)}

def resample_subject(X,y,rng):
    ix=[]
    for cls in (0,1):
        ids=np.flatnonzero(y==cls)
        if len(ids):ix.extend(rng.choice(ids,len(ids),replace=True).tolist())
    return np.asarray(X)[ix],np.asarray(y)[ix]

def fit_source_grouped(target,features,data,source_sids,selection,kgrid,cgrid,seed=42):
    assert_no_target_leak(target,source_sids)
    X,y,g=source_arrays(features,data,source_sids)
    return grouped_select(X,y,g,selection,kgrid,cgrid,seed)

def explanation_stability(a,b):
    cos=[];over=[];spear=[]
    for x,y in zip(a,b):
        den=np.linalg.norm(x)*np.linalg.norm(y);cos.append(float(np.dot(x,y)/den) if den else np.nan)
        over.append(len(set(np.argsort(-np.abs(x),kind='stable')[:5])&set(np.argsort(-np.abs(y),kind='stable')[:5]))/5)
        r=spearmanr(x,y).statistic;spear.append(float(r) if np.isfinite(r) else np.nan)
    return float(np.nanmean(cos)),float(np.nanmean(over)),float(np.nanmean(spear))

def bootstrap_fit(features,data,source_sids,target,selection,k,C,lambdas,rng):
    chunks=[];ys=[];gs=[]
    for s in source_sids:
        x,y=resample_subject(features[s],data[s]['y'],rng);chunks.append(x);ys.append(y);gs.extend([s]*len(y))
    X=np.concatenate(chunks);y=np.concatenate(ys);g=np.asarray(gs)
    fit=fixed_fit(X,y,g,C,k,selection)
    xc,yc=resample_subject(features[target][:20],data[target]['y'][:20],rng)
    mix=weighted_bundle(fit,X,y,xc,yc);targ=target_bundle(fit,xc,yc);inter=intercept_bundle(fit,xc,yc)
    bundles={'general':fit,'mix10':mix,'target_only':targ,'intercept_only':inter}
    for variant,lam in (lambdas or {}).items():bundles[variant]=shrink_bundle(fit,xc,yc,lam)
    return {'bundles':bundles,'background':X}

def run_full(args):
    from .report import write_report
    threadpool_limits(1);t0=time.time();stamp=time.strftime('%Y%m%d_%H%M%S');out=Path(args.out).resolve() if args.out else ROOT/'results'/stamp;out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=out/'run.log',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',encoding='utf8')
    data,refs,refpath,manifest=load_all();allsubs=sorted(data);chosen=allsubs[:args.subjects] if args.subjects is not None else allsubs
    scaleaudit,thresholdaudit=artifact_scale_audit({s:data[s] for s in chosen})
    if args.quick:chosen=chosen[:6]
    file_sigs={int(r.subject):str(r.sha256) for r in manifest.dropna(subset=['sha256']).itertuples()};sig={s:file_sigs[s] for s in chosen}
    manifest[manifest.subject.isin(chosen)].to_csv(out/'split_manifest.csv',index=False)
    scaleaudit.to_csv(out/'signal_scale_audit.csv',index=False);thresholdaudit.to_csv(out/'artifact_threshold_sweep.csv',index=False)
    Xraw={};rejlist=[];cache=ROOT/'cache'
    for w in [2,4,8,56]:
        stage_t=time.time()
        f,r=cached_features({s:data[s] for s in chosen},w,cache,sig);Xraw[w]=f;rejlist.append(r)
        logging.info('feature stage window=%ss elapsed_seconds=%.2f',w,time.time()-stage_t)
    rejdetail=pd.concat(rejlist,ignore_index=True);rejdetail=rejdetail[rejdetail.subject.isin(chosen)].drop_duplicates(['subject','presentation_trial','window_s'],keep='last')
    rejdetail.to_csv(out/'artifact_rejection_trials.csv',index=False)
    rejsummary=rejdetail.groupby(['subject','window_s']).agg(n_trials=('presentation_trial','count'),n_windows=('n_windows','sum'),n_rejected=('rejected','sum'),fallback_trials=('fallback_keep_all','sum')).reset_index()
    rejsummary['rejection_rate']=rejsummary.n_rejected/(rejsummary.n_windows.replace(0,np.nan));rejsummary.to_csv(out/'artifact_rejection.csv',index=False)
    Xnorm=norm_subject(Xraw[4],chosen)
    Xstatraw={}
    if not args.skip_exploratory:
        Xstatraw,_=cached_features({s:data[s] for s in chosen},4,cache,sig,stats=True)
        Xstat=norm_subject(Xstatraw,chosen)
    Xseq={};gru_targets=[];exploratory_failures=[]
    if not args.skip_exploratory:
        try:
            import torch
            from .exploratory import sequence_cache,normalize_sequences
            stage_t=time.time()
            Xseq=normalize_sequences(sequence_cache({s:data[s] for s in chosen},cache,sig),chosen)
            gru_targets=chosen[:args.gru_subjects] if args.gru_subjects is not None else chosen
            logging.info('GRU sequence cache stage elapsed_seconds=%.2f',time.time()-stage_t)
        except ImportError:
            exploratory_failures.append({'subject':'','step':'GRU','reason':'PyTorch unavailable; CPU GRU skipped'})
        except Exception as ex:
            logging.exception('Could not prepare GRU sequence cache');exploratory_failures.append({'subject':'','step':'GRU','reason':str(ex),'traceback':traceback.format_exc()})
    if all(s in Xnorm for s in chosen):
        if all(np.allclose(Xraw[4][s],Xnorm[s]) for s in chosen):raise AssertionError('L2 normalization is a no-op; expected subject-calibrated features to differ from L1')
        for s in chosen:
            if not np.allclose(Xnorm[s][:20].mean(0),0,atol=1e-7) or not np.allclose(Xnorm[s][:20].std(0),1,atol=1e-6):raise AssertionError(f'Subject {s}: L2 calibration window is not standardized')
    import sklearn,scipy,hashlib
    versions={'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'scipy':scipy.__version__}
    try:
        import torch;versions['torch']=torch.__version__
    except ImportError:versions['torch']=None
    config={'seed':42,'subjects':chosen,'quick':args.quick,'fs_hz':128,'n_jobs':1,'threadpool_limit':1,'calibration_trials':20,'test_trials':20,'presentation_split':[1,20,21,40],
            'bands_hz':{'theta':[4,8],'alpha':[8,13],'beta':[13,30],'gamma':[30,45]},'log_power_floor':1e-12,'welch':{'nperseg':256,'noverlap':128},
            'window_sensitivity_seconds':[2,4,8,'full'],'window_default_seconds':4,'window_overlap':.5,'window_start_s':5,'window_end_s':60,
            'artifact_rejection':{'rms_robust_z_max':ARTIFACT_RMS_Z_MAX,'peak_to_peak_max':ARTIFACT_P2P_MAX,'minimum_kept_fraction':.5,'units':'input signal units; DEAP EEG stored at the dataset-provided amplitude scale','scale_audit':'signal_scale_audit.csv','threshold_sweep':'artifact_threshold_sweep.csv'},
            'correlation_threshold':.90,'mi_n_neighbors':3,'mi_k_grid':[10,20,30,40,'all'],'C_grid':[.01,.1,1],
            'lambda_grid':[0,.01,.1,1,10,100,1000,'inf'],'lambda_selection':['shrinkage_brier: mean Brier, log-loss tie-break','shrinkage_ba: mean balanced accuracy, ties to larger lambda','shrinkage_finite: mean Brier/log-loss tie-break over finite lambda only'],'mix_calibration_weight_ratio':10,
            'bootstrap_refits':5 if args.quick else 20,'statistics_resamples':1000 if args.quick else 20000,'faithfulness_masks_per_trial':100,'channel_mask_probability':.25,'shuffled_attribution_repeats':10,'shuffled_calibration_repeats':10,'random_k_draws':10,'deletion_k':[0,1,3,5,10,20,32],
            'feature_cache':str(cache.relative_to(ROOT)),'reference_file':str(refpath.relative_to(ROOT)),'reference_sha256':hashlib.sha256(refpath.read_bytes()).hexdigest(),'dataset_sha256':file_sigs,'BA_noninferiority_margin':-.03,'margin_prechosen':True,'packages':versions,
            'gru':{'hidden_size':16,'dropout':.3,'weight_decay':.001,'early_stopping_patience':10,'ig_steps':24,'gru_subjects_requested':args.gru_subjects,'gru_subjects_run':gru_targets}}
    config['gru_subjects_requested']=args.gru_subjects;config['gru_subjects_run']=gru_targets
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    arms=['L0','L1','L2','L3','L4'];selmode={'L0':'corr','L1':'corr','L2':'corr','L3':'mi','L4':'mi'}
    loaded,done=load_fold_checkpoints(out) if args.resume else ({k:[] for k in CKPT_NAMES},set())
    allrows=loaded['allrows'];sweeps=loaded['sweeps'];lambdaselect=loaded['lambdaselect'];ctrl=loaded['ctrl'];stab=loaded['stab'];deletions=loaded['deletions'];shuffle=loaded['shuffle'];sens=loaded['sens'];attrib_subject=loaded['attrib_subject'];explore_rows=loaded['explore'];gru_deletions=loaded['gru_deletions'];gru_attributions=loaded['gru_attributions'];corr_example=loaded['corr_example'];shuffle_cal=loaded['shuffle_cal'];fails=loaded['fails']
    for oi,target in enumerate(chosen):
      if target in done:logging.info('resume: skipping completed subject %s',target);continue
      try:
        src=[s for s in chosen if s!=target];assert_no_target_leak(target,src);ysrc=np.concatenate([data[s]['y'] for s in src]);groups=np.repeat(src,40);ycal=data[target]['y'][:20];ytest=data[target]['y'][20:]
        majority=int(np.bincount(ysrc,minlength=2).argmax());mp=np.full(20,float(majority))
        byarm={'L0':Xraw[56],'L1':Xraw[4],'L2':Xnorm,'L3':Xnorm,'L4':Xnorm};orig={};targetfeatures={}
        for arm in arms:
          arm_t=time.time()
          features=byarm[arm];X,y,g=source_arrays(features,data,src);xcal=features[target][:20];xtest=features[target][20:];targetfeatures[arm]=(features,xcal,xtest)
          kgrid=['all'] if selmode[arm]=='corr' else [10,20,30,40,'all'];fit=grouped_select(X,y,g,selmode[arm],kgrid,[.01,.1,1],SEED)
          if arm=='L2' and not corr_example:
            before=np.nan_to_num(np.corrcoef(X,rowvar=False),nan=0,posinf=0,neginf=0);keep=fit['pipeline'].named_steps['correlation'].get_support();after=np.nan_to_num(np.corrcoef(X[:,keep],rowvar=False),nan=0,posinf=0,neginf=0)
            corr_example.extend([{'subject':target,'stage':'before','feature_i':i,'feature_j':j,'correlation':float(before[i,j])} for i in range(before.shape[0]) for j in range(before.shape[1])]);ids=np.flatnonzero(keep)
            corr_example.extend([{'subject':target,'stage':'after','feature_i':int(ids[i]),'feature_j':int(ids[j]),'correlation':float(after[i,j])} for i in range(after.shape[0]) for j in range(after.shape[1])])
          lambda_map={'shrinkage_brier':np.inf};lambda_cv={}
          if arm=='L4':
            lambda_map,lambda_cv=pseudo_lambda(features,data,src,'mi',fit['k'],fit['C'],kgrid,[.01,.1,1],basefit=fit)
            for variant,selected in lambda_map.items():
              scores={str(k):v for k,v in lambda_cv.items()}
              lambdaselect.append({'subject':target,'selection_variant':variant,'selected_lambda':str(selected),'pseudo_subjects':len(src),
                  'selection_brier':json.dumps({k:float(v['brier']) for k,v in scores.items()}),
                  'selection_log_loss':json.dumps({k:float(v['log_loss']) for k,v in scores.items()}),
                  'selection_balanced_accuracy':json.dumps({k:(float(v['balanced_accuracy']) if np.isfinite(v['balanced_accuracy']) else None) for k,v in scores.items()})})
          else:
            lambda_map={'shrinkage_brier':np.inf}
          mix=weighted_bundle(fit,X,y,xcal,ycal);targ=target_bundle(fit,xcal,ycal);inter=intercept_bundle(fit,xcal,ycal)
          models={'general':fit,'mix10':mix,'intercept_only':inter}
          for variant,selected in lambda_map.items():models[variant]=shrink_bundle(fit,xcal,ycal,selected)
          if targ is not None:models['target_only']=targ
          orig[arm]=models
          bg=X
          for name,bundle in models.items():
            p=predict_prob(bundle,xtest);m=metrics(ytest,p);faith,attr,adderr=faithfulness(bundle,xtest,bg,100 if not args.quick else 100,SEED+target);shuf,shufreps=shuffled_faithfulness(bundle,xtest,bg,10,100,SEED+target)
            allrows.append({'subject':target,'arm':arm,'model':name,**m,'compact_faithfulness':faith,'shap_additivity_max_error':adderr,'selected_k':fit['k'],'C':fit['C'],'lambda':str(lambda_map[name]) if name in lambda_map else '', 'lambda_variant':name if name.startswith('shrinkage_') else ''})
            shuffle.extend({'subject':target,'arm':arm,'model':name,'repeat':i,'score':v} for i,v in enumerate(shufreps))
            dr,auc=deletion(bundle,xtest,bg,attr,target,name,SEED+target);deletions.extend(dr)
            allrows[-1]['deletion_auc']=auc
            for c in range(32):attrib_subject.append({'subject':target,'arm':arm,'model':name,'channel':c+1,'mean_abs_attribution':float(np.mean(np.abs(attr[:,c])))})
            if name=='shrinkage_brier' and arm=='L4':
              for l in LAMBDAS:
                sb=shrink_bundle(fit,xcal,ycal,l);ps=predict_prob(sb,xtest);mf=metrics(ytest,ps);ff,aa,_=faithfulness(sb,xtest,bg,100,SEED+target)
                sweeps.append({'subject':target,'lambda':str(l),'selection':'descriptive, not used for selection','balanced_accuracy':mf['balanced_accuracy'],'brier':mf['brier'],'compact_faithfulness':ff})
          allrows.append({'subject':target,'arm':arm,'model':'majority',**metrics(ytest,mp),'compact_faithfulness':np.nan,'shap_additivity_max_error':np.nan,'selected_k':'','C':np.nan,'lambda':''})
          for repeat in range(10):
            sy=np.random.default_rng(SEED+target*100+repeat).permutation(ycal);cb=weighted_bundle(fit,X,y,xcal,sy);cp=predict_prob(cb,xtest);shuffle_cal.append({'subject':target,'arm':arm,'repeat':repeat,**metrics(ytest,cp)})
          if arm=='L3':
            mf,ma,_=faithfulness(fit,xtest,X,100,SEED+target)
            ctrl.append({'subject':target,'arm':arm,'method':'MI','draw':0,'k':fit['k'],'balanced_accuracy':metrics(ytest,predict_prob(fit,xtest))['balanced_accuracy'],'compact_faithfulness':mf})
            kk=fit['k']
            for seed in range(10):
              rf=grouped_select(X,y,g,'random',[kk],[fit['C']],SEED+seed); rp=predict_prob(rf,xtest);rfai,_,_=faithfulness(rf,xtest,X,100,SEED+target+seed)
              ctrl.append({'subject':target,'arm':'L3','method':'random-k','draw':seed,'k':kk,'balanced_accuracy':metrics(ytest,rp)['balanced_accuracy'],'compact_faithfulness':rfai})
            corrfit=grouped_select(X,y,g,'corr',['all'],[.01,.1,1],SEED);cfa,_,_=faithfulness(corrfit,xtest,X,100,SEED+target)
            ctrl.append({'subject':target,'arm':'L3','method':'corr-only','draw':0,'k':'all','balanced_accuracy':metrics(ytest,predict_prob(corrfit,xtest))['balanced_accuracy'],'compact_faithfulness':cfa})
          if arm in ('L0','L1','L2','L3','L4'):
            # paired bootstrap refits; refit source general model and each calibration model.
            st_names=list(models);refitattr={name:[] for name in st_names if name in models}; originals={name:explain(models[name],xtest,bg)[0] for name in refitattr}
            nboot=5 if args.quick else 20
            rng=np.random.default_rng(SEED+target)
            for bi in range(nboot):
              boot=bootstrap_fit(features,data,src,target,selmode[arm],fit['k'],fit['C'],lambda_map,rng)
              for name,b in boot['bundles'].items():
                if name not in refitattr:continue
                if b is None:continue
                refitattr[name].append(explain(b,xtest,boot['background'])[0])
            for name,attrs in refitattr.items():
              if attrs:
                per=[explanation_stability(originals[name],a) for a in attrs];v=np.nanmean(per,axis=0)
                stab.append({'subject':target,'arm':arm,'model':name,'cosine':v[0],'top5_overlap':v[1],'spearman':v[2],'n_refits':len(attrs)})
          if arm in ('L0','L1'):
            pass
          logging.info('subject=%s arm=%s completed elapsed_seconds=%.2f',target,arm,time.time()-arm_t)
        # Window-length sensitivity uses correlation-only general classifier on each setting.
        for win,features in [(2,Xraw[2]),(4,Xraw[4]),(8,Xraw[8]),(56,Xraw[56])]:
          X,y,g=source_arrays(features,data,src);fitw=grouped_select(X,y,g,'corr',['all'],[.01,.1,1]);xt=features[target][20:];p=predict_prob(fitw,xt);sfaith,_,_=faithfulness(fitw,xt,X,100,SEED+target);sens.append({'subject':target,'window_s':'full' if win==56 else win,'balanced_accuracy':metrics(ytest,p)['balanced_accuracy'],'compact_faithfulness':sfaith})
        if Xstatraw:
          try:
            X,y,g=source_arrays(Xstat,data,src);xcal=Xstat[target][:20];xt=Xstat[target][20:];kg=[10,20,30,40,'all'];efit=grouped_select(X,y,g,'mi',kg,[.01,.1,1]);elams,_=pseudo_lambda(Xstat,data,src,'mi',efit['k'],efit['C'],kg,[.01,.1,1],basefit=efit);elam=elams['shrinkage_brier']
            for ename,eb in [('general',efit),('mix10',weighted_bundle(efit,X,y,xcal,ycal)),('shrinkage',shrink_bundle(efit,xcal,ycal,elam))]:
              ep=predict_prob(eb,xt);explore_rows.append({'subject':target,'arm':'EXPLORATORY - not part of the main claim','features':'mean_std_slope','model':ename,'lambda':str(elam) if ename=='shrinkage' else '',**metrics(ytest,ep)})
          except Exception as ex:exploratory_failures.append({'subject':target,'step':'4a','reason':str(ex),'traceback':traceback.format_exc()})
        if target in gru_targets:
          try:
            from .exploratory import run_gru_fold
            grow,gdel,gia=run_gru_fold(Xseq,data,src,target,SEED+target);explore_rows.extend(grow);gru_deletions.extend(gdel);gru_attributions.extend(gia)
          except Exception as ex:exploratory_failures.append({'subject':target,'step':'GRU','reason':str(ex),'traceback':traceback.format_exc()})
        logging.info('completed subject %s',target)
        save_fold_checkpoint(out,target,{'allrows':allrows,'sweeps':sweeps,'lambdaselect':lambdaselect,'ctrl':ctrl,'stab':stab,'deletions':deletions,'shuffle':shuffle,'sens':sens,'attrib_subject':attrib_subject,'explore':explore_rows,'gru_deletions':gru_deletions,'gru_attributions':gru_attributions,'corr_example':corr_example,'shuffle_cal':shuffle_cal,'fails':fails})
      except Exception as e:
        logging.exception('subject %s failed',target);fails.append({'subject':target,'error':str(e),'traceback':traceback.format_exc()});pd.DataFrame([fails[-1]]).to_csv(Path(out)/'checkpoints'/f'subject_{target:02d}_fails.csv',index=False)
    metricdf=pd.DataFrame(allrows);stdf=pd.DataFrame(stab);controldf=pd.DataFrame(ctrl)
    metricdf.to_csv(out/'subject_metrics.csv',index=False);pd.DataFrame(fails,columns=['subject','error','traceback']).to_csv(out/'failures.csv',index=False)
    pd.DataFrame(sweeps).to_csv(out/'lambda_sweep.csv',index=False);pd.DataFrame(lambdaselect).to_csv(out/'lambda_selection.csv',index=False);pd.DataFrame(stab).to_csv(out/'stability_refits.csv',index=False);pd.DataFrame(deletions).to_csv(out/'deletion_curves.csv',index=False);pd.DataFrame(shuffle).to_csv(out/'shuffled_controls.csv',index=False);pd.DataFrame(sens).to_csv(out/'window_sensitivity.csv',index=False);pd.DataFrame(ctrl).to_csv(out/'feature_selection_controls.csv',index=False);pd.DataFrame(attrib_subject).to_csv(out/'channel_attributions.csv',index=False);pd.DataFrame(explore_rows).to_csv(out/'exploratory_metrics.csv',index=False)
    from .stats import build_paired_summary,predictive_preservation
    paired=build_paired_summary(metricdf,stdf,controldf,1000 if args.quick else 20000);paired.to_csv(out/'paired_summary.csv',index=False)
    pd.DataFrame([predictive_preservation(metricdf,n_resamples=1000 if args.quick else 20000)]).to_csv(out/'predictive_preservation.csv',index=False)
    pd.DataFrame(exploratory_failures).to_csv(out/'exploratory_failures.csv',index=False)
    pd.DataFrame(gru_deletions).to_csv(out/'exploratory_deletion_curves.csv',index=False)
    pd.DataFrame(gru_attributions).to_csv(out/'exploratory_ig_attributions.csv',index=False)
    pd.DataFrame(corr_example).to_csv(out/'correlation_example.csv',index=False)
    pd.DataFrame(shuffle_cal).to_csv(out/'shuffled_calibration.csv',index=False)
    if not metricdf.empty:
        l0g=metricdf[(metricdf.arm=='L0')&(metricdf.model=='general')];l0m=metricdf[(metricdf.arm=='L0')&(metricdf.model=='mix10')]
        base={'general_ba':.49,'mix10_ba':.56,'general_faithfulness':.68,'mix10_faithfulness':.60}
        observed={'general_ba':l0g.balanced_accuracy.mean(),'mix10_ba':l0m.balanced_accuracy.mean(),'general_faithfulness':l0g.compact_faithfulness.mean(),'mix10_faithfulness':l0m.compact_faithfulness.mean()}
        for key,val in observed.items():
            if np.isfinite(val) and abs(val-base[key])>.05:logging.warning('L0 notebook replication check differs from reference expectation by more than 0.05: %s=%.4f expected approximately %.2f',key,val,base[key])
    from .postprocess import generate
    generate(out,args.quick)
    if not args.quick:
        from .report import write_report
        write_report(out,config,'EXPLORATORY - not part of the main claim. Step 4a/GRU status recorded in exploratory_metrics.csv.')
    logging.info('pipeline finished; completed=%d failed=%d elapsed_seconds=%.1f',len(chosen)-len(fails),len(fails),time.time()-t0)
    if not args.quick:
        latest=ROOT/'latest';latest.mkdir(exist_ok=True)
        keep={f.name for f in out.iterdir() if f.is_file()}
        if latest.resolve().parent==ROOT.resolve():
            for old in latest.iterdir():
                if old.is_file() and old.name not in keep:old.unlink()
        for f in out.iterdir():
            if f.is_file(): (latest/f.name).write_bytes(f.read_bytes())
    print(f'Output: {out}; folds completed={len(chosen)-len(fails)}/{len(chosen)}; failed={len(fails)}; elapsed={time.time()-t0:.1f}s')
    if fails:print(pd.DataFrame(fails)[['subject','error']].to_string(index=False))
