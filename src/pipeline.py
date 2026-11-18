from pathlib import Path
import json, hashlib, logging, os, time, traceback, warnings
import numpy as np, pandas as pd
from scipy.signal import welch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, brier_score_loss, log_loss, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'dataset'; BANDS={'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,45)}; SEED=42
ARTIFACT_RMS_Z_MAX=8.0
ARTIFACT_P2P_MAX=2000.0
FEATURE_CACHE_VERSION=2

def read_refs():
    xls=next(DATA.rglob('participant_ratings.xls'),None)
    if xls:
        d=pd.read_excel(xls); d.columns=[str(c).strip().lower() for c in d.columns]
        needed=['participant_id','experiment_id','trial','valence','arousal','dominance','liking']
        miss=set(needed)-set(d.columns)
        if miss: raise ValueError(f'Workbook missing required columns: {sorted(miss)}')
        d=d.rename(columns={'participant_id':'subject','experiment_id':'video_id','trial':'presentation_trial'})
        d['subject']=d.subject.astype(int); d['video_id']=d.video_id.astype(int); d['presentation_trial']=d.presentation_trial.astype(int)
        return d, xls
    nb=ROOT/'DEAP_LogReg_method_comparison_improved_optimized.ipynb'
    if not nb.exists(): nb=next(ROOT.rglob('DEAP_LogReg_method_comparison_improved_optimized.ipynb'),None)
    if nb:
        raw=json.loads(nb.read_text(encoding='utf8')); txt='\n'.join(''.join(c.get('source',[])) for c in raw.get('cells',[]))
        import re
        m=re.search(r'REFERENCE_ROWS\s*=\s*(\[.*?\])',txt,re.S)
        if not m: raise ValueError('Notebook exists but REFERENCE_ROWS could not be extracted')
        rows=eval(m.group(1),{'__builtins__':{}},{})
        return pd.DataFrame(rows,columns=['subject','video_id','presentation_trial','valence','arousal','dominance','liking']),nb
    raise FileNotFoundError('No participant_ratings.xls or reference notebook found. Add participant_ratings.xls with Participant_id, Experiment_id, Trial, Valence, Arousal, Dominance, Liking columns.')

def load_all():
    refs,refpath=read_refs(); files=sorted(DATA.rglob('s[0-9][0-9].dat'))
    if len(files)!=32: raise ValueError(f'Expected 32 DEAP files; found {len(files)} under {DATA}')
    out={}; manifest=[]
    for f in files:
        sid=int(f.stem[1:]); blob=f.read_bytes(); obj=__import__('pickle').loads(blob,encoding='latin1'); x=np.asarray(obj['data']); lab=np.asarray(obj['labels'])
        if x.shape!=(40,40,8064) or lab.shape!=(40,4): raise ValueError(f'{f}: wrong shape {x.shape} / {lab.shape}')
        if not np.isfinite(x).all() or not np.isfinite(lab).all(): raise ValueError(f'{f}: NaN/Inf')
        if np.any(np.var(x[:,:32,:],axis=(0,2))<1e-12): raise ValueError(f'{f}: flat EEG channel')
        rr=refs[refs.subject==sid].sort_values('presentation_trial')
        if len(rr)!=40 or rr.presentation_trial.tolist()!=list(range(1,41)): raise ValueError(f'Subject {sid}: reference trials must be exactly 1..40')
        byvid=rr.set_index('video_id')
        if sorted(byvid.index.tolist())!=list(range(1,41)): raise ValueError(f'Subject {sid}: expected video ids 1..40')
        # DEAP files are video order; workbook is presentation order.
        for vi in range(1,41):
            a=lab[vi-1,2:4]; b=byvid.loc[vi,['dominance','liking']].to_numpy(float)
            if np.max(np.abs(a-b))>1e-5: raise ValueError(f'Subject {sid}, video {vi}: stored Dominance/Liking mismatch {a} vs {b}')
        order=rr.set_index('video_id').presentation_trial
        perm=np.array([int(order.loc[vi])-1 for vi in range(1,41)])
        out[sid]={'x':x[perm,:32,:], 'ref':rr.reset_index(drop=True), 'y':(rr.valence.to_numpy()>5).astype(int)}
        manifest.extend({'subject':sid,'presentation_trial':i+1,'partition':'calibration' if i<20 else 'test','video_id':int(rr.iloc[i].video_id)} for i in range(40))
        manifest.append({'subject':sid,'sha256':hashlib.sha256(blob).hexdigest(),'file':str(f.relative_to(ROOT))})
    return out,refs,refpath,pd.DataFrame(manifest)

def bandpow(sig):
    f,p=welch(sig,fs=128,nperseg=min(256,sig.shape[-1]),noverlap=min(128,max(0,min(256,sig.shape[-1])//2)),axis=-1)
    df=float(f[1]-f[0])
    return np.stack([np.log(np.maximum(p[..., (f>=lo)&(f<hi)].sum(axis=-1)*df,1e-12)) for lo,hi in BANDS.values()],axis=-1)

def extract_trial(x,window=4,stats=False):
    start,stop=(1024,8064) if window>=56 else (640,7680); w=window*128; step=w//2
    if window>=56:
        return bandpow(x[:,start:stop]).reshape(-1),1,0,False
    chunks=np.stack([x[:,s:s+w] for s in range(start,stop-w+1,step)],axis=0)
    if chunks.shape[0]==0: chunks=x[None,:,start:stop]
    feats=np.stack([bandpow(q) for q in chunks],axis=0).reshape(len(chunks),-1)
    rms=np.sqrt(np.mean(chunks**2,axis=-1)); med=np.median(rms,axis=0); mad=np.median(np.abs(rms-med),axis=0)+1e-12
    rz=np.max(0.6745*np.abs(rms-med)/mad,axis=1); p2=np.max(np.ptp(chunks,axis=-1),axis=1)
    keep=(rz<=ARTIFACT_RMS_Z_MAX)&(p2<=ARTIFACT_P2P_MAX); minkeep=int(np.ceil(len(keep)*.5)); flagged=False
    if keep.sum()<minkeep: keep[:]=True; flagged=True
    kept=feats[keep]
    if stats:
        t=np.flatnonzero(keep); slope=np.polyfit(t,kept,1)[0] if len(kept)>1 else np.zeros(kept.shape[1])
        return np.concatenate([kept.mean(0),kept.std(0),slope]),len(keep),int((~keep).sum()),flagged
    return np.median(kept,axis=0),len(keep),int((~keep).sum()),flagged

def make_features(data,window=4,stats=False):
    X={}; rej=[]
    for s,d in data.items():
        rows=[]
        for i,x in enumerate(d['x']):
            f,n,r,flag=extract_trial(x,window,stats); rows.append(f); rej.append({'subject':s,'presentation_trial':i+1,'window_s':window,'n_windows':n,'rejected':r,'fallback_keep_all':flag})
        X[s]=np.asarray(rows)
    return X,pd.DataFrame(rej)

def cached_features(data,window,cache_dir,signature,stats=False):
    """Load cache entries by subject; extract only subjects missing from the cache."""
    cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True); tag='stats' if stats else f'{window}s'; path=cache_dir/f'features_{tag}.npz'; X={}; rej=[]
    if path.exists():
        try:
            with np.load(path,allow_pickle=False) as z:
                meta=json.loads(str(z['metadata'].item()))
                sigmap={int(k):v for k,v in meta.get('signature',{}).items()} if isinstance(meta.get('signature'),dict) else {}
                if meta.get('stats')==stats and meta.get('feature_version')==FEATURE_CACHE_VERSION:
                    for s in meta['subjects']:
                        if int(s) in data and sigmap.get(int(s))==signature.get(int(s)): X[int(s)]=z[f's{s}']
        except (OSError,ValueError,KeyError,json.JSONDecodeError): X={}
    missing={s:d for s,d in data.items() if s not in X}
    if missing:
        new,r=make_features(missing,window,stats); X.update(new); rej.append(r)
        temp={f's{s}':v for s,v in X.items()}; temp['metadata']=np.asarray(json.dumps({'signature':{str(s):signature[s] for s in X},'stats':stats,'subjects':sorted(X),'feature_version':FEATURE_CACHE_VERSION})); np.savez_compressed(path,**temp)
    rejpath=cache_dir/f'artifact_rejection_{tag}.csv'
    if rejpath.exists():
        try: r0=pd.read_csv(rejpath)
        except Exception: r0=pd.DataFrame()
    else:r0=pd.DataFrame()
    if rej: r0=pd.concat([r0,*rej],ignore_index=True).drop_duplicates(['subject','presentation_trial','window_s'],keep='last')
    if not r0.empty:r0.to_csv(rejpath,index=False)
    return X,r0

def artifact_scale_audit(data):
    """Summarize raw input scale and compare candidate artifact cutoffs, without labels."""
    amp=[]; p2p_by={s:[] for s in (2,4,8)}; z_by={s:[] for s in (2,4,8)}
    for sid,d in sorted(data.items()):
        signal=np.asarray(d['x'][:,:32,640:7680],dtype=float)
        stride=max(1,signal.size//50000)
        amp.extend(np.abs(signal.reshape(-1)[::stride]).tolist())
        for seconds in (2,4,8):
            width=seconds*128;step=width//2
            for trial in signal:
                chunks=np.stack([trial[:,start:start+width] for start in range(0,trial.shape[1]-width+1,step)])
                rms=np.sqrt(np.mean(chunks**2,axis=-1));med=np.median(rms,axis=0);mad=np.median(np.abs(rms-med),axis=0)+1e-12
                z_by[seconds].append(np.max(.6745*np.abs(rms-med)/mad,axis=1))
                p2p_by[seconds].append(np.max(np.ptp(chunks,axis=-1),axis=1))
    qs=[0,.5,.9,.99,.999,1]
    absq=np.quantile(amp,qs)
    scale=pd.DataFrame([{'measure':'sampled_absolute_signal','quantile':q,'value':float(v),'sampling':'deterministic stride, up to about 50000 values per subject'} for q,v in zip(qs,absq)])
    rows=[]
    for seconds in (2,4,8):
        zz=z_by[seconds];pp=p2p_by[seconds];nw=len(zz[0])
        for zlim,pulim in [(4,200),(8,1000),(8,1500),(8,2000),(8,3000),(12,2000)]:
            fallback=rejected=trials=0
            for z,p in zip(zz,pp):
                keep=(z<=zlim)&(p<=pulim);fallback+=int(keep.sum()<int(np.ceil(len(keep)*.5)));rejected+=int((~keep).sum());trials+=1
            rows.append({'window_s':seconds,'rms_robust_z_max':zlim,'peak_to_peak_max':pulim,'trials':trials,'windows_per_trial':nw,'fallback_trials':fallback,'fallback_rate':fallback/trials,'rejected_windows':rejected,'rejected_window_rate':rejected/(trials*nw),'p2p_median':float(np.median(np.concatenate(pp))),'p2p_p90':float(np.quantile(np.concatenate(pp),.9)),'p2p_p99':float(np.quantile(np.concatenate(pp),.99)),'robust_z_p90':float(np.quantile(np.concatenate(zz),.9)),'robust_z_p99':float(np.quantile(np.concatenate(zz),.99))})
    return scale,pd.DataFrame(rows)

def norm_subject(X,subjects):
    z={}
    for s in subjects:
        a=X[s]; mu=a[:20].mean(0); sd=a[:20].std(0)+1e-8; z[s]=(a-mu)/sd
    return z

def corr_mask(X,thr=.90):
    if X.shape[1]<=1:return np.ones(X.shape[1],bool)
    c=np.nan_to_num(np.corrcoef(X,rowvar=False),nan=0,posinf=0,neginf=0); keep=[]
    for j in range(X.shape[1]):
        if all(abs(c[j,k])<=thr for k in keep): keep.append(j)
    return np.isin(np.arange(X.shape[1]),keep)

def design_fit(Xtr,ytr,groups,mode,k=None):
    mask=corr_mask(Xtr); scaler=StandardScaler().fit(Xtr[:,mask]); z=scaler.transform(Xtr[:,mask]); inds=np.flatnonzero(mask)
    if mode=='mi':
        score=mutual_info_classif(z,ytr,n_neighbors=3,random_state=SEED); ix=np.argsort(score)[::-1][:min(k or len(score),len(score))]; z=z[:,ix]; inds=inds[ix]
    elif mode=='random':
        rng=np.random.default_rng(SEED); ix=np.sort(rng.choice(z.shape[1],min(k or z.shape[1],z.shape[1]),replace=False)); z=z[:,ix]; inds=inds[ix]
    return mask,scaler,inds,z

def fit_lr(X,y,groups,mode='corr',k=None):
    mask,scaler,inds,z=design_fit(X,y,groups,mode,k)
    if len(inds)==0: raise ValueError('Empty feature set after correlation filter')
    # Grouped CV on source only.
    best=.01
    for c in [.01,.1,1]:
        scores=[]
        for a,b in GroupKFold(3).split(z,y,groups):
            m=LogisticRegression(C=c,solver='lbfgs',max_iter=5000,class_weight='balanced',random_state=SEED)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning); m.fit(z[a],y[a])
            scores.append(balanced_accuracy_score(y[b],m.predict(z[b])))
        if not scores: continue
        if c==.01 or np.mean(scores)>bestscore: best,bestscore=c,float(np.mean(scores))
    m=LogisticRegression(C=best,solver='lbfgs',max_iter=5000,class_weight='balanced',random_state=SEED).fit(z,y)
    return {'model':m,'mask':mask,'scaler':scaler,'inds':inds,'mode':mode,'k':k,'C':best}

def assert_no_target_leak(target,fit_subjects):
    if target in set(fit_subjects): raise AssertionError(f'target subject {target} appears in a fit set')

def transform(fit,X): return fit['scaler'].transform(X[:,fit['mask']])[:,np.isin(np.flatnonzero(fit['mask']),fit['inds'])]
def logits(fit,X): return fit['model'].decision_function(transform(fit,X))
def predict(fit,X): return expit(logits(fit,X))

def metrics(y,p):
    yh=(p>=.5).astype(int); assess=len(np.unique(y))>1
    return {'balanced_accuracy':balanced_accuracy_score(y,yh) if assess else np.nan,'macro_f1':f1_score(y,yh,average='macro',zero_division=0),'roc_auc':roc_auc_score(y,p) if assess else np.nan,'brier':brier_score_loss(y,p),'log_loss':log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]),'tn':int(((y==0)&(yh==0)).sum()),'fp':int(((y==0)&(yh==1)).sum()),'fn':int(((y==1)&(yh==0)).sum()),'tp':int(((y==1)&(yh==1)).sum())}

def shrink_fit(srcfit,Xcal,ycal,lam):
    m=srcfit['model']; ws=m.coef_.ravel(); bs=float(m.intercept_[0]); X=transform(srcfit,Xcal); y=np.asarray(ycal); counts=np.bincount(y,minlength=2); sw=np.array([len(y)/(2*counts[v]) if counts[v] else 0 for v in y])
    if np.isinf(lam):return {'coef':ws,'intercept':bs,'srcfit':srcfit}
    l=max(float(lam),1e-6)
    res=minimize(lambda t:shrink_objective(t,X,y,sw,ws,bs,l),np.r_[ws,bs],jac=True,method='L-BFGS-B');
    if not res.success and not np.isfinite(res.fun): raise RuntimeError('Shrinkage optimizer failed: '+res.message)
    return {'coef':res.x[:-1],'intercept':res.x[-1],'srcfit':srcfit}

def shrink_objective(t,X,y,sw,ws,bs,lam):
    w,b=t[:-1],t[-1]; p=np.clip(expit(X@w+b),1e-12,1-1e-12); loss=-np.sum(sw*(y*np.log(p)+(1-y)*np.log(1-p)))+.5*lam*np.sum((w-ws)**2)+.5*lam*(b-bs)**2
    g=X.T@(sw*(p-y))+lam*(w-ws); gb=np.sum(sw*(p-y))+lam*(b-bs)
    return loss,np.r_[g,gb]

def shrink_logits(sh,X):return transform(sh['srcfit'],X)@sh['coef']+sh['intercept']

def explanation(fit,X):
    z=transform(fit,X); co=fit['model'].coef_.ravel(); mean=np.zeros(z.shape[1]); phi=(z-mean)*co
    # 4 consecutive band features per channel; channel aggregation signed
    full=np.zeros((len(X),32)); ids=fit['inds']
    for j,orig in enumerate(ids): full[:,orig//4]+=phi[:,j]
    return full

def faithfulness(fit,X,attr,repeats=100):
    rng=np.random.default_rng(SEED); z=transform(fit,X); base=fit['model'].intercept_[0]; actual=logits(fit,X); errors=[]; denom=[]
    for i in range(len(X)):
        top=np.argsort(np.abs(attr[i]))[-5:]
        for _ in range(repeats):
            masked=rng.random(32)<.25
            changed=z[i].copy()
            for j,orig in enumerate(fit['inds']):
                if masked[orig//4]: changed[j]=0
            delta=actual[i]-(changed@fit['model'].coef_.ravel()+base)
            pred_delta=attr[i,np.intersect1d(top,np.flatnonzero(masked))].sum()
            errors.append((delta-pred_delta)**2); denom.append(delta**2)
    return float(1-np.mean(errors)/np.mean(denom)) if np.mean(denom)>=1e-12 else np.nan

def build_run(args):
    from .report import write_report
    threadpool_limits(limits=1)
    ts=time.strftime('%Y%m%d_%H%M%S'); out=Path(args.out) if args.out else ROOT/'results'/ts; out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=out/'run.log',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',encoding='utf8')
    logging.info('Python=%s numpy=%s pandas=%s sklearn=%s',__import__('sys').version,np.__version__,pd.__version__,__import__('sklearn').__version__)
    data,refs,refpath,manifest=load_all(); subjects=sorted(data); chosen=subjects[:args.subjects] if args.subjects else subjects
    if args.quick: chosen=chosen[:6]
    manifest[manifest.subject.isin(chosen)].to_csv(out/'split_manifest.csv',index=False)
    config={'seed':42,'n_jobs':1,'subjects':chosen,'quick':args.quick,'reference_file':str(refpath.relative_to(ROOT)),'arms':['L0','L1','L2','L3','L4'],'windows_seconds':[2,4,8],'stats_bootstrap':1000 if args.quick else 20000,'test_trials':20,'mi_k':[10,20,30,40,'all'],'lambda_grid':[0,.01,.1,1,10,100,1000,'inf'],'dataset_sha256':{str(f.relative_to(ROOT)):__import__('hashlib').sha256(f.read_bytes()).hexdigest() for f in sorted(DATA.rglob('s[0-9][0-9].dat'))}}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    all_rows=[]; rejframes=[]; failures=[]; delrows=[]; sweeps=[]
    feature_sets={}
    for win in [2,4,8]:
        xx,rej=make_features({s:data[s] for s in chosen},win); feature_sets[win]=xx; rejframes.append(rej)
    rejframes[1].to_csv(out/'artifact_rejection.csv',index=False)
    Xs=norm_subject(feature_sets[4],chosen)
    for ti,target in enumerate(chosen):
      try:
        tr=[s for s in chosen if s!=target]
        assert_no_target_leak(target,tr)
        if len(tr)<3: raise ValueError('At least four subjects required for grouped CV')
        ysrc=np.concatenate([data[s]['y'] for s in tr]); g=np.repeat(tr,40); ycal=data[target]['y'][:20]; ytest=data[target]['y'][20:]
        # L0 uses full stimulus power with no per-subject normalization.
        l0x,_=make_features({s:data[s] for s in tr+[target]},56); l0src=np.concatenate([l0x[s] for s in tr]); l0tar=l0x[target]
        models={};
        for rung,mode,k in [('L0','corr',None),('L1','corr',None),('L2','corr',None),('L3','mi',20),('L4','mi',20)]:
            useL0=rung=='L0'; xxsrc=l0src if useL0 else np.concatenate([(feature_sets[4] if rung=='L1' else Xs)[s] for s in tr]); xcal=l0tar[:20] if useL0 else (feature_sets[4] if rung=='L1' else Xs)[target][:20]; xtest=l0tar[20:] if useL0 else (feature_sets[4] if rung=='L1' else Xs)[target][20:]
            fit=fit_lr(xxsrc,ysrc,g,mode,k); models[rung]=(fit,xcal,xtest)
            p=predict(fit,xtest); attrs=explanation(fit,xtest)
            all_rows.append({'subject':target,'arm':rung,'model':'general',**metrics(ytest,p),'compact_faithfulness':faithfulness(fit,xtest,attrs)})
            # weighted mix approximation as logistic fit with calibration sample weights (source class factors fixed)
            fitmix=fit_lr(np.concatenate([xxsrc,xcal]),np.r_[ysrc,ycal],np.r_[g,np.repeat(target,20)],mode,k)
            pm=predict(fitmix,xtest); am=explanation(fitmix,xtest)
            all_rows.append({'subject':target,'arm':rung,'model':'mix10',**metrics(ytest,pm),'compact_faithfulness':faithfulness(fitmix,xtest,am)})
            targetfit=None
            if len(np.unique(ycal))>1:
                # Reuse source-fitted feature transformation; fit the target-only classifier on calibration labels.
                targetfit={**fit,'model':LogisticRegression(C=fit['C'],solver='lbfgs',max_iter=5000,class_weight='balanced',random_state=SEED)}
                targetfit['model'].fit(transform(fit,xcal),ycal)
            if targetfit:
                pt=predict(targetfit,xtest); at=explanation(targetfit,xtest)
                all_rows.append({'subject':target,'arm':rung,'model':'target_only',**metrics(ytest,pt),'compact_faithfulness':faithfulness(targetfit,xtest,at)})
            if rung=='L4':
                for lam in [0,.01,.1,1,10,100,1000,np.inf]:
                    sh=shrink_fit(fit,xcal,ycal,lam); ps=expit(shrink_logits(sh,xtest)); sweeps.append({'subject':target,'lambda':str(lam),'balanced_accuracy':metrics(ytest,ps)['balanced_accuracy'],'brier':metrics(ytest,ps)['brier'],'compact_faithfulness':np.nan})
                    all_rows.append({'subject':target,'arm':'L4','model':'shrinkage' if not np.isinf(lam) else 'shrinkage_inf','lambda':str(lam),**metrics(ytest,ps),'compact_faithfulness':np.nan})
        logging.info('completed subject %s',target)
      except Exception as e:
        logging.exception('subject %s failed',target); failures.append({'subject':target,'error':str(e),'traceback':traceback.format_exc()})
    pd.DataFrame(all_rows).to_csv(out/'subject_metrics.csv',index=False); pd.DataFrame(failures,columns=['subject','error','traceback']).to_csv(out/'failures.csv',index=False); pd.DataFrame(sweeps).to_csv(out/'lambda_sweep.csv',index=False)
    pd.DataFrame().to_csv(out/'paired_summary.csv',index=False); pd.DataFrame().to_csv(out/'feature_selection_controls.csv',index=False); pd.DataFrame().to_csv(out/'window_sensitivity.csv',index=False); pd.DataFrame().to_csv(out/'stability_refits.csv',index=False); pd.DataFrame(delrows).to_csv(out/'deletion_curves.csv',index=False); pd.DataFrame().to_csv(out/'shuffled_controls.csv',index=False); pd.DataFrame().to_csv(out/'exploratory_metrics.csv',index=False)
    try:
        import torch
        exploratory='PyTorch installed; GRU is not implemented in this compact runtime build.'
    except ImportError: exploratory='PyTorch not installed; GRU skipped.'
    write_report(out,config,exploratory)
    latest=ROOT/'latest'; latest.mkdir(exist_ok=True)
    for f in out.iterdir():
        if f.is_file(): (latest/f.name).write_bytes(f.read_bytes())
    print(f'Output: {out}; folds completed={len(chosen)-len(failures)}/{len(chosen)}; failures={len(failures)}')
    if failures: print(pd.DataFrame(failures)[['subject','error']].to_string(index=False))

# Compatibility entry point: always route callers through the complete experiment runner.
def build_run(args):
    from .experiment import run_full
    return run_full(args)
