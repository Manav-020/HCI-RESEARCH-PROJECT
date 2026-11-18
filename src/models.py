"""Leakage-aware grouped model selection and attribution helpers."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

class CorrelationFilter(BaseEstimator,TransformerMixin):
    def __init__(self,threshold=.90): self.threshold=threshold
    def fit(self,X,y=None):
        X=np.asarray(X);self.active_=np.flatnonzero(np.var(X,axis=0)>1e-12)
        if not len(self.active_):raise ValueError('No variable features for correlation filtering')
        c=np.nan_to_num(np.corrcoef(X[:,self.active_],rowvar=False),nan=0,posinf=0,neginf=0); keep=[]
        if len(self.active_)==1: keep=[0]
        else:
            for j in range(len(self.active_)):
                if all(abs(c[j,k])<=self.threshold for k in keep): keep.append(j)
        self.support_=np.zeros(X.shape[1],bool); self.support_[self.active_[keep]]=True
        return self
    def transform(self,X): return np.asarray(X)[:,self.support_]
    def get_support(self): return self.support_.copy()

class MISelector(BaseEstimator,TransformerMixin):
    """MI is fit by Pipeline.fit, hence only on each inner-training fold."""
    def __init__(self,k='all',random_state=42,n_neighbors=3,kind='mi'):
        self.k=k; self.random_state=random_state; self.n_neighbors=n_neighbors; self.kind=kind
    def fit(self,X,y):
        X=np.asarray(X); n=X.shape[1]; self.n_samples_fit_=len(X)
        if self.kind=='corr' or self.k=='all':
            self.scores_=np.zeros(n); keep=np.arange(n)
        elif self.kind=='random':
            self.scores_=np.zeros(n); keep=np.sort(np.random.default_rng(self.random_state).choice(n,min(int(self.k),n),replace=False))
        else:
            self.scores_=mutual_info_classif(X,y,n_neighbors=self.n_neighbors,random_state=self.random_state)
            keep=np.argsort(self.scores_)[::-1][:min(int(self.k),n)]
        self.support_=np.zeros(n,bool); self.support_[keep]=True
        return self
    def transform(self,X): return np.asarray(X)[:,self.support_]
    def get_support(self): return self.support_.copy()

def build_pipeline(C=.1,k='all',selection='corr',seed=42,class_weight='balanced'):
    selector_kind='corr' if selection=='corr' else selection
    return Pipeline([('correlation',CorrelationFilter()),('scale',StandardScaler()),('selection',MISelector(k=k,random_state=seed,kind=selector_kind)),('model',LogisticRegression(C=C,solver='lbfgs',max_iter=5000,class_weight=class_weight,random_state=seed))])

def grouped_select(X,y,groups,selection='corr',k_grid=('all',),c_grid=(.01,.1,1),seed=42,weights=None):
    """Tune C/k with GroupKFold; each pipeline transformer refits inside each fold."""
    folds=list(GroupKFold(3).split(X,y,groups)); best=None
    for k in k_grid:
      for C in c_grid:
        vals=[]
        for tr,va in folds:
            pipe=build_pipeline(C,k,selection,seed)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                fitargs={}
                if weights is not None: fitargs['model__sample_weight']=np.asarray(weights)[tr]
                try: pipe.fit(X[tr],y[tr],**fitargs)
                except ConvergenceWarning:
                    pipe.set_params(model__max_iter=10000); pipe.fit(X[tr],y[tr],**fitargs)
            vals.append(balanced_accuracy_score(y[va],pipe.predict(X[va])))
        score=float(np.mean(vals))
        if best is None or score>best['score']:
            best={'score':score,'C':C,'k':k}
    final=build_pipeline(best['C'],best['k'],selection,seed)
    fitargs={}
    if weights is not None: fitargs['model__sample_weight']=np.asarray(weights)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        try: final.fit(X,y,**fitargs)
        except ConvergenceWarning:
            final.set_params(model__max_iter=10000); final.fit(X,y,**fitargs)
    return {'pipeline':final,'selection':selection,'C':best['C'],'k':best['k'],'cv_score':best['score']}

def transform(bundle,X): return bundle['pipeline'][:-1].transform(X)
def params(bundle):
    if 'coef' in bundle:return bundle['coef'],bundle['intercept']
    m=bundle['pipeline'].named_steps['model'];return m.coef_.ravel(),float(m.intercept_[0])
def predict_logit(bundle,X):
    if 'coef' in bundle:
        c,b=params(bundle);return transform(bundle,X)@c+b
    return bundle['pipeline'].decision_function(X)
def predict_prob(bundle,X): return 1/(1+np.exp(-np.clip(predict_logit(bundle,X),-50,50)))

def selected_original_indices(bundle):
    p=bundle['pipeline']; corr=p.named_steps['correlation'].get_support(); sel=p.named_steps['selection'].get_support()
    return np.flatnonzero(corr)[sel]

def explain(bundle,X,background_X):
    z=transform(bundle,X); bg=transform(bundle,background_X).mean(axis=0); coef,_=params(bundle)
    phi=(z-bg)*coef
    channel=np.zeros((len(X),32)); orig=selected_original_indices(bundle)
    for j,idx in enumerate(orig): channel[:,idx//4]+=phi[:,j]
    logit=predict_logit(bundle,X); _,inter=params(bundle); base=float(np.dot(bg,coef)+inter)
    err=float(np.max(np.abs(channel.sum(axis=1)+base-logit)))
    if err>=1e-7: raise AssertionError(f'SHAP additivity max error {err:.3g} >= 1e-7')
    return channel,phi,bg,err

def masked_logit(bundle,x,bg,channels):
    z=transform(bundle,np.asarray(x)[None,:])[0].copy(); orig=selected_original_indices(bundle)
    for j,idx in enumerate(orig):
        if channels[idx//4]: z[j]=bg[j]
    c,b=params(bundle);return float(z@c+b)

def faithfulness(bundle,X,background_X,repeats=100,seed=42):
    attr,phi,bg,err=explain(bundle,X,background_X);rng=np.random.default_rng(seed);n=len(X)
    # Match the notebook's vectorized seeded mask draw (repeat, trial, channel).
    masks=rng.random((repeats,n,32))<.25;z=transform(bundle,X);coef,inter=params(bundle);orig=selected_original_indices(bundle);channel_ids=orig//4
    zmasked=np.where(masks[:,:,channel_ids],bg[None,None,:],z[None,:,:])
    original=predict_logit(bundle,X);actual=original[None,:]-(zmasked@coef+inter)
    full_pred=(masks*attr[None,:,:]).sum(axis=2)
    if not np.allclose(actual,full_pred,atol=1e-7):raise AssertionError(f'Masked SHAP reconstruction error {np.max(np.abs(actual-full_pred)):.3g}')
    order=np.argsort(-np.abs(attr),axis=1,kind='stable');compact=np.zeros_like(attr);rows=np.arange(n)[:,None];compact[rows,order[:,:5]]=attr[rows,order[:,:5]]
    compact_pred=(masks*compact[None,:,:]).sum(axis=2);den=float(np.mean(actual**2))
    return (float(1-np.mean((actual-compact_pred)**2)/den) if den>=1e-12 else np.nan),attr,err

def shuffled_faithfulness(bundle,X,background_X,repeats=10,mask_repeats=100,seed=42):
    attr,_,_,_=explain(bundle,X,background_X);rng=np.random.default_rng(seed);scores=[];n=len(X);z=transform(bundle,X);bg=transform(bundle,background_X).mean(0);coef,inter=params(bundle);channels=selected_original_indices(bundle)//4
    masks=rng.random((mask_repeats,n,32))<.25;zm=np.where(masks[:,:,channels],bg[None,None,:],z[None,:,:]);actual=predict_logit(bundle,X)[None,:]-(zm@coef+inter)
    order=np.argsort(-np.abs(attr),axis=1,kind='stable');compact=np.zeros_like(attr);rows=np.arange(n)[:,None];compact[rows,order[:,:5]]=attr[rows,order[:,:5]];den=float(np.mean(actual**2))
    for _ in range(repeats):
      shuffled=np.take_along_axis(compact,np.argsort(rng.random(compact.shape),axis=1),axis=1);pred=(masks*shuffled[None,:,:]).sum(2)
      scores.append(1-float(np.mean((actual-pred)**2))/den if den>=1e-12 else np.nan)
    return float(np.nanmean(scores)),scores

def deletion(bundle,X,background_X,attr,subject,model_name,seed=42):
    from scipy.integrate import trapezoid
    rng=np.random.default_rng(seed); ks=[0,1,3,5,10,20,32]; rows=[]; topcur=[]; randcur=[]; bg=transform(bundle,background_X).mean(0)
    for i,x in enumerate(X):
      top=np.argsort(-np.abs(attr[i]),kind='stable'); least=np.argsort(np.abs(attr[i]),kind='stable'); rand=rng.permutation(32); original=float(predict_logit(bundle,x[None,:])[0]); originalp=float(predict_prob(bundle,x[None,:])[0]); values={}
      for name,order in [('top',top),('least',least),('random',rand)]:
        for k in ks:
          mask=np.zeros(32,bool); mask[order[:k]]=True; v=masked_logit(bundle,x,bg,mask); dp=float(originalp-1/(1+np.exp(-np.clip(v,-50,50))))
          rows.append({'subject':subject,'model':model_name,'trial_index':i,'order':name,'k':k,'log_odds_change':original-v,'probability_change':dp}); values[(name,k)]=original-v
      topcur.append([values['top',k] for k in ks]); randcur.append([values['random',k] for k in ks])
    denom=32
    area=float(np.mean([trapezoid(np.abs(np.asarray(t)-np.asarray(r)),x=ks)/denom for t,r in zip(topcur,randcur)]))
    return rows,area
