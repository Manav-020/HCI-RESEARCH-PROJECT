"""EXPLORATORY - not part of the main claim: statistics features and CPU GRU."""
from pathlib import Path
import json
import copy
import numpy as np
from scipy.special import expit
from .pipeline import bandpow

def extract_sequence_trial(x):
    start,stop,w,step=640,7680,512,256
    chunks=np.stack([x[:,s:s+w] for s in range(start,stop-w+1,step)])
    feat=np.stack([bandpow(c).reshape(-1) for c in chunks])
    rms=np.sqrt(np.mean(chunks**2,axis=-1));med=np.median(rms,axis=0);mad=np.median(np.abs(rms-med),axis=0)+1e-12
    rz=np.max(.6745*np.abs(rms-med)/mad,axis=1);p2=np.max(np.ptp(chunks,axis=-1),axis=1);keep=(rz<=4)&(p2<=200)
    if keep.sum()<int(np.ceil(.5*len(keep))):keep[:]=True
    feat[~keep]=np.median(feat[keep],axis=0)
    return feat

def sequence_cache(data,cache_dir,signature):
    p=Path(cache_dir);p.mkdir(parents=True,exist_ok=True);path=p/'sequences_4s.npz';out={}
    if path.exists():
        try:
            with np.load(path,allow_pickle=False) as z:
                meta=json.loads(str(z['metadata'].item()));old={int(k):v for k,v in meta['signature'].items()}
                for s in meta['subjects']:
                    if int(s) in data and old.get(int(s))==signature.get(int(s)):out[int(s)]=z[f's{s}']
        except Exception:out={}
    for s in data:
        if s not in out:out[s]=np.stack([extract_sequence_trial(x) for x in data[s]['x']]).astype(np.float32)
    if len(out)!=len(data) or not path.exists() or any(s not in out for s in data):
        pass
    if out:
        payload={f's{s}':x for s,x in out.items()};payload['metadata']=np.asarray(json.dumps({'subjects':sorted(out),'signature':{str(s):signature[s] for s in out}}));np.savez_compressed(path,**payload)
    return out

def normalize_sequences(sequences,subjects):
    out={}
    for s in subjects:
        x=sequences[s];mu=x[:20].mean(axis=(0,1),keepdims=True);sd=x[:20].std(axis=(0,1),keepdims=True)+1e-8;out[s]=(x-mu)/sd
    return out

def train_gru(source,source_y,source_groups,seed=42,max_epochs=80):
    import torch
    from torch import nn
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    class Net(nn.Module):
        def __init__(self):
            super().__init__();self.gru=nn.GRU(128,16,batch_first=True);self.drop=nn.Dropout(.3);self.head=nn.Linear(16,1)
        def encode(self,x):
            _,h=self.gru(x);return h[-1]
        def forward(self,x):return self.head(self.drop(self.encode(x))).squeeze(-1)
    net=Net();groups=np.asarray(source_groups);valgroup=sorted(set(groups.tolist()))[-1];tr=np.flatnonzero(groups!=valgroup);va=np.flatnonzero(groups==valgroup)
    X=torch.tensor(source,dtype=torch.float32);Y=torch.tensor(source_y,dtype=torch.float32);opt=torch.optim.Adam(net.parameters(),lr=.003,weight_decay=.001);lossfn=nn.BCEWithLogitsLoss()
    best=np.inf;best_epoch=1;stale=0;generator=torch.Generator().manual_seed(seed)
    for epoch in range(max_epochs):
        net.train();order=torch.tensor(tr)[torch.randperm(len(tr),generator=generator)]
        for ids in order.split(64):
            opt.zero_grad();loss=lossfn(net(X[ids]),Y[ids]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():vl=float(lossfn(net(X[va]),Y[va]))
        if vl<best-1e-5:best=vl;best_epoch=epoch+1;stale=0
        else:stale+=1
        if stale>=10:break
    # Refit on every outer-source subject for the selected epoch count.
    torch.manual_seed(seed);net=Net();opt=torch.optim.Adam(net.parameters(),lr=.003,weight_decay=.001);all_ids=torch.arange(len(X));generator=torch.Generator().manual_seed(seed)
    for _ in range(best_epoch):
        net.train();order=all_ids[torch.randperm(len(all_ids),generator=generator)]
        for ids in order.split(64):
            opt.zero_grad();loss=lossfn(net(X[ids]),Y[ids]);loss.backward();opt.step()
    return net,best_epoch

def fine_tune_head(net,xcal,ycal,seed=42,steps=100,l2=1.):
    import torch
    from torch.nn import functional as F
    torch.manual_seed(seed);net.eval()
    for p in net.gru.parameters():p.requires_grad=False
    with torch.no_grad():h=net.encode(torch.tensor(xcal,dtype=torch.float32))
    y=torch.tensor(ycal,dtype=torch.float32);w0=net.head.weight.detach().clone();b0=net.head.bias.detach().clone();opt=torch.optim.Adam(net.head.parameters(),lr=.01)
    if len(np.unique(ycal))<2:return net
    for _ in range(steps):
        opt.zero_grad();logit=net.head(h).squeeze(-1);loss=F.binary_cross_entropy_with_logits(logit,y)+.5*l2*((net.head.weight-w0).pow(2).sum()+(net.head.bias-b0).pow(2).sum());loss.backward();opt.step()
    return net

def gru_logit(net,x):
    import torch
    net.eval()
    with torch.no_grad():return net(torch.tensor(x,dtype=torch.float32)).cpu().numpy()

def integrated_gradients(net,x,baseline,steps=24):
    import torch
    net.eval();xt=torch.tensor(x,dtype=torch.float32);bt=torch.tensor(baseline,dtype=torch.float32);delta=xt-bt;gr=[]
    for alpha in torch.linspace(0,1,steps):
        v=(bt+alpha*delta).detach().requires_grad_(True);o=net(v[None,:,:]).sum();g=torch.autograd.grad(o,v)[0];gr.append(g.detach())
    ig=delta*torch.stack(gr).mean(0);feat=ig.sum(0).cpu().numpy();channels=feat.reshape(32,4).sum(axis=1)
    with torch.no_grad():fx=float(net(xt[None,:,:]).item());fb=float(net(bt[None,:,:]).item())
    return channels,abs((fx-fb)-channels.sum())

def gru_faithfulness(net,X,baseline,attrs,repeats=100,seed=42):
    rng=np.random.default_rng(seed);errors=[];den=[];net.eval()
    for i,x in enumerate(X):
        top=np.argsort(abs(attrs[i]))[-5:];actual0=float(gru_logit(net,x[None])[0])
        for _ in range(repeats):
            mask=rng.random(32)<.25;xm=x.copy()
            for c in np.flatnonzero(mask):xm[:,c*4:(c+1)*4]=baseline[:,c*4:(c+1)*4]
            actual=actual0-float(gru_logit(net,xm[None])[0]);pred=float(attrs[i,np.intersect1d(top,np.flatnonzero(mask))].sum());errors.append((actual-pred)**2);den.append(actual**2)
    d=float(np.mean(den));return 1-float(np.mean(errors))/d if d>=1e-12 else np.nan

def run_gru_fold(features,data,source,target,seed=42):
    ysrc=np.concatenate([data[s]['y'] for s in source]);groups=np.repeat(source,40);Xsrc=np.concatenate([features[s] for s in source])
    net,epochs=train_gru(Xsrc,ysrc,groups,seed);xc=features[target][:20];yc=data[target]['y'][:20];xt=features[target][20:];yt=data[target]['y'][20:]
    personalized=fine_tune_head(copy.deepcopy(net),xc,yc,seed)
    baseline=Xsrc.mean(axis=0);rows=[];deletion_rows=[];ig_rows=[]
    for name,m in [('general',net),('shrinkage_personalized_head',personalized)]:
        logits=gru_logit(m,xt);probs=expit(logits);attrs=[];complete=[]
        for x in xt:
            a,e=integrated_gradients(m,x,baseline);attrs.append(a);complete.append(e)
        for channel in range(32):ig_rows.append({'subject':target,'model':name,'channel':channel+1,'mean_signed_ig':float(np.mean([a[channel] for a in attrs])),'mean_abs_ig':float(np.mean([abs(a[channel]) for a in attrs]))})
        from .pipeline import metrics
        faith=gru_faithfulness(m,xt,baseline,np.asarray(attrs),100,seed)
        aucs=[];ks=[0,1,3,5,10,20,32]
        from scipy.integrate import trapezoid
        rng=np.random.default_rng(seed)
        for i,x in enumerate(xt):
            original=float(logits[i]);p0=float(probs[i]); orders={'top':np.argsort(-abs(attrs[i])),'least':np.argsort(abs(attrs[i])),'random':rng.permutation(32)};curve={}
            for ordername,order in orders.items():
                for k in ks:
                    mask=np.zeros(32,bool);mask[order[:k]]=True;xm=x.copy()
                    for c in np.flatnonzero(mask):xm[:,4*c:4*c+4]=baseline[:,4*c:4*c+4]
                    vl=float(gru_logit(m,xm[None])[0]);pv=float(expit(vl));change=original-vl;curve[(ordername,k)]=change
                    deletion_rows.append({'subject':target,'model':name,'trial_index':i,'order':ordername,'k':k,'log_odds_change':change,'probability_change':p0-pv})
            aucs.append(float(trapezoid([abs(curve['top',k]-curve['random',k]) for k in ks],x=ks)/32))
        rows.append({'subject':target,'arm':'EXPLORATORY - not part of the main claim','model':name,**metrics(yt,probs),'compact_faithfulness':faith,'ig_completeness_error':float(np.mean(complete)),'deletion_auc':float(np.mean(aucs)),'epochs':epochs})
    return rows,deletion_rows,ig_rows
