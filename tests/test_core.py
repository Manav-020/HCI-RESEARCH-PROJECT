import numpy as np
from src.pipeline import extract_trial, norm_subject, shrink_fit, shrink_logits, explanation, fit_lr, shrink_objective, assert_no_target_leak, bandpow, ARTIFACT_RMS_Z_MAX, ARTIFACT_P2P_MAX, artifact_scale_audit
from sklearn.linear_model import LogisticRegression
from src.models import grouped_select, explain, faithfulness, deletion, MISelector, predict_logit, CorrelationFilter
from src.experiment import (fixed_fit, shrink_bundle, shrink_objective, weighted_mix_sample_weights,
                            resample_subject, explanation_stability, fit_source_grouped, select_lambda_variants, intercept_bundle)
from src.pipeline import cached_features
from src.exploratory import extract_sequence_trial
from scipy.signal import welch

def test_window_count_and_minimum_kept():
    x=np.random.default_rng(1).normal(size=(32,8064))
    _,n,rejected,_=extract_trial(x,4)
    assert n==26 and n-rejected>=13

def test_artifact_thresholds_and_audit_meet_fallback_target():
    assert ARTIFACT_RMS_Z_MAX==8 and ARTIFACT_P2P_MAX==2000
    x=np.random.default_rng(91).normal(0,20,size=(2,32,8064))
    _,sweep=artifact_scale_audit({1:{'x':x}})
    chosen=sweep[(sweep.rms_robust_z_max==8)&(sweep.peak_to_peak_max==2000)]
    assert len(chosen)==3 and (chosen.fallback_rate<.10).all()

def test_l0_bandpower_matches_notebook_integrated_welch():
    x=np.random.default_rng(92).normal(size=(32,8064));f,p=welch(x,fs=128,nperseg=256,noverlap=128,axis=-1);df=f[1]-f[0]
    bands=((4,8),(8,13),(13,30),(30,45));expected=np.stack([np.log(np.maximum(p[:,(f>=lo)&(f<hi)].sum(-1)*df,1e-12)) for lo,hi in bands],axis=1).reshape(-1)
    np.testing.assert_allclose(bandpow(x).reshape(-1),expected,rtol=0,atol=0)

def test_correlation_filter_drops_constants_and_keeps_greedy_feature_order():
    x=np.random.default_rng(93).normal(size=(100,4));x[:,1]=x[:,0];x[:,3]=4
    f=CorrelationFilter(.9).fit(x)
    assert f.get_support().tolist()==[True,False,True,False]

def test_lambda_criteria_and_ba_tie_break_choose_larger_lambda():
    vals={'0':{'brier':.3,'log_loss':.8,'balanced_accuracy':.7},'1':{'brier':.2,'log_loss':.6,'balanced_accuracy':.8},'inf':{'brier':.1,'log_loss':.5,'balanced_accuracy':.8}}
    chosen=select_lambda_variants(vals)
    assert chosen=={'shrinkage_brier':np.inf,'shrinkage_ba':np.inf,'shrinkage_finite':1.0}

def test_normalization_uses_first_twenty():
    rng=np.random.default_rng(4); a=rng.normal(size=(40,3)); b=a.copy(); b[20:]+=10000
    z=norm_subject({1:a},[1])[1]; z2=norm_subject({1:b},[1])[1]
    np.testing.assert_allclose(z[:20],z2[:20]); assert not np.allclose(z[20:],z2[20:])

def test_l2_features_differ_from_l1_and_first_twenty_are_standardized():
    rng=np.random.default_rng(104)
    l1={s:rng.normal(loc=s*3,scale=2+s*.1,size=(40,16)) for s in [1,2,3]}
    l2=norm_subject(l1,[1,2,3])
    assert any(not np.allclose(l1[s],l2[s]) for s in l1)
    for s in l2:
        np.testing.assert_allclose(l2[s][:20].mean(axis=0),0,atol=1e-12)
        np.testing.assert_allclose(l2[s][:20].std(axis=0),1,atol=1e-7)

def test_shrink_infinity_reproduces_source():
    rng=np.random.default_rng(3); X=rng.normal(size=(80,8)); y=np.tile([0,1],40); groups=np.repeat(np.arange(4),20)
    fit=fit_lr(X,y,groups); sh=shrink_fit(fit,X[:10],y[:10],np.inf)
    np.testing.assert_allclose(sh['coef'],fit['model'].coef_.ravel())
    np.testing.assert_allclose(shrink_logits(sh,X[10:]),fit['model'].decision_function(fit['scaler'].transform(X[10:,fit['mask']])[:,np.isin(np.flatnonzero(fit['mask']),fit['inds'])]))

def test_shrink_analytic_gradient_matches_finite_difference():
    rng=np.random.default_rng(8); X=rng.normal(size=(9,4)); y=rng.integers(0,2,9); sw=np.ones(9); ws=rng.normal(size=4); bs=.2; lam=.7; t=rng.normal(size=5)
    loss,g=shrink_objective(t,X,y,sw,ws,bs,lam); eps=1e-6; numeric=[]
    for j in range(len(t)):
        step=np.zeros_like(t); step[j]=eps
        numeric.append((shrink_objective(t+step,X,y,sw,ws,bs,lam)[0]-shrink_objective(t-step,X,y,sw,ws,bs,lam)[0])/(2*eps))
    np.testing.assert_allclose(g,numeric,rtol=1e-5,atol=1e-6)

def test_target_subject_never_in_fit_subjects():
    assert_no_target_leak(7,[1,2,3])
    try: assert_no_target_leak(7,[1,7,3])
    except AssertionError: pass
    else: raise AssertionError('leak was not caught')

def test_additivity_for_logistic_model():
    rng=np.random.default_rng(5); X=rng.normal(size=(80,8)); y=np.tile([0,1],40); groups=np.repeat(np.arange(4),20)
    fit=fit_lr(X,y,groups); phi=explanation(fit,X); z=fit['scaler'].transform(X[:,fit['mask']])[:,np.isin(np.flatnonzero(fit['mask']),fit['inds'])]
    # Explanation uses zero background in standardized space; grouped signed contributions sum to centered logit.
    np.testing.assert_allclose(phi.sum(1)+fit['model'].intercept_[0],fit['model'].decision_function(z),atol=1e-7)

def test_exact_masked_shap_and_compact_faithfulness():
    rng=np.random.default_rng(31); X=rng.normal(size=(100,12)); y=(X[:,0]-X[:,5]>0).astype(int);g=np.repeat(np.arange(5),20)
    b=grouped_select(X[:80],y[:80],g[:80],'corr',['all'],[.1]); score,attr,err=faithfulness(b,X[80:],X[:80],repeats=8)
    assert err<1e-7 and np.isfinite(score)

def test_mix10_weights_are_normalized_and_calibration_is_tenfold():
    ys=np.array([0]*8+[1]*12);yc=np.array([0,0,1,1]);w=weighted_mix_sample_weights(ys,yc)
    assert np.isclose(w.sum(),len(ys))
    # Source class factors stay fixed; each calibration sample weighs 10x its same-class source trial.
    assert np.isclose(w[-4]/w[0],10) and np.isclose(w[-2]/w[8],10)

def test_mi_selection_excludes_target_subject_data():
    rng=np.random.default_rng(67); features={s:rng.normal(size=(40,12)) for s in [1,2,3,4]}; data={s:{'y':np.tile([0,1],20)} for s in features}
    b1=fit_source_grouped(4,features,data,[1,2,3],'mi',[10],[.1])
    features[4]=rng.normal(loc=1e5,size=(40,12));data[4]['y']=1-data[4]['y']
    b2=fit_source_grouped(4,features,data,[1,2,3],'mi',[10],[.1])
    np.testing.assert_array_equal(b1['pipeline'].named_steps['selection'].get_support(),b2['pipeline'].named_steps['selection'].get_support())
    assert b1['pipeline'].named_steps['selection'].n_samples_fit_==120

def test_bootstrap_resampling_is_stratified_and_stability_is_bounded():
    rng=np.random.default_rng(9); y=np.array([0]*13+[1]*7);X=rng.normal(size=(20,3));xb,yb=resample_subject(X,y,rng)
    assert np.bincount(yb,minlength=2).tolist()==[13,7]
    m=explanation_stability(np.eye(1,32,0),np.eye(1,32,0));assert np.allclose(m,[1,1,1])

def test_deletion_curve_has_required_k_and_orders():
    rng=np.random.default_rng(19);X=rng.normal(size=(80,8));y=np.tile([0,1],40);g=np.repeat(np.arange(4),20);b=grouped_select(X,y,g,'corr',['all'],[.1])
    from src.models import explain
    a=explain(b,X[:2],X[2:40])[0];rows,auc=deletion(b,X[:2],X[2:40],a,1,'general')
    assert {r['order'] for r in rows}=={'top','least','random'}
    assert {r['k'] for r in rows}=={0,1,3,5,10,20,32} and np.isfinite(auc) and auc>=0

def test_new_shrinkage_infinity_and_gradient():
    rng=np.random.default_rng(28);X=rng.normal(size=(80,8));y=np.tile([0,1],40);g=np.repeat(np.arange(4),20);b=fixed_fit(X,y,g,.1,'all','corr')
    sh=shrink_bundle(b,X[:20],y[:20],np.inf);np.testing.assert_allclose(sh['coef'],b['pipeline'].named_steps['model'].coef_.ravel());np.testing.assert_allclose(predict_logit(sh,X[20:]),predict_logit(b,X[20:]),atol=1e-12)
    xx=rng.normal(size=(7,3));yy=rng.integers(0,2,7);sw=np.ones(7);ws=rng.normal(size=3);t=rng.normal(size=4);f,grad=shrink_objective(t,xx,yy,sw,ws,.1,.5);eps=1e-6
    num=np.array([(shrink_objective(t+np.eye(4)[j]*eps,xx,yy,sw,ws,.1,.5)[0]-shrink_objective(t-np.eye(4)[j]*eps,xx,yy,sw,ws,.1,.5)[0])/(2*eps) for j in range(4)])
    np.testing.assert_allclose(grad,num,rtol=1e-5,atol=1e-6)

def test_intercept_personalization_freezes_coefficients():
    rng=np.random.default_rng(95);X=rng.normal(size=(80,8));y=np.tile([0,1],40);g=np.repeat(np.arange(4),20);fit=fixed_fit(X,y,g,.1,'all','corr')
    b=intercept_bundle(fit,X[:20],1-y[:20]);np.testing.assert_array_equal(b['coef'],fit['pipeline'].named_steps['model'].coef_.ravel())
    assert np.isfinite(b['intercept'])

def test_feature_cache_is_reused(tmp_path,monkeypatch):
    import src.pipeline as p
    calls=[]
    def fake(data,window,stats=False):
        calls.append(window);return {s:np.ones((40,4))*s for s in data},__import__('pandas').DataFrame([{'subject':s,'presentation_trial':1,'window_s':window,'n_windows':1,'rejected':0,'fallback_keep_all':False} for s in data])
    monkeypatch.setattr(p,'make_features',fake); data={1:{},2:{}};sig={1:'a',2:'b'}
    x,_=cached_features(data,4,tmp_path,sig);assert len(calls)==1
    y,_=cached_features(data,4,tmp_path,sig);assert len(calls)==1
    np.testing.assert_array_equal(x[2],y[2])

def test_gru_sequence_window_shape():
    x=np.random.default_rng(101).normal(size=(32,8064));seq=extract_sequence_trial(x)
    assert seq.shape==(26,128) and np.isfinite(seq).all()
