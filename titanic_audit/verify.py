"""Read-only audit of source data/submissions; new outputs only in this directory."""
from pathlib import Path
import sys, json, hashlib, platform
import numpy as np
import pandas as pd
import sklearn, catboost
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OTHER = Path(r'C:\dev\project\kaggle_skn35 v1')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common.interaction_experiments import FeaturePreprocessor, fit_cat
from common.utils import train_test_split_by_target

train = pd.read_csv(ROOT/'csv/train.csv')
test = pd.read_csv(ROOT/'csv/test.csv')
template = pd.read_csv(ROOT/'csv/submission.csv')
X = train.drop(columns=['passengerid','survived']); y = train.survived
Xt = test.drop(columns='passengerid')
report = {'environment': {'python':platform.python_version(),'pandas':pd.__version__,
          'catboost':catboost.__version__,'sklearn':sklearn.__version__}, 'data':{},'csv':[], 'models':{}}
for name in ['train','test','submission']:
 a=ROOT/f'csv/{name}.csv'; b=OTHER/f'data/{name}.csv'
 report['data'][name]={'equal_values':pd.read_csv(a).equals(pd.read_csv(b)),
     'sha256_study':hashlib.sha256(a.read_bytes()).hexdigest(),
     'sha256_project':hashlib.sha256(b.read_bytes()).hexdigest()}
preds={}
for root in [ROOT, OTHER]:
 for p in (root/'submission').glob('*.csv'):
  df=pd.read_csv(p); prob=df['survived']
  key=p.name; preds[key]=df.set_index('passengerid').survived.sort_index()
  report['csv'].append({'file':str(p),'shape':list(df.shape),'columns':df.columns.tolist(),
     'id_match':df.passengerid.equals(template.passengerid),'id_unique':df.passengerid.is_unique,
     'finite':bool(np.isfinite(prob).all()),'range_ok':bool(prob.between(0,1).all()),
     'min':float(prob.min()),'max':float(prob.max()),'unique':int(prob.nunique()),
     'hard_labels':bool(prob.isin([0,1]).all()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
report['identical_submissions']=[[a,b] for i,a in enumerate(preds) for b in list(preds)[i+1:]
                                 if np.array_equal(preds[a].values,preds[b].values)]
tr,va=train_test_split_by_target(train,target_name='survived')
sets={'Base':[], '11':['GenderClass','GenderIsChild','ClassIsChild'],
      '15':['GenderClass','GenderIsChild','ClassIsChild','AgeBand']}
saved_names={'Base':'submission_result_1.csv','11':'titanic_result_11.csv','15':'titanic_result_15.csv'}
oofs=pd.DataFrame({'passengerid':train.passengerid,'survived':y})
for label, additions in sets.items():
 prep=FeaturePreprocessor(additions)
 A=prep.fit_transform(X.loc[tr.index]); B=prep.transform(X.loc[va.index]); m=fit_cat(A,y.loc[tr.index])
 p=m.predict_proba(B)[:,list(m.classes_).index(1)]
 ta=roc_auc_score(y.loc[tr.index],m.predict_proba(A)[:,list(m.classes_).index(1)])
 va_auc=roc_auc_score(y.loc[va.index],p)
 item={'train_auc':ta,'validation_auc':va_auc,'gap':ta-va_auc,
       'confusion_matrix_holdout':confusion_matrix(y.loc[va.index],p>=0.5).tolist(),
       'holdout_params':m.get_all_params()}
 full=FeaturePreprocessor(additions); F=full.fit_transform(X); m=fit_cat(F,y)
 tp=m.predict_proba(full.transform(Xt))[:,list(m.classes_).index(1)]
 saved=preds[saved_names[label]].reindex(test.passengerid).to_numpy()
 item.update({'full_params':m.get_all_params(),'features':F.columns.tolist(),
              'submission_max_absolute_difference':float(np.max(np.abs(tp-saved))),
              'submission_reproduced':bool(np.allclose(tp,saved,atol=1e-12,rtol=1e-12))})
 oof=np.full(len(y),np.nan); visits=np.zeros(len(y),dtype=int); folds=[]
 for f,(ti,vi) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y),1):
  prep=FeaturePreprocessor(additions); A=prep.fit_transform(X.iloc[ti]); B=prep.transform(X.iloc[vi])
  m=fit_cat(A,y.iloc[ti]); p=m.predict_proba(B)[:,list(m.classes_).index(1)]
  oof[vi]=p; visits[vi]+=1
  folds.append(float(roc_auc_score(y.iloc[vi],p)))
  print(label,'fold',f,folds[-1],flush=True)
 assert np.isfinite(oof).all() and (visits==1).all()
 item.update({'folds':folds,'cv_mean':float(np.mean(folds)),'cv_std_ddof1':float(np.std(folds,ddof=1)),
              'oof_auc':float(roc_auc_score(y,oof)),'confusion_matrix_oof':confusion_matrix(y,oof>=.5).tolist()})
 report['models'][label]=item; oofs[label]=oof
 (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(label,'completed',item['submission_reproduced'],flush=True)
oofs.to_csv(OUT/'oof_audit.csv',index=False)
p11=preds['titanic_result_11.csv'].reindex(test.passengerid).to_numpy()
p15=preds['titanic_result_15.csv'].reindex(test.passengerid).to_numpy()
delta=p15-p11
full=FeaturePreprocessor(sets['15']); full.fit_transform(X)
clean=full.feature_frame(Xt,full.transform_missing(Xt))
changes=pd.DataFrame({'passengerid':test.passengerid,'age':test.age,'AgeBand':clean.AgeBand,
                      'pclass':test.pclass,'gender':test.gender,'p11':p11,'p15':p15,'delta':delta})
changes.to_csv(OUT/'test_prediction_changes.csv',index=False)
report['prediction_changes']={'mean_abs':float(np.abs(delta).mean()),'max_abs':float(np.abs(delta).max()),
    'spearman':float(pd.Series(p11).corr(pd.Series(p15),method='spearman')),
    'class_flips_at_05':int(((p11>=.5)!=(p15>=.5)).sum()),
    'by_ageband':changes.groupby('AgeBand').agg(n=('delta','size'),mean_delta=('delta','mean'),
             mean_abs_delta=('delta',lambda z:z.abs().mean())).reset_index().to_dict('records')}
report['distributions']={}
for label,d in [('train',train),('test',test),('holdout_train',tr),('holdout_valid',va)]:
 report['distributions'][label]={'n':len(d),'female_fraction':float(d.gender.eq('female').mean()),
    'age_missing':float(d.age.isna().mean()),'cabin_missing':float(d.cabin.isna().mean()),
    'age_mean_observed':float(d.age.mean()),'fare_mean':float(d.fare.mean()),
    'pclass':d.pclass.value_counts(normalize=True).to_dict(),
    'ticket_overlap_other':int(d.ticket.isin({'train':test,'test':train,
             'holdout_train':va,'holdout_valid':tr}[label].ticket).sum())}
local=pd.DataFrame([{'label':k,'private':v,**{m:report['models'][k][m] for m in
        ['validation_auc','cv_mean','gap','oof_auc']}} for k,v in [('Base',.91118),('11',.91254),('15',.91468)]]).set_index('label')
report['local_private_correlations_n3']={'pearson':local.corr()['private'].to_dict(),
                                      'spearman':local.corr(method='spearman')['private'].to_dict()}
(OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('AUDIT COMPLETE',flush=True)
