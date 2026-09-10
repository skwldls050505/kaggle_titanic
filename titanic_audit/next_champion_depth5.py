"""Champion 15 versus depth=5 only. Does not overwrite existing submissions.
Run from any directory. --smoke runs one fold only; default uses 3 x 5 folds.
"""
from pathlib import Path
import sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.interaction_experiments import FeaturePreprocessor
from common.feature_experiments import baseline_parameters, fit_cat

CHAMPION_FEATURES = ['GenderClass', 'GenderIsChild', 'ClassIsChild', 'AgeBand']

def fit_candidate(encoded, target):
    # depth만 변경. learning_rate 자동 계산 규칙도 Champion과 동일.
    model = CatBoostClassifier(**(baseline_parameters() | {'depth': 5}))
    model.fit(encoded, target.astype('int8'))
    return model

def run(smoke=False):
    train = pd.read_csv(ROOT/'csv/train.csv')
    X = train.drop(columns=['passengerid', 'survived'])
    y = train['survived']
    rows, summaries, oof_tables = [], [], []
    seeds = [42] if smoke else [42, 137, 2026]
    for seed in seeds:
        oof = {label: np.full(len(y), np.nan) for label in ['Champion15','Depth5']}
        seen = np.zeros(len(y), dtype=int)
        for fold, (ti, vi) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X,y), 1):
            prep = FeaturePreprocessor(CHAMPION_FEATURES)
            A = prep.fit_transform(X.iloc[ti])
            B = prep.transform(X.iloc[vi])
            assert A.columns.equals(B.columns)
            champion = fit_cat(A, y.iloc[ti])
            candidate = fit_candidate(A, y.iloc[ti])
            pa, pb = champion.get_all_params(), candidate.get_all_params()
            changed = {k for k in set(pa)|set(pb) if pa.get(k) != pb.get(k)}
            assert changed == {'depth','max_leaves'}, changed
            for label, model in [('Champion15',champion),('Depth5',candidate)]:
                positive_index = list(model.classes_).index(1)
                probability = model.predict_proba(B)[:,positive_index]
                assert np.isfinite(probability).all()
                oof[label][vi] = probability
                train_auc = roc_auc_score(y.iloc[ti],model.predict_proba(A)[:,positive_index])
                val_auc = roc_auc_score(y.iloc[vi], probability)
                rows.append(dict(split_seed=seed, model_seed=42, fold=fold, model=label,
                                 train_auc=train_auc, validation_auc=val_auc, gap=train_auc-val_auc))
            seen[vi] += 1
            print(f'split_seed={seed} fold={fold} completed',flush=True)
            if smoke:
                break
        if smoke:
            continue
        assert (seen==1).all()
        for label in oof:
            assert np.isfinite(oof[label]).all()
            fold_values=[r['validation_auc'] for r in rows if r['split_seed']==seed and r['model']==label]
            summaries.append(dict(split_seed=seed,model=label,cv_mean=float(np.mean(fold_values)),
                cv_std=float(np.std(fold_values,ddof=1)),oof_auc=float(roc_auc_score(y,oof[label]))))
        oof_tables.append(pd.DataFrame(dict(passengerid=train.passengerid,survived=y,split_seed=seed,**oof)))
    out = Path(__file__).resolve().parent / ('depth5_smoke' if smoke else 'depth5_results')
    out.mkdir(exist_ok=False)
    pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
    if not smoke:
        pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
        pd.concat(oof_tables,ignore_index=True).to_csv(out/'oof.csv',index=False)
        # Export the experimental candidate, without promoting it automatically.
        test=pd.read_csv(ROOT/'csv/test.csv'); template=pd.read_csv(ROOT/'csv/submission.csv')
        prep=FeaturePreprocessor(CHAMPION_FEATURES)
        full=prep.fit_transform(X); model=fit_candidate(full,y)
        transformed=prep.transform(test.drop(columns='passengerid'))
        assert full.columns.equals(transformed.columns)
        prob=model.predict_proba(transformed)[:,list(model.classes_).index(1)]
        assert set(test.passengerid)==set(template.passengerid)
        assert test.passengerid.is_unique and template.passengerid.is_unique
        result=template.copy()
        result['survived']=result.passengerid.map(pd.Series(prob,index=test.passengerid))
        assert result.survived.notna().all() and result.survived.between(0,1).all()
        result.to_csv(out/'titanic_champion15_depth5_candidate.csv',index=False)
        (out/'parameters.json').write_text(json.dumps(model.get_all_params(),indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries if summaries else rows).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args().smoke)
