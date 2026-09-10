"""10~15차 전용 전처리. 과거 기록을 보존하고 GenderClass 이름을 사용한다."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from common.preprocessing import TitanicPreprocessor
from common.feature_experiments import baseline_parameters, fit_cat, scores
from common.evaluations import get_auc_score


class FeaturePreprocessor(TitanicPreprocessor):
    def __init__(self, additions=()):
        self.additions = tuple(additions)

    @staticmethod
    def ticket_prefix(X):
        ticket = X['ticket'].fillna('').astype(str).str.upper().str.strip()
        # 마지막 숫자 일련번호를 떼고 표기 구두점만 정규화한다.
        prefix = ticket.str.replace(r'\s*\d+$', '', regex=True)
        prefix = prefix.str.replace(r'[^A-Z0-9]', '', regex=True)
        return prefix.replace('', 'NUMERIC')

    def fit_missing(self, X):
        super().fit_missing(X)
        counts = self.ticket_prefix(X).value_counts()
        self.common_prefixes_ = counts[counts > 5].index.tolist()
        return self

    def feature_frame(self, raw, clean):
        data = self.make_features(clean)
        if 'GenderClass' in self.additions:
            data['GenderClass'] = data['gender'].astype(str) + '_' + data['pclass'].astype(str)
        # IsChild는 조합 계산에만 사용하고 독립 입력으로 추가하지 않는다.
        child = (data['age'] < 16).astype(int).astype(str)
        if 'GenderIsChild' in self.additions:
            data['GenderIsChild'] = data['gender'].astype(str) + '_' + child
        if 'ClassIsChild' in self.additions:
            data['ClassIsChild'] = data['pclass'].astype(str) + '_' + child
        if 'GenderIsAlone' in self.additions:
            data['GenderIsAlone'] = data['gender'].astype(str) + '_' + data['IsAlone'].astype(str)
        if 'EmbarkedClass' in self.additions:
            data['EmbarkedClass'] = data['embarked'].astype(str) + '_' + data['pclass'].astype(str)
        if 'TicketPrefix' in self.additions:
            prefix = self.ticket_prefix(raw)
            data['TicketPrefix'] = prefix.where(prefix.isin(self.common_prefixes_), 'Rare')
        if 'AgeBand' in self.additions:
            data['AgeBand'] = pd.cut(data['age'], [0,16,20,35,60,np.inf], right=False,
                                     labels=['Child','Teen','YoungAdult','Adult','Senior']).astype(str)
        return self.select_features(data)

    def fit_transform(self, X):
        self.fit_missing(X)
        data = self.feature_frame(X, self.transform_missing(X))
        self.fit_encoding(data)
        return self.transform_encoding(data)

    def transform(self, X):
        data = self.feature_frame(X, self.transform_missing(X))
        return self.transform_encoding(data)


def fit_pair(X_train, y_train, X_valid, additions):
    prep = FeaturePreprocessor(additions)
    encoded_train = prep.fit_transform(X_train)
    encoded_valid = prep.transform(X_valid)
    model = fit_cat(encoded_train, y_train)
    return prep, model, encoded_train, encoded_valid


def cv_compare(X, y, previous, candidate):
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, va) in enumerate(splitter.split(X, y), 1):
        reference = None
        for label, additions in [('Original baseline', []), ('Previous best', previous), ('Candidate', candidate)]:
            # 피처 집합이 같으면 해당 fold 안에서만 결과를 재사용한다.
            key = tuple(additions)
            if label == 'Original baseline':
                cache = {}
            if key not in cache:
                _, model, _, valid = fit_pair(X.iloc[tr], y.iloc[tr], X.iloc[va], additions)
                if reference is None:
                    reference = model.get_all_params()
                assert model.get_all_params() == reference, '동일 fold에서 실효 파라미터가 달라졌습니다.'
                auc = get_auc_score(y.iloc[va], model.predict_proba(valid)[:, list(model.classes_).index(1)])
                cache[key] = auc
            rows.append({'fold': fold, 'model': label, 'auc': cache[key]})
        print('Fold', fold, '완료')
    table = pd.DataFrame(rows)
    return table, table.groupby('model')['auc'].agg(['mean', 'std'])
