"""3~9차 전용 도구. 기존 baseline 모듈과 모델 설정을 수정하지 않는다."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from common.preprocessing import TitanicPreprocessor
from common.modeling import BoostModelType
from common.utils import reset_seeds
from common.evaluations import get_auc_score


class FeaturePreprocessor(TitanicPreprocessor):
    def __init__(self, additions=()):
        self.additions = tuple(additions)

    @staticmethod
    def titles(X):
        return X['name'].str.extract(r',\s*([^.]*)\.', expand=False).str.strip().fillna('Unknown')

    def fit_missing(self, X):
        super().fit_missing(X)
        # 학습 표본에서 4개 주요 호칭은 36명 이상, 나머지는 5명 이하였다.
        # 빈도 기준만 사용하며 각 fold에서 다시 계산한다.
        counts = self.titles(X).value_counts()
        self.common_titles_ = counts[counts > 5].index.tolist()
        self.ticket_counts_ = X['ticket'].value_counts()
        return self

    def feature_frame(self, raw, clean):
        data = self.make_features(clean)
        if 'Title' in self.additions:
            title = self.titles(raw)
            data['Title'] = title.where(title.isin(self.common_titles_), 'Rare')
        if 'SexPclass' in self.additions:
            data['SexPclass'] = data['gender'].astype(str) + '_' + data['pclass'].astype(str)
        if 'FamilyGroup' in self.additions:
            # EDA에서 1명, 2~4명, 5명 이상이 다른 패턴. 검증 threshold 탐색 없음.
            data['FamilyGroup'] = pd.cut(data['FamilySize'], [0, 1, 4, np.inf],
                                          labels=['Alone', 'Small', 'Large']).astype(str)
        if 'TicketGroupSize' in self.additions or 'FarePerPerson' in self.additions:
            ticket_size = raw['ticket'].map(self.ticket_counts_).fillna(1).clip(lower=1)
            if 'TicketGroupSize' in self.additions:
                data['TicketGroupSize'] = ticket_size.astype(float)
            if 'FarePerPerson' in self.additions:
                data['FarePerPerson'] = data['fare'] / ticket_size
        if 'IsChild' in self.additions:
            # 치환 후 계산: 결측 승객을 자동으로 성인 취급하는 것을 피한다.
            data['IsChild'] = (data['age'] < 16).astype(int)
        if 'CabinDeck' in self.additions:
            data['CabinDeck'] = raw['cabin'].str.extract(r'^\s*([A-Za-z])', expand=False).str.upper().fillna('Unknown')
        return self.select_features(data)

    def fit_transform(self, X):
        self.fit_missing(X)
        data = self.feature_frame(X, self.transform_missing(X))
        self.fit_encoding(data)
        return self.transform_encoding(data)

    def transform(self, X):
        data = self.feature_frame(X, self.transform_missing(X))
        return self.transform_encoding(data)


def baseline_parameters():
    # titanic_1 Modeling.__fit과 같은 명시적 설정. 미지정 기본값도 변경하지 않는다.
    return dict(BoostModelType.cat.value[2]) | {
        'random_state': 42, 'cat_features': [], 'allow_writing_files': False}


@reset_seeds()
def fit_cat(X, y):
    model = BoostModelType.cat.value[1](**baseline_parameters())
    model.fit(X, y.astype('int8'))
    assert model.get_params() == baseline_parameters()
    return model


def scores(model, X_train, y_train, X_valid, y_valid):
    index = list(model.classes_).index(1)
    train_auc = get_auc_score(y_train, model.predict_proba(X_train)[:, index])
    valid_auc = get_auc_score(y_valid, model.predict_proba(X_valid)[:, index])
    return {'train_auc': train_auc, 'validation_auc': valid_auc, 'gap': train_auc-valid_auc}


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
