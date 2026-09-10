"""Titanic baseline 전처리. fit은 학습 부분에만, transform은 검증/test에 사용."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class TitanicPreprocessor:
    def fit_missing(self, X):
        self.columns_ = X.columns.tolist()
        self.missing_ratio_ = X.isna().mean()
        # 객실번호는 대부분 결측이며 baseline에서 원본 번호를 제외한다.
        self.drop_missing_ = ['cabin'] if self.missing_ratio_.get('cabin', 0) >= 0.2 else []
        data = X.drop(columns=self.drop_missing_)
        self.numeric_ = data.select_dtypes(include='number').columns.tolist()
        self.fill_ = {}
        for col in data:
            self.fill_[col] = data[col].median() if col in self.numeric_ else data[col].mode().iloc[0]
        # 나이는 성별/객실등급별 중앙값. 미관측 그룹은 전체 학습 중앙값 사용.
        self.age_groups_ = data.groupby(['gender', 'pclass'])['age'].median()
        return self

    def transform_missing(self, X):
        if X.columns.tolist() != self.columns_:
            raise ValueError('학습/예측 입력 컬럼과 순서가 다릅니다.')
        data = X.drop(columns=self.drop_missing_).copy()
        keys = pd.MultiIndex.from_frame(data[['gender', 'pclass']])
        grouped_age = pd.Series(self.age_groups_.reindex(keys).to_numpy(), index=data.index)
        data['age'] = data['age'].fillna(grouped_age)
        return data.fillna(self.fill_)

    @staticmethod
    def make_features(X):
        data = X.copy()
        # 기존 노트북의 가족 크기/단독 승선 피처만 재사용한다.
        data['FamilySize'] = data['sibsp'] + data['parch'] + 1
        data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
        return data

    @staticmethod
    def select_features(X):
        # 이름/티켓 원문은 고유값이 많아 baseline 원핫 차원을 과도하게 늘린다.
        # family_size는 FamilySize와 완전히 같으므로 이중 생성하지 않는다.
        return X.drop(columns=['name', 'ticket'])

    def fit_encoding(self, X):
        self.cat_cols_ = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        self.num_cols_ = X.columns.difference(self.cat_cols_, sort=False).tolist()
        self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64)
        self.encoder_.fit(X[self.cat_cols_])
        self.output_columns_ = self.num_cols_ + self.encoder_.get_feature_names_out(self.cat_cols_).tolist()
        return self

    def transform_encoding(self, X):
        values = np.column_stack([X[self.num_cols_].to_numpy(dtype=float),
                                  self.encoder_.transform(X[self.cat_cols_])])
        if not np.isfinite(values).all():
            raise ValueError('전처리 결과에 NaN 또는 무한대가 있습니다.')
        return pd.DataFrame(values, index=X.index, columns=self.output_columns_)

    def fit_transform(self, X):
        self.fit_missing(X)
        data = self.select_features(self.make_features(self.transform_missing(X)))
        self.fit_encoding(data)
        return self.transform_encoding(data)

    def transform(self, X):
        data = self.select_features(self.make_features(self.transform_missing(X)))
        return self.transform_encoding(data)
