"""Base Model과 이름 기반 단일 피처 후보를 공정하게 비교하는 도구."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common.evaluations import get_auc_score
from common.feature_experiments import fit_cat
from common.preprocessing import TitanicPreprocessor


NOBLE_TITLES = ["Sir", "Lady", "Countess", "Don", "Dona", "Jonkheer"]


def extract_title(data):
    """Name의 쉼표와 마침표 사이 문자열을 호칭으로 추출한다."""
    return (
        data["name"]
        .str.extract(r",\s*([^.]*)\.", expand=False)
        .str.strip()
        .fillna("Unknown")
    )


def make_profession_group(title):
    if title == "Dr":
        return "Medical"
    if title == "Rev":
        return "Clergy"
    if title in ["Capt", "Col", "Major"]:
        return "Military"
    return "Other"


class TitleFeaturePreprocessor(TitanicPreprocessor):
    """기존 전처리를 유지하고 요청된 이름 기반 피처 하나만 추가한다."""

    def __init__(self, feature=None):
        if feature not in {None, "IsNoble", "ProfessionGroup", "LogFare"}:
            raise ValueError(f"지원하지 않는 피처입니다: {feature}")
        self.feature = feature

    def feature_frame(self, raw, clean):
        data = self.make_features(clean)
        title = extract_title(raw)
        if self.feature == "IsNoble":
            data["IsNoble"] = title.isin(NOBLE_TITLES).astype(int)
        elif self.feature == "ProfessionGroup":
            data["ProfessionGroup"] = title.map(make_profession_group)
        elif self.feature == "LogFare":
            data["LogFare"] = np.log1p(clean["fare"])
        # Title은 중간 변수일 뿐이며 select_features가 name 원문을 제거한다.
        return self.select_features(data)

    def fit_transform(self, X):
        self.fit_missing(X)
        data = self.feature_frame(X, self.transform_missing(X))
        self.fit_encoding(data)
        return self.transform_encoding(data)

    def transform(self, X):
        data = self.feature_frame(X, self.transform_missing(X))
        return self.transform_encoding(data)


def fit_feature_model(X_train, y_train, X_valid, feature=None):
    prep = TitleFeaturePreprocessor(feature)
    encoded_train = prep.fit_transform(X_train)
    encoded_valid = prep.transform(X_valid)
    model = fit_cat(encoded_train, y_train)
    return prep, model, encoded_train, encoded_valid


def model_metrics(model, X_train, y_train, X_valid, y_valid):
    positive_index = list(model.classes_).index(1)
    train_probability = model.predict_proba(X_train)[:, positive_index]
    valid_probability = model.predict_proba(X_valid)[:, positive_index]
    return {
        "train_auc": get_auc_score(y_train, train_probability),
        "validation_auc": get_auc_score(y_valid, valid_probability),
        "gap": get_auc_score(y_train, train_probability)
        - get_auc_score(y_valid, valid_probability),
        "valid_probability": valid_probability,
        "valid_prediction": model.predict(X_valid).astype(int),
    }


def five_fold_compare(X, y, feature):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for fold, (train_index, valid_index) in enumerate(splitter.split(X, y), 1):
        fold_models = []
        for label, candidate in [("Base", None), ("Candidate", feature)]:
            _, model, encoded_train, encoded_valid = fit_feature_model(
                X.iloc[train_index], y.iloc[train_index], X.iloc[valid_index], candidate
            )
            fold_models.append(model)
            positive_index = list(model.classes_).index(1)
            fold_auc = get_auc_score(
                y.iloc[valid_index],
                model.predict_proba(encoded_valid)[:, positive_index],
            )
            rows.append({"fold": fold, "model": label, "auc": fold_auc})
        assert fold_models[0].get_all_params() == fold_models[1].get_all_params(), (
            "동일 fold에서 Base와 Candidate의 CatBoost 실효 파라미터가 다릅니다."
        )
        print(f"Fold {fold} 완료")
    fold_table = pd.DataFrame(rows)
    summary = fold_table.groupby("model")["auc"].agg(["mean", "std"])
    return fold_table, summary
