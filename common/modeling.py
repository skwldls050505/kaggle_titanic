import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import enum 

from xgboost import XGBClassifier
from xgboost import plot_importance as xgb_plot_importance
from lightgbm import LGBMClassifier
from lightgbm import plot_importance as lgb_plot_importance
from catboost import CatBoostClassifier

# 이걸 쓰면 제일 좋은 걸 리턴해줌

from .utils import reset_seeds
from .evaluations import get_auc_score

# 그래프 그리는 거
def cat_plot_importance(cat):
    feature_importance = cat.feature_importances_
    feature_names = np.array(cat.feature_names_)
    sorted_idx = np.argsort(feature_importance)

    plt.figure(figsize=(12, 6))
    plt.barh(
        range(len(sorted_idx)),
        feature_importance[sorted_idx]
    )
    plt.yticks(
        range(len(sorted_idx)),
        feature_names[sorted_idx]
    )
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.show()

# 어떤 모델인지 확인하는 것 
# 사용할 부스팅 모델들을 한 군데에 정리해놓은 설정표
class BoostModelType(enum.Enum):
    xgb = (enum.auto(), XGBClassifier, 
            {'tree_method':'hist', 'enable_categorical':True}, xgb_plot_importance) # XGBoost에 필요한 정보들을 하나로 묶은 거
    lgb = (enum.auto(), LGBMClassifier, 
            {'verbose': -1}, lgb_plot_importance)
    cat = (enum.auto(), CatBoostClassifier, 
            {'verbose':0}, cat_plot_importance)
    

#value[0] → 고유 번호
# value[1] → 모델 클래스
# value[2] → 기본 하이퍼파라미터
# value[3] → feature importance 함수

class Modeling:

    def __init__( # 생 성 함  수 __ 는 외부에서 직접 접근 못하게 막는 것
        self, x_tr:pd.DataFrame, y_tr:pd.DataFrame, # x_tr : 모델에게 줄 문제, y_tr : 정답
        vali_func=get_auc_score, cat_cols = [ # 모델 평가 방법
            'pclass', 'sex', 'embarked', 'who', 'adult_male', 'deck', 'alone' # 인코딩해야 될 컬럼들 (카데고리화 할 컬럼들)
        ]) -> None:
        self.__best_model = { # 나중에 가장 성능 좋은 모델의 정보를 저장할 공간
            'model_name': None,
            'hpo': None,
            'train_score': 0.0,
            'test_score': 0.0,
            'score_type': None
            # 이건 다 초기값
        }
        # 필요한 것들을 객체 안에 저장
        self.__vali_func = vali_func # 평가 함수
        self.__cat_cols = cat_cols # 범주형 컬럼 이름
        self.__targets = y_tr # 학습 데이터에 대한 정답 데이터
        self.__features = x_tr  # 학습 데이터에 대한 피처 데이터
        self.__models = []
        self.__results = []
        self.__convert_dtype(self.__features) # 데이터 타입 변환
        self.__valid_features_targets() #함수
        # 왜 self.에 저장하냐면 뒤의 다른 메소드에서도 계속 사용해야 하기 때문
    
    # X_train의 데이터 타입을 모델들이 처리하기 좋은 형태로 바꾸는 함수
    def __convert_dtype(self, features):
        bool_cols = features.select_dtypes(include=["bool", "boolean"]).columns # bool 컬럼 찾기
        features[bool_cols] = features[bool_cols].astype("int8") # bool 컬럼을 int8로 바꿔서 모델이 처리할 수 있게끔! 아까 bool이라서 오류 발생
        # 전부 다 사용 가능하게끔 강사님이 바꿈
        # 기존에 사용 안되던 타입들이 다 바꿔짐

        features[self.__cat_cols] = features[self.__cat_cols].astype('category') # 범주형 컬럼은 카테고리로 변환 - > 이유 : XGBoost, LightGBM, CatBoost가 범주형 데이터라는 사실을 알 수 있도록 해주는 거

    def __valid_features_targets(self) -> None: # 검증하는 함수
        assert self.__features.isnull().sum().sum() == 0, "features에 결측치가 있습니다." # 결측치 검사
        assert len(self.__features) == len(self.__targets), "features와 targets의 데이터 수가 다릅니다." # x랑 y의 길이 검사


    def get_best_model(self):
        return self.__best_model

    def get_results(self):
        """test_score는 기존 API 이름이며 실제로는 validation AUC입니다."""
        return pd.DataFrame(self.__results).sort_values('validation_auc', ascending=False)

    def __fit(self, model_type:BoostModelType, add_hpo:dict):

        assert model_type in BoostModelType, "정상적인 모델 타입이 아닙니다." #XGB / LGB / CatBoost 중 하나인지 확인

        # 하이퍼 파라미터 정의 
        hpo = model_type.value[2] | add_hpo # 모델 타입에 따라 하이퍼 파라미터 정의 (model_type.value[2] 이건 기본값, or add_hpo는 사용자 지정값)
        if model_type is BoostModelType.cat: # CatBoost인 경우는 특별 처리
            hpo = hpo | {'cat_features': self.__cat_cols} # "이 컬럼들은 범주형 feature야" 라고 cat한테 알려쥼 
            hpo['allow_writing_files'] = False
        # 모델 생성 
        model = model_type.value[1](**hpo) # 실제 모델 생성

        
        # 모델 학습 
        model.fit(self.__features, self.__targets.astype("int8")) 

        return model, hpo #① 학습 완료된 모델, ② 사용한 하이퍼파라미터 반환

    def __evaluation(self, model, hpo, y_te, x_te):
        # 모델 평가  (auc : 실제 정답 y_te vs 모델 예측값 비교)
        positive_index = list(model.classes_).index(1)
        test_score = self.__vali_func(y=y_te, pred=model.predict_proba(x_te)[:, positive_index])
        train_score = self.__vali_func(y=self.__targets, pred=model.predict_proba(self.__features)[:, positive_index])
        self.__results.append({'model': model.__class__.__name__, 'train_auc': train_score,
                               'validation_auc': test_score})

        if self.__best_model['model_name'] is None or self.__best_model['test_score'] < test_score:
            self.__best_model = { # 그럼 best_model에 저장
                'model_name': model.__class__.__name__, # XGBClassifier , ..  이런 거
                'hpo': hpo,
                'train_score': train_score,
                'model': model,
                'test_score': test_score,
                'score_type': 'auc'
            }


    @reset_seeds()  # 같은 조건에서는 최대한 같은 결과가 나오도록 만드는 역할
    def fit_evaluation(self, y_te, x_te, add_hpo:dict={} ): # 빈 값 초기화
        self.fit(add_hpo)
        self.evaluate(y_te, x_te)

    @reset_seeds()
    def fit(self, add_hpo=None):
        """학습과 평가를 별도 단계로도 실행할 수 있도록 기존 로직 재사용."""
        self.__models = [self.__fit(kind, {'random_state': 42} | (add_hpo or {}))
                         for kind in BoostModelType]
        return self

    def evaluate(self, y_te, x_te):
        self.__convert_dtype(x_te)
        self.__results = []
        self.__best_model['model_name'] = None
        for model, hpo in self.__models:
            self.__evaluation(model, hpo, y_te, x_te)
        return self.get_results()
        
        


