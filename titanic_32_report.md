# Titanic 32 최종 분석 보고서

## 1. Titanic_1 ~ Titanic_31 분석 결과

기준 모델은 Private 0.91468 / Public 0.89121로 기록된 원본 Titanic 15이다. 이 값은 이전 사용자 제공 점수를 보존한 titanic_audit/REPORT.md에서 확인했으며 Kaggle 서버를 새로 조회한 값은 아니다. Age+AgeBand 33열 모델을 복원해 원본 제출과 수치 일치를 검증했다. 28/31은 현재 age 제외 32열이며 28의 fold별 early stopping 점수 0.914325와 고정 모델의 CV를 혼동하지 않았다.

## 2. Feature 분석

유지한 최종 인코딩 입력: pclass, age, sibsp, fare, FamilySize, IsAlone, gender_female, gender_male, embarked_C, embarked_Q, embarked_S, GenderClass_female_1, GenderClass_female_2, GenderClass_female_3, GenderClass_male_1, GenderClass_male_2, GenderClass_male_3, GenderIsChild_female_0, GenderIsChild_female_1, GenderIsChild_male_0, GenderIsChild_male_1, ClassIsChild_1_0, ClassIsChild_1_1, ClassIsChild_2_0, ClassIsChild_2_1, ClassIsChild_3_0, ClassIsChild_3_1, AgeBand_Adult, AgeBand_Child, AgeBand_Senior, AgeBand_Teen, AgeBand_YoungAdult

제거한 의미 피처: ['parch']. 상관계수만으로 제거하지 않고 동일 파라미터의 단일 제거 실험을 비교했다.

추가한 피처: []. 추가 후보 두 개는 family_size를 각 행에서 계산해 티켓 관측 빈도 의존을 피했다. 개선 조건 미달 후보는 사용하지 않았다.

| Model | CV Mean | CV Std | OOF AUC | Train AUC | Train-CV Gap | Feature Count | CV delta | OOF delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drop pclass | 0.905717 | 0.019149 | 0.903385 | 0.965290 | 0.059573 | 32 | -0.000557 | -0.000862 |
| drop gender | 0.905845 | 0.018214 | 0.903491 | 0.964723 | 0.058877 | 31 | -0.000429 | -0.000756 |
| drop age | 0.907518 | 0.013868 | 0.905643 | 0.958796 | 0.051278 | 32 | 0.001244 | 0.001397 |
| drop sibsp | 0.906485 | 0.020001 | 0.903851 | 0.964815 | 0.058330 | 32 | 0.000211 | -0.000395 |
| drop parch | 0.908099 | 0.017823 | 0.904551 | 0.965532 | 0.057433 | 32 | 0.001824 | 0.000304 |
| drop fare | 0.899753 | 0.020324 | 0.897488 | 0.955564 | 0.055810 | 32 | -0.006521 | -0.006759 |
| drop embarked | 0.905866 | 0.021729 | 0.904652 | 0.963916 | 0.058050 | 30 | -0.000408 | 0.000406 |
| drop FamilySize | 0.906103 | 0.017031 | 0.903476 | 0.965022 | 0.058918 | 32 | -0.000171 | -0.000771 |
| drop IsAlone | 0.905642 | 0.016868 | 0.903810 | 0.965270 | 0.059628 | 32 | -0.000633 | -0.000436 |
| drop GenderClass | 0.904100 | 0.016280 | 0.902152 | 0.965473 | 0.061374 | 27 | -0.002175 | -0.002094 |
| drop GenderIsChild | 0.904785 | 0.018387 | 0.902523 | 0.965300 | 0.060515 | 29 | -0.001490 | -0.001724 |
| drop ClassIsChild | 0.902990 | 0.018164 | 0.901519 | 0.965457 | 0.062467 | 27 | -0.003285 | -0.002728 |
| drop AgeBand | 0.906233 | 0.018564 | 0.904389 | 0.964405 | 0.058172 | 28 | -0.000042 | 0.000142 |

새 피처 실험:
| Model | CV Mean | CV Std | OOF AUC | Train AUC | Train-CV Gap | Feature Count |
| --- | --- | --- | --- | --- | --- | --- |
| new FamilyFarePerPerson | 0.906449 | 0.016838 | 0.904541 | 0.968407 | 0.061959 | 33 |
| new ChildFamilySize | 0.906532 | 0.017471 | 0.903739 | 0.964995 | 0.058463 | 33 |

기준 모델의 OOF SHAP/그룹 permutation(상관된 대체 입력이 남아 있으므로 고유 기여를 과소평가할 수 있음):
| Feature | mean | std |
| --- | --- | --- |
| gender | 0.109866 | 0.018619 |
| GenderIsChild | 0.038616 | 0.014656 |
| fare | 0.018024 | 0.010001 |
| pclass | 0.006255 | 0.005509 |
| GenderClass | 0.004163 | 0.003929 |
| ClassIsChild | 0.002999 | 0.003633 |
| FamilySize | 0.001528 | 0.004687 |
| IsAlone | 0.000195 | 0.001762 |
| age | -0.000598 | 0.005379 |
| embarked | -0.000754 | 0.004002 |
| parch | -0.000904 | 0.002254 |
| sibsp | -0.002594 | 0.006310 |
| AgeBand | -0.003159 | 0.002708 |

## 3. Feature Correlation 분석

FamilySize=sibsp+parch+1은 정확한 합성 관계이고 IsAlone은 FamilySize의 결정적 이진 함수이다. AgeBand는 age의 구간화이고 GenderClass는 gender/pclass를 그대로 결합한다. categorical 관계는 임의 정수 Pearson 대신 Cramers V로 확인했다. Ticket/NonFamily 파생은 탐색용 train 통계이며 모델 입력에 자동 추가하지 않았다. Heatmap은 노트북에 저장되어 있다.

Pearson:
| index | age | AgeBandOrdinal | fare | LogFare | sibsp | parch | FamilySize | IsAlone | TicketGroupSize | FarePerPerson | NonFamilyGroupSize | NonFamilyGroupRatio | FamilyFarePerPerson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 1.000000 | 0.905553 | 0.161373 | 0.176120 | -0.216922 | -0.155138 | -0.227052 | 0.127893 | -0.190279 | 0.281307 | 0.084824 | 0.044076 | 0.187609 |
| AgeBandOrdinal | 0.905553 | 1.000000 | 0.141358 | 0.111938 | -0.283004 | -0.206572 | -0.298429 | 0.202091 | -0.242098 | 0.259550 | 0.120652 | 0.067998 | 0.198496 |
| fare | 0.161373 | 0.141358 | 1.000000 | 0.790558 | 0.162144 | 0.181826 | 0.204587 | -0.254572 | 0.371077 | 0.838195 | 0.152230 | 0.106485 | 0.845483 |
| LogFare | 0.176120 | 0.111938 | 0.790558 | 1.000000 | 0.314721 | 0.288180 | 0.362824 | -0.456914 | 0.482919 | 0.736190 | 0.069327 | 0.024180 | 0.620623 |
| sibsp | -0.216922 | -0.283004 | 0.162144 | 0.314721 | 1.000000 | 0.386491 | 0.872679 | -0.606890 | 0.585127 | 0.009600 | -0.493507 | -0.369110 | -0.089553 |
| parch | -0.155138 | -0.206572 | 0.181826 | 0.288180 | 0.386491 | 1.000000 | 0.787633 | -0.545113 | 0.554079 | 0.008936 | -0.415659 | -0.301286 | -0.080486 |
| FamilySize | -0.227052 | -0.298429 | 0.204587 | 0.362824 | 0.872679 | 0.787633 | 1.000000 | -0.694039 | 0.684247 | 0.011144 | -0.549755 | -0.406098 | -0.102439 |
| IsAlone | 0.127893 | 0.202091 | -0.254572 | -0.456914 | -0.606890 | -0.545113 | -0.694039 | 1.000000 | -0.438152 | -0.174885 | 0.423637 | 0.475451 | 0.053897 |
| TicketGroupSize | -0.190279 | -0.242098 | 0.371077 | 0.482919 | 0.585127 | 0.554079 | 0.684247 | -0.438152 | 1.000000 | 0.021848 | 0.232994 | 0.201308 | 0.194762 |
| FarePerPerson | 0.281307 | 0.259550 | 0.838195 | 0.736190 | 0.009600 | 0.008936 | 0.011144 | -0.174885 | 0.021848 | 1.000000 | 0.010165 | -0.050778 | 0.760160 |
| NonFamilyGroupSize | 0.084824 | 0.120652 | 0.152230 | 0.069327 | -0.493507 | -0.415659 | -0.549755 | 0.423637 | 0.232994 | 0.010165 | 1.000000 | 0.772134 | 0.359697 |
| NonFamilyGroupRatio | 0.044076 | 0.067998 | 0.106485 | 0.024180 | -0.369110 | -0.301286 | -0.406098 | 0.475451 | 0.201308 | -0.050778 | 0.772134 | 1.000000 | 0.246876 |
| FamilyFarePerPerson | 0.187609 | 0.198496 | 0.845483 | 0.620623 | -0.089553 | -0.080486 | -0.102439 | 0.053897 | 0.194762 | 0.760160 | 0.359697 | 0.246876 | 1.000000 |

Spearman:
| index | age | AgeBandOrdinal | fare | LogFare | sibsp | parch | FamilySize | IsAlone | TicketGroupSize | FarePerPerson | NonFamilyGroupSize | NonFamilyGroupRatio | FamilyFarePerPerson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 1.000000 | 0.908485 | 0.200352 | 0.200352 | -0.127383 | -0.226705 | -0.169844 | 0.116533 | -0.121853 | 0.389107 | 0.130453 | 0.116524 | 0.393049 |
| AgeBandOrdinal | 0.908485 | 1.000000 | 0.171460 | 0.171460 | -0.148424 | -0.235790 | -0.183772 | 0.127913 | -0.139260 | 0.368009 | 0.126464 | 0.111337 | 0.372966 |
| fare | 0.200352 | 0.171460 | 1.000000 | 1.000000 | 0.446109 | 0.359848 | 0.501159 | -0.511189 | 0.616053 | 0.739585 | -0.000251 | 0.013806 | 0.666638 |
| LogFare | 0.200352 | 0.171460 | 1.000000 | 1.000000 | 0.446109 | 0.359848 | 0.501159 | -0.511189 | 0.616053 | 0.739585 | -0.000251 | 0.013806 | 0.666638 |
| sibsp | -0.127383 | -0.148424 | 0.446109 | 0.446109 | 1.000000 | 0.454015 | 0.864700 | -0.848190 | 0.528458 | 0.105353 | -0.500407 | -0.491569 | -0.167797 |
| parch | -0.226705 | -0.235790 | 0.359848 | 0.359848 | 0.454015 | 1.000000 | 0.785937 | -0.669007 | 0.556900 | -0.076167 | -0.459141 | -0.424322 | -0.267430 |
| FamilySize | -0.169844 | -0.183772 | 0.501159 | 0.501159 | 0.864700 | 0.785937 | 1.000000 | -0.965260 | 0.631950 | 0.070171 | -0.559221 | -0.547312 | -0.215215 |
| IsAlone | 0.116533 | 0.127913 | -0.511189 | -0.511189 | -0.848190 | -0.669007 | -0.965260 | 1.000000 | -0.580234 | -0.150819 | 0.517221 | 0.524329 | 0.126229 |
| TicketGroupSize | -0.121853 | -0.139260 | 0.616053 | 0.616053 | 0.528458 | 0.556900 | 0.631950 | -0.580234 | 1.000000 | 0.012086 | 0.172579 | 0.218530 | 0.098141 |
| FarePerPerson | 0.389107 | 0.368009 | 0.739585 | 0.739585 | 0.105353 | -0.076167 | 0.070171 | -0.150819 | 0.012086 | 1.000000 | -0.031193 | -0.051944 | 0.834291 |
| NonFamilyGroupSize | 0.130453 | 0.126464 | -0.000251 | -0.000251 | -0.500407 | -0.459141 | -0.559221 | 0.517221 | 0.172579 | -0.031193 | 1.000000 | 0.993632 | 0.411215 |
| NonFamilyGroupRatio | 0.116524 | 0.111337 | 0.013806 | 0.013806 | -0.491569 | -0.424322 | -0.547312 | 0.524329 | 0.218530 | -0.051944 | 0.993632 | 1.000000 | 0.396670 |
| FamilyFarePerPerson | 0.393049 | 0.372966 | 0.666638 | 0.666638 | -0.167797 | -0.267430 | -0.215215 | 0.126229 | 0.098141 | 0.834291 | 0.411215 | 0.396670 | 1.000000 |

Cramers V:
| index | gender | pclass | GenderClass | GenderIsChild | ClassIsChild | AgeBand |
| --- | --- | --- | --- | --- | --- | --- |
| gender | 1.000000 | 0.116144 | 1.000000 | 1.000000 | 0.161557 | 0.109819 |
| pclass | 0.116144 | 1.000000 | 1.000000 | 0.140746 | 1.000000 | 0.345829 |
| GenderClass | 1.000000 | 1.000000 | 1.000000 | 0.594211 | 0.638101 | 0.257953 |
| GenderIsChild | 1.000000 | 0.140746 | 0.594211 | 1.000000 | 0.586276 | 0.578755 |
| ClassIsChild | 0.161557 | 1.000000 | 0.638101 | 0.586276 | 1.000000 | 0.554630 |
| AgeBand | 0.109819 | 0.345829 | 0.257953 | 0.578755 | 0.554630 | 1.000000 |

## 4. Hyperparameter Tuning 결과

피처 집합 확정 후 16개 고정 조합을 평가했다. iterations 300/600/817/1000, learning_rate .005~.03, depth 4~6, L2 약1.945~20, random_strength .2~1, border_count 64/128/254, rsm .8/1을 검토했다. bagging_temperature 0/1은 Bayesian bootstrap에서만, min_data_in_leaf 10/20은 Depthwise에서만 사용했다. 각 recipe의 iterations는 fold 밖에서 고정되어 validation으로 early stopping을 하지 않는다. 최고점 0.0005 이내에서는 낮은 gap을 선호했다. 최종 탐색 선택 편향까지 제거한 nested CV는 아니며, 신뢰구간/통계적 우월성을 주장하지 않는다.

| Model | CV Mean | CV Std | OOF AUC | Train AUC | Train-CV Gap | Feature Count | Parameter |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tune 04 | 0.910049 | 0.016111 | 0.906589 | 0.935022 | 0.024973 | 32 | {'iterations': 600, 'learning_rate': 0.01, 'depth': 4, 'l2_leaf_reg': 10} |
| tune 05 | 0.909487 | 0.016209 | 0.906772 | 0.944506 | 0.035019 | 32 | {'iterations': 1000, 'learning_rate': 0.007, 'depth': 4, 'l2_leaf_reg': 3} |
| tune 06 | 0.909475 | 0.015828 | 0.906396 | 0.940704 | 0.031229 | 32 | {'iterations': 300, 'learning_rate': 0.03, 'depth': 4, 'l2_leaf_reg': 10} |
| tune 14 | 0.909229 | 0.015200 | 0.906214 | 0.950715 | 0.041486 | 32 | {'iterations': 600, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 10, 'grow_policy': 'Depthwise', 'min_data_in_leaf': 10} |
| tune 08 | 0.908916 | 0.015448 | 0.906701 | 0.952438 | 0.043522 | 32 | {'iterations': 1000, 'learning_rate': 0.005, 'depth': 6, 'l2_leaf_reg': 3} |
| tune 16 | 0.908448 | 0.018992 | 0.906711 | 0.949650 | 0.041202 | 32 | {'iterations': 817, 'learning_rate': 0.006971115660907851, 'depth': 4, 'l2_leaf_reg': 1.9452208847287389, 'random_strength': 0.3746261155838959, 'subsample': 0.9328729111737557, 'border_count': 128} |
| tune 13 | 0.908339 | 0.015078 | 0.905359 | 0.942617 | 0.034278 | 32 | {'iterations': 600, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 1} |
| tune 12 | 0.908264 | 0.015176 | 0.905405 | 0.941275 | 0.033011 | 32 | {'iterations': 600, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 0} |
| tune 15 | 0.907949 | 0.014831 | 0.905202 | 0.949181 | 0.041232 | 32 | {'iterations': 600, 'learning_rate': 0.01, 'depth': 6, 'l2_leaf_reg': 20, 'grow_policy': 'Depthwise', 'min_data_in_leaf': 20} |
| tune 01 | 0.907768 | 0.015017 | 0.905709 | 0.937244 | 0.029476 | 32 | {'iterations': 300, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 3} |
| tune 02 | 0.907566 | 0.017864 | 0.905694 | 0.948789 | 0.041223 | 32 | {'iterations': 600, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 3} |
| tune 10 | 0.907445 | 0.015695 | 0.905717 | 0.950512 | 0.043068 | 32 | {'iterations': 600, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 3, 'border_count': 64} |
| tune 11 | 0.907086 | 0.016930 | 0.905010 | 0.949265 | 0.042179 | 32 | {'iterations': 600, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 3, 'border_count': 128} |
| tune 03 | 0.905941 | 0.018372 | 0.904143 | 0.961581 | 0.055640 | 32 | {'iterations': 1000, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 3} |
| tune 09 | 0.905390 | 0.019270 | 0.903367 | 0.954688 | 0.049298 | 32 | {'iterations': 600, 'learning_rate': 0.015, 'depth': 5, 'l2_leaf_reg': 10, 'random_strength': 0.2, 'rsm': 0.8} |
| tune 07 | 0.905128 | 0.017022 | 0.902898 | 0.954056 | 0.048928 | 32 | {'iterations': 600, 'learning_rate': 0.01, 'depth': 6, 'l2_leaf_reg': 10, 'random_strength': 0.2} |

## 5. Candidate 모델 비교

| Model | CV Mean | CV Std | OOF AUC | Train AUC | Train-CV Gap | Feature Count |
| --- | --- | --- | --- | --- | --- | --- |
| A: historical Private champion 15 | 0.906275 | 0.016868 | 0.904247 | 0.965376 | 0.059101 | 33.000000 |
| B: noise-removal candidate | 0.908099 | 0.017823 | 0.904551 | 0.965532 | 0.057433 | 32.000000 |
| C: selected features + tuning | 0.910049 | 0.016111 | 0.906589 | 0.935022 | 0.024973 | 32.000000 |
| E: champion/new weights 0.25/0.75 | 0.909949 | 0.016049 | 0.906488 | 0.949380 | 0.039431 | 33.000000 |
| D: no new feature passed acceptance criteria | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 |

기준 15와 새 단일 모델 OOF 상관을 먼저 확인했고, 조건을 만족하면 25/50/75%의 세 비율만 비교했다.

| index | A: historical Private champion 15 | B: noise-removal candidate | C: selected features + tuning |
| --- | --- | --- | --- |
| A: historical Private champion 15 | 1.000000 | 0.999017 | 0.990931 |
| B: noise-removal candidate | 0.999017 | 1.000000 | 0.990461 |
| C: selected features + tuning | 0.990931 | 0.990461 | 1.000000 |

## 6. 최종 Titanic_32 설정

**선택:** C: selected features + tuning

최종 의미 피처 설정: {'drop': ['parch'], 'add': []}

가중치: None

실효 파라미터:

{
  "nan_mode": "Min",
  "eval_metric": "Logloss",
  "iterations": 600,
  "sampling_frequency": "PerTree",
  "leaf_estimation_method": "Newton",
  "random_score_type": "NormalWithModelSizeDecrease",
  "grow_policy": "SymmetricTree",
  "penalties_coefficient": 1,
  "boosting_type": "Plain",
  "model_shrink_mode": "Constant",
  "feature_border_type": "GreedyLogSum",
  "bayesian_matrix_reg": 0.10000000149011612,
  "eval_fraction": 0,
  "force_unit_auto_pair_weights": false,
  "l2_leaf_reg": 10,
  "random_strength": 1,
  "rsm": 1,
  "boost_from_average": false,
  "model_size_reg": 0.5,
  "pool_metainfo_options": {
    "tags": {}
  },
  "subsample": 0.800000011920929,
  "use_best_model": false,
  "class_names": [
    0,
    1
  ],
  "random_seed": 42,
  "depth": 4,
  "posterior_sampling": false,
  "border_count": 254,
  "classes_count": 0,
  "auto_class_weights": "None",
  "sparse_features_conflict_fraction": 0,
  "leaf_estimation_backtracking": "AnyImprovement",
  "best_model_min_trees": 1,
  "model_shrink_rate": 0,
  "min_data_in_leaf": 1,
  "loss_function": "Logloss",
  "learning_rate": 0.009999999776482582,
  "score_function": "Cosine",
  "task_type": "CPU",
  "leaf_estimation_iterations": 10,
  "bootstrap_type": "MVS",
  "max_leaves": 16
}

## 7. Previous Best vs Titanic_32

| Metric | Previous Best | Titanic_32 | Difference |
| --- | --- | --- | --- |
| Train AUC | 0.965299 | 0.940006 | -0.025293 |
| Validation AUC | 0.903643 | 0.902667 | -0.000976 |
| CV Mean | 0.906275 | 0.910049 | 0.003774 |
| CV Std | 0.016868 | 0.016111 | -0.000757 |
| OOF AUC | 0.904247 | 0.906589 | 0.002343 |
| Feature Count | 33.000000 | 32.000000 | -1.000000 |

Train AUC는 이 표에서 687행 holdout 학습 점수이고 후보표에서는 fold train 평균이다. Holdout은 과거부터 반복 사용된 보조 진단이므로 독립 검증으로 해석하지 않는다.

| Fold | Previous Best | Titanic_32 | Difference |
| --- | --- | --- | --- |
| 1 | 0.901065 | 0.897118 | -0.003947 |
| 2 | 0.923786 | 0.933829 | 0.010043 |
| 3 | 0.879799 | 0.893275 | 0.013476 |
| 4 | 0.912154 | 0.911200 | -0.000953 |
| 5 | 0.914569 | 0.914823 | 0.000254 |

OOF confusion (threshold=0.5):
| index | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| Previous Private best | 529 | 41 | 82 | 264 |
| Titanic 32 | 529 | 41 | 76 | 270 |

Ticket-group stress CV (seed=42, 진단만 수행하고 재탐색하지 않음):
| Model | CV Mean | CV Std | OOF AUC | Train AUC | Train-CV Gap | Feature Count |
| --- | --- | --- | --- | --- | --- | --- |
| Champion: ticket-group stress CV | 0.890189 | 0.036584 | 0.889991 | 0.966934 | 0.076745 | 33 |
| Titanic32 group stress | 0.898528 | 0.037083 | 0.896671 | 0.936663 | 0.038134 | 32 |

## 8. Prediction 변화

| Pearson | Spearman | Mean Absolute Difference | Max Absolute Difference | Mean absolute rank change | Max absolute rank change | Passengers with rank change | Meaningful probability change >= 0.01 | Class changes at 0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.990602 | 0.958623 | 0.033538 | 0.281796 | 17.928753 | 166.000000 | 380 | 285 | 8 |

의미 있는 확률 변화 기준은 절대 변화 0.01 이상으로 사전 정의했다. 순위는 평균 순위로 계산했다.

## 9. 최종 판단

**Titanic_32 제출해볼 가치 있음**. 동일 fold에서 여러 지표를 함께 비교한 개발 후보다. 실제 새로운 Private/Public AUC는 제출 전 알 수 없으며 기존 Private 점수를 반복 최적화 목표로 사용하지 않았다.

CSV 검증: 393행, test/template ID 순서 동일, 중복 ID 없음, 지정 컬럼만 존재, NaN 없음, 확률 0~1. SEED=42. 재실행 검증 결과는 노트북 metadata 및 마지막 검증 셀에 기록한다.

CSV SHA-256: 62f59e4bc85b15e6604b9145b68b01133757a33b28cce47393299be57ed38f15

## 부록 A. 전체 실험 이력과 증거

### 分析 범위

관련 프로젝트 파일을 내용과 SHA-256으로 읽었다. 가상환경/캐시 라이브러리는 제외했다. 오래된 출력은 현재 코드로 다시 생성한 값과 구분한다. 미기록을 0으로 취급하지 않는다.

노트북 42개, 텍스트/데이터 파일 147개, CSV 45개 확인.

| Experiment | Added | Removed | Train AUC | Validation AUC | CV Mean | CV Std | OOF AUC | Private | Public |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  | 0.963079 | 0.900065 | 0.901775 | 0.016322 | 0.899876 | 0.911180 | 0.896720 |
| 2 submission=Base |  |  | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 |
| 2 / HasCabin | HasCabin |  | 0.964623 | 0.898439 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 |
| 2 / AgeMissing | AgeMissing |  | 0.964178 | 0.899943 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 |
| 3 | Title |  | 0.965168 | 0.902992 | 0.900010 | 0.016136 | 미기록 | 미기록 | 미기록 |
| 4 | SexPclass |  | 0.963047 | 0.902911 | 0.904676 | 0.016132 | 미기록 | 미기록 | 미기록 |
| 5 | FamilyGroup |  | 0.964434 | 0.902748 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 |
| 6 | TicketGroupSize |  | 0.964691 | 0.899984 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 |
| 7 | FarePerPerson |  | 0.965835 | 0.902748 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 |
| 8 | IsChild |  | 0.964218 | 0.903155 | 0.902251 | 0.017383 | 미기록 | 미기록 | 미기록 |
| 9 | CabinDeck |  | 0.964011 | 0.904781 | 0.902823 | 0.013704 | 미기록 | 미기록 | 미기록 |
| 10 | GenderIsChild |  | 0.965065 | 0.904293 | 0.905907 | 0.017324 | 미기록 | 미기록 | 미기록 |
| 11 | ClassIsChild |  | 0.964795 | 0.906489 | 0.906233 | 0.018564 | 0.904389 | 0.912540 | 0.896720 |
| 12 | GenderIsAlone |  | 0.963471 | 0.904619 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 |
| 13 | EmbarkedClass |  | 0.963489 | 0.905757 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 |
| 14 | TicketPrefix |  | 0.969699 | 0.902423 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 |
| 15 | AgeBand |  | 0.965299 | 0.903643 | 0.906275 | 0.016868 | 0.904247 | 0.914680 | 0.891210 |
| 16_randomized_search |  |  | 0.952909 | 0.900065 | 0.910481 | 0.017304 | 0.907781 | 미기록 | 미기록 |
| 16_delete_age |  | age | 0.965299 | 0.903643 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 |
| 16_isnoble | IsNoble |  | 0.963227 | 0.899821 | 0.898784 | 0.014441 | 미기록 | 미기록 | 미기록 |
| 17_profession_group | ProfessionGroup |  | 0.962182 | 0.899496 | 0.900609 | 0.016918 | 미기록 | 미기록 | 미기록 |
| 18 | LogFare |  | 0.965128 | 0.900309 | 0.898336 | 0.015689 | 미기록 | 미기록 | 미기록 |
| 19 | AgeBand | age | 0.964272 | 0.895552 | 0.904387 | 0.011004 | 미기록 | 미기록 | 미기록 |
| 20 |  | LogFare | 0.963079 | 0.900065 | 0.901775 | 0.016322 | 미기록 | 미기록 | 미기록 |
| 21 |  | IsAlone | 0.964155 | 0.903480 | 0.900862 | 0.018278 | 미기록 | 미기록 | 미기록 |
| 22 | NonFamilyGroupSize |  | 0.965556 | 0.899333 | 0.900966 | 0.016310 | 미기록 | 미기록 | 미기록 |
| 23 | TicketFarePerPerson |  | 0.967393 | 0.901935 | 0.900752 | 0.015635 | 미기록 | 미기록 | 미기록 |
| 24 | NonFamilyGroupRatio |  | 0.966574 | 0.896975 | 0.900748 | 0.014903 | 미기록 | 미기록 | 미기록 |
| 25 | NonFamilyGroupSize,TicketFarePerPerson,NonFamilyGroupRatio |  | 0.967303 | 0.896731 | 0.898545 | 0.014154 | 미기록 | 미기록 | 미기록 |
| 26 |  |  | 0.963754 | 0.900390 | 0.909335 | 0.014127 | 미기록 | 미기록 | 미기록 |
| 27 |  |  | 0.973396 | 0.895633 | 0.909254 | 0.015288 | 미기록 | 미기록 | 미기록 |
| 28 |  |  | 0.925486 | 0.906367 | 0.914325 | 0.012962 | 0.878288 | 미기록 | 미기록 |
| 29 |  |  | 0.929900 | 0.898845 | 0.907632 | 0.009901 | 0.869942 | 미기록 | 미기록 |
| 30 |  |  | 0.926211 | 0.905107 | 0.911666 | 0.013079 | 0.886363 | 미기록 | 미기록 |
| 31 |  |  | 0.925486 | 0.906367 | 0.914325 | 0.012962 | 0.878288 | 미기록 | 미기록 |

### 피처 분류 (확실함은 이 개발 기록 내에서만 의미)

A: GenderClass/SexPclass, GenderIsChild, ClassIsChild — 당시 누적 holdout/CV와 일부 Private 방향이 일치. 기본 gender/pclass/fare는 강한 신호지만 각각 독립적인 역사적 제거 실험은 없어 이번 ablation으로 보완한다.

B: AgeBand — Private15에는 이득, 로컬 지표 혼재; age와 함께 쓴 champion을 보존. CabinDeck/HasCabin/AgeMissing/FamilyGroup/EmbarkedClass/GenderIsAlone/FarePerPerson — 단일 지표 또는 미미한 차이, 충분한 반복 근거 없음.

C: LogFare — 단조 변환/중복 후보; 18의 Base 오염으로 독립 추가 효과 주장 불가. IsAlone/FamilySize/sibsp/parch — 구조적 중복 후보이나 삭제 효과는 반드시 실험. NonFamilyGroupSize의 .000104 CV 차이는 실질적 이득 근거 부족.

D: Title/IsChild/IsNoble/ProfessionGroup/TicketPrefix/TicketGroupSize/NonFamilyGroupRatio/세 Ticket 결합 — 해당 조건 CV 또는 holdout 악화. 일반적으로 무효라 단정하지 않음.

원천 목록: passengerid(ID, 제외), survived(target), name/ticket/cabin(원문 제외 및 파생 원천), pclass, gender(별도 프로젝트 sex와 의미상 대응), age, sibsp, parch, fare, embarked. 파생 전체: FamilySize/family_size, IsAlone, HasCabin, AgeMissing, Title, SexPclass/GenderClass, FamilyGroup, TicketGroupSize, FarePerPerson/TicketFarePerPerson, IsChild, CabinDeck/Deck, GenderIsChild, ClassIsChild, GenderIsAlone, EmbarkedClass, TicketPrefix, AgeBand, IsNoble, ProfessionGroup/Occupation, LogFare, NonFamilyGroupSize, NonFamilyGroupRatio. 별도 seaborn 학습 파일의 who/adult_male/alive 등은 현재 대회 피처로 합치지 않는다.

### 실험별 상세

#### Titanic 1

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S'] | 0.963079 | 0.900065 | 0.901775 | 0.016322 | [0.8919172932330827, 0.9190821256038647, 0.8787821001779812, 0.9064327485380117, 0.9126620900076278] | 0.899876 | 0.911180 | 0.896720 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) |  |

#### Titanic 2 submission=Base

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 submission=Base |  |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S'] | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | HasCabin/AgeMissing both rejected; CSV is Base. |

#### Titanic 2 / HasCabin

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 / HasCabin | HasCabin |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'HasCabin'] | 0.964623 | 0.898439 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | CV not run; holdout -0.001626 vs Base; not repeated proof of noise. |

#### Titanic 2 / AgeMissing

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 / AgeMissing | AgeMissing |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'AgeMissing'] | 0.964178 | 0.899943 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | CV not run; holdout -0.000122, inconclusive. |

#### Titanic 3

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | Title |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare', 'Title_Rev'] | 0.965168 | 0.902992 | 0.900010 | 0.016136 | [0.8914160401002506, 0.9139333841851005, 0.8759852529875414, 0.9060513602847698, 0.9126620900076278] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical accepted=False; holdout delta vs then-best=+0.002927. |

#### Titanic 4

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | SexPclass |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'SexPclass_female_1', 'SexPclass_female_2', 'SexPclass_female_3', 'SexPclass_male_1', 'SexPclass_male_2', 'SexPclass_male_3'] | 0.963047 | 0.902911 | 0.904676 | 0.016132 | [0.8954887218045113, 0.9222603610475464, 0.8817060767861683, 0.9093567251461988, 0.9145690312738367] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical accepted=True; holdout delta vs then-best=+0.002846. |

#### Titanic 5

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | FamilyGroup |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'SexPclass_female_1', 'SexPclass_female_2', 'SexPclass_female_3', 'SexPclass_male_1', 'SexPclass_male_2', 'SexPclass_male_3', 'FamilyGroup_Alone', 'FamilyGroup_Large', 'FamilyGroup_Small'] | 0.964434 | 0.902748 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical accepted=False; holdout delta vs then-best=-0.000163. |

#### Titanic 6

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | TicketGroupSize |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'TicketGroupSize', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'SexPclass_female_1', 'SexPclass_female_2', 'SexPclass_female_3', 'SexPclass_male_1', 'SexPclass_male_2', 'SexPclass_male_3'] | 0.964691 | 0.899984 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical accepted=False; holdout delta vs then-best=-0.002927. |

#### Titanic 7

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | FarePerPerson |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'FarePerPerson', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'SexPclass_female_1', 'SexPclass_female_2', 'SexPclass_female_3', 'SexPclass_male_1', 'SexPclass_male_2', 'SexPclass_male_3'] | 0.965835 | 0.902748 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical accepted=False; holdout delta vs then-best=-0.000163. |

#### Titanic 8

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | IsChild |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'IsChild', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'SexPclass_female_1', 'SexPclass_female_2', 'SexPclass_female_3', 'SexPclass_male_1', 'SexPclass_male_2', 'SexPclass_male_3'] | 0.964218 | 0.903155 | 0.902251 | 0.017383 | [0.8944235588972432, 0.919209255021612, 0.8758581235697941, 0.9082125603864734, 0.9135519959318587] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical accepted=False; holdout delta vs then-best=+0.000244. |

#### Titanic 9

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | CabinDeck |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'SexPclass_female_1', 'SexPclass_female_2', 'SexPclass_female_3', 'SexPclass_male_1', 'SexPclass_male_2', 'SexPclass_male_3', 'CabinDeck_A', 'CabinDeck_B', 'CabinDeck_C', 'CabinDeck_D', 'CabinDeck_E', 'CabinDeck_F', 'CabinDeck_G', 'CabinDeck_T', 'CabinDeck_Unknown'] | 0.964011 | 0.904781 | 0.902823 | 0.013704 | [0.8944862155388471, 0.9185736079328757, 0.8841215357233663, 0.9052885837782864, 0.9116450546656496] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical accepted=False; holdout delta vs then-best=+0.001870. |

#### Titanic 10

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | GenderIsChild |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1'] | 0.965065 | 0.904293 | 0.905907 | 0.017324 | [0.8964285714285715, 0.92442156114925, 0.8809433002796847, 0.9108822781591661, 0.9168573607932875] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical accepted=True; holdout delta vs then-best=+0.001382. |

#### Titanic 11

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | ClassIsChild |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1', 'ClassIsChild_1_0', 'ClassIsChild_1_1', 'ClassIsChild_2_0', 'ClassIsChild_2_1', 'ClassIsChild_3_0', 'ClassIsChild_3_1'] | 0.964795 | 0.906489 | 0.906233 | 0.018564 | [0.8979323308270678, 0.9263285024154589, 0.8785278413424867, 0.9117721840833969, 0.9166031019577929] | 0.904389 | 0.912540 | 0.896720 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical accepted=True; holdout delta vs then-best=+0.002195. |

#### Titanic 12

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | GenderIsAlone |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1', 'ClassIsChild_1_0', 'ClassIsChild_1_1', 'ClassIsChild_2_0', 'ClassIsChild_2_1', 'ClassIsChild_3_0', 'ClassIsChild_3_1', 'GenderIsAlone_female_0', 'GenderIsAlone_female_1', 'GenderIsAlone_male_0', 'GenderIsAlone_male_1'] | 0.963471 | 0.904619 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical accepted=False; holdout delta vs then-best=-0.001870. |

#### Titanic 13

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | EmbarkedClass |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1', 'ClassIsChild_1_0', 'ClassIsChild_1_1', 'ClassIsChild_2_0', 'ClassIsChild_2_1', 'ClassIsChild_3_0', 'ClassIsChild_3_1', 'EmbarkedClass_C_1', 'EmbarkedClass_C_2', 'EmbarkedClass_C_3', 'EmbarkedClass_Q_1', 'EmbarkedClass_Q_2', 'EmbarkedClass_Q_3', 'EmbarkedClass_S_1', 'EmbarkedClass_S_2', 'EmbarkedClass_S_3'] | 0.963489 | 0.905757 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical accepted=False; holdout delta vs then-best=-0.000732. |

#### Titanic 14

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | TicketPrefix |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1', 'ClassIsChild_1_0', 'ClassIsChild_1_1', 'ClassIsChild_2_0', 'ClassIsChild_2_1', 'ClassIsChild_3_0', 'ClassIsChild_3_1', 'TicketPrefix_A4', 'TicketPrefix_A5', 'TicketPrefix_C', 'TicketPrefix_CA', 'TicketPrefix_FCC', 'TicketPrefix_NUMERIC', 'TicketPrefix_PC', 'TicketPrefix_Rare', 'TicketPrefix_SCPARIS', 'TicketPrefix_SOC', 'TicketPrefix_SOPP', 'TicketPrefix_SOTONOQ', 'TicketPrefix_STONO2', 'TicketPrefix_WC'] | 0.969699 | 0.902423 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical accepted=False; holdout delta vs then-best=-0.004066. |

#### Titanic 15

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15 | AgeBand |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1', 'ClassIsChild_1_0', 'ClassIsChild_1_1', 'ClassIsChild_2_0', 'ClassIsChild_2_1', 'ClassIsChild_3_0', 'ClassIsChild_3_1', 'AgeBand_Adult', 'AgeBand_Child', 'AgeBand_Senior', 'AgeBand_Teen', 'AgeBand_YoungAdult'] | 0.965299 | 0.903643 | 0.906275 | 0.016868 | [0.9010651629072682, 0.9237859140605136, 0.8797991355199594, 0.9121535723366387, 0.9145690312738368] | 0.904247 | 0.914680 | 0.891210 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical accepted=False; holdout delta vs then-best=-0.002846. Historical Private CSV retained age+AgeBand (33 columns). Current common code removes age (32); stored markdown and recent output disagree. |

#### Titanic 16_randomized_search

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16_randomized_search |  |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1', 'ClassIsChild_1_0', 'ClassIsChild_1_1', 'ClassIsChild_2_0', 'ClassIsChild_2_1', 'ClassIsChild_3_0', 'ClassIsChild_3_1', 'AgeBand_Adult', 'AgeBand_Child', 'AgeBand_Senior', 'AgeBand_Teen', 'AgeBand_YoungAdult'] | 0.952909 | 0.900065 | 0.910481 | 0.017304 | 미기록 | 0.907781 | 미기록 | 미기록 | CatBoost | {'border_count': 128, 'depth': 4, 'iterations': 817, 'l2_leaf_reg': 1.9452208847287389, 'learning_rate': 0.006971115660907851, 'random_strength': 0.3746261155838959, 'subsample': 0.9328729111737557} | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Historical 33-column input; 30-trial search on same CV, private/public placeholders not scores. |

#### Titanic 16_delete_age

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16_delete_age |  | age | ['pclass', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1', 'ClassIsChild_1_0', 'ClassIsChild_1_1', 'ClassIsChild_2_0', 'ClassIsChild_2_1', 'ClassIsChild_3_0', 'ClassIsChild_3_1', 'AgeBand_Adult', 'AgeBand_Child', 'AgeBand_Senior', 'AgeBand_Teen', 'AgeBand_YoungAdult'] | 0.965299 | 0.903643 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Stored outputs are legacy 33-column numbers; current imported module deletes age. No reliable independent CV for this named variant. |

#### Titanic 16_isnoble

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16_isnoble | IsNoble |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'IsNoble'] | 0.963227 | 0.899821 | 0.898784 | 0.014441 | [0.888784, 0.911073, 0.879036, 0.903, 0.912026] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Base-derived, not a cumulative 15 experiment; only 1/5 improved folds. |

#### Titanic 17_profession_group

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17_profession_group | ProfessionGroup |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'ProfessionGroup'] | 0.962182 | 0.899496 | 0.900609 | 0.016918 | [0.890664, 0.917684, 0.876112, 0.907958, 0.910628] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Base-derived; only 1/5 improved folds. |

#### Titanic 18

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | LogFare |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'LogFare'] | 0.965128 | 0.900309 | 0.898336 | 0.015689 | 미기록 | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | 18 branch: no GenderClass/GenderIsChild/ClassIsChild interactions. LogFare already added to both Base and Candidate before comparison; equality is not a valid isolated ablation. |

#### Titanic 19

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | AgeBand | age | ['pclass', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'LogFare', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'AgeBand_Adult', 'AgeBand_Child', 'AgeBand_Senior', 'AgeBand_Teen', 'AgeBand_YoungAdult'] | 0.964272 | 0.895552 | 0.904387 | 0.011004 | [0.903258, 0.91673, 0.888762, 0.900394, 0.912789] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | 18 branch: no GenderClass/GenderIsChild/ClassIsChild interactions. |

#### Titanic 20

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 |  | LogFare | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S'] | 0.963079 | 0.900065 | 0.901775 | 0.016322 | [0.891917, 0.919082, 0.878782, 0.906433, 0.912662] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | 18 branch: no GenderClass/GenderIsChild/ClassIsChild interactions. |

#### Titanic 21

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 21 |  | IsAlone | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S'] | 0.964155 | 0.903480 | 0.900862 | 0.018278 | [0.891291, 0.920353, 0.874587, 0.904526, 0.913552] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | 18 branch: no GenderClass/GenderIsChild/ClassIsChild interactions. chosen on holdout despite 19 having higher CV; do not equate branch best with global best. |

#### Titanic 22

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22 | NonFamilyGroupSize |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'NonFamilyGroupSize', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S'] | 0.965556 | 0.899333 | 0.900966 | 0.016310 | [0.890414, 0.919718, 0.879926, 0.90122, 0.913552] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | 18 branch: no GenderClass/GenderIsChild/ClassIsChild interactions. Train-only ticket frequencies differ between fit and validation observations; subtracting FamilySize can be negative. |

#### Titanic 23

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 23 | TicketFarePerPerson |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'TicketFarePerPerson', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S'] | 0.967393 | 0.901935 | 0.900752 | 0.015635 | [0.885401, 0.91654, 0.883613, 0.903636, 0.914569] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | 18 branch: no GenderClass/GenderIsChild/ClassIsChild interactions. Train-only ticket frequencies differ between fit and validation observations; subtracting FamilySize can be negative. |

#### Titanic 24

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | NonFamilyGroupRatio |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'NonFamilyGroupRatio', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S'] | 0.966574 | 0.896975 | 0.900748 | 0.014903 | [0.891228, 0.920099, 0.885139, 0.89461, 0.912662] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | 18 branch: no GenderClass/GenderIsChild/ClassIsChild interactions. Train-only ticket frequencies differ between fit and validation observations; subtracting FamilySize can be negative. |

#### Titanic 25

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | NonFamilyGroupSize,TicketFarePerPerson,NonFamilyGroupRatio |  | ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'NonFamilyGroupSize', 'TicketFarePerPerson', 'NonFamilyGroupRatio', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S'] | 0.967303 | 0.896731 | 0.898545 | 0.014154 | [0.883396, 0.915395, 0.893656, 0.888762, 0.911518] | 미기록 | 미기록 | 미기록 | CatBoost | verbose=0, seed=42, iterations=1000, depth=6, L2=3, random_strength=1, MVS, auto learning_rate; OHE cat_features=[] | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | 18 branch: no GenderClass/GenderIsChild/ClassIsChild interactions. Train-only ticket frequencies differ between fit and validation observations; subtracting FamilySize can be negative. |

#### Titanic 26

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 26 |  |  | ['pclass', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1', 'ClassIsChild_1_0', 'ClassIsChild_1_1', 'ClassIsChild_2_0', 'ClassIsChild_2_1', 'ClassIsChild_3_0', 'ClassIsChild_3_1', 'AgeBand_Adult', 'AgeBand_Child', 'AgeBand_Senior', 'AgeBand_Teen', 'AgeBand_YoungAdult'] | 0.963754 | 0.900390 | 0.909335 | 0.014127 | 미기록 | 미기록 | 미기록 | 미기록 | CatBoost | {'depth': 4, 'l2_leaf_reg': 20} | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Current 32-column age-excluded feature set; comparison baseline CV .907518, not historical Private15 .906275. |

#### Titanic 27

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 |  |  | ['pclass', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1', 'ClassIsChild_1_0', 'ClassIsChild_1_1', 'ClassIsChild_2_0', 'ClassIsChild_2_1', 'ClassIsChild_3_0', 'ClassIsChild_3_1', 'AgeBand_Adult', 'AgeBand_Child', 'AgeBand_Senior', 'AgeBand_Teen', 'AgeBand_YoungAdult'] | 0.973396 | 0.895633 | 0.909254 | 0.015288 | 미기록 | 미기록 | 미기록 | 미기록 | CatBoost | {'random_strength': 0.2, 'subsample': 0.8, 'rsm': 1} | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Current 32-column age-excluded feature set; comparison baseline CV .907518, not historical Private15 .906275. |

#### Titanic 28

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 28 |  |  | ['pclass', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1', 'ClassIsChild_1_0', 'ClassIsChild_1_1', 'ClassIsChild_2_0', 'ClassIsChild_2_1', 'ClassIsChild_3_0', 'ClassIsChild_3_1', 'AgeBand_Adult', 'AgeBand_Child', 'AgeBand_Senior', 'AgeBand_Teen', 'AgeBand_YoungAdult'] | 0.925486 | 0.906367 | 0.914325 | 0.012962 | [0.9087719298245613, 0.9295703025680143, 0.8952453597762523, 0.9195906432748538, 0.9184464785151285] | 0.878288 | 미기록 | 미기록 | CatBoost | {'verbose': 0, 'random_state': 42, 'cat_features': [], 'allow_writing_files': False, 'learning_rate': 0.03, 'iterations': 20, 'loss_function': 'Logloss', 'eval_metric': 'AUC'} | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Outer validation selects early-stop iteration. CV is optimistic for model selection; final full fit uses median 20 |

#### Titanic 29

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29 |  |  | ['pclass', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1', 'ClassIsChild_1_0', 'ClassIsChild_1_1', 'ClassIsChild_2_0', 'ClassIsChild_2_1', 'ClassIsChild_3_0', 'ClassIsChild_3_1', 'AgeBand_Adult', 'AgeBand_Child', 'AgeBand_Senior', 'AgeBand_Teen', 'AgeBand_YoungAdult'] | 0.929900 | 0.898845 | 0.907632 | 0.009901 | [0.9018796992481203, 0.922133231629799, 0.8962623951182305, 0.9061149249936437, 0.9117721840833969] | 0.869942 | 미기록 | 미기록 | CatBoost | {'verbose': 0, 'random_state': 42, 'cat_features': [], 'allow_writing_files': False, 'learning_rate': 0.03, 'depth': 4, 'l2_leaf_reg': 20, 'iterations': 8, 'loss_function': 'Logloss', 'eval_metric': 'AUC'} | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Outer validation selects early-stop iteration. CV is optimistic for model selection; final full fit uses median 8 |

#### Titanic 30

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 |  |  | ['pclass', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1', 'ClassIsChild_1_0', 'ClassIsChild_1_1', 'ClassIsChild_2_0', 'ClassIsChild_2_1', 'ClassIsChild_3_0', 'ClassIsChild_3_1', 'AgeBand_Adult', 'AgeBand_Child', 'AgeBand_Senior', 'AgeBand_Teen', 'AgeBand_YoungAdult'] | 0.926211 | 0.905107 | 0.911666 | 0.013079 | [0.9093358395989976, 0.930142384947877, 0.8934655479277904, 0.9141240783117214, 0.9112636664124076] | 0.886363 | 미기록 | 미기록 | CatBoost | {'verbose': 0, 'random_state': 42, 'cat_features': [], 'allow_writing_files': False, 'learning_rate': 0.03, 'random_strength': 0.2, 'subsample': 0.8, 'rsm': 1.0, 'iterations': 50, 'loss_function': 'Logloss', 'eval_metric': 'AUC'} | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | Outer validation selects early-stop iteration. CV is optimistic for model selection; final full fit uses median 50 |

#### Titanic 31

| Experiment | Added | Removed | Features | Train AUC | Validation AUC | CV Mean | CV Std | Fold AUC | OOF AUC | Private | Public | Model | Parameters | Preprocessing | Missing | Encoding | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 31 |  |  | ['pclass', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'GenderClass_female_1', 'GenderClass_female_2', 'GenderClass_female_3', 'GenderClass_male_1', 'GenderClass_male_2', 'GenderClass_male_3', 'GenderIsChild_female_0', 'GenderIsChild_female_1', 'GenderIsChild_male_0', 'GenderIsChild_male_1', 'ClassIsChild_1_0', 'ClassIsChild_1_1', 'ClassIsChild_2_0', 'ClassIsChild_2_1', 'ClassIsChild_3_0', 'ClassIsChild_3_1', 'AgeBand_Adult', 'AgeBand_Child', 'AgeBand_Senior', 'AgeBand_Teen', 'AgeBand_YoungAdult'] | 0.925486 | 0.906367 | 0.914325 | 0.012962 | [0.9087719298245613, 0.9295703025680143, 0.8952453597762523, 0.9195906432748538, 0.9184464785151285] | 0.878288 | 미기록 | 미기록 | CatBoost | Titanic28, learning_rate=.03, iterations=20 | fold-train fit; cabin>=20% missing dropped; name/ticket raw excluded; no scaling/SMOTE | age gender×pclass median -> global median; other numeric median/categorical mode | fold-fitted OneHotEncoder(handle_unknown=ignore, sparse_output=False) | A combined params and B blend rejected; CSV SHA identical to28. B OOF .889025, weights .5/.3/.2, CV .910230. |

### 노트북별 저장 출력/설정 증거


<details><summary>C:\dev\study\machine_learning\titanic_1 copy.ipynb — stored plots 11, stored errors 1</summary>

실행 오류 기록: ["KeyError: 'age'"]

설정 근거:

~~~python

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
train shape: (916, 12)


test shape: (393, 11)


             model  train_auc  validation_auc
CatBoostClassifier   0.963079        0.900065
    LGBMClassifier   0.994433        0.894088
     XGBClassifier   0.996897        0.888396
선택: CatBoostClassifier Validation AUC: 0.900065051227842


검증: {'test 행 수 일치': True, '원본 shape 유지': True, '컬럼명/순서 유지': True, 'ID 값/순서 유지': True, 'NaN 없음': np.True_, '확률 실수형': True, '확률 0~1 범위': np.True_}
최종 shape: (393, 2)
 passengerid  survived
         916  0.832588
         917  0.906221
         918  0.879633
         919  0.075956
         920  0.962691
저장: submission_result_1.csv

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_1.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
train shape: (916, 12)


test shape: (393, 11)


             model  train_auc  validation_auc
CatBoostClassifier   0.963079        0.900065
    LGBMClassifier   0.994433        0.894088
     XGBClassifier   0.996897        0.888396
선택: CatBoostClassifier Validation AUC: 0.900065051227842


검증: {'test 행 수 일치': True, '원본 shape 유지': True, '컬럼명/순서 유지': True, 'ID 값/순서 유지': True, 'NaN 없음': np.True_, '확률 실수형': True, '확률 0~1 범위': np.True_}
최종 shape: (393, 2)
 passengerid  survived
         916  0.832588
         917  0.906221
         918  0.879633
         919  0.075956
         920  0.962691
저장: submission_result_1.csv

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_10.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.interaction_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission = pd.read_csv('csv/submission.csv')
previous_features = ['GenderClass']
new_feature = 'GenderIsChild'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9046761832116523, 'std': 0.01613237786295489}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
eda_frame['EDA_IsChild'] = (eda_frame['age'] < 16).astype(int)
display(eda_frame.groupby(['gender','EDA_IsChild'])[target_col].agg(['count','mean']))
display(eda_frame.groupby(new_feature,dropna=False)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 24: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
assert len(previous_features) != 1 or abs(previous_scores['validation_auc'] - 0.902911) < 0.000001
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 26: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] >= previous_cv['mean']
            and candidate_cv['std'] <= previous_cv['std'] + 0.005
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 28: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.963047   0.965065
validation_auc           0.900065       0.902911   0.904293
gap                      0.063014       0.060136   0.060771
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_11.ipynb — stored plots 4, stored errors 2</summary>

실행 오류 기록: ['AssertionError: 기존 결과를 덮어쓰지 않습니다.', "NameError: name 'train' is not defined"]

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.interaction_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission = pd.read_csv('csv/submission.csv')
previous_features = ['GenderClass', 'GenderIsChild']
new_feature = 'ClassIsChild'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9059066143619919, 'std': 0.017323969521877203}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
eda_frame['EDA_IsChild'] = (eda_frame['age'] < 16).astype(int)
display(eda_frame.groupby(['pclass','EDA_IsChild'])[target_col].agg(['count','mean']))
display(eda_frame.groupby(new_feature,dropna=False)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 34: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
assert len(previous_features) != 1 or abs(previous_scores['validation_auc'] - 0.902911) < 0.000001
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 36: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] >= previous_cv['mean']
            and candidate_cv['std'] <= previous_cv['std'] + 0.005
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 38: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.965065   0.964795
validation_auc           0.900065       0.904293   0.906489
gap                      0.063014       0.060771   0.058306
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_12.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.interaction_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission = pd.read_csv('csv/submission.csv')
previous_features = ['GenderClass', 'GenderIsChild', 'ClassIsChild']
new_feature = 'GenderIsAlone'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9062327921252408, 'std': 0.018564264845361396}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
display(eda_frame.groupby(['gender','IsAlone'])[target_col].agg(['count','mean']))
display(eda_frame.groupby(new_feature,dropna=False)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 24: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
assert len(previous_features) != 1 or abs(previous_scores['validation_auc'] - 0.902911) < 0.000001
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 26: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] >= previous_cv['mean']
            and candidate_cv['std'] <= previous_cv['std'] + 0.005
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 28: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.964795   0.963471
validation_auc           0.900065       0.906489   0.904619
gap                      0.063014       0.058306   0.058852
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_13.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.interaction_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission = pd.read_csv('csv/submission.csv')
previous_features = ['GenderClass', 'GenderIsChild', 'ClassIsChild']
new_feature = 'EmbarkedClass'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9062327921252408, 'std': 0.018564264845361396}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
display(eda_frame.groupby(['embarked','pclass'])[target_col].agg(['count','mean']))
display(eda_frame.groupby(new_feature,dropna=False)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 24: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
assert len(previous_features) != 1 or abs(previous_scores['validation_auc'] - 0.902911) < 0.000001
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 26: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] >= previous_cv['mean']
            and candidate_cv['std'] <= previous_cv['std'] + 0.005
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 28: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.964795   0.963489
validation_auc           0.900065       0.906489   0.905757
gap                      0.063014       0.058306   0.057732
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_14.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.interaction_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission = pd.read_csv('csv/submission.csv')
previous_features = ['GenderClass', 'GenderIsChild', 'ClassIsChild']
new_feature = 'TicketPrefix'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9062327921252408, 'std': 0.018564264845361396}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
display(X_tr_raw['ticket'].value_counts().head(30))
raw_prefix = eda_prep.ticket_prefix(X_tr_raw)
print(raw_prefix.value_counts().to_string())
display(pd.crosstab(eda_frame[new_feature], eda_frame['pclass']))
display(pd.crosstab(eda_frame[new_feature], eda_frame['gender']))
display(eda_frame.groupby(new_feature,dropna=False)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 24: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
assert len(previous_features) != 1 or abs(previous_scores['validation_auc'] - 0.902911) < 0.000001
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 26: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] >= previous_cv['mean']
            and candidate_cv['std'] <= previous_cv['std'] + 0.005
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 28: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.964795   0.969699
validation_auc           0.900065       0.906489   0.902423
gap                      0.063014       0.058306   0.067276
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_15.ipynb — stored plots 3, stored errors 1</summary>

실행 오류 기록: ['AssertionError: 기존 결과를 덮어쓰지 않습니다.']

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.interaction_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')


submission = pd.read_csv('csv/submission.csv')
previous_features = ['GenderClass', 'GenderIsChild', 'ClassIsChild']
new_feature = 'AgeBand'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9062327921252408, 'std': 0.018564264845361396}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
X_tr_raw['age'].hist(bins=25)
plt.title('Observed age distribution - training split')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
print('Age missing:', X_tr_raw['age'].isna().sum())
display(eda_frame.groupby(new_feature,dropna=False)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 24: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
assert len(previous_features) != 1 or abs(previous_scores['validation_auc'] - 0.902911) < 0.000001
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 26: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] >= previous_cv['mean']
            and candidate_cv['std'] <= previous_cv['std'] + 0.005
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 28: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.964795   0.962336
validation_auc           0.900065       0.906489   0.900024
gap                      0.063014       0.058306   0.062311

C:\Users\Playdata\AppData\Local\Temp\ipykernel_29532\3384964745.py:1: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(shap_values, X_test_model)


TN: 0.909
FP: 0.091
FN: 0.209
TP: 0.791

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_16_delete_age.ipynb — stored plots 2, stored errors 2</summary>

실행 오류 기록: ["ModuleNotFoundError: No module named 'numpy'", 'AssertionError: 기존 결과를 덮어쓰지 않습니다.']

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.interaction_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission = pd.read_csv('csv/submission.csv')
previous_features = ['GenderClass', 'GenderIsChild', 'ClassIsChild']
new_feature = 'AgeBand'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9062327921252408, 'std': 0.018564264845361396}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
X_tr_raw['age'].hist(bins=25)
plt.title('Observed age distribution - training split')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
print('Age missing:', X_tr_raw['age'].isna().sum())
display(eda_frame.groupby(new_feature,dropna=False)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 24: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
assert len(previous_features) != 1 or abs(previous_scores['validation_auc'] - 0.902911) < 0.000001
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 26: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] >= previous_cv['mean']
            and candidate_cv['std'] <= previous_cv['std'] + 0.005
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 28: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.964795   0.965299
validation_auc           0.900065       0.906489   0.903643
gap                      0.063014       0.058306   0.061656

C:\Users\Playdata\AppData\Local\Temp\ipykernel_29532\3384964745.py:1: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(shap_values, X_test_model)

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_16_isnoble.ipynb — stored plots 1, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
Base shape: (687, 12)
Candidate shape: (687, 13)
추가된 모델 입력 열: ['IsNoble']
Base에 유지된 파생 피처: ['FamilySize', 'IsAlone']
범주형 fit 대상: ['gender', 'embarked']
Scaling: 적용하지 않음 (Base Model과 동일)


Fold 1 완료
Fold 2 완료
Fold 3 완료
Fold 4 완료
Fold 5 완료

[5-Fold AUC]
model     Base  Candidate  Difference
fold                                 
1     0.891917   0.888784   -0.003133
2     0.919082   0.911073   -0.008009
3     0.878782   0.879036    0.000254
4     0.906433   0.903000   -0.003432
5     0.912662   0.912026   -0.000636

[CV 요약]
              mean      std
model                      
Base      0.901775 0.016322
Candidate 0.898784 0.014441
Candidate가 개선된 fold: 1/5


        Metric     Base  Candidate  Difference
     Train AUC 0.963079   0.963227    0.000149
Validation AUC 0.900065   0.899821   -0.000244
           Gap 0.063014   0.063406    0.000393
       CV Mean 0.901775   0.898784   -0.002991
        CV Std 0.016322   0.014441   -0.001881
      Accuracy 0.860262   0.864629    0.004367
     Precision 0.829268   0.839506    0.010238
        Recall 0.790698   0.790698    0.000000
            F1 0.809524   0.814371    0.004847

판단 조건: {'holdout_up': False, 'cv_up': np.False_, 'improved_folds': 1, 'std_stable': np.True_, 'gap_stable': True, 'small_sample': True}
최종 판단: 제외


## 실험 결론

### 가설
이름에 포함된 귀족·상류 신분 호칭이 생존 여부에 추가 정보를 제공할 수 있다.

### 결과
- Base Validation AUC: 0.900065
- Candidate Validation AUC: 0.899821
- 변화: -0.000244
- Base CV Mean: 0.901775
- Candidate CV Mean: 0.898784
- 변화: -0.002991
- CV Std: Base 0.016322 / Candidate 0.014441
- Gap: Base 0.063014 / Candidate 0.063406
- 개선된 CV Fold: 1/5

### Confusion Matrix 변화
- FP: 14 → 13 (-1)
- FN: 18 → 18 (+0)
- TP: 68 → 68 (+0)
- TN: 129 → 130 (+1)

### 해석
후보는 Base Model에 `IsNoble` 하나만 추가했다. 같은 split과 같은 CatBoost 실효 파라미터로 비교했으며, 5-Fold CV도 각 fold의 학습 부분에서 전처리와 encoder를 새로 fit했다. 최소 관련 그룹 표본은 3명이다. 표본이 30명보다 작으면 일부 승객의 결과가 그룹 통계와 모델 평가를 크게 움직일 수 있으므로 성능 수치와 함께 신중하게 해석한다.

### 최종 판단
**제외**


~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_16_randomized_search.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 6: space={'model__iterations':randint(500,1501),'model__depth':randint(4,9),'model__learning_rate':loguniform(.005,.08),'model__l2_leaf_reg':loguniform(1.,20.),'model__random_strength':loguniform(.1,5.),'model__subsample':uniform(.65,.30),'model__border_count':[32,64,128,254]}
search=RandomizedSearchCV(pipe(BASE),space,n_iter=30,scoring='roc_auc',cv=StratifiedKFold(5,shuffle=True,random_state=SEED),random_state=SEED,n_jobs=-1,verbose=1,refit=True,return_train_score=True)
search.fit(X,y); print(search.best_params_); print(search.best_score_); print(search.best_index_)
r=pd.DataFrame(search.cv_results_); r['rank']=r.rank_test_score; r['train_auc']=r.mean_train_score; r['cv_auc']=r.mean_test_score; r['cv_std']=r.std_test_score; r['gap']=r.train_auc-r.cv_auc
for p in ['iterations','depth','learning_rate','l2_leaf_reg','random_strength','subsample','border_count']: r[p]=r['param_model__'+p].astype(float)
display(r.sort_values('rank_test_score')[['rank','iterations','depth','learning_rate','l2_leaf_reg','random_strength','subsample','border_count','train_auc','cv_auc','cv_std','gap','params']].head(10))
Path('results').mkdir(exist_ok=True); out=Path('results/titanic_16_randomized_search_results.csv'); assert not out.exists(); r.to_csv(out,index=False)

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
5 0.9145690312738368
Titanic 15 Champion {'train_auc': 0.965299045217078, 'validation_auc': 0.9036428687591479, 'gap': 0.06165617645793009} CV Mean 0.9062745632196434 CV Std 0.016867827959369617 OOF AUC 0.9042465267214278


    rank  iterations  depth  learning_rate  l2_leaf_reg  random_strength  \
22     1       817.0    4.0       0.006971     1.945221         0.374626   
7      2       705.0    6.0       0.008287     3.226871         1.920155   
26     3       938.0    7.0       0.009663     9.792418         1.726703   
14     4       532.0    5.0       0.011842     1.209738         0.356843   
15     5      1046.0    6.0       0.006966     4.114962         1.628477   
19     6      1056.0    5.0       0.006699     2.424534         0.596539   
18     7      1040.0    6.0       0.006190     1.984601         0.310655   
16     8       892.0    6.0       0.006800     8.406084         0.557724   
9      9       769.0    7.0       0.012364     8.834921         0.931466   
1     10       958.0    6.0       0.026472    13.394335         1.595857   

    subsample  border_count  train_auc    cv_auc    cv_std       gap  \
22   0.932873         128.0   0.948592  0.910481  0.015477  0.038111   
7    0.777547         128.0   0.948800  0.909173  0.012931  0.039627   
26   0.760335         254.0   0.954679  0.907221  0.014807  0.047458   
14   0.868882         254.0   0.962758  0.907001  0.018388  0.055757   
15   0.878236         254.0   0.953397  0.906959  0.014956  0.046438   
19   0.715532          64.0   0.959370  0.906691  0.016279  0.052679   
18   0.698366         128.0   0.968869  0.906284  0.017401  0.062584   
16   0.710516          64.0   0.953711  0.905943  0.015852  0.047768   
9    0.806250          64.0   0.963399  0.904716  0.016680  0.058684   
1    0.656175         128.0   0.976864  0.901976  0.019332  0.074888   

                                               params  
22  {'model__border_count': 128, 'model__depth': 4...  
7   {'model__border_count': 128, 'model__depth': 6...  
26  {'model__border_count': 254, 'model__depth': 7...  
14  {'model__border_count': 254, 'model__depth': 5...  
15  {'model__border_count': 254, 'model__depth': 6...  
19  {'model__border_count': 64, 'model__depth': 5,...  
18  {'model__border_count': 128, 'model__depth': 6...  
16  {'model__border_count': 64, 'model__depth': 6,...  
9   {'model__border_count': 64, 'model__depth': 7,...  
1   {'model__border_count': 128, 'model__depth': 6...  

   fold  Champion AUC  Tuned AUC     Delta
0     1      0.901065   0.897243 -0.003822
1     2      0.923786   0.935100  0.011315
2     3      0.879799   0.890987  0.011187
3     4      0.912154   0.913234  0.001081
4     5      0.914569   0.915840  0.001271

           Metric  Titanic 15  Titanic 16     Delta
0       Train AUC    0.965299    0.952909 -0.012390
1  Validation AUC    0.903643    0.900065 -0.003578
2             Gap    0.061656    0.052844 -0.008812
3         CV Mean    0.906275    0.910481  0.004206
4          CV Std    0.016868    0.017304  0.000436
5         OOF AUC    0.904247    0.907781  0.003534
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_17_profession_group.ipynb — stored plots 1, stored errors 1</summary>

실행 오류 기록: ["ValueError: could not convert string to float: 'Aks, Master. Philip Frank'"]

설정 근거:

~~~python

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
Base shape: (687, 12)
Candidate shape: (687, 16)
추가된 모델 입력 열: ['ProfessionGroup_Clergy', 'ProfessionGroup_Medical', 'ProfessionGroup_Military', 'ProfessionGroup_Other']
Base에 유지된 파생 피처: ['FamilySize', 'IsAlone']
범주형 fit 대상: ['gender', 'embarked', 'ProfessionGroup']
Scaling: 적용하지 않음 (Base Model과 동일)


Fold 1 완료
Fold 2 완료
Fold 3 완료
Fold 4 완료
Fold 5 완료

[5-Fold AUC]
model     Base  Candidate  Difference
fold                                 
1     0.891917   0.890664   -0.001253
2     0.919082   0.917684   -0.001398
3     0.878782   0.876112   -0.002670
4     0.906433   0.907958    0.001526
5     0.912662   0.910628   -0.002034

[CV 요약]
              mean      std
model                      
Base      0.901775 0.016322
Candidate 0.900609 0.016918
Candidate가 개선된 fold: 1/5


        Metric     Base  Candidate  Difference
     Train AUC 0.963079   0.962182   -0.000896
Validation AUC 0.900065   0.899496   -0.000569
           Gap 0.063014   0.062687   -0.000327
       CV Mean 0.901775   0.900609   -0.001166
        CV Std 0.016322   0.016918    0.000597
      Accuracy 0.860262   0.864629    0.004367
     Precision 0.829268   0.839506    0.010238
        Recall 0.790698   0.790698    0.000000
            F1 0.809524   0.814371    0.004847

판단 조건: {'holdout_up': False, 'cv_up': np.False_, 'improved_folds': 1, 'std_stable': np.True_, 'gap_stable': True, 'small_sample': True}
최종 판단: 제외


## 실험 결론

### 가설
이름에서 추출한 직업·사회적 역할 호칭이 생존 여부에 추가 정보를 제공할 수 있다.

### 결과
- Base Validation AUC: 0.900065
- Candidate Validation AUC: 0.899496
- 변화: -0.000569
- Base CV Mean: 0.901775
- Candidate CV Mean: 0.900609
- 변화: -0.001166
- CV Std: Base 0.016322 / Candidate 0.016918
- Gap: Base 0.063014 / Candidate 0.062687
- 개선된 CV Fold: 1/5

### Confusion Matrix 변화
- FP: 14 → 13 (-1)
- FN: 18 → 18 (+0)
- TP: 68 → 68 (+0)
- TN: 129 → 130 (+1)

### 해석
후보는 Base Model에 `ProfessionGroup` 하나만 추가했다. 같은 split과 같은 CatBoost 실효 파라미터로 비교했으며, 5-Fold CV도 각 fold의 학습 부분에서 전처리와 encoder를 새로 fit했다. 최소 관련 그룹 표본은 4명이다. 표본이 30명보다 작으면 일부 승객의 결과가 그룹 통계와 모델 평가를 크게 움직일 수 있으므로 성능 수치와 함께 신중하게 해석한다.

### 최종 판단
**제외**


~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_18.ipynb — stored plots 1, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
Base Train AUC: 0.965128
Base Validation AUC: 0.900309
Candidate Train AUC: 0.965128
Candidate Validation AUC: 0.900309


Fold 1 완료
Fold 2 완료
Fold 3 완료
Fold 4 완료
Fold 5 완료

[5-Fold AUC]
model     Base  Candidate  Difference
fold                                 
1     0.889724   0.889724    0.000000
2     0.913107   0.913107    0.000000
3     0.875222   0.875222    0.000000
4     0.904144   0.904144    0.000000
5     0.909484   0.909484    0.000000

[CV 요약]
              mean      std
model                      
Base      0.898336 0.015689
Candidate 0.898336 0.015689


        Metric     Base  Candidate  Difference
     Train AUC 0.965128   0.965128    0.000000
Validation AUC 0.900309   0.900309    0.000000
           Gap 0.064819   0.064819    0.000000
       CV Mean 0.898336   0.898336    0.000000
        CV Std 0.015689   0.015689    0.000000

Previous Best Validation AUC: 0.903643
Titanic 18 Validation AUC: 0.900309
Difference: -0.003334

Previous Best CV Mean: 0.906275
Titanic 18 CV Mean: 0.898336
Difference: -0.007938


전체 train/test 변환 검증 완료: (916, 13) (393, 13)
생성 파일: c:\dev\study\machine_learning\submission\titanic_result_18.csv
result shape: (393, 2)
 passengerid  survived
         916  0.829912
         917  0.895525
         918  0.883352
         919  0.079155
         920  0.956514



Experiment: Titanic 18
Added Feature: LogFare
Removed Feature: 없음
Fare: 유지
Train AUC: 0.965128
Validation AUC: 0.900309
Gap: 0.064819
CV Mean: 0.898336
CV Std: 0.015689
결론: 보류
생성 파일:
- titanic_18.ipynb
- submission/titanic_result_18.csv


C:\Users\Playdata\AppData\Local\Temp\ipykernel_32428\3632369571.py:2: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(shap_values, X_test)

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_19.ipynb — stored plots 2, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 10: class ExperimentPreprocessor(TitanicPreprocessor):
    """Titanic 18 설정에서 지정된 단일 변경만 적용하는 leakage-safe 전처리."""

    def __init__(self, *, use_logfare, use_ageband, remove_isalone, ticket_feature):
        self.use_logfare = use_logfare
        self.use_ageband = use_ageband
        self.remove_isalone = remove_isalone
        self.ticket_feature = ticket_feature

    def fit_missing(self, X):
        super().fit_missing(X)
        if self.ticket_feature is not None:
            # 기존 TicketGroupSize 방식: fit 표본의 ticket 빈도만 저장한다.
            self.ticket_counts_ = X["ticket"].value_counts()
        return self

    def feature_frame(self, raw, clean):
        data = self.make_features(clean)
        if self.use_logfare:
            data["LogFare"] = np.log1p(data["fare"])
        if self.use_ageband:
            # 기존 프로젝트 정의: [0,16), [16,20), [20,35), [35,60), [60,inf)
            data["AgeBand"] = pd.cut(
                data["age"], [0, 16, 20, 35, 60, np.inf], right=False,
                labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
            ).astype(str)
            data = data.drop(columns="age")
        if self.remove_isalone:
            data = data.drop(columns="IsAlone")
        if self.ticket_feature is not None:
            ticket_size = raw["ticket"].map(self.ticket_counts_).fillna(1).clip(lower=1).astype(float)
            if self.ticket_feature == "NonFamilyGroupSize":
                data["NonFamilyGroupSize"] = ticket_size - data["FamilySize"]
            elif self.ticket_feature == "TicketFarePerPerson":
                data["TicketFarePerPerson"] = data["fare"] / ticket_size
            elif self.ticket_feature == "NonFamilyGroupRatio":
                non_family_size = ticket_size - data["FamilySize"]
                data["NonFamilyGroupRatio"] = non_family_size / ticket_size
            else:
                raise ValueError(f"지원하지 않는 Ticket feature: {self.ticket_feature}")
        return self.select_features(data)

    def fit_transform(self, X):
        self.fit_missing(X)
        data = self.feature_frame(X, self.transform_missing(X))
        self.fit_encoding(data)
        return self.transform_encoding(data)

    def transform(self, X):
        data = self.feature_frame(X, self.transform_missing(X))
        return self.transform_encoding(data)

experiment_prep = ExperimentPreprocessor(
    use_logfare=True,
    use_ageband=True,
    remove_isalone=False,
    ticket_feature=None,
)
experiment_prep.fit_missing(X_tr_raw)
X_tr_clean = experiment_prep.transform_missing(X_tr_raw)
X_valid_clean = experiment_prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()
print("제외 결측 컬럼:", experiment_prep.drop_missing_)
print("age 그룹 중앙값:\n", experiment_prep.age_groups_.to_string())

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
범주형: ['gender', 'embarked', 'AgeBand']
인코딩 후: (687, 17) (229, 17)
Encoded Feature 목록: ['pclass', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'LogFare', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'AgeBand_Adult', 'AgeBand_Child', 'AgeBand_Senior', 'AgeBand_Teen', 'AgeBand_YoungAdult']


Train AUC: 0.964272
Validation AUC: 0.895552
Train / Validation Gap: 0.068720


전체 train/test 변환: (916, 17) (393, 17)
최종 Feature 목록: ['pclass', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'LogFare', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'AgeBand_Adult', 'AgeBand_Child', 'AgeBand_Senior', 'AgeBand_Teen', 'AgeBand_YoungAdult']


저장: c:\dev\study\machine_learning\submission\titanic_19_result.csv
검증: {'test 행 수 일치': True, '원본 shape 유지': True, '컬럼명/순서 유지': True, 'ID 값/순서 유지': True, 'NaN 없음': np.True_, '확률 0~1 범위': np.True_}


Experiment: Titanic 19
Change: Age 제거 / AgeBand only
Train AUC: 0.964272
Validation AUC: 0.895552
Gap: 0.068720
CV Mean: 0.904387
CV Std: 0.011004
Feature 목록: ['pclass', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'LogFare', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'AgeBand_Adult', 'AgeBand_Child', 'AgeBand_Senior', 'AgeBand_Teen', 'AgeBand_YoungAdult']
Submission: c:\dev\study\machine_learning\submission\titanic_19_result.csv


TN: 0.888
FP: 0.112
FN: 0.209
TP: 0.791

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_2.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 15: # Baseline 처리 흐름을 그대로 확인합니다.
# 실험용 클래스는 원본 마스크 하나만 추가하고 나머지는 상속해 재사용합니다.
class ExperimentPreprocessor(TitanicPreprocessor):
    def __init__(self, variant='Baseline'):
        self.variant = variant

    def transform_missing(self, X):
        # 원본 X는 변경하지 않으므로 삭제/치환 이전 결측 여부를 읽을 수 있습니다.
        data = super().transform_missing(X)
        if self.variant == 'HasCabin':
            data['HasCabin'] = X['cabin'].notna().astype(int)
        elif self.variant == 'AgeMissing':
            data['AgeMissing'] = X['age'].isna().astype(int)
        elif self.variant != 'Baseline':
            raise ValueError('알 수 없는 실험 이름입니다.')
        return data

prep = ExperimentPreprocessor('Baseline').fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
print('학습 결측 비율:', prep.missing_ratio_.to_dict())
print('나이 그룹 중앙값:', prep.age_groups_.to_dict())
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
train shape: (916, 12)
test shape: (393, 11)


            train_auc  validation_auc       gap     delta
experiment                                               
Baseline     0.963079        0.900065  0.063014  0.000000
HasCabin     0.964623        0.898439  0.066185 -0.001626
AgeMissing   0.964178        0.899943  0.064235 -0.000122

            train_auc  validation_auc       gap     delta
experiment                                               
Baseline     0.963079        0.900065  0.063014  0.000000
HasCabin     0.964623        0.898439  0.066185 -0.001626
AgeMissing   0.964178        0.899943  0.064235 -0.000122
5-fold 검토 후보: Baseline


            train_auc  validation_auc       gap     delta decision
experiment                                                        
Baseline     0.963079        0.900065  0.063014  0.000000    최종 채택
HasCabin     0.964623        0.898439  0.066185 -0.001626    최종 제외
AgeMissing   0.964178        0.899943  0.064235 -0.000122    최종 제외

Fold 1 Baseline AUC: 0.8919172932330827
Fold 2 Baseline AUC: 0.9190821256038647
Fold 3 Baseline AUC: 0.8787821001779812
Fold 4 Baseline AUC: 0.9064327485380117
Fold 5 Baseline AUC: 0.9126620900076278
                mean       std
experiment                    
Baseline    0.901775  0.016322
최종 채택: Baseline


검증: {'파일 존재': True, 'test 행수 일치': True, '템플릿 shape 유지': True, '컬럼 순서 유지': True, 'ID 및 행 순서 유지': True, 'NaN 없음': np.True_, '실수형 확률': True, '확률 범위': np.True_}
최종 shape: (393, 2)

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_20.ipynb — stored plots 2, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 10: class ExperimentPreprocessor(TitanicPreprocessor):
    """Titanic 18 설정에서 지정된 단일 변경만 적용하는 leakage-safe 전처리."""

    def __init__(self, *, use_logfare, use_ageband, remove_isalone, ticket_feature):
        self.use_logfare = use_logfare
        self.use_ageband = use_ageband
        self.remove_isalone = remove_isalone
        self.ticket_feature = ticket_feature

    def fit_missing(self, X):
        super().fit_missing(X)
        if self.ticket_feature is not None:
            # 기존 TicketGroupSize 방식: fit 표본의 ticket 빈도만 저장한다.
            self.ticket_counts_ = X["ticket"].value_counts()
        return self

    def feature_frame(self, raw, clean):
        data = self.make_features(clean)
        if self.use_logfare:
            data["LogFare"] = np.log1p(data["fare"])
        if self.use_ageband:
            # 기존 프로젝트 정의: [0,16), [16,20), [20,35), [35,60), [60,inf)
            data["AgeBand"] = pd.cut(
                data["age"], [0, 16, 20, 35, 60, np.inf], right=False,
                labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
            ).astype(str)
            data = data.drop(columns="age")
        if self.remove_isalone:
            data = data.drop(columns="IsAlone")
        if self.ticket_feature is not None:
            ticket_size = raw["ticket"].map(self.ticket_counts_).fillna(1).clip(lower=1).astype(float)
            if self.ticket_feature == "NonFamilyGroupSize":
                data["NonFamilyGroupSize"] = ticket_size - data["FamilySize"]
            elif self.ticket_feature == "TicketFarePerPerson":
                data["TicketFarePerPerson"] = data["fare"] / ticket_size
            elif self.ticket_feature == "NonFamilyGroupRatio":
                non_family_size = ticket_size - data["FamilySize"]
                data["NonFamilyGroupRatio"] = non_family_size / ticket_size
            else:
                raise ValueError(f"지원하지 않는 Ticket feature: {self.ticket_feature}")
        return self.select_features(data)

    def fit_transform(self, X):
        self.fit_missing(X)
        data = self.feature_frame(X, self.transform_missing(X))
        self.fit_encoding(data)
        return self.transform_encoding(data)

    def transform(self, X):
        data = self.feature_frame(X, self.transform_missing(X))
        return self.transform_encoding(data)

experiment_prep = ExperimentPreprocessor(
    use_logfare=False,
    use_ageband=False,
    remove_isalone=False,
    ticket_feature=None,
)
experiment_prep.fit_missing(X_tr_raw)
X_tr_clean = experiment_prep.transform_missing(X_tr_raw)
X_valid_clean = experiment_prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()
print("제외 결측 컬럼:", experiment_prep.drop_missing_)
print("age 그룹 중앙값:\n", experiment_prep.age_groups_.to_string())

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
범주형: ['gender', 'embarked']
인코딩 후: (687, 12) (229, 12)
Encoded Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']


Train AUC: 0.963079
Validation AUC: 0.900065
Train / Validation Gap: 0.063014


전체 train/test 변환: (916, 12) (393, 12)
최종 Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']


저장: c:\dev\study\machine_learning\submission\titanic_20_result.csv
검증: {'test 행 수 일치': True, '원본 shape 유지': True, '컬럼명/순서 유지': True, 'ID 값/순서 유지': True, 'NaN 없음': np.True_, '확률 0~1 범위': np.True_}


Experiment: Titanic 20
Change: LogFare 제거 / Fare only
Train AUC: 0.963079
Validation AUC: 0.900065
Gap: 0.063014
CV Mean: 0.901775
CV Std: 0.016322
Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']
Submission: c:\dev\study\machine_learning\submission\titanic_20_result.csv


TN: 0.902
FP: 0.098
FN: 0.209
TP: 0.791

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_21.ipynb — stored plots 3, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 10: class ExperimentPreprocessor(TitanicPreprocessor):
    """Titanic 18 설정에서 지정된 단일 변경만 적용하는 leakage-safe 전처리."""

    def __init__(self, *, use_logfare, use_ageband, remove_isalone, ticket_feature):
        self.use_logfare = use_logfare
        self.use_ageband = use_ageband
        self.remove_isalone = remove_isalone
        self.ticket_feature = ticket_feature

    def fit_missing(self, X):
        super().fit_missing(X)
        if self.ticket_feature is not None:
            # 기존 TicketGroupSize 방식: fit 표본의 ticket 빈도만 저장한다.
            self.ticket_counts_ = X["ticket"].value_counts()
        return self

    def feature_frame(self, raw, clean):
        data = self.make_features(clean)
        if self.use_logfare:
            data["LogFare"] = np.log1p(data["fare"])
        if self.use_ageband:
            # 기존 프로젝트 정의: [0,16), [16,20), [20,35), [35,60), [60,inf)
            data["AgeBand"] = pd.cut(
                data["age"], [0, 16, 20, 35, 60, np.inf], right=False,
                labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
            ).astype(str)
            data = data.drop(columns="age")
        if self.remove_isalone:
            data = data.drop(columns="IsAlone")
        if self.ticket_feature is not None:
            ticket_size = raw["ticket"].map(self.ticket_counts_).fillna(1).clip(lower=1).astype(float)
            if self.ticket_feature == "NonFamilyGroupSize":
                data["NonFamilyGroupSize"] = ticket_size - data["FamilySize"]
            elif self.ticket_feature == "TicketFarePerPerson":
                data["TicketFarePerPerson"] = data["fare"] / ticket_size
            elif self.ticket_feature == "NonFamilyGroupRatio":
                non_family_size = ticket_size - data["FamilySize"]
                data["NonFamilyGroupRatio"] = non_family_size / ticket_size
            else:
                raise ValueError(f"지원하지 않는 Ticket feature: {self.ticket_feature}")
        return self.select_features(data)

    def fit_transform(self, X):
        self.fit_missing(X)
        data = self.feature_frame(X, self.transform_missing(X))
        self.fit_encoding(data)
        return self.transform_encoding(data)

    def transform(self, X):
        data = self.feature_frame(X, self.transform_missing(X))
        return self.transform_encoding(data)

experiment_prep = ExperimentPreprocessor(
    use_logfare=True,
    use_ageband=False,
    remove_isalone=True,
    ticket_feature=None,
)
experiment_prep.fit_missing(X_tr_raw)
X_tr_clean = experiment_prep.transform_missing(X_tr_raw)
X_valid_clean = experiment_prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()
print("제외 결측 컬럼:", experiment_prep.drop_missing_)
print("age 그룹 중앙값:\n", experiment_prep.age_groups_.to_string())

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
범주형: ['gender', 'embarked']
인코딩 후: (687, 12) (229, 12)
Encoded Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']


Train AUC: 0.964155
Validation AUC: 0.903480
Train / Validation Gap: 0.060675


      Version                  Change  Train AUC   Val AUC       Gap  \
0  Titanic 18          Fare + LogFare   0.965128  0.900309  0.064819   
1  Titanic 19   Age 제거 / AgeBand only   0.964272  0.895552  0.068720   
2  Titanic 20  LogFare 제거 / Fare only   0.963079  0.900065  0.063014   
3  Titanic 21              IsAlone 제거   0.964155  0.903480  0.060675   

    CV Mean    CV Std  
0  0.898336  0.015689  
1  0.904387  0.011004  
2  0.901775  0.016322  
3  0.900862  0.018278  

전체 train/test 변환: (916, 12) (393, 12)
최종 Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']


저장: c:\dev\study\machine_learning\submission\titanic_21_result.csv
검증: {'test 행 수 일치': True, '원본 shape 유지': True, '컬럼명/순서 유지': True, 'ID 값/순서 유지': True, 'NaN 없음': np.True_, '확률 0~1 범위': np.True_}


Experiment: Titanic 21
Change: IsAlone 제거
Train AUC: 0.964155
Validation AUC: 0.903480
Gap: 0.060675
CV Mean: 0.900862
CV Std: 0.018278
Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']
Submission: c:\dev\study\machine_learning\submission\titanic_21_result.csv


TN: 0.895
FP: 0.105
FN: 0.209
TP: 0.791


C:\Users\Playdata\AppData\Local\Temp\ipykernel_31360\2747367835.py:7: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(shap_values, X_test_model)

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_22.ipynb — stored plots 2, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 10: class ExperimentPreprocessor(TitanicPreprocessor):
    """Titanic 18 설정에서 지정된 단일 변경만 적용하는 leakage-safe 전처리."""

    def __init__(self, *, use_logfare, use_ageband, remove_isalone, ticket_feature):
        self.use_logfare = use_logfare
        self.use_ageband = use_ageband
        self.remove_isalone = remove_isalone
        self.ticket_feature = ticket_feature

    def fit_missing(self, X):
        super().fit_missing(X)
        if self.ticket_feature is not None:
            # 기존 TicketGroupSize 방식: fit 표본의 ticket 빈도만 저장한다.
            self.ticket_counts_ = X["ticket"].value_counts()
        return self

    def feature_frame(self, raw, clean):
        data = self.make_features(clean)
        if self.use_logfare:
            data["LogFare"] = np.log1p(data["fare"])
        if self.use_ageband:
            # 기존 프로젝트 정의: [0,16), [16,20), [20,35), [35,60), [60,inf)
            data["AgeBand"] = pd.cut(
                data["age"], [0, 16, 20, 35, 60, np.inf], right=False,
                labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
            ).astype(str)
            data = data.drop(columns="age")
        if self.remove_isalone:
            data = data.drop(columns="IsAlone")
        if self.ticket_feature is not None:
            ticket_size = raw["ticket"].map(self.ticket_counts_).fillna(1).clip(lower=1).astype(float)
            if self.ticket_feature == "NonFamilyGroupSize":
                data["NonFamilyGroupSize"] = ticket_size - data["FamilySize"]
            elif self.ticket_feature == "TicketFarePerPerson":
                data["TicketFarePerPerson"] = data["fare"] / ticket_size
            elif self.ticket_feature == "NonFamilyGroupRatio":
                non_family_size = ticket_size - data["FamilySize"]
                data["NonFamilyGroupRatio"] = non_family_size / ticket_size
            else:
                raise ValueError(f"지원하지 않는 Ticket feature: {self.ticket_feature}")
        return self.select_features(data)

    def fit_transform(self, X):
        self.fit_missing(X)
        data = self.feature_frame(X, self.transform_missing(X))
        self.fit_encoding(data)
        return self.transform_encoding(data)

    def transform(self, X):
        data = self.feature_frame(X, self.transform_missing(X))
        return self.transform_encoding(data)

experiment_prep = ExperimentPreprocessor(
    use_logfare=True,
    use_ageband=False,
    remove_isalone=True,
    ticket_feature='NonFamilyGroupSize',
)
experiment_prep.fit_missing(X_tr_raw)
X_tr_clean = experiment_prep.transform_missing(X_tr_raw)
X_valid_clean = experiment_prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()
print("제외 결측 컬럼:", experiment_prep.drop_missing_)
print("age 그룹 중앙값:\n", experiment_prep.age_groups_.to_string())

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
범주형: ['gender', 'embarked']
인코딩 후: (687, 13) (229, 13)
Encoded Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'NonFamilyGroupSize', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']


Train AUC: 0.965556
Validation AUC: 0.899333
Train / Validation Gap: 0.066223


전체 train/test 변환: (916, 13) (393, 13)
최종 Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'NonFamilyGroupSize', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']


저장: c:\dev\study\machine_learning\submission\titanic_22_result.csv
검증: {'test 행 수 일치': True, '원본 shape 유지': True, '컬럼명/순서 유지': True, 'ID 값/순서 유지': True, 'NaN 없음': np.True_, '확률 0~1 범위': np.True_}


Experiment: Titanic 22
Change: Titanic 21 + NonFamilyGroupSize
Train AUC: 0.965556
Validation AUC: 0.899333
Gap: 0.066223
CV Mean: 0.900966
CV Std: 0.016310
Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'NonFamilyGroupSize', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']
Submission: c:\dev\study\machine_learning\submission\titanic_22_result.csv


TN: 0.867
FP: 0.133
FN: 0.209
TP: 0.791

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_23.ipynb — stored plots 2, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 10: class ExperimentPreprocessor(TitanicPreprocessor):
    """Titanic 18 설정에서 지정된 단일 변경만 적용하는 leakage-safe 전처리."""

    def __init__(self, *, use_logfare, use_ageband, remove_isalone, ticket_feature):
        self.use_logfare = use_logfare
        self.use_ageband = use_ageband
        self.remove_isalone = remove_isalone
        self.ticket_feature = ticket_feature

    def fit_missing(self, X):
        super().fit_missing(X)
        if self.ticket_feature is not None:
            # 기존 TicketGroupSize 방식: fit 표본의 ticket 빈도만 저장한다.
            self.ticket_counts_ = X["ticket"].value_counts()
        return self

    def feature_frame(self, raw, clean):
        data = self.make_features(clean)
        if self.use_logfare:
            data["LogFare"] = np.log1p(data["fare"])
        if self.use_ageband:
            # 기존 프로젝트 정의: [0,16), [16,20), [20,35), [35,60), [60,inf)
            data["AgeBand"] = pd.cut(
                data["age"], [0, 16, 20, 35, 60, np.inf], right=False,
                labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
            ).astype(str)
            data = data.drop(columns="age")
        if self.remove_isalone:
            data = data.drop(columns="IsAlone")
        if self.ticket_feature is not None:
            ticket_size = raw["ticket"].map(self.ticket_counts_).fillna(1).clip(lower=1).astype(float)
            if self.ticket_feature == "NonFamilyGroupSize":
                data["NonFamilyGroupSize"] = ticket_size - data["FamilySize"]
            elif self.ticket_feature == "TicketFarePerPerson":
                data["TicketFarePerPerson"] = data["fare"] / ticket_size
            elif self.ticket_feature == "NonFamilyGroupRatio":
                non_family_size = ticket_size - data["FamilySize"]
                data["NonFamilyGroupRatio"] = non_family_size / ticket_size
            else:
                raise ValueError(f"지원하지 않는 Ticket feature: {self.ticket_feature}")
        return self.select_features(data)

    def fit_transform(self, X):
        self.fit_missing(X)
        data = self.feature_frame(X, self.transform_missing(X))
        self.fit_encoding(data)
        return self.transform_encoding(data)

    def transform(self, X):
        data = self.feature_frame(X, self.transform_missing(X))
        return self.transform_encoding(data)

experiment_prep = ExperimentPreprocessor(
    use_logfare=True,
    use_ageband=False,
    remove_isalone=True,
    ticket_feature='TicketFarePerPerson',
)
experiment_prep.fit_missing(X_tr_raw)
X_tr_clean = experiment_prep.transform_missing(X_tr_raw)
X_valid_clean = experiment_prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()
print("제외 결측 컬럼:", experiment_prep.drop_missing_)
print("age 그룹 중앙값:\n", experiment_prep.age_groups_.to_string())

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
범주형: ['gender', 'embarked']
인코딩 후: (687, 13) (229, 13)
Encoded Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'TicketFarePerPerson', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']


Train AUC: 0.967393
Validation AUC: 0.901935
Train / Validation Gap: 0.065458


전체 train/test 변환: (916, 13) (393, 13)
최종 Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'TicketFarePerPerson', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']


저장: c:\dev\study\machine_learning\submission\titanic_23_result.csv
검증: {'test 행 수 일치': True, '원본 shape 유지': True, '컬럼명/순서 유지': True, 'ID 값/순서 유지': True, 'NaN 없음': np.True_, '확률 0~1 범위': np.True_}


Experiment: Titanic 23
Change: Titanic 21 + TicketFarePerPerson
Train AUC: 0.967393
Validation AUC: 0.901935
Gap: 0.065458
CV Mean: 0.900752
CV Std: 0.015635
Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'TicketFarePerPerson', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']
Submission: c:\dev\study\machine_learning\submission\titanic_23_result.csv


TN: 0.860
FP: 0.140
FN: 0.209
TP: 0.791

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_24.ipynb — stored plots 3, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 10: class ExperimentPreprocessor(TitanicPreprocessor):
    """Titanic 18 설정에서 지정된 단일 변경만 적용하는 leakage-safe 전처리."""

    def __init__(self, *, use_logfare, use_ageband, remove_isalone, ticket_feature):
        self.use_logfare = use_logfare
        self.use_ageband = use_ageband
        self.remove_isalone = remove_isalone
        self.ticket_feature = ticket_feature

    def fit_missing(self, X):
        super().fit_missing(X)
        if self.ticket_feature is not None:
            # 기존 TicketGroupSize 방식: fit 표본의 ticket 빈도만 저장한다.
            self.ticket_counts_ = X["ticket"].value_counts()
        return self

    def feature_frame(self, raw, clean):
        data = self.make_features(clean)
        if self.use_logfare:
            data["LogFare"] = np.log1p(data["fare"])
        if self.use_ageband:
            # 기존 프로젝트 정의: [0,16), [16,20), [20,35), [35,60), [60,inf)
            data["AgeBand"] = pd.cut(
                data["age"], [0, 16, 20, 35, 60, np.inf], right=False,
                labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
            ).astype(str)
            data = data.drop(columns="age")
        if self.remove_isalone:
            data = data.drop(columns="IsAlone")
        if self.ticket_feature is not None:
            ticket_size = raw["ticket"].map(self.ticket_counts_).fillna(1).clip(lower=1).astype(float)
            if self.ticket_feature == "NonFamilyGroupSize":
                data["NonFamilyGroupSize"] = ticket_size - data["FamilySize"]
            elif self.ticket_feature == "TicketFarePerPerson":
                data["TicketFarePerPerson"] = data["fare"] / ticket_size
            elif self.ticket_feature == "NonFamilyGroupRatio":
                non_family_size = ticket_size - data["FamilySize"]
                data["NonFamilyGroupRatio"] = non_family_size / ticket_size
            else:
                raise ValueError(f"지원하지 않는 Ticket feature: {self.ticket_feature}")
        return self.select_features(data)

    def fit_transform(self, X):
        self.fit_missing(X)
        data = self.feature_frame(X, self.transform_missing(X))
        self.fit_encoding(data)
        return self.transform_encoding(data)

    def transform(self, X):
        data = self.feature_frame(X, self.transform_missing(X))
        return self.transform_encoding(data)

experiment_prep = ExperimentPreprocessor(
    use_logfare=True,
    use_ageband=False,
    remove_isalone=True,
    ticket_feature='NonFamilyGroupRatio',
)
experiment_prep.fit_missing(X_tr_raw)
X_tr_clean = experiment_prep.transform_missing(X_tr_raw)
X_valid_clean = experiment_prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()
print("제외 결측 컬럼:", experiment_prep.drop_missing_)
print("age 그룹 중앙값:\n", experiment_prep.age_groups_.to_string())

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
범주형: ['gender', 'embarked']
인코딩 후: (687, 13) (229, 13)
Encoded Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'NonFamilyGroupRatio', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']


Train AUC: 0.966574
Validation AUC: 0.896975
Train / Validation Gap: 0.069598


전체 train/test 변환: (916, 13) (393, 13)
최종 Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'NonFamilyGroupRatio', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']


저장: c:\dev\study\machine_learning\submission\titanic_24_result.csv
검증: {'test 행 수 일치': True, '원본 shape 유지': True, '컬럼명/순서 유지': True, 'ID 값/순서 유지': True, 'NaN 없음': np.True_, '확률 0~1 범위': np.True_}


Experiment: Titanic 24
Change: Titanic 21 + NonFamilyGroupRatio
Train AUC: 0.966574
Validation AUC: 0.896975
Gap: 0.069598
CV Mean: 0.900748
CV Std: 0.014903
Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'NonFamilyGroupRatio', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']
Submission: c:\dev\study\machine_learning\submission\titanic_24_result.csv


TN: 0.881
FP: 0.119
FN: 0.209
TP: 0.791


c:\dev\study\machine_learning\.venv\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
C:\Users\Playdata\AppData\Local\Temp\ipykernel_27288\2747367835.py:7: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(shap_values, X_test_model)

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_25.ipynb — stored plots 4, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 2: from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold

from common.evaluations import get_auc_score
from common.feature_experiments import fit_cat, baseline_parameters
from common.preprocessing import TitanicPreprocessor
from common.utils import train_test_split_by_target

ROOT = Path.cwd()
train = pd.read_csv(ROOT / "csv" / "train.csv")
test = pd.read_csv(ROOT / "csv" / "test.csv")
submission = pd.read_csv(ROOT / "csv" / "submission.csv")
ADDED_FEATURES = ["NonFamilyGroupSize", "TicketFarePerPerson", "NonFamilyGroupRatio"]

print("train:", train.shape, "test:", test.shape, "submission:", submission.shape)
display(train.head())
display(test.head())


cell 10: class Titanic25Preprocessor(TitanicPreprocessor):
    """Titanic 21에 기존 22~24 Ticket Feature 세 개만 동시에 추가한다."""

    def fit_missing(self, X):
        super().fit_missing(X)
        self.ticket_counts_ = X["ticket"].value_counts()
        return self

    def feature_frame(self, raw, clean):
        data = self.make_features(clean)
        # Titanic 21: LogFare 유지, IsAlone 제거
        data["LogFare"] = np.log1p(data["fare"])
        data = data.drop(columns="IsAlone")

        # Titanic 22~24와 동일한 leakage-safe TicketGroupSize 정의
        ticket_group_size = raw["ticket"].map(self.ticket_counts_).fillna(1).clip(lower=1).astype(float)
        non_family_group_size = ticket_group_size - data["FamilySize"]

        data["NonFamilyGroupSize"] = non_family_group_size
        data["TicketFarePerPerson"] = data["fare"] / ticket_group_size
        data["NonFamilyGroupRatio"] = non_family_group_size / ticket_group_size
        # TicketGroupSize는 중간 계산값이며 모델 Feature로 추가하지 않는다.
        return self.select_features(data)

    def fit_transform(self, X):
        self.fit_missing(X)
        data = self.feature_frame(X, self.transform_missing(X))
        self.fit_encoding(data)
        return self.transform_encoding(data)

    def transform(self, X):
        data = self.feature_frame(X, self.transform_missing(X))
        return self.transform_encoding(data)


def feature_diagnostics(label, raw, features, prep):
    ticket_group_size = raw["ticket"].map(prep.ticket_counts_).fillna(1).clip(lower=1).astype(float)
    checks = {
        "TicketGroupSize_zero": int((ticket_group_size == 0).sum()),
        "NonFamilyGroupSize_negative": int((features["NonFamilyGroupSize"] < 0).sum()),
        "NonFamilyGroupRatio_negative": int((features["NonFamilyGroupRatio"] < 0).sum()),
        "NaN": int(features[ADDED_FEATURES].isna().sum().sum()),
        "Inf": int(np.isinf(features[ADDED_FEATURES].to_numpy()).sum()),
    }
    print(f"[{label}]")
    print("TicketGroupSize == 0 개수:", checks["TicketGroupSize_zero"])
    print("NonFamilyGroupSize 음수 개수:", checks["NonFamilyGroupSize_negative"])
    print("NonFamilyGroupRatio 음수 개수:", checks["NonFamilyGroupRatio_negative"])
    print("NaN 개수:", checks["NaN"])
    print("Inf 개수:", checks["Inf"])
    return checks


cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
assert all(feature in X_tr_features.columns for feature in ADDED_FEATURES)
assert "TicketGroupSize" not in X_tr_features.columns
assert "IsAlone" not in X_tr_features.columns
print("모델 입력 전 Feature:", X_tr_features.columns.tolist())
display(X_tr_features[ADDED_FEATURES].describe())
for feature in ADDED_FEATURES:
    print(f"\n[{feature} value counts]")
    print(X_tr_features[feature].value_counts().sort_index().head(30).to_string())


cell 19: for feature in ADDED_FEATURES:
    relation = pd.DataFrame({feature: X_tr_features[feature], target_col: y_tr})
    relation["bin"] = pd.qcut(relation[feature], q=4, duplicates="drop")
    print(f"\n[{feature} vs target]")
    print(relation.groupby("bin", observed=True)[target_col].agg(["size", "mean"]).to_string())


cell 21: prep.fit_encoding(X_tr_features)
X_tr_model = prep.transform_encoding(X_tr_features)
X_valid_model = prep.transform_encoding(X_valid_features)
assert X_tr_model.columns.equals(X_valid_model.columns)
assert all(feature in X_tr_model.columns for feature in ADDED_FEATURES)
assert "TicketGroupSize" not in X_tr_model.columns
print("범주형:", prep.cat_cols_)
print("인코딩 후:", X_tr_model.shape, X_valid_model.shape)
print("Encoded Feature 목록:", X_tr_model.columns.tolist())


cell 33: full_prep = Titanic25Preprocessor()
X_full_model = full_prep.fit_transform(X)
X_test_model = full_prep.transform(X_test_raw)
assert X_full_model.columns.equals(X_test_model.columns)
assert all(feature in X_full_model.columns for feature in ADDED_FEATURES)
assert "TicketGroupSize" not in X_full_model.columns
assert "IsAlone" not in X_full_model.columns
assert np.isfinite(X_full_model.to_numpy()).all()
assert np.isfinite(X_test_model.to_numpy()).all()
full_model = fit_cat(X_full_model, y)

X_full_features = full_prep.feature_frame(X, full_prep.transform_missing(X))
X_test_features = full_prep.feature_frame(X_test_raw, full_prep.transform_missing(X_test_raw))
full_train_checks = feature_diagnostics("full train", X, X_full_features, full_prep)
test_checks = feature_diagnostics("test", X_test_raw, X_test_features, full_prep)
print("전체 train/test 변환:", X_full_model.shape, X_test_model.shape)
print("최종 Feature 목록:", X_full_model.columns.tolist())


cell 39: feature_importance = pd.Series(
    full_model.feature_importances_, index=X_full_model.columns, name="importance"
).sort_values(ascending=False)
ticket_feature_importance = feature_importance.reindex(ADDED_FEATURES)
print("전체 Feature Importance:\n", feature_importance.to_string())
print("\n세 Ticket Feature Importance:\n", ticket_feature_importance.to_string())
feature_importance.sort_values().plot.barh(figsize=(11, 8), title="Titanic 25 Feature Importance")
plt.tight_layout()
plt.show()


cell 41: import shap
import skimage

explainer = shap.TreeExplainer(full_model)
shap_values = explainer.shap_values(X_test_model)

if isinstance(shap_values, list):
    shap_for_plot = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
else:
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_for_plot = shap_array[:, :, 1]
    else:
        shap_for_plot = shap_array

assert shap_for_plot.shape == X_test_model.shape, (shap_for_plot.shape, X_test_model.shape)
ticket_shap_importance = pd.Series(
    np.abs(shap_for_plot).mean(axis=0), index=X_test_model.columns, name="mean_abs_shap"
).reindex(ADDED_FEATURES)
print("SHAP shape:", shap_for_plot.shape, "X_test_model shape:", X_test_model.shape)
print("세 Ticket Feature mean(|SHAP|):\n", ticket_shap_importance.to_string())
shap.summary_plot(shap_for_plot, X_test_model, show=False)
plt.tight_layout()
plt.show()


cell 47: individual_best_val = max(0.899333, 0.901935, 0.896975)
individual_best_cv = max(0.900966, 0.900752, 0.900748)
synergy_vs_individual = validation_auc > individual_best_val and cv_mean > individual_best_cv
adopt = validation_auc > 0.903480 and cv_mean > 0.900862 and gap <= 0.060675
decision = "채택" if adopt else "미채택"
synergy_assessment = "시너지 확인" if synergy_vs_individual else "일관된 시너지 없음"

summary = {
    "experiment": "Titanic 25",
    "baseline": "Titanic 21",
    "added_features": ADDED_FEATURES,
    "features": X_full_model.columns.tolist(),
    "train_auc": float(train_auc),
    "validation_auc": float(validation_auc),
    "gap": float(gap),
    "cv_mean": float(cv_mean),
    "cv_std": float(cv_std),
    "fold_scores": [float(value) for value in fold_scores],
    "vs_titanic_21": {
        "validation_auc_delta": float(baseline_val_delta),
        "cv_mean_delta": float(baseline_cv_delta),
        "gap_delta": float(baseline_gap_delta),
    },
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "normalized_confusion_matrix": {
        "tn": float(normalized_tn), "fp": float(normalized_fp),
        "fn": float(normalized_fn), "tp": float(normalized_tp),
    },
    "ticket_feature_importance": {key: float(value) for key, value in ticket_feature_importance.items()},
    "ticket_feature_mean_abs_shap": {key: float(value) for key, value in ticket_shap_importance.items()},
    "diagnostics": {
        "holdout_train": holdout_train_checks,
        "holdout_validation": holdout_valid_checks,
        "full_train": full_train_checks,
        "test": test_checks,
    },
    "submission_file": "submission/titanic_25_result.csv",
    "submission_checks": {key: bool(value) for key, value in submission_checks.items()},
    "execution_errors": 0,
    "synergy_assessment": synergy_assessment,
    "decision": decision,
}
summary_path = ROOT / "results" / "titanic_25_summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Train AUC: {train_auc:.6f}")
print(f"Validation AUC: {validation_auc:.6f}")
print(f"Gap: {gap:.6f}")
print(f"CV Mean: {cv_mean:.6f}")
print(f"CV Std: {cv_std:.6f}")
print("Ticket Feature Importance:\n", ticket_feature_importance.to_string())
print("Ticket Feature mean(|SHAP|):\n", ticket_shap_importance.to_string())
print("Synergy:", synergy_assessment)
print("최종 판단:", decision)
print("Summary:", summary_path)

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
범주형: ['gender', 'embarked']
인코딩 후: (687, 15) (229, 15)
Encoded Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'NonFamilyGroupSize', 'TicketFarePerPerson', 'NonFamilyGroupRatio', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']


Train AUC: 0.967303
Validation AUC: 0.896731
Train / Validation Gap: 0.070572


Fold 1: 0.883396
Fold 2: 0.915395
Fold 3: 0.893656
Fold 4: 0.888762
Fold 5: 0.911518
5-Fold CV Mean: 0.898545
5-Fold CV Std: 0.014154


      Version               Features  Train AUC  ...       Gap   CV Mean    CV Std
0  Titanic 21               Baseline   0.964155  ...  0.060675  0.900862  0.018278
1  Titanic 22   + NonFamilyGroupSize   0.965556  ...  0.066223  0.900966  0.016310
2  Titanic 23  + TicketFarePerPerson   0.967393  ...  0.065458  0.900752  0.015635
3  Titanic 24  + NonFamilyGroupRatio   0.966574  ...  0.069598  0.900748  0.014903
4  Titanic 25    + 3 Ticket Features   0.967303  ...  0.070572  0.898545  0.014154

[5 rows x 7 columns]
vs Titanic 21 Validation AUC: -0.006749
vs Titanic 21 CV Mean: -0.002317
vs Titanic 21 Gap: +0.009897


[full train]
TicketGroupSize == 0 개수: 0
NonFamilyGroupSize 음수 개수: 182
NonFamilyGroupRatio 음수 개수: 182
NaN 개수: 0
Inf 개수: 0
[test]
TicketGroupSize == 0 개수: 0
NonFamilyGroupSize 음수 개수: 148
NonFamilyGroupRatio 음수 개수: 148
NaN 개수: 0
Inf 개수: 0
전체 train/test 변환: (916, 15) (393, 15)
최종 Feature 목록: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'LogFare', 'NonFamilyGroupSize', 'TicketFarePerPerson', 'NonFamilyGroupRatio', 'gender_female', 'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S']


저장: C:\dev\study\machine_learning\submission\titanic_25_result.csv
검증: {'rows_393': True, 'template_shape': True, 'column_order': True, 'same_id_order': True, 'prediction_0_1': np.True_, 'no_nan': np.True_, 'no_duplicate_id': True}
   passengerid  survived
0          916  0.818665
1          917  0.894548
2          918  0.881269
3          919  0.076384
4          920  0.987190


전체 Feature Importance:
 gender_female          26.023005
gender_male            24.963045
TicketFarePerPerson     9.722970
age                     8.358497
pclass                  6.539717
LogFare                 5.316640
fare                    5.177672
FamilySize              3.231600
NonFamilyGroupRatio     2.508304
sibsp                   2.477969
NonFamilyGroupSize      1.700425
parch                   1.494057
embarked_S              1.346281
embarked_C              0.697531
embarked_Q              0.442289

세 Ticket Feature Importance:
 NonFamilyGroupSize     1.700425
TicketFarePerPerson    9.722970
NonFamilyGroupRatio    2.508304


SHAP shape: (393, 15) X_test_model shape: (393, 15)
세 Ticket Feature mean(|SHAP|):
 NonFamilyGroupSize     0.037456
TicketFarePerPerson    0.199704
NonFamilyGroupRatio    0.057431
titanic_25.ipynb:cell-41:22: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.


TN: 124
FP: 19
FN: 18
TP: 68


TN: 0.867
FP: 0.133
FN: 0.209
TP: 0.791
TN + FP: 1.0
FN + TP: 1.0


Train AUC: 0.967303
Validation AUC: 0.896731
Gap: 0.070572
CV Mean: 0.898545
CV Std: 0.014154
Ticket Feature Importance:
 NonFamilyGroupSize     1.700425
TicketFarePerPerson    9.722970
NonFamilyGroupRatio    2.508304
Ticket Feature mean(|SHAP|):
 NonFamilyGroupSize     0.037456
TicketFarePerPerson    0.199704
NonFamilyGroupRatio    0.057431
Synergy: 일관된 시너지 없음
최종 판단: 미채택
Summary: C:\dev\study\machine_learning\results\titanic_25_summary.json

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_26.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 1: from pathlib import Path
import random
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from common.interaction_experiments import FeaturePreprocessor
from common.feature_experiments import baseline_parameters

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission_template = pd.read_csv('csv/submission.csv')

target_col = 'survived'
id_col = 'passengerid'
BASE_FEATURES = ['GenderClass', 'GenderIsChild', 'ClassIsChild', 'AgeBand']
X = train.drop(columns=[target_col, id_col])
y = train[target_col].astype('int8')
X_test_raw = test.drop(columns=[id_col])

assert X.columns.equals(X_test_raw.columns)
assert train[id_col].is_unique and test[id_col].is_unique
assert set(train[id_col]).isdisjoint(test[id_col])

cell 5: train_part, valid_part = train_test_split(
    train, test_size=0.25, stratify=train[target_col], random_state=SEED
)
X_tr_raw = train_part.drop(columns=[target_col, id_col])
y_tr = train_part[target_col].astype('int8')
X_valid_raw = valid_part.drop(columns=[target_col, id_col])
y_valid = valid_part[target_col].astype('int8')
assert len(train_part) == 687 and len(valid_part) == 229

BASE_PARAMS = baseline_parameters()
assert BASE_PARAMS['random_state'] == SEED
print('Titanic 15 명시 파라미터:', BASE_PARAMS)
print('고정 피처:', BASE_FEATURES)

cell 6: def prepare_fold(X_train_raw, X_valid_raw):
    prep = FeaturePreprocessor(BASE_FEATURES)
    X_train_model = prep.fit_transform(X_train_raw)
    X_valid_model = prep.transform(X_valid_raw)
    assert X_train_model.columns.equals(X_valid_model.columns)
    assert np.isfinite(X_train_model.to_numpy()).all()
    assert np.isfinite(X_valid_model.to_numpy()).all()
    return prep, X_train_model, X_valid_model


def fit_model(X_train_model, y_train, overrides=None, **fit_kwargs):
    params = BASE_PARAMS | (overrides or {})
    model = CatBoostClassifier(**params)
    model.fit(X_train_model, y_train, **fit_kwargs)
    return model


def auc_scores(model, X_train_model, y_train, X_valid_model, y_valid):
    positive_index = list(model.classes_).index(1)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train_model)[:, positive_index])
    valid_auc = roc_auc_score(y_valid, model.predict_proba(X_valid_model)[:, positive_index])
    return train_auc, valid_auc, train_auc - valid_auc


def baseline_cv():
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        _, X_fold_train, X_fold_valid = prepare_fold(X.iloc[tr_idx], X.iloc[va_idx])
        model = fit_model(X_fold_train, y.iloc[tr_idx])
        train_auc, valid_auc, gap = auc_scores(
            model, X_fold_train, y.iloc[tr_idx], X_fold_valid, y.iloc[va_idx]
        )
        rows.append({'fold': fold, 'train_auc': train_auc,
                     'validation_auc': valid_auc, 'gap': gap})
    folds = pd.DataFrame(rows)
    summary = {
        'cv_mean': folds['validation_auc'].mean(),
        'cv_std': folds['validation_auc'].std(ddof=1),
        'train_auc': folds['train_auc'].mean(),
        'gap': folds['gap'].mean(),
    }
    return folds, summary


baseline_folds, baseline_cv_summary = baseline_cv()
display(baseline_folds)
print('Titanic 15 actual-submission baseline CV:', baseline_cv_summary)

cell 9: _, X_holdout_train, X_holdout_valid = prepare_fold(X_tr_raw, X_valid_raw)
baseline_holdout_model = fit_model(X_holdout_train, y_tr)
baseline_holdout = auc_scores(baseline_holdout_model, X_holdout_train, y_tr,
                              X_holdout_valid, y_valid)
best_holdout_model = fit_model(X_holdout_train, y_tr, BEST_TUNED)
best_holdout = auc_scores(best_holdout_model, X_holdout_train, y_tr,
                          X_holdout_valid, y_valid)

comparison = pd.DataFrame([
    {'Model': 'Baseline Titanic 15', 'Parameter': str(BASE_PARAMS),
     'CV Mean': baseline_cv_summary['cv_mean'], 'CV Std': baseline_cv_summary['cv_std'],
     'Validation AUC': baseline_holdout[1], 'Gap': baseline_holdout[2]},
    {'Model': 'Titanic 26 Best', 'Parameter': str(BEST_TUNED),
     'CV Mean': best['cv_mean'], 'CV Std': best['cv_std'],
     'Validation AUC': best_holdout[1], 'Gap': best_holdout[2]},
])
display(comparison)
print('Final Parameter:', BEST_TUNED)
FINAL_PARAMS = BASE_PARAMS | BEST_TUNED

cell 11: final_prep = FeaturePreprocessor(BASE_FEATURES)
X_full_model = final_prep.fit_transform(X)
X_test_model = final_prep.transform(X_test_raw)
assert X_full_model.columns.equals(X_test_model.columns)
final_model = CatBoostClassifier(**FINAL_PARAMS)
final_model.fit(X_full_model, y)
positive_index = list(final_model.classes_).index(1)
predictions = final_model.predict_proba(X_test_model)[:, positive_index]

# Titanic 15와 같은 template/id mapping 방식으로 생성한다.
assert submission_template.columns.tolist() == [id_col, target_col]
assert len(submission_template) == len(test)
assert submission_template[id_col].is_unique and submission_template[id_col].notna().all()
assert set(submission_template[id_col]) == set(test[id_col])
result = submission_template.copy(deep=True)
by_id = pd.Series(predictions, index=test[id_col].to_numpy())
result[target_col] = result[id_col].map(by_id)

# 저장 직전 검증: 행 수, ID/순서, 확률 dtype, NaN, 범위.
assert len(result) == len(test) == len(submission_template)
pd.testing.assert_series_equal(result[id_col], submission_template[id_col], check_names=True)
assert pd.api.types.is_float_dtype(result[target_col])
assert result[target_col].notna().all()
assert np.isfinite(result[target_col].to_numpy()).all()
assert result[target_col].between(0.0, 1.0).all()

output_path = Path('titanic_26_result.csv')
result.to_csv(output_path, index=False)
saved = pd.read_csv(output_path)
assert saved.shape == submission_template.shape == (len(test), 2)
assert saved.columns.equals(submission_template.columns)
pd.testing.assert_frame_equal(saved.drop(columns=target_col), submission_template.drop(columns=target_col))
assert pd.api.types.is_float_dtype(saved[target_col])
assert saved[target_col].notna().all() and saved[target_col].between(0.0, 1.0).all()
np.testing.assert_allclose(saved[target_col], by_id.loc[submission_template[id_col]], rtol=1e-12, atol=1e-15)
print('Submission 검증 통과:', output_path, saved.shape)
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
   fold  train_auc  validation_auc       gap
0     1   0.957912        0.906140  0.051772
1     2   0.958143        0.924358  0.033785
2     3   0.963202        0.886728  0.076475
3     4   0.956995        0.905861  0.051135
4     5   0.957728        0.914505  0.043222

Titanic 15 actual-submission baseline CV: {'cv_mean': np.float64(0.9075184337655735), 'cv_std': np.float64(0.013868055749294898), 'train_auc': np.float64(0.9587961681724849), 'gap': np.float64(0.05127773440691141)}


    depth  l2_leaf_reg  fold  train_auc  validation_auc       gap
0       4            3     1   0.975528        0.902820  0.072708
1       4            3     2   0.975458        0.923659  0.051799
2       4            3     3   0.979994        0.887490  0.092504
3       4            3     4   0.976249        0.898042  0.078207
4       4            3     5   0.979060        0.916095  0.062965
..    ...          ...   ...        ...             ...       ...
95      7           20     1   0.968873        0.904386  0.064487
96      7           20     2   0.969742        0.920163  0.049579
97      7           20     3   0.973118        0.885329  0.087789
98      7           20     4   0.967984        0.912726  0.055258
99      7           20     5   0.969896        0.917493  0.052403

[100 rows x 6 columns]

    depth  l2_leaf_reg   cv_mean    cv_std  train_auc       gap
0       4           20  0.909335  0.014127   0.957081  0.047746
1       4           12  0.909253  0.015216   0.963083  0.053830
2       5           20  0.909082  0.013205   0.962255  0.053172
3       7           12  0.908964  0.014120   0.976655  0.067691
4       6            8  0.908649  0.014067   0.977790  0.069141
5       6           12  0.908500  0.014870   0.973657  0.065158
6       4            8  0.908073  0.013809   0.967842  0.059769
7       7           20  0.908019  0.014030   0.969923  0.061903
8       6           20  0.907702  0.012770   0.967184  0.059482
9       5           12  0.907523  0.015185   0.968448  0.060925
10      7            8  0.907446  0.013718   0.981337  0.073891
11      5            8  0.907067  0.016003   0.973202  0.066135
12      6            5  0.906464  0.015740   0.982966  0.076502
13      4            5  0.905712  0.016073   0.972750  0.067038
14      4            3  0.905621  0.014397   0.977258  0.071637
15      5            3  0.905504  0.016220   0.982709  0.077204
16      7            5  0.905054  0.011914   0.985304  0.080250
17      5            5  0.904938  0.014691   0.978224  0.073286
18      6            3  0.904887  0.015949   0.986631  0.081744
19      7            3  0.901336  0.012958   0.988877  0.087541

                 Model                                          Parameter  \
0  Baseline Titanic 15  {'verbose': 0, 'random_state': 42, 'cat_featur...   
1      Titanic 26 Best                  {'depth': 4, 'l2_leaf_reg': 20.0}   

    CV Mean    CV Std  Validation AUC       Gap  
0  0.907518  0.013868        0.900024  0.062311  
1  0.909335  0.014127        0.900390  0.063364  
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_27.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 1: from pathlib import Path
import random
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from common.interaction_experiments import FeaturePreprocessor
from common.feature_experiments import baseline_parameters

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission_template = pd.read_csv('csv/submission.csv')

target_col = 'survived'
id_col = 'passengerid'
BASE_FEATURES = ['GenderClass', 'GenderIsChild', 'ClassIsChild', 'AgeBand']
X = train.drop(columns=[target_col, id_col])
y = train[target_col].astype('int8')
X_test_raw = test.drop(columns=[id_col])

assert X.columns.equals(X_test_raw.columns)
assert train[id_col].is_unique and test[id_col].is_unique
assert set(train[id_col]).isdisjoint(test[id_col])

cell 5: train_part, valid_part = train_test_split(
    train, test_size=0.25, stratify=train[target_col], random_state=SEED
)
X_tr_raw = train_part.drop(columns=[target_col, id_col])
y_tr = train_part[target_col].astype('int8')
X_valid_raw = valid_part.drop(columns=[target_col, id_col])
y_valid = valid_part[target_col].astype('int8')
assert len(train_part) == 687 and len(valid_part) == 229

BASE_PARAMS = baseline_parameters()
assert BASE_PARAMS['random_state'] == SEED
print('Titanic 15 명시 파라미터:', BASE_PARAMS)
print('고정 피처:', BASE_FEATURES)

cell 6: def prepare_fold(X_train_raw, X_valid_raw):
    prep = FeaturePreprocessor(BASE_FEATURES)
    X_train_model = prep.fit_transform(X_train_raw)
    X_valid_model = prep.transform(X_valid_raw)
    assert X_train_model.columns.equals(X_valid_model.columns)
    assert np.isfinite(X_train_model.to_numpy()).all()
    assert np.isfinite(X_valid_model.to_numpy()).all()
    return prep, X_train_model, X_valid_model


def fit_model(X_train_model, y_train, overrides=None, **fit_kwargs):
    params = BASE_PARAMS | (overrides or {})
    model = CatBoostClassifier(**params)
    model.fit(X_train_model, y_train, **fit_kwargs)
    return model


def auc_scores(model, X_train_model, y_train, X_valid_model, y_valid):
    positive_index = list(model.classes_).index(1)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train_model)[:, positive_index])
    valid_auc = roc_auc_score(y_valid, model.predict_proba(X_valid_model)[:, positive_index])
    return train_auc, valid_auc, train_auc - valid_auc


def baseline_cv():
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        _, X_fold_train, X_fold_valid = prepare_fold(X.iloc[tr_idx], X.iloc[va_idx])
        model = fit_model(X_fold_train, y.iloc[tr_idx])
        train_auc, valid_auc, gap = auc_scores(
            model, X_fold_train, y.iloc[tr_idx], X_fold_valid, y.iloc[va_idx]
        )
        rows.append({'fold': fold, 'train_auc': train_auc,
                     'validation_auc': valid_auc, 'gap': gap})
    folds = pd.DataFrame(rows)
    summary = {
        'cv_mean': folds['validation_auc'].mean(),
        'cv_std': folds['validation_auc'].std(ddof=1),
        'train_auc': folds['train_auc'].mean(),
        'gap': folds['gap'].mean(),
    }
    return folds, summary


baseline_folds, baseline_cv_summary = baseline_cv()
display(baseline_folds)
print('Titanic 15 actual-submission baseline CV:', baseline_cv_summary)

cell 8: from sklearn.model_selection import ParameterSampler

search_space = {
    'random_strength': [0.2, 0.5, 1, 2, 4],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'rsm': [0.7, 0.85, 1.0],
}
sampled_params = list(ParameterSampler(search_space, n_iter=36, random_state=SEED))
assert len(sampled_params) == 36
assert len({tuple(sorted(p.items())) for p in sampled_params}) == 36

candidate_rows = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for params in sampled_params:
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        _, X_fold_train, X_fold_valid = prepare_fold(X.iloc[tr_idx], X.iloc[va_idx])
        model = fit_model(X_fold_train, y.iloc[tr_idx], params)
        train_auc, valid_auc, gap = auc_scores(
            model, X_fold_train, y.iloc[tr_idx], X_fold_valid, y.iloc[va_idx]
        )
        candidate_rows.append({**params, 'fold': fold, 'train_auc': train_auc,
                               'validation_auc': valid_auc, 'gap': gap})
    print('완료:', params)

cv_folds = pd.DataFrame(candidate_rows)
cv_summary = (cv_folds.groupby(['random_strength', 'subsample', 'rsm'], as_index=False)
              .agg(cv_mean=('validation_auc', 'mean'),
                   cv_std=('validation_auc', 'std'),
                   train_auc=('train_auc', 'mean'),
                   gap=('gap', 'mean'))
              .sort_values(['cv_mean', 'cv_std'], ascending=[False, True], ignore_index=True))
display(cv_folds)
display(cv_summary)
best = cv_summary.iloc[0]
BEST_TUNED = {'random_strength': float(best['random_strength']),
              'subsample': float(best['subsample']), 'rsm': float(best['rsm'])}
print('Selected Best Parameter:', BEST_TUNED)

cell 9: _, X_holdout_train, X_holdout_valid = prepare_fold(X_tr_raw, X_valid_raw)
baseline_holdout_model = fit_model(X_holdout_train, y_tr)
baseline_holdout = auc_scores(baseline_holdout_model, X_holdout_train, y_tr,
                              X_holdout_valid, y_valid)
best_holdout_model = fit_model(X_holdout_train, y_tr, BEST_TUNED)
best_holdout = auc_scores(best_holdout_model, X_holdout_train, y_tr,
                          X_holdout_valid, y_valid)

comparison = pd.DataFrame([
    {'Model': 'Baseline Titanic 15', 'Parameter': str(BASE_PARAMS),
     'CV Mean': baseline_cv_summary['cv_mean'], 'CV Std': baseline_cv_summary['cv_std'],
     'Validation AUC': baseline_holdout[1], 'Gap': baseline_holdout[2]},
    {'Model': 'Titanic 27 Best', 'Parameter': str(BEST_TUNED),
     'CV Mean': best['cv_mean'], 'CV Std': best['cv_std'],
     'Validation AUC': best_holdout[1], 'Gap': best_holdout[2]},
])
display(comparison)
print('Final Parameter:', BEST_TUNED)
FINAL_PARAMS = BASE_PARAMS | BEST_TUNED

cell 11: final_prep = FeaturePreprocessor(BASE_FEATURES)
X_full_model = final_prep.fit_transform(X)
X_test_model = final_prep.transform(X_test_raw)
assert X_full_model.columns.equals(X_test_model.columns)
final_model = CatBoostClassifier(**FINAL_PARAMS)
final_model.fit(X_full_model, y)
positive_index = list(final_model.classes_).index(1)
predictions = final_model.predict_proba(X_test_model)[:, positive_index]

# Titanic 15와 같은 template/id mapping 방식으로 생성한다.
assert submission_template.columns.tolist() == [id_col, target_col]
assert len(submission_template) == len(test)
assert submission_template[id_col].is_unique and submission_template[id_col].notna().all()
assert set(submission_template[id_col]) == set(test[id_col])
result = submission_template.copy(deep=True)
by_id = pd.Series(predictions, index=test[id_col].to_numpy())
result[target_col] = result[id_col].map(by_id)

# 저장 직전 검증: 행 수, ID/순서, 확률 dtype, NaN, 범위.
assert len(result) == len(test) == len(submission_template)
pd.testing.assert_series_equal(result[id_col], submission_template[id_col], check_names=True)
assert pd.api.types.is_float_dtype(result[target_col])
assert result[target_col].notna().all()
assert np.isfinite(result[target_col].to_numpy()).all()
assert result[target_col].between(0.0, 1.0).all()

output_path = Path('titanic_27_result.csv')
result.to_csv(output_path, index=False)
saved = pd.read_csv(output_path)
assert saved.shape == submission_template.shape == (len(test), 2)
assert saved.columns.equals(submission_template.columns)
pd.testing.assert_frame_equal(saved.drop(columns=target_col), submission_template.drop(columns=target_col))
assert pd.api.types.is_float_dtype(saved[target_col])
assert saved[target_col].notna().all() and saved[target_col].between(0.0, 1.0).all()
np.testing.assert_allclose(saved[target_col], by_id.loc[submission_template[id_col]], rtol=1e-12, atol=1e-15)
print('Submission 검증 통과:', output_path, saved.shape)
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
   fold  train_auc  validation_auc       gap
0     1   0.957912        0.906140  0.051772
1     2   0.958143        0.924358  0.033785
2     3   0.963202        0.886728  0.076475
3     4   0.956995        0.905861  0.051135
4     5   0.957728        0.914505  0.043222

Titanic 15 actual-submission baseline CV: {'cv_mean': np.float64(0.9075184337655735), 'cv_std': np.float64(0.013868055749294898), 'train_auc': np.float64(0.9587961681724849), 'gap': np.float64(0.05127773440691141)}


     subsample  rsm  random_strength  fold  train_auc  validation_auc  \
0          0.7  0.7              0.2     1   0.964209        0.905890   
1          0.7  0.7              0.2     2   0.965518        0.923722   
2          0.7  0.7              0.2     3   0.970771        0.883931   
3          0.7  0.7              0.2     4   0.965716        0.905225   
4          0.7  0.7              0.2     5   0.967921        0.915205   
..         ...  ...              ...   ...        ...             ...   
175        0.7  1.0              1.0     1   0.957555        0.908145   
176        0.7  1.0              1.0     2   0.958413        0.924612   
177        0.7  1.0              1.0     3   0.964675        0.887999   
178        0.7  1.0              1.0     4   0.959165        0.902428   
179        0.7  1.0              1.0     5   0.958179        0.914823   

          gap  
0    0.058319  
1    0.041796  
2    0.086840  
3    0.060491  
4    0.052716  
..        ...  
175  0.049409  
176  0.033800  
177  0.076676  
178  0.056736  
179  0.043356  

[180 rows x 7 columns]

    random_strength  subsample   rsm   cv_mean    cv_std  train_auc       gap
0               0.2        0.8  1.00  0.909254  0.015288   0.967199  0.057945
1               1.0        0.7  0.70  0.908888  0.012778   0.958011  0.049123
2               0.5        1.0  0.70  0.908770  0.016024   0.963058  0.054287
3               2.0        0.7  0.85  0.908713  0.012145   0.954337  0.045624
4               0.2        0.7  1.00  0.908343  0.014845   0.968019  0.059676
5               4.0        0.7  0.85  0.908324  0.011613   0.949914  0.041591
6               0.2        0.8  0.85  0.908252  0.015711   0.966947  0.058695
7               4.0        0.9  1.00  0.908249  0.011782   0.949236  0.040987
8               0.2        0.7  0.85  0.908201  0.013713   0.967569  0.059369
9               4.0        0.7  0.70  0.908180  0.012220   0.949505  0.041325
10              0.5        1.0  0.85  0.908174  0.013490   0.963503  0.055330
11              1.0        0.9  0.70  0.908162  0.014429   0.957902  0.049739
12              0.2        1.0  0.70  0.908146  0.014537   0.966758  0.058613
13              0.5        0.8  0.85  0.907870  0.013859   0.962733  0.054862
14              4.0        0.9  0.70  0.907853  0.011711   0.948748  0.040895
15              4.0        1.0  0.85  0.907798  0.011338   0.949252  0.041454
16              1.0        0.9  1.00  0.907694  0.011914   0.959515  0.051821
17              0.2        1.0  1.00  0.907603  0.015517   0.968286  0.060683
18              1.0        0.7  1.00  0.907602  0.013719   0.959597  0.051995
19              0.5        0.7  0.85  0.907589  0.014328   0.963560  0.055971
20              1.0        0.8  1.00  0.907518  0.013868   0.958796  0.051278
21              4.0        0.7  1.00  0.907514  0.012779   0.949444  0.041930
22              1.0        0.8  0.70  0.907331  0.014812   0.957667  0.050336
23              1.0        0.9  0.85  0.907300  0.015788   0.958785  0.051484
24              0.5        0.7  0.70  0.907235  0.014805   0.962330  0.055095
25              0.5        0.8  0.70  0.907157  0.013918   0.962681  0.055524
26              2.0        0.7  0.70  0.907080  0.011666   0.953810  0.046730
27              1.0        1.0  0.70  0.907075  0.015233   0.958748  0.051673
28              0.2        0.9  0.85  0.906942  0.014954   0.967030  0.060089
29              2.0        1.0  0.85  0.906850  0.013639   0.953843  0.046992
30              0.2        0.7  0.70  0.906795  0.014860   0.966827  0.060032
31              4.0        0.8  1.00  0.906674  0.011892   0.949544  0.042870
32              2.0        0.8  1.00  0.906574  0.013482   0.954253  0.047679
33              2.0        0.9  1.00  0.906568  0.014427   0.955148  0.048580
34              1.0        1.0  0.85  0.906083  0.013772   0.959061  0.052978
35              4.0        0.9  0.85  0.905543  0.012396   0.949321  0.043778

                 Model                                          Parameter  \
0  Baseline Titanic 15  {'verbose': 0, 'random_state': 42, 'cat_featur...   
1      Titanic 27 Best  {'random_strength': 0.2, 'subsample': 0.8, 'rs...   

    CV Mean    CV Std  Validation AUC       Gap  
0  0.907518  0.013868        0.900024  0.062311  
1  0.909254  0.015288        0.895633  0.077763  
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_28.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 1: from pathlib import Path
import random
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from common.interaction_experiments import FeaturePreprocessor
from common.feature_experiments import baseline_parameters

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission_template = pd.read_csv('csv/submission.csv')

target_col = 'survived'
id_col = 'passengerid'
BASE_FEATURES = ['GenderClass', 'GenderIsChild', 'ClassIsChild', 'AgeBand']
X = train.drop(columns=[target_col, id_col])
y = train[target_col].astype('int8')
X_test_raw = test.drop(columns=[id_col])

assert X.columns.equals(X_test_raw.columns)
assert train[id_col].is_unique and test[id_col].is_unique
assert set(train[id_col]).isdisjoint(test[id_col])

cell 5: train_part, valid_part = train_test_split(
    train, test_size=0.25, stratify=train[target_col], random_state=SEED
)
X_tr_raw = train_part.drop(columns=[target_col, id_col])
y_tr = train_part[target_col].astype('int8')
X_valid_raw = valid_part.drop(columns=[target_col, id_col])
y_valid = valid_part[target_col].astype('int8')
assert len(train_part) == 687 and len(valid_part) == 229

BASE_PARAMS = baseline_parameters()
assert BASE_PARAMS['random_state'] == SEED
print('Titanic 15 명시 파라미터:', BASE_PARAMS)
print('고정 피처:', BASE_FEATURES)

cell 6: def prepare_fold(X_train_raw, X_valid_raw):
    prep = FeaturePreprocessor(BASE_FEATURES)
    X_train_model = prep.fit_transform(X_train_raw)
    X_valid_model = prep.transform(X_valid_raw)
    assert X_train_model.columns.equals(X_valid_model.columns)
    assert np.isfinite(X_train_model.to_numpy()).all()
    assert np.isfinite(X_valid_model.to_numpy()).all()
    return prep, X_train_model, X_valid_model


def fit_model(X_train_model, y_train, overrides=None, **fit_kwargs):
    params = BASE_PARAMS | (overrides or {})
    model = CatBoostClassifier(**params)
    model.fit(X_train_model, y_train, **fit_kwargs)
    return model


def auc_scores(model, X_train_model, y_train, X_valid_model, y_valid):
    positive_index = list(model.classes_).index(1)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train_model)[:, positive_index])
    valid_auc = roc_auc_score(y_valid, model.predict_proba(X_valid_model)[:, positive_index])
    return train_auc, valid_auc, train_auc - valid_auc


def baseline_cv():
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        _, X_fold_train, X_fold_valid = prepare_fold(X.iloc[tr_idx], X.iloc[va_idx])
        model = fit_model(X_fold_train, y.iloc[tr_idx])
        train_auc, valid_auc, gap = auc_scores(
            model, X_fold_train, y.iloc[tr_idx], X_fold_valid, y.iloc[va_idx]
        )
        rows.append({'fold': fold, 'train_auc': train_auc,
                     'validation_auc': valid_auc, 'gap': gap})
    folds = pd.DataFrame(rows)
    summary = {
        'cv_mean': folds['validation_auc'].mean(),
        'cv_std': folds['validation_auc'].std(ddof=1),
        'train_auc': folds['train_auc'].mean(),
        'gap': folds['gap'].mean(),
    }
    return folds, summary


baseline_folds, baseline_cv_summary = baseline_cv()
display(baseline_folds)
print('Titanic 15 actual-submission baseline CV:', baseline_cv_summary)

cell 9: _, X_holdout_train, X_holdout_valid = prepare_fold(X_tr_raw, X_valid_raw)
baseline_holdout_model = fit_model(X_holdout_train, y_tr)
baseline_holdout = auc_scores(baseline_holdout_model, X_holdout_train, y_tr,
                              X_holdout_valid, y_valid)
best_holdout_params = {'learning_rate': BEST_LEARNING_RATE, 'iterations': 3000,
                       'loss_function': 'Logloss', 'eval_metric': 'AUC'}
best_holdout_model = fit_model(
    X_holdout_train, y_tr, best_holdout_params,
    eval_set=(X_holdout_valid, y_valid), early_stopping_rounds=150,
    use_best_model=True, verbose=False,
)
best_holdout = auc_scores(best_holdout_model, X_holdout_train, y_tr,
                          X_holdout_valid, y_valid)
holdout_best_iteration = int(best_holdout_model.get_best_iteration()) + 1

comparison = pd.DataFrame([
    {'Model': 'Baseline Titanic 15', 'Parameter': str(BASE_PARAMS),
     'CV Mean': baseline_cv_summary['cv_mean'], 'CV Std': baseline_cv_summary['cv_std'],
     'Validation AUC': baseline_holdout[1], 'Gap': baseline_holdout[2]},
    {'Model': 'Titanic 28 Best',
     'Parameter': str({'learning_rate': BEST_LEARNING_RATE,
                       'final_iterations': FINAL_ITERATIONS,
                       'early_stopping_rounds': 150}),
     'CV Mean': best['cv_mean'], 'CV Std': best['cv_std'],
     'Validation AUC': best_holdout[1], 'Gap': best_holdout[2]},
])
display(comparison)
print('Best learning_rate:', BEST_LEARNING_RATE)
print('Holdout Best Iteration:', holdout_best_iteration)
print('Final iterations:', FINAL_ITERATIONS)
print('CV Mean:', best['cv_mean'], 'CV Std:', best['cv_std'])
FINAL_PARAMS = BASE_PARAMS | {
    'learning_rate': BEST_LEARNING_RATE,
    'iterations': FINAL_ITERATIONS,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
}

cell 11: final_prep = FeaturePreprocessor(BASE_FEATURES)
X_full_model = final_prep.fit_transform(X)
X_test_model = final_prep.transform(X_test_raw)
assert X_full_model.columns.equals(X_test_model.columns)
final_model = CatBoostClassifier(**FINAL_PARAMS)
final_model.fit(X_full_model, y)
positive_index = list(final_model.classes_).index(1)
predictions = final_model.predict_proba(X_test_model)[:, positive_index]

# Titanic 15와 같은 template/id mapping 방식으로 생성한다.
assert submission_template.columns.tolist() == [id_col, target_col]
assert len(submission_template) == len(test)
assert submission_template[id_col].is_unique and submission_template[id_col].notna().all()
assert set(submission_template[id_col]) == set(test[id_col])
result = submission_template.copy(deep=True)
by_id = pd.Series(predictions, index=test[id_col].to_numpy())
result[target_col] = result[id_col].map(by_id)

# 저장 직전 검증: 행 수, ID/순서, 확률 dtype, NaN, 범위.
assert len(result) == len(test) == len(submission_template)
pd.testing.assert_series_equal(result[id_col], submission_template[id_col], check_names=True)
assert pd.api.types.is_float_dtype(result[target_col])
assert result[target_col].notna().all()
assert np.isfinite(result[target_col].to_numpy()).all()
assert result[target_col].between(0.0, 1.0).all()

output_path = Path('titanic_28_result.csv')
result.to_csv(output_path, index=False)
saved = pd.read_csv(output_path)
assert saved.shape == submission_template.shape == (len(test), 2)
assert saved.columns.equals(submission_template.columns)
pd.testing.assert_frame_equal(saved.drop(columns=target_col), submission_template.drop(columns=target_col))
assert pd.api.types.is_float_dtype(saved[target_col])
assert saved[target_col].notna().all() and saved[target_col].between(0.0, 1.0).all()
np.testing.assert_allclose(saved[target_col], by_id.loc[submission_template[id_col]], rtol=1e-12, atol=1e-15)
print('Submission 검증 통과:', output_path, saved.shape)
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
   fold  train_auc  validation_auc       gap
0     1   0.957912        0.906140  0.051772
1     2   0.958143        0.924358  0.033785
2     3   0.963202        0.886728  0.076475
3     4   0.956995        0.905861  0.051135
4     5   0.957728        0.914505  0.043222

Titanic 15 actual-submission baseline CV: {'cv_mean': np.float64(0.9075184337655735), 'cv_std': np.float64(0.013868055749294898), 'train_auc': np.float64(0.9587961681724849), 'gap': np.float64(0.05127773440691141)}


    learning_rate  fold  best_iteration  train_auc  validation_auc       gap
0           0.010     1              42   0.923003        0.904699  0.018304
1           0.010     2             182   0.928586        0.929189 -0.000603
2           0.010     3               6   0.922798        0.893974  0.028824
3           0.010     4               2   0.909854        0.919209 -0.009355
4           0.010     5              22   0.921904        0.911200  0.010704
5           0.015     1              41   0.925307        0.906140  0.019167
6           0.015     2             126   0.929255        0.929761 -0.000506
7           0.015     3               9   0.923645        0.897343  0.026302
8           0.015     4               2   0.909696        0.919209 -0.009513
9           0.015     5              19   0.923143        0.914760  0.008383
10          0.020     1              39   0.925784        0.907206  0.018579
11          0.020     2              17   0.919639        0.930714 -0.011075
12          0.020     3               6   0.922862        0.893974  0.028888
13          0.020     4               2   0.909696        0.919209 -0.009513
14          0.020     5              38   0.923103        0.917874  0.005229
15          0.030     1             150   0.943578        0.908772  0.034806
16          0.030     2              70   0.929983        0.929570  0.000413
17          0.030     3               9   0.923238        0.895245  0.027992
18          0.030     4               2   0.909791        0.919591 -0.009800
19          0.030     5              20   0.921543        0.918446  0.003097
20          0.050     1              17   0.925729        0.909524  0.016205
21          0.050     2              21   0.923661        0.930079 -0.006418
22          0.050     3              20   0.927430        0.894165  0.033265
23          0.050     4               2   0.909799        0.919591 -0.009792
24          0.050     5              30   0.928752        0.914823  0.013929

   learning_rate   cv_mean    cv_std  mean_best_iteration  \
0          0.030  0.914325  0.012962                 50.2   
1          0.020  0.913796  0.013861                 20.4   
2          0.050  0.913636  0.013260                 18.0   
3          0.015  0.913443  0.012387                 39.4   
4          0.010  0.911654  0.013470                 50.8   

   median_best_iteration  train_auc       gap  
0                   20.0   0.925627  0.011302  
1                   17.0   0.920217  0.006421  
2                   20.0   0.923074  0.009438  
3                   19.0   0.922209  0.008767  
4                   22.0   0.921229  0.009575  

Best learning_rate: 0.03
CV mean/median Best Iteration: 50.2 20.0
Final iterations (rounded CV median): 20


                 Model                                          Parameter  \
0  Baseline Titanic 15  {'verbose': 0, 'random_state': 42, 'cat_featur...   
1      Titanic 28 Best  {'learning_rate': 0.03, 'final_iterations': 20...   

    CV Mean    CV Std  Validation AUC       Gap  
0  0.907518  0.013868        0.900024  0.062311  
1  0.914325  0.012962        0.906367  0.019120  

Best learning_rate: 0.03
Holdout Best Iteration: 24
Final iterations: 20
CV Mean: 0.914324942791762 CV Std: 0.012961501591864434

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_29.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 1: from pathlib import Path
import random
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from common.interaction_experiments import FeaturePreprocessor
from common.feature_experiments import baseline_parameters

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission_template = pd.read_csv('csv/submission.csv')

target_col = 'survived'
id_col = 'passengerid'
BASE_FEATURES = ['GenderClass', 'GenderIsChild', 'ClassIsChild', 'AgeBand']
X = train.drop(columns=[target_col, id_col])
y = train[target_col].astype('int8')
X_test_raw = test.drop(columns=[id_col])

assert X.columns.equals(X_test_raw.columns)
assert train[id_col].is_unique and test[id_col].is_unique
assert set(train[id_col]).isdisjoint(test[id_col])
train_part, valid_part = train_test_split(
    train, test_size=0.25, stratify=train[target_col], random_state=SEED
)
X_tr_raw = train_part.drop(columns=[target_col, id_col])
y_tr = train_part[target_col].astype('int8')
X_valid_raw = valid_part.drop(columns=[target_col, id_col])
y_valid = valid_part[target_col].astype('int8')
assert len(train_part) == 687 and len(valid_part) == 229

BASE_PARAMS = baseline_parameters()
assert BASE_PARAMS['random_state'] == SEED
print('Titanic 15 명시 파라미터:', BASE_PARAMS)
print('고정 피처:', BASE_FEATURES)
import sys
import catboost, sklearn
print("Runtime:", sys.version)
print("Versions:", catboost.__version__, sklearn.__version__, pd.__version__)
PARAMS = {
 '28': {'learning_rate': 0.03},
 '29': {'learning_rate': 0.03, 'depth': 4, 'l2_leaf_reg': 20},
 '30': {'learning_rate': 0.03, 'random_strength': 0.2, 'subsample': 0.8, 'rsm': 1.0},
 'A': {'learning_rate': 0.03, 'depth': 4, 'l2_leaf_reg': 20,
       'random_strength': 0.2, 'subsample': 0.8, 'rsm': 1.0},
}
SPLITS=list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X,y))
# Declare practical stability criteria before seeing results.
MIN_GAIN=0.001
MAX_STD_INCREASE=0.003
MAX_GAP_INCREASE=0.01
MAX_HOLDOUT_DROP=0.005
MIN_IMPROVED_FOLDS=3
MAX_SINGLE_FOLD_SHARE=0.60

cell 4: def prepare_fold(X_train_raw, X_valid_raw):
    prep = FeaturePreprocessor(BASE_FEATURES)
    X_train_model = prep.fit_transform(X_train_raw)
    X_valid_model = prep.transform(X_valid_raw)
    assert X_train_model.columns.equals(X_valid_model.columns)
    assert np.isfinite(X_train_model.to_numpy()).all()
    assert np.isfinite(X_valid_model.to_numpy()).all()
    return prep, X_train_model, X_valid_model


def fit_model(X_train_model, y_train, overrides=None, **fit_kwargs):
    params = BASE_PARAMS | (overrides or {})
    model = CatBoostClassifier(**params)
    model.fit(X_train_model, y_train, **fit_kwargs)
    return model


def auc_scores(model, X_train_model, y_train, X_valid_model, y_valid):
    positive_index = list(model.classes_).index(1)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train_model)[:, positive_index])
    valid_auc = roc_auc_score(y_valid, model.predict_proba(X_valid_model)[:, positive_index])
    return train_auc, valid_auc, train_auc - valid_auc


def positive(model, values):
    assert model.get_all_params()['random_seed']==SEED
    return model.predict_proba(values)[:,list(model.classes_).index(1)]

def evaluate(label):
    rows, train_predictions=[],[]
    oof=np.full(len(y),np.nan)
    visits=np.zeros(len(y),dtype=int)
    params=PARAMS[label]|{'iterations':3000,'loss_function':'Logloss','eval_metric':'AUC'}
    for fold,(tr,va) in enumerate(SPLITS,1):
        assert not set(tr).intersection(va)
        _,xt,xv=prepare_fold(X.iloc[tr],X.iloc[va])
        model=fit_model(xt,y.iloc[tr],params,eval_set=(xv,y.iloc[va]),
                        early_stopping_rounds=150,use_best_model=True,verbose=False)
        pt,pv=positive(model,xt),positive(model,xv)
        oof[va]=pv
        visits[va]+=1
        train_predictions.append(pt)
        ta,vauc=roc_auc_score(y.iloc[tr],pt),roc_auc_score(y.iloc[va],pv)
        best_iteration=int(model.get_best_iteration())+1
        assert best_iteration==model.tree_count_ and 1<=best_iteration<=3000
        rows.append(dict(fold=fold,train_auc=ta,validation_auc=vauc,gap=ta-vauc,best_iteration=best_iteration))
        print(f"Titanic {label} Fold {fold}: AUC={vauc:.9f}, Best Iteration={best_iteration}")
    assert (visits==1).all() and np.isfinite(oof).all()
    folds=pd.DataFrame(rows)
    iterations=int(folds.best_iteration.median())
    assert 1<=iterations<=3000
    _,ht,hv=prepare_fold(X_tr_raw,X_valid_raw)
    hm=fit_model(ht,y_tr,params,eval_set=(hv,y_valid),early_stopping_rounds=150,
                 use_best_model=True,verbose=False)
    hp_train,hp_valid=positive(hm,ht),positive(hm,hv)
    ht_auc,hv_auc=roc_auc_score(y_tr,hp_train),roc_auc_score(y_valid,hp_valid)
    summary={
      'Model':label,'CV Mean':float(folds.validation_auc.mean()),
      'CV Std':float(folds.validation_auc.std(ddof=1)),'OOF AUC':float(roc_auc_score(y,oof)),
      'Holdout AUC':float(hv_auc),'Train/Validation Gap':float(ht_auc-hv_auc),
      'CV Gap':float(folds.gap.mean()),'Best Iteration Mean':float(folds.best_iteration.mean()),
      'Best Iteration Median':float(folds.best_iteration.median()),'Final Iterations':iterations,
      'Parameter':str(BASE_PARAMS|params|{'iterations':iterations})}
    print("Best iteration distribution:",folds.best_iteration.tolist())
    print("Full-train iterations = CV median:",iterations,"(no override)")
    display(folds)
    display(pd.DataFrame([summary]))
    if label=='28':
        np.testing.assert_allclose(summary['CV Mean'],0.914324942791762,atol=1e-12,rtol=0)
        np.testing.assert_allclose(summary['CV Std'],0.012961501591864434,atol=1e-12,rtol=0)
        assert iterations==20
        print("Titanic 28 recorded result reproduced.")
    return dict(summary=summary,folds=folds,oof=oof,train_predictions=train_predictions,
                holdout_train=hp_train,holdout_valid=hp_valid,iterations=iterations)

def full_prediction(label,evaluation):
    prep=FeaturePreprocessor(BASE_FEATURES)
    full=prep.fit_transform(X)
    test_model=prep.transform(X_test_raw)
    assert full.columns.equals(test_model.columns)
    params=PARAMS[label]|{'iterations':evaluation['iterations'],'loss_function':'Logloss','eval_metric':'AUC'}
    model=fit_model(full,y,params)
    assert model.tree_count_==evaluation['iterations']
    return positive(model,test_model)

def save_submission(predictions,number):
    assert submission_template.columns.tolist()==[id_col,target_col]
    assert len(submission_template)==len(test)
    assert submission_template[id_col].is_unique and submission_template[id_col].notna().all()
    assert set(submission_template[id_col])==set(test[id_col])
    assert predictions.shape==(len(test),)
    by_id=pd.Series(predictions,index=test[id_col].to_numpy())
    result=submission_template.copy(deep=True)
    result[target_col]=result[id_col].map(by_id)
    assert result.shape==submission_template.shape and result.columns.equals(submission_template.columns)
    pd.testing.assert_series_equal(result[id_col],submission_template[id_col],check_names=True)
    p=result[target_col].to_numpy()
    assert pd.api.types.is_float_dtype(result[target_col])
    assert np.isfinite(p).all() and result[target_col].between(0,1).all()
    assert ((p>0)&(p<1)).any(),'Hard labels forbidden'
    output_path=Path(f'titanic_{number}_result.csv')
    result.to_csv(output_path,index=False)
    saved=pd.read_csv(output_path)
    assert saved.shape==(len(test),2) and saved.columns.equals(submission_template.columns)
    pd.testing.assert_series_equal(saved[id_col],submission_template[id_col])
    assert saved[target_col].notna().all() and saved[target_col].between(0,1).all()
    assert pd.api.types.is_float_dtype(saved[target_col])
    assert ((saved[target_col]>0)&(saved[target_col]<1)).any()
    np.testing.assert_allclose(saved[target_col],p,rtol=1e-12,atol=1e-15)
    print("Submission assertions passed:",output_path,saved.shape)
    return result
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
Titanic 28 Fold 1: AUC=0.908771930, Best Iteration=150
Titanic 28 Fold 2: AUC=0.929570303, Best Iteration=70
Titanic 28 Fold 3: AUC=0.895245360, Best Iteration=9
Titanic 28 Fold 4: AUC=0.919590643, Best Iteration=2
Titanic 28 Fold 5: AUC=0.918446479, Best Iteration=20
Best iteration distribution: [150, 70, 9, 2, 20]
Full-train iterations = CV median: 20 (no override)
Titanic 28 recorded result reproduced.
Titanic 29 Fold 1: AUC=0.901879699, Best Iteration=332
Titanic 29 Fold 2: AUC=0.922133232, Best Iteration=4
Titanic 29 Fold 3: AUC=0.896262395, Best Iteration=4
Titanic 29 Fold 4: AUC=0.906114925, Best Iteration=92
Titanic 29 Fold 5: AUC=0.911772184, Best Iteration=8
Best iteration distribution: [332, 4, 4, 92, 8]
Full-train iterations = CV median: 8 (no override)


   fold  train_auc  validation_auc       gap  best_iteration
0     1   0.943578        0.908772  0.034806             150
1     2   0.929983        0.929570  0.000413              70
2     3   0.923238        0.895245  0.027992               9
3     4   0.909791        0.919591 -0.009800               2
4     5   0.921543        0.918446  0.003097              20

   fold  train_auc  validation_auc       gap  best_iteration
0     1   0.934580        0.901880  0.032700             332
1     2   0.897480        0.922133 -0.024653               4
2     3   0.911972        0.896262  0.015710               4
3     4   0.918808        0.906115  0.012693              92
4     5   0.908722        0.911772 -0.003050               8

CV Mean change vs Titanic 28: -0.006692455777123851
Submission assertions passed: titanic_29_result.csv (393, 2)

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_3.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.feature_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission = pd.read_csv('csv/submission.csv')
previous_features = []
new_feature = 'Title'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9017752715121137, 'std': 0.016321824128332087}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
raw_titles = eda_prep.titles(X_tr_raw)
display(pd.DataFrame({'Title':raw_titles,'survived':y_tr}).groupby('Title')['survived'].agg(['count','mean']))
display(pd.crosstab(eda_frame['Title'], X_tr_raw['gender']))
display(eda_frame.groupby('Title')['age'].agg(['count','median']))
display(eda_frame.groupby(new_feature,dropna=False)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 24: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 26: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] > previous_cv['mean']
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 28: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.963079   0.965168
validation_auc           0.900065       0.900065   0.902992
gap                      0.063014       0.063014   0.062176
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_30.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 1: from pathlib import Path
import random
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from common.interaction_experiments import FeaturePreprocessor
from common.feature_experiments import baseline_parameters

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission_template = pd.read_csv('csv/submission.csv')

target_col = 'survived'
id_col = 'passengerid'
BASE_FEATURES = ['GenderClass', 'GenderIsChild', 'ClassIsChild', 'AgeBand']
X = train.drop(columns=[target_col, id_col])
y = train[target_col].astype('int8')
X_test_raw = test.drop(columns=[id_col])

assert X.columns.equals(X_test_raw.columns)
assert train[id_col].is_unique and test[id_col].is_unique
assert set(train[id_col]).isdisjoint(test[id_col])
train_part, valid_part = train_test_split(
    train, test_size=0.25, stratify=train[target_col], random_state=SEED
)
X_tr_raw = train_part.drop(columns=[target_col, id_col])
y_tr = train_part[target_col].astype('int8')
X_valid_raw = valid_part.drop(columns=[target_col, id_col])
y_valid = valid_part[target_col].astype('int8')
assert len(train_part) == 687 and len(valid_part) == 229

BASE_PARAMS = baseline_parameters()
assert BASE_PARAMS['random_state'] == SEED
print('Titanic 15 명시 파라미터:', BASE_PARAMS)
print('고정 피처:', BASE_FEATURES)
import sys
import catboost, sklearn
print("Runtime:", sys.version)
print("Versions:", catboost.__version__, sklearn.__version__, pd.__version__)
PARAMS = {
 '28': {'learning_rate': 0.03},
 '29': {'learning_rate': 0.03, 'depth': 4, 'l2_leaf_reg': 20},
 '30': {'learning_rate': 0.03, 'random_strength': 0.2, 'subsample': 0.8, 'rsm': 1.0},
 'A': {'learning_rate': 0.03, 'depth': 4, 'l2_leaf_reg': 20,
       'random_strength': 0.2, 'subsample': 0.8, 'rsm': 1.0},
}
SPLITS=list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X,y))
# Declare practical stability criteria before seeing results.
MIN_GAIN=0.001
MAX_STD_INCREASE=0.003
MAX_GAP_INCREASE=0.01
MAX_HOLDOUT_DROP=0.005
MIN_IMPROVED_FOLDS=3
MAX_SINGLE_FOLD_SHARE=0.60

cell 4: def prepare_fold(X_train_raw, X_valid_raw):
    prep = FeaturePreprocessor(BASE_FEATURES)
    X_train_model = prep.fit_transform(X_train_raw)
    X_valid_model = prep.transform(X_valid_raw)
    assert X_train_model.columns.equals(X_valid_model.columns)
    assert np.isfinite(X_train_model.to_numpy()).all()
    assert np.isfinite(X_valid_model.to_numpy()).all()
    return prep, X_train_model, X_valid_model


def fit_model(X_train_model, y_train, overrides=None, **fit_kwargs):
    params = BASE_PARAMS | (overrides or {})
    model = CatBoostClassifier(**params)
    model.fit(X_train_model, y_train, **fit_kwargs)
    return model


def auc_scores(model, X_train_model, y_train, X_valid_model, y_valid):
    positive_index = list(model.classes_).index(1)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train_model)[:, positive_index])
    valid_auc = roc_auc_score(y_valid, model.predict_proba(X_valid_model)[:, positive_index])
    return train_auc, valid_auc, train_auc - valid_auc


def positive(model, values):
    assert model.get_all_params()['random_seed']==SEED
    return model.predict_proba(values)[:,list(model.classes_).index(1)]

def evaluate(label):
    rows, train_predictions=[],[]
    oof=np.full(len(y),np.nan)
    visits=np.zeros(len(y),dtype=int)
    params=PARAMS[label]|{'iterations':3000,'loss_function':'Logloss','eval_metric':'AUC'}
    for fold,(tr,va) in enumerate(SPLITS,1):
        assert not set(tr).intersection(va)
        _,xt,xv=prepare_fold(X.iloc[tr],X.iloc[va])
        model=fit_model(xt,y.iloc[tr],params,eval_set=(xv,y.iloc[va]),
                        early_stopping_rounds=150,use_best_model=True,verbose=False)
        pt,pv=positive(model,xt),positive(model,xv)
        oof[va]=pv
        visits[va]+=1
        train_predictions.append(pt)
        ta,vauc=roc_auc_score(y.iloc[tr],pt),roc_auc_score(y.iloc[va],pv)
        best_iteration=int(model.get_best_iteration())+1
        assert best_iteration==model.tree_count_ and 1<=best_iteration<=3000
        rows.append(dict(fold=fold,train_auc=ta,validation_auc=vauc,gap=ta-vauc,best_iteration=best_iteration))
        print(f"Titanic {label} Fold {fold}: AUC={vauc:.9f}, Best Iteration={best_iteration}")
    assert (visits==1).all() and np.isfinite(oof).all()
    folds=pd.DataFrame(rows)
    iterations=int(folds.best_iteration.median())
    assert 1<=iterations<=3000
    _,ht,hv=prepare_fold(X_tr_raw,X_valid_raw)
    hm=fit_model(ht,y_tr,params,eval_set=(hv,y_valid),early_stopping_rounds=150,
                 use_best_model=True,verbose=False)
    hp_train,hp_valid=positive(hm,ht),positive(hm,hv)
    ht_auc,hv_auc=roc_auc_score(y_tr,hp_train),roc_auc_score(y_valid,hp_valid)
    summary={
      'Model':label,'CV Mean':float(folds.validation_auc.mean()),
      'CV Std':float(folds.validation_auc.std(ddof=1)),'OOF AUC':float(roc_auc_score(y,oof)),
      'Holdout AUC':float(hv_auc),'Train/Validation Gap':float(ht_auc-hv_auc),
      'CV Gap':float(folds.gap.mean()),'Best Iteration Mean':float(folds.best_iteration.mean()),
      'Best Iteration Median':float(folds.best_iteration.median()),'Final Iterations':iterations,
      'Parameter':str(BASE_PARAMS|params|{'iterations':iterations})}
    print("Best iteration distribution:",folds.best_iteration.tolist())
    print("Full-train iterations = CV median:",iterations,"(no override)")
    display(folds)
    display(pd.DataFrame([summary]))
    if label=='28':
        np.testing.assert_allclose(summary['CV Mean'],0.914324942791762,atol=1e-12,rtol=0)
        np.testing.assert_allclose(summary['CV Std'],0.012961501591864434,atol=1e-12,rtol=0)
        assert iterations==20
        print("Titanic 28 recorded result reproduced.")
    return dict(summary=summary,folds=folds,oof=oof,train_predictions=train_predictions,
                holdout_train=hp_train,holdout_valid=hp_valid,iterations=iterations)

def full_prediction(label,evaluation):
    prep=FeaturePreprocessor(BASE_FEATURES)
    full=prep.fit_transform(X)
    test_model=prep.transform(X_test_raw)
    assert full.columns.equals(test_model.columns)
    params=PARAMS[label]|{'iterations':evaluation['iterations'],'loss_function':'Logloss','eval_metric':'AUC'}
    model=fit_model(full,y,params)
    assert model.tree_count_==evaluation['iterations']
    return positive(model,test_model)

def save_submission(predictions,number):
    assert submission_template.columns.tolist()==[id_col,target_col]
    assert len(submission_template)==len(test)
    assert submission_template[id_col].is_unique and submission_template[id_col].notna().all()
    assert set(submission_template[id_col])==set(test[id_col])
    assert predictions.shape==(len(test),)
    by_id=pd.Series(predictions,index=test[id_col].to_numpy())
    result=submission_template.copy(deep=True)
    result[target_col]=result[id_col].map(by_id)
    assert result.shape==submission_template.shape and result.columns.equals(submission_template.columns)
    pd.testing.assert_series_equal(result[id_col],submission_template[id_col],check_names=True)
    p=result[target_col].to_numpy()
    assert pd.api.types.is_float_dtype(result[target_col])
    assert np.isfinite(p).all() and result[target_col].between(0,1).all()
    assert ((p>0)&(p<1)).any(),'Hard labels forbidden'
    output_path=Path(f'titanic_{number}_result.csv')
    result.to_csv(output_path,index=False)
    saved=pd.read_csv(output_path)
    assert saved.shape==(len(test),2) and saved.columns.equals(submission_template.columns)
    pd.testing.assert_series_equal(saved[id_col],submission_template[id_col])
    assert saved[target_col].notna().all() and saved[target_col].between(0,1).all()
    assert pd.api.types.is_float_dtype(saved[target_col])
    assert ((saved[target_col]>0)&(saved[target_col]<1)).any()
    np.testing.assert_allclose(saved[target_col],p,rtol=1e-12,atol=1e-15)
    print("Submission assertions passed:",output_path,saved.shape)
    return result
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
Titanic 28 Fold 1: AUC=0.908771930, Best Iteration=150
Titanic 28 Fold 2: AUC=0.929570303, Best Iteration=70
Titanic 28 Fold 3: AUC=0.895245360, Best Iteration=9
Titanic 28 Fold 4: AUC=0.919590643, Best Iteration=2
Titanic 28 Fold 5: AUC=0.918446479, Best Iteration=20
Best iteration distribution: [150, 70, 9, 2, 20]
Full-train iterations = CV median: 20 (no override)
Titanic 28 recorded result reproduced.
Titanic 30 Fold 1: AUC=0.909335840, Best Iteration=51
Titanic 30 Fold 2: AUC=0.930142385, Best Iteration=63
Titanic 30 Fold 3: AUC=0.893465548, Best Iteration=50
Titanic 30 Fold 4: AUC=0.914124078, Best Iteration=28
Titanic 30 Fold 5: AUC=0.911263666, Best Iteration=1
Best iteration distribution: [51, 63, 50, 28, 1]
Full-train iterations = CV median: 50 (no override)


   fold  train_auc  validation_auc       gap  best_iteration
0     1   0.943578        0.908772  0.034806             150
1     2   0.929983        0.929570  0.000413              70
2     3   0.923238        0.895245  0.027992               9
3     4   0.909791        0.919591 -0.009800               2
4     5   0.921543        0.918446  0.003097              20

   fold  train_auc  validation_auc       gap  best_iteration
0     1   0.937127        0.909336  0.027791              51
1     2   0.941090        0.930142  0.010948              63
2     3   0.943889        0.893466  0.050423              50
3     4   0.929045        0.914124  0.014921              28
4     5   0.920878        0.911264  0.009615               1

CV Mean change vs Titanic 28: -0.002658639352003278
Submission assertions passed: titanic_30_result.csv (393, 2)

~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_31.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 1: from pathlib import Path
import random
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from common.interaction_experiments import FeaturePreprocessor
from common.feature_experiments import baseline_parameters

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission_template = pd.read_csv('csv/submission.csv')

target_col = 'survived'
id_col = 'passengerid'
BASE_FEATURES = ['GenderClass', 'GenderIsChild', 'ClassIsChild', 'AgeBand']
X = train.drop(columns=[target_col, id_col])
y = train[target_col].astype('int8')
X_test_raw = test.drop(columns=[id_col])

assert X.columns.equals(X_test_raw.columns)
assert train[id_col].is_unique and test[id_col].is_unique
assert set(train[id_col]).isdisjoint(test[id_col])
train_part, valid_part = train_test_split(
    train, test_size=0.25, stratify=train[target_col], random_state=SEED
)
X_tr_raw = train_part.drop(columns=[target_col, id_col])
y_tr = train_part[target_col].astype('int8')
X_valid_raw = valid_part.drop(columns=[target_col, id_col])
y_valid = valid_part[target_col].astype('int8')
assert len(train_part) == 687 and len(valid_part) == 229

BASE_PARAMS = baseline_parameters()
assert BASE_PARAMS['random_state'] == SEED
print('Titanic 15 명시 파라미터:', BASE_PARAMS)
print('고정 피처:', BASE_FEATURES)
import sys
import catboost, sklearn
print("Runtime:", sys.version)
print("Versions:", catboost.__version__, sklearn.__version__, pd.__version__)
PARAMS = {
 '28': {'learning_rate': 0.03},
 '29': {'learning_rate': 0.03, 'depth': 4, 'l2_leaf_reg': 20},
 '30': {'learning_rate': 0.03, 'random_strength': 0.2, 'subsample': 0.8, 'rsm': 1.0},
 'A': {'learning_rate': 0.03, 'depth': 4, 'l2_leaf_reg': 20,
       'random_strength': 0.2, 'subsample': 0.8, 'rsm': 1.0},
}
SPLITS=list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X,y))
# Declare practical stability criteria before seeing results.
MIN_GAIN=0.001
MAX_STD_INCREASE=0.003
MAX_GAP_INCREASE=0.01
MAX_HOLDOUT_DROP=0.005
MIN_IMPROVED_FOLDS=3
MAX_SINGLE_FOLD_SHARE=0.60

cell 4: def prepare_fold(X_train_raw, X_valid_raw):
    prep = FeaturePreprocessor(BASE_FEATURES)
    X_train_model = prep.fit_transform(X_train_raw)
    X_valid_model = prep.transform(X_valid_raw)
    assert X_train_model.columns.equals(X_valid_model.columns)
    assert np.isfinite(X_train_model.to_numpy()).all()
    assert np.isfinite(X_valid_model.to_numpy()).all()
    return prep, X_train_model, X_valid_model


def fit_model(X_train_model, y_train, overrides=None, **fit_kwargs):
    params = BASE_PARAMS | (overrides or {})
    model = CatBoostClassifier(**params)
    model.fit(X_train_model, y_train, **fit_kwargs)
    return model


def auc_scores(model, X_train_model, y_train, X_valid_model, y_valid):
    positive_index = list(model.classes_).index(1)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train_model)[:, positive_index])
    valid_auc = roc_auc_score(y_valid, model.predict_proba(X_valid_model)[:, positive_index])
    return train_auc, valid_auc, train_auc - valid_auc


def positive(model, values):
    assert model.get_all_params()['random_seed']==SEED
    return model.predict_proba(values)[:,list(model.classes_).index(1)]

def evaluate(label):
    rows, train_predictions=[],[]
    oof=np.full(len(y),np.nan)
    visits=np.zeros(len(y),dtype=int)
    params=PARAMS[label]|{'iterations':3000,'loss_function':'Logloss','eval_metric':'AUC'}
    for fold,(tr,va) in enumerate(SPLITS,1):
        assert not set(tr).intersection(va)
        _,xt,xv=prepare_fold(X.iloc[tr],X.iloc[va])
        model=fit_model(xt,y.iloc[tr],params,eval_set=(xv,y.iloc[va]),
                        early_stopping_rounds=150,use_best_model=True,verbose=False)
        pt,pv=positive(model,xt),positive(model,xv)
        oof[va]=pv
        visits[va]+=1
        train_predictions.append(pt)
        ta,vauc=roc_auc_score(y.iloc[tr],pt),roc_auc_score(y.iloc[va],pv)
        best_iteration=int(model.get_best_iteration())+1
        assert best_iteration==model.tree_count_ and 1<=best_iteration<=3000
        rows.append(dict(fold=fold,train_auc=ta,validation_auc=vauc,gap=ta-vauc,best_iteration=best_iteration))
        print(f"Titanic {label} Fold {fold}: AUC={vauc:.9f}, Best Iteration={best_iteration}")
    assert (visits==1).all() and np.isfinite(oof).all()
    folds=pd.DataFrame(rows)
    iterations=int(folds.best_iteration.median())
    assert 1<=iterations<=3000
    _,ht,hv=prepare_fold(X_tr_raw,X_valid_raw)
    hm=fit_model(ht,y_tr,params,eval_set=(hv,y_valid),early_stopping_rounds=150,
                 use_best_model=True,verbose=False)
    hp_train,hp_valid=positive(hm,ht),positive(hm,hv)
    ht_auc,hv_auc=roc_auc_score(y_tr,hp_train),roc_auc_score(y_valid,hp_valid)
    summary={
      'Model':label,'CV Mean':float(folds.validation_auc.mean()),
      'CV Std':float(folds.validation_auc.std(ddof=1)),'OOF AUC':float(roc_auc_score(y,oof)),
      'Holdout AUC':float(hv_auc),'Train/Validation Gap':float(ht_auc-hv_auc),
      'CV Gap':float(folds.gap.mean()),'Best Iteration Mean':float(folds.best_iteration.mean()),
      'Best Iteration Median':float(folds.best_iteration.median()),'Final Iterations':iterations,
      'Parameter':str(BASE_PARAMS|params|{'iterations':iterations})}
    print("Best iteration distribution:",folds.best_iteration.tolist())
    print("Full-train iterations = CV median:",iterations,"(no override)")
    display(folds)
    display(pd.DataFrame([summary]))
    if label=='28':
        np.testing.assert_allclose(summary['CV Mean'],0.914324942791762,atol=1e-12,rtol=0)
        np.testing.assert_allclose(summary['CV Std'],0.012961501591864434,atol=1e-12,rtol=0)
        assert iterations==20
        print("Titanic 28 recorded result reproduced.")
    return dict(summary=summary,folds=folds,oof=oof,train_predictions=train_predictions,
                holdout_train=hp_train,holdout_valid=hp_valid,iterations=iterations)

def full_prediction(label,evaluation):
    prep=FeaturePreprocessor(BASE_FEATURES)
    full=prep.fit_transform(X)
    test_model=prep.transform(X_test_raw)
    assert full.columns.equals(test_model.columns)
    params=PARAMS[label]|{'iterations':evaluation['iterations'],'loss_function':'Logloss','eval_metric':'AUC'}
    model=fit_model(full,y,params)
    assert model.tree_count_==evaluation['iterations']
    return positive(model,test_model)

def save_submission(predictions,number):
    assert submission_template.columns.tolist()==[id_col,target_col]
    assert len(submission_template)==len(test)
    assert submission_template[id_col].is_unique and submission_template[id_col].notna().all()
    assert set(submission_template[id_col])==set(test[id_col])
    assert predictions.shape==(len(test),)
    by_id=pd.Series(predictions,index=test[id_col].to_numpy())
    result=submission_template.copy(deep=True)
    result[target_col]=result[id_col].map(by_id)
    assert result.shape==submission_template.shape and result.columns.equals(submission_template.columns)
    pd.testing.assert_series_equal(result[id_col],submission_template[id_col],check_names=True)
    p=result[target_col].to_numpy()
    assert pd.api.types.is_float_dtype(result[target_col])
    assert np.isfinite(p).all() and result[target_col].between(0,1).all()
    assert ((p>0)&(p<1)).any(),'Hard labels forbidden'
    output_path=Path(f'titanic_{number}_result.csv')
    result.to_csv(output_path,index=False)
    saved=pd.read_csv(output_path)
    assert saved.shape==(len(test),2) and saved.columns.equals(submission_template.columns)
    pd.testing.assert_series_equal(saved[id_col],submission_template[id_col])
    assert saved[target_col].notna().all() and saved[target_col].between(0,1).all()
    assert pd.api.types.is_float_dtype(saved[target_col])
    assert ((saved[target_col]>0)&(saved[target_col]<1)).any()
    np.testing.assert_allclose(saved[target_col],p,rtol=1e-12,atol=1e-15)
    print("Submission assertions passed:",output_path,saved.shape)
    return result
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
Titanic 28 Fold 1: AUC=0.908771930, Best Iteration=150
Titanic 28 Fold 2: AUC=0.929570303, Best Iteration=70
Titanic 28 Fold 3: AUC=0.895245360, Best Iteration=9
Titanic 28 Fold 4: AUC=0.919590643, Best Iteration=2
Titanic 28 Fold 5: AUC=0.918446479, Best Iteration=20
Best iteration distribution: [150, 70, 9, 2, 20]
Full-train iterations = CV median: 20 (no override)
Titanic 28 recorded result reproduced.
Titanic 29 Fold 1: AUC=0.901879699, Best Iteration=332
Titanic 29 Fold 2: AUC=0.922133232, Best Iteration=4
Titanic 29 Fold 3: AUC=0.896262395, Best Iteration=4
Titanic 29 Fold 4: AUC=0.906114925, Best Iteration=92
Titanic 29 Fold 5: AUC=0.911772184, Best Iteration=8
Best iteration distribution: [332, 4, 4, 92, 8]
Full-train iterations = CV median: 8 (no override)
Titanic 30 Fold 1: AUC=0.909335840, Best Iteration=51
Titanic 30 Fold 2: AUC=0.930142385, Best Iteration=63
Titanic 30 Fold 3: AUC=0.893465548, Best Iteration=50
Titanic 30 Fold 4: AUC=0.914124078, Best Iteration=28
Titanic 30 Fold 5: AUC=0.911263666, Best Iteration=1
Best iteration distribution: [51, 63, 50, 28, 1]
Full-train iterations = CV median: 50 (no override)
Verified actual previous execution: 29
Verified actual previous execution: 30


   fold  train_auc  validation_auc       gap  best_iteration
0     1   0.943578        0.908772  0.034806             150
1     2   0.929983        0.929570  0.000413              70
2     3   0.923238        0.895245  0.027992               9
3     4   0.909791        0.919591 -0.009800               2
4     5   0.921543        0.918446  0.003097              20

   fold  train_auc  validation_auc       gap  best_iteration
0     1   0.934580        0.901880  0.032700             332
1     2   0.897480        0.922133 -0.024653               4
2     3   0.911972        0.896262  0.015710               4
3     4   0.918808        0.906115  0.012693              92
4     5   0.908722        0.911772 -0.003050               8

   fold  train_auc  validation_auc       gap  best_iteration
0     1   0.937127        0.909336  0.027791              51
1     2   0.941090        0.930142  0.010948              63
2     3   0.943889        0.893466  0.050423              50
3     4   0.929045        0.914124  0.014921              28
4     5   0.920878        0.911264  0.009615               1

Titanic A Fold 1: AUC=0.903634085, Best Iteration=80
Titanic A Fold 2: AUC=0.925819985, Best Iteration=7
Titanic A Fold 3: AUC=0.891113654, Best Iteration=262
Titanic A Fold 4: AUC=0.920480549, Best Iteration=1
Titanic A Fold 5: AUC=0.915713196, Best Iteration=448
Best iteration distribution: [80, 7, 262, 1, 448]
Full-train iterations = CV median: 80 (no override)


   fold  train_auc  validation_auc       gap  best_iteration
0     1   0.922074        0.903634  0.018440              80
1     2   0.908576        0.925820 -0.017244               7
2     3   0.938482        0.891114  0.047368             262
3     4   0.903184        0.920481 -0.017296               1
4     5   0.942626        0.915713  0.026913             448

Best coarse-grid weights (28,29,30): [0.5, 0.3, 0.2]
Best grid OOF AUC: 0.8890249467599636
Simple Average OOF AUC: 0.883766859344894
Candidate B weights: [0.5, 0.3, 0.2]
Blend OOF AUC: 0.8890249467599636


    w28  w29  w30   oof_auc  active
0   0.5  0.3  0.2  0.889025       3
1   0.6  0.3  0.1  0.887605       3
2   0.4  0.3  0.3  0.886987       3
3   0.5  0.2  0.3  0.886865       3
4   0.5  0.4  0.1  0.886667       3
..  ...  ...  ...       ...     ...
61  0.1  0.9  0.0  0.872690       2
62  0.2  0.8  0.0  0.872538       2
63  0.0  0.6  0.4  0.871544       2
64  0.0  1.0  0.0  0.869942       1
65  0.0  0.7  0.3  0.869765       2

[66 rows x 5 columns]

   fold  train_auc  validation_auc       gap
0     1   0.941167        0.905013  0.036154
1     2   0.934155        0.929252  0.004903
2     3   0.937544        0.893466  0.044078
3     4   0.923808        0.906814  0.016994
4     5   0.920012        0.916603  0.003408

  Model   OOF AUC
0    28  0.878288
1    29  0.869942
2    30  0.886363
3     B  0.889025

A paired fold AUC differences: [-0.0051378446115286636, -0.0037503178235444024, -0.004131706076786257, 0.0008899059242308827, -0.002733282481566346] largest positive contribution: 1.0
B paired fold AUC differences: [-0.003759398496240518, -0.0003178235443681565, -0.0017798118484617653, -0.012776506483600292, -0.0018433765573355743] largest positive contribution: 1.0
Selected final method: 28
Thresholds declared before fitting; no test labels or Public Score used.
OOF early stopping/weight selection and reused holdout introduce selection optimism.
Scores are development estimates, not an independent final assessment.
Submission assertions passed: titanic_31_result.csv (393, 2)
FINAL REPORT: {'summaries': [{'Model': '28', 'CV Mean': 0.914324942791762, 'CV Std': 0.012961501591864433, 'OOF AUC': 0.8782882060642936, 'Holdout AUC': 0.9063668889250285, 'Train/Validation Gap': 0.019119509922026157, 'CV Gap': 0.011301714594916024, 'Best Iteration Mean': 50.2, 'Best Iteration Median': 20.0, 'Final Iterations': 20, 'Parameter': "{'verbose': 0, 'random_state': 42, 'cat_features': [], 'allow_writing_files': False, 'learning_rate': 0.03, 'iterations': 20, 'loss_function': 'Logloss', 'eval_metric': 'AUC'}"}, {'Model': '29', 'CV Mean': 0.9076324870146382, 'CV Std': 0.009900571164358699, 'OOF AUC': 0.869942196531792, 'Holdout AUC': 0.8988453407058059, 'Train/Validation Gap': 0.03105467730896616, 'CV Gap': 0.006679959115992218, 'Best Iteration Mean': 88.0, 'Best Iteration Median': 8.0, 'Final Iterations': 8, 'Parameter': "{'verbose': 0, 'random_state': 42, 'cat_features': [], 'allow_writing_files': False, 'learning_rate': 0.03, 'depth': 4, 'l2_leaf_reg': 20, 'iterations': 8, 'loss_function': 'Logloss', 'eval_metric': 'AUC'}"}, {'Model': '30', 'CV Mean': 0.9116663034397587, 'CV Std': 0.013079248112097427, 'OOF AUC': 0.8863629449345909, 'Holdout AUC': 0.9051065213855911, 'Train/Validation Gap': 0.021104972039017178, 'CV Gap': 0.02273950077543041, 'Best Iteration Mean': 38.6, 'Best Iteration Median': 50.0, 'Final Iterations': 50, 'Parameter': "{'verbose': 0, 'random_state': 42, 'cat_features': [], 'allow_writing_files': False, 'learning_rate': 0.03, 'random_strength': 0.2, 'subsample': 0.8, 'rsm': 1.0, 'iterations': 50, 'loss_function': 'Logloss', 'eval_metric': 'AUC'}"}, {'Model': 'A', 'CV Mean': 0.9113522937779232, 'CV Std': 0.01397166680372443, 'OOF AUC': 0.8793504715546091, 'Holdout AUC': 0.9033582696373395, 'Train/Validation Gap': 0.05280278242535186, 'CV Gap': 0.011635966371089036, 'Best Iteration Mean': 159.6, 'Best Iteration Median': 80.0, 'Final Iterations': 80, 'Parameter': "{'verbose': 0, 'random_state': 42, 'cat_features': [], 'allow_writing_files': False, 'learning_rate': 0.03, 'depth': 4, 'l2_leaf_reg': 20, 'random_strength': 0.2, 'subsample': 0.8, 'rsm': 1.0, 'iterations': 80, 'loss_function': 'Logloss', 'eval_metric': 'AUC'}"}, {'Model': 'B', 'CV Mean': 0.9102295594057608, 'CV Std': 0.013438825189363608, 'OOF AUC': 0.8890249467599636, 'Holdout AUC': 0.903846153846154, 'Train/Validation Gap': 0.02603134570347676, 'CV Gap': 0.021107403197078645, 'Final Iterations': "{'28': 20, '29': 8, '30': 50}", 'Parameter': "{'28': 0.5, '29': 0.3, '30': 0.2}"}], 'fold_auc': {'28': [0.9087719298245613, 0.9295703025680143, 0.8952453597762523, 0.9195906432748538, 0.9184464785151285], '29': [0.9018796992481203, 0.922133231629799, 0.8962623951182305, 0.9061149249936437, 0.9117721840833969], '30': [0.9093358395989976, 0.930142384947877, 0.8934655479277904, 0.9141240783117214, 0.9112636664124076], 'A': [0.9036340852130327, 0.9258199847444699, 0.891113653699466, 0.9204805491990847, 0.9157131960335622], 'B': [0.9050125313283208, 0.9292524790236462, 0.8934655479277905, 0.9068141367912536, 0.9166031019577929]}, 'best_iterations': {'28': [150, 70, 9, 2, 20], '29': [332, 4, 4, 92, 8], '30': [51, 63, 50, 28, 1], 'A': [80, 7, 262, 1, 448]}, 'selected': '28', 'weights': None, 'final_iterations': {'28': 20}, 'best_grid_weights': [0.5, 0.3, 0.2], 'candidate_B_weights': [0.5, 0.3, 0.2], 'simple_average_oof_auc': 0.883766859344894, 'random_state': 42}


  Candidate  CV gain >= 0.001  ...  Holdout diagnostic drop <= 0.005  Eligible
0         A             False  ...                              True     False
1         B             False  ...                              True     False

[2 rows x 9 columns]
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_4.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.feature_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission = pd.read_csv('csv/submission.csv')
previous_features = []
new_feature = 'SexPclass'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9017752715121136, 'std': 0.016321824128332073}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
display(train_part.groupby(['gender','pclass'])[target_col].agg(['count','mean']))
display(eda_frame.groupby(new_feature,dropna=False)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 24: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 26: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] > previous_cv['mean']
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 28: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.963079   0.963047
validation_auc           0.900065       0.900065   0.902911
gap                      0.063014       0.063014   0.060136
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_5.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.feature_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission = pd.read_csv('csv/submission.csv')
previous_features = ['SexPclass']
new_feature = 'FamilyGroup'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9046761832116523, 'std': 0.01613237786295489}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
display(eda_frame.groupby('FamilySize')[target_col].agg(['count','mean']))
display(pd.crosstab(eda_frame['FamilyGroup'], eda_frame['IsAlone']))
display(eda_frame.groupby(new_feature,dropna=False)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 24: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 26: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] > previous_cv['mean']
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 28: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.963047   0.964434
validation_auc           0.900065       0.902911   0.902748
gap                      0.063014       0.060136   0.061686
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_6.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.feature_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission = pd.read_csv('csv/submission.csv')
previous_features = ['SexPclass']
new_feature = 'TicketGroupSize'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9046761832116523, 'std': 0.01613237786295489}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
display(eda_frame.groupby(new_feature,dropna=False)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 24: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 26: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] > previous_cv['mean']
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 28: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.963047   0.964691
validation_auc           0.900065       0.902911   0.899984
gap                      0.063014       0.060136   0.064707
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_7.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.feature_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission = pd.read_csv('csv/submission.csv')
previous_features = ['SexPclass']
new_feature = 'FarePerPerson'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9046761832116523, 'std': 0.01613237786295489}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
groups = pd.qcut(eda_frame[new_feature], 4, duplicates='drop')
display(eda_frame.groupby(groups,observed=True)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 24: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 26: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] > previous_cv['mean']
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 28: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.963047   0.965835
validation_auc           0.900065       0.902911   0.902748
gap                      0.063014       0.060136   0.063087
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_8.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.feature_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission = pd.read_csv('csv/submission.csv')
previous_features = ['SexPclass']
new_feature = 'IsChild'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9046761832116523, 'std': 0.01613237786295489}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
age_band = pd.cut(X_tr_raw['age'], [0,5,12,16,25,40,60,100], right=False)
display(pd.DataFrame({'AgeBand':age_band,'survived':y_tr}).groupby('AgeBand',observed=True)['survived'].agg(['count','mean']))
display(eda_frame.groupby(new_feature,dropna=False)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 24: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 26: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] > previous_cv['mean']
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 28: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.963047   0.964218
validation_auc           0.900065       0.902911   0.903155
gap                      0.063014       0.060136   0.061063
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_9.ipynb — stored plots 0, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python
cell 2: from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from common.utils import train_test_split_by_target
from common.feature_experiments import FeaturePreprocessor, fit_cat, scores, cv_compare, baseline_parameters

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')
submission = pd.read_csv('csv/submission.csv')
previous_features = ['SexPclass']
new_feature = 'CabinDeck'
candidate_features = previous_features + [new_feature]
previous_cv = {'mean': 0.9046761832116523, 'std': 0.01613237786295489}
print(train.shape, test.shape)
display(train.head())
display(test.head())

cell 10: eda_prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
eda_frame = eda_prep.feature_frame(X_tr_raw, eda_prep.transform_missing(X_tr_raw))
eda_frame[target_col] = y_tr
display(pd.crosstab(eda_frame['CabinDeck'], eda_frame['pclass']))
display(eda_frame.groupby(new_feature,dropna=False)[target_col].agg(['count','mean']))
print('EDA 표본 수:', len(eda_frame))

cell 13: prep = FeaturePreprocessor(candidate_features).fit_missing(X_tr_raw)
X_tr_clean = prep.transform_missing(X_tr_raw)
X_valid_clean = prep.transform_missing(X_valid_raw)
assert not X_tr_clean.isna().any().any()
assert not X_valid_clean.isna().any().any()

cell 15: X_tr_features = prep.feature_frame(X_tr_raw, X_tr_clean)
X_valid_features = prep.feature_frame(X_valid_raw, X_valid_clean)
print('추가 피처:', candidate_features)
print('인코딩 전:', X_tr_features.columns.tolist())
display(X_tr_features.head())

cell 24: reference_prep = FeaturePreprocessor()
reference_X = reference_prep.fit_transform(X_tr_raw)
reference_valid = reference_prep.transform(X_valid_raw)
reference_model = fit_cat(reference_X, y_tr)
reference_scores = scores(reference_model, reference_X, y_tr, reference_valid, y_valid)
assert abs(reference_scores['validation_auc'] - 0.900065) < 0.000001
previous_prep = FeaturePreprocessor(previous_features)
previous_X = previous_prep.fit_transform(X_tr_raw)
previous_valid = previous_prep.transform(X_valid_raw)
previous_model = reference_model if not previous_features else fit_cat(previous_X, y_tr)
previous_scores = scores(previous_model, previous_X, y_tr, previous_valid, y_valid)
candidate_model = fit_cat(X_tr_model, y_tr)
assert candidate_model.get_all_params() == reference_model.get_all_params()
assert previous_model.get_all_params() == reference_model.get_all_params()
print('명시적 설정:', baseline_parameters())
print('실효 설정:', candidate_model.get_all_params())

cell 26: candidate_scores = scores(candidate_model, X_tr_model, y_tr, X_valid_model, y_valid)
holdout_delta = candidate_scores['validation_auc'] - previous_scores['validation_auc']
original_delta = candidate_scores['validation_auc'] - reference_scores['validation_auc']
display(pd.DataFrame({'Original baseline':reference_scores, 'Previous best':previous_scores, 'Candidate':candidate_scores}))
print('이전 Best 대비:', holdout_delta, '최초 baseline 대비:', original_delta)
candidate_cv = None
cv_folds = None
if holdout_delta > 0:
    cv_folds, cv_table = cv_compare(X, y, previous_features, candidate_features)
    display(cv_folds.pivot(index='fold', columns='model', values='auc'))
    display(cv_table)
    previous_cv = cv_table.loc['Previous best'].to_dict()
    candidate_cv = cv_table.loc['Candidate'].to_dict()
accepted = (holdout_delta > 0 and candidate_cv is not None
            and candidate_cv['mean'] > previous_cv['mean']
            and candidate_scores['gap'] <= previous_scores['gap'] + 0.01)
best_features = candidate_features if accepted else previous_features
best_scores = candidate_scores if accepted else previous_scores
best_cv = candidate_cv if accepted else previous_cv
print('채택:', accepted, '다음 버전의 Best:', best_features)

cell 28: full_reference_prep = FeaturePreprocessor()
full_reference_X = full_reference_prep.fit_transform(X)
full_reference_model = fit_cat(full_reference_X, y)
final_prep = FeaturePreprocessor(candidate_features)
X_full_model = final_prep.fit_transform(X)
submission_model = fit_cat(X_full_model, y)
assert submission_model.get_all_params() == full_reference_model.get_all_params()
print('이 버전 CSV의 피처:', X_full_model.columns.tolist())
print('최종 학습 실효 파라미터:', submission_model.get_all_params())
~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
                Original baseline  Previous best  Candidate
train_auc                0.963079       0.963047   0.964011
validation_auc           0.900065       0.902911   0.904781
gap                      0.063014       0.060136   0.059230
~~~

</details>


<details><summary>C:\dev\study\machine_learning\titanic_base_smote_experiment.ipynb — stored plots 1, stored errors 1</summary>

실행 오류 기록: ["NameError: name 'submission' is not defined"]

설정 근거:

~~~python

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
전체 Target 비율
survived
0    570
1    346
Name: count, dtype: int64
survived
0    0.622271
1    0.377729
Name: proportion, dtype: float64

Train Target 비율
survived
0    427
1    260
Name: count, dtype: int64
survived
0    0.621543
1    0.378457
Name: proportion, dtype: float64

Train / Validation shape
(687, 10) (229, 10)


Encoding 후 shape
Train     : (687, 12)
Validation: (229, 12)

Encoding 후 dtype


===== Original Baseline =====
Train AUC      : 0.963079
Validation AUC : 0.900065
Gap            : 0.063014


===== SMOTE 적용 전 =====
survived
0    427
1    260
Name: count, dtype: int64

===== SMOTE 적용 후 =====
survived
0    427
1    427
Name: count, dtype: int64

Train shape
(687, 12) → (854, 12)


===== SMOTE Model =====
Train AUC      : 0.967727
Validation AUC : 0.900146
Gap            : 0.067580


                Baseline     SMOTE  Difference
Train AUC       0.963079  0.967727    0.004648
Validation AUC  0.900065  0.900146    0.000081
Gap             0.063014  0.067580    0.004566

=======================================================
SMOTE 실험 결과 요약
=======================================================
Validation AUC : 0.900065 → 0.900146 (+0.000081)
Accuracy       : 0.860262 → 0.851528 (-0.008734)
Recall         : 0.790698 → 0.779070 (-0.011628)
F1             : 0.809524 → 0.797619 (-0.011905)

FP : 14 → 15 (+1)
FN : 18 → 19 (+1)

AUC 기준: SMOTE가 개선되었습니다.
FN 기준: 생존자를 놓치는 경우가 증가했습니다.
FP 기준: 사망자를 생존으로 잘못 예측하는 경우가 증가했습니다.

~~~

</details>


<details><summary>C:\dev\study\machine_learning\my_folder\3-3. 앙상블 예제 - 모듈화.ipynb — stored plots 0, stored errors 2</summary>

실행 오류 기록: ["ModuleNotFoundError: No module named 'common.evaluation_plots'", "AttributeError: type object 'BoostModelType' has no attribute 'show_plot_importance'"]

설정 근거:

~~~python

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
{'model_name': 'CatBoostClassifier',
 'hpo': {'verbose': 0,
  'cat_features': ['pclass',
   'sex',
   'embarked',
   'who',
   'adult_male',
   'deck',
   'alone']},
 'train_score': 0.8386643711489671,
 'test_score': 0.787599728399253,
 'score_type': 'auc'}
~~~

</details>


<details><summary>C:\dev\study\machine_learning\my_folder\titanic_eda.ipynb — stored plots 10, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text

~~~

</details>


<details><summary>C:\dev\study\machine_learning\my_folder\타이타닉 내가 순서 정리한 것.ipynb — stored plots 5, stored errors 1</summary>

실행 오류 기록: ['KeyError: "[\'alive\', \'adult_male\'] not found in axis"']

설정 근거:

~~~python

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
훈련용 AUC: 0.9227 / 테스트용 AUC: 0.8796


      Model  Train AUC  Test AUC
0  CatBoost   0.922721  0.879582
1  LightGBM   0.991290  0.838776
2   XGBoost   0.997094  0.835422
~~~

</details>


<details><summary>C:\dev\project\kaggle_skn35 v1\1. base model.ipynb — stored plots 2, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
{'model': XGBClassifier(base_score=None, booster=None, callbacks=None,
               colsample_bylevel=None, colsample_bynode=None,
               colsample_bytree=None, device=None, early_stopping_rounds=None,
               enable_categorical=True, eval_metric=None, feature_types=None,
               feature_weights=None, gamma=None, grow_policy=None,
               importance_type=None, interaction_constraints=None,
               learning_rate=None, max_bin=None, max_cat_threshold=None,
               max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
               max_leaves=None, min_child_weight=None, missing=nan,
               monotone_constraints=None, multi_strategy=None, n_estimators=None,
               n_jobs=None, num_parallel_tree=None, ...),
 'model_name': 'XGBClassifier',
 'hpo': {'tree_method': 'hist', 'enable_categorical': True},
 'train_score': 0.9536558158401784,
 'test_score': 0.9536558158401784,
 'score_type': 'get_auc_score'}
~~~

</details>


<details><summary>C:\dev\project\kaggle_skn35 v1\2. model.ipynb — stored plots 0, stored errors 1</summary>

실행 오류 기록: ["ValueError: feature_names mismatch: ['pclass', 'gender', 'age', 'sibsp', 'parch', 'fare'] ['pclass', 'gender', 'age', 'sibsp', 'parch', 'fare', 'FamilySize', 'IsAlone']\ntraining data did not have the following fields: IsAlone, FamilySize"]

설정 근거:

~~~python

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
train shape: (916, 12)
test shape: (393, 11)
submission shape: (393, 2)


features shape: (916, 6)
target shape: (916,)



Fold 1:
Train 데이터 크기: (732, 6)
Validation 데이터 크기: (184, 6)
Best Model: CatBoostClassifier
AUC: 0.862155
Accuracy: 0.880435
--------------------------------------------------



Fold 2:
Train 데이터 크기: (733, 6)
Validation 데이터 크기: (183, 6)
Best Model: CatBoostClassifier
AUC: 0.840389
Accuracy: 0.868852
--------------------------------------------------



Fold 3:
Train 데이터 크기: (733, 6)
Validation 데이터 크기: (183, 6)
Best Model: LGBMClassifier
AUC: 0.851640
Accuracy: 0.857923
--------------------------------------------------



Fold 4:
Train 데이터 크기: (733, 6)
Validation 데이터 크기: (183, 6)
Best Model: CatBoostClassifier
AUC: 0.860412
Accuracy: 0.868852
--------------------------------------------------



Fold 5:
Train 데이터 크기: (733, 6)
Validation 데이터 크기: (183, 6)
Best Model: CatBoostClassifier
AUC: 0.848970
Accuracy: 0.868852
--------------------------------------------------

[Base Model Cross Validation 결과]
Fold 1: AUC=0.862155, Accuracy=0.880435
Fold 2: AUC=0.840389, Accuracy=0.868852
Fold 3: AUC=0.851640, Accuracy=0.857923
Fold 4: AUC=0.860412, Accuracy=0.868852
Fold 5: AUC=0.848970, Accuracy=0.868852
CV Mean (AUC): 0.852713
CV Std (AUC): 0.007943
CV Mean (Accuracy): 0.868983
CV Std (Accuracy): 0.007120



Fold 1:
Train 데이터 크기: (912, 6)
Validation 데이터 크기: (184, 6)
Best Model: CatBoostClassifier
AUC: 0.853383
Accuracy: 0.869565
--------------------------------------------------

Fold 2 - SMOTE 적용 전
survived
0    456
1    277
Name: count, dtype: int64
Fold 2 - SMOTE 적용 후
survived
0    456
1    456
Name: count, dtype: int64



Fold 2:
Train 데이터 크기: (912, 6)
Validation 데이터 크기: (183, 6)
Best Model: CatBoostClassifier
AUC: 0.828566
Accuracy: 0.846995
--------------------------------------------------

Fold 3 - SMOTE 적용 전
survived
0    456
1    277
Name: count, dtype: int64
Fold 3 - SMOTE 적용 후
survived
0    456
1    456
Name: count, dtype: int64



Fold 3:
Train 데이터 크기: (912, 6)
Validation 데이터 크기: (183, 6)
Best Model: CatBoostClassifier
AUC: 0.837147
Accuracy: 0.846995
--------------------------------------------------

Fold 4 - SMOTE 적용 전
survived
0    456
1    277
Name: count, dtype: int64
Fold 4 - SMOTE 적용 후
survived
0    456
1    456
Name: count, dtype: int64



Fold 4:
Train 데이터 크기: (912, 6)
Validation 데이터 크기: (183, 6)
Best Model: CatBoostClassifier
AUC: 0.863272
Accuracy: 0.868852
--------------------------------------------------

Fold 5 - SMOTE 적용 전
survived
0    456
1    277
Name: count, dtype: int64
Fold 5 - SMOTE 적용 후
survived
0    456
1    456
Name: count, dtype: int64



Fold 5:
Train 데이터 크기: (912, 6)
Validation 데이터 크기: (183, 6)
Best Model: CatBoostClassifier
AUC: 0.840198
Accuracy: 0.857923
--------------------------------------------------

[Base Model + SMOTE Cross Validation 결과]
Fold 1: AUC=0.853383, Accuracy=0.869565
Fold 2: AUC=0.828566, Accuracy=0.846995
Fold 3: AUC=0.837147, Accuracy=0.846995
Fold 4: AUC=0.863272, Accuracy=0.868852
Fold 5: AUC=0.840198, Accuracy=0.857923
CV Mean (AUC): 0.844513
CV Std (AUC): 0.012311
CV Mean (Accuracy): 0.858066
CV Std (Accuracy): 0.009937


Base CV 평균 (AUC): 0.852713
SMOTE CV 평균 (AUC): 0.844513
차이 (AUC): -0.008200

Base CV 평균 (Accuracy): 0.868983
SMOTE CV 평균 (Accuracy): 0.858066
차이 (Accuracy): -0.010917


{'model': XGBClassifier(base_score=None, booster=None, callbacks=None,
               colsample_bylevel=None, colsample_bynode=None,
               colsample_bytree=None, device=None, early_stopping_rounds=None,
               enable_categorical=True, eval_metric=None, feature_types=None,
               feature_weights=None, gamma=None, grow_policy=None,
               importance_type=None, interaction_constraints=None,
               learning_rate=None, max_bin=None, max_cat_threshold=None,
               max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
               max_leaves=None, min_child_weight=None, missing=nan,
               monotone_constraints=None, multi_strategy=None, n_estimators=None,
               n_jobs=None, num_parallel_tree=None, ...),
 'model_name': 'XGBClassifier',
 'hpo': {'tree_method': 'hist', 'enable_categorical': True},
 'train_score': 0.9640350877192981,
 'test_score': 0.9640350877192981,
 'score_type': 'get_auc_score'}
~~~

</details>


<details><summary>C:\dev\project\kaggle_skn35 v1\3. model_feature_add.ipynb — stored plots 2, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
train shape: (916, 12)
test shape: (393, 11)
submission shape: (393, 2)



Fold 1:
Train 데이터 크기: (732, 11)
Validation 데이터 크기: (184, 11)
Best Model: CatBoostClassifier
AUC: 0.922744
Accuracy: 0.896739
--------------------------------------------------



Fold 2:
Train 데이터 크기: (733, 11)
Validation 데이터 크기: (183, 11)
Best Model: CatBoostClassifier
AUC: 0.891940
Accuracy: 0.874317
--------------------------------------------------



Fold 3:
Train 데이터 크기: (733, 11)
Validation 데이터 크기: (183, 11)
Best Model: LGBMClassifier
AUC: 0.888825
Accuracy: 0.846995
--------------------------------------------------



Fold 4:
Train 데이터 크기: (733, 11)
Validation 데이터 크기: (183, 11)
Best Model: CatBoostClassifier
AUC: 0.918065
Accuracy: 0.874317
--------------------------------------------------



Fold 5:
Train 데이터 크기: (733, 11)
Validation 데이터 크기: (183, 11)
Best Model: CatBoostClassifier
AUC: 0.877320
Accuracy: 0.863388
--------------------------------------------------

[3. model_feature_add Cross Validation 결과]
Fold 1: AUC=0.922744, Accuracy=0.896739
Fold 2: AUC=0.891940, Accuracy=0.874317
Fold 3: AUC=0.888825, Accuracy=0.846995
Fold 4: AUC=0.918065, Accuracy=0.874317
Fold 5: AUC=0.877320, Accuracy=0.863388
AUC Mean: 0.899779
AUC Std: 0.017593
Accuracy Mean: 0.871151
Accuracy Std: 0.016239



Fold 1:
Train 데이터 크기: (912, 11)
Validation 데이터 크기: (184, 11)
Best Model: XGBClassifier
AUC: 0.917481
Accuracy: 0.853261
--------------------------------------------------

Fold 2 - SMOTE 적용 전
survived
0    456
1    277
Name: count, dtype: int64
Fold 2 - SMOTE 적용 후
survived
0    456
1    456
Name: count, dtype: int64



Fold 2:
Train 데이터 크기: (912, 11)
Validation 데이터 크기: (183, 11)
Best Model: CatBoostClassifier
AUC: 0.888380
Accuracy: 0.846995
--------------------------------------------------

Fold 3 - SMOTE 적용 전
survived
0    456
1    277
Name: count, dtype: int64
Fold 3 - SMOTE 적용 후
survived
0    456
1    456
Name: count, dtype: int64



Fold 3:
Train 데이터 크기: (912, 11)
Validation 데이터 크기: (183, 11)
Best Model: CatBoostClassifier
AUC: 0.891241
Accuracy: 0.846995
--------------------------------------------------

Fold 4 - SMOTE 적용 전
survived
0    456
1    277
Name: count, dtype: int64
Fold 4 - SMOTE 적용 후
survived
0    456
1    456
Name: count, dtype: int64



Fold 4:
Train 데이터 크기: (912, 11)
Validation 데이터 크기: (183, 11)
Best Model: CatBoostClassifier
AUC: 0.916857
Accuracy: 0.868852
--------------------------------------------------

Fold 5 - SMOTE 적용 전
survived
0    456
1    277
Name: count, dtype: int64
Fold 5 - SMOTE 적용 후
survived
0    456
1    456
Name: count, dtype: int64



Fold 5:
Train 데이터 크기: (912, 11)
Validation 데이터 크기: (183, 11)
Best Model: CatBoostClassifier
AUC: 0.884884
Accuracy: 0.879781
--------------------------------------------------

[3. model_feature_add + SMOTE Cross Validation 결과]
Fold 1: AUC=0.917481, Accuracy=0.853261
Fold 2: AUC=0.888380, Accuracy=0.846995
Fold 3: AUC=0.891241, Accuracy=0.846995
Fold 4: AUC=0.916857, Accuracy=0.868852
Fold 5: AUC=0.884884, Accuracy=0.879781
AUC Mean: 0.899769
AUC Std: 0.014351
Accuracy Mean: 0.859177
Accuracy Std: 0.013038


[Feature 추가 전 참고]
현재 Kaggle 최고 점수: 0.91254

Feature CV AUC Mean: 0.899779
Feature CV AUC Std: 0.017593
Feature + SMOTE CV AUC Mean: 0.899769
Feature + SMOTE CV AUC Std: 0.014351

Feature CV Accuracy Mean: 0.871151
Feature + SMOTE CV Accuracy Mean: 0.859177


{'model': XGBClassifier(base_score=None, booster=None, callbacks=None,
               colsample_bylevel=None, colsample_bynode=None,
               colsample_bytree=None, device=None, early_stopping_rounds=None,
               enable_categorical=True, eval_metric=None, feature_types=None,
               feature_weights=None, gamma=None, grow_policy=None,
               importance_type=None, interaction_constraints=None,
               learning_rate=None, max_bin=None, max_cat_threshold=None,
               max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
               max_leaves=None, min_child_weight=None, missing=nan,
               monotone_constraints=None, multi_strategy=None, n_estimators=None,
               n_jobs=None, num_parallel_tree=None, ...),
 'model_name': 'XGBClassifier',
 'hpo': {'tree_method': 'hist', 'enable_categorical': True},
 'train_score': 0.9960541705140042,
 'test_score': 0.9960541705140042,
 'score_type': 'get_auc_score'}

Train 데이터 개수: 916
OOF 0/1 예측 개수: 916
OOF 확률값 개수: 916
TN: 516
FP: 54
FN: 75
TP: 271
[[0.90526316 0.09473684]
 [0.21676301 0.78323699]]

~~~

</details>


<details><summary>C:\dev\project\kaggle_skn35 v1\4. model_feature_add2.ipynb — stored plots 2, stored errors 0</summary>

실행 오류 기록: []

설정 근거:

~~~python

~~~

저장된 지표/Confusion/Importance 근거 (같은 셀에서 출력된 원문, 표의 기준 모델을 구분해야 함):

~~~text
train shape: (916, 12)
test shape: (393, 11)
submission shape: (393, 2)


Train 크기: (912, 15)
Validation 크기: (184, 15)
CatBoost AUC: 0.932331
CatBoost Accuracy: 0.885870
--------------------------------------------------

Fold 2
SMOTE 적용 전 클래스 개수
survived
0    456
1    277
Name: count, dtype: int64
SMOTE 적용 후 클래스 개수
survived
0    456
1    456
Name: count, dtype: int64


Train 크기: (912, 15)
Validation 크기: (183, 15)
CatBoost AUC: 0.899186
CatBoost Accuracy: 0.863388
--------------------------------------------------

Fold 3
SMOTE 적용 전 클래스 개수
survived
0    456
1    277
Name: count, dtype: int64
SMOTE 적용 후 클래스 개수
survived
0    456
1    456
Name: count, dtype: int64


Train 크기: (912, 15)
Validation 크기: (183, 15)
CatBoost AUC: 0.903318
CatBoost Accuracy: 0.841530
--------------------------------------------------

Fold 4
SMOTE 적용 전 클래스 개수
survived
0    456
1    277
Name: count, dtype: int64
SMOTE 적용 후 클래스 개수
survived
0    456
1    456
Name: count, dtype: int64


Train 크기: (912, 15)
Validation 크기: (183, 15)
CatBoost AUC: 0.915840
CatBoost Accuracy: 0.857923
--------------------------------------------------

Fold 5
SMOTE 적용 전 클래스 개수
survived
0    456
1    277
Name: count, dtype: int64
SMOTE 적용 후 클래스 개수
survived
0    456
1    456
Name: count, dtype: int64


Train 크기: (912, 15)
Validation 크기: (183, 15)
CatBoost AUC: 0.911709
CatBoost Accuracy: 0.874317
--------------------------------------------------

[Feature Add2 + SMOTE + CatBoost CV 결과]
Fold 1: AUC=0.932331, Accuracy=0.885870
Fold 2: AUC=0.899186, Accuracy=0.863388
Fold 3: AUC=0.903318, Accuracy=0.841530
Fold 4: AUC=0.915840, Accuracy=0.857923
Fold 5: AUC=0.911709, Accuracy=0.874317
AUC Mean: 0.912477
AUC Std: 0.011546
Accuracy Mean: 0.864606
Accuracy Std: 0.015001
OOF Prediction 길이 확인 완료: 916


TN: 515
FP: 55
FN: 69
TP: 277


========== Experiment 4 Summary ==========

Model: CatBoost
SMOTE: Yes

Features Added:
- HasCabin
- Deck
- TicketGroupSize
- FarePerPerson

Age Imputation:
- Title × Pclass Median

CV AUC Mean: 0.912477
CV AUC Std: 0.011546
CV Accuracy Mean: 0.864606
CV Accuracy Std: 0.015001

OOF Confusion Matrix:
TN: 515
FP: 55
FN: 69
TP: 277

Submission Path:
./submission/4. model_feature_add2_result.csv

~~~

</details>

### CSV 전체 점검

| file | rows | columns | sha256 | ID_unique | finite | range_ok | hard_labels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C:\dev\study\machine_learning\titanic_26_result.csv | 393 | passengerid,survived | 2d5aafda6324747b14faa378ab6521486a674f84908cfe3ea9b6bd762044207e | True | True | True | False |
| C:\dev\study\machine_learning\titanic_27_result.csv | 393 | passengerid,survived | 1b8eab10d73fffc4a82626622f9e93e9551eacb0a23b8687e444570ed2946a71 | True | True | True | False |
| C:\dev\study\machine_learning\titanic_28_result.csv | 393 | passengerid,survived | cdee59cf0e73921c288c21c310df88e792206a2f941a3d05bf142334580366b0 | True | True | True | False |
| C:\dev\study\machine_learning\titanic_29_result.csv | 393 | passengerid,survived | a28f8e1ec3b6abb29e94dc1aa138aefc25a4d4d6bd62848f8ac32d40fbedadcf | True | True | True | False |
| C:\dev\study\machine_learning\titanic_30_result.csv | 393 | passengerid,survived | 88901d44560b2ebf739c9578ec16c06e333695444bcd6a30a3286bfcb05a3562 | True | True | True | False |
| C:\dev\study\machine_learning\titanic_31_result.csv | 393 | passengerid,survived | cdee59cf0e73921c288c21c310df88e792206a2f941a3d05bf142334580366b0 | True | True | True | False |
| C:\dev\study\machine_learning\csv\submission.csv | 393 | passengerid,survived | 723ab89193140f3c5a901ca96a7fcbb9869bba674e385a1c7ff77cdbd029e697 | True | True | True | False |
| C:\dev\study\machine_learning\csv\test.csv | 393 | passengerid,pclass,name,gender,age,sibsp,parch,ticket,fare,cabin,embarked | 9df3361de7838476cc9a9d216e09f58edbf6b1267dfd188d8a504b943671a070 | 미기록 | 미기록 | 미기록 | 미기록 |
| C:\dev\study\machine_learning\csv\train.csv | 916 | passengerid,survived,pclass,name,gender,age,sibsp,parch,ticket,fare,cabin,embarked | f752a254eaf503f4cfd0dacd6854b4e9638d78d65ee9e9944b8db8895d991617 | 미기록 | 미기록 | 미기록 | 미기록 |
| C:\dev\study\machine_learning\results\titanic_16_randomized_search_results.csv | 30 | mean_fit_time,std_fit_time,mean_score_time,std_score_time,param_model__border_count,param_model__depth,param_model__iterations,param_model__l2_leaf_reg,param_model__learning_rate,param_model__random_strength,param_model__subsample,params,split0_test_score,split1_test_score,split2_test_score,split3_test_score,split4_test_score,mean_test_score,std_test_score,rank_test_score,split0_train_score,split1_train_score,split2_train_score,split3_train_score,split4_train_score,mean_train_score,std_train_score,rank,train_auc,cv_auc,cv_std,gap,iterations,depth,learning_rate,l2_leaf_reg,random_strength,subsample,border_count | 728529b71da1d4783bf627e0a08e9041e1af4cc9c924508fd17155e9f46983bc | 미기록 | 미기록 | 미기록 | 미기록 |
| C:\dev\study\machine_learning\submission\submission_result_1.csv | 393 | passengerid,survived | 63accf77b10d71f2966e9e9af63a506c88c7e9c85f0b738e81e0a4ad5e3d7794 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_19_result.csv | 393 | passengerid,survived | a540adaf18bf7857c34394c9f02c75a1dcbcf0b0f7c418e4a2a342a39f99c94a | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_20_result.csv | 393 | passengerid,survived | 63accf77b10d71f2966e9e9af63a506c88c7e9c85f0b738e81e0a4ad5e3d7794 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_21_result.csv | 393 | passengerid,survived | 2406c9d699a34083ad2df8b56ba525ef7ec5e7ac4e7299dea49dbb1a7ffe5399 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_22_result.csv | 393 | passengerid,survived | bc92773242a343508c07659d3b3ca080ab6e230a8b3e2d5a2c78def2cf1badc0 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_23_result.csv | 393 | passengerid,survived | 407403ab07b949f3a38df735a828f60f2fc9c5c89e0bf3338c602e82de7e8283 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_24_result.csv | 393 | passengerid,survived | a59ab2759e0669526849809dff776c0864a00edfb9041c55e78db29e39f13467 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_25_result.csv | 393 | passengerid,survived | d05088f5c8c5d7a2752a184f713aaac675810916132a914dfa895864741a2ed4 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_10.csv | 393 | passengerid,survived | ed9dfd6c863106db1c40ba54f50dace9eeae9601d15e9fe2ba8601df73049f55 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_11.csv | 393 | passengerid,survived | f57b05ba7bf63e7ae295747a82a631485a592d19428bd9f53374ada72be3bac1 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_12.csv | 393 | passengerid,survived | e65829d63f05f320619d1f21eb0b2f0af00ddaa79e7b69ba9ee5f09b18d61f5c | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_13.csv | 393 | passengerid,survived | 58318c32a6311b4bf2fa93001ecf5d7655425485104b191dcaea92155f83b65b | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_14.csv | 393 | passengerid,survived | 4a8ba5b7517883603f4f24a39ef82d1c4515644ce44212be2417a38e442cad2c | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_15.csv | 393 | passengerid,survived | fbbfa51e5673d38611077685770cd3393d5334ccf9ec1a799b84234d1b00a3c7 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_16_delete_age.csv | 393 | passengerid,survived | b4904a6a22e09c44a22b2466468a38fc60facce1be0b5e911245e1a8af56c1b0 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_16_randomized_search.csv | 393 | passengerid,survived | b70566d8417162069f69de0a11bf833004ab1161841a9cc5ba6de9bc7675cd8f | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_18.csv | 393 | passengerid,survived | 3d3ee469b76c45e881f3711af99d240f00a5ab97a91f4694b8e60d4ab4c9ff0a | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_2.csv | 393 | passengerid,survived | 63accf77b10d71f2966e9e9af63a506c88c7e9c85f0b738e81e0a4ad5e3d7794 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_3.csv | 393 | passengerid,survived | 7016cab089648b80d7f43f09381a3b2a189b57ed1f7e11b1020d232c11934c44 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_4.csv | 393 | passengerid,survived | 314b0416a4175fcd79549c44db6cebaefddf8351ce9468a10920bea7d7da8da3 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_5.csv | 393 | passengerid,survived | 43a50e7ac00e19b2b1a9ceaf2ccb6dfdf98b16902d0014cb1c772e60d89ffb3b | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_6.csv | 393 | passengerid,survived | da61a62ac3a22c80da6f68da10d3942e30e04695216da348f25433208d90aa45 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_7.csv | 393 | passengerid,survived | 58127cbff6289196d2b7e90563ae2ff258e5527885d89248940ac04020c205dc | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_8.csv | 393 | passengerid,survived | 6a01d2ef80b781ee391fe27b924d332d8aba89493efa060d8c64b0d80aebea46 | True | True | True | False |
| C:\dev\study\machine_learning\submission\titanic_result_9.csv | 393 | passengerid,survived | db3b1d75d3820d9016eea1d89a03e0101d33bc2524fbefdc4d1072f470829eb0 | True | True | True | False |
| C:\dev\study\machine_learning\titanic_audit\oof_audit.csv | 916 | passengerid,survived,Base,11,15 | 304e31752c308fe1df6ecaf8d2add042f762e425f9bfadf1bb588865c52d20ed | 미기록 | 미기록 | 미기록 | 미기록 |
| C:\dev\study\machine_learning\titanic_audit\test_prediction_changes.csv | 393 | passengerid,age,AgeBand,pclass,gender,p11,p15,delta | 2e2c081264ca39034735bf00785c8557e6fe7d208d4b07bdc85eb493aa8a3601 | 미기록 | 미기록 | 미기록 | 미기록 |
| C:\dev\study\machine_learning\titanic_audit\depth5_smoke\fold_metrics.csv | 2 | split_seed,model_seed,fold,model,train_auc,validation_auc,gap | 0aa97f8c5e4e7f38c02ba4e8c13eaa6294ed10406bc52a5218d785de07d0792f | 미기록 | 미기록 | 미기록 | 미기록 |
| C:\dev\project\kaggle_skn35 v1\data\submission.csv | 393 | passengerid,survived | 723ab89193140f3c5a901ca96a7fcbb9869bba674e385a1c7ff77cdbd029e697 | True | True | True | False |
| C:\dev\project\kaggle_skn35 v1\data\test.csv | 393 | passengerid,pclass,name,gender,age,sibsp,parch,ticket,fare,cabin,embarked | 9df3361de7838476cc9a9d216e09f58edbf6b1267dfd188d8a504b943671a070 | 미기록 | 미기록 | 미기록 | 미기록 |
| C:\dev\project\kaggle_skn35 v1\data\train.csv | 916 | passengerid,survived,pclass,name,gender,age,sibsp,parch,ticket,fare,cabin,embarked | f752a254eaf503f4cfd0dacd6854b4e9638d78d65ee9e9944b8db8895d991617 | 미기록 | 미기록 | 미기록 | 미기록 |
| C:\dev\project\kaggle_skn35 v1\submission\2. model_result.csv | 393 | passengerid,survived | cfb459e2d17c6e4532447756ae5826b90a4950c4616cbb84e0646847505a1acb | True | True | True | True |
| C:\dev\project\kaggle_skn35 v1\submission\3. model_feature_add_result.csv | 393 | passengerid,survived | d6c990c8a2ec4ed96417cc3162f58cce6b53d66b3339c38b12093321192ba075 | True | True | True | False |
| C:\dev\project\kaggle_skn35 v1\submission\4. model_feature_add2_result.csv | 393 | passengerid,survived | 44df0c345325055e78cedf630161de72437af79a6e933525e58054b19bea65f6 | True | True | True | False |
| C:\dev\project\kaggle_skn35 v1\submission\base_model_0909_01_(95.36).csv | 393 | passengerid,survived | 651b807c49042e2301811e685046713173b89e08f6a01fcb61d777ae617b3386 | True | True | True | True |

### 관련 로그/보조 결과

catboost_info는 공용 경로로 실험 ID가 없어 1~31의 특정 버전에 점수를 귀속시키지 않았다. titanic_audit/depth5_smoke/fold_metrics.csv는 한 fold smoke만 있어 전체 CV로 사용하지 않았다. 과거 SHAP 중 test 기반 그림은 최종 성능 증거가 아니며 이번에는 OOF SHAP을 새로 계산했다. 별도 kaggle_skn35 계열은 seed0/SMOTE/전처리 누수/모델 선택 차이로 주계열 점수와 직접 순위 비교하지 않는다. 기존 REPORT.md의 Private/Public 기록만 출처를 밝혀 사용한다.

### 파일 manifest

| file | sha256 | bytes |
| --- | --- | --- |
| C:\dev\study\machine_learning\build_titanic_18.py | fa035d83f26cce9da11e12d0d16954608fee619711fd4f96d24dad274ad9402b | 19778 |
| C:\dev\study\machine_learning\feature_experiment_report_10_15.md | d9667d94f7e3672c80292de7e3ffcce9f0fffd06031d90d10475f6170e75901f | 7222 |
| C:\dev\study\machine_learning\feature_experiment_report_3_9.md | a58a60d323ebf43498bf753f14c8c71a9ed87c976ed36e44bb1b50f4b166a9f0 | 5299 |
| C:\dev\study\machine_learning\feature_experiment_results_10_15.json | 69d5cab01bbaf7146cdc7036e84d8fd3c88a2ad9a12f7297dec8ed437902a35b | 15382 |
| C:\dev\study\machine_learning\feature_experiment_results_3_9.json | 985b18abc61fe0f936bf030ac8940a4f28cc7cd6cfe7ec1aa4f97915d1cd3834 | 17208 |
| C:\dev\study\machine_learning\pyproject.toml | 431bd88632fd49811cb1c287369d02beef38eddf1027e9d4530265b4396fee23 | 389 |
| C:\dev\study\machine_learning\README.md | f91688b392475c7cd98d83462800be9ac631a7bb5fa6f873df5693acba30811e | 16 |
| C:\dev\study\machine_learning\titanic_1 copy.ipynb | 5e231bd220e35be86edd9a9b018100dd6dee31a861034bd3f299e932bb222954 | 286605 |
| C:\dev\study\machine_learning\titanic_1.ipynb | e8b73f5a92982d674834e84532cf4c6c2c2be03f76f906197af8f5331907e918 | 56841 |
| C:\dev\study\machine_learning\titanic_10.ipynb | cb93e870f7ab03f0c3cadd3c4d89b678fce2aa2e48321068658c9d4ad5bb000f | 46708 |
| C:\dev\study\machine_learning\titanic_11.ipynb | 31da9adc2fb9192e768d28c3f7fe3aee7aee5e8b85f42a88f006005c800184e5 | 321396 |
| C:\dev\study\machine_learning\titanic_12.ipynb | 68b0584d4a735eead198069e3aa59575a8d11ccd86734fdf518adb4f058b9a3d | 44120 |
| C:\dev\study\machine_learning\titanic_13.ipynb | cec2ed3a1c22a38c0f4772b3accb0c79e90e3814f1db6937cc943c2931598cf4 | 47697 |
| C:\dev\study\machine_learning\titanic_14.ipynb | 11313e88f6c2170360b35f95f3c2630e9b66c4ccdfce97101a3ab0334aa2d4c0 | 54043 |
| C:\dev\study\machine_learning\titanic_15.ipynb | 746a7f7872cf723fd8c99458e50c88d2fe5fff52d0a48fdf1be42476572f24a8 | 243282 |
| C:\dev\study\machine_learning\titanic_16_delete_age.ipynb | 401434032c3a35fc9b263c7755eb4658a20f0b203cc6809cc401a6c7dd929844 | 204149 |
| C:\dev\study\machine_learning\titanic_16_isnoble.ipynb | 4b81bd2f0ef260ff218eb397c11b1874e2292046ede3e57f2d6b700993e579e7 | 57817 |
| C:\dev\study\machine_learning\titanic_16_randomized_search.ipynb | 93c69ff0a7e4cb13063afaf70f9fbaf7f75b107e79c61c612cbcec4f96749ade | 32228 |
| C:\dev\study\machine_learning\titanic_17_profession_group.ipynb | 71fa436ed0e626c1078e82a0d36ee1d5e70dc024e62788a3cadf8229fa1b8929 | 73479 |
| C:\dev\study\machine_learning\titanic_18.ipynb | 1806f9980afeacb79986457c3f29624293bf975790a0bafa66dd7391a39a6ba4 | 118570 |
| C:\dev\study\machine_learning\titanic_19.ipynb | f0faf2665b637edd52c5a3ebe81f468e2bb1fad2f00be26081b84f6271077cda | 117110 |
| C:\dev\study\machine_learning\titanic_2.ipynb | 1ddc68b5509506c673ba520750936e825db9b73541ea126d323be6e2274976f9 | 82411 |
| C:\dev\study\machine_learning\titanic_20.ipynb | 00172c62e842f0dcca5e0633413644ab373ee86803a5242d59d8b9dae9396aa2 | 103182 |
| C:\dev\study\machine_learning\titanic_21.ipynb | 8751307ff12ba2ff86841baf31df746c9ffe5d0c11d47789e1923a51dc1cb86c | 209071 |
| C:\dev\study\machine_learning\titanic_22.ipynb | 8823c93ab307b458e1e1bc1a5c422ba87eef09107a4cd8e2423d835bb2849839 | 108906 |
| C:\dev\study\machine_learning\titanic_23.ipynb | d854fea9aeb972a8d00a5b5820e7f3c6e6f28c93470c2cfd8797d91b9589fa27 | 108310 |
| C:\dev\study\machine_learning\titanic_24.ipynb | 62281d2902ea28d7bbe49a1609ef18e5bcde250492d8d66c1600c098db3d8876 | 212570 |
| C:\dev\study\machine_learning\titanic_25.ipynb | 6b83427f956e0c249aea5d6959c394a8b688a7dcc7ced83163686dc1841e33c1 | 300160 |
| C:\dev\study\machine_learning\titanic_26.ipynb | 8054303c200a03f1cd3bbe1d2faa8bc0f2a101ecd65ce750e9f6c757e3deb8b9 | 41012 |
| C:\dev\study\machine_learning\titanic_26_result.csv | 2d5aafda6324747b14faa378ab6521486a674f84908cfe3ea9b6bd762044207e | 9974 |
| C:\dev\study\machine_learning\titanic_27.ipynb | d94d16e1fa661d85dceb1f256f4be8f266ac2c70f803204776b3fa0efb672ebc | 53238 |
| C:\dev\study\machine_learning\titanic_27_result.csv | 1b8eab10d73fffc4a82626622f9e93e9551eacb0a23b8687e444570ed2946a71 | 9998 |
| C:\dev\study\machine_learning\titanic_28.ipynb | f7e49718d9174a1050740e96dd4bef05fa9f4989c249ec52f14eaa1e9b416a66 | 41461 |
| C:\dev\study\machine_learning\titanic_28_result.csv | cdee59cf0e73921c288c21c310df88e792206a2f941a3d05bf142334580366b0 | 9878 |
| C:\dev\study\machine_learning\titanic_29.ipynb | 928bb285c3c7943c845e2abc517ea907f2ffc47e6d12aafaad7d7ba486df69e9 | 28892 |
| C:\dev\study\machine_learning\titanic_29_result.csv | a28f8e1ec3b6abb29e94dc1aa138aefc25a4d4d6bd62848f8ac32d40fbedadcf | 9839 |
| C:\dev\study\machine_learning\titanic_3.ipynb | cb72a20a29343d62049db22921b3f99d7daf1172c0b9e2ec881dabd1cd7241ee | 47999 |
| C:\dev\study\machine_learning\titanic_30.ipynb | 28dfd41d8303a8947a559a7a9a10f9160f802c3b9aa20c29c85f60825bcc60b6 | 28947 |
| C:\dev\study\machine_learning\titanic_30_result.csv | 88901d44560b2ebf739c9578ec16c06e333695444bcd6a30a3286bfcb05a3562 | 9926 |
| C:\dev\study\machine_learning\titanic_31.ipynb | 86458608b37750cc1162140477d832d08b6dd7d2f847609bf86e5b8e022d0cb5 | 61174 |
| C:\dev\study\machine_learning\titanic_31_result.csv | cdee59cf0e73921c288c21c310df88e792206a2f941a3d05bf142334580366b0 | 9878 |
| C:\dev\study\machine_learning\titanic_4.ipynb | 28f66a969c4bb19390bd6cdda65c94c0bfdc816e2b8befcf848cc9676a95aff4 | 44877 |
| C:\dev\study\machine_learning\titanic_5.ipynb | eb91000ea5da19925214ca5dcaacf84d095a62d06dee961a5a63cfa1bc1ca73c | 42428 |
| C:\dev\study\machine_learning\titanic_6.ipynb | 22cc0b543d9d35c46763c5646a0088981db29c19d314c7efc157f75b8c5f5cc3 | 37963 |
| C:\dev\study\machine_learning\titanic_7.ipynb | 0b166fe7a68d3d0aafcb7809a0ed4001a14dbe9d8cebf2db1c877b1040332d26 | 39877 |
| C:\dev\study\machine_learning\titanic_8.ipynb | 26a43e4334c21fe8aa445b8e69e6f8310ba9db8545f5c022c62f08104660238b | 42248 |
| C:\dev\study\machine_learning\titanic_9.ipynb | 4f71d0b6a985b5f61114c017015c45e5bfdbe55e6937f3ede8c8f018dcdfb254 | 48520 |
| C:\dev\study\machine_learning\titanic_base_smote_experiment.ipynb | 3c4ee1f0ed34fa38ec8a0b93dec8887092326a765389fdd56f1451d9cf92ebca | 92604 |
| C:\dev\study\machine_learning\catboost_info\catboost_training.json | 0f5dd96ba794b4eb9c59c6e5ed7766bb36004b0eeca37cf603eb89c21ce579e1 | 77516 |
| C:\dev\study\machine_learning\catboost_info\learn_error.tsv | a607e6afcbc9edaa1ec88ff7893b7921bcbf3f57bc1eaa1310062594db8850d2 | 13421 |
| C:\dev\study\machine_learning\catboost_info\time_left.tsv | d3ba1bed11f35d07ce8416c6ef8cee74d72785c388247fcbc70d0b73baaa4441 | 11788 |
| C:\dev\study\machine_learning\common\evaluations.py | f817f4f8fef34d7ad4aef5a96d89726630a2a6603c92845d40e9ceecd0fcc015 | 277 |
| C:\dev\study\machine_learning\common\feature_experiments.py | 2a92e1d4c733ccd9ce962c111c4f12d56c9babf3abc72bf4865624551c29b787 | 5231 |
| C:\dev\study\machine_learning\common\interaction_experiments.py | f25e4901e9e33afc15aacf794265068ccea3de691ff4082e01a17e4d7f25bfcf | 4494 |
| C:\dev\study\machine_learning\common\modeling.py | 380a5a63849e4021140aee280a726fc092ad3a181f344eb7245b2162974dfd77 | 7266 |
| C:\dev\study\machine_learning\common\preprocessing.py | e89ef18b0665faad939a6eae9277b4bc1874acd005273e1b755facaf4e70bc09 | 3357 |
| C:\dev\study\machine_learning\common\title_feature_experiments.py | a7022165b2791bff0c0ccb3a2743f3919d5e2bba73355c5cd100d4e0bb1485c1 | 4385 |
| C:\dev\study\machine_learning\common\utils.py | 004ee1974091c6fe90b251d80ee19d166682c3f1cbeffec76e111fb0529fe418 | 763 |
| C:\dev\study\machine_learning\common\__pycache__\evaluation_plots.py | af84b436fc95457ec5de30d31e7f3e7119282f4b36a94401b53f336007a824f9 | 388 |
| C:\dev\study\machine_learning\csv\submission.csv | 723ab89193140f3c5a901ca96a7fcbb9869bba674e385a1c7ff77cdbd029e697 | 3868 |
| C:\dev\study\machine_learning\csv\test.csv | 9df3361de7838476cc9a9d216e09f58edbf6b1267dfd188d8a504b943671a070 | 27384 |
| C:\dev\study\machine_learning\csv\train.csv | f752a254eaf503f4cfd0dacd6854b4e9638d78d65ee9e9944b8db8895d991617 | 64952 |
| C:\dev\study\machine_learning\my_folder\3-3. 앙상블 예제 - 모듈화.ipynb | 93523ba39b7ff701d9c9b291d971df4b6407da56d143738b4066f91c3359ccd0 | 11969 |
| C:\dev\study\machine_learning\my_folder\titanic_baseline_guide.md | 6b7d6e2a95942458e1b931a76305e10efb335a8bf31f68040624fb2588c7f39f | 11501 |
| C:\dev\study\machine_learning\my_folder\titanic_eda.ipynb | c487949aec24c72e4aebfd0812893feb7fd27845e2be45f573723e9ae8923d0d | 569399 |
| C:\dev\study\machine_learning\my_folder\타이타닉 내가 순서 정리한 것.ipynb | d47991f0dba9c94607754785582ff8fee9ba4620c6fb082351d9100d2b89a04f | 182920 |
| C:\dev\study\machine_learning\results\titanic_16_randomized_search_results.csv | 728529b71da1d4783bf627e0a08e9041e1af4cc9c924508fd17155e9f46983bc | 27977 |
| C:\dev\study\machine_learning\results\titanic_16_summary.json | 1d96c1b2427d70d07f830d68885a1dc6c0073fdc729cdb192faee2dfa0f1f2b9 | 1610 |
| C:\dev\study\machine_learning\results\titanic_25_summary.json | 7c6fc617b30428683b6c6e571c2c5316ad2e3c584c5b62a9322724f4cd2db487 | 2742 |
| C:\dev\study\machine_learning\submission\submission_result_1.csv | 63accf77b10d71f2966e9e9af63a506c88c7e9c85f0b738e81e0a4ad5e3d7794 | 9958 |
| C:\dev\study\machine_learning\submission\titanic_19_result.csv | a540adaf18bf7857c34394c9f02c75a1dcbcf0b0f7c418e4a2a342a39f99c94a | 9979 |
| C:\dev\study\machine_learning\submission\titanic_20_result.csv | 63accf77b10d71f2966e9e9af63a506c88c7e9c85f0b738e81e0a4ad5e3d7794 | 9958 |
| C:\dev\study\machine_learning\submission\titanic_21_result.csv | 2406c9d699a34083ad2df8b56ba525ef7ec5e7ac4e7299dea49dbb1a7ffe5399 | 9964 |
| C:\dev\study\machine_learning\submission\titanic_22_result.csv | bc92773242a343508c07659d3b3ca080ab6e230a8b3e2d5a2c78def2cf1badc0 | 9956 |
| C:\dev\study\machine_learning\submission\titanic_23_result.csv | 407403ab07b949f3a38df735a828f60f2fc9c5c89e0bf3338c602e82de7e8283 | 9958 |
| C:\dev\study\machine_learning\submission\titanic_24_result.csv | a59ab2759e0669526849809dff776c0864a00edfb9041c55e78db29e39f13467 | 9971 |
| C:\dev\study\machine_learning\submission\titanic_25_result.csv | d05088f5c8c5d7a2752a184f713aaac675810916132a914dfa895864741a2ed4 | 9985 |
| C:\dev\study\machine_learning\submission\titanic_result_10.csv | ed9dfd6c863106db1c40ba54f50dace9eeae9601d15e9fe2ba8601df73049f55 | 9960 |
| C:\dev\study\machine_learning\submission\titanic_result_11.csv | f57b05ba7bf63e7ae295747a82a631485a592d19428bd9f53374ada72be3bac1 | 9972 |
| C:\dev\study\machine_learning\submission\titanic_result_12.csv | e65829d63f05f320619d1f21eb0b2f0af00ddaa79e7b69ba9ee5f09b18d61f5c | 9992 |
| C:\dev\study\machine_learning\submission\titanic_result_13.csv | 58318c32a6311b4bf2fa93001ecf5d7655425485104b191dcaea92155f83b65b | 9970 |
| C:\dev\study\machine_learning\submission\titanic_result_14.csv | 4a8ba5b7517883603f4f24a39ef82d1c4515644ce44212be2417a38e442cad2c | 9962 |
| C:\dev\study\machine_learning\submission\titanic_result_15.csv | fbbfa51e5673d38611077685770cd3393d5334ccf9ec1a799b84234d1b00a3c7 | 9974 |
| C:\dev\study\machine_learning\submission\titanic_result_16_delete_age.csv | b4904a6a22e09c44a22b2466468a38fc60facce1be0b5e911245e1a8af56c1b0 | 9996 |
| C:\dev\study\machine_learning\submission\titanic_result_16_randomized_search.csv | b70566d8417162069f69de0a11bf833004ab1161841a9cc5ba6de9bc7675cd8f | 9948 |
| C:\dev\study\machine_learning\submission\titanic_result_18.csv | 3d3ee469b76c45e881f3711af99d240f00a5ab97a91f4694b8e60d4ab4c9ff0a | 9974 |
| C:\dev\study\machine_learning\submission\titanic_result_2.csv | 63accf77b10d71f2966e9e9af63a506c88c7e9c85f0b738e81e0a4ad5e3d7794 | 9958 |
| C:\dev\study\machine_learning\submission\titanic_result_3.csv | 7016cab089648b80d7f43f09381a3b2a189b57ed1f7e11b1020d232c11934c44 | 9972 |
| C:\dev\study\machine_learning\submission\titanic_result_4.csv | 314b0416a4175fcd79549c44db6cebaefddf8351ce9468a10920bea7d7da8da3 | 9970 |
| C:\dev\study\machine_learning\submission\titanic_result_5.csv | 43a50e7ac00e19b2b1a9ceaf2ccb6dfdf98b16902d0014cb1c772e60d89ffb3b | 9960 |
| C:\dev\study\machine_learning\submission\titanic_result_6.csv | da61a62ac3a22c80da6f68da10d3942e30e04695216da348f25433208d90aa45 | 9968 |
| C:\dev\study\machine_learning\submission\titanic_result_7.csv | 58127cbff6289196d2b7e90563ae2ff258e5527885d89248940ac04020c205dc | 9992 |
| C:\dev\study\machine_learning\submission\titanic_result_8.csv | 6a01d2ef80b781ee391fe27b924d332d8aba89493efa060d8c64b0d80aebea46 | 9979 |
| C:\dev\study\machine_learning\submission\titanic_result_9.csv | db3b1d75d3820d9016eea1d89a03e0101d33bc2524fbefdc4d1072f470829eb0 | 9953 |
| C:\dev\study\machine_learning\titanic_audit\0_3-3. 앙상블 예제 - 모듈화.txt | 71ed009b24cf4e2e5cfef604a0139262b5d1dfa4ebeaa74ed99fc6cbec76d888 | 4649 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_1 copy.txt | 084efc09b02bfe9fc7b95fa264e2c3d352b8603d3df75837d86c10b28948ff62 | 44645 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_1.txt | 75d045941ae5a2e99e4516750f10e00728ad94a87e1fd75dc0d0ab87f58636fc | 24443 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_10.txt | 1e72bfa3b37fa36d810e00a8a10178543be8d4f4f0e84c24bf0594032cdd6448 | 23188 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_11.txt | 43339341ce901e193737ae6f76dbe25d55721df152e4dbbc52634e26ada358c2 | 30952 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_12.txt | 4aceb4aeef62cc8ed004c2207539a9ac43b41477c1a4cc11bd73dc8d20bfb26e | 22685 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_13.txt | 77f0c4edf57db2eec27263afb2eb4a0d71a4814a7345fee48455c4118174905b | 23840 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_14.txt | 78b43659e66900d5ae071eec36bac95d4ea40fc8d8ee1a1b4fa6afc259e555a7 | 26404 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_15.txt | d0908de1bfc3e969f00d4c47e3a6949fe13f7d0b260df51bc8a624bc6a4223a7 | 29417 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_16_isnoble.txt | c09d1b3a5bd76fb0cacbddc7e7d3802e1eb9f1fde657b49bc1d205483356c756 | 17728 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_17_profession_group.txt | bd23b8d61379b98b69c4b40e5179da827caeb97f068ed852096d99b6e45c3931 | 18286 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_2.txt | 4fbd082c92a7b512ba9d89ecfddef1e61e869b8c4401eb29765f84027fe0c396 | 35631 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_3.txt | 47d5d7944738c23a0f3879efdf8a1fb0b4f8711ca5cfea2daf1a1257cfadac02 | 23151 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_4.txt | 7f034fcc219480e32112a5e1ab681bc008561be3a4513dcd8cf115ea9ca5104d | 22374 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_5.txt | f0ce73d614edd00082f2ce0ee15d5abec9d959dce41239ed7cf8c74cc060ae55 | 21929 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_6.txt | 07af8c6a223ff08d0340450c920fc7aaef25e14a3855ce66d34233c87540f93a | 21443 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_7.txt | 61760811a19342d79f62477e345c3abd13e406cd26ddeae6f208b32b7e8becbb | 22154 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_8.txt | 21c00d9a35cf1e31c11caab956b86d4949ca6eba23619629e1aa0da511faa0d7 | 22401 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_9.txt | 243e9ada3f28510365e7c0add1b2ff7b82d7e32baa76d68c295e0fafa1ffe17c | 24725 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_base_smote_experiment.txt | 4a4c352cf44dc2b78444ba97e77e17d22587acf7595663fca6a8fac5d3be2adc | 21350 |
| C:\dev\study\machine_learning\titanic_audit\0_titanic_eda.txt | 6915d2568c939a781055a61c799a469c3c07dc17b5885cb45a098a8d576502bb | 24230 |
| C:\dev\study\machine_learning\titanic_audit\0_타이타닉 내가 순서 정리한 것.txt | ee59d67f4d32f4ebed501b226fadec7a7465fab0be2f7be5db3bb6476be55298 | 16285 |
| C:\dev\study\machine_learning\titanic_audit\1_1. base model.txt | a2cf8891aecf6898145602adf58020573af498540b1e519bf73ef1c671f10fe7 | 8614 |
| C:\dev\study\machine_learning\titanic_audit\1_2. model.txt | 90344e3b972a6cc0c8d709ed0a1f5b3cd090d2842611d58663e0366285cddab7 | 19491 |
| C:\dev\study\machine_learning\titanic_audit\1_3. model_feature_add.txt | 1b38466aa7f8b17aaf438c8bc4261005b214afda5f0f39d3bb64d48409ac3ec3 | 31661 |
| C:\dev\study\machine_learning\titanic_audit\1_4. model_feature_add2.txt | 253ce2145a9d34b2bdcce3ea20c811bdc473ad1f7c61d63d2c3ef9db634b4ab8 | 28470 |
| C:\dev\study\machine_learning\titanic_audit\next_champion_depth5.py | aed225ed1535c54c0521dba751cae7c43ecc0868e7ab9f7754b574db568c6624 | 4920 |
| C:\dev\study\machine_learning\titanic_audit\oof_audit.csv | 304e31752c308fe1df6ecaf8d2add042f762e425f9bfadf1bb588865c52d20ed | 60086 |
| C:\dev\study\machine_learning\titanic_audit\REPORT.md | b8c85489831feebc0c27d49f5093f7c53c67d862ea0aab6a57b9eb3807b5cf7a | 57789 |
| C:\dev\study\machine_learning\titanic_audit\source_manifest.json | 890388a34c72bc7d8c326769a277fe9a96ae4f6511e05068b605e8b404a68e26 | 13145 |
| C:\dev\study\machine_learning\titanic_audit\test_prediction_changes.csv | 2e2c081264ca39034735bf00785c8557e6fe7d208d4b07bdc85eb493aa8a3601 | 33676 |
| C:\dev\study\machine_learning\titanic_audit\verification.json | dc9fddd8ebacb3738c2bc57835a3da97805f5e9b303c3d8190e722679634654f | 28277 |
| C:\dev\study\machine_learning\titanic_audit\verify.py | ceb99f4b9527009c0d2b6d1a52edbe109c4b0d60f376fa2c824feb73fa95c6eb | 6667 |
| C:\dev\study\machine_learning\titanic_audit\depth5_smoke\fold_metrics.csv | 0aa97f8c5e4e7f38c02ba4e8c13eaa6294ed10406bc52a5218d785de07d0792f | 215 |
| C:\dev\project\kaggle_skn35 v1\1. base model.ipynb | 035410b5dcc30dd285960cf8ac2d5a7aae66dfadded100c163108cc22845d77e | 66103 |
| C:\dev\project\kaggle_skn35 v1\2. model.ipynb | d257adfc826e4cc6cb20e4c7748b05575e2a954a8a5b87c1151ff0b2d2dd7ca1 | 176517 |
| C:\dev\project\kaggle_skn35 v1\3. model_feature_add.ipynb | 001b3f56d182272639cdc6384cc58eb259cfef150b25edaed4fa48d94f90d42b | 382626 |
| C:\dev\project\kaggle_skn35 v1\4. model_feature_add2.ipynb | 4ba19cd007ce36d07cf42669c4ca0ce52877ab415fdabeb2243f445fac4b0e7b | 354686 |
| C:\dev\project\kaggle_skn35 v1\pyproject.toml | 5c806a710f53d36c6f0fb3899ff07680a6b75ef9c61e4786abc5eaa804bb7157 | 404 |
| C:\dev\project\kaggle_skn35 v1\catboost_info\catboost_training.json | 7bc85b740f501093155822d129349e74bd493d9944a32ef5ca9cb3097a1ff092 | 96806 |
| C:\dev\project\kaggle_skn35 v1\catboost_info\learn_error.tsv | 73616d81d582c790b522bfb26f77ed49061fd4a121d5f9f6d7655e8c25dbf44d | 16788 |
| C:\dev\project\kaggle_skn35 v1\catboost_info\time_left.tsv | cbe6555789f6ccc45877c506e071c96b0a870ab6e9411535cc1c6ffe7979b163 | 15360 |
| C:\dev\project\kaggle_skn35 v1\common\evaluations.py | f817f4f8fef34d7ad4aef5a96d89726630a2a6603c92845d40e9ceecd0fcc015 | 277 |
| C:\dev\project\kaggle_skn35 v1\common\evaluation_plots.py | af84b436fc95457ec5de30d31e7f3e7119282f4b36a94401b53f336007a824f9 | 388 |
| C:\dev\project\kaggle_skn35 v1\common\modeling.py | c707970c8bfb9554e4f8df4e4179a1fd47c29f54aeb48d5355e5194fb90a2692 | 5652 |
| C:\dev\project\kaggle_skn35 v1\common\utils.py | 004ee1974091c6fe90b251d80ee19d166682c3f1cbeffec76e111fb0529fe418 | 763 |
| C:\dev\project\kaggle_skn35 v1\data\submission.csv | 723ab89193140f3c5a901ca96a7fcbb9869bba674e385a1c7ff77cdbd029e697 | 3868 |
| C:\dev\project\kaggle_skn35 v1\data\test.csv | 9df3361de7838476cc9a9d216e09f58edbf6b1267dfd188d8a504b943671a070 | 27384 |
| C:\dev\project\kaggle_skn35 v1\data\train.csv | f752a254eaf503f4cfd0dacd6854b4e9638d78d65ee9e9944b8db8895d991617 | 64952 |
| C:\dev\project\kaggle_skn35 v1\submission\2. model_result.csv | cfb459e2d17c6e4532447756ae5826b90a4950c4616cbb84e0646847505a1acb | 3082 |
| C:\dev\project\kaggle_skn35 v1\submission\3. model_feature_add_result.csv | d6c990c8a2ec4ed96417cc3162f58cce6b53d66b3339c38b12093321192ba075 | 6640 |
| C:\dev\project\kaggle_skn35 v1\submission\4. model_feature_add2_result.csv | 44df0c345325055e78cedf630161de72437af79a6e933525e58054b19bea65f6 | 9935 |
| C:\dev\project\kaggle_skn35 v1\submission\base_model_0909_01_(95.36).csv | 651b807c49042e2301811e685046713173b89e08f6a01fcb61d777ae617b3386 | 3082 |

## 부록 B. 이번 실행의 모든 평가

| Model | CV Mean | CV Std | OOF AUC | Train AUC | Train-CV Gap | Feature Count | spec | params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: historical Private champion 15 | 0.906275 | 0.016868 | 0.904247 | 0.965376 | 0.059101 | 33 | {'drop': [], 'add': []} | {} |
| 28: fixed deployment recipe | 0.908135 | 0.013804 | 0.904607 | 0.922685 | 0.014550 | 32 | {'drop': ['age'], 'add': []} | {'iterations': 20, 'learning_rate': 0.03, 'loss_function': 'Logloss', 'eval_metric': 'AUC'} |
| drop pclass | 0.905717 | 0.019149 | 0.903385 | 0.965290 | 0.059573 | 32 | {'drop': ['pclass'], 'add': []} | {} |
| drop gender | 0.905845 | 0.018214 | 0.903491 | 0.964723 | 0.058877 | 31 | {'drop': ['gender'], 'add': []} | {} |
| drop age | 0.907518 | 0.013868 | 0.905643 | 0.958796 | 0.051278 | 32 | {'drop': ['age'], 'add': []} | {} |
| drop sibsp | 0.906485 | 0.020001 | 0.903851 | 0.964815 | 0.058330 | 32 | {'drop': ['sibsp'], 'add': []} | {} |
| drop parch | 0.908099 | 0.017823 | 0.904551 | 0.965532 | 0.057433 | 32 | {'drop': ['parch'], 'add': []} | {} |
| drop fare | 0.899753 | 0.020324 | 0.897488 | 0.955564 | 0.055810 | 32 | {'drop': ['fare'], 'add': []} | {} |
| drop embarked | 0.905866 | 0.021729 | 0.904652 | 0.963916 | 0.058050 | 30 | {'drop': ['embarked'], 'add': []} | {} |
| drop FamilySize | 0.906103 | 0.017031 | 0.903476 | 0.965022 | 0.058918 | 32 | {'drop': ['FamilySize'], 'add': []} | {} |
| drop IsAlone | 0.905642 | 0.016868 | 0.903810 | 0.965270 | 0.059628 | 32 | {'drop': ['IsAlone'], 'add': []} | {} |
| drop GenderClass | 0.904100 | 0.016280 | 0.902152 | 0.965473 | 0.061374 | 27 | {'drop': ['GenderClass'], 'add': []} | {} |
| drop GenderIsChild | 0.904785 | 0.018387 | 0.902523 | 0.965300 | 0.060515 | 29 | {'drop': ['GenderIsChild'], 'add': []} | {} |
| drop ClassIsChild | 0.902990 | 0.018164 | 0.901519 | 0.965457 | 0.062467 | 27 | {'drop': ['ClassIsChild'], 'add': []} | {} |
| drop AgeBand | 0.906233 | 0.018564 | 0.904389 | 0.964405 | 0.058172 | 28 | {'drop': ['AgeBand'], 'add': []} | {} |
| new FamilyFarePerPerson | 0.906449 | 0.016838 | 0.904541 | 0.968407 | 0.061959 | 33 | {'drop': ['parch'], 'add': ['FamilyFarePerPerson']} | {} |
| new ChildFamilySize | 0.906532 | 0.017471 | 0.903739 | 0.964995 | 0.058463 | 33 | {'drop': ['parch'], 'add': ['ChildFamilySize']} | {} |
| tune 01 | 0.907768 | 0.015017 | 0.905709 | 0.937244 | 0.029476 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 300, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 3} |
| tune 02 | 0.907566 | 0.017864 | 0.905694 | 0.948789 | 0.041223 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 600, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 3} |
| tune 03 | 0.905941 | 0.018372 | 0.904143 | 0.961581 | 0.055640 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 1000, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 3} |
| tune 04 | 0.910049 | 0.016111 | 0.906589 | 0.935022 | 0.024973 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 600, 'learning_rate': 0.01, 'depth': 4, 'l2_leaf_reg': 10} |
| tune 05 | 0.909487 | 0.016209 | 0.906772 | 0.944506 | 0.035019 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 1000, 'learning_rate': 0.007, 'depth': 4, 'l2_leaf_reg': 3} |
| tune 06 | 0.909475 | 0.015828 | 0.906396 | 0.940704 | 0.031229 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 300, 'learning_rate': 0.03, 'depth': 4, 'l2_leaf_reg': 10} |
| tune 07 | 0.905128 | 0.017022 | 0.902898 | 0.954056 | 0.048928 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 600, 'learning_rate': 0.01, 'depth': 6, 'l2_leaf_reg': 10, 'random_strength': 0.2} |
| tune 08 | 0.908916 | 0.015448 | 0.906701 | 0.952438 | 0.043522 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 1000, 'learning_rate': 0.005, 'depth': 6, 'l2_leaf_reg': 3} |
| tune 09 | 0.905390 | 0.019270 | 0.903367 | 0.954688 | 0.049298 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 600, 'learning_rate': 0.015, 'depth': 5, 'l2_leaf_reg': 10, 'random_strength': 0.2, 'rsm': 0.8} |
| tune 10 | 0.907445 | 0.015695 | 0.905717 | 0.950512 | 0.043068 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 600, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 3, 'border_count': 64} |
| tune 11 | 0.907086 | 0.016930 | 0.905010 | 0.949265 | 0.042179 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 600, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 3, 'border_count': 128} |
| tune 12 | 0.908264 | 0.015176 | 0.905405 | 0.941275 | 0.033011 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 600, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 0} |
| tune 13 | 0.908339 | 0.015078 | 0.905359 | 0.942617 | 0.034278 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 600, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 1} |
| tune 14 | 0.909229 | 0.015200 | 0.906214 | 0.950715 | 0.041486 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 600, 'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 10, 'grow_policy': 'Depthwise', 'min_data_in_leaf': 10} |
| tune 15 | 0.907949 | 0.014831 | 0.905202 | 0.949181 | 0.041232 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 600, 'learning_rate': 0.01, 'depth': 6, 'l2_leaf_reg': 20, 'grow_policy': 'Depthwise', 'min_data_in_leaf': 20} |
| tune 16 | 0.908448 | 0.018992 | 0.906711 | 0.949650 | 0.041202 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 817, 'learning_rate': 0.006971115660907851, 'depth': 4, 'l2_leaf_reg': 1.9452208847287389, 'random_strength': 0.3746261155838959, 'subsample': 0.9328729111737557, 'border_count': 128} |
| Champion: ticket-group stress CV | 0.890189 | 0.036584 | 0.889991 | 0.966934 | 0.076745 | 33 | {'drop': [], 'add': []} | {} |
| Final: ticket-group stress CV | 0.898528 | 0.037083 | 0.896671 | 0.936663 | 0.038134 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 600, 'learning_rate': 0.01, 'depth': 4, 'l2_leaf_reg': 10} |
| Final OOF explanations | 0.910049 | 0.016111 | 0.906589 | 0.935022 | 0.024973 | 32 | {'drop': ['parch'], 'add': []} | {'iterations': 600, 'learning_rate': 0.01, 'depth': 4, 'l2_leaf_reg': 10} |

## 참고 문서

[CatBoost 파라미터](https://catboost.ai/docs/en/references/training-parameters/common), [scikit-learn 모델 선택과 nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)