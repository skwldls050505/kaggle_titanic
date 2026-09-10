# Titanic 전체 실험 분석 — Private Leaderboard 우선

분석일: 2026-09-10. **현재 Champion은 `titanic_result_15.csv`, Private 0.91468, 사용자 제공 최종 순위 1등이다.** Public 0.89121이라는 이유로 11번으로 되돌릴 근거는 없다.

이번 분석에서 새롭게 확인한 핵심은 다음과 같다. **15번은 11번에 AgeBand만 추가한 모델이며, 기존 기록에서는 holdout이 낮다는 이유로 CV를 실행하지 않았다. 직접 보완 실행한 결과, 15번 CV 평균은 0.906275로 11번 0.906233과 사실상 동률이고 fold 간 표준편차는 감소했다.** 다만 전체 OOF AUC는 약간 낮다. 따라서 Private 우승은 확정된 대회 결과로 받아들이되, 모든 모집단에서 더 우수하다는 주장으로 확대하지 않는다.

## 1. 분석 범위와 증거의 구분

분석한 두 작업 폴더:

- [machine_learning](C:/dev/study/machine_learning): 22개 notebook, 7개 Python 공통 모듈, 원본 CSV 3개, 제출 CSV 15개, 과거 결과 JSON 2개와 보고서·가이드.
- [kaggle_skn35 v1](<C:/dev/project/kaggle_skn35 v1>): 4개 notebook, 4개 Python 공통 모듈, 원본 CSV 3개, 제출 CSV 4개.

가상환경의 외부 라이브러리와 `__pycache__`는 프로젝트 실험 코드에 포함하지 않았다. `pyproject.toml`과 환경 버전, CatBoost 로그 파일의 존재도 확인했다. 공유 `catboost_info`는 실험별 식별자가 없고 후속 실행이 덮어쓸 수 있어 특정 제출의 검증 점수로 사용하지 않았다.

26개 notebook의 코드·마크다운·저장된 텍스트 출력을 [분석 폴더](C:/dev/study/machine_learning/titanic_audit)에 추출했다. `0_`는 machine_learning, `1_`는 kaggle_skn35 계열이다. 아래의 **셀 번호는 마크다운을 포함한 1부터 시작하는 물리적 셀 번호**로, 실행 카운터와 다르다.

증거를 세 종류로 구분했다.

| 종류 | 사용 범위 |
|---|---|
| 사용자 제공 결과 | Private/Public 5개 점수와 15번 최종 1등. Kaggle 서버에서 별도로 조회한 결과는 아님 |
| 기존 코드·저장 출력 | 실험 가설, 피처, 기존 holdout/CV, 제출 생성 코드 |
| 이번 재실행 | Base·11·15 holdout, 전체 train 재학습, 제출 재현, 동일 5-fold CV, OOF, 모든 제출 CSV 검사 |

두 폴더의 train/test/submission 원본은 **파일 SHA-256까지 동일**하다. 데이터는 train 916명, test 393명이며 train 생존 346명, 사망 570명이다. CSV 해시는 [verification.json](C:/dev/study/machine_learning/titanic_audit/verification.json)의 데이터·제출 기록, 전체 분석 대상 코드와 데이터의 해시는 [source_manifest.json](C:/dev/study/machine_learning/titanic_audit/source_manifest.json)에서 확인할 수 있다.

재현 환경은 Python 3.12.13, pandas 3.0.5, CatBoost 1.2.10, scikit-learn 1.9.0이다. Base·11·15의 저장된 제출 확률과 재학습 확률은 **최대 절대 차이 1.1102230246251565e-16**으로 일치했다. 원본 notebook과 제출 CSV는 수정하지 않았다.

### 평가 Metric 확인 범위

프로젝트의 `get_auc_score()`는 ROC curve의 면적을 계산한다. 주 실험 notebook도 사용자가 지정한 AUC를 평가 기준으로 명시하고, 제출 템플릿은 `survived=0.5`인 실수형이다. 따라서 이 보고서는 **요청대로 ROC-AUC 대회로 분석**한다. 다만 대회 URL·Evaluation 규정 파일·Public/Private 행 구분은 없어서 공식 규정과 분할 비율을 독립 확인하지 못했다. 템플릿의 0.5만으로 AUC 대회임을 증명할 수는 없다. 다른 Titanic 대회의 규정을 가져와 이 대회에 적용하지 않았다.

ROC-AUC는 생존자에게 사망자보다 높은 점수를 주는 정도를 측정한다. 확률 또는 threshold를 적용하지 않은 연속 점수를 사용할 수 있으므로, **수학적으로 반드시 보정된 확률이어야 하는 것은 아니다. 이 프로젝트에서는 `predict_proba()`의 생존 확률을 제출하는 것이 적절하다.** 0/1로 자르면 다수의 순위 정보가 사라진다. [scikit-learn ROC-AUC 문서](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

## 2. 파일명에 의존하지 않고 복원한 실험 흐름

두 계열의 숫자는 같은 실험 번호가 아니다. 또한 파일 수정 시각은 복사·편집 시각일 수 있어 서로 다른 계열의 절대 실행 순서는 확정하지 않았다.

**주 실험 계열(machine_learning)**

`titanic_1`의 CatBoost Base → `titanic_2` 내부의 가설 1 HasCabin / 가설 2 AgeMissing → 3 Title → 4 SexPclass 채택 → 5~9 각 후보 비교 → 10 GenderIsChild 채택 → 11 ClassIsChild 채택 → 12~15 각 후보 비교.

4~9의 `SexPclass`와 10~15의 `GenderClass`는 모두 `gender + '_' + pclass`다. **이름 변경이지 새 원천 피처 추가가 아니다.** 12~15의 직전 기준은 모두 11번이다. 12→13→14→15의 실패한 피처를 계속 누적하지 않았다.

`titanic_1.ipynb`는 Base다. **실험 1=HasCabin과 파일 titanic_1은 다르다.** 두 첫 가설은 `titanic_2.ipynb`에 함께 있다. 이 파일의 제출은 둘 다 제외한 Base이므로 `titanic_result_2.csv`와 `submission_result_1.csv`는 실제 예측값도 완전히 같다.

**별도 프로젝트 계열(kaggle_skn35)**

`1. base model`(학습셋 재평가, hard label 제출) → `2. model`(SMOTE/CV 시도) → `3. model_feature_add`(가족·호칭·직업·귀족) → `4. model_feature_add2`(객실·티켓·요금·나이 치환 + CatBoost 단독).

이는 위 가설 3·4와 다르다. 특히 `3. model_feature_add_result.csv`는 **최종 XGBoost**, `4. model_feature_add2_result.csv`는 **최종 CatBoost**다. 같은 CatBoost에서 피처만 늘린 비교로 해석하면 안 된다.

**보조·후속 파일**

| 파일 | 확인한 역할과 한계 |
|---|---|
| `titanic_1 copy.ipynb` | EDA가 확장된 Base 복사본. 후반 모델·제출 코드가 동일 Base이며 동일 제출명을 씀. 실행 카운터가 비어 있는 셀에도 저장 출력이 있어 전체 실행 순서의 증거로 삼기 어려움 |
| `my_folder/titanic_eda.ipynb` | 동일 916명 데이터, 동일 학습 부분 중심 EDA. 대회 제출 생성 모델 아님 |
| `my_folder/타이타닉 내가 순서 정리한 것.ipynb` | seaborn Titanic 891명, 30% 검증. 중복 제거 후 564명 학습/268명 검증. 별도 학습 예제이며 현재 대회 점수와 직접 비교 불가 |
| `my_folder/3-3. 앙상블 예제 - 모듈화.ipynb` | seaborn 891명, 25% 검증. `alive` 제거. 현재 공통 모듈과 과거 저장 출력이 동일 버전이라는 보장은 없음 |
| `titanic_16_isnoble.ipynb` | **Base+IsNoble** 실험. Val 0.899821, CV 0.898784±0.014441. 15번 후속 누적 모델이 아님 |
| `titanic_17_profession_group.ipynb` | **Base+ProfessionGroup** 실험. Val 0.899496, CV 0.900609±0.016918. 역시 15번 기준 아님 |
| `titanic_base_smote_experiment.ipynb` | Base+SMOTE 별도 실험. Val 0.900065→0.900146. 후반 제출 셀에는 미정의 변수가 있어 단독 재실행 불완전 |

## 3. Base~15 전체 비교표

피처 표기:

- **B** = `pclass, gender, age, sibsp, parch, fare, embarked, FamilySize, IsAlone` 9개 의미 피처. OHE 후 12열.
- **G** = SexPclass/GenderClass, OHE 6열.
- **C** = GenderIsChild, OHE 4열.
- **D** = ClassIsChild, OHE 6열.
- **A** = AgeBand, OHE 5열.

표의 Train/Val은 동일 687명/229명 holdout의 점수다. Gap=Train−Val. `—`는 **후보 CV 미실행**이다. 피처의 “제외”는 그 조건에서 누적하지 않는 운영 판단이며, 통계적으로 무효라는 뜻은 아니다. 작은 holdout 하락만 있는 후보는 이번 분석에서 보류로 완화했다.

| 실험 | 가설 | 변경·Feature | Train AUC | Val AUC | Gap | CV Mean | CV Std | 결과와 현재 판단 |
|---|---|---|---:|---:|---:|---:|---:|---|
| Base | 기본 승객 정보와 가족 규모로 기준 순위 성능을 확보한다 | B, 12열 | .963079 | .900065 | .063014 | .901775 | .016322 | CatBoost 선정; 채택(비교 기준) |
| 1 | 객실 기록 존재 여부가 등급·요금 외 신호를 보존한다 | B+HasCabin, 13열 | .964623 | .898439 | .066185 | — | — | 과거 제외; 현재 보류(CV 없음) |
| 2 | 나이 누락 자체가 치환 후 사라지는 정보를 담는다 | B+AgeMissing, 13열 | .964178 | .899943 | .064235 | — | — | 과거 제외; 보류(차이 극소) |
| 3 | 호칭이 이름에서 성별·연령·신분 정보를 낮은 차원으로 보존한다 | B+Title, 전체 재학습 18열 | .965168 | .902992 | .062176 | .900010 | .016136 | CV 하락; 제외 |
| 4 | 성별에 따라 객실 등급의 생존 효과가 다르다 | B+G, 18열 | .963047 | .902911 | .060136 | .904676 | .016132 | holdout/CV 상승; 채택 |
| 5 | 가족 규모를 단독·소가족·대가족으로 묶으면 비선형 패턴이 쉬워진다 | B+G+FamilyGroup, 21열 | .964434 | .902748 | .061686 | — | — | 과거 제외; 보류 |
| 6 | 공유 티켓의 인원수가 가족 외 동행 정보를 담는다 | B+G+TicketGroupSize, 19열 | .964691 | .899984 | .064707 | — | — | train 상승·val 하락; 제외 |
| 7 | 공동 요금을 인원수로 나누면 개인 수준 요금 신호가 개선된다 | B+G+FarePerPerson, 19열 | .965835 | .902748 | .063087 | — | — | 과거 제외; 보류 |
| 8 | 16세 미만 표시가 어린이의 다른 생존 패턴을 표현한다 | B+G+IsChild, 19열 | .964218 | .903155 | .061063 | .902251 | .017383 | CV 하락; 제외 |
| 9 | 객실 갑판이 등급 이상의 위치 정보를 담는다 | B+G+CabinDeck, 전체 27열 | .964011 | .904781 | .059230 | .902823 | .013704 | 평균 하락·변동 감소; 보류 |
| 10 | 어린이 효과는 성별에 따라 다르다 | B+G+C, 22열 | .965065 | .904293 | .060771 | .905907 | .017324 | holdout/CV 상승; 채택 |
| 11 | 어린이 효과는 객실 등급에 따라 다르다 | B+G+C+D, 28열 | .964795 | .906489 | .058306 | .906233 | .018564 | Private .91254; 채택 |
| 12 | 동반 승선 효과는 성별에 따라 다르다 | B+G+C+D+GenderIsAlone, 32열 | .963471 | .904619 | .058852 | — | — | 과거 제외; 보류 |
| 13 | 항구와 등급 조합이 승객 구성 차이를 담는다 | B+G+C+D+EmbarkedClass, 37열 | .963489 | .905757 | .057732 | — | — | 과거 제외; 보류 |
| 14 | 티켓 접두사가 발권 유형 정보를 담는다 | B+G+C+D+TicketPrefix, 전체 42열 | .969699 | .902423 | .067276 | — | — | train 상승·val 하락 뚜렷; 제외 |
| 15 | 연령을 여러 구간으로 표현하면 어린이/성인 이외의 패턴을 학습한다 | B+G+C+D+A, 33열 | .965299 | .903643 | .061656 | **.906275*** | **.016868*** | **Private .91468, 최종 1등; 채택·Champion** |

\* 15번 CV는 **이번 분석에서 새로 계산**했다. 기존 JSON은 `cv: null`, `accepted: false`이며 `best_cv=.906233`은 11번 값이다. 그 값을 15번 CV로 옮겨 적으면 안 된다. Base CV는 원래 Base notebook이 아니라 후속 비교에서 계산했고 이번에 재현했다. 희귀 범주 피처의 OHE 열 수는 학습 표본마다 달라질 수 있다.

근거: [실험 1·2 셀 26~36](C:/dev/study/machine_learning/titanic_audit/0_titanic_2.txt), [기존 3~9 JSON](C:/dev/study/machine_learning/feature_experiment_results_3_9.json), [기존 10~15 JSON](C:/dev/study/machine_learning/feature_experiment_results_10_15.json), 각 후보 notebook의 피처·평가·재학습·제출 셀 및 [이번 재검증 결과](C:/dev/study/machine_learning/titanic_audit/verification.json). 11번은 중간 시각화 셀이 추가되어 모델 학습/평가/재학습/예측/저장이 각각 셀 35/37/39/41/43이다.

### 주 실험의 공통 실행 조건

| 항목 | Base~15 실제 조건 |
|---|---|
| 원본 제외 | passengerid는 식별자, survived는 target으로 분리. cabin 원문은 학습 결측률 ≥20%라 제거. name/ticket 원문 제거 |
| 나이 결측 | 학습 부분의 gender×pclass 중앙값 → 없으면 학습 전체 중앙값 |
| 기타 결측 | 수치형 학습 중앙값, 범주형 학습 최빈값 |
| FE | FamilySize=sibsp+parch+1, IsAlone=(FamilySize==1). 후보별 파생 피처 추가 |
| Encoding | 학습에서만 OneHotEncoder(handle_unknown='ignore', sparse_output=False). pclass는 수치형 유지. CatBoost cat_features=[] |
| Scaling / SMOTE | 둘 다 없음 |
| Holdout | stratify=survived, seed 42, validation 25%=229명, train 687명 |
| CV | StratifiedKFold(5, shuffle=True, random_state=42), 전체 916명. 매 fold에서 전처리·모델 신규 fit |
| 모델 | Base는 XGB/LGB/Cat 비교 후 Cat 선택. 이후 후보는 CatBoost 고정 |
| 명시 설정 | verbose=0, random_state=42, cat_features=[], allow_writing_files=False |
| 주요 실효 설정 | iterations=1000, depth=6, l2_leaf_reg=3, random_strength=1, loss_function=Logloss, bootstrap_type=MVS, subsample≈0.8, CPU |
| 학습률 | 자동 결정. 687명 holdout 학습 ≈0.008776, 916명 최종 학습 ≈0.009923 |
| Threshold | AUC·제출에는 없음. 혼동행렬용 기본 분류 경계는 0.5 |
| 평가 | 외부 ROC-AUC. 내부 loss/eval_metric=Logloss는 AUC 계산 오류가 아님. early stopping 없음 |
| 최종 제출 | 후보 피처로 916명 전체 재학습 → test transform → 생존 확률 → ID 매핑 → index=False |

Base 모델 비교의 저장 결과: CatBoost .963079/.900065, LightGBM .994433/.894088, XGBoost .996897/.888396(Train/Val). 이 조건에서는 CatBoost가 더 작은 Gap과 높은 Val을 보였다. 모든 문제에서 CatBoost가 우수하다는 뜻은 아니다.

자동 learning_rate는 데이터와 iteration 수 등에 따라 정해진다. 따라서 같은 명시적 설정이라도 holdout 모델과 전체 재학습 모델의 실효 학습률은 달라질 수 있다. **Base·11·15끼리는 같은 학습 크기에서 실효 파라미터가 일치한다.** [CatBoost 파라미터 문서](https://catboost.ai/docs/en/references/training-parameters/common)

## 4. 요청한 다섯 제출 파일과 생성 코드 연결

### `3. model_feature_add_result.csv`

- 대응 코드: [3. model_feature_add.ipynb](<C:/dev/project/kaggle_skn35 v1/3. model_feature_add.ipynb>), 셀 14·19·21(피처), 25(CV), 33(최종 학습), 35(확률), 37·39(저장).
- 최종 모델: **XGBClassifier**. `Modeling`이 XGB/LGB/Cat를 학습한 뒤 **전체 SMOTE 학습 데이터에 대한 점수**로 선택했다. 저장 출력에서 XGB가 확인된다.
- 피처: `pclass, gender, age, sibsp, parch, fare, FamilySize, IsAlone, Occupation, IsNoble, Title` 11개. embarked 제외.
- Title: 전체 train에서 빈도 ≥10인 Mr/Miss/Mrs/Master, 나머지 Rare. Occupation: Dr→Doctor, Rev→Clergy, Col/Major/Capt→Military, 나머지 Other. 귀족 호칭 6종을 IsNoble로 변환.
- 결측: CV 분리 전 전체 train 평균/최빈값. SMOTE용 OrdinalEncoder는 fold train에서 fit, SMOTE(seed=0), 범주 코드를 반올림. Scaling 없음.
- CV: seed=0, 5-fold. SMOTE 없음 .899779±.017593, SMOTE 있음 **.899769±.014351**(ddof=0). 각 fold에서 최상의 모델을 골라 같은 fold에 보고하는 선택 편향이 있다.
- 최종 저장 train_score/test_score 모두 **.996054**이나, 동일한 합성 학습셋 재평가다. 독립 Val AUC가 아니다.
- Private **.90242**, Public **.90621**.
- 가능한 해석: 이름·가족 신호가 도움이 될 수 있으나 독립 효과를 측정하지 않았다. Public 우위만으로 좋은 일반화라고 판단할 수 없다. 주 계열 Base와는 모델, 인코딩, SMOTE, 결측치, embarked가 함께 다르다.

### `4. model_feature_add2_result.csv`

- 대응 코드: [4. model_feature_add2.ipynb](<C:/dev/project/kaggle_skn35 v1/4. model_feature_add2.ipynb>), 셀 12·14·16·18·22(전처리), 30(CV), 37(전체 학습), 39(확률), 41(저장).
- 최종 모델: **CatBoostClassifier(verbose=0, cat_features=[pclass,gender,Title,Occupation,Deck])**. random_seed를 명시하지 않아 라이브러리 기본 0을 사용한다.
- 피처: 위 11개 + **HasCabin, Deck, TicketGroupSize, FarePerPerson** =15개.
- 나이 치환도 전체 train의 **Title×pclass 중앙값 → Title 중앙값 → 전체 중앙값**으로 변경.
- TicketGroupSize는 **train에서 train 빈도, test에서 test 빈도를 각각 계산**한다. FarePerPerson=fare/TicketGroupSize.
- CV: CatBoost 고정, SMOTE seed=0, split seed=0. **.912477±.011546**(ddof=0). CV 전 전체 train 통계가 사용돼 완전한 fold 격리가 아니다. 별도 holdout과 train AUC는 보고되지 않았다.
- Private **.90242**, Public **.89774**.
- 이전 대비 CV +.012708, Private는 표시 정밀도상 동일, Public −.00847. 여러 변화가 동시에 있어 Deck이나 요금 하나를 원인으로 지목할 수 없다. CV 누수·학습/예측 통계 차이·모델 변경이 해석을 제한한다.

### `submission_result_1.csv`

- 대응 코드: [titanic_1.ipynb](C:/dev/study/machine_learning/titanic_1.ipynb), 셀 25·27(모델 선택), 29(전체 재학습), 31(확률), 33(저장). 복사본도 같은 파일명을 사용한다.
- 모델/피처: CatBoost, **B 12열**, SMOTE 없음, 학습 부분에만 전처리 fit.
- Val .900065, Train .963079, Gap .063014, CV .901775±.016322.
- Private **.91118**, Public **.89672**.
- 코드에는 루트 `submission_result_1.csv`를 저장하지만 현재 파일은 `submission/`에 있다. 이동 이력은 없으나 **동일 Base 코드로 예측값을 재현**했다.
- 이름 계열 제출보다 높은 Private를 “특정 피처 삭제 덕분”으로 단정할 수 없다. 검증 설계, 모델, 인코딩, SMOTE 여부까지 다르다.

### `titanic_result_11.csv`

- 대응 코드: [titanic_11.ipynb](C:/dev/study/machine_learning/titanic_11.ipynb), 셀 3·39·41·43. `candidate_features`로 재학습한 `submission_model.predict_proba()` 결과를 저장한다.
- 모델/피처: CatBoost, **B+G+C+D =28열**.
- 직전 10번 대비 ClassIsChild만 추가. Base 대비 GenderClass, GenderIsChild, ClassIsChild의 세 상호작용 추가.
- Train .964795, Val .906489, Gap .058306, CV .906233±.018564.
- Private **.91254**, Public **.89672**.
- 가능한 향상 원인: 성별·등급별 어린이/성인 관계를 트리가 직접 활용할 수 있는 입력으로 표현. Base 대비 로컬과 Private의 개선 방향이 일치한다.

### `titanic_result_15.csv`

- 대응 코드: [titanic_15.ipynb](C:/dev/study/machine_learning/titanic_15.ipynb), 셀 3에서 AgeBand 추가, 29에서 **candidate_features** 재학습, 31 확률 예측, 33 저장.
- 모델/피처: CatBoost, **B+G+C+D+A =33열**. 11번 피처는 모두 유지.
- AgeBand: 치환한 age를 `[0,16), [16,20), [20,35), [35,60), [60,∞)`로 구분. 순서형 정수가 아닌 원핫 범주다. raw age도 유지한다.
- Train .965299, Val .903643, Gap .061656. **기존 후보 CV 없음 → 이번 .906275±.016868**, OOF .904247.
- Private **.91468**, Public **.89121**, **최종 1등**.
- `best_features`는 과거 holdout 규칙 때문에 11번 피처를 가리킨다. **제출된 15번은 `best_final_model`이 아니라 AgeBand를 포함한 `submission_model`**이다.
- 셀 34의 “CabinDeck 후보”는 복사된 잘못된 설명이다. 코드·33열 출력·제출 재현 모두 AgeBand를 확인한다.

## 5. Public은 떨어졌는데 Private은 왜 올랐는가

### 점수 변화 자체

| 제출 | Private | Public | Private−Public | Private 순위(제공된 5개) |
|---|---:|---:|---:|---|
| feature_add | .90242 | .90621 | −.00379 | 공동 4 |
| feature_add2 | .90242 | .89774 | +.00468 | 공동 4 |
| Base 제출 | .91118 | .89672 | +.01446 | 3 |
| 11 | .91254 | .89672 | +.01582 | 2 |
| **15** | **.91468** | **.89121** | **+.02347** | **1** |

15−11은 Private **+.00214**, Public **−.00551**다. 동일한 모델이 서로 다른 평가 표본에서 다른 순위 성능을 보인 것이다. 점수 두 개로 Public/Private 집단의 어떤 특징이 원인인지 역산할 수는 없다.

### 실제 확인한 데이터·예측 변화

| 항목 | Train 전체 | Test 전체 | Holdout 학습 | Holdout 검증 |
|---|---:|---:|---:|---:|
| 표본 수 | 916 | 393 | 687 | 229 |
| 여성 비율 | 35.70% | 35.37% | 34.93% | 37.99% |
| 2등급 비율 | 21.18% | 21.12% | 23.14% | 15.28% |
| 3등급 비율 | 54.04% | 54.45% | 52.84% | 57.64% |
| 나이 결측 | 19.65% | 21.12% | 18.34% | 23.58% |
| 객실 결측 | 78.38% | 75.32% | 79.04% | 76.42% |
| 관측 나이 평균 | 29.70 | 30.32 | 29.51 | 30.31 |
| 운임 평균 | 32.40 | 35.38 | 30.96 | 36.72 |

전체 Train/Test의 성별·등급 구성은 꽤 비슷하다. **이 결과만으로 큰 분포 이동이 발생했다고 주장할 수 없다.** 반면 holdout의 2등급 비중은 학습보다 낮고 나이 결측은 더 높다. target stratify는 이 변수들까지 균형화하지 않는다. 단일 검증셋이 연령·등급 상호작용 평가에 다른 가중치를 줄 가능성은 실제 데이터로 뒷받침된다.

Test 393명 중 152명은 train에도 등장하는 티켓을 가진다. 이는 가족·동행 표본의 의존 가능성을 보여주지만 그 자체로 target leakage는 아니다. 새 가족에 대한 성능을 따로 평가하려면 티켓 그룹 분할이 필요하다.

11→15의 test 확률 차이:

- 평균 절대 변화 **.008921**, 최대 **.082489**.
- 예측 순위 Spearman 상관 **.988985**: 전체적으로 비슷하지만 세부 순위가 바뀐다.
- 0.5 기준 class가 바뀐 승객은 **1명뿐**이다. 확률 제출이므로 나머지 승객 간 순서 변화도 AUC를 바꿀 수 있다.

| AgeBand | Test 인원 | 15−11 평균 확률 변화 | 평균 절대 변화 |
|---|---:|---:|---:|
| Child | 36 | −.001034 | .014163 |
| Teen | 33 | −.006123 | .009137 |
| YoungAdult | 204 | +.002839 | .006113 |
| Adult | 107 | −.007134 | .012424 |
| Senior | 13 | +.001098 | .009083 |

YoungAdult 확률은 평균적으로 올라가고 Adult/Teen은 내려갔다. **어느 변화가 정답 방향인지, Public/Private 중 어디에 더 많았는지는 정답과 분할 마스크가 없어 모른다.** 이 표를 “Private에는 청년이 많다”는 주장으로 사용하면 안 된다. 승객별 수치는 [test_prediction_changes.csv](C:/dev/study/machine_learning/titanic_audit/test_prediction_changes.csv)에 있다.

### 요청한 원인 후보별 판정

| 가능성 | 확인 결과 |
|---|---|
| 1. Public/Private 분포 차이 | 가능하지만 직접 확인 불가. 전체 Test만 있고 분할 마스크가 없음 |
| 2. 작은 데이터의 Leaderboard 분산 | 유력한 일반적 설명. Test 전체가 393명이고 각 subset은 더 작음. 정확한 표준오차·유의성은 계산 불가 |
| 3. AgeBand가 Private에 더 적합 | 해당 Private 평가에서는 실제 순위 개선. 새로운 데이터에서도 같은 효과인지 미확정 |
| 4. FE에 의한 variance 변화 | 새 CV의 fold 간 Std .018564→.016868. 제한적 근거는 있지만 모델 자체의 반복학습 분산을 직접 측정한 값은 아님 |
| 5. Over/Underfitting | 15는 holdout Train 상승·Val 하락·Gap 증가. 과적합 가능성도 남는다. 과소적합이라고 단정할 근거 없음 |
| 6. Validation 편향 | 동일 holdout 반복 사용과 실제 구성 차이 존재. 15를 CV 전에 탈락시킨 규칙은 취약 |
| 7. CV와 Leaderboard 방향 | 새 CV 평균은 Base<11<15. 다만 11→15 차이가 .000042뿐이고 OOF는 반대 |
| 8. Seed 영향 | 11/15 split/model seed 모두 42이므로 seed 변경이 원인은 아님. 같은 seed여도 피처가 달라지면 트리 학습 결과는 달라짐 |
| 9. SMOTE | Base/11/15는 미사용. 따라서 11→15 개선 원인이 될 수 없음. 별도 프로젝트 계열 비교에는 교란요인 |
| 10. Threshold | 11/15 확률 제출, threshold tuning 없음. 0.5 조정으로 제출 AUC가 개선된 결과가 아님 |
| 11~12. predict/proba | 핵심 5개 모두 확률 제출. 초기 별도 Base·2번은 class label 제출 문제 있음 |
| 13. Leakage | 주 Base~15에서 target/test가 fit에 들어간 코드 없음. 별도 프로젝트 2~4에는 CV 전 통계 계산 문제 있음 |
| 14. Train/Test 처리 차이 | 주 계열은 같은 fit/transform. 별도 feature_add2 티켓 빈도는 train/test 각각 계산 |
| 15. Encoding fit | 주 계열은 fold train에서만 OHE fit. 별도 SMOTE 계열도 OrdinalEncoder 자체는 fold train fit |
| 16. Test 통계 fit 사용 | 주 계열에서는 없음. feature_add2는 test 자체 티켓 빈도를 파생 피처로 사용. test 결측치 중앙값 fit은 확인되지 않음 |

ROC-AUC는 양성·음성 쌍의 순위에 기반하므로 단순 생존 비율 차이만으로 기대 AUC가 바뀌는 것은 아니다. **각 class 내부의 연령·성별·등급 구성과 예측 난도, 작은 표본에서의 쌍 구성**이 더 직접적인 요인이다. Public과 Private AUC의 단순 가중평균도 전체 Test AUC가 되지 않는다. 서로 다른 subset 사이의 양성·음성 쌍이 빠지기 때문이다.

## 6. Base→11→15 집중 분석

| 항목 | Base | 11 | 15 | 11→15 |
|---|---:|---:|---:|---:|
| 입력 열 수 | 12 | 28 | 33 | +5 |
| Train AUC | .963079 | .964795 | .965299 | +.000504 |
| Holdout AUC | .900065 | .906489 | .903643 | −.002846 |
| Gap | .063014 | .058306 | .061656 | +.003350 |
| CV Mean | .901775 | .906233 | .906275 | +.000042 |
| CV Std(ddof=1) | .016322 | .018564 | .016868 | −.001696 |
| OOF AUC(신규) | .899876 | .904389 | .904247 | −.000142 |
| Private | .91118 | .91254 | .91468 | +.00214 |
| Public | .89672 | .89672 | .89121 | −.00551 |

**Base→11:** GenderClass, GenderIsChild, ClassIsChild가 추가됐다. Val +.006424, CV +.004458, Gap −.004708, Private +.00136이다. 새 원천 정보를 얻은 것은 아니지만, 트리가 여러 조건을 거쳐 만들 관계를 입력에서 바로 사용할 수 있게 했다. 제한된 깊이·훈련 횟수에서 유용한 표현이 된 것으로 해석할 수 있다. 다만 CV Std는 커져 “분산까지 모두 줄었다”고 말할 수 없다.

**11→15:** raw age를 유지하면서 AgeBand를 추가했다. 연령 구간은 경계가 고정된 요약 표현이므로 비슷한 나이끼리 패턴을 공유하는 데 도움을 줄 수 있다. 동시에 입력 선택지가 증가하므로 복잡성이 늘거나 noise를 학습할 수도 있다. **피처가 추가됐다는 사실만으로 정규화가 됐다고 말할 수 없다.**

새 5-fold 결과:

| Fold | 11 | 15 | 차이 |
|---|---:|---:|---:|
| 1 | .897932 | .901065 | +.003133 |
| 2 | .926329 | .923786 | −.002543 |
| 3 | .878528 | .879799 | +.001271 |
| 4 | .911772 | .912154 | +.000381 |
| 5 | .916603 | .914569 | −.002034 |

낮은 점수의 일부 fold는 개선되고 높은 일부 fold는 하락해 표준편차가 줄었다. 3/5 fold 개선, 평균 차이 .000042는 실질적으로 거의 동률이다. 표준편차는 fold 난도와 구성 차이도 포함하므로 순수한 bias/variance 분해가 아니다. 이번 결과는 **“holdout 하락이 곧 일반화 실패라는 결론은 부정확했다”**는 점을 뒷받침한다.

OOF AUC는 모든 OOF 예측을 한 번에 모아 구한다. fold별 AUC 평균과는 계산하는 양성·음성 쌍이 다르므로 두 지표의 개선 방향이 달라도 오류가 아니다. Fold 간 확률 스케일 차이도 pooled OOF AUC에 영향을 줄 수 있다.

## 7. Feature별 효과와 중복 정보

아래 변화량은 **추가 당시 직전 채택 모델 대비**다. 각기 다른 기준의 변화량을 피처의 절대 중요도 순위로 비교하지 않는다. 대회 점수가 없는 피처에 Leaderboard 효과를 만들어 붙이지 않았다.

| Feature | 가설·생성법 | Val 영향 | CV Mean 영향 | Leaderboard 근거 | 최종 판단 |
|---|---|---:|---:|---|---|
| FamilySize | sibsp+parch+1, 가족 규모의 합성 표현 | 단독 분리 실험 없음 | 없음 | Base부터 포함 | 유지 |
| IsAlone | FamilySize==1 | 독립 효과 미측정 | 미측정 | Champion 포함 | 유지, 제거 실험은 후보 |
| HasCabin | 원본 cabin.notna | −.001626 | 미실행 | 별도 복합 제출만 존재 | 보류 |
| AgeMissing | 원본 age.isna | −.000122 | 미실행 | 없음 | 보류 |
| Title | 이름 호칭, fold 빈도>5는 유지 | +.002927 | −.001766 | 주 3번 LB 없음 | 현재 제외 |
| GenderClass/SexPclass | gender×pclass | +.002846 | +.002901 | 11·15에 포함, 독립 LB 없음 | 유지 |
| FamilyGroup | 1명 / 2~4명 / ≥5명 | −.000163 | 미실행 | 없음 | 보류 |
| TicketGroupSize | 학습 티켓 빈도, 미관측=1 | −.002927 | 미실행 | 별도 복합 제출 외 없음 | 현재 제외 |
| FarePerPerson | fare/학습 티켓 빈도 | −.000163 | 미실행 | 별도 복합 제출 외 없음 | 보류 |
| IsChild | 치환 age<16 | +.000244 | −.002425 | 없음 | 독립 입력 제외 |
| CabinDeck | 객실 첫 영문자, 누락 Unknown | +.001870 | −.001853 | 주 9번 LB 없음 | 보류(분산 감소) |
| GenderIsChild | gender×(age<16) | +.001382 | +.001230 | 11·15에 포함 | 유지 |
| ClassIsChild | pclass×(age<16) | +.002195 | +.000326 | 11 Private .91254 | 유지 |
| GenderIsAlone | gender×IsAlone | −.001870 | 미실행 | 없음 | 보류 |
| EmbarkedClass | embarked×pclass | −.000732 | 미실행 | 없음 | 보류 |
| TicketPrefix | 일련번호·구두점 제거, fold 빈도>5 유지 | −.004066 | 미실행 | 없음 | 현재 제외 |
| **AgeBand** | [0,16,20,35,60,∞) | −.002846 | **+.000042(신규)** | **11→15 Private +.00214** | **유지·Champion** |
| Occupation / ProfessionGroup | 호칭에서 의료·성직·군인 추출 | 후속 Base 대비 −.000569 | −.001166 | 별도 3·4 복합 모델 포함 | Champion에 추가할 근거 부족 |
| IsNoble | 귀족 호칭 표시 | 후속 Base 대비 −.000244 | −.002991 | 별도 3·4 복합 모델 포함 | 제외, 호칭 버그도 확인 |
| Deck / 별도 티켓·요금 피처 | 별도 4번은 4개 FE와 치환·모델 동시 변경 | 없음 | 복합 +.012708 | Private 변화 없음 | 개별 판단 불가 |

주 계열 Title의 Rare 경계는 `>5`, 별도 프로젝트 Title은 `>=10`이다. 이름이 같아도 같은 변수가 아니다. 주 계열 FarePerPerson도 실제 개인당 운임의 정답이 아니라 **관측된 학습 티켓 인원수로 나눈 비율 피처**다. 학습 승객 자신의 빈도까지 포함하지만 미관측 티켓은 1이므로 train/새 티켓에서 통계적 의미가 달라질 수 있다.

중복의 유형을 구분해야 한다.

1. **완전 동일 열:** seaborn 학습 예제의 `family_size`와 `FamilySize`는 같은 공식이다. 주 Base~15는 FamilySize 하나만 만든다.
2. **결정적 함수:** IsAlone은 FamilySize의 함수, AgeBand는 age의 함수, GenderClass는 gender/pclass의 조합이다. 원천 정보는 늘지 않지만 트리의 분할 방식이 달라진다. 선형모델의 다중공선성 논리를 그대로 적용해 삭제할 이유는 없다.
3. **부분 중복:** Title↔gender/age, TicketPrefix↔pclass, CabinDeck/HasCabin↔pclass/fare. 같은 정보를 일부 공유하나 동일하지 않다.
4. **희소 범주:** 희귀 호칭·갑판·접두사는 적은 표본의 결과를 외울 위험이 있다. 기준 피처와 함께 ablation이나 grouped permutation으로 추가 가치를 평가해야 한다.

Base 학습 EDA의 FamilySize–sibsp 상관 .886, FamilySize–parch .784는 합성 공식에 따른 자연스러운 결과다. 이를 이유로 세 열 중 하나를 무조건 제거하면 가족 관계의 세부 정보가 손실될 수 있다. 성별 원핫 2열은 상호 보완적이고 GenderClass도 성별·등급을 복원할 수 있지만, 어떤 표현이 CatBoost에서 유리한지는 별도 한 가지 제거 실험으로 확인해야 한다.

학습 예제에는 `WomanChild=(sex=='female') OR (age<16)`, `Sex_Pclass`, `Child_Pclass`, `who`, `class`, `embark_town`, `alone`도 있다. 이들은 현재 Champion 입력에 없다. 특히 WomanChild의 OR 표현은 4범주 GenderIsChild와 같은 피처가 아니다.

## 8. Confusion Matrix는 보조 지표로만 해석

이번 재계산의 동일 holdout, threshold=.5:

| 모델 | TN | FP | FN | TP | AUC |
|---|---:|---:|---:|---:|---:|
| Base | 129 | 14 | 18 | 68 | .900065 |
| 11 | 130 | 13 | 18 | 68 | .906489 |
| 15 | 130 | 13 | 18 | 68 | .903643 |

**11과 15의 혼동행렬이 정확히 같아도 AUC는 다르다.** 0.5의 어느 쪽에 있는지는 같지만 그 안에서 생존 확률의 순서가 달라질 수 있기 때문이다.

| OOF 출처 | TN | FP | FN | TP | 해석 |
|---|---:|---:|---:|---:|---|
| Base, 신규 seed42 | 529 | 41 | 83 | 263 | 전체 916명 OOF |
| 11, 신규 seed42 | 529 | 41 | 80 | 266 | Base보다 FN 3 감소 |
| 15, 신규 seed42 | 529 | 41 | 82 | 264 | 11보다 FN 2 증가하나 Private 최고 |
| 별도 feature_add, 저장 seed0 | 516 | 54 | 75 | 271 | fold별 모델 선택 편향 있음 |
| 별도 feature_add2, 저장 seed0 | 515 | 55 | 69 | 277 | FN 6 감소·FP 1 증가, Private 변화 없음 |

서로 다른 split과 파이프라인의 혼동행렬은 직접적인 피처 효과 비교가 아니다. Base+SMOTE 별도 실험도 FP 14→15, FN 18→19로 나빠졌지만 확률 AUC는 .000081 상승했다. **FP/FN 감소와 AUC 향상은 같은 목표가 아니다.**

## 9. Local Validation 신뢰성

같은 설계로 직접 비교 가능한 제공 LB 제출은 Base·11·15, **3개뿐**이다.

| Local Metric ↔ Private | Pearson r | Spearman ρ | 판단 |
|---|---:|---:|---|
| Holdout AUC | .445 | .500 | Base→11은 일치, 11→15는 반대 |
| CV Mean | .800 | 1.000 | 세 제출 순서는 일치하나 11/15는 거의 동률 |
| Train−Val Gap | −.155 | −.500 | Gap 감소만으로 우승 모델 선택 불가 |
| OOF AUC | .778 | .500 | Base→11 일치, 11→15 소폭 반대 |

이는 **3점에 대한 기술 통계**다. CV가 Private를 예측한다고 검증한 상관 분석이 아니다. 15번 CV는 Private 결과를 안 뒤 계산했으므로 사전 예측 성공이라고 주장해서도 안 된다. 추가로 별도 3·4 모델의 CV까지 섞으면 전처리·seed·모델선택·표준편차 정의가 달라 상관계수의 해석이 더 나빠진다.

현재 검증의 개선점:

- holdout이 하락했다는 이유로 후보 CV를 생략하지 않는다. 15번이 그 규칙의 반례다.
- Champion 15를 기준으로 같은 fold의 **쌍별 AUC 차이**를 기록한다. fold Std 자체와 차이의 불확실성을 혼동하지 않는다.
- split seed와 모델 seed를 구분한다. 여러 split에서 비교하되 좋은 seed 하나를 골라 보고하지 않는다.
- OOF를 저장하고 각 행이 정확히 한 번 검증됐는지 검사한다. 이번 audit은 이를 검사했다.
- CV를 반복 사용한 모델 선택의 낙관성은 남는다. 앞으로 공개되는 새 holdout이나 외부 평가가 없다면 완전한 독립 최종 추정이라고 표현하지 않는다.
- 최종 Private 결과를 기준으로 Champion을 보존하되, 앞으로 같은 Private 결과를 계속 보며 튜닝하면 그것도 선택 데이터가 된다.

## 10. 코드 문제: 위치·원인·수정 방법

| 우선도 | 파일·위치 | 확인한 문제 | 수정 방법 |
|---|---|---|---|
| 높음 | `titanic_15.ipynb` 셀 27 | holdout gate 때문에 후보 CV 미실행, best_cv를 후보 CV로 오인하기 쉬움 | CV를 조건 없이 실행하고 candidate_metrics와 historical_best_metrics를 분리. Champion 목록은 15의 candidate_features 4개로 고정 |
| 높음 | 별도 `1. base model.ipynb` 셀 24~28 | 학습한 데이터 자체를 평가. .953656을 validation처럼 표기 | raw 데이터부터 train/val 분리, 학습 부분에만 전처리 fit. 기존 점수는 resubstitution으로 명시 |
| 높음 | 별도 `1. base model` 셀 34~37, `2. model` 셀 16·26~29 | predict()로 AUC 계산/제출. 실제 CSV가 0/1 두 값 | AUC·제출은 predict_proba의 class 1 열, accuracy/CM만 predict 사용 |
| 높음 | 별도 `2. model` 셀 10, `3. model_feature_add` 셀 9·14, `4. model_feature_add2` 셀 11·16·18·20 | CV 전에 전체 train의 결측 통계·희귀 범주·티켓 빈도 계산 | raw fold 분리 후 해당 fold train에서 통계 fit. **test label leakage가 아니라 validation 전처리 누수** |
| 높음 | 별도 `2. model` 셀 16, `3. model_feature_add` 셀 25 | 같은 검증 fold에서 3개 모델 중 best 선택 후 그 점수 보고 | 모델별 CV를 별도 집계. 최종 알고리즘 선택의 독립 성능이 필요하면 nested CV |
| 높음 | 별도 `3. model_feature_add` 셀 33 | 전체 SMOTE train의 재평가 점수로 최종 XGB 선택 | CV에서 정한 알고리즘/파라미터를 그대로 전체 train 재학습 |
| 높음 | 별도 `2. model` 셀 11→12→24→26 | features를 만든 뒤 train/test에 FamilySize·IsAlone 추가. 학습 features는 6열, 예측 test는 8열이 되는 코드 경로 | FE 후 features/target 생성. `x_test_final=...reindex(columns=features.columns)`와 원본 집합 검사. 현재 CSV는 있으나 현 코드의 깨끗한 재실행 성공을 보증하지 못함 |
| 중간 | 별도 `4. model_feature_add2` 셀 16 | train/test 티켓 인원수를 각 데이터에서 따로 계산 | fold train ticket_counts를 저장해 val/test에 map, unknown=1. 기존 점수와 비교할 때 처리 의미가 바뀐 별도 실험으로 기록 |
| 중간 | 별도 2·3·4의 SMOTE 블록 | 범주 코드를 일반 SMOTE로 보간 후 반올림, 무스케일 거리. 0/1·파생식의 논리적 관계도 깨질 수 있음 | 일반 SMOTE 제거를 별도 실험하거나 mixed-type SMOTENC 검토. 파생 피처 재계산과 거리 설정도 각각 검증 |
| 중간 | `titanic_base_smote_experiment.ipynb` 셀 40·41 | final_prep, X_test_raw, X_full_model, submission_model, submission 등이 해당 notebook에서 정의되지 않은 제출 블록 | 전체 train 전처리→SMOTE→새 모델 학습과 템플릿 로딩을 명시하거나 미완성 제출 셀 제거. 현재 `titanic_result_16.csv`는 없음 |
| 중간 | 같은 SMOTE notebook 셀 22 | `get_all_params()` 동일 여부 출력이 **False**. 같은 명시 설정이어도 표본 수 변경으로 실효 설정이 달라짐 | 자동 learning_rate 등의 차이를 기록. SMOTE만의 효과를 분리하려면 기준 모델 실효 학습률을 후보에도 적용 |
| 중간 | `titanic_11.ipynb` 셀 25·27 | 원본 train에 없는 파생 피처를 존재 여부로 거르므로 새 피처 상관 분석이 빠짐. `sex`는 실제 `gender`와도 불일치. 원본 train target 전체를 이용한 EDA | `X_tr_features`를 복사해 학습 부분 target만 붙이고 gender를 사용. 요구한 열이 있는지 assert해 조용한 누락 방지 |
| 중간 | `titanic_1.ipynb` 셀 2·33 | 읽기/쓰기 루트 경로와 현재 csv/submission 폴더 불일치 | csv/train.csv 등과 submission/submission_result_1.csv로 경로 통일. 기존 파일 덮어쓰기 방지 |
| 중간 | `common/title_feature_experiments.py` 12행 | `Countess`와 실제 추출값 `the Countess` 불일치 | 실제 문자열을 목록에 넣거나 extract 단계에서 일관 정규화. 해당 16번 결과는 버그가 포함된 정의의 결과 |
| 중간 | 별도 `common/modeling.py` 88행 | validation 길이 검사에서 다시 학습 길이를 검사 | `assert len(x_te)==len(y_te)` 및 column 이름·순서 확인 |
| 중간 | 같은 파일 149행 | bare except가 학습 실패를 숨기고 class 이름도 잘못 출력할 수 있음 | `except Exception as exc: raise RuntimeError(...) from exc`로 원인을 보존 |
| 낮음 | `titanic_15.ipynb` 셀 15·34 및 결과 설명 | 실제 interaction 모듈 대신 다른 모듈명을 적고 CabinDeck이라고 잘못 적음; 11번을 최종 Best라고 서술 | 역사적 local-best와 현재 Private champion을 구분해 별도 기록. AgeBand 설명으로 정정 |
| 낮음 | `titanic_2.ipynb` 셀 36 | “덮어쓰지 않는다” 설명과 달리 to_csv 전 exists 검사 없음 | 경로 exists 검사 또는 실행별 고유 폴더 |

SMOTENC는 수치형과 범주형이 섞인 데이터의 oversampling을 지원한다. 단, 파생 피처의 논리적 일관성까지 자동으로 보장하는 것은 아니다. [imbalanced-learn oversampling 문서](https://imbalanced-learn.org/stable/over_sampling.html)

해당 위치 바로가기: [주 전처리](C:/dev/study/machine_learning/common/preprocessing.py:8), [상호작용](C:/dev/study/machine_learning/common/interaction_experiments.py:28), [모델 설정](C:/dev/study/machine_learning/common/feature_experiments.py:63), [귀족 호칭](C:/dev/study/machine_learning/common/title_feature_experiments.py:12), [별도 Modeling 검사](<C:/dev/project/kaggle_skn35 v1/common/modeling.py:88>), [별도 예외 처리](<C:/dev/project/kaggle_skn35 v1/common/modeling.py:149>).

### 요청한 점검 항목 중 발견하지 않은 문제와 적용 범위

- **Base~15:** validation/test에 SMOTE 적용 없음, scaling 자체 없음, CV 밖 encoder fit 없음, test 통계로 결측치 fit 없음, target를 모델 입력 피처로 사용하는 코드 없음.
- 기본 EDA에서 target를 붙인 표는 학습 입력과 별도 객체다. target에 대한 EDA 자체를 target encoding leakage로 부르지 않는다. 다만 11번의 추가 heatmap은 전체 train의 target도 보므로, 이를 보고 피처를 선택하면 validation 정보의 간접 사용이 된다. 반복 피처 선택에 의한 적응적 편향도 별개다.
- **별도 SMOTE 계열:** SMOTE는 fold train 안에 있으며 validation/test를 재표본화하지 않는다. CV 밖 SMOTE 오류가 아니라 **그 앞의 전처리가 CV 밖인 오류**다.
- 별도 `4`의 test 티켓 빈도 사용은 test label 누수와 다르다. test batch에 의존하는 전처리이므로 엄격한 train-fit/transform 원칙과 차이가 있고 train/test 행 수 차이의 영향을 받는다.
- 실제 제출 **19개 모두** 393행, `passengerid,survived`, ID 고유·순서 일치, NaN/무한대 없음, 값 0~1이다. 핵심 5개는 연속 확률이다. 초기 별도 Base·2만 hard label이다.
- 초기 파일명 `(95.36)`이나 test_score 필드는 Private/Val 점수를 뜻하지 않는다.
- 주 계열은 입력 원본의 열·순서와 변환 후 열 일치를 검사한다. 별도 2번에는 위의 불일치 코드가 있고, 별도 3·4는 현재 CSV ID 정렬은 맞지만 저장 시 ID 집합 검사 강화가 필요하다.
- 별도 3·4의 OOF 대입 `oof_proba[valid_idx]`는 `.iloc` 위치와 대응해 구조상 맞다. 3번은 모델 선택 편향, 두 파일은 CV 전처리 누수가 남아 있어 “완전한 독립 OOF”라고 부를 수 없다.
- 학습 예제의 `alive`는 실제 모델 입력 전에 제거한다. target 직접 복사 열이 Champion에 포함된 흔적은 없다.

## 11. Champion 15 기준 다음 실험 8개와 우선순위

현재 폴더에 이미 16/17이 있으므로 새 제안은 **N16~N23**으로 구분한다. 기존 16/17 파일을 덮어쓰지 않는다. **모든 제안의 출발점은 B+G+C+D+A인 15번**이다. 하나를 채택하기 전 다른 변경과 누적하지 않는다.

공통 검증: holdout은 참고만 하고, split seed `[42,137,2026]`의 동일 5-fold에서 Champion과 후보를 쌍으로 비교한다. 모델 seed는 42로 고정(모델 seed 실험만 예외). 매 fold마다 새 전처리 fit. CV Mean/Std(ddof=1), OOF AUC, fold별 Δ, Train−Val Gap을 저장한다. 반복 fold를 서로 독립된 표본 15개로 간주해 유의확률을 과장하지 않는다.

공통 판단 기준은 사전에 고정한다. 예를 들어 **평균 CV와 seed별 OOF가 함께 개선되고 최소 2/3 split seed에서 같은 방향, CV Std의 큰 악화 없음**이면 후보로 유지한다. .0001 수준의 개선만 있거나 지표가 엇갈리면 보류한다. 이 기준도 운영 규칙이지 우승 보증이 아니다. Private 정답은 다음 모델 튜닝 입력으로 사용하지 않는다.

| 우선순위 | 실험 | 핵심 변경 하나 | 성공 가능성 | 과적합 위험 | 구현 난이도 | 기대 효과 크기 |
|---|---|---|---|---|---|---|
| 1 | N16 | depth 6→5 | 중 | 하 | 하 | 중 |
| 2 | N17 | 동일 15 모델의 고정 5개 seed 확률 평균 | 중 | 하 | 중 | 중 |
| 3 | N18 | l2_leaf_reg 3→5 | 중 | 하 | 중 | 중 |
| 4 | N19 | IsAlone 입력 열만 제거 | 중 | 하 | 하 | 하 |
| 5 | N20 | 원본 AgeMissing 한 열 추가 | 중 | 중 | 하 | 하 |
| 6 | N21 | age 치환 그룹을 gender×pclass→Title×pclass로 교체 | 중 | 중 | 중 | 중 |
| 7 | N22 | iterations 1000→800 | 중 | 하 | 중 | 하 |
| 8 | N23 | raw age만 제거하고 AgeBand/child 상호작용 유지 | 하 | 하 | 하 | 중 |

성공 가능성 등급은 검증된 확률이 아니라 실행 전 우선순위 판단이다. 기존 자료만으로 “성공 가능성 상”이라고 보장할 후보는 없다. 특히 N16 코드의 **한 fold 실행 확인**에서는 Champion .901065, depth5 .896053으로 낮았다. 이 한 fold로 채택하거나 전체 실패를 확정하지 않는다. 완전한 반복 CV 전 N16은 미채택이다.

### N16 — 트리 깊이를 한 단계 낮춰 일반화를 점검

**가설:** 33개 입력의 세부 상호작용을 depth6로 학습하는 것이 작은 표본의 우연까지 반영할 수 있다.

**변경:** depth만 6→5. AgeBand 및 11번에서 채택한 피처 전부 유지.

**검증 이유:** 15의 holdout Gap .061656이 남아 있고, 복잡도를 한 단계 줄이는 명확한 대조 실험이 가능하다.

**기대 효과:** 과적합 완화. 반대로 필요한 상호작용을 못 배워 underfitting이 될 수 있다.

**판단:** 공통 CV/OOF 기준. train 점수나 Gap이 낮아지는 것만으로 채택하지 않는다. 아래 실행 코드를 제공한다.

### N17 — 모델 seed 평균으로 학습 변동 완화

**가설:** 동일 피처에서도 부스팅의 무작위성이 세부 순위를 바꾸므로 여러 seed의 평균이 안정적일 수 있다.

**변경:** 15의 설정을 유지한 model seeds `[42,43,44,45,46]` 확률 평균을 사용한다. 좋은 seed만 골라내지 않는다.

**검증 이유:** 11/15 예측은 비슷하며 순위의 작은 차이가 중요했다. 각 fold에서 같은 train으로 5개를 학습하고 평균 확률을 평가한다.

**기대 효과:** seed 의존 감소, 연산량 5배. 피처 변경을 동시에 하지 않는다.

**판단:** Champion 단일 seed 대비 CV/OOF와 seed 민감도. test 예측 상관만으로 효과를 단정하지 않는다.

### N18 — L2 정규화 소폭 강화

**가설:** leaf 값의 큰 조정이 일부 희소 그룹을 과하게 반영할 수 있다.

**변경:** l2_leaf_reg만 3→5.

**검증 이유:** 학습 세부값의 변동을 줄이면서 기존 interaction을 유지할 수 있다.

**기대 효과:** 과도한 leaf 값 완화. 너무 강하면 순위 구별력 손실.

**판단:** 공통 기준. **주의: l2_leaf_reg를 명시하면 CatBoost의 자동 learning_rate 결정 조건도 바뀔 수 있다.** 각 fold의 Champion을 먼저 fit한 뒤 그 실효 learning_rate를 후보에 명시해 학습률을 동일하게 유지한다. 전체 재학습에서도 같은 방식으로 Champion의 full learning_rate를 사용한다.

### N19 — IsAlone의 추가 표현 필요성 검증

**가설:** FamilySize가 있어 IsAlone의 이진 표현은 분할 경쟁만 늘릴 수 있다.

**변경:** 생성 자체는 유지하고 최종 모델 입력에서 IsAlone 한 열만 제거한다. 나머지 원본·interaction 유지.

**검증 이유:** 정확한 결정적 함수의 중복을 근거로 한 ablation이다. “상관이 높으니 무조건 삭제”가 아니다.

**기대 효과:** 표현 단순화 또는 정보 접근성 손실. 효과는 작을 가능성.

**판단:** CV/OOF가 동률이면 단순성의 이점은 있지만, Private 우승 원본 제출은 계속 보존한다.

### N20 — AgeBand와 함께 결측 표시의 가치 재검증

**가설:** 치환된 나이와 관측된 나이가 같은 AgeBand로 들어가도 신뢰도가 다르다.

**변경:** raw age.isna()를 AgeMissing 한 열로 추가. 결측치 값·AgeBand 경계는 유지.

**검증 이유:** 기존 실험 2는 B 기준이고 CV도 없었다. 15 기준의 조건부 효과는 미검증이다.

**기대 효과:** 나이 치환의 불확실성을 구분. 결측 패턴에 대한 과적합 가능성도 있다.

**판단:** 전체 CV/OOF를 우선하고 결측 승객 하위집단은 보조 진단만 사용한다.

### N21 — 호칭 기반 나이 치환

**가설:** Master/Miss/Mr 등의 호칭이 gender만 사용하는 것보다 나이 대체를 개선한다.

**변경:** 나이 치환 규칙 하나만 Title×pclass→Title→전체 중앙값으로 교체한다. Title은 중간 변수이며 최종 입력에 추가하지 않는다.

**검증 이유:** 별도 4번에서 여러 변화와 섞여 있던 효과를 15에서 분리한다.

**기대 효과:** child/ageband 분류 오류 감소. 희귀 호칭의 불안정성 위험.

**판단:** 각 fold train 통계만 사용. 공통 기준과 치환된 연령대 변화 기록. target 기반 호칭 규칙 금지.

### N22 — 학습 회차 축소

**가설:** 후반 tree가 작은 패턴을 추가 학습하며 순위 noise를 늘릴 수 있다.

**변경:** iterations 1000→800.

**검증 이유:** 모델 복잡도를 다른 방식으로 조절한다. N16과 동시에 변경하지 않는다.

**기대 효과:** 학습시간 감소와 일반화 개선 가능성.

**판단:** iteration 변경도 자동 learning_rate를 바꿀 수 있어 fold별 Champion의 실효 learning_rate를 고정해서 후보를 학습한다. early stopping을 동시에 넣지 않는다.

### N23 — 원시 나이의 세밀함이 필요한지 검증

**가설:** raw age의 세부 경계가 noise이고 AgeBand와 child interaction만으로 충분할 수 있다.

**변경:** 전처리와 AgeBand 생성은 그대로 수행한 뒤 모델 입력의 age 한 열만 제거한다.

**검증 이유:** AgeBand의 상대적 역할을 확인하는 명확한 ablation. 현재 raw age의 유효성이 부정된 것은 아니다.

**기대 효과:** 세부 나이 noise 감소 또는 중요한 순위 정보 손실. 그래서 후순위다.

**판단:** CV/OOF 동시 개선이 없으면 제거하지 않는다.

별도의 **검증 설계 진단**으로 티켓을 그룹으로 묶은 StratifiedGroupKFold를 권한다. 이는 모델 개선 실험 번호와 분리한다. Test에는 train과 공유 티켓도 있으므로 그룹 CV는 현재 대회보다 더 어려운 “새 그룹” 문제일 수 있으며, 이것만으로 Champion을 교체하지 않는다.

## 12. 지금 실행할 다음 실험 코드

추천은 **15번을 고정 기준으로 depth만 5로 바꾼 후보를 반복 CV로 비교하는 N16**이다. 현재 15의 성능을 개선했다고 주장하는 코드가 아니라 검증할 가설의 구현이다.

원본 `titanic_15.ipynb`를 복사해서 다음처럼 바꾸면 된다.

1. 셀 3의 기준 목록은 과거 `best_features`가 아니라 아래 4개로 고정한다.
2. 셀 25의 후보 학습만 depth5 함수로 교체한다. 원본 Champion은 기존 fit_cat 사용.
3. 셀 27의 `if holdout_delta > 0:` gate를 제거한다. **기존 cv_compare는 모든 모델을 동일 fit_cat로 학습하므로 depth 비교에 그대로 쓰면 안 된다.** 아래 별도 비교 루프를 사용한다.
4. 셀 29는 피처를 그대로 유지하고 선택한 depth5 후보를 전체 train에서 새로 학습한다. 제출명에 Champion15 기준과 depth5를 명시한다.

```python
from catboost import CatBoostClassifier
from common.interaction_experiments import FeaturePreprocessor
from common.feature_experiments import baseline_parameters, fit_cat

champion_features = [
    'GenderClass', 'GenderIsChild', 'ClassIsChild', 'AgeBand'
]

def fit_depth5(X_encoded, y):
    model = CatBoostClassifier(**(baseline_parameters() | {'depth': 5}))
    model.fit(X_encoded, y.astype('int8'))
    return model

# 각 fold에서 raw train/valid 분리 후:
prep = FeaturePreprocessor(champion_features)
encoded_train = prep.fit_transform(X_train_raw)
encoded_valid = prep.transform(X_valid_raw)
champion = fit_cat(encoded_train, y_train)
candidate = fit_depth5(encoded_train, y_train)

# depth가 만드는 max_leaves 차이 외에 실효 설정이 바뀌지 않았는지 검사
p0, p1 = champion.get_all_params(), candidate.get_all_params()
changed = {k for k in set(p0) | set(p1) if p0.get(k) != p1.get(k)}
assert changed == {'depth', 'max_leaves'}

p_champion = champion.predict_proba(encoded_valid)[:, list(champion.classes_).index(1)]
p_candidate = candidate.predict_proba(encoded_valid)[:, list(candidate.classes_).index(1)]
```

독립 실행 가능한 전체 코드는 [next_champion_depth5.py](C:/dev/study/machine_learning/titanic_audit/next_champion_depth5.py)에 작성했다. 실제 한 fold 실행으로 입력 열 일치·학습·확률 평가·실효 파라미터 변경 검사를 통과했다. **3×5-fold 전체 실험은 아직 실행하지 않았다.** 한 fold의 하락 결과를 숨기지 않고 위 우선순위 표에 명시했다.

PowerShell 실행 명령:

```powershell
& 'C:\dev\project\kaggle_skn35 v1\.venv\Scripts\python.exe' 'C:\dev\study\machine_learning\titanic_audit\next_champion_depth5.py'
```

실행하면 별도 `titanic_audit/depth5_results` 폴더에 fold 지표, seed별 CV/OOF 요약, OOF 확률, 후보 제출 파일과 실효 파라미터를 저장한다. 기존 제출은 덮어쓰지 않는다. 결과 폴더가 이미 있으면 중단한다. Kaggle 업로드나 Champion 자동 교체는 하지 않는다.

## 현재까지 실험에서 알게 된 것

1. **현재 최종 1등은 15번**이며 Public 점수로 순위를 뒤집어 해석하면 안 된다.
2. 15번은 11번+AgeBand이고 33열이다. 11번의 28열 best_final_model과 다르다.
3. Base·11·15는 기존 CSV와 수치적으로 재현돼 코드 연결이 확인됐다.
4. 15번의 기존 CV는 없었다. 새 CV는 11번과 거의 동률이며 fold Std가 작다.
5. GenderClass와 child 상호작용의 누적 추가는 holdout/CV 개선 근거가 있다.
6. holdout 한 번으로 후보 CV를 막는 규칙이 Private 우승 후보를 제외했다.
7. 11/15의 holdout 혼동행렬은 같아도 AUC는 다르다.
8. 별도 프로젝트 계열에는 class label AUC, CV 전처리 누수, 학습셋 모델선택 문제가 있어 직접 비교를 제한한다.
9. 소수 Leaderboard 결과로 피처별 인과효과와 통계적 유의성을 확정할 수 없다.

## 성능이 좋아진 Feature

**GenderClass, GenderIsChild, ClassIsChild**는 당시 직전 기준 대비 holdout/CV가 함께 개선됐다. **AgeBand**는 11 대비 실제 Private +.00214로 현재 Champion에 유지한다. 새 CV 평균의 차이는 거의 없으므로 일반적 우월성은 제한적으로 해석한다.

## 성능이 나빠진 Feature

Title과 독립 IsChild는 CV 평균이 낮아졌다. TicketGroupSize와 TicketPrefix는 train 상승·holdout 하락으로 현재 제외 근거가 있다. 다만 후보 CV가 없는 경우 영구적인 열등성으로 단정하지 않는다.

## 아직 판단하기 어려운 Feature

HasCabin, AgeMissing, FamilyGroup, FarePerPerson, CabinDeck, GenderIsAlone, EmbarkedClass. FamilySize/IsAlone도 Champion에 들어 있지만 독립 기여를 분리한 실험은 없다. 별도 4번에서 한꺼번에 바뀐 객실·티켓·치환 효과는 개별 판단 불가다.

## 현재 Champion Model

**`titanic_result_15.csv`**

**Private AUC: 0.91468 / Public: 0.89121 / 사용자 제공 최종 순위: 1등**

CatBoost, 33열, AgeBand 포함, SMOTE·Scaling 없음, random_seed=42, 전체 916명 재학습 후 생존 확률 제출.

## Public과 Private 차이에 대한 결론

AgeBand 추가로 일부 승객의 확률 순위가 바뀌었고, 그 변화가 이번 Private에서는 유리하게 평가됐다. 작은 평가 표본과 구성 차이가 가능한 설명이다. **Private 구성이나 정답이 없으므로 어느 집단이 원인인지 확정할 수 없다.** Public 하락은 Private 우승을 부정하지 않고, Private 우승도 모든 로컬·미래 데이터에서 우수함을 증명하지 않는다.

## 현재 코드에서 가장 먼저 고쳐야 할 문제

현재 실험 흐름에서는 **holdout 하락 시 CV를 건너뛰는 gate와 11번 local-best를 Champion으로 취급하는 기록 방식**부터 바로잡아야 한다. 과거 별도 계열을 재사용할 경우에는 hard label AUC와 CV 밖 전처리부터 수정한다.

## 다음 실험 TOP 5

1. 15번 유지, depth 6→5 반복 CV 대조.
2. 15번 유지, 고정 5개 model seed의 확률 평균.
3. 15번 유지, 실효 learning_rate를 맞춘 L2 3→5.
4. 15번에서 IsAlone 한 열만 제거하는 ablation.
5. 15번에 AgeMissing 한 열만 추가.

## 내가 지금 바로 진행해야 할 다음 실험

[N16 실행 코드](C:/dev/study/machine_learning/titanic_audit/next_champion_depth5.py)로 Champion15와 depth5를 동일 반복 CV에서 비교한다. 피처·전처리·seed를 함께 바꾸지 않는다. 단일 fold 실행에서는 depth5가 낮았으므로 **전체 결과 전에는 교체하지 않고 `titanic_result_15.csv`를 Champion으로 유지한다.**
