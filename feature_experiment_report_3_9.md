# Titanic 3~9 Feature Engineering 실험 결과

모든 실험은 CatBoost baseline 생성 인자를 동일하게 사용했고 같은 학습 크기의 원본 baseline과 get_all_params() 전체 일치를 검증했다. 모델 튜닝·앙상블·threshold tuning은 수행하지 않았다.

| 버전 | 이번 피처 | Train AUC | Validation AUC | Gap | CV Mean | CV Std | 이전 Best 대비 | 판단 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 3 | Title | 0.965168 | 0.902992 | 0.062176 | 0.900010 | 0.016136 | +0.002927 | 제외 |
| 4 | SexPclass | 0.963047 | 0.902911 | 0.060136 | 0.904676 | 0.016132 | +0.002846 | 채택 |
| 5 | FamilyGroup | 0.964434 | 0.902748 | 0.061686 | 미실행 | 미실행 | -0.000163 | 제외 |
| 6 | TicketGroupSize | 0.964691 | 0.899984 | 0.064707 | 미실행 | 미실행 | -0.002927 | 제외 |
| 7 | FarePerPerson | 0.965835 | 0.902748 | 0.063087 | 미실행 | 미실행 | -0.000163 | 제외 |
| 8 | IsChild | 0.964218 | 0.903155 | 0.061063 | 0.902251 | 0.017383 | +0.000244 | 제외 |
| 9 | CabinDeck | 0.964011 | 0.904781 | 0.059230 | 0.902823 | 0.013704 | +0.001870 | 제외 |

## Titanic 3
학습 표본에서 Mr 393명 생존율 10.2%, Mrs 100명 89.0%, Miss 139명 82.7%, Master 36명 33.3%, Rare 19명 21.1%다. 호칭별 차이가 크지만 성별/연령 정보와 겹친다. 단변수 생존율 차이가 크다는 사실이 모델의 독립적인 추가 기여를 보장하지 않는다.

## Titanic 4
같은 학습 표본에서 1등급 여성 66명은 모두 생존, 3등급 여성 115명은 73.0% 생존이다. 1등급 남성은 22.2%, 2·3등급 남성은 약 9~10%다. 원본의 결정적 조합이지만 이번에는 holdout과 CV 모두에서 표현의 효과가 확인됐다. 100%인 표본 비율을 예측 규칙으로 하드코딩하지 않았다.

## Titanic 5
Alone 426명 생존율 30.0%, Small(2~4명) 221명 56.1%, Large(5명 이상) 40명 20.0%다. target 관계는 비선형이지만 FamilySize의 구간 표현이고 Alone은 IsAlone과 동일하다. 기존 트리에 추가했을 때 holdout 이득은 없었다.

## Titanic 6
학습 기준 티켓 인원 1명 집단은 468명/생존율 33.5%, 2명 집단은 132명/52.3%, 3명 집단은 54명/48.1%다. FamilySize와 상관은 0.560으로 완전 복제는 아니다. 그러나 학습 인원수는 실제 전체 티켓 그룹 크기가 아니며 validation 미관측 티켓은 1이 된다. 이번 holdout에서는 악화됐다.

## Titanic 7
fare와 FarePerPerson 상관은 0.856, pclass와는 -0.545, TicketGroupSize와는 -0.001이다. 높은 운임과 공유 수가 섞인 비율이지만 원본 fare와 상당 부분 겹친다. 이번에는 TicketGroupSize가 탈락해 분모 계산에만 사용했으며 입력에는 별도로 추가하지 않았다.

## Titanic 8
치환 후 16세 미만 62명 생존율 51.6%, 나머지 625명 36.5%다. age와 상관은 -0.530이다. 연령대 원본 EDA에서도 어린이 내부의 차이와 작은 표본을 확인했다. age의 결정적 threshold 표현으로 holdout은 소폭 상승했으나 이전 Best보다 CV 평균이 낮았다.

## Titanic 9
Unknown 543명 생존율 32.8%, B 31명 64.5%, E 19명 63.2%, C 50명 52.0%다. A/B/C는 학습 표본에서 모두 1등급이고, Unknown의 355명은 3등급이다. Deck와 등급은 많이 겹치지만 동일하지는 않다. 희귀 Deck의 표본이 작아 극단적 생존율을 일반화하지 않는다. holdout 최고지만 이전 Best보다 CV 평균이 낮았다.

## 최종 Best
추가 채택은 SexPclass 하나다. gender/pclass 조합을 6개 원핫 열로 인코딩하여 baseline 12개 + 6개 = 18개 입력이다.
최종 Best holdout Validation AUC 0.902911, CV Mean 0.904676, CV Std 0.016132.
원본 baseline 대비 holdout +0.002846, CV 평균 +0.002901. 통계적으로 확실한 우월성을 증명한 것은 아니며 반복된 모델 선택의 낙관성이 남는다.
9차 notebook의 best_final_model은 채택된 피처만 사용한다. 9차 CSV는 CabinDeck을 넣은 후보 실험 기록이다. 채택 기준에 맞는 제출을 고르면 4차 CSV다.
3차 Title은 원본 baseline보다 CV 평균이 낮아 제외, 5/6/7차는 이전 Best 대비 holdout 개선 없음, 8/9차는 이전 Best 대비 CV 평균 개선 없음으로 제외했다.
HasCabin, AgeMissing은 이번 어느 후보에도 포함하지 않았다.

## 파일 및 검증
- csv/train.csv, csv/test.csv, csv/submission.csv만 사용.
- titanic_3~9.ipynb의 모든 코드 셀 실행 완료.
- submission/titanic_result_3~9.csv는 각각 실제 후보를 전체 train에 재학습한 생존 확률이다.
- 각 CSV (393, 2), 템플릿 컬럼/ID/순서 유지, NaN 없음, float64 확률 0~1, index=False.
- 기존 노트북, 입력 CSV, 제출 파일, 공통 모듈은 해시를 대조하여 보존을 확인.
- 신규 common/feature_experiments.py만 사용하며 기존 모듈의 전처리/seed/AUC/모델 설정을 재사용.
- 기본 learning_rate는 baseline에서도 자동 결정된다. 이를 수동으로 바꾸지 않았으며 holdout끼리, 동일 fold끼리, 전체 train끼리 실효 파라미터 전체가 같음을 확인했다. 서로 학습 행 수가 다른 단계의 자동값이 다른 것은 baseline 자체의 동작이다.
