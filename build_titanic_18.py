from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from common.utils import train_test_split_by_target
from common.title_feature_experiments import (
    TitleFeaturePreprocessor,
    fit_feature_model,
    model_metrics,
    five_fold_compare,
)
from common.feature_experiments import fit_cat

ROOT = Path.cwd()
NOTEBOOK_PATH = ROOT / "titanic_18.ipynb"

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Titanic Experiment 18\n",
                "\n",
                "가설:\n",
                "fare 원본값을 유지하면서 LogFare를 추가하면\n",
                "Fare의 원본 정보와 로그 변환 정보를 동시에 활용할 수 있어\n",
                "AUC와 일반화 성능이 개선될 수 있다.\n",
                "\n",
                "변경사항:\n",
                "+ LogFare\n",
                "fare 유지\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n",
                "\n",
                "from common.utils import train_test_split_by_target\n",
                "from common.title_feature_experiments import fit_feature_model, model_metrics, five_fold_compare\n",
                "from common.feature_experiments import fit_cat\n",
                "\n",
                "ROOT = Path.cwd()\n",
                "TRAIN_PATH = ROOT / \"csv\" / \"train.csv\"\n",
                "TEST_PATH = ROOT / \"csv\" / \"test.csv\"\n",
                "SUBMISSION_PATH = ROOT / \"csv\" / \"submission.csv\"\n",
                "train = pd.read_csv(TRAIN_PATH)\n",
                "test = pd.read_csv(TEST_PATH)\n",
                "submission = pd.read_csv(SUBMISSION_PATH)\n",
                "\n",
                "target_col = next(col for col in train.columns if col not in test.columns)\n",
                "id_col = submission.columns[0]\n",
                "feature_name = \"LogFare\"\n",
                "print(\"train:\", train.shape, \"test:\", test.shape, \"submission:\", submission.shape)\n",
                "print(\"target:\", target_col, \"ID:\", id_col, \"candidate:\", feature_name)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train[\"LogFare\"] = np.log1p(train[\"fare\"])\n",
                "test[\"LogFare\"] = np.log1p(test[\"fare\"])\n",
                "\n",
                "assert \"fare\" in train.columns and \"fare\" in test.columns\n",
                "assert \"LogFare\" in train.columns and \"LogFare\" in test.columns\n",
                "print(\"LogFare 생성 완료\")\n",
                "print(\"train LogFare missing count:\", train[\"LogFare\"].isna().sum())\n",
                "print(\"test LogFare missing count:\", test[\"LogFare\"].isna().sum())\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X = train.drop(columns=[target_col, id_col])\n",
                "y = train[target_col]\n",
                "X_test_raw = test.drop(columns=id_col)\n",
                "train_part, valid_part = train_test_split_by_target(train, target_name=target_col)\n",
                "X_tr_raw = train_part.drop(columns=[target_col, id_col])\n",
                "y_tr = train_part[target_col]\n",
                "X_valid_raw = valid_part.drop(columns=[target_col, id_col])\n",
                "y_valid = valid_part[target_col]\n",
                "\n",
                "assert set(y.unique()) == {0, 1}\n",
                "assert X.columns.equals(X_test_raw.columns)\n",
                "assert \"LogFare\" in X_tr_raw.columns and \"LogFare\" in X_valid_raw.columns\n",
                "assert len(train_part) == 687 and len(valid_part) == 229\n",
                "print(\"train split:\", X_tr_raw.shape, \"validation split:\", X_valid_raw.shape)\n",
                "print(y_tr.value_counts(normalize=True).sort_index().to_string())\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Base와 LogFare 후보 모델 비교\n",
                "\n",
                "기존 Feature와 모델 설정은 그대로 유지하고, `fare`는 유지한 채 `LogFare`만 추가합니다.\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "_, base_model, X_base_train, X_base_valid = fit_feature_model(X_tr_raw, y_tr, X_valid_raw, None)\n",
                "_, candidate_model, X_candidate_train, X_candidate_valid = fit_feature_model(X_tr_raw, y_tr, X_valid_raw, feature_name)\n",
                "\n",
                "assert base_model.get_all_params() == candidate_model.get_all_params()\n",
                "\n",
                "base_result = model_metrics(base_model, X_base_train, y_tr, X_base_valid, y_valid)\n",
                "candidate_result = model_metrics(candidate_model, X_candidate_train, y_tr, X_candidate_valid, y_valid)\n",
                "\n",
                "print(\"Base Train AUC:\", round(base_result[\"train_auc\"], 6))\n",
                "print(\"Base Validation AUC:\", round(base_result[\"validation_auc\"], 6))\n",
                "print(\"Candidate Train AUC:\", round(candidate_result[\"train_auc\"], 6))\n",
                "print(\"Candidate Validation AUC:\", round(candidate_result[\"validation_auc\"], 6))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def classification_metrics(result):\n",
                "    prediction = result[\"valid_prediction\"]\n",
                "    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()\n",
                "    return {\n",
                "        \"accuracy\": accuracy_score(y_valid, prediction),\n",
                "        \"precision\": precision_score(y_valid, prediction, zero_division=0),\n",
                "        \"recall\": recall_score(y_valid, prediction, zero_division=0),\n",
                "        \"f1\": f1_score(y_valid, prediction, zero_division=0),\n",
                "        \"tn\": int(tn), \"fp\": int(fp), \"fn\": int(fn), \"tp\": int(tp),\n",
                "    }\n",
                "\n",
                "base_class = classification_metrics(base_result)\n",
                "candidate_class = classification_metrics(candidate_result)\n",
                "print(\"Base metrics:\", base_class)\n",
                "print(\"Candidate metrics:\", candidate_class)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. 5-Fold CV (기존 구조 유지)\n",
                "\n",
                "`LogFare`만 추가한 후보와 기존 Base를 같은 fold에서 공정하게 비교한다.\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fold_table, cv_summary = five_fold_compare(X, y, feature_name)\n",
                "fold_pivot = fold_table.pivot(index=\"fold\", columns=\"model\", values=\"auc\")\n",
                "fold_pivot[\"Difference\"] = fold_pivot[\"Candidate\"] - fold_pivot[\"Base\"]\n",
                "print(\"\\n[5-Fold AUC]\")\n",
                "print(fold_pivot.to_string(float_format=lambda value: f\"{value:.6f}\"))\n",
                "print(\"\\n[CV 요약]\")\n",
                "print(cv_summary.to_string(float_format=lambda value: f\"{value:.6f}\"))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "metrics = [\"Train AUC\", \"Validation AUC\", \"Gap\", \"CV Mean\", \"CV Std\"]\n",
                "base_values = [\n",
                "    base_result[\"train_auc\"], base_result[\"validation_auc\"], base_result[\"gap\"],\n",
                "    cv_summary.loc[\"Base\", \"mean\"], cv_summary.loc[\"Base\", \"std\"],\n",
                "]\n",
                "candidate_values = [\n",
                "    candidate_result[\"train_auc\"], candidate_result[\"validation_auc\"], candidate_result[\"gap\"],\n",
                "    cv_summary.loc[\"Candidate\", \"mean\"], cv_summary.loc[\"Candidate\", \"std\"],\n",
                "]\n",
                "comparison = pd.DataFrame({\"Metric\": metrics, \"Base\": base_values, \"Candidate\": candidate_values})\n",
                "comparison[\"Difference\"] = comparison[\"Candidate\"] - comparison[\"Base\"]\n",
                "print(comparison.to_string(index=False, float_format=lambda value: f\"{value:.6f}\"))\n",
                "\n",
                "with (ROOT / \"results\" / \"titanic_16_summary.json\").open(\"r\", encoding=\"utf-8\") as f:\n",
                "    previous_best = json.load(f)\n",
                "\n",
                "prev_val = previous_best[\"champion_holdout\"][\"validation_auc\"]\n",
                "prev_cv = previous_best[\"champion_cv_mean\"]\n",
                "print(f\"\\nPrevious Best Validation AUC: {prev_val:.6f}\")\n",
                "print(f\"Titanic 18 Validation AUC: {candidate_result['validation_auc']:.6f}\")\n",
                "print(f\"Difference: {candidate_result['validation_auc'] - prev_val:+.6f}\")\n",
                "print(f\"\\nPrevious Best CV Mean: {prev_cv:.6f}\")\n",
                "print(f\"Titanic 18 CV Mean: {cv_summary.loc['Candidate', 'mean']:.6f}\")\n",
                "print(f\"Difference: {cv_summary.loc['Candidate', 'mean'] - prev_cv:+.6f}\")\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. 최종 Train/Test 재학습 및 제출 파일 생성\n",
                "\n",
                "현재 실험의 `LogFare`를 전체 train에 fit하고, test에 transform해서 Kaggle submission용 결과를 생성한다.\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from common.title_feature_experiments import TitleFeaturePreprocessor\n",
                "\n",
                "full_prep = TitleFeaturePreprocessor(feature_name)\n",
                "X_full = full_prep.fit_transform(X)\n",
                "X_test = full_prep.transform(X_test_raw)\n",
                "\n",
                "assert X_full.columns.equals(X_test.columns)\n",
                "assert np.isfinite(X_full.to_numpy()).all()\n",
                "assert np.isfinite(X_test.to_numpy()).all()\n",
                "print(\"전체 train/test 변환 검증 완료:\", X_full.shape, X_test.shape)\n",
                "\n",
                "full_model = fit_cat(X_full, y)\n",
                "positive_index = list(full_model.classes_).index(1)\n",
                "test_probability = full_model.predict_proba(X_test)[:, positive_index]\n",
                "\n",
                "result = pd.DataFrame({\n",
                "    id_col: test[id_col],\n",
                "    target_col: test_probability,\n",
                "})\n",
                "result.columns = submission.columns\n",
                "\n",
                "result_path = ROOT / \"submission\" / \"titanic_result_18.csv\"\n",
                "result.to_csv(result_path, index=False)\n",
                "print(\"생성 파일:\", result_path)\n",
                "print(\"result shape:\", result.shape)\n",
                "print(result.head().to_string(index=False))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=== 검증 체크 ===\")\n",
                "print(\"LogFare 생성 여부:\", \"LogFare\" in train.columns and \"LogFare\" in test.columns)\n",
                "print(\"fare 유지 여부:\", \"fare\" in train.columns and \"fare\" in test.columns)\n",
                "print(\"Feature column 일치 여부:\", X_full.columns.equals(X_test.columns))\n",
                "print(\"NaN/Inf 검사:\", np.isfinite(X_full.to_numpy()).all(), np.isfinite(X_test.to_numpy()).all())\n",
                "print(\"모델 학습 완료 여부:\", full_model is not None)\n",
                "print(\"제출 파일 존재 여부:\", (ROOT / \"submission\" / \"titanic_result_18.csv\").exists())\n",
                "print(\"제출 row 수와 test row 수 비교:\", len(result), len(test))\n",
                "print(\"제출 컬럼 구조:\", result.columns.tolist())\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"\\nExperiment: Titanic 18\")\n",
                "print(\"Added Feature: LogFare\")\n",
                "print(\"Removed Feature: 없음\")\n",
                "print(\"Fare: 유지\")\n",
                "print(f\"Train AUC: {candidate_result['train_auc']:.6f}\")\n",
                "print(f\"Validation AUC: {candidate_result['validation_auc']:.6f}\")\n",
                "print(f\"Gap: {candidate_result['gap']:.6f}\")\n",
                "print(f\"CV Mean: {cv_summary.loc['Candidate', 'mean']:.6f}\")\n",
                "print(f\"CV Std: {cv_summary.loc['Candidate', 'std']:.6f}\")\n",
                "\n",
                "combined_decision = \"채택\" if candidate_result['validation_auc'] > prev_val and cv_summary.loc['Candidate', 'mean'] > prev_cv else \"보류\"\n",
                "print(f\"결론: {combined_decision}\")\n",
                "print(\"생성 파일:\")\n",
                "print(\"- titanic_18.ipynb\")\n",
                "print(\"- submission/titanic_result_18.csv\")\n",
            ],
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Created notebook: {NOTEBOOK_PATH}")

# Run the experiment logic in the current workspace.
TRAIN_PATH = ROOT / "csv" / "train.csv"
TEST_PATH = ROOT / "csv" / "test.csv"
SUBMISSION_PATH = ROOT / "csv" / "submission.csv"
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SUBMISSION_PATH)

target_col = next(col for col in train.columns if col not in test.columns)
id_col = submission.columns[0]
feature_name = "LogFare"

train["LogFare"] = np.log1p(train["fare"])
test["LogFare"] = np.log1p(test["fare"])

assert "fare" in train.columns and "fare" in test.columns
assert "LogFare" in train.columns and "LogFare" in test.columns
print("raw train LogFare missing count:", train["LogFare"].isna().sum())
print("raw test LogFare missing count:", test["LogFare"].isna().sum())

X = train.drop(columns=[target_col, id_col])
y = train[target_col]
X_test_raw = test.drop(columns=id_col)
train_part, valid_part = train_test_split_by_target(train, target_name=target_col)
X_tr_raw = train_part.drop(columns=[target_col, id_col])
y_tr = train_part[target_col]
X_valid_raw = valid_part.drop(columns=[target_col, id_col])
y_valid = valid_part[target_col]

_, base_model, X_base_train, X_base_valid = fit_feature_model(X_tr_raw, y_tr, X_valid_raw, None)
_, candidate_model, X_candidate_train, X_candidate_valid = fit_feature_model(X_tr_raw, y_tr, X_valid_raw, feature_name)
base_result = model_metrics(base_model, X_base_train, y_tr, X_base_valid, y_valid)
candidate_result = model_metrics(candidate_model, X_candidate_train, y_tr, X_candidate_valid, y_valid)

fold_table, cv_summary = five_fold_compare(X, y, feature_name)

with (ROOT / "results" / "titanic_16_summary.json").open("r", encoding="utf-8") as f:
    previous_best = json.load(f)

prev_val = previous_best["champion_holdout"]["validation_auc"]
prev_cv = previous_best["champion_cv_mean"]

full_prep = TitleFeaturePreprocessor(feature_name)
X_full = full_prep.fit_transform(X)
X_test = full_prep.transform(X_test_raw)

assert X_full.columns.equals(X_test.columns)
assert np.isfinite(X_full.to_numpy()).all()
assert np.isfinite(X_test.to_numpy()).all()

full_model = fit_cat(X_full, y)
positive_index = list(full_model.classes_).index(1)
test_probability = full_model.predict_proba(X_test)[:, positive_index]
result = pd.DataFrame({id_col: test[id_col], target_col: test_probability})
result.columns = submission.columns
result_path = ROOT / "submission" / "titanic_result_18.csv"
result.to_csv(result_path, index=False)

print("\n=== Result Summary ===")
print(f"Train AUC: {candidate_result['train_auc']:.6f}")
print(f"Validation AUC: {candidate_result['validation_auc']:.6f}")
print(f"Gap: {candidate_result['gap']:.6f}")
print(f"CV Mean: {cv_summary.loc['Candidate', 'mean']:.6f}")
print(f"CV Std: {cv_summary.loc['Candidate', 'std']:.6f}")
print(f"Previous Best Validation AUC: {prev_val:.6f}")
print(f"Titanic 18 Validation AUC: {candidate_result['validation_auc']:.6f}")
print(f"Difference: {candidate_result['validation_auc'] - prev_val:+.6f}")
print(f"Previous Best CV Mean: {prev_cv:.6f}")
print(f"Titanic 18 CV Mean: {cv_summary.loc['Candidate', 'mean']:.6f}")
print(f"Difference: {cv_summary.loc['Candidate', 'mean'] - prev_cv:+.6f}")
print("\n=== Verification ===")
print("LogFare created in train/test:", "LogFare" in train.columns and "LogFare" in test.columns)
print("fare retained:", "fare" in train.columns and "fare" in test.columns)
print("Feature columns matched:", X_full.columns.equals(X_test.columns))
print("No NaN/Inf in encoded train/test:", np.isfinite(X_full.to_numpy()).all(), np.isfinite(X_test.to_numpy()).all())
print("Model fit finished:", full_model is not None)
print("Submission file exists:", result_path.exists())
print("Submission rows:", len(result), "Test rows:", len(test))
print("Submission columns:", result.columns.tolist())
print("\nGenerated files:")
print("- titanic_18.ipynb")
print("- submission/titanic_result_18.csv")
print("\nDecision:", "채택" if candidate_result["validation_auc"] > prev_val and cv_summary.loc["Candidate", "mean"] > prev_cv else "보류")
