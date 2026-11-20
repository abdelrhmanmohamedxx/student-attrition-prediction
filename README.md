# Student Attrition Prediction

Predicting which first-year university students will **not return for their second year**, so student-success teams can reach them before they leave.

The project compares eight classification models with a leak-free scikit-learn pipeline. It evaluates them with metrics suited to an imbalanced outcome and turns the best model into risk tiers that advisors can act on.

## Results at a glance

| | Value |
|---|---|
| Students / cohorts | 3,400 students, entry cohorts 2005–2010 |
| At-risk (did not return) rate | 21.3% |
| Best test ROC-AUC | **0.77** (Random Forest, Logistic Regression) |
| Selected model | Gradient Boosting (HistGB), best cross-validated ROC-AUC |
| At-risk students found (recall) | **54%**, with 55% precision (F1 0.55) |
| Highest-risk 20% of students | contain **53% of all leavers** (57% actual non-return rate vs 21% average) |
| New-cohort check (train 2005–09 → test 2010) | ROC-AUC 0.76 |

**Top drivers:** second-term hours attempted, second-term core GPA, major, distance from home, and first-term completion and GPA. Financial-aid figures and entrance-test scores add little once academic performance is known.

### Model comparison (held-out test set, 680 students)

| Model | ROC-AUC | PR-AUC | Recall (at-risk) | Precision (at-risk) | Accuracy |
|---|---|---|---|---|---|
| Random Forest | 0.771 | 0.586 | 0.455 | 0.653 | 0.832 |
| Logistic Regression | 0.771 | 0.583 | 0.607 | 0.463 | 0.766 |
| Gradient Boosting (HistGB) | 0.758 | 0.577 | 0.579 | 0.519 | 0.796 |
| Extra Trees | 0.758 | 0.579 | 0.490 | 0.592 | 0.819 |
| Decision Tree | 0.742 | 0.511 | 0.510 | 0.481 | 0.778 |
| SVM (RBF) | 0.738 | 0.482 | 0.372 | 0.635 | 0.821 |
| KNN | 0.702 | 0.551 | 0.290 | 0.875 | 0.840 |
| Baseline (always "returns") | 0.500 | 0.213 | 0.000 | – | 0.787 |

Recall and precision use the default 0.5 cutoff. The notebook also tunes the cutoff on training data only. Note that the do-nothing baseline already scores 79% accuracy, which is why accuracy is not used to pick models.

## Methodology

1. **EDA:** target balance, missing-value patterns, at-risk rates by student group, academic performance vs outcome.
2. **Feature engineering (row-level only):**
   - letter grades mapped to points (A=4 … F=0)
   - per-term core GPA, course counts, F counts, incomplete counts across all six course slots
   - earned ÷ attempted completion ratios
   - a `NO_SECOND_TERM` flag
   - entry cohort year
3. **Stratified 80/20 train/test split.** The test set is used only for final evaluation.
4. **Preprocessing inside a `Pipeline`:**
   - numeric columns: median imputation with missing-value indicators, then standard scaling
   - categorical columns: one-hot encoding
5. **Class imbalance:** `class_weight='balanced'`. An optional cell runs SMOTE + XGBoost correctly, inside an `imblearn` pipeline.
6. **Tuning:** `GridSearchCV` with 5-fold stratified CV, scored on ROC-AUC.
7. **Evaluation:** ROC-AUC, PR-AUC, recall/precision/F1 for at-risk students, confusion matrix, ROC and PR curves.
8. **Decision threshold** chosen from out-of-fold training predictions. **Risk tiers** show how leavers concentrate by predicted risk.
9. **Explainability:** permutation importance on raw features, plus logistic-regression coefficients for direction of effect.
10. **Robustness checks:**
    - temporal validation on the unseen 2010 cohort
    - an early-warning model that uses first-term data only

### Improvements over the first version

| Issue in v1 | Fix in this version |
|---|---|
| Imputer, scaler and feature selector fitted on the full dataset before the split (test-set leakage) | All learned preprocessing lives in a pipeline fitted on training folds only |
| SMOTE applied before cross-validation, inflating CV scores (0.93 CV vs 0.82 test) | Class weights instead; optional SMOTE runs inside an `imblearn` pipeline |
| Models ranked by accuracy, although "everyone returns" scores ~80% | Ranked by ROC-AUC / PR-AUC; recall and precision reported for at-risk students |
| Missing second-term data filled with the mean, hiding the strongest signal (13% of those students returned vs 83% overall) | Missing-value indicators and a `NO_SECOND_TERM` flag |
| `LabelEncoder` gave categories and grades a meaningless numeric order | One-hot encoding for categories; explicit grade-point mapping |
| Only core-course slots 1–2 used | All six slots summarised per term |

## Repository structure

```
├── final_project.ipynb      # full analysis with outputs
├── Student_Attrition.xlsx   # dataset (sheets: University information, Variable Dictionary)
├── requirements.txt
└── README.md
```

## How to run

**Google Colab:** open `final_project.ipynb`, upload `Student_Attrition.xlsx` with the Files panel, then choose *Runtime → Run all*.

**Locally:**

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook final_project.ipynb
```

The full notebook runs in about a minute on a laptop. The SMOTE + XGBoost comparison runs only when `imbalanced-learn` and `xgboost` are installed; otherwise it is skipped automatically.

## Dataset

The data has one row per first-year student. It includes demographics, residency, major and minor, entrance-test scores, high-school GPA, first- and second-term core-course grades, hours attempted and earned, housing, distance from home, parental education, and financial-aid figures (gross need, cost of attendance, family contribution, unmet need). The target is `RETURNED_2ND_YR`. Column definitions are in the workbook's **Variable Dictionary** sheet.

## Limitations

- **Timing:** the main model needs second-term data, so it scores students only at the end of year 1. The early-warning version (first term only) gives more lead time but is weaker (ROC-AUC 0.67).
- **Missing reasons:** about half of leavers look like returners in the data. Family, health, transfer and work reasons are not recorded.
- **Data age and scope:** the data comes from one institution and 2005–2010 cohorts, so the model should be re-validated on recent data.
- **Fairness:** demographic attributes are model inputs. Error rates should be audited by group before any deployment.
- **Correlation, not cause:** feature importance shows what predicts leaving, not what would prevent it.

## Next steps

- Pilot advisor outreach to the highest-risk tier and measure the change in return rate against a comparison group.
- Use the early-warning model after term 1 as a wider, low-cost screen.
- Add earlier signals: mid-term grades, LMS activity, attendance, advising contacts, financial-aid holds.
- Retrain yearly and monitor performance and fairness by group.

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn (optional: imbalanced-learn, XGBoost)
