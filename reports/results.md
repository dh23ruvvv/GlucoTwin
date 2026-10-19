# GlucoTwin model results

Synthetic data only; not clinical validation.

## Frozen evaluation protocol

- Baseline: StandardScaler + logistic regression on current glucose only.
- Main model: fixed histogram gradient boosting on CGM history, clock and baseline EHR.
- Ablation: identical boosting settings with EHR removed.
- No hyperparameter search; random internal early stopping is disabled.
- Train on days 1-13; choose each threshold on validation days 14-17; test on days 18-21.
- Threshold rule: maximize validation window F1, with higher threshold breaking exact ties.
- Main model was selected by design, before examining test results.

## Test results

| Model | ROC-AUC | Average precision | Precision | Recall | F1 | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_current_glucose | 0.7584 | 0.1021 | 0.1175 | 0.2308 | 0.1557 | 0.1058 |
| wearable_only_hgb | 0.9537 | 0.4665 | 0.4243 | 0.5449 | 0.4771 | 0.2460 |
| fused_hgb | 0.9624 | 0.5023 | 0.4562 | 0.5342 | 0.4921 | 0.2978 |

Test windows: 14,258; event windows: 468; prevalence: 3.28%.

Fused confusion counts: TN=13492, FP=298, FN=218, TP=250.

Fused minus wearable-only average precision: +0.0358. This is a descriptive simulation result, not proof that EHR improves real-world outcomes.

## Features and interpretation

Dynamic: glucose_current, glucose_lag_30m, glucose_lag_60m, glucose_lag_120m, glucose_mean_2h, glucose_std_2h, glucose_min_2h, glucose_max_2h, glucose_slope_30m, glucose_delta_2h, hour_sin, hour_cos.

Baseline EHR: age_years, sex_female, bmi_kg_m2, hypertension, diabetes_duration_years, hba1c_percent.

Patient ID, dates, target labels, future glucose and hidden simulator parameters are excluded.

Model scores have not undergone separate probability calibration; the alert threshold is a prototype F1 operating point, not a clinical decision threshold.

## Limitations

- Fully synthetic data; good metrics do not imply clinical validity or Indian population representativeness.
- Train and test include the same patients at different times; unseen-patient generalization is untested.
- Overlapping prediction windows and repeated patients are correlated; counts are not independent clinical events.
- Window recall is not episode-level recall, and repeated alerts are not deduplicated.
- EHR-CGM associations are engineered simulator assumptions, not established biological effect estimates.
- Complete regular CGM omits realistic dropout, missingness, treatment changes and device bias.
- F1 threshold selection does not represent clinical costs, safe alert burden or a treatment policy.
- Probabilities are not independently calibrated; no diagnosis, treatment or medication advice is provided.

## Reproduce

```powershell
.\.venv\Scripts\python.exe generate_data.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe verify_model.py
```

The same code, data and dependency versions are required. Retraining this fixed protocol is reproduction; do not revise it in response to test scores and still call that test independent.
