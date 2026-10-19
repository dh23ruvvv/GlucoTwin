# Step 3: train, evaluate and verify

Merge this update's contents into your EXISTING project folder. It does not replace your
README, data, config, requirements or Git settings. Bundled model/results are from the checked
reference run; train locally before using the model so it matches your dependency versions.

## Run in Windows PowerShell

```powershell
.\.venv\Scripts\python.exe validate_data.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe verify_model.py
.\.venv\Scripts\python.exe predict.py --patient SYN_0001 --at "2026-01-18T12:00:00"
```

If generation/config/checksum validation fails, regenerate data first and repeat these commands.
Do not regenerate data after training unless you also retrain. Stop on errors.
No additional packages beyond Step 1 are needed; threadpoolctl is installed with scikit-learn.

## What trains

1. Baseline: StandardScaler fitted on TRAINING current glucose, followed by logistic regression.
2. Wearable-only ablation: fixed HistGradientBoostingClassifier on CGM history and current clock time.
3. Main model: same boosting settings, with baseline EHR added (18 total features).

The main model is chosen in advance, not by test ranking. There is no hyperparameter sweep.
Internal random early stopping is disabled. Each model's alert threshold maximizes VALIDATION
window F1; the test set is used after all models and thresholds have been fixed.

Historical mean/std/min/max include the current observation and the preceding two hours,
which is nine readings with 15-minute sampling. Slope is in mg/dL/minute. Clock time uses sine/cosine.
The binary sex indicator is a deterministic encoding, not a learned preprocessing step.

The same patients recur across calendar splits. No patient ID or future-derived information is a
predictor. The model is a population-trained predictor with patient-specific inputs and replayed
state updates, not a separately fitted physiological simulator or online-learning system.

## Output

- models/model.joblib: saved main estimator, ordered feature list, validation threshold,
  configuration, runtime versions and source-data/code checksums.
- reports/protocol.json: modeling decisions written before test evaluation.
- reports/metrics.json: validation/test metrics and exact confusion counts for all three models.
- reports/results.md: human-readable results and limitations.
- reports/model_comparison.png: precision-recall and ROC curves.
- reports/confusion_matrix.png: main model test counts at the validation-selected threshold.
- reports/feature_examples.csv: small inspection sample, including separate metadata/target columns.
- reports/model_checks.json: boundary, causal-input and replay verification.

Only the explicit feature_columns list enters the estimator. The inspection CSV intentionally
also contains target/metadata for review; never feed every CSV column to a classifier.
The score is not independently calibrated. This prototype's F1 threshold is not a medical threshold.

## Interpretation

Read LOCAL reports/results.md after training. Do not copy reference numbers if local results differ.
Average precision matters with rare events; accuracy is not the headline. Window recall does not
measure episode-level recall. Overlapping windows and repeated patients are correlated. Report the
EHR ablation honestly even if adding EHR does not improve a metric. Synthetic performance cannot
establish clinical benefit or validation on real patients.

## Commit

```powershell
git add features.py train.py predict.py verify_model.py models/model.joblib reports docs/STEP3.md
git commit -m "Train and evaluate fused glucose risk model without temporal leakage"
git push
```

The README can link to reports/results.md and contain the run commands above. Its mandatory final
cleanup is in Step 5; do not describe the dashboard as implemented until Step 4 is complete.

## Common errors

- Model missing: run train.py first.
- scikit-learn version mismatch: retrain locally; do not rely on a model serialized with a different version.
- Feature code/config/data changed: regenerate if needed, then retrain and reverify.
- Missing sklearn/threadpoolctl: rerun Step 1 installation with the same .venv Python.
- No complete history: choose an actual timestamp with at least two hours of prior readings.
- Already above event threshold: inference deliberately withholds NEW-ONSET risk; this is not an error.
- Any joblib model file must come from trusted project code; the simplest route is to train your own locally.

Next step: a one-page Streamlit dashboard reusing make_feature_row and score_patient unchanged.
