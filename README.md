# GlucoTwin

Synthetic patient twin for next-hour glucose alerts in Type 2 diabetes.

**Status:** Data generator, trained model, chronological evaluation and Streamlit dashboard are implemented. Final submission still requires video access and fresh-clone/public-link checks.

## Team and College / Incubator

- Team: **GlucoTwin** (working name; editable in project_info.json)
- Team leader and sole member: **Anusha Gupta**
- College: **IIT Kanpur**
- Team size: 1
- Roles: Data simulation, ML, dashboard, documentation and demonstration
- Incubator: Not applicable
- Suggested repository/folder name: `GlucoTwin_IIT_Kanpur` (update if the team name changes).

## Problem Statement and Healthcare Use Case

Build a proof of concept in which a doctor reviews a patient's baseline EHR alongside continuously replayed CGM measurements and a next-hour event score. The use case is advance review of a possible glucose threshold excursion, without prescribing treatment or asserting clinical effectiveness.

## Digital Twin Behavior

Each synthetic patient has linked baseline EHR and a timestamped CGM stream. Changing the selected patient or advancing twin time updates the available history, feature vector, model score and alert state. A shared trained model serves patient-specific inputs; the model does not retrain during replay. This is a data-driven patient-state prototype, not a validated physiological simulator.

## Prediction Target

- Eligible observation: current glucose is below **180 mg/dL**.
- Event: at least one CGM reading reaches **180 mg/dL** in **(t, t + 60 minutes]**.
- Current glucose is excluded from the future target window.
- This is a documented prototype definition, not a personalized treatment threshold.
- Already-above-threshold observations display 'Not scored' for new-onset risk.

## Architecture

[Architecture PDF](docs/architecture.pdf)

Synthetic EHR + synthetic CGM -> patient ID join -> historical features -> chronological training/validation/test -> saved fused model -> patient replay dashboard. Future CGM is accessed for evaluation labels only; the dashboard's optional retrospective reveal never enters inference.

## Synthetic Data and Provenance

- 80 baseline EHR records and 161,280 CGM readings, covering 21 simulated days.
- CGM every 15 minutes; synthetic IDs link both tables; seed `42`.
- EHR includes age, sex, BMI, diagnoses, diabetes duration and baseline HbA1c.
- Baseline EHR is generated before CGM. HbA1c is not computed from future observations.
- Glucose includes meal-like responses, daily patterns and stochastic noise. Hidden simulator factors are not exported as features.
- No real patient records, identifiers, external datasets or genetic markers are used.

[Data dictionary](data/DATA_DICTIONARY.md) | [Provenance and checksums](data/provenance.json) | [Data reuse terms](data/DATA_LICENSE.md)

## Technical Stack and AI/ML Details

Python, NumPy, pandas, scikit-learn, Streamlit, matplotlib, joblib and ReportLab. CPU only; CUDA is unnecessary. Risk inference uses scikit-learn, with no LLM or external inference API.

- Baseline: StandardScaler + LogisticRegression on current glucose only; scaler fits on training data.
- Wearable-only ablation: HistGradientBoostingClassifier on 12 dynamic/clock features.
- Main model: the same boosting settings with six baseline EHR features added (18 total inputs).
- Dynamic features: current glucose, 30/60/120-minute lags, two-hour mean/std/min/max, slope, delta, and sine/cosine clock time.
- Static predictors: age, deterministic sex indicator, BMI, hypertension, diabetes duration and HbA1c.
- Patient ID, baseline date, target labels and future glucose are excluded from predictors.

Full settings are in [protocol.json](reports/protocol.json); no hyperparameter search was performed. Internal random early stopping is disabled.

## Chronological Evaluation and Leakage Controls

| Split | Days | Eligible windows | Event windows |
| --- | --- | ---: | ---: |
| Train | 1-13 | 47,339 | 1,657 |
| Validation | 14-17 | 14,209 | 514 |
| Test | 18-21 | 14,258 | 468 |

Every two-hour history and one-hour label window stays completely inside its split. Evaluation anchors are sampled every 30 minutes. Train first, choose each alert cutoff by maximum validation window F1, then evaluate the test set after settings and cutoffs are frozen. The main model is preselected by design, not by test ranking.

The same patients recur across time splits. This tests later periods for known synthetic patients, not unseen-patient or real-world generalization. Overlapping windows are correlated.

## Test Results

Results below are generated from this checkout's reports/metrics.json.

| Model | ROC-AUC | Average precision | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current glucose baseline | 0.7584 | 0.1021 | 11.8% | 23.1% | 0.1557 |
| CGM history + clock | 0.9537 | 0.4665 | 42.4% | 54.5% | 0.4771 |
| CGM history + clock + EHR | 0.9624 | 0.5023 | 45.6% | 53.4% | 0.4921 |

Test prevalence: **3.28%**. Fused cutoff: **0.2978**, selected on validation only.

Fused confusion counts: TN **13,492**, FP **298**, FN **218**, TP **250**. About half the event windows are missed at this operating point. A below-cutoff score does not establish safety.

[Detailed results](reports/results.md) | [Machine-readable metrics](reports/metrics.json) | [Precision-recall / ROC curves](reports/model_comparison.png) | [Confusion matrix](reports/confusion_matrix.png)

## Windows Setup and Reproduction

Python 3.12 is a practical starting point. Open PowerShell in the project folder. Use the submitted requirements.txt to reproduce the checked environment; requirements.in lists direct dependencies.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify_setup.py
.\.venv\Scripts\python.exe generate_data.py
.\.venv\Scripts\python.exe validate_data.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe verify_model.py
.\.venv\Scripts\python.exe verify_dashboard.py
.\.venv\Scripts\python.exe build_docs.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501 and keep the terminal running. Stop with Ctrl+C. No environment activation, GPU, cloud service, API key or account is required to run locally.

Regenerate data and retrain together if the config or dependency versions change. Load only a model you generated or otherwise trust. The saved bundle records feature order, validation cutoff, runtime versions and checksums.

### Record the final environment

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding ascii requirements.txt
```

Use a clean project .venv, not a global Python environment, for that snapshot. Same script, seed and dependency versions reproduce generated CSV bytes; rerunning a fixed modeling protocol is reproduction, not permission to tune on test results.

## Doctor Dashboard

1. Select a synthetic patient and inspect baseline EHR.
2. Move Twin time or click Next reading to update patient state and score.
3. Inspect the available glucose history, model score, cutoff and alert.
4. Optionally reveal the observed next hour. Orange dashed values are later observations, not a predicted trajectory.
5. Change time or patient; revealed follow-up hides automatically.
6. Review model evaluation and limitations. At the dataset end, incomplete follow-up is explicitly reported.

[Automated dashboard checks and local demo cases](reports/dashboard_checks.json). The default chart shows eight hours of history; the model uses only the latest two hours.

## Demonstration Video - Minimum 20 Minutes

[Watch the demonstration video](paste_link)

Required duration: at least 20 minutes. Verify actual duration and signed-out access.

## Presentation

[Project presentation PDF](docs/presentation.pdf) | [Editable content source](docs/presentation.md)

## Open-Source License

Original code and project-generated synthetic data are distributed under the [MIT License](LICENSE); generated-data terms are also documented in data/DATA_LICENSE.md. Dependencies retain their own licenses; [third-party notices](THIRD_PARTY_NOTICES.md) are generated from installed distribution metadata and packaged notice files.

## Limitations and Ethics

- Fully synthetic data; good metrics do not imply clinical validity or Indian population representativeness.
- Train and test include the same patients at different times; unseen-patient generalization is untested.
- Overlapping prediction windows and repeated patients are correlated; counts are not independent clinical events.
- Window recall is not episode-level recall, and repeated alerts are not deduplicated.
- EHR-CGM associations are engineered simulator assumptions, not established biological effect estimates.
- Complete regular CGM omits realistic dropout, missingness, treatment changes and device bias.
- F1 threshold selection does not represent clinical costs, safe alert burden or a treatment policy.
- Probabilities are not independently calibrated; no diagnosis, treatment or medication advice is provided.
- No real patient data is collected. Synthetic-only operation is not a claim of DPDP/HIPAA certification or regulatory approval.

## Future Work

Evaluate unseen patients, realistic missingness, calibration and episode-level alert burden; then consider appropriately governed external and prospective validation. None of these validations is claimed here.

## Submission Checklist

- [x] Team leader and college details provided
- [x] Linked synthetic EHR and CGM with generator, dictionary and provenance
- [x] Trained fused predictor, baseline, chronological evaluation and leakage checks
- [x] Working doctor-facing dashboard with patient/time replay
- [x] Architecture PDF and project presentation PDF
- [x] MIT license and dependency metadata/notices
- [ ] Confirm final team name and official repository naming convention
- [ ] Verify full-time-student eligibility and any incubator information
- [ ] Verify linked recording duration and signed-out playback
- [ ] Test from a fresh clone and clean environment
- [ ] Verify public repository, PDFs and all links while signed out
- [ ] Submit leader name, phone, email and GitHub URL; retain confirmation

Author contact phone/email are supplied on the submission platform, not required in this public README.
