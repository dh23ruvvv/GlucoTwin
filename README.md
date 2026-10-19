# GlucoTwin

Synthetic patient digital twin for a Type 2 diabetes demonstration.

**Status: Step 1 setup only. Data generation, model and dashboard are not implemented yet.**

## Team and College / Incubator

- Team name: TODO
- Team leader: TODO
- Members and roles: TODO (solo developer)
- College: TODO
- Incubator: TODO or Not applicable

## Project Summary

Combine synthetic baseline EHR and synthetic CGM readings to estimate whether glucose will reach at least 180 mg/dL within 60 minutes, for observations currently below that threshold. This threshold defines a prototype event; it is not personalized medical advice.

## Problem Statement and Healthcare Use Case

Demonstrate how a clinician could review a patient's baseline context, recent glucose history and a model-estimated alert as new simulated readings arrive. No treatment recommendations are provided.

## Prediction Target and Doctor Workflow

- Static EHR: demographics, BMI, diagnoses, diabetes duration and baseline HbA1c.
- Dynamic stream: CGM sampled every 15 minutes, joined by synthetic patient ID.
- Inputs at time t: baseline EHR and readings at or before t only.
- Label: any CGM reading at or above 180 mg/dL in (t, t + 60 minutes].
- Only examples with current CGM below 180 mg/dL are eligible.
- Twin updates use replay of synthetic observations; this does not imply online retraining.

## Architecture

TODO: add docs/architecture.pdf in Step 5.

## Synthetic Data and Provenance

TODO: generator and data dictionary in Step 2. All planned records are generated from code with a fixed seed. No real patient data or external datasets are used.

## Technical Stack and Model

Python, NumPy, pandas, scikit-learn, Streamlit, matplotlib and joblib. ReportLab is used for PDF deliverables. CPU only; CUDA is unnecessary.

Planned baseline: logistic regression using current glucose. Planned fused model: HistGradientBoostingClassifier with EHR and CGM history. Final model and settings will be documented after evaluation.

## Features, Chronological Splits and Leakage Prevention

Planned split: days 1-13 training; 14-17 validation; 18-21 testing. Every historical input window and future label window must be wholly inside its split. Fit learned preprocessing on training data only; choose alert threshold on validation data only. Exclude patient IDs from features. Disable random internal validation in the boosting model.

Patients recur across time splits; this evaluates future observations from known synthetic patients, not generalization to unseen patients. Baseline labs must not be derived from future CGM.

## Baseline Comparison and Test Results

TODO: actual results in Step 3. No performance claims have been made.

## Windows Setup and Reproduction

Open PowerShell inside this extracted project folder. Python 3.12 is suggested; the setup checker requires at least Python 3.10. Package support is confirmed by installation and imports, not assumed.

```powershell
py --version
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify_setup.py
```

After SETUP OK, record this environment's exact package versions:

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding ascii requirements.txt
git --version
git init
git branch -M main
git add .
git commit -m "Set up GlucoTwin project"
```

requirements.in retains the direct dependency list. requirements.txt initially contains that list; after the successful installation command above it becomes the environment's exact dependency snapshot. Future reproduction uses requirements.txt. No environment activation or PowerShell execution-policy change is necessary.

Create an empty public GitHub repository named with your real TeamName_CollegeName. Do not add a remote README, license or gitignore. Then substitute your own URL:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_TEAM_YOUR_COLLEGE.git
git push -u origin main
```

TODO: add data generation, training and dashboard run commands as those scripts are implemented.

## Dashboard Walkthrough

TODO: Step 4.

## Demonstration Video - At Least 20 Minutes

TODO: user will provide an accessible video link before submission.

## Presentation

TODO: docs/presentation.pdf in Step 5.

## Limitations, Ethics and Intended Use

Synthetic demonstration only. Patterns and measured performance depend on the simulator. There is no established clinical validity, population representativeness or verified probability calibration. It is not for diagnosis, treatment or real patient monitoring. Do not claim regulatory compliance solely because data are synthetic.

## Open-Source License and Dependency Notices

Original code is provided under the MIT License; see LICENSE. Dependency licenses remain their respective owners' licenses. THIRD_PARTY_NOTICES.md will be completed from installed package metadata. Explicit generated-data reuse terms will be added when the generator is implemented.

## Future Work

External validation on appropriately governed real-world data, patient-level evaluation, calibration and prospective clinical assessment. These are outside this submission's implementation scope.

## Submission Checklist

- [ ] Real team and institution details
- [ ] Linked synthetic EHR and CGM, generator and dictionary
- [ ] Trained fused model, baseline and chronological evaluation
- [ ] Working doctor-facing dashboard
- [ ] Reproduction checked from a fresh clone
- [ ] Architecture PDF
- [ ] Presentation PDF
- [ ] Accessible video of at least 20 minutes
- [ ] License and dependency notices complete
- [ ] Public repository with correct name and no unfinished placeholders
- [ ] Platform submission confirmed before official deadline
