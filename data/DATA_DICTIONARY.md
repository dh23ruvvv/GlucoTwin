# GlucoTwin synthetic data dictionary

These are project-generated demonstration records, not a named external dataset.
There are no real patients, names, contact details, external identifiers or genetic records.
No claims of clinical calibration or Indian population representativeness are made.

## ehr.csv: one baseline record per synthetic patient

| Field | Type / units | Meaning |
| --- | --- | --- |
| patient_id | string, SYN_0001 etc. | Join key; exclude from predictors |
| baseline_date | YYYY-MM-DD | Synthetic EHR snapshot date, before all CGM |
| age_years | integer years | Simulated age, 25-78 |
| sex_at_birth | female / male | Simplified simulated categorical demographic; not comprehensive |
| bmi_kg_m2 | numeric kg/m^2 | Baseline BMI, 19-42 |
| type2_diabetes | integer 1 | Existing diagnosis for every simulated patient; constant, exclude from predictors |
| hypertension | integer 0 / 1 | Synthetic comorbid diagnosis |
| diabetes_duration_years | numeric years | Duration, 0.5 to 25, capped relative to age |
| hba1c_percent | numeric percent | Baseline lab, 5.8-10.8; sampled BEFORE CGM, never computed from future observations |

The lab and CGM share hidden simulator severity to create an explicit, artificial association.
Any predictive gain from EHR may therefore reflect simulation assumptions rather than clinical value.
No medications, treatments or genetic markers are modeled.

## cgm.csv: one measurement per patient every 15 minutes

| Field | Type / units | Meaning |
| --- | --- | --- |
| patient_id | string | Foreign key to ehr.csv |
| timestamp | YYYY-MM-DDTHH:MM:SS | Synthetic local time in Asia/Kolkata; stored without offset |
| glucose_mg_dl | numeric mg/dL | Rounded simulated sensor reading, clipped to generator bounds 65-340 |

Default configuration: 80 patients x 21 days x 96 readings/day = 161,280 CGM rows.
Every patient has the same observation dates, with no missing measurements.
Day 1 is 2026-01-01; the EHR baseline is 2025-12-31. These dates are simulation time,
not the hackathon deadline or a real patient's observation dates.

## Data generation

`generate_data.py` uses independent seeded random streams for baseline EHR and patient trajectories.
CGM combines patient baseline, daily periodic variation, causal meal-like responses, unobserved
activity/stress responses, correlated noise and measurement noise. Event response is zero before onset.
The event threshold does not control glucose generation. Hidden severity, meal timing, response
parameters and event schedules are not exported or available to the prediction model.

## Planned target and chronological protocol

At an anchor time t, use baseline EHR and glucose observed no later than t.
If current glucose is below 180 mg/dL, label 1 when any of the next four 15-minute readings
is at least 180 mg/dL; otherwise label 0. The window is (t, t+60 minutes], excluding t.
This is an operational prototype definition, not a treatment recommendation.

Anchors are every 30 minutes. Inputs use up to two hours of past readings.
Days 1-13 are training, 14-17 validation, 18-21 test. Discard anchors whose history or
target would leave their split. The same patients recur across splits. Never use patient ID,
future glucose, future labels or simulator-only variables as predictors.

`validate_data.py` reports counts using this protocol. It fits no model and exports no
future-derived feature table. Freeze modeling decisions using training/validation only;
data-quality counts are not permission to tune on test performance.

## Reproducibility and provenance

The seed and all settings are in config.json. Runtime versions, generator checksum,
file checksums and full settings are stored in provenance.json. Same config, script and
dependency versions reproduce the CSV bytes. Pin requirements on the machine used for submission.
Different dependency versions may produce different bytes; rerun generation and validation together.
Changing the seed intentionally creates a different simulated cohort.

## Limitations

Complete, regularly sampled synthetic trajectories are easier than real sensor data. Missingness,
device failure, treatment changes, clinical confounding, population diversity and measurement bias
are not adequately represented. Good synthetic test results do not establish medical safety,
clinical benefit, external validity, probability calibration or regulatory compliance.
