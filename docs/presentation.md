# GlucoTwin presentation source

Anusha Gupta | IIT Kanpur | Team GlucoTwin

## 1. GlucoTwin
Synthetic patient twin for next-hour glucose alerts; solo proof of concept.

## 2. Healthcare use case and target
Current glucose <180 mg/dL; any next-hour CGM >=180 mg/dL. Prototype definition only.

## 3. Synthetic data
Linked baseline EHR and CGM. Baseline labs precede the generated stream.

## 4. Architecture
EHR + CGM -> causal historical features -> chronological fitting/evaluation -> saved model -> replay UI.

## 5. Evaluation
Train days 1-13; validation 14-17; test 18-21. Complete windows inside each split; same patients recur.

## 6. Results
See ../reports/results.md for the exact local metrics used in this build.

## 7. Errors
Report precision, recall and confusion counts. Below-cutoff does not mean safe.

## 8. Doctor workflow
Select patient; advance time; inspect history and score; optionally reveal follow-up without changing inference.

## 9. Limitations and future work
Synthetic patterns, known patients, correlated windows and uncalibrated scores. External and clinical validation remain future work.
