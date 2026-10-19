"""Meaningful checks for boundary leakage, causal replay and trained artifacts."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import (EHR_FEATURES, FUSED_FEATURES, build_training_tables, make_feature_row,
                      read_inputs, split_intervals)
from generate_data import load_config
from predict import check_bundle, score_patient
from validate_data import check_data

ROOT = Path(__file__).resolve().parent


def main():
    config = load_config(ROOT / "config.json")
    quality = check_data(config, ROOT / "data")
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    check_bundle(bundle, config, ROOT / "data")
    ehr, cgm = read_inputs(ROOT / "data")
    tables = build_training_tables(ehr, cgm, config)
    ehr_by_id = ehr.set_index("patient_id", drop=False)
    intervals = split_intervals(config)
    checks = []
    if set(FUSED_FEATURES) & {"patient_id", "timestamp", "target_event", "history_start", "label_end", "split"}:
        raise AssertionError("Metadata or labels entered the predictor list")
    if len(bundle["feature_columns"]) != bundle["estimator"].n_features_in_:
        raise AssertionError("Estimator feature count mismatch")
    if not set(EHR_FEATURES).issubset(bundle["feature_columns"]):
        raise AssertionError("EHR is absent from the saved model inputs")
    if bundle["estimator"].early_stopping is not False:
        raise AssertionError("Random internal early stopping must remain disabled")
    checks.append("Saved estimator ingests all 18 intended dynamic and static features; no labels or IDs")

    for split, frame in tables.items():
        start, end = intervals[split]
        if not ((frame["history_start"] >= start) & (frame["label_end"] < end)).all():
            raise AssertionError("A feature or label window crosses a split boundary")
        if len(frame) != quality["splits"][split]["eligible_windows"]:
            raise AssertionError("Target counts changed")
        for position in np.linspace(0, len(frame) - 1, 3, dtype=int):
            example = frame.iloc[position]
            patient = cgm.loc[cgm["patient_id"] == example["patient_id"]].copy()
            static = ehr_by_id.loc[example["patient_id"]]
            at = example["timestamp"]
            online = make_feature_row(static, patient, at, config)
            np.testing.assert_allclose(online.iloc[0].to_numpy(dtype=float),
                                       example[FUSED_FEATURES].to_numpy(dtype=float), rtol=0, atol=1e-12)
            future = patient.loc[(patient["timestamp"] > at) & (patient["timestamp"] <= example["label_end"])]
            expected_y = int((future["glucose_mg_dl"] >= config["event_threshold_mg_dl"]).any())
            if expected_y != example["target_event"]:
                raise AssertionError("Target is not the next-hour event")
            changed = patient.copy()
            changed.loc[changed["timestamp"] > at, "glucose_mg_dl"] = 333.0
            modified_row = make_feature_row(static, changed, at, config)
            pd.testing.assert_frame_equal(online, modified_row)
            truncated = make_feature_row(static, patient.loc[patient["timestamp"] <= at], at, config)
            pd.testing.assert_frame_equal(online, truncated)
    checks.append("All history and label windows stay wholly within chronological splits")
    checks.append("Training features equal replay features at sampled anchors in all splits")
    checks.append("Changing or removing all future readings leaves prediction inputs unchanged")
    checks.append("Sampled targets equal the observed next-hour event, excluding current glucose")

    # Check an actual saved-model inference, and a real past-input perturbation.
    example = tables["validation"].iloc[0]
    static = ehr_by_id.loc[example["patient_id"]]
    patient = cgm.loc[cgm["patient_id"] == example["patient_id"]].copy()
    at = example["timestamp"]
    original = score_patient(bundle, static, patient, at)
    changed = patient.copy()
    changed.loc[changed["timestamp"] > at, "glucose_mg_dl"] = 333.0
    if score_patient(bundle, static, changed, at) != original:
        raise AssertionError("Future values changed actual inference")
    if original["risk_score"] is None or not 0 <= original["risk_score"] <= 1:
        raise AssertionError("Invalid saved model probability")
    changed_past = patient.copy()
    changed_past.loc[changed_past["timestamp"] == at, "glucose_mg_dl"] += 1
    old_row = make_feature_row(static, patient, at, config)
    new_row = make_feature_row(static, changed_past, at, config)
    if old_row.equals(new_row):
        raise AssertionError("Current observations do not update twin features")
    checks.append("Saved-model inference ignores future values; available input changes update twin features")
    output = {"status": "PASS", "checks": checks, "example_prediction": original,
              "note": "Software and leakage checks only; not clinical validation"}
    (ROOT / "reports" / "model_checks.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for check in checks:
        print("PASS:", check)
    print("\nMODEL CHECKS PASSED - ready for Step 4: Streamlit dashboard.")


if __name__ == "__main__":
    main()
