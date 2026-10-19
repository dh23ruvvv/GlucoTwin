"""Exercise actual Streamlit widgets and inference paths using AppTest."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from threadpoolctl import threadpool_limits

from features import build_training_tables, read_inputs
from generate_data import load_config
from predict import score_patient

ROOT = Path(__file__).resolve().parent


def no_errors(app):
    if len(app.exception):
        raise AssertionError("Dashboard exception: " + str(app.exception[0].message))
    if len(app.error):
        raise AssertionError("Dashboard setup error: " + str(app.error[0].value))


def snapshot(app):
    no_errors(app)
    return dict(app.session_state["twin_snapshot"])


def select_snapshot(app, patient_id, timestamp):
    app.selectbox(key="patient_id").select(patient_id).run()
    no_errors(app)
    app.select_slider(key="replay_time").set_value(pd.Timestamp(timestamp).to_pydatetime()).run()
    return snapshot(app)


def main():
    config = load_config(ROOT / "config.json")
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    ehr, cgm = read_inputs(ROOT / "data")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    initial = snapshot(app)
    checks = []
    if initial["future_revealed"] or initial["visible_future_points"]:
        raise AssertionError("Future observations must be hidden initially")
    if pd.Timestamp(initial["chart_last_timestamp"]) > pd.Timestamp(initial["as_of"]):
        raise AssertionError("Default chart displays future observations")
    checks.append("Dashboard loads with EHR, model inference and no future observations visible")

    app.button(key="next_reading").click().run()
    moved = snapshot(app)
    if pd.Timestamp(moved["as_of"]) - pd.Timestamp(initial["as_of"]) != pd.Timedelta(minutes=config["sampling_minutes"]):
        raise AssertionError("Next reading did not advance exactly one sample")
    static = ehr.loc[ehr["patient_id"] == moved["patient_id"]].iloc[0]
    patient = cgm.loc[cgm["patient_id"] == moved["patient_id"]]
    reference = score_patient(bundle, static, patient, moved["as_of"])
    for key in ("risk_score", "current_glucose_mg_dl", "alert"):
        if moved[key] != reference[key]:
            raise AssertionError(f"UI diverges from verified inference: {key}")
    checks.append("Time navigation advances one reading and matches shared saved-model inference")

    app.button(key="previous_reading").click().run()
    if snapshot(app)["as_of"] != initial["as_of"]:
        raise AssertionError("Previous reading did not return to the original anchor")
    app.button(key="next_reading").click().run()
    moved = snapshot(app)

    app.checkbox(key="reveal_outcome").check().run()
    revealed = snapshot(app)
    if not revealed["future_revealed"] or revealed["visible_future_points"] == 0:
        raise AssertionError("Outcome reveal did not display follow-up")
    if revealed["risk_score"] != moved["risk_score"]:
        raise AssertionError("Outcome reveal changed the prediction")
    app.button(key="next_reading").click().run()
    hidden = snapshot(app)
    if hidden["future_revealed"] or hidden["visible_future_points"]:
        raise AssertionError("Advancing time must hide revealed outcomes")
    checks.append("Revealing follow-up leaves the prediction unchanged; navigation hides follow-up again")

    # Choose demonstration cases using validation scores/current values, not future outcomes.
    tables = build_training_tables(ehr, cgm, config)
    validation = tables["validation"]
    with threadpool_limits(limits=4):
        probabilities = bundle["estimator"].predict_proba(validation[bundle["feature_columns"]])[:, 1]
    cases = []
    for title, mask in (("Prototype alert", probabilities >= bundle["threshold"]),
                        ("Below cutoff", probabilities < bundle["threshold"])):
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise AssertionError(f"No validation example for {title}")
        row = validation.iloc[indices[0]]
        state = select_snapshot(app, row["patient_id"], row["timestamp"])
        if state["alert"] != (title == "Prototype alert"):
            raise AssertionError("Alert state is incorrect")
        cases.append({"case": title, "patient_id": row["patient_id"], "as_of": row["timestamp"].isoformat(),
                      "risk_score": state["risk_score"]})
    checks.append("Alert and below-cutoff paths both render using validation examples")

    val_start = pd.Timestamp(config["start_date"]) + pd.Timedelta(days=config["train_days"])
    val_end = val_start + pd.Timedelta(days=config["validation_days"])
    high = cgm.loc[(cgm["timestamp"] >= val_start) & (cgm["timestamp"] < val_end) &
                   (cgm["glucose_mg_dl"] >= config["event_threshold_mg_dl"])].iloc[0]
    state = select_snapshot(app, high["patient_id"], high["timestamp"])
    if state["eligible_for_incident_event_prediction"] or state["risk_score"] is not None:
        raise AssertionError("Already-high state must withhold new-onset risk")
    cases.append({"case": "Already above threshold", "patient_id": high["patient_id"],
                  "as_of": high["timestamp"].isoformat(), "risk_score": None})
    checks.append("Already-above-threshold state withholds new-onset risk rather than displaying zero")

    app.checkbox(key="reveal_outcome").check().run()
    other_id = next(value for value in ehr["patient_id"] if value != high["patient_id"])
    app.selectbox(key="patient_id").select(other_id).run()
    state = snapshot(app)
    if state["patient_id"] != other_id or state["future_revealed"]:
        raise AssertionError("Patient switch must update twin and hide follow-up")
    checks.append("Patient switching updates the twin and resets retrospective reveal")

    last = cgm.loc[cgm["patient_id"] == other_id, "timestamp"].max()
    select_snapshot(app, other_id, last)
    app.checkbox(key="reveal_outcome").check().run()
    state = snapshot(app)
    if state["visible_future_points"] != 0:
        raise AssertionError("The last reading cannot have observed future follow-up")
    if not any("full follow-up window" in str(item.value) for item in app.info):
        raise AssertionError("Incomplete follow-up must be explicitly explained")
    checks.append("End-of-data state reports unavailable follow-up without manufacturing an outcome")

    app.button(key="reset_replay").click().run()
    state = snapshot(app)
    if state["as_of"] != initial["as_of"] or state["future_revealed"]:
        raise AssertionError("Reset must restore default time and hide retrospective follow-up")
    checks.append("Previous and reset controls restore the expected time and hide follow-up")

    output = {"status": "PASS", "checks": checks, "validation_demo_cases": cases,
              "note": "UI and software checks, not clinical validation. Cases selected without using future labels."}
    (ROOT / "reports" / "dashboard_checks.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for check in checks:
        print("PASS:", check)
    print("\nDASHBOARD CHECKS PASSED - ready for Step 5: documentation and PDFs.")


if __name__ == "__main__":
    main()
