"""Command-line patient inference. No future target is constructed or accessed."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from features import FEATURE_VERSION, FUSED_FEATURES, make_feature_row, read_inputs
from generate_data import load_config, sha256

ROOT = Path(__file__).resolve().parent


def check_bundle(bundle, config, data_dir=None):
    if bundle["feature_version"] != FEATURE_VERSION or bundle["feature_columns"] != FUSED_FEATURES:
        raise ValueError("Feature schema mismatch; run train.py again")
    if bundle["features_sha256"] != sha256(ROOT / "features.py"):
        raise ValueError("Feature code changed; run train.py again")
    if bundle["config"] != config:
        raise ValueError("Config changed after training; regenerate and retrain")
    if bundle["runtime"]["scikit-learn"] != sklearn.__version__:
        raise ValueError("scikit-learn version differs; train the model in this environment")
    if data_dir is not None:
        for name, checksum in bundle["source_hashes"].items():
            if sha256(Path(data_dir) / name) != checksum:
                raise ValueError("Training data changed; run train.py again")


def score_patient(bundle, ehr_row, patient_cgm, as_of):
    config = bundle["config"]
    row = make_feature_row(ehr_row, patient_cgm, as_of, config)
    current = float(row["glucose_current"].iloc[0])
    eligible = current < config["event_threshold_mg_dl"]
    result = {
        "patient_id": str(ehr_row["patient_id"]), "as_of": pd.Timestamp(as_of).isoformat(),
        "current_glucose_mg_dl": current, "eligible_for_incident_event_prediction": eligible,
        "prediction_horizon_minutes": config["prediction_horizon_minutes"],
        "risk_score": None, "alert_threshold": float(bundle["threshold"]), "alert": False,
        "status": "Already at or above the prototype event threshold; new-onset risk is not scored",
        "note": "Synthetic prototype; score is not clinically calibrated or treatment advice",
    }
    if eligible:
        with threadpool_limits(limits=4):
            score = float(bundle["estimator"].predict_proba(row[bundle["feature_columns"]])[0, 1])
        result.update(risk_score=score, alert=bool(score >= bundle["threshold"]),
                      status="Prototype alert for review" if score >= bundle["threshold"] else "Below prototype alert threshold")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="SYN_0001")
    parser.add_argument("--at", default="2026-01-18T12:00:00")
    args = parser.parse_args()
    config = load_config(ROOT / "config.json")
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    check_bundle(bundle, config, ROOT / "data")
    ehr, cgm = read_inputs(ROOT / "data")
    rows = ehr.loc[ehr["patient_id"] == args.patient]
    if len(rows) != 1:
        raise ValueError("Unknown synthetic patient ID")
    patient_cgm = cgm.loc[cgm["patient_id"] == args.patient]
    print(json.dumps(score_patient(bundle, rows.iloc[0], patient_cgm, args.at), indent=2))


if __name__ == "__main__":
    main()
