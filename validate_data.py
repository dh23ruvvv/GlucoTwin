"""Check linked synthetic inputs and target availability, without fitting a model."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generate_data import GENERATOR_VERSION, load_config, sha256

ROOT = Path(__file__).resolve().parent
EHR_COLUMNS = ["patient_id", "baseline_date", "age_years", "sex_at_birth", "bmi_kg_m2",
               "type2_diabetes", "hypertension", "diabetes_duration_years", "hba1c_percent"]
CGM_COLUMNS = ["patient_id", "timestamp", "glucose_mg_dl"]


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def check_data(config, data_dir):
    ehr = pd.read_csv(data_dir / "ehr.csv")
    cgm = pd.read_csv(data_dir / "cgm.csv", parse_dates=["timestamp"])
    provenance = json.loads((data_dir / "provenance.json").read_text(encoding="utf-8"))
    require(list(ehr.columns) == EHR_COLUMNS, "Unexpected EHR schema")
    require(list(cgm.columns) == CGM_COLUMNS, "Unexpected CGM schema")
    require(not ehr.isna().any().any() and not cgm.isna().any().any(), "Missing values found")
    require(len(ehr) == config["n_patients"], "Wrong EHR row count")
    require(ehr["patient_id"].is_unique, "Duplicate EHR patient IDs")
    require(set(ehr["patient_id"]) == set(cgm["patient_id"]), "EHR/CGM patient IDs do not match")
    require(ehr["patient_id"].str.fullmatch(r"SYN_\d{4,}").all(), "Non-synthetic patient ID format")
    require(not cgm.duplicated(["patient_id", "timestamp"]).any(), "Duplicate patient/time pairs")
    start = pd.Timestamp(config["start_date"])
    require((pd.to_datetime(ehr["baseline_date"]) < start).all(), "EHR baseline must precede CGM")
    require(provenance["config"] == config, "Data was generated with a different config; regenerate")
    require(provenance["generator_version"] == GENERATOR_VERSION, "Generator version mismatch")
    require(provenance["generator_sha256"] == sha256(ROOT / "generate_data.py"),
            "Generator changed after data generation; regenerate the data")
    for name in ("ehr.csv", "cgm.csv"):
        require(provenance["files"][name]["sha256"] == sha256(data_dir / name),
                f"Checksum mismatch for {name}; regenerate the data")
    for column, low, high in (("age_years", 25, 78), ("bmi_kg_m2", 19, 42),
                              ("diabetes_duration_years", 0.5, 25), ("hba1c_percent", 5.8, 10.8)):
        require(ehr[column].between(low, high).all(), f"{column} outside generator bounds")
    require(ehr["sex_at_birth"].isin(["female", "male"]).all(), "Unexpected sex category")
    require(ehr["type2_diabetes"].eq(1).all(), "All simulated patients should have Type 2 diabetes")
    require(ehr["hypertension"].isin([0, 1]).all(), "Invalid hypertension flag")
    require((ehr["diabetes_duration_years"] <= ehr["age_years"] - 18).all(), "Invalid diabetes duration")
    require(cgm["glucose_mg_dl"].between(65, 340).all(), "CGM outside simulator bounds")

    step = config["sampling_minutes"]
    n_samples = config["n_days"] * 1440 // step
    require(len(cgm) == config["n_patients"] * n_samples, "Wrong CGM row count")
    expected = pd.date_range(start, periods=n_samples, freq=f"{step}min")
    arrays = []
    for patient_id, group in cgm.groupby("patient_id", sort=True):
        require(pd.DatetimeIndex(group["timestamp"]).equals(expected),
                f"{patient_id}: timestamps must be complete, sorted and regular")
        arrays.append(group["glucose_mg_dl"].to_numpy(dtype=float))
    values = np.stack(arrays)
    lookback = config["lookback_minutes"] // step
    horizon = config["prediction_horizon_minutes"] // step
    stride = config["prediction_stride_minutes"] // step
    threshold = config["event_threshold_mg_dl"]
    split_reports, first = {}, 0
    for name, days_key in (("train", "train_days"), ("validation", "validation_days"), ("test", "test_days")):
        last = first + config[days_key] * 1440 // step
        block = values[:, first:last]
        anchors = np.arange(lookback, block.shape[1] - horizon, stride)
        require(len(anchors) > 0, f"{name}: no complete prediction windows")
        # Future values are inspected ONLY for labels/counts, never exported as inputs.
        future_max = np.max(np.stack([block[:, anchors + h] for h in range(1, horizon + 1)]), axis=0)
        eligible = block[:, anchors] < threshold
        events = (future_max >= threshold) & eligible
        count = int(eligible.sum())
        positives = int(events.sum())
        require(0 < positives < count, f"{name}: both event and non-event examples are required")
        split_reports[name] = {
            "start_inclusive": expected[first].isoformat(),
            "end_exclusive": (expected[last - 1] + pd.Timedelta(minutes=step)).isoformat(),
            "candidate_windows": int(eligible.size),
            "eligible_windows": count,
            "event_windows": positives,
            "non_event_windows": count - positives,
            "event_prevalence": round(positives / count, 6),
            "boundary_rule": "Full history and future horizon are inside this split",
        }
        first = last

    return {
        "status": "PASS",
        "ehr_rows": len(ehr),
        "cgm_rows": len(cgm),
        "n_patients": len(ehr),
        "samples_per_patient": n_samples,
        "glucose_summary_mg_dl": {
            "minimum": float(values.min()), "median": round(float(np.median(values)), 1),
            "maximum": float(values.max()),
        },
        "checks": ["schemas", "synthetic IDs", "linked patients", "no missing values",
                   "unique keys", "baseline predates stream", "value bounds", "regular timestamps",
                   "config and checksum integrity", "complete split windows", "both target classes"],
        "target": f"Current glucose < {threshold}; any reading >= {threshold} in next {config['prediction_horizon_minutes']} minutes",
        "splits": split_reports,
        "note": "Data integrity and target counts only; not model performance or clinical validation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "data_quality.json")
    args = parser.parse_args()
    config = load_config(args.config)
    report = check_data(config, args.data_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"DATA CHECKS PASSED: {report['ehr_rows']:,} EHR rows; {report['cgm_rows']:,} CGM rows.")
    for name, stats in report["splits"].items():
        print(f"{name}: {stats['eligible_windows']:,} eligible windows; "
              f"{stats['event_windows']:,} events ({stats['event_prevalence']:.1%})")
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
