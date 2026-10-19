"""Shared causal feature construction for training and patient replay."""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

FEATURE_VERSION = "1.0"
DYNAMIC_FEATURES = [
    "glucose_current", "glucose_lag_30m", "glucose_lag_60m", "glucose_lag_120m",
    "glucose_mean_2h", "glucose_std_2h", "glucose_min_2h", "glucose_max_2h",
    "glucose_slope_30m", "glucose_delta_2h", "hour_sin", "hour_cos",
]
EHR_FEATURES = ["age_years", "sex_female", "bmi_kg_m2", "hypertension",
                "diabetes_duration_years", "hba1c_percent"]
FUSED_FEATURES = DYNAMIC_FEATURES + EHR_FEATURES
BASELINE_FEATURES = ["glucose_current"]


def feature_steps(config):
    step = config["sampling_minutes"]
    if config["lookback_minutes"] != 120 or any(minutes % step for minutes in (30, 60, 120)):
        raise ValueError("This feature version requires a 120-minute lookback and sampling that divides 30 minutes")
    return step, 120 // step


def encode_ehr(row):
    return {
        "age_years": float(row["age_years"]),
        "sex_female": float(row["sex_at_birth"] == "female"),
        "bmi_kg_m2": float(row["bmi_kg_m2"]),
        "hypertension": float(row["hypertension"]),
        "diabetes_duration_years": float(row["diabetes_duration_years"]),
        "hba1c_percent": float(row["hba1c_percent"]),
    }


def summarize_history(windows, timestamps, config):
    """Windows end at their anchor. This function has no future-label input."""
    step, lookback = feature_steps(config)
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 2 or windows.shape[1] != lookback + 1:
        raise ValueError("Unexpected historical window shape")
    times = pd.DatetimeIndex(timestamps)
    if len(times) != len(windows) or not np.isfinite(windows).all():
        raise ValueError("Invalid history values or timestamps")
    current = windows[:, -1]
    lag30 = windows[:, -1 - 30 // step]
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60
    return pd.DataFrame({
        "glucose_current": current,
        "glucose_lag_30m": lag30,
        "glucose_lag_60m": windows[:, -1 - 60 // step],
        "glucose_lag_120m": windows[:, 0],
        "glucose_mean_2h": windows.mean(axis=1),
        "glucose_std_2h": windows.std(axis=1, ddof=0),
        "glucose_min_2h": windows.min(axis=1),
        "glucose_max_2h": windows.max(axis=1),
        "glucose_slope_30m": (current - lag30) / 30,
        "glucose_delta_2h": current - windows[:, 0],
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    })


def split_intervals(config):
    start = pd.Timestamp(config["start_date"])
    result = {}
    for split, key in (("train", "train_days"), ("validation", "validation_days"), ("test", "test_days")):
        end = start + pd.Timedelta(days=config[key])
        result[split] = (start, end)
        start = end
    return result


def build_training_tables(ehr, cgm, config):
    """Build split-contained historical inputs and separately computed future labels.

    Caller must run validate_data.check_data before using untrusted input tables.
    Patient ID and timestamps are metadata, never predictor columns.
    """
    step, lookback = feature_steps(config)
    horizon = config["prediction_horizon_minutes"] // step
    stride = config["prediction_stride_minutes"] // step
    intervals = split_intervals(config)
    parts = {name: [] for name in intervals}
    ehr_by_id = ehr.set_index("patient_id")
    for patient_id, patient in cgm.groupby("patient_id", sort=True):
        patient = patient.sort_values("timestamp")
        static = ehr_by_id.loc[patient_id]
        if pd.Timestamp(static["baseline_date"]) >= patient["timestamp"].min():
            raise ValueError("Baseline EHR must precede the dynamic stream")
        for split, (start, end) in intervals.items():
            block = patient.loc[(patient["timestamp"] >= start) & (patient["timestamp"] < end)]
            values = block["glucose_mg_dl"].to_numpy(dtype=float)
            times = pd.DatetimeIndex(block["timestamp"])
            anchors = np.arange(lookback, len(values) - horizon, stride)
            if len(anchors) == 0:
                raise ValueError(f"No complete windows for {patient_id} in {split}")
            eligible = values[anchors] < config["event_threshold_mg_dl"]
            anchors = anchors[eligible]
            if len(anchors) == 0:
                continue
            windows = sliding_window_view(values, lookback + 1)[anchors - lookback]
            frame = summarize_history(windows, times[anchors], config)
            for column, value in encode_ehr(static).items():
                frame[column] = value
            # The only future access occurs in target creation, after inputs exist.
            future_max = np.max(np.stack([values[anchors + h] for h in range(1, horizon + 1)]), axis=0)
            frame["target_event"] = (future_max >= config["event_threshold_mg_dl"]).astype(int)
            frame["patient_id"] = patient_id
            frame["timestamp"] = times[anchors]
            frame["history_start"] = times[anchors - lookback]
            frame["label_end"] = times[anchors + horizon]
            frame["split"] = split
            parts[split].append(frame)
    result = {}
    for split, frames in parts.items():
        if not frames:
            raise ValueError(f"No eligible observations in {split}")
        result[split] = pd.concat(frames, ignore_index=True)
        if not np.isfinite(result[split][FUSED_FEATURES].to_numpy()).all():
            raise ValueError("Nonfinite predictor found")
        if result[split]["target_event"].nunique() != 2:
            raise ValueError(f"Both target classes are required in {split}")
    return result


def make_feature_row(ehr_row, patient_cgm, as_of, config):
    """Training-identical inputs for ONE patient, excluding every reading after as_of."""
    step, lookback = feature_steps(config)
    anchor = pd.Timestamp(as_of)
    if pd.Timestamp(ehr_row["baseline_date"]) >= anchor:
        raise ValueError("EHR baseline is not available before this anchor")
    ids = patient_cgm["patient_id"].unique()
    if len(ids) != 1 or ids[0] != ehr_row["patient_id"]:
        raise ValueError("Pass CGM for exactly the selected EHR patient")
    observations = patient_cgm.copy()
    observations["timestamp"] = pd.to_datetime(observations["timestamp"])
    first = anchor - pd.Timedelta(minutes=config["lookback_minutes"])
    history = observations.loc[(observations["timestamp"] >= first) &
                               (observations["timestamp"] <= anchor)].sort_values("timestamp")
    expected = pd.date_range(first, anchor, freq=f"{step}min")
    if not pd.DatetimeIndex(history["timestamp"]).equals(expected):
        raise ValueError("Need a complete two-hour history ending at the selected reading")
    values = history["glucose_mg_dl"].to_numpy(dtype=float)
    if len(values) != lookback + 1:
        raise ValueError("Unexpected history length")
    frame = summarize_history(values.reshape(1, -1), [anchor], config)
    for column, value in encode_ehr(ehr_row).items():
        frame[column] = value
    return frame[FUSED_FEATURES]


def read_inputs(data_dir):
    data_dir = Path(data_dir)
    return (pd.read_csv(data_dir / "ehr.csv"),
            pd.read_csv(data_dir / "cgm.csv", parse_dates=["timestamp"]))
