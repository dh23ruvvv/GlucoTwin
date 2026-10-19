"""Generate fully synthetic baseline EHR and 15-minute CGM. No model training.

The simulator is illustrative, not a physiological or clinically calibrated model.
The glucose event threshold is NOT used to generate the glucose trajectory.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GENERATOR_VERSION = "1.0"


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for key in ("n_patients", "n_days", "sampling_minutes", "prediction_stride_minutes",
                "lookback_minutes", "prediction_horizon_minutes", "train_days",
                "validation_days", "test_days"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if 1440 % config["sampling_minutes"]:
        raise ValueError("sampling_minutes must divide 1440 exactly")
    if sum(config[k] for k in ("train_days", "validation_days", "test_days")) != config["n_days"]:
        raise ValueError("Split day counts must sum to n_days")
    for key in ("lookback_minutes", "prediction_horizon_minutes", "prediction_stride_minutes"):
        if config[key] % config["sampling_minutes"]:
            raise ValueError(f"{key} must be divisible by sampling_minutes")
    if min(config[k] for k in ("train_days", "validation_days", "test_days")) * 1440 <= (
        config["lookback_minutes"] + config["prediction_horizon_minutes"]
    ):
        raise ValueError("Every split needs room for history and future target windows")
    if not np.isfinite(config["event_threshold_mg_dl"]) or not 65 < config["event_threshold_mg_dl"] < 340:
        raise ValueError("event_threshold_mg_dl must be between the simulator bounds 65 and 340")
    start = pd.Timestamp(config["start_date"])
    if start.tzinfo is not None or start != start.normalize():
        raise ValueError("start_date must be a timezone-naive calendar date, e.g. 2026-01-01")
    return config


def add_response(signal, hours, onset, amplitude, peak_hours):
    """Causal rise/decay response: no effect at any time before its onset."""
    elapsed = hours - onset
    mask = (elapsed >= 0) & (elapsed <= 6 * peak_hours)
    scaled = elapsed[mask] / peak_hours
    signal[mask] += amplitude * scaled * np.exp(1 - scaled)


def generate(config):
    start = pd.Timestamp(config["start_date"])
    samples = config["n_days"] * 1440 // config["sampling_minutes"]
    times = pd.date_range(start, periods=samples, freq=f"{config['sampling_minutes']}min")
    hours = np.arange(samples, dtype=float) * config["sampling_minutes"] / 60
    static_seed, dynamic_seed = np.random.SeedSequence(config["seed"]).spawn(2)
    static_rng = np.random.default_rng(static_seed)
    patient_seeds = dynamic_seed.spawn(config["n_patients"])
    ehr_rows, streams = [], []

    for index, patient_seed in enumerate(patient_seeds):
        patient_id = f"SYN_{index + 1:04d}"
        # These variables are drawn BEFORE any future CGM exists. Severity is a
        # simulation-only latent variable and is never exported as a feature.
        severity = float(np.clip(static_rng.normal(), -1.7, 2.2))
        age = int(np.clip(static_rng.normal(51, 10), 25, 78))
        sex = str(static_rng.choice(["female", "male"]))
        bmi = round(float(np.clip(static_rng.normal(27 + 1.7 * severity, 3), 19, 42)), 1)
        duration = round(float(np.clip(static_rng.gamma(2, 3), 0.5, min(25, age - 18))), 1)
        hba1c = round(float(np.clip(7.3 + 0.65 * severity + static_rng.normal(0, 0.35), 5.8, 10.8)), 1)
        hypertension = int(static_rng.random() < np.clip(0.25 + (age - 40) * 0.008, 0.1, 0.65))
        ehr_rows.append({
            "patient_id": patient_id,
            "baseline_date": (start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "age_years": age,
            "sex_at_birth": sex,
            "bmi_kg_m2": bmi,
            "type2_diabetes": 1,
            "hypertension": hypertension,
            "diabetes_duration_years": duration,
            "hba1c_percent": hba1c,
        })

        rng = np.random.default_rng(patient_seed)
        basal = 117 + 12 * severity + rng.normal(0, 5)
        meal_sensitivity = float(np.clip(42 + 8 * severity + 0.45 * (bmi - 27) + rng.normal(0, 4), 18, 70))
        peak_hours = float(rng.uniform(0.65, 1.1))
        glucose = basal + 5 * np.cos(2 * np.pi * (hours - 6) / 24)
        # One preceding day warms up decaying event responses at the start.
        for day in range(-1, config["n_days"]):
            for meal_hour, meal_scale in ((8, 0.9), (13.5, 1.1), (20, 1.15)):
                if rng.random() < 0.08:
                    continue
                onset = day * 24 + meal_hour + rng.normal(0, 0.65)
                amplitude = meal_sensitivity * meal_scale * rng.uniform(0.65, 1.5)
                add_response(glucose, hours, onset, amplitude, peak_hours * rng.uniform(0.85, 1.15))
            if rng.random() < 0.4:
                add_response(glucose, hours, day * 24 + rng.uniform(10, 23),
                             rng.uniform(12, 30), rng.uniform(0.4, 0.9))
            if rng.random() < 0.6:
                add_response(glucose, hours, day * 24 + rng.uniform(7, 21),
                             -rng.uniform(8, 22), rng.uniform(0.6, 1.4))
            if rng.random() < 0.18:
                add_response(glucose, hours, day * 24 + rng.uniform(6, 22),
                             rng.uniform(10, 25), rng.uniform(1, 2))

        # Continuous stochastic variation; no explicit train/validation/test regime.
        drift = np.empty(samples)
        drift[0] = rng.normal(0, 4)
        innovations = rng.normal(0, 2.2, samples)
        for i in range(1, samples):
            drift[i] = 0.88 * drift[i - 1] + innovations[i]
        glucose = np.round(np.clip(glucose + drift + rng.normal(0, 2, samples), 65, 340), 1)
        streams.append(pd.DataFrame({
            "patient_id": patient_id,
            "timestamp": times,
            "glucose_mg_dl": glucose,
        }))

    return pd.DataFrame(ehr_rows), pd.concat(streams, ignore_index=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    config = load_config(args.config)
    ehr, cgm = generate(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ehr_path, cgm_path = args.output_dir / "ehr.csv", args.output_dir / "cgm.csv"
    ehr.to_csv(ehr_path, index=False, float_format="%.1f", lineterminator="\n")
    cgm.to_csv(cgm_path, index=False, float_format="%.1f",
               date_format="%Y-%m-%dT%H:%M:%S", lineterminator="\n")
    provenance = {
        "dataset_name": "GlucoTwin synthetic demonstration data (project-generated)",
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": sha256(Path(__file__)),
        "source": "Generated entirely by generate_data.py; no external dataset or real records",
        "license": "MIT; see repository LICENSE and data/DATA_LICENSE.md",
        "timezone": "Asia/Kolkata; timestamps stored as local time without an offset",
        "units": {"glucose_mg_dl": "mg/dL", "hba1c_percent": "%", "bmi_kg_m2": "kg/m^2"},
        "config": config,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "files": {"ehr.csv": {"rows": len(ehr), "sha256": sha256(ehr_path)},
                  "cgm.csv": {"rows": len(cgm), "sha256": sha256(cgm_path)}},
        "limitations": [
            "Illustrative stochastic rules; not physiological calibration or an Indian population sample",
            "Baseline EHR generated before CGM; HbA1c is not calculated from future CGM",
            "Meal timing, severity and other hidden simulator factors are not exported",
            "Complete regular CGM; no missingness, device dropout or treatment policy simulation",
            "Binary simulated sex categories are a simplification, not population representation",
            "Synthetic event labels do not establish clinical validity or legal compliance",
        ],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(ehr):,} EHR rows and {len(cgm):,} CGM rows.")
    print(f"Output: {args.output_dir.resolve()}")
    print("Next: run validate_data.py. This step does not train a model.")


if __name__ == "__main__":
    main()
