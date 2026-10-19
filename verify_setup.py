"""Verify Step 1 dependencies and configuration; no data or models are created."""
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "streamlit": "streamlit",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "reportlab": "reportlab",
}


def main():
    if sys.version_info < (3, 10):
        raise SystemExit("Use Python 3.10 or newer; Python 3.12 is a practical choice.")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if config["n_days"] != sum(config[k] for k in ("train_days", "validation_days", "test_days")):
        raise SystemExit("Configuration error: split lengths must sum to n_days.")
    for key in ("lookback_minutes", "prediction_horizon_minutes", "prediction_stride_minutes"):
        if config[key] % config["sampling_minutes"]:
            raise SystemExit(f"Configuration error: {key} must be a sampling interval multiple.")

    versions = {}
    missing = []
    for distribution, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            versions[distribution] = importlib.metadata.version(distribution)
            print(f"OK  {distribution}: {versions[distribution]}")
        except Exception as exc:
            missing.append(distribution)
            print(f"FAIL  {distribution}: {exc}")
    if missing:
        raise SystemExit("Setup incomplete. Install requirements and rerun this check.")

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": versions,
        "config": config,
    }
    (ROOT / "setup_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nSETUP OK - ready for Step 2: synthetic data.")


if __name__ == "__main__":
    main()
