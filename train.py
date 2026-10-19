"""Train fixed baseline, wearable-only and fused models; evaluate chronologically."""
import argparse
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from features import (BASELINE_FEATURES, DYNAMIC_FEATURES, EHR_FEATURES, FEATURE_VERSION,
                      FUSED_FEATURES, build_training_tables, read_inputs, split_intervals)
from generate_data import load_config, sha256
from validate_data import check_data

ROOT = Path(__file__).resolve().parent
FINAL_MODEL = "fused_hgb"
HGB_SETTINGS = {"learning_rate": 0.08, "max_iter": 120, "max_leaf_nodes": 15,
                "min_samples_leaf": 30, "l2_regularization": 2.0,
                "early_stopping": False, "categorical_features": None}


def choose_threshold(y, probabilities):
    """Maximize validation window F1; break exact ties toward a higher threshold."""
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = np.flatnonzero(scores == scores.max())[-1]
    return float(thresholds[index])


def evaluate(y, probabilities, threshold):
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "n_windows": int(len(y)), "n_events": int(np.sum(y)),
        "event_prevalence": float(np.mean(y)), "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "alert_fraction": float(np.mean(predicted)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_results(test, probabilities, scores, output):
    y = test["target_event"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    names = {"baseline_current_glucose": "Current glucose baseline",
             "wearable_only_hgb": "CGM history + clock",
             "fused_hgb": "CGM history + clock + EHR"}
    for key, p in probabilities.items():
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(recall, precision, label=f"{names[key]} (AP={scores[key]['test']['average_precision']:.3f})")
        fpr, tpr, _ = roc_curve(y, p)
        axes[1].plot(fpr, tpr, label=f"{names[key]} (AUC={scores[key]['test']['roc_auc']:.3f})")
    axes[0].axhline(y.mean(), color="gray", linestyle="--", label=f"Prevalence: {y.mean():.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Synthetic test: precision-recall", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="Synthetic test: ROC", xlim=(0, 1), ylim=(0, 1.02))
    for ax in axes:
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.2)
    fig.savefig(output / "model_comparison.png", dpi=160)
    plt.close(fig)
    cm = scores[FINAL_MODEL]["test"]["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5.3, 4.3), constrained_layout=True)
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=15,
                color="white" if value > matrix.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No alert", "Alert"],
           yticklabels=["No event", "Event"], xlabel="Prediction", ylabel="Observed future label",
           title="Fused model: synthetic test windows")
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def write_results(report, output):
    lines = ["# GlucoTwin model results", "", "Synthetic data only; not clinical validation.", "",
             "## Frozen evaluation protocol", "",
             "- Baseline: StandardScaler + logistic regression on current glucose only.",
             "- Main model: fixed histogram gradient boosting on CGM history, clock and baseline EHR.",
             "- Ablation: identical boosting settings with EHR removed.",
             "- No hyperparameter search; random internal early stopping is disabled.",
             "- Train on days 1-13; choose each threshold on validation days 14-17; test on days 18-21.",
             "- Threshold rule: maximize validation window F1, with higher threshold breaking exact ties.",
             "- Main model was selected by design, before examining test results.", "",
             "## Test results", "",
             "| Model | ROC-AUC | Average precision | Precision | Recall | F1 | Threshold |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, result in report["models"].items():
        m = result["test"]
        lines.append(f"| {key} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} | "
                     f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['threshold']:.4f} |")
    main = report["models"][FINAL_MODEL]["test"]
    wear = report["models"]["wearable_only_hgb"]["test"]
    cm = main["confusion_matrix"]
    lines += ["", f"Test windows: {main['n_windows']:,}; event windows: {main['n_events']:,}; "
              f"prevalence: {main['event_prevalence']:.2%}.", "",
              f"Fused confusion counts: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}.", "",
              f"Fused minus wearable-only average precision: {main['average_precision'] - wear['average_precision']:+.4f}. "
              "This is a descriptive simulation result, not proof that EHR improves real-world outcomes.", "",
              "## Features and interpretation", "",
              "Dynamic: " + ", ".join(DYNAMIC_FEATURES) + ".", "",
              "Baseline EHR: " + ", ".join(EHR_FEATURES) + ".", "",
              "Patient ID, dates, target labels, future glucose and hidden simulator parameters are excluded.", "",
              "Model scores have not undergone separate probability calibration; the alert threshold is a "
              "prototype F1 operating point, not a clinical decision threshold.", "",
              "## Limitations", ""]
    lines += ["- " + item for item in report["limitations"]]
    lines += ["", "## Reproduce", "", "```powershell", ".\\.venv\\Scripts\\python.exe generate_data.py",
              ".\\.venv\\Scripts\\python.exe train.py", ".\\.venv\\Scripts\\python.exe verify_model.py", "```", "",
              "The same code, data and dependency versions are required. Retraining this fixed protocol is "
              "reproduction; do not revise it in response to test scores and still call that test independent.", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")


def run(config, data_dir, model_dir, reports_dir):
    print("Checking synthetic inputs...", flush=True)
    quality = check_data(config, data_dir)
    ehr, cgm = read_inputs(data_dir)
    tables = build_training_tables(ehr, cgm, config)
    for split, frame in tables.items():
        expected = quality["splits"][split]
        if len(frame) != expected["eligible_windows"] or int(frame["target_event"].sum()) != expected["event_windows"]:
            raise ValueError("Feature table target counts disagree with independent data validation")
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(HGB_SETTINGS, random_state=config["seed"])
    protocol = {
        "main_model_chosen_before_test": FINAL_MODEL,
        "config": config, "feature_version": FEATURE_VERSION,
        "feature_groups": {"dynamic": DYNAMIC_FEATURES, "ehr": EHR_FEATURES},
        "boosting_settings": settings,
        "baseline_settings": {"estimator": "StandardScaler + LogisticRegression", "max_iter": 1000,
                              "random_state": config["seed"]},
        "threshold_rule": "Maximum validation window F1; exact ties choose higher threshold",
        "thread_limit": 4,
        "no_test_tuning": True,
    }
    (reports_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    specs = {
        "baseline_current_glucose": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=config["seed"])), BASELINE_FEATURES),
        "wearable_only_hgb": (HistGradientBoostingClassifier(**settings), DYNAMIC_FEATURES),
        FINAL_MODEL: (HistGradientBoostingClassifier(**settings), FUSED_FEATURES),
    }
    trained, scores = {}, {}
    train, validation, test = (tables[k] for k in ("train", "validation", "test"))
    for name, (estimator, columns) in specs.items():
        print(f"Training {name}: {len(train):,} windows, {len(columns)} features...", flush=True)
        estimator.fit(train[columns], train["target_event"])
        p_val = estimator.predict_proba(validation[columns])[:, 1]
        threshold = choose_threshold(validation["target_event"], p_val)
        scores[name] = {"features": columns, "validation": evaluate(validation["target_event"], p_val, threshold)}
        trained[name] = (estimator, columns, threshold)

    print("All models and thresholds frozen. Evaluating synthetic test once...", flush=True)
    probabilities = {}
    for name, (estimator, columns, threshold) in trained.items():
        p_test = estimator.predict_proba(test[columns])[:, 1]
        if not np.isfinite(p_test).all():
            raise ValueError("Nonfinite prediction")
        probabilities[name] = p_test
        scores[name]["test"] = evaluate(test["target_event"], p_test, threshold)

    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}
    report = {
        "status": "TRAINED_AND_EVALUATED", "main_model": FINAL_MODEL,
        "runtime": versions, "protocol": protocol,
        "split_counts": quality["splits"], "models": scores,
        "limitations": [
            "Fully synthetic data; good metrics do not imply clinical validity or Indian population representativeness.",
            "Train and test include the same patients at different times; unseen-patient generalization is untested.",
            "Overlapping prediction windows and repeated patients are correlated; counts are not independent clinical events.",
            "Window recall is not episode-level recall, and repeated alerts are not deduplicated.",
            "EHR-CGM associations are engineered simulator assumptions, not established biological effect estimates.",
            "Complete regular CGM omits realistic dropout, missingness, treatment changes and device bias.",
            "F1 threshold selection does not represent clinical costs, safe alert burden or a treatment policy.",
            "Probabilities are not independently calibrated; no diagnosis, treatment or medication advice is provided.",
        ],
    }
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (reports_dir / "data_quality.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    plot_results(test, probabilities, scores, reports_dir)
    write_results(report, reports_dir)

    # The main model choice never depends on the test ranking.
    estimator, columns, threshold = trained[FINAL_MODEL]
    bundle = {
        "artifact_version": "1.0", "model_name": FINAL_MODEL, "estimator": estimator,
        "feature_columns": columns, "feature_version": FEATURE_VERSION,
        "features_sha256": sha256(ROOT / "features.py"),
        "threshold": threshold, "config": config, "runtime": versions,
        "source_hashes": {name: sha256(data_dir / name) for name in ("ehr.csv", "cgm.csv")},
        "training_period": quality["splits"]["train"],
        "validation_period": quality["splits"]["validation"],
        "intended_use": "Synthetic prototype only; not clinically validated",
    }
    joblib.dump(bundle, model_dir / "model.joblib", compress=3)
    loaded = joblib.load(model_dir / "model.joblib")
    expected = estimator.predict_proba(validation[columns].iloc[:20])[:, 1]
    restored = loaded["estimator"].predict_proba(validation[columns].iloc[:20])[:, 1]
    np.testing.assert_array_equal(expected, restored)
    sample = pd.concat([frame.head(5) for frame in tables.values()], ignore_index=True)
    sample.to_csv(reports_dir / "feature_examples.csv", index=False, lineterminator="\n")
    print("\nTEST RESULTS (synthetic windows only)", flush=True)
    for name, result in scores.items():
        m = result["test"]
        print(f"{name}: AUC={m['roc_auc']:.4f}; AP={m['average_precision']:.4f}; "
              f"precision={m['precision']:.4f}; recall={m['recall']:.4f}; F1={m['f1']:.4f}", flush=True)
    print(f"\nMODEL SAVED: {model_dir / 'model.joblib'}", flush=True)
    print("Next: run verify_model.py, then build the dashboard in Step 4.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    config = load_config(args.config)
    # Keeps OpenMP/BLAS overhead manageable on ordinary laptops and large hosts.
    with threadpool_limits(limits=4):
        run(config, args.data_dir, args.model_dir, args.reports_dir)


if __name__ == "__main__":
    main()
