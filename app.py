"""One-page doctor-facing synthetic patient twin. Run with Streamlit."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from features import read_inputs
from generate_data import load_config
from predict import check_bundle, score_patient
from validate_data import check_data

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="GlucoTwin | Patient workspace", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner=False)
def load_workspace(signature):
    """File signature invalidates the cache after data/model/code changes."""
    config = load_config(ROOT / "config.json")
    check_data(config, ROOT / "data")
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    check_bundle(bundle, config, ROOT / "data")
    ehr, cgm = read_inputs(ROOT / "data")
    report_path = ROOT / "reports" / "metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    if report and (report.get("runtime") != bundle["runtime"] or
                   report.get("protocol", {}).get("config") != config):
        report = None
    return config, bundle, ehr, cgm, report


def workspace_signature():
    files = ["config.json", "generate_data.py", "features.py", "predict.py", "validate_data.py",
             "data/ehr.csv", "data/cgm.csv", "data/provenance.json", "models/model.joblib"]
    if (ROOT / "reports" / "metrics.json").exists():
        files.append("reports/metrics.json")
    return tuple((name, (ROOT / name).stat().st_mtime_ns, (ROOT / name).stat().st_size) for name in files)


def hide_future():
    st.session_state["reveal_outcome"] = False


def move_reading(options, direction):
    current = options.index(st.session_state["replay_time"])
    st.session_state["replay_time"] = options[max(0, min(len(options) - 1, current + direction))]
    hide_future()


def reset_replay(default_time):
    st.session_state["replay_time"] = default_time
    hide_future()


def history_chart(history, as_of, config, future=None):
    """Only visible observations are passed to the figure; future is opt-in."""
    fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    fig.patch.set_facecolor("#F6F8FB")
    ax.set_facecolor("white")
    ax.plot(history["timestamp"], history["glucose_mg_dl"], color="#087F8C", linewidth=2,
            label="Available CGM")
    ax.axhline(config["event_threshold_mg_dl"], color="#B7791F", linestyle=":", linewidth=1.5,
               label=f"Prototype event threshold: {config['event_threshold_mg_dl']} mg/dL")
    ax.axvspan(as_of - pd.Timedelta(minutes=config["lookback_minutes"]), as_of,
               color="#087F8C", alpha=0.06, label="Two-hour feature window")
    ax.axvline(as_of, color="#475569", linestyle="--", linewidth=1, label="Twin time")
    if future is not None and len(future):
        joined = pd.concat([history.tail(1), future], ignore_index=True)
        ax.plot(joined["timestamp"], joined["glucose_mg_dl"], color="#C97316", linestyle="--",
                linewidth=2, label="Observed later (revealed; not a forecast)")
    ax.set_ylabel("Glucose (mg/dL)", color="#153047")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(axis="y", alpha=0.15)
    ax.tick_params(colors="#475569")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    return fig


def main():
    st.markdown("""<style>
        .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {background: white; border: 1px solid #DEE6EC;
            padding: 14px 16px; border-radius: 10px;}
    </style>""", unsafe_allow_html=True)
    st.title("GlucoTwin")
    st.caption("DOCTOR WORKSPACE  ·  TYPE 2 DIABETES  ·  SYNTHETIC PATIENT REPLAY")
    st.caption("Demonstration only. No established clinical validity; not for diagnosis or treatment.")
    try:
        with st.spinner("Loading the patient twins and verified model…"):
            config, bundle, ehr, cgm, report = load_workspace(workspace_signature())
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        st.error("The workspace is not ready. Generate the data and train the model before opening the dashboard.")
        st.code("python generate_data.py\npython train.py\npython verify_model.py", language="powershell")
        with st.expander("Setup details"):
            st.write(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Patient twin")
        patient_id = st.selectbox("Synthetic patient", ehr["patient_id"].tolist(), key="patient_id", on_change=hide_future)
        static = ehr.loc[ehr["patient_id"] == patient_id].iloc[0]
        st.caption("Baseline EHR")
        fields = {
            "Age": f"{int(static['age_years'])} years",
            "Sex at birth": str(static["sex_at_birth"]).title(),
            "BMI": f"{static['bmi_kg_m2']:.1f} kg/m²",
            "HbA1c": f"{static['hba1c_percent']:.1f}%",
            "Diabetes duration": f"{static['diabetes_duration_years']:.1f} years",
            "Hypertension": "Yes" if static["hypertension"] else "No",
        }
        st.dataframe(pd.DataFrame({"Field": fields.keys(), "Value": fields.values()}), hide_index=True)
        st.caption(f"Existing diagnosis: Type 2 diabetes. EHR recorded {static['baseline_date']}, before the CGM stream.")
        st.divider()
        st.caption("80 synthetic twins by default. No real patient records or live devices are connected.")

    patient = cgm.loc[cgm["patient_id"] == patient_id].copy()
    first = patient["timestamp"].min() + pd.Timedelta(minutes=config["lookback_minutes"])
    options = pd.DatetimeIndex(patient.loc[patient["timestamp"] >= first, "timestamp"]).to_pydatetime().tolist()
    desired_default = pd.Timestamp(config["start_date"]) + pd.Timedelta(days=config["train_days"], hours=12)
    default_time = min(options, key=lambda value: abs(pd.Timestamp(value) - desired_default))
    if "replay_time" not in st.session_state or st.session_state["replay_time"] not in options:
        st.session_state["replay_time"] = default_time
    if "reveal_outcome" not in st.session_state:
        st.session_state["reveal_outcome"] = False

    st.subheader("Replay the patient")
    selected = st.select_slider("Twin time", options=options, key="replay_time",
                                format_func=lambda value: value.strftime("%d %b %Y · %H:%M"), on_change=hide_future)
    index = options.index(selected)
    previous, following, reset, _ = st.columns([1, 1, 1, 2])
    previous.button("Previous reading", key="previous_reading", on_click=move_reading,
                    args=(options, -1), disabled=index == 0)
    following.button("Next reading", key="next_reading", on_click=move_reading,
                     args=(options, 1), disabled=index == len(options) - 1)
    reset.button("Reset replay", key="reset_replay", on_click=reset_replay, args=(default_time,))
    st.caption("All dates are simulated, in Asia/Kolkata time. Each reading advances the twin by 15 minutes in the default dataset.")
    as_of = pd.Timestamp(selected)
    prediction = score_patient(bundle, static, patient, as_of)
    available = patient.loc[patient["timestamp"] <= as_of]
    history = available.loc[available["timestamp"] >= as_of - pd.Timedelta(hours=8)]

    st.markdown(f"### {patient_id} · {as_of.strftime('%d %b %Y, %H:%M')}")
    current, score, cutoff, horizon = st.columns(4)
    current.metric("Current glucose", f"{prediction['current_glucose_mg_dl']:.1f} mg/dL")
    score.metric("Next-hour model score", "Not scored" if prediction["risk_score"] is None else f"{prediction['risk_score']:.1%}",
                 help="Raw model-estimated event probability on synthetic data. Not independently calibrated or a clinical risk estimate.")
    cutoff.metric("Prototype alert cutoff", f"{prediction['alert_threshold']:.1%}",
                  help="Chosen on validation data to maximize window F1, then fixed before test evaluation.")
    horizon.metric("Prediction horizon", f"{config['prediction_horizon_minutes']} min")
    if not prediction["eligible_for_incident_event_prediction"]:
        st.info("Current glucose is already at or above the prototype event threshold. New-onset event risk is not scored.")
    elif prediction["alert"]:
        st.warning("Prototype alert for review: the next-hour model score meets the validation-selected cutoff.")
    else:
        st.success("Below the prototype alert cutoff. Missed events remain possible.")
    st.caption(f"Target: any reading ≥{config['event_threshold_mg_dl']} mg/dL during the next "
               f"{config['prediction_horizon_minutes']} minutes, when current glucose is below that threshold.")

    reveal = st.checkbox("Reveal observed next hour (retrospective demo only)", key="reveal_outcome")
    future = None
    if reveal:
        end = as_of + pd.Timedelta(minutes=config["prediction_horizon_minutes"])
        future = patient.loc[(patient["timestamp"] > as_of) & (patient["timestamp"] <= end)]
    st.subheader("Glucose history")
    figure = history_chart(history, as_of, config, future)
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
    n_inputs = config["lookback_minutes"] // config["sampling_minutes"] + 1
    st.caption(f"Twin inputs: baseline EHR + {n_inputs} CGM readings from "
               f"{(as_of - pd.Timedelta(minutes=config['lookback_minutes'])).strftime('%H:%M')} to {as_of.strftime('%H:%M')}. "
               "The teal shaded window supplies the historical features. Future observations never enter inference.")
    if reveal:
        st.caption("Orange dashed values are observations revealed after the prediction. They are not predicted glucose values.")
        expected_followup = config["prediction_horizon_minutes"] // config["sampling_minutes"]
        if len(future) != expected_followup:
            st.info("The dataset ends before the full follow-up window. A complete next-hour outcome is unavailable.")
        elif not prediction["eligible_for_incident_event_prediction"]:
            st.info("Follow-up is shown, but the new-onset target is not applicable because current glucose was already above threshold.")
        else:
            observed_event = bool((future["glucose_mg_dl"] >= config["event_threshold_mg_dl"]).any())
            st.info(f"Observed next hour: {'event present' if observed_event else 'no threshold event'}. "
                    f"Maximum observed glucose: {future['glucose_mg_dl'].max():.1f} mg/dL. "
                    "This reveal does not change the saved prediction.")

    with st.expander("Model evaluation and limitations"):
        if report:
            labels = {"baseline_current_glucose": "Current glucose baseline",
                      "wearable_only_hgb": "CGM history + clock",
                      "fused_hgb": "CGM history + clock + EHR"}
            summary = []
            for name, result in report["models"].items():
                m = result["test"]
                summary.append({"Model": labels.get(name, name), "ROC-AUC": f"{m['roc_auc']:.3f}",
                                "Average precision": f"{m['average_precision']:.3f}",
                                "Precision": f"{m['precision']:.1%}", "Recall": f"{m['recall']:.1%}"})
            st.dataframe(pd.DataFrame(summary), hide_index=True)
        else:
            st.caption("Matching evaluation report unavailable. Regenerate it by training locally.")
        st.write("These are synthetic test-window results. The same patients appear in different time periods; "
                 "unseen-patient and real-world performance are untested. Overlapping windows are correlated.")
        st.write("EHR associations and glucose patterns come from simulation assumptions. Scores are not independently "
                 "calibrated, and this alert cutoff does not represent clinical costs or a treatment policy.")
        st.caption("The twin updates its patient state during replay; the trained model remains fixed. No online retraining or treatment simulation is claimed.")

    # Internal snapshot for automated UI verification; no hidden future values stored.
    st.session_state["twin_snapshot"] = {
        **prediction, "future_revealed": bool(reveal),
        "visible_future_points": len(future) if future is not None else 0,
        "chart_last_timestamp": (future["timestamp"].max() if future is not None and len(future)
                                 else history["timestamp"].max()).isoformat(),
        "history_last_timestamp": history["timestamp"].max().isoformat(),
    }


if __name__ == "__main__":
    main()
