# Step 4: doctor dashboard

Merge this update into the existing project. It adds app.py, verify_dashboard.py,
.streamlit/config.toml and these instructions. Your trained model and source data are unchanged.

## Run in Windows PowerShell

```powershell
.\.venv\Scripts\python.exe verify_dashboard.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501 if the browser does not open automatically. Keep the terminal running.
Stop the server with Ctrl+C. No GPU, external hosting or new dependency is required.

## Demonstrate

1. Select a synthetic patient in the sidebar and show their baseline EHR.
2. Move the Twin time slider or use Next reading. The CGM history, score and alert update.
3. Show that only observations at or before the selected time appear in the default chart.
4. Optionally check Reveal observed next hour. The orange dashed trace shows actual later
   synthetic observations, NOT a predicted glucose trajectory. The score remains unchanged.
5. Change time or patient: retrospective follow-up hides again automatically.
6. Open Model evaluation and limitations for the baseline, wearable-only and fused comparison.

Run verify_dashboard.py to generate reports/dashboard_checks.json. Its validation_demo_cases
list gives patient IDs/timestamps for an alert, below-cutoff state and already-high state.
Examples are selected using current values and model scores, not future outcome labels.
Use your LOCAL examples after local training; the reference examples may differ.

The chart displays up to eight hours of available history. Only the most recent two hours
(nine readings by default) enter the feature vector. Replay advances every 15 minutes, while
reported evaluation samples anchors every 30 minutes. No online model retraining is performed.

## Verification

The optional automated check uses Streamlit's built-in AppTest. It exercises patient/time
navigation, both alert states, the already-high state, future reveal and incomplete follow-up.
It does not require pytest, a browser driver or separate test dependencies.

## Commit

```powershell
git add app.py verify_dashboard.py .streamlit/config.toml docs/STEP4.md reports/dashboard_checks.json
git commit -m "Add doctor-facing patient twin dashboard with verified replay"
git push
```

## Troubleshooting

- Missing Streamlit: install requirements.txt using the existing .venv Python.
- Missing/stale model or data: run generate_data.py, train.py and verify_model.py, then restart.
- Port in use: stop the old app with Ctrl+C or run with --server.port 8502 and open that port.
- Changes not visible: stop and restart the Streamlit process. Data/model cache keys track file changes.
- New-onset risk says Not scored: current glucose is already above the prototype threshold; this is intended.
- Outcome unavailable at the end: no complete future hour remains in the synthetic dataset; this is intended.
- AppTest timeout on a slow machine: increase default_timeout=40 in verify_dashboard.py to 90 and rerun.
- AppTest may print a missing ScriptRunContext warning before the checks. If it finishes with
  DASHBOARD CHECKS PASSED, that bare-mode warning is harmless.

Do not describe below-cutoff predictions as safe or diagnoses. No real-patient clinical validity is established.
Next: final README, architecture PDF and presentation PDF. The video is handled by the user.
