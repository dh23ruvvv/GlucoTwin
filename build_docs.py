"""Build submission documentation from LOCAL metrics and installed dependencies.

Run after generation, training, model verification and dashboard verification.
Writes README.md, THIRD_PARTY_NOTICES.md, PDFs and editable Markdown sources.
"""
import hashlib
import importlib.metadata as metadata
import json
import re
import shutil
from collections import deque
from html import escape
from pathlib import Path

import joblib
import pandas as pd
import reportlab
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from generate_data import load_config
from predict import check_bundle

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
W, H = 960, 540
NAVY = "#153047"
TEAL = "#087F8C"
GRAY = "#526779"
LIGHT = "#EDF5F6"
AMBER = "#A96012"
FONT = "GlucoSans"


def register_fonts():
    # ReportLab bundles these fonts on Windows and Linux. Embed them in PDFs so
    # viewing does not depend on system Helvetica substitution.
    directory = Path(reportlab.__file__).resolve().parent / "fonts"
    pdfmetrics.registerFont(TTFont(FONT, str(directory / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont(FONT + "-Bold", str(directory / "VeraBd.ttf")))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def color(value):
    return colors.HexColor(value)


def text(c, value, x, y, width, size=17, leading=None, bold=False, fill=NAVY, max_height=500):
    # y is the top of a text block; only this function wraps body text.
    style = ParagraphStyle("body", fontName=FONT + "-Bold" if bold else FONT,
                           fontSize=size, leading=leading or size * 1.35, textColor=color(fill))
    paragraph = Paragraph(escape(str(value)).replace("\n", "<br/>"), style)
    _, height = paragraph.wrap(width, max_height)
    if height > max_height + 0.1 or y - height < 31:
        raise ValueError(f"Text overflow: {str(value)[:80]}")
    paragraph.drawOn(c, x, y - height)
    return y - height


def page(c, title, info, number=None, subtitle=None):
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(color(TEAL))
    c.rect(40, 504, 38, 5, fill=1, stroke=0)
    text(c, title, 40, 487, 880, size=30, bold=True, max_height=75)
    if subtitle:
        text(c, subtitle, 40, 438, 880, size=12, fill=GRAY, max_height=38)
    c.setStrokeColor(color("#DAE4EA"))
    c.line(40, 35, 920, 35)
    c.setFillColor(color(GRAY))
    c.setFont(FONT, 9)
    c.drawString(40, 19, f"{info['team_leader']} | {info['college']} | GlucoTwin")
    if number is not None:
        c.drawRightString(920, 19, f"{number} / 9")


def bullet_list(c, items, x, y, width, size=18, gap=18):
    for item in items:
        c.setFillColor(color(TEAL))
        c.circle(x + 3, y - size * 0.55, 2.3, fill=1, stroke=0)
        y = text(c, item, x + 17, y, width - 17, size=size) - gap
    return y


def box(c, x, y, width, height, title, lines, fill=LIGHT):
    c.setFillColor(color(fill))
    c.setStrokeColor(color("#C6DFE1"))
    c.roundRect(x, y, width, height, 7, fill=1, stroke=1)
    bottom = text(c, title, x + 12, y + height - 12, width - 24, size=11.5, bold=True, max_height=height - 18)
    text(c, "\n".join(lines), x + 12, bottom - 7, width - 24, size=11,
         leading=14, max_height=bottom - y - 10)


def arrow(c, points, fill=TEAL):
    c.setStrokeColor(color(fill))
    c.setFillColor(color(fill))
    c.setLineWidth(1.5)
    p = c.beginPath()
    p.moveTo(*points[0])
    for point in points[1:]:
        p.lineTo(*point)
    c.drawPath(p)
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    p = c.beginPath()
    p.moveTo(x2, y2)
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 > x1 else -1
        p.lineTo(x2 - sign * 7, y2 + 3.5)
        p.lineTo(x2 - sign * 7, y2 - 3.5)
    else:
        sign = 1 if y2 > y1 else -1
        p.lineTo(x2 + 3.5, y2 - sign * 7)
        p.lineTo(x2 - 3.5, y2 - sign * 7)
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def architecture(c, info, number=None):
    page(c, "System architecture", info, number,
         "Two linked streams feed the same causal feature code during training and patient replay")
    text(c, "OFFLINE TRAINING AND EVALUATION", 285, 413, 625, size=11, bold=True, fill=TEAL)
    box(c, 40, 327, 180, 82, "Synthetic baseline EHR", ["Demographics, diagnoses", "BMI, HbA1c, duration"])
    box(c, 40, 218, 180, 82, "Synthetic CGM stream", ["Patient ID + timestamp", "Glucose every 15 minutes"])
    box(c, 285, 277, 220, 105, "Join + causal features", ["Linked by patient ID", "Inputs available at time t", "12 dynamic + 6 EHR inputs"])
    box(c, 555, 277, 150, 105, "Time split", ["Train: 13 days", "Validation: 4 days", "Test: 4 days"])
    box(c, 755, 277, 165, 105, "Train and evaluate", ["Histogram boosting", "Threshold on validation", "Baseline + EHR ablation"])
    arrow(c, [(220, 368), (250, 368), (250, 330), (285, 330)])
    arrow(c, [(220, 259), (250, 259), (250, 330), (285, 330)])
    arrow(c, [(505, 330), (555, 330)])
    arrow(c, [(705, 330), (755, 330)])
    text(c, "PATIENT REPLAY", 285, 211, 220, size=11, bold=True, fill=TEAL)
    box(c, 285, 103, 220, 88, "Snapshot at selected time", ["Baseline EHR + past CGM", "Same features.py code"])
    box(c, 555, 103, 150, 88, "Saved model", ["model.joblib", "Fixed alert cutoff"])
    box(c, 755, 103, 165, 88, "Doctor dashboard", ["EHR + glucose history", "Score, alert, time replay"])
    arrow(c, [(250, 259), (250, 147), (285, 147)])
    arrow(c, [(505, 147), (555, 147)])
    arrow(c, [(705, 147), (755, 147)])
    arrow(c, [(837, 277), (837, 231), (630, 231), (630, 191)], fill=GRAY)
    text(c, "saved weights + cutoff", 650, 250, 170, size=10, fill=GRAY)
    text(c, "Future CGM creates labels only. It never enters prediction inputs; follow-up reveal is opt-in.",
         40, 78, 880, size=12, fill=GRAY)
    text(c, "Synthetic demonstration only. Same patients across time splits; clinical validity is untested.",
         40, 58, 880, size=11, fill=GRAY)


def table(c, headers, rows, widths, x, y, row_height=48, font_size=15, highlight=None):
    total = sum(widths)
    c.setFillColor(color(NAVY))
    c.rect(x, y - row_height, total, row_height, fill=1, stroke=0)
    positions = [x + sum(widths[:i]) for i in range(len(widths))]
    for position, width, value in zip(positions, widths, headers):
        text(c, value, position + 9, y - 11, width - 18, size=font_size - 1, bold=True, fill="#FFFFFF", max_height=row_height - 15)
    y -= row_height
    for index, row in enumerate(rows):
        c.setFillColor(color(LIGHT if index == highlight else ("#F6F8FB" if index % 2 == 0 else "#FFFFFF")))
        c.rect(x, y - row_height, total, row_height, fill=1, stroke=0)
        for position, width, value in zip(positions, widths, row):
            text(c, value, position + 9, y - 12, width - 18, size=font_size,
                 bold=index == highlight, max_height=row_height - 15)
        y -= row_height
    return y


def make_pdfs(info, metrics, quality, demo, ehr, cgm):
    DOCS.mkdir(exist_ok=True)
    arch_path = DOCS / "architecture.pdf"
    c = canvas.Canvas(str(arch_path), pagesize=(W, H))
    c.setTitle("GlucoTwin system architecture")
    c.setAuthor(info["team_leader"])
    architecture(c, info)
    c.save()

    path = DOCS / "presentation.pdf"
    c = canvas.Canvas(str(path), pagesize=(W, H))
    c.setTitle("GlucoTwin - Synthetic patient twin for glucose alerts")
    c.setAuthor(info["team_leader"])
    cfg = metrics["protocol"]["config"]
    fused = metrics["models"]["fused_hgb"]["test"]
    counts = fused["confusion_matrix"]

    # 1. Cover
    page(c, "GlucoTwin", info, 1)
    text(c, "A synthetic patient twin\nfor next-hour glucose alerts", 40, 402, 820, size=35, bold=True, max_height=120)
    text(c, "Baseline EHR + CGM history + time-aware prediction", 40, 278, 820, size=22, fill=TEAL)
    text(c, f"{info['team_leader']}\n{info['college']}\nTeam {info['team_name']} | Solo proof of concept", 40, 212, 800, size=19)
    text(c, "Digital Twin Challenge 2026 | Phase 1 prototype", 40, 101, 820, size=15, fill=GRAY)
    text(c, "All patient records are synthetic. No established clinical validity.", 40, 73, 820, size=13, fill=GRAY)
    c.showPage()

    # 2. Scope and operational target
    page(c, "Healthcare use case and prediction target", info, 2,
         "A doctor reviews patient context, recent glucose and a model-estimated alert at a chosen time")
    text(c, "One operational question", 40, 376, 415, size=23, bold=True)
    text(c, f"When current glucose is below {cfg['event_threshold_mg_dl']} mg/dL, will any reading reach that threshold "
         f"during the next {cfg['prediction_horizon_minutes']} minutes?", 40, 325, 415, size=20)
    text(c, "This threshold defines the prototype event. It is not a personalized treatment threshold.",
         40, 176, 415, size=15, fill=GRAY)
    text(c, "Available at prediction time", 525, 376, 395, size=23, bold=True)
    bullet_list(c, ["Baseline EHR recorded before the stream", "Two hours of CGM, including the current reading",
                    "Current clock time", "No future readings or patient ID as predictors"], 525, 323, 395, size=18, gap=19)
    c.showPage()

    # 3. Data
    page(c, "Synthetic data and patient linkage", info, 3,
         "Project-generated records only; no external dataset, real patient records or live devices")
    table(c, ["Stream", "Contents", "Default scale"], [
        ["Baseline EHR", "Demographics, diagnoses, BMI, HbA1c, duration", f"{quality['ehr_rows']:,} patients"],
        ["Wearable CGM", "Patient ID, timestamp, glucose (mg/dL)", f"{quality['cgm_rows']:,} readings"],
    ], [180, 490, 210], 40, 384, row_height=66, font_size=17)
    text(c, "Fusion key: patient_id", 40, 155, 400, size=22, bold=True, fill=TEAL)
    text(c, "Baseline labs are sampled before CGM. They are not calculated from future glucose.", 40, 109, 880, size=17)
    text(c, "The simulator includes meal-like rises, daily patterns and noise. Associations are engineered assumptions.",
         40, 73, 880, size=13, fill=GRAY)
    c.showPage()

    # 4. Architecture
    architecture(c, info, 4)
    c.showPage()

    # 5. Evaluation
    page(c, "Chronological evaluation", info, 5,
         "Both history and future-label windows remain wholly inside their assigned time split")
    splits = metrics["split_counts"]
    table(c, ["Split", "Days", "Eligible windows", "Event windows"], [
        ["Train", f"1-{cfg['train_days']}", f"{splits['train']['eligible_windows']:,}", f"{splits['train']['event_windows']:,}"],
        ["Validation", f"{cfg['train_days'] + 1}-{cfg['train_days'] + cfg['validation_days']}", f"{splits['validation']['eligible_windows']:,}", f"{splits['validation']['event_windows']:,}"],
        ["Test", f"{cfg['train_days'] + cfg['validation_days'] + 1}-{cfg['n_days']}", f"{splits['test']['eligible_windows']:,}", f"{splits['test']['event_windows']:,}"],
    ], [180, 170, 280, 250], 40, 385, row_height=48, font_size=17)
    bullet_list(c, ["Fit models on training data; choose alert cutoffs on validation F1.",
                    "Freeze model settings and cutoffs before test evaluation.",
                    "The same patients recur across splits; unseen-patient performance is untested."],
                40, 160, 880, size=16, gap=12)
    c.showPage()

    # 6. Results
    page(c, "Synthetic test results", info, 6,
         "Current-glucose baseline, wearable-only ablation and the preselected fused model")
    rows = []
    labels = ["Current glucose", "CGM + clock", "CGM + clock + EHR"]
    for label, key in zip(labels, ("baseline_current_glucose", "wearable_only_hgb", "fused_hgb")):
        m = metrics["models"][key]["test"]
        rows.append([label, f"{m['roc_auc']:.4f}", f"{m['average_precision']:.4f}",
                     f"{m['precision']:.1%}", f"{m['recall']:.1%}", f"{m['f1']:.4f}"])
    table(c, ["Model", "ROC-AUC", "Avg precision", "Precision", "Recall", "F1"], rows,
          [270, 110, 150, 120, 110, 120], 40, 385, row_height=58, font_size=16, highlight=2)
    text(c, f"Test event prevalence: {fused['event_prevalence']:.2%}. Average precision makes the rare-event comparison visible.",
         40, 130, 880, size=17)
    text(c, "Adding EHR changes simulator performance; this does not demonstrate clinical benefit.",
         40, 84, 880, size=15, fill=GRAY)
    c.showPage()

    # 7. Errors matter
    page(c, "Alert errors remain substantial", info, 7,
         "Fused model at its validation-selected cutoff; counts refer to overlapping test windows")
    table(c, ["Observed label", "No alert", "Alert"], [
        ["No event", f"TN {counts['tn']:,}", f"FP {counts['fp']:,}"],
        ["Event", f"FN {counts['fn']:,}", f"TP {counts['tp']:,}"],
    ], [220, 170, 170], 40, 371, row_height=64, font_size=20)
    text(c, f"{fused['precision']:.1%}", 655, 360, 265, size=38, bold=True, fill=TEAL)
    text(c, "Precision: event among alerted windows", 655, 305, 265, size=17)
    text(c, f"{fused['recall']:.1%}", 655, 235, 265, size=38, bold=True, fill=AMBER)
    text(c, "Recall: alerted among event windows", 655, 180, 265, size=17)
    text(c, "About half the event windows are missed at this operating point. A below-cutoff score does not mean safe.",
         40, 128, 880, size=19, bold=True)
    text(c, "Window-level metrics are not episode-level sensitivity, and repeated alerts have not been deduplicated.",
         40, 67, 880, size=13, fill=GRAY)
    c.showPage()

    # 8. Faithful schematic with actual available data; never presented as a screenshot.
    page(c, "Doctor interaction with the twin", info, 8,
         "Workflow schematic using an actual validation snapshot; app.py contains the working interface")
    case = demo["validation_demo_cases"][0]
    at = pd.Timestamp(case["as_of"])
    patient_id = case["patient_id"]
    static = ehr.loc[ehr["patient_id"] == patient_id].iloc[0]
    selected = cgm.loc[(cgm["patient_id"] == patient_id) & (cgm["timestamp"] <= at) &
                       (cgm["timestamp"] >= at - pd.Timedelta(hours=2))]
    box(c, 40, 192, 205, 195, patient_id, [f"Age: {int(static['age_years'])} years", f"BMI: {static['bmi_kg_m2']:.1f}",
        f"HbA1c: {static['hba1c_percent']:.1f}%", f"EHR: {static['baseline_date']}", "Diagnosis: Type 2 diabetes"])
    text(c, "Available CGM history", 290, 386, 400, size=20, bold=True)
    x0, y0, width, height = 300, 218, 340, 115
    glucose = selected["glucose_mg_dl"].tolist()
    low = min(min(glucose) - 8, 100)
    high = max(max(glucose) + 8, cfg["event_threshold_mg_dl"] + 8)
    c.setStrokeColor(color("#B3C5D0")); c.line(x0, y0, x0 + width, y0)
    c.setStrokeColor(color(TEAL)); c.setLineWidth(2)
    for i in range(1, len(glucose)):
        c.line(x0 + (i - 1) * width / (len(glucose) - 1), y0 + (glucose[i - 1] - low) / (high - low) * height,
               x0 + i * width / (len(glucose) - 1), y0 + (glucose[i] - low) / (high - low) * height)
    text(c, f"{(at - pd.Timedelta(hours=2)).strftime('%H:%M')} - {at.strftime('%H:%M')} | mg/dL | no future readings",
         290, 195, 400, size=12, fill=GRAY)
    text(c, "Model score", 715, 386, 205, size=20, bold=True)
    text(c, f"{case['risk_score']:.1%}", 715, 338, 205, size=38, bold=True, fill=AMBER)
    text(c, f"Cutoff: {fused['threshold']:.1%}\nPrototype alert for review", 715, 276, 205, size=17)
    text(c, f"Snapshot: {at.strftime('%d %b %Y, %H:%M')} | Synthetic local time", 40, 158, 880, size=15, fill=GRAY)
    text(c, "Select patient -> advance time -> inspect history and score -> optionally reveal observed follow-up",
         40, 129, 880, size=18)
    text(c, "Revealed follow-up is never a forecast and never changes inference. Already-high states withhold new-onset scoring.",
         40, 69, 880, size=13, fill=GRAY)
    c.showPage()

    # 9. Limitations and next work
    page(c, "Limitations and next validation work", info, 9,
         "Implemented outcome: reproducible fusion, trained prediction and an updating doctor-facing patient state")
    text(c, "Current limitations", 40, 378, 415, size=24, bold=True)
    bullet_list(c, ["Synthetic patterns and engineered EHR associations", "Known patients across time splits",
                    "Correlated windows; no episode-level evaluation", "No independent calibration or clinical validation"],
                40, 327, 415, size=18, gap=16)
    text(c, "Future work", 525, 378, 395, size=24, bold=True)
    bullet_list(c, ["Assess unseen patients and realistic missingness", "Study calibration and alert burden",
                    "Validate externally under appropriate data governance", "Prospective evaluation before any clinical use"],
                525, 327, 395, size=18, gap=16)
    text(c, "No real patient data, diagnosis, medication advice or regulatory approval is claimed.",
         40, 78, 880, size=16, fill=GRAY)
    c.save()
    return arch_path, path


def dependency_notices():
    roots = [line.strip() for line in (ROOT / "requirements.in").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    pending = deque(Requirement(line) for line in roots)
    seen, distributions = {}, {}
    while pending:
        requirement = pending.popleft()
        name = canonicalize_name(requirement.name)
        extras = set(requirement.extras) | {""}
        if name in seen and extras.issubset(seen[name]):
            continue
        seen.setdefault(name, set()).update(extras)
        distribution = metadata.distribution(requirement.name)
        distributions[name] = distribution
        for raw in distribution.requires or []:
            child = Requirement(raw)
            if child.marker is None or any(child.marker.evaluate({"extra": extra}) for extra in seen[name]):
                pending.append(child)

    destination = DOCS / "dependency_licenses"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(exist_ok=True)
    records, table_rows = [], []
    for name, distribution in sorted(distributions.items()):
        expression = distribution.metadata.get("License-Expression")
        license_text = distribution.metadata.get("License")
        classifiers = [value for value in distribution.metadata.get_all("Classifier", []) if value.startswith("License ::")]
        label = expression or (classifiers[-1].split(" :: ")[-1] if classifiers else None)
        if not label:
            first_line = (license_text or "Not declared in installed metadata").splitlines()[0]
            label = first_line if len(first_line) <= 90 else "See full installed license metadata"
        copied = []
        for relative in sorted(distribution.files or [], key=str):
            filename = Path(str(relative)).name
            if not any(part in filename.lower() for part in ("license", "licence", "copying", "notice")):
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file():
                continue
            package_dir = destination / name
            package_dir.mkdir(exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
            target = package_dir / f"{len(copied) + 1:02d}_{safe_name}"
            shutil.copyfile(source, target)
            copied.append({"distribution_path": str(relative), "repository_path": target.relative_to(ROOT).as_posix()})
        record = {"name": distribution.metadata["Name"], "version": distribution.version,
                  "license_expression": expression, "license_metadata": license_text,
                  "license_classifiers": classifiers, "project_urls": distribution.metadata.get_all("Project-URL", []),
                  "copied_notices": copied}
        records.append(record)
        location = f"[Copied texts]({copied[0]['repository_path']})" if copied else "No license file packaged; see metadata JSON"
        table_rows.append(f"| {record['name']} | {record['version']} | {label.replace('|', '/')} | {location} |")
    (DOCS / "dependency_metadata.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    content = "\n".join([
        "# Third-party dependency notices", "",
        "Original GlucoTwin code and project-generated data use the repository MIT License. Dependencies retain their own terms.", "",
        "This inventory is generated from the actually installed dependency closure of requirements.in. License expressions, "
        "classifiers and texts are read from installed distributions, not guessed from package names. "
        "A summarized label is not a substitute for the full license text.", "",
        "[Full package metadata](docs/dependency_metadata.json) includes original license fields and the copied-notice mapping. "
        "Where a distribution packages no license file, that absence is explicitly recorded; consult its declared license and upstream distribution.", "",
        "| Dependency | Installed version | Declared license / classifier | Packaged notice |",
        "| --- | --- | --- | --- |", *table_rows, "",
        "Requirements are installed separately; no third-party executable package or model service is bundled into the original project code.", "",
    ])
    (ROOT / "THIRD_PARTY_NOTICES.md").write_text(content, encoding="utf-8")
    return len(records)


def make_readme(info, metrics, quality, demo):
    cfg = metrics["protocol"]["config"]
    main = metrics["models"]["fused_hgb"]["test"]
    video = info["video_url"].strip()
    video_section = f"[Watch the demonstration video]({video})\n\nRequired duration: at least 20 minutes. Verify actual duration and signed-out access." if video else (
        "**REQUIRED BEFORE SUBMISSION:** Add your accessible recording URL to `project_info.json` under `video_url`, "
        "then rerun `build_docs.py`. The recording must be at least 20 minutes. No video link has been supplied yet.")
    lines = ["# GlucoTwin", "", "Synthetic patient twin for next-hour glucose alerts in Type 2 diabetes.", "",
             "**Status:** Data generator, trained model, chronological evaluation and Streamlit dashboard are implemented. "
             "Follow the final submission checks below before submitting to the organizer.", "",
             "## Team and College / Incubator", "",
             f"- Team: **{info['team_name']}**" + (" (working name; editable in project_info.json)" if info["team_name_provisional"] else ""),
             f"- Team leader and sole member: **{info['team_leader']}**", f"- College: **{info['college']}**",
             f"- Team size: {info['team_size']}", f"- Roles: {info['roles']}", f"- Incubator: {info['incubator']}",
             "- Suggested repository/folder name: `GlucoTwin_IIT_Kanpur` (update if the team name changes).", "",
             "## Problem Statement and Healthcare Use Case", "",
             "Build a proof of concept in which a doctor reviews a patient's baseline EHR alongside continuously replayed "
             "CGM measurements and a next-hour event score. The use case is advance review of a possible glucose threshold "
             "excursion, without prescribing treatment or asserting clinical effectiveness.", "",
             "## Digital Twin Behavior", "",
             "Each synthetic patient has linked baseline EHR and a timestamped CGM stream. Changing the selected patient "
             "or advancing twin time updates the available history, feature vector, model score and alert state. "
             "A shared trained model serves patient-specific inputs; the model does not retrain during replay. "
             "This is a data-driven patient-state prototype, not a validated physiological simulator.", "",
             "## Prediction Target", "",
             f"- Eligible observation: current glucose is below **{cfg['event_threshold_mg_dl']} mg/dL**.",
             f"- Event: at least one CGM reading reaches **{cfg['event_threshold_mg_dl']} mg/dL** in **(t, t + {cfg['prediction_horizon_minutes']} minutes]**.",
             "- Current glucose is excluded from the future target window.",
             "- This is a documented prototype definition, not a personalized treatment threshold.",
             "- Already-above-threshold observations display 'Not scored' for new-onset risk.", "",
             "## Architecture", "", "[Architecture PDF](docs/architecture.pdf)", "",
             "Synthetic EHR + synthetic CGM -> patient ID join -> historical features -> chronological training/validation/test "
             "-> saved fused model -> patient replay dashboard. Future CGM is accessed for evaluation labels only; "
             "the dashboard's optional retrospective reveal never enters inference.", "",
             "## Synthetic Data and Provenance", "",
             f"- {quality['ehr_rows']:,} baseline EHR records and {quality['cgm_rows']:,} CGM readings, covering {cfg['n_days']} simulated days.",
             f"- CGM every {cfg['sampling_minutes']} minutes; synthetic IDs link both tables; seed `{cfg['seed']}`.",
             "- EHR includes age, sex, BMI, diagnoses, diabetes duration and baseline HbA1c.",
             "- Baseline EHR is generated before CGM. HbA1c is not computed from future observations.",
             "- Glucose includes meal-like responses, daily patterns and stochastic noise. Hidden simulator factors are not exported as features.",
             "- No real patient records, identifiers, external datasets or genetic markers are used.", "",
             "[Data dictionary](data/DATA_DICTIONARY.md) | [Provenance and checksums](data/provenance.json) | "
             "[Data reuse terms](data/DATA_LICENSE.md)", "",
             "## Technical Stack and AI/ML Details", "",
             "Python, NumPy, pandas, scikit-learn, Streamlit, matplotlib, joblib and ReportLab. CPU only; CUDA is unnecessary. "
             "Risk inference uses scikit-learn, with no LLM or external inference API.", "",
             "- Baseline: StandardScaler + LogisticRegression on current glucose only; scaler fits on training data.",
             "- Wearable-only ablation: HistGradientBoostingClassifier on 12 dynamic/clock features.",
             "- Main model: the same boosting settings with six baseline EHR features added (18 total inputs).",
             "- Dynamic features: current glucose, 30/60/120-minute lags, two-hour mean/std/min/max, slope, delta, and sine/cosine clock time.",
             "- Static predictors: age, deterministic sex indicator, BMI, hypertension, diabetes duration and HbA1c.",
             "- Patient ID, baseline date, target labels and future glucose are excluded from predictors.", "",
             "Full settings are in [protocol.json](reports/protocol.json); no hyperparameter search was performed. "
             "Internal random early stopping is disabled.", "",
             "## Chronological Evaluation and Leakage Controls", "",
             "| Split | Days | Eligible windows | Event windows |", "| --- | --- | ---: | ---: |"]
    first_day = 1
    for split, key in (("train", "train_days"), ("validation", "validation_days"), ("test", "test_days")):
        final_day = first_day + cfg[key] - 1
        stats = metrics["split_counts"][split]
        lines.append(f"| {split.title()} | {first_day}-{final_day} | {stats['eligible_windows']:,} | {stats['event_windows']:,} |")
        first_day = final_day + 1
    lines += ["", "Every two-hour history and one-hour label window stays completely inside its split. "
              "Evaluation anchors are sampled every 30 minutes. Train first, choose each alert cutoff by maximum validation "
              "window F1, then evaluate the test set after settings and cutoffs are frozen. "
              "The main model is preselected by design, not by test ranking.", "",
              "The same patients recur across time splits. This tests later periods for known synthetic patients, "
              "not unseen-patient or real-world generalization. Overlapping windows are correlated.", "",
              "## Test Results", "",
              "Results below are generated from this checkout's reports/metrics.json.", "",
              "| Model | ROC-AUC | Average precision | Precision | Recall | F1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, key in (("Current glucose baseline", "baseline_current_glucose"), ("CGM history + clock", "wearable_only_hgb"),
                       ("CGM history + clock + EHR", "fused_hgb")):
        m = metrics["models"][key]["test"]
        lines.append(f"| {label} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.4f} |")
    cm = main["confusion_matrix"]
    lines += ["", f"Test prevalence: **{main['event_prevalence']:.2%}**. Fused cutoff: **{main['threshold']:.4f}**, selected on validation only.", "",
              f"Fused confusion counts: TN **{cm['tn']:,}**, FP **{cm['fp']:,}**, FN **{cm['fn']:,}**, TP **{cm['tp']:,}**. "
              "About half the event windows are missed at this operating point. A below-cutoff score does not establish safety.", "",
              "[Detailed results](reports/results.md) | [Machine-readable metrics](reports/metrics.json) | "
              "[Precision-recall / ROC curves](reports/model_comparison.png) | [Confusion matrix](reports/confusion_matrix.png)", "",
              "## Windows Setup and Reproduction", "",
              "Python 3.12 is a practical starting point. Open PowerShell in the project folder. "
              "Use the submitted requirements.txt to reproduce the checked environment; requirements.in lists direct dependencies.", "",
              "```powershell", "py -m venv .venv", ".\\.venv\\Scripts\\python.exe -m pip install --upgrade pip",
              ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt", ".\\.venv\\Scripts\\python.exe verify_setup.py",
              ".\\.venv\\Scripts\\python.exe generate_data.py", ".\\.venv\\Scripts\\python.exe validate_data.py",
              ".\\.venv\\Scripts\\python.exe train.py", ".\\.venv\\Scripts\\python.exe verify_model.py",
              ".\\.venv\\Scripts\\python.exe verify_dashboard.py", ".\\.venv\\Scripts\\python.exe build_docs.py",
              ".\\.venv\\Scripts\\python.exe -m streamlit run app.py", "```", "",
              "Open http://localhost:8501 and keep the terminal running. Stop with Ctrl+C. "
              "No environment activation, GPU, cloud service, API key or account is required to run locally.", "",
              "Regenerate data and retrain together if the config or dependency versions change. Load only a model "
              "you generated or otherwise trust. The saved bundle records feature order, validation cutoff, runtime versions and checksums.", "",
              "### Record the final environment", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe -m pip freeze | Set-Content -Encoding ascii requirements.txt", "```", "",
              "Use a clean project .venv, not a global Python environment, for that snapshot. "
              "Same script, seed and dependency versions reproduce generated CSV bytes; rerunning a fixed modeling protocol "
              "is reproduction, not permission to tune on test results.", "",
              "## Final Submission Checks", "",
              "[Step 7 instructions](docs/STEP7.md) explain team/video confirmations, public GitHub checks and "
              "fresh-clone reproduction. Run `preflight.py --source-only` for local technical checks; "
              "run `verify_clean_clone.py` after pushing your final public repository, then `preflight.py`. "
              "Only the full check can report READY_FOR_MANUAL_SUBMISSION; no script submits to the platform.", "",
              "## Doctor Dashboard", "",
              "1. Select a synthetic patient and inspect baseline EHR.",
              "2. Move Twin time or click Next reading to update patient state and score.",
              "3. Inspect the available glucose history, model score, cutoff and alert.",
              "4. Optionally reveal the observed next hour. Orange dashed values are later observations, not a predicted trajectory.",
              "5. Change time or patient; revealed follow-up hides automatically.",
              "6. Review model evaluation and limitations. At the dataset end, incomplete follow-up is explicitly reported.", "",
              "[Automated dashboard checks and local demo cases](reports/dashboard_checks.json). "
              "The default chart shows eight hours of history; the model uses only the latest two hours.", "",
              "## Demonstration Video - Minimum 20 Minutes", "", video_section, "",
              "## Presentation", "", "[Project presentation PDF](docs/presentation.pdf) | [Editable content source](docs/presentation.md)", "",
              "## Open-Source License", "", "Original code and project-generated synthetic data are distributed under the "
              "[MIT License](LICENSE); generated-data terms are also documented in data/DATA_LICENSE.md. "
              "Dependencies retain their own licenses; [third-party notices](THIRD_PARTY_NOTICES.md) are generated from installed distribution metadata and packaged notice files.", "",
              "## Limitations and Ethics", ""]
    lines += ["- " + limitation for limitation in metrics["limitations"]]
    lines += ["- No real patient data is collected. Synthetic-only operation is not a claim of DPDP/HIPAA certification or regulatory approval.", "",
              "## Future Work", "", "Evaluate unseen patients, realistic missingness, calibration and episode-level alert burden; "
              "then consider appropriately governed external and prospective validation. None of these validations is claimed here.", "",
              "## Submission Checklist", "", "- [x] Team leader and college details provided",
              "- [x] Linked synthetic EHR and CGM with generator, dictionary and provenance",
              "- [x] Trained fused predictor, baseline, chronological evaluation and leakage checks",
              "- [x] Working doctor-facing dashboard with patient/time replay",
              "- [x] Architecture PDF and project presentation PDF",
              "- [x] MIT license and dependency metadata/notices",
              "- [ ] Confirm final team name and official repository naming convention",
              "- [ ] Verify full-time-student eligibility and any incubator information",
              "- [ ] Add and verify a recording of at least 20 minutes" if not video else "- [ ] Verify linked recording duration and signed-out playback",
              "- [ ] Test from a fresh clone and clean environment",
              "- [ ] Verify public repository, PDFs and all links while signed out",
              "- [ ] Submit leader name, phone, email and GitHub URL; retain confirmation", "",
              "Author contact phone/email are supplied on the submission platform, not required in this public README.", ""]
    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    info = read_json(ROOT / "project_info.json")
    for key in ("project_title", "team_name", "college", "team_leader"):
        if not str(info.get(key, "")).strip():
            raise ValueError(f"Missing project_info field: {key}")
    config = load_config(ROOT / "config.json")
    metrics = read_json(ROOT / "reports" / "metrics.json")
    quality = read_json(ROOT / "reports" / "data_quality.json")
    checks = read_json(ROOT / "reports" / "model_checks.json")
    demo = read_json(ROOT / "reports" / "dashboard_checks.json")
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    check_bundle(bundle, config, ROOT / "data")
    if metrics["protocol"]["config"] != config or metrics["runtime"] != bundle["runtime"]:
        raise ValueError("Metrics do not match the current configuration/model; retrain")
    if any(record["status"] != "PASS" for record in (quality, checks, demo)):
        raise ValueError("Data, model and dashboard checks must pass before building documentation")
    ehr = pd.read_csv(ROOT / "data" / "ehr.csv")
    cgm = pd.read_csv(ROOT / "data" / "cgm.csv", parse_dates=["timestamp"])
    register_fonts()
    arch, presentation = make_pdfs(info, metrics, quality, demo, ehr, cgm)
    n_dependencies = dependency_notices()
    make_readme(info, metrics, quality, demo)
    # Editable text source comes from the same local evidence as the PDFs.
    contents = "\n\n".join([
        "# GlucoTwin presentation source", f"{info['team_leader']} | {info['college']} | Team {info['team_name']}",
        "## 1. GlucoTwin\nSynthetic patient twin for next-hour glucose alerts; solo proof of concept.",
        "## 2. Healthcare use case and target\nCurrent glucose <180 mg/dL; any next-hour CGM >=180 mg/dL. Prototype definition only.",
        "## 3. Synthetic data\nLinked baseline EHR and CGM. Baseline labs precede the generated stream.",
        "## 4. Architecture\nEHR + CGM -> causal historical features -> chronological fitting/evaluation -> saved model -> replay UI.",
        "## 5. Evaluation\nTrain days 1-13; validation 14-17; test 18-21. Complete windows inside each split; same patients recur.",
        "## 6. Results\nSee ../reports/results.md for the exact local metrics used in this build.",
        "## 7. Errors\nReport precision, recall and confusion counts. Below-cutoff does not mean safe.",
        "## 8. Doctor workflow\nSelect patient; advance time; inspect history and score; optionally reveal follow-up without changing inference.",
        "## 9. Limitations and future work\nSynthetic patterns, known patients, correlated windows and uncalibrated scores. External and clinical validation remain future work.",
    ]) + "\n"
    (DOCS / "presentation.md").write_text(contents, encoding="utf-8")
    manifest = {"status": "BUILT", "team": info, "runtime": metrics["runtime"], "dependency_count": n_dependencies,
                "files": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (arch, presentation, ROOT / "README.md", ROOT / "THIRD_PARTY_NOTICES.md")},
                "pending": ["Confirm working team name", "Verify video duration and signed-out access",
                            "Complete fresh-clone and public-link checks", "Verify eligibility and incubator details"]}
    (DOCS / "documentation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"DOCUMENTATION BUILT: README, architecture PDF, 9-slide presentation PDF; {n_dependencies} dependency records.")
    if not info["video_url"].strip():
        print("PENDING: add video_url to project_info.json and rerun build_docs.py before submission.")
    if info["team_name_provisional"]:
        print("WORKING TEAM NAME: GlucoTwin. Confirm or edit project_info.json; this does not affect the trained model.")


if __name__ == "__main__":
    main()
