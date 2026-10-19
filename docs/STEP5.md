# Step 5: final documentation and PDF artifacts

Merge this update into your EXISTING repository, replacing README.md and THIRD_PARTY_NOTICES.md.
Review local README changes first if you have added custom information. Your model, config,
requirements and source data are not replaced. Keep your .venv and .git folders.

## Team information

project_info.json records Anusha Gupta, IIT Kanpur and the working team name GlucoTwin.
You may change team_name and set team_name_provisional to false. The suggested repository
name is GlucoTwin_IIT_Kanpur. Confirm the official naming convention before renaming GitHub.
If an incubator applies, replace Not specified with its actual name; otherwise use Not applicable.

## Regenerate using YOUR actual results

```powershell
.\.venv\Scripts\python.exe build_docs.py
```

This rebuilds README.md, docs/architecture.pdf, docs/presentation.pdf, editable presentation
content, dependency metadata, packaged license copies and THIRD_PARTY_NOTICES.md. It reads
local reports/metrics.json; it does not train or tune models. The PDF presentation has 9 slides.
Rerun verify_model.py and verify_dashboard.py first if those reports are absent.

## Video is supplied by the user

Add the recording URL in project_info.json under video_url, then rerun build_docs.py.
Check the recording is at least 20 minutes and accessible while signed out. A missing URL is
explicitly marked pending; no recording or link has been invented.

## Version snapshot

After the application checks succeed in your dedicated .venv:

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding ascii requirements.txt
```

Keep requirements.in as the list of direct dependencies. Do not use a global environment's freeze.

## Inspect

- Open README.md and both PDFs. Check names, college, results, links and page readability.
- Confirm README results match your local reports/results.md.
- Leave video/fresh-clone/public-access items unchecked until actually verified.
- Confirm licensing text is present. Third-party notices are based on installed metadata,
  with packaged license texts copied where available; missing packaged texts are disclosed.

## Commit

```powershell
git add README.md THIRD_PARTY_NOTICES.md project_info.json build_docs.py docs requirements.txt
git commit -m "Document GlucoTwin with architecture and presentation PDFs"
git push
```

No website migration is needed. The next step is packaging/fresh-clone/public-link verification;
video recording remains the user's responsibility. Submission is not complete until the video
link and all external access checks pass.
