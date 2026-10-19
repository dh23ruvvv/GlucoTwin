# Step 2: generate and validate synthetic data

Copy the contents of this update into the existing project folder. Merge data, reports and
docs with the existing folders. This update does not include or overwrite your README,
requirements, config, license or Git configuration. It is NOT a replacement repository.

From PowerShell in the project folder:

```powershell
.\.venv\Scripts\python.exe generate_data.py
.\.venv\Scripts\python.exe validate_data.py
```

Default expected row counts: 80 EHR and 161,280 CGM. The validation command must finish
with DATA CHECKS PASSED and positive event and non-event counts in each time split.
Pre-generated files in this update are provided for inspection; regenerate locally to record
your actual runtime and config. Check local results; event counts can vary with dependency versions.

## Byte-for-byte reproducibility check

```powershell
$before = (Get-FileHash .\data\ehr.csv).Hash + (Get-FileHash .\data\cgm.csv).Hash
.\.venv\Scripts\python.exe generate_data.py
$after = (Get-FileHash .\data\ehr.csv).Hash + (Get-FileHash .\data\cgm.csv).Hash
if ($before -ne $after) { throw "Reproducibility check failed" }
Write-Output "REPRODUCIBILITY OK"
.\.venv\Scripts\python.exe validate_data.py
```

## Commit

```powershell
git add .gitattributes generate_data.py validate_data.py data reports/data_quality.json docs/STEP2.md
git commit -m "Add reproducible synthetic EHR and CGM data"
git push
```

## README update

In the existing README, change the status to:

> Steps 1-2 complete: reproducible linked synthetic EHR and CGM data, with integrity checks.
> Model training and dashboard are not implemented yet.

Replace the synthetic-data TODO with links to data/DATA_DICTIONARY.md, data/provenance.json
and data/DATA_LICENSE.md. Add the two run commands above under reproduction instructions.
The simulation is illustrative: baseline HbA1c is sampled before CGM, not derived from future data.

## Troubleshooting

- Missing config: run inside your existing project; do not use this update as a standalone project.
- Checksum or config mismatch: run generate_data.py, then validate_data.py without editing CSVs.
- Missing package: install requirements.txt using the same .venv Python executable.
- Git LF/CRLF warning: the included .gitattributes keeps project text files in LF format.
  After a fresh clone, run generation and validation together; do not hand-edit generated files.
- Excel: inspect CSVs without saving them back over the originals. Resaving can alter values and checksums.

The next step is feature construction and honest model evaluation. No model has been trained here.
