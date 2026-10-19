# Step 7: final verification and submission

Use this update inside the EXISTING repository. It adds preflight.py, verify_clean_clone.py
and these instructions. It does not replace your team information or requirements.txt.
The separate full-project ZIP is a backup/reference project, not a statement that the video
or public GitHub submission has been completed.

The included reports/reference_environment_check.json records an isolated Linux reference
run: pinned dependency installation, data/model/UI/docs checks, exact CSV regeneration and
matching model metrics. It is NOT a fresh-clone report for your public repository, and it
cannot satisfy the final public-repository/recording checks.

## 1. Initialize final-check fields without losing your values

```powershell
.\.venv\Scripts\python.exe preflight.py --init-fields
```

Edit project_info.json:

- Confirm team_name and set team_name_provisional to false.
- College: IIT Kanpur. Leader: Anusha Gupta.
- Set incubator to the actual name, or Not applicable if none.
- Set github_url to the actual https://github.com/OWNER/REPOSITORY URL.
- Set video_url to your accessible recording URL.
- Set video_duration_minutes to its ACTUAL duration, which must be at least 20.
- Set video_access_verified to true only after playing it while signed out.
- Set eligibility_confirmed to true only after confirming the organizer's student eligibility rules.

Use JSON numbers/booleans, not strings: 23 and true, not "23" or "true".
Do not put your phone/email or credentials in the public project file; those go on the platform.

The recommended repository name for the working name is GlucoTwin_IIT_Kanpur.
If renaming an existing repository, use GitHub repository Settings and then update origin:

```powershell
git remote set-url origin https://github.com/YOUR_USERNAME/GlucoTwin_IIT_Kanpur.git
```

Replace YOUR_USERNAME with your actual account, and use your actual team name if different.
Do not create another repository merely to rename the existing one.

## 2. Finish the artifacts locally

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding ascii requirements.txt
.\.venv\Scripts\python.exe build_docs.py
.\.venv\Scripts\python.exe preflight.py --source-only
```

The source-only check must end LOCAL_SOURCE_CHECKS_PASS. It does not check external access,
eligibility or recording playback. Inspect the regenerated PDFs and README yourself.

## 3. Keep transient local check reports outside Git

Add these two lines to .gitignore if not already present:

```text
reports/submission_check.json
reports/clean_clone_check.json
```

Then commit and push your intended project files. Review git status first.

```powershell
git status --short
git add preflight.py verify_clean_clone.py build_docs.py .gitignore project_info.json requirements.txt README.md THIRD_PARTY_NOTICES.md docs reports/reference_environment_check.json
git commit -m "Prepare final submission checks and documentation"
git push
```

If earlier work was not pushed, ensure the source, data, model and reports from Steps 1-5
are also committed. Never add .venv, secrets or the video file. The preflight checks required
tracked paths, but filename checks are not a substitute for reviewing what you publish.

## 4. Test as a stranger

```powershell
.\.venv\Scripts\python.exe verify_clean_clone.py
```

This clones the PUBLIC default branch without credential helpers, creates a new .venv,
installs the submitted pins and runs the data/model/UI/docs pipeline. Installation can take
several minutes. It leaves the temporary directory for inspection and records the tested commit.
It never modifies, deletes or pushes your original repository.

## 5. Final preflight

```powershell
.\.venv\Scripts\python.exe preflight.py
```

READY_FOR_MANUAL_SUBMISSION requires all checks to pass, including public GitHub metadata,
local HEAD matching the public default branch, the clean-clone report matching that commit,
and your truthful video/eligibility confirmations. API rate limits/network failures remain
unverified checks, not evidence a repository is private. NOT_READY exits with code 2 and
lists missing items; it is not necessarily a software failure.

Open the repository in an incognito/signed-out browser and independently verify:

- README is visible and contains correct team/college/leader details.
- Architecture PDF and presentation PDF open from README.
- Video plays without requesting access and duration is at least 20 minutes.
- All code, linked synthetic data, model, requirements and licenses are accessible.
- The video demonstrates the actual model and identifies synthetic-data limitations.

If you change the final commit after testing, push it and rerun the clean-clone check.

## 6. Submit on the official platform

Enter ONLY the requested platform fields:

- Team leader: Anusha Gupta.
- Leader phone: enter your own accurate number on the platform.
- Leader email: enter your own accurate email on the platform.
- Public GitHub repository URL.

Verify the official cutoff. The stated deadline is 20 October 2026, 11:59 p.m. IST.
Aim to submit by 8:30 p.m. and keep the final hours for contingency. Save the submission
confirmation. Nothing in these scripts submits to the organizer or confirms receipt.

## Common errors

- Empty GitHub URL: fill project_info.json before verify_clean_clone.py.
- NOT_READY while video is pending: expected; finish that item rather than changing the model.
- Requirement package cannot install: use the tested Python version or a clean .venv; paste the exact pip error.
- Fresh clone lacks a file: commit/push it to the repository's DEFAULT branch, then rerun.
- Commit mismatch: push the final commit, then rerun the clone check.
- GitHub API rate limit: retry later or independently verify public access; do not claim automated success if the check did not pass.
- No changes to commit: fine if the intended files are already committed; continue with push/checks.
