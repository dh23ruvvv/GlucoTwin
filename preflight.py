"""Check local artifacts and final submission prerequisites. Does not submit or push."""
import argparse
import ast
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "requirements.txt", "requirements.in",
    ".gitignore", ".gitattributes", ".streamlit/config.toml", "config.json", "project_info.json",
    "generate_data.py", "validate_data.py", "features.py", "train.py", "predict.py", "app.py",
    "verify_setup.py", "verify_model.py", "verify_dashboard.py", "build_docs.py",
    "preflight.py", "verify_clean_clone.py", "data/ehr.csv", "data/cgm.csv",
    "data/DATA_DICTIONARY.md", "data/DATA_LICENSE.md", "data/provenance.json",
    "models/model.joblib", "reports/metrics.json", "reports/protocol.json",
    "reports/data_quality.json", "reports/model_checks.json", "reports/dashboard_checks.json",
    "docs/architecture.pdf", "docs/presentation.pdf", "docs/dependency_metadata.json", "docs/STEP7.md",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def github_parts(url):
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com" or parsed.query or parsed.fragment:
        raise ValueError("Use https://github.com/OWNER/REPOSITORY with no credentials or query string")
    parts = parsed.path.strip("/").removesuffix(".git").split("/")
    if len(parts) != 2 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in parts):
        raise ValueError("GitHub URL must identify one repository")
    return parts


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=40)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def check(source_only=False):
    checks = []

    def record(name, condition, detail):
        checks.append({"check": name, "passed": bool(condition), "detail": detail})

    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    record("Required files", not missing, "All required files present" if not missing else ", ".join(missing))
    if missing:
        return checks
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=path.name)
    record("Python syntax", True, "Top-level source files parse successfully")

    pins = [line.strip() for line in (ROOT / "requirements.txt").read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    pinned = True
    for line in pins:
        try:
            requirement = Requirement(line)
            specifiers = list(requirement.specifier)
            pinned = pinned and not requirement.url and len(specifiers) == 1 and specifiers[0].operator == "==" and "*" not in specifiers[0].version
        except Exception:
            pinned = False
    record("Pinned dependencies", bool(pins) and pinned, "Use pip freeze from the project .venv to pin exact versions")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    broken = []
    for target in re.findall(r"\]\(([^)]+)\)", readme):
        if not urllib.parse.urlparse(target).scheme and not target.startswith("#"):
            if not (ROOT / target.split("#", 1)[0]).is_file():
                broken.append(target)
    record("README local links", not broken, "No broken local links" if not broken else ", ".join(broken))
    for name in ("architecture.pdf", "presentation.pdf"):
        content = (ROOT / "docs" / name).read_bytes()
        record(name, content.startswith(b"%PDF-") and b"%%EOF" in content[-1024:], "PDF header and end marker checked; inspect visually too")

    for name in ("data_quality", "model_checks", "dashboard_checks"):
        report = read_json(ROOT / "reports" / (name + ".json"))
        record(name, report.get("status") == "PASS", "Recorded verification status; rerun the corresponding checker if source/data changed")
    from generate_data import load_config
    from validate_data import check_data
    check_data(load_config(ROOT / "config.json"), ROOT / "data")
    record("Current synthetic data integrity", True, "Schemas, patient linkage, timestamps, checksums and target classes pass")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    record("Original license", "MIT License" in license_text, "MIT text present; dependencies retain their own terms")

    if source_only:
        return checks
    info = read_json(ROOT / "project_info.json")
    record("Team and institution", all(str(info.get(k, "")).strip() for k in ("team_name", "team_leader", "college")), "Actual team, leader and college must be provided")
    record("Final team name", info.get("team_name_provisional") is False, "Confirm name and set team_name_provisional to false")
    record("Incubator details", str(info.get("incubator", "")).strip().lower() not in ("", "not specified", "todo"), "Supply actual incubator, or Not applicable if none")
    record("Student eligibility", info.get("eligibility_confirmed") is True, "Confirm the actual team meets the organizer's full-time-student requirements")
    video_url = str(info.get("video_url", "")).strip()
    parsed_video = urllib.parse.urlparse(video_url)
    record("Video link", parsed_video.scheme == "https" and bool(parsed_video.netloc) and video_url in readme,
           "Add the accessible video_url and rerun build_docs.py so README includes it")
    duration = info.get("video_duration_minutes")
    record("Video duration", isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 20,
           "Record the ACTUAL duration in video_duration_minutes; minimum is 20")
    record("Video signed-out access", info.get("video_access_verified") is True,
           "Manually play the recording while signed out, then set video_access_verified=true; no automatic playback claim is made")

    current_commit = None
    try:
        current_commit = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=normal", "--", ".",
                    ":(exclude)reports/submission_check.json", ":(exclude)reports/clean_clone_check.json")
        record("Working tree committed", not dirty, "Commit all intended changes before the final clone test" if dirty else "No uncommitted project changes")
        tracked = set(git("ls-files").splitlines())
        absent = sorted(set(REQUIRED) - tracked)
        record("Required files tracked by Git", not absent, ", ".join(absent) if absent else "All required artifacts are committed/tracked")
        unwanted = [name for name in tracked if name.startswith((".venv/", "__pycache__/")) or
                    Path(name).name.startswith(".env") or name == ".streamlit/secrets.toml" or
                    Path(name).suffix.lower() in (".mp4", ".mov", ".avi")]
        record("No environment/secrets/video files tracked", not unwanted, ", ".join(unwanted) if unwanted else "Checked tracked paths; video should be linked")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        record("Git repository", False, str(exc))

    try:
        owner, repository = github_parts(info.get("github_url", ""))
        expected = "_".join(re.sub(r"[^A-Za-z0-9]+", "_", info[key]).strip("_") for key in ("team_name", "college"))
        record("Repository name", repository.lower() == expected.lower(), f"Expected naming pattern: {expected}; verify against official instructions")
        request = urllib.request.Request(f"https://api.github.com/repos/{owner}/{repository}", headers={
            "Accept": "application/vnd.github+json", "User-Agent": "GlucoTwin-submission-check",
            "X-GitHub-Api-Version": "2026-03-10"})
        with urllib.request.urlopen(request, timeout=15) as response:
            remote = json.load(response)
        record("Public repository without authentication", remote.get("private") is False, "GitHub metadata fetched without an Authorization header")
        default_branch = remote["default_branch"]
        result = git("-c", "credential.helper=", "-c", "core.askPass=", "ls-remote", "--exit-code",
                     f"https://github.com/{owner}/{repository}.git", f"refs/heads/{default_branch}")
        remote_commit = result.split()[0]
        record("Latest commit pushed to default branch", current_commit is not None and remote_commit == current_commit,
               "Local HEAD must match the public repository's default branch")
    except (OSError, ValueError, KeyError, urllib.error.URLError, subprocess.TimeoutExpired) as exc:
        record("Public GitHub access / URL", False, f"Not verified: {exc}. API rate limits or network failures do not prove a repository is private.")

    clone_path = ROOT / "reports" / "clean_clone_check.json"
    clone = read_json(clone_path) if clone_path.exists() else {}
    record("Fresh clone of final commit", clone.get("status") == "PASS" and current_commit is not None and clone.get("source_commit") == current_commit,
           "Run verify_clean_clone.py after pushing the final commit; report must match this HEAD")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true", help="Local technical checks; never declares submission readiness")
    parser.add_argument("--init-fields", action="store_true", help="Add missing manual-check fields without replacing existing values")
    args = parser.parse_args()
    if args.init_fields:
        path = ROOT / "project_info.json"
        info = read_json(path)
        for key, default in {"video_duration_minutes": None, "video_access_verified": False, "eligibility_confirmed": False}.items():
            info.setdefault(key, default)
        path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print("Added missing manual-check fields to project_info.json. Existing values were preserved.")
        return
    try:
        checks = check(args.source_only)
    except Exception as exc:
        checks = [{"check": "Local validation", "passed": False, "detail": str(exc)}]
    passed = all(item["passed"] for item in checks)
    status = ("LOCAL_SOURCE_CHECKS_PASS" if args.source_only else "READY_FOR_MANUAL_SUBMISSION") if passed else "NOT_READY"
    output = {"status": status, "mode": "source_only" if args.source_only else "submission",
              "checks": checks, "note": "No platform submission is performed. Source-only mode does not verify external access or video."}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "submission_check.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for item in checks:
        print(f"{'PASS' if item['passed'] else 'PENDING'}: {item['check']} - {item['detail']}", flush=True)
    print("\n" + status)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
