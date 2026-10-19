"""Clone the user's PUBLIC GitHub repo and run it in a fresh virtual environment."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from preflight import ROOT, github_parts, read_json


def main():
    info = read_json(ROOT / "project_info.json")
    try:
        owner, repository = github_parts(info.get("github_url", ""))
    except ValueError as exc:
        raise SystemExit(f"SETUP NEEDED: {exc}. Edit github_url in project_info.json first.")
    destination = Path(tempfile.mkdtemp(prefix="glucotwin-fresh-"))
    checkout = destination / repository
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment.pop("GIT_ASKPASS", None)
    environment.pop("SSH_ASKPASS", None)
    environment.pop("PYTHONPATH", None)
    completed = []
    report = {"status": "FAIL", "github_url": f"https://github.com/{owner}/{repository}",
              "source_commit": None, "completed_steps": completed}
    report_path = ROOT / "reports" / "clean_clone_check.json"

    def run(label, command, cwd):
        print(f"\n=== {label} ===", flush=True)
        subprocess.run(command, cwd=cwd, env=environment, check=True, timeout=600)
        completed.append(label)

    try:
        run("Clone without Git credentials", ["git", "-c", "credential.helper=", "-c", "core.askPass=", "clone",
            "--depth", "1", f"https://github.com/{owner}/{repository}.git", str(checkout)], destination)
        report["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True, timeout=30).strip()
        run("Create isolated environment", [sys.executable, "-m", "venv", str(checkout / ".venv")], checkout)
        python = checkout / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run("Install pinned dependencies", [str(python), "-m", "pip", "install", "-r", "requirements.txt"], checkout)
        for label, script, extra in [
            ("Original committed files and pins", "preflight.py", ["--source-only"]),
            ("Dependency setup", "verify_setup.py", []),
            ("Regenerate synthetic inputs", "generate_data.py", []),
            ("Validate synthetic inputs", "validate_data.py", []),
            ("Retrain fixed protocol", "train.py", []),
            ("Verify causal model inference", "verify_model.py", []),
            ("Exercise doctor dashboard", "verify_dashboard.py", []),
            ("Rebuild documentation", "build_docs.py", []),
            ("Final local source checks", "preflight.py", ["--source-only"]),
        ]:
            run(label, [str(python), script, *extra], checkout)
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nTemporary clone retained for inspection: {checkout}", flush=True)
        print(f"Report: {report_path}", flush=True)
    print("\nCLEAN CLONE CHECK PASSED. Run preflight.py for final submission prerequisites.", flush=True)


if __name__ == "__main__":
    main()
