"""
Step 5 (V2) - Run records: provenance, unique output folders, task logs.
================================================================================
ROMAN URDU: Har run ke saath itna "saboot" (evidence) save hota hai ke koi
bhi (aap, supervisor, ya aap khud 6 mahine baad) result ko dobara bana sake:
poori config ki copy, seed, window/drain settings, git commit hash + kya
working tree "dirty" thi (uncommitted code), Python aur libraries ke
versions. Har run ka apna UNIQUE folder hota hai -- dobara chalane se purana
result kabhi chupke se overwrite nahi hota.

ENGLISH: Every saved run carries enough provenance to reproduce it (full
config snapshot, seed, protocol settings, git commit + dirty status, Python
and library versions). Each run writes to a NEW unique folder, so a rerun
never silently overwrites earlier evidence.
"""

import csv
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from typing import Dict, Iterable, List, Optional

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True, check=True)
        return out.stdout.rstrip()  # rstrip only: porcelain lines start with a space
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def parse_porcelain(text: Optional[str]) -> List[str]:
    """Paths from `git status --porcelain` output ("XY path" per line)."""
    return [line[3:] for line in (text or "").splitlines() if line.strip()]


def collect_provenance() -> Dict:
    """Code revision + runtime environment of THIS process (verified at run
    time, not assumed). dirty_tracked_files lists modified tracked files --
    if non-empty, the result came from uncommitted code."""
    dirty_lines = _git("status", "--porcelain", "--untracked-files=no")
    dirty_files = parse_porcelain(dirty_lines)
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty_files) if dirty_lines is not None else None,
        "dirty_tracked_files": dirty_files,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _version(name) for name in ("numpy", "PyYAML", "pytest", "gymnasium")},
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def unique_run_dir(base_dir: str, name: str, commit: Optional[str] = None) -> str:
    """Create and return a NEW folder base_dir/<name>_<UTC time>_<commit7>[_k].
    Never reuses an existing folder (exist_ok=False)."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{name}_{stamp}_{(commit or 'nogit')[:7]}"
    os.makedirs(base_dir, exist_ok=True)
    for k in range(1, 1000):
        path = os.path.join(base_dir, stem if k == 1 else f"{stem}_{k}")
        try:
            os.makedirs(path, exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique run folder under {base_dir}")


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def write_csv(path: str, rows: Iterable[Dict], columns: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row[k] is None else row[k]) for k in columns})
