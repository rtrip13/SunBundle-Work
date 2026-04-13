#!/usr/bin/env python3
"""
Remove Git LFS for this repo so ``data/irs/eo2.csv`` is stored as a normal Git blob.

Run once from repo root **after** you have a slim ``eo2.csv`` (e.g. from ``trim_eo2_for_app.py``)
under 100 MB:

  python scripts/remove_git_lfs_for_eo2.py

What it does:
  - Deletes ``.gitattributes`` if it only tracks eo2 (or removes eo2 LFS lines).
  - Removes the Git LFS ``pre-push`` hook that blocks pushes when ``git-lfs`` is missing.
  - Strips ``[lfs]`` sections from ``.git/config`` in this repo.
  - Re-stages ``eo2.csv`` using ``git hash-object`` + ``git update-index`` so the **raw CSV**
    is stored even if your **global** ``~/.gitconfig`` still defines ``filter.lfs`` (a plain
    ``git add`` would otherwise keep writing LFS pointer blobs).

You still need to ``git commit`` and ``git push`` yourself.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIT_DIR = ROOT / ".git"
CONFIG = GIT_DIR / "config"
ATTRS = ROOT / ".gitattributes"
PRE_PUSH = GIT_DIR / "hooks" / "pre-push"


def _strip_lfs_config() -> None:
    if not CONFIG.is_file():
        return
    text = CONFIG.read_text(encoding="utf-8")
    # Drop [lfs] and [lfs "..."] sections (repo-local LFS config from ``git lfs install``).
    out = re.sub(r"\[lfs[^\]]*\][^\[]*", "", text, flags=re.MULTILINE)
    # Clean excessive blank lines
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    CONFIG.write_text(out, encoding="utf-8")


def main() -> int:
    if not GIT_DIR.is_dir():
        print("Not a git repository.", file=sys.stderr)
        return 1

    if ATTRS.is_file():
        lines = [ln for ln in ATTRS.read_text(encoding="utf-8").splitlines() if "eo2.csv" not in ln and ln.strip()]
        if lines:
            ATTRS.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            ATTRS.unlink()

    if PRE_PUSH.is_file() and "git-lfs" in PRE_PUSH.read_text(encoding="utf-8", errors="replace"):
        PRE_PUSH.unlink()

    _strip_lfs_config()

    eo2 = ROOT / "data" / "irs" / "eo2.csv"
    if not eo2.is_file():
        print(f"Missing {eo2} — add your slim CSV, then re-run.", file=sys.stderr)
        return 1

    subprocess.run(["git", "rm", "--cached", "-f", "data/irs/eo2.csv"], cwd=ROOT, check=False)
    ho = subprocess.run(
        ["git", "hash-object", "-w", "data/irs/eo2.csv"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    oid = ho.stdout.strip()
    subprocess.run(
        ["git", "update-index", "--add", "--cacheinfo", f"100644,{oid},data/irs/eo2.csv"],
        cwd=ROOT,
        check=True,
    )
    print("Re-staged data/irs/eo2.csv as a normal Git blob (bypasses global LFS clean filter).")
    print("Next: git status  →  git commit  →  git push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
