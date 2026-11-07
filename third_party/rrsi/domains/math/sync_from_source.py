"""Sync the editable MathPilot subset into third_party/mathpilot.

Why this exists: RRSI drafts every candidate in a git worktree of THIS repo, so
the harness has to live inside it. The source of truth stays in the MathPilot
checkout; this copy is a working mirror. Run this before any evolution round —
a stale mirror means you are evolving last week's code.

    python domains/math/sync_from_source.py --dry-run     # show what would change
    python domains/math/sync_from_source.py               # apply

Reports three sets so drift is visible: updated, missing-in-mirror,
extra-in-mirror (files that exist only here, e.g. an edit made in the wrong
place). Nothing is deleted automatically.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIRROR = HERE.parent.parent / "third_party" / "mathpilot"
DIRS = ["agent", "utils", "contracts", "prompts", "tools"]
FILES = ["run_eval.py", "run_eval__classify.py", "run_eval__engine.py",
         "run_eval__scoring.py", "user_agent.py", "user_agent__config.py",
         "user_agent__config_base.py", "requirements.txt", "lean-toolchain"]
# 13 MB of Lean retrieval corpora; resolvable at runtime from the source tree.
SKIP_DIRS = {Path("tools") / "lean_local"}
SKIP_NAMES = {"__pycache__", ".pytest_cache"}


def source_root() -> Path:
    env = os.environ.get("MATHPILOT_SOURCE") or os.environ.get("MATHPILOT_ROOT")
    return Path(env).resolve() if env else Path(r"D:\对标leap的数学智能体项目")


def iter_source_files(src: Path):
    for d in DIRS:
        base = src / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            rel = p.relative_to(src)
            if any(part in SKIP_NAMES for part in rel.parts):
                continue
            if any(rel == s or s in rel.parents for s in SKIP_DIRS):
                continue
            if p.is_file():
                yield rel
    for name in FILES:
        p = src / name
        if p.is_file():
            yield Path(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--source", default="")
    args = ap.parse_args()

    src = Path(args.source).resolve() if args.source else source_root()
    if not (src / "run_eval.py").exists():
        print(f"[FAIL] source looks wrong (no run_eval.py): {src}")
        return 1
    if not MIRROR.exists():
        print(f"[FAIL] mirror missing: {MIRROR}")
        return 1

    updated, same, missing = [], 0, []
    for rel in sorted(iter_source_files(src)):
        s, d = src / rel, MIRROR / rel
        if d.exists() and filecmp.cmp(s, d, shallow=False):
            same += 1
            continue
        updated.append(str(rel))
        if not d.exists():
            missing.append(str(rel))
        if not args.dry_run:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)

    known = {str(r) for r in iter_source_files(src)}
    extra = sorted(str(p.relative_to(MIRROR)) for p in MIRROR.rglob("*")
                   if p.is_file()
                   and "__pycache__" not in p.parts
                   and str(p.relative_to(MIRROR)) not in known
                   and not str(p.relative_to(MIRROR)).startswith("题库"))

    print(f"source   {src}")
    print(f"mirror   {MIRROR}")
    print(f"unchanged {same}   {'would update' if args.dry_run else 'updated'} {len(updated)}")
    for rel in updated[:40]:
        print("  ~", rel)
    if len(updated) > 40:
        print(f"  ... {len(updated) - 40} more")
    if extra:
        print(f"extra-in-mirror {len(extra)} (not in source; NOT deleted):")
        for rel in extra[:15]:
            print("  +", rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
