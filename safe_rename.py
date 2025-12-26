#!/usr/bin/env python3
"""
safe_rename.py

Recursively sanitize filenames (and optionally folders) under a root directory
so Windows tools & scripts won’t choke on characters like %, #, =, etc.

Rules:
- Allowed characters: letters, digits, dot, underscore, hyphen  [A-Za-z0-9._-]
- Everything else becomes "_"
- Collapse repeated "_" and strip leading/trailing "_" and spaces
- Strip trailing dots/spaces (invalid on Windows)
- Preserve the last extension (e.g., .mp4, .wav, .csv)
- Avoid Windows reserved names (CON, PRN, AUX, NUL, COM1..9, LPT1..9)
- Ensure uniqueness in each directory: add _1, _2, ... if needed
- Optionally sanitize directory names too (use --rename-dirs)

Usage:
    python safe_rename.py "D:\path\to\root" --rename-dirs --dry-run
    python safe_rename.py "D:\path\to\root" --log rename_map.csv
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

# Windows reserved device names (case-insensitive, with optional extension)
_RESERVED = {
    "con","prn","aux","nul",
    "com1","com2","com3","com4","com5","com6","com7","com8","com9",
    "lpt1","lpt2","lpt3","lpt4","lpt5","lpt6","lpt7","lpt8","lpt9",
}

SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
UNDERSCORE_RUNS = re.compile(r"_+")

def sanitize_stem(stem: str) -> str:
    """
    Sanitize the filename stem (without extension).
    """
    # Replace unsafe chars with '_'
    s = SAFE_CHARS_RE.sub("_", stem)
    # Collapse consecutive underscores
    s = UNDERSCORE_RUNS.sub("_", s)
    # Strip leading/trailing underscores and spaces
    s = s.strip(" _")
    # Strip trailing dots and spaces (Windows hates them)
    s = s.rstrip(". ")
    if not s:
        s = "file"
    # Avoid reserved device names (case-insensitive)
    if s.lower() in _RESERVED:
        s = f"_{s}"
    return s

def unique_name_in_dir(dirpath: Path, candidate: str, used: set) -> str:
    """
    Ensure candidate filename is unique in dirpath. If exists or would collide
    with other renames, append _1, _2, ...
    """
    base, dot, ext = candidate.partition(".")
    # Keep the dot+ext if present (ext may be empty)
    suffix = f".{ext}" if dot else ""
    final = candidate
    idx = 1
    while (dirpath / final).exists() or final.lower() in used:
        final = f"{base}_{idx}{suffix}"
        idx += 1
    used.add(final.lower())
    return final

def plan_file_renames(root: Path) -> Dict[Path, Path]:
    """
    Compute file rename plan (Path old -> Path new), preserving extension.
    """
    plan: Dict[Path, Path] = {}
    # Process files directory-by-directory to resolve collisions locally
    for dirpath in sorted({p.parent for p in root.rglob("*") if p.is_file()}):
        used = {p.name.lower() for p in dirpath.iterdir() if p.is_file()}
        # We'll add new names we create to 'used' to avoid collisions
        for p in sorted([x for x in dirpath.iterdir() if x.is_file()]):
            old_name = p.name
            ext = p.suffix  # includes leading dot, e.g. ".mp4"
            stem = p.stem
            safe_stem = sanitize_stem(stem)
            safe_name = safe_stem + ext  # keep extension verbatim
            # If unchanged, skip
            if safe_name == old_name:
                continue
            # Ensure unique
            safe_name = unique_name_in_dir(dirpath, safe_name, used)
            plan[p] = dirpath / safe_name
    return plan

def plan_dir_renames(root: Path) -> Dict[Path, Path]:
    """
    Compute directory rename plan (Path old -> Path new).
    Important: rename deepest-first to avoid breaking paths mid-walk.
    """
    plan: Dict[Path, Path] = {}
    # Gather all dirs (excluding root), deepest-first
    dirs = sorted([d for d in root.rglob("*") if d.is_dir()], key=lambda p: len(p.parts), reverse=True)
    for d in dirs:
        parent = d.parent
        old_name = d.name
        safe_name = sanitize_stem(old_name)
        if safe_name == old_name:
            continue
        # Ensure uniqueness among sibling directories
        siblings = {x.name.lower() for x in parent.iterdir() if x.is_dir()}
        used = set(siblings)
        # reserve currently planned targets too
        for oldp, newp in plan.items():
            if newp.parent == parent:
                used.add(newp.name.lower())
        final = safe_name
        idx = 1
        while (parent / final).exists() or final.lower() in used:
            final = f"{safe_name}_{idx}"
            idx += 1
        plan[d] = parent / final
    return plan

def write_log(log_path: Path, mapping: Dict[Path, Path]):
    import csv
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old_path", "new_path"])
        for old, new in sorted(mapping.items()):
            w.writerow([str(old), str(new)])

def main():
    ap = argparse.ArgumentParser(description="Recursively sanitize & rename files (and optionally folders) to Windows-friendly names.")
    ap.add_argument("root", help="Root folder to process")
    ap.add_argument("--rename-dirs", action="store_true", help="Also sanitize & rename directories")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be renamed, but do not change anything")
    ap.add_argument("--log", help="Optional CSV to log rename mapping (old_path,new_path)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[ERROR] Root folder not found: {root}")

    total_plan: Dict[Path, Path] = {}

    if args.rename_dirs:
        dir_plan = plan_dir_renames(root)
        total_plan.update(dir_plan)

    file_plan = plan_file_renames(root)
    total_plan.update(file_plan)

    if not total_plan:
        print("[INFO] Nothing to rename. All names look safe.")
        return

    print(f"[INFO] Planned renames: {len(total_plan)}")
    for old, new in sorted(total_plan.items()):
        print(f"  {old}  ->  {new}")

    if args.log:
        write_log(Path(args.log), total_plan)
        print(f"[INFO] Wrote log: {Path(args.log).resolve()}")

    if args.dry_run:
        print("[DRY-RUN] No changes were made.")
        return

    # Apply renames: directories deepest-first, files arbitrary (paths independent)
    # If we have dir renames, ensure we rename those first (they are deepest-first already).
    # Build ordered list: (dirs first if present)
    ops = []
    if args.rename_dirs:
        # Only directory mappings
        ops.extend(sorted([kv for kv in total_plan.items() if kv[0].is_dir()],
                          key=lambda kv: len(kv[0].parts), reverse=True))
    # Then files
    ops.extend([kv for kv in total_plan.items() if kv[0].is_file()])

    # Perform moves
    errors = 0
    for old, new in ops:
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
        except Exception as e:
            print(f"[ERROR] Failed to rename: {old} -> {new} :: {e}")
            errors += 1

    if errors:
        print(f"[DONE] Completed with {errors} error(s).")
    else:
        print("[DONE] All renames applied successfully.")

if __name__ == "__main__":
    main()
