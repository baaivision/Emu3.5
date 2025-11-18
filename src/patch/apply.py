# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path

REQUIRED_VLLM_VERSION = "0.11.0"


def get_vllm_site():
    try:
        import vllm
        if getattr(vllm, "__version__", None) != REQUIRED_VLLM_VERSION:
            print(f"[FATAL] vLLM version must be {REQUIRED_VLLM_VERSION}, "
                  f"but found {vllm.__version__}. Aborting.")
            sys.exit(10)
        print(f"[INFO] vLLM version verified: {vllm.__version__}")
        return Path(vllm.__file__).parent
    except ImportError:
        print("[ERROR] vllm is not installed. Please run: pip install vllm==0.11.0")
        sys.exit(1)


def run_patch(patch_file, site_dir, dry_run=False):
    cmd = ["patch", f"-p2"]
    if dry_run:
        cmd.insert(1, "--dry-run")
    with open(patch_file, "r") as f:
        result = subprocess.run(
            cmd,
            cwd=str(site_dir),
            stdin=f,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    if not dry_run:
        print(f"[INFO] Applied patch: {patch_file}")
    if result.returncode != 0:
        print(f"[ERROR] Patch failed: {patch_file}")
        print(result.stdout)
        print(result.stderr)
    return result.returncode == 0, result.stdout, result.stderr


def extract_patch_targets(patch_file):
    targets = []
    with open(patch_file, "r") as f:
        for line in f:
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                path = line.split("\t")[0].split(" ", 1)[-1]
                if path not in ("a/dev/null", "b/dev/null"):
                    idx = len("a/vllm/")
                    targets.append(path[idx:-1])
    return list(set(targets))


def backup_files(targets, site_dir, backup_root):
    for rel in targets:
        src = site_dir / rel
        if src.exists():
            dst = backup_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Backing up {src} to {dst}")
            shutil.copy2(src, dst)


def restore_backup(backup_root, site_dir):
    if not backup_root.exists():
        print("[WARN] No backup directory found.")
        return
    for root, _, files in os.walk(backup_root):
        for f in files:
            bfile = Path(root) / f
            rel = bfile.relative_to(backup_root)
            orig = site_dir / rel
            orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bfile, orig)
    print("[INFO] Restore completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", type=str, default="./third_party/vllm/",
                        help="Directory containing .patch files")
    args = parser.parse_args()
    patch_dir = Path(args.patch_dir)

    if not patch_dir.exists() or not patch_dir.is_dir():
        print(f"[ERROR] patch-dir does not exist: {patch_dir}")
        sys.exit(1)

    site_dir = get_vllm_site()
    print(f"[INFO] vLLM site-packages: {site_dir}")

    patch_files = sorted(p for p in patch_dir.rglob("*.patch"))
    if not patch_files:
        print("[ERROR] No patch files found.")
        sys.exit(1)

    print(f"[INFO] Found {len(patch_files)} patch(es).")

    # Backup root folder
    backup_root = site_dir.parent / "vllm_patch_backup"
    if backup_root.exists():
        print("[WARN] Removing previous backup...")
        shutil.rmtree(backup_root)
    backup_root.mkdir(parents=True)

    print("[INFO] Running dry-run...")
    for p in patch_files:
        ok, out, err = run_patch(p, site_dir, dry_run=True)
        if not ok:
            print(f"[FATAL] Dry-run failed for patch: {p}\n{err}")
            sys.exit(2)
    print("[INFO] Dry-run passed.")

    print("[INFO] Backing up modified files...")
    for p in patch_files:
        targets = extract_patch_targets(p)
        backup_files(targets, site_dir, backup_root)

    print("[INFO] Applying patches...")
    for p in patch_files:
        ok, out, err = run_patch(p, site_dir, dry_run=False)
        if not ok:
            print(f"[ERROR] Failed to apply patch: {p}\n{err}")
            print("[INFO] Restoring from backup...")
            restore_backup(backup_root, site_dir)
            sys.exit(3)
    print("[SUCCESS] All patches applied successfully.")
    print(f"[INFO] Backup stored at: {backup_root}")

if __name__ == "__main__":
    main()
