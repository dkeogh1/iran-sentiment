"""
Off-box backup of data/ to S3 (settings.BACKUP_SYNC), quant-style: `aws s3
sync` with no --delete, into a versioned bucket the writer cannot delete from.
"""

from __future__ import annotations

import os
import subprocess

from config import settings


def sync_commands(bucket: str, dry_run: bool = False) -> list[list[str]]:
    bucket = bucket.rstrip("/")
    if not bucket.startswith("s3://"):
        bucket = f"s3://{bucket}"
    cmds = []
    for local, prefix, storage in settings.BACKUP_SYNC:
        if not local.exists():
            continue
        cmd = [str(settings.AWS_BIN), "s3", "sync", str(local), f"{bucket}/{prefix}",
               "--storage-class", storage, "--only-show-errors", "--exclude", "*.partial.jsonl"]
        if dry_run:
            cmd.append("--dryrun")
        cmds.append(cmd)
    return cmds


def run(dry_run: bool = False) -> list[tuple[str, int]]:
    """Sync every BACKUP_SYNC entry; returns (prefix, exit code) per entry."""
    from dotenv import load_dotenv
    load_dotenv(settings.PROJECT_ROOT / ".env", override=True)
    bucket = os.environ.get("IRAN_BACKUP_S3_BUCKET")
    if not bucket:
        raise RuntimeError("IRAN_BACKUP_S3_BUCKET is not set in .env (e.g. iran-sentiment-backups-<account>; s3:// optional)")
    results = []
    for cmd in sync_commands(bucket, dry_run):
        rc = subprocess.run(cmd, env=os.environ.copy()).returncode
        results.append((cmd[4], rc))
    return results
