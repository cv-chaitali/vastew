#!/usr/bin/env python3
"""
hf_upload.py — tiny CLI to create (if needed) a Hugging Face repo and upload a folder.

Examples:
  python hf_upload.py \
    --repo-id username/name \
    --folder local model \
    --repo-type model \
    --private or --public \
    --commit-message "Initial upload" \
    --token hf_xxx...    # or set env HUGGINGFACE_TOKEN and omit this
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from huggingface_hub import login

login("yourtoken")
def repo_web_url(repo_id: str, repo_type: str) -> str:
    base = "https://huggingface.co"
    if repo_type == "dataset":
        base += "/datasets"
    elif repo_type == "space":
        base += "/spaces"
    return f"{base}/{repo_id}"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Create (if needed) a HF repo and upload a local folder."
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help="Repo id like 'username/reponame' or 'org/reponame'.",
    )
    p.add_argument(
        "--folder",
        required=True,
        help="Local folder path to upload.",
    )
    p.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Type of repo to create/upload to (default: model).",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create/reconfigure repo as private.",
    )
    p.add_argument(
        "--commit-message",
        default="Upload via hf_upload.py",
        help="Commit message for the upload.",
    )
    p.add_argument(
        "--path-in-repo",
        default=None,
        help="Optional subdirectory inside the repo to place the upload.",
    )
    p.add_argument(
        "--ignore",
        nargs="*",
        default=None,
        help="Glob patterns to ignore (space-separated). Example: --ignore '*.pt' '*.bin'",
    )
    p.add_argument(
        "--allow",
        nargs="*",
        default=None,
        help="Glob patterns to include (space-separated). Mutually exclusive with --ignore.",
    )
    p.add_argument(
        "--token",
        default=os.getenv("HUGGINGFACE_TOKEN"),
        help="HF access token. If omitted, uses HUGGINGFACE_TOKEN env var or your saved login.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Set up the repo but do not upload files.",
    )
    args = p.parse_args(argv)

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        print(f"❌ Folder not found or not a directory: {folder}", file=sys.stderr)
        return 2

    if args.ignore and args.allow:
        print("❌ Use either --ignore or --allow, not both.", file=sys.stderr)
        return 2

    api = HfApi()

    # Create repo idempotently (won't error if it already exists)
    try:
        api.create_repo(
            repo_id=args.repo_id,
            private=args.private,
            repo_type=args.repo_type,
            exist_ok=True,
            token=args.token,
        )
        vis = "private" if args.private else "public"
        print(f"✅ Repo ready: {args.repo_id} [{args.repo_type}, {vis}]")
        print(f"   {repo_web_url(args.repo_id, args.repo_type)}")
    except HfHubHTTPError as e:
        # Not fatal if it already exists or we lack perms — surface the error.
        print(f"⚠️  create_repo warning: {e}", file=sys.stderr)

    if args.dry_run:
        print("🧪 Dry run enabled — skipping upload.")
        return 0

    try:
        commit_info = api.upload_folder(
            folder_path=str(folder),
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            path_in_repo=args.path_in_repo,
            token=args.token,
            commit_message=args.commit_message,
            ignore_patterns=args.ignore,
            allow_patterns=args.allow,
        )
        # commit_info fields vary across versions; printing the object is safest.
        print("🚀 Upload complete.")
        print(f"   Repo: {repo_web_url(args.repo_id, args.repo_type)}")
        print(f"   Commit: {getattr(commit_info, 'oid', getattr(commit_info, 'commit_hash', 'unknown'))}")
        return 0
    except Exception as e:
        print(f"❌ Upload failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
