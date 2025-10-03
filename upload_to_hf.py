#!/usr/bin/env python3
"""Upload a local file or directory to a private Hugging Face organization repository."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a file or folder to a private repository under a Hugging Face organization."
        )
    )
    parser.add_argument(
        "src",
        type=Path,
        help="Path to the local file or directory to upload.",
    )
    parser.add_argument(
        "repo_name",
        help="Name of the repository to create or update (without organization prefix).",
    )
    parser.add_argument(
        "--org",
        required=True,
        help="Hugging Face organization where the repository lives.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset", "space"),
        help="Type of Hugging Face repository to target (default: model).",
    )
    parser.add_argument(
        "--token",
        help="User access token; falls back to HF_TOKEN environment variable if omitted.",
    )
    parser.add_argument(
        "--commit-message",
        help="Optional commit message to use for the upload.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch or revision to push to (default: main).",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help=(
            "Destination path inside the repository. When uploading a file the default is the file name; "
            "for folders the default is the repository root."
        ),
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional glob patterns of files to include when uploading a folder.",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=None,
        help="Optional glob patterns of files to skip when uploading a folder.",
    )
    parser.add_argument(
        "--delete-patterns",
        nargs="*",
        default=None,
        help="Optional glob patterns of repo files to delete when uploading a folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_path = args.src.expanduser().resolve()
    if not src_path.exists():
        sys.exit(f"Path not found: {src_path}")

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        sys.exit("Hugging Face token missing. Supply --token or set HF_TOKEN.")

    repo_id = f"{args.org.strip('/')}/{args.repo_name.strip()}"

    api = HfApi()
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type=args.repo_type,
            private=True,
            exist_ok=True,
            token=token,
        )
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Failed to create or access repository {repo_id}: {exc}")

    commit_message = args.commit_message or f"Upload {src_path.name}"

    try:
        if src_path.is_file():
            target_path = args.path_in_repo or src_path.name
            api.upload_file(
                path_or_fileobj=str(src_path),
                path_in_repo=target_path,
                repo_id=repo_id,
                repo_type=args.repo_type,
                token=token,
                revision=args.revision,
                commit_message=commit_message,
            )
        else:
            api.upload_folder(
                folder_path=str(src_path),
                path_in_repo=args.path_in_repo,
                repo_id=repo_id,
                repo_type=args.repo_type,
                token=token,
                revision=args.revision,
                commit_message=commit_message,
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
                delete_patterns=args.delete_patterns,
            )
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Upload failed: {exc}")

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"Uploaded {src_path} to {repo_url} ({args.repo_type}, private)")


if __name__ == "__main__":
    main()
