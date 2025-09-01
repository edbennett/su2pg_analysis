#!/usr/bin/env python3

"""
Tools for annotating output with provenance information.
"""

from datetime import datetime, timezone
import hashlib
import os
import pathlib
import psutil
import socket
import subprocess


def get_commit_id():
    git_result = subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        capture_output=True,
    )
    if git_result.returncode != 0:
        return "[No commit ID available]"

    return git_result.stdout.decode().strip()


def sha256_file(filename):
    with open(filename, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def get_basic_metadata(*data_filenames):
    now = datetime.now(timezone.utc).isoformat()
    metadata = {}
    metadata["_comment"] = (
        "This file and all the files in this directory were generated automatically. "
        "Do not modify them; re-run the analysis workflow!"
    )
    metadata["workflow_run"] = {
        "completed": now,
        "user_name": os.getlogin(),
        "machine_name": socket.gethostname(),
    }
    metadata["analysis_code"] = {"version": get_commit_id()}
    metadata["workflow_step"] = {
        "command_line": " ".join(psutil.Process(os.getpid()).cmdline()),
        "completed": now,
    }
    if data_filenames:
        metadata["input_data"] = {
            filename: {
                "filename": filename,
                "last_updated": datetime.fromtimestamp(
                    pathlib.Path(filename).stat().st_mtime, timezone.utc
                ).isoformat(),
                "sha256": sha256_file(filename),
            }
            for filename in data_filenames
        }

    return metadata


def flatten_metadata(metadata):
    flat_metadata = {
        "_comment": (
            "This file was generated automatically. "
            "Do not modify it directly; re-run the analysis workflow!"
        )
    }

    for outer_key, inner_metadata in metadata.items():
        if outer_key == "_comment":
            continue
        flat_metadata.update(
            {f"{outer_key}::{k}": v for k, v in inner_metadata.items()}
        )

    flat_metadata["generated"] = flat_metadata["workflow_run::completed"]
    del flat_metadata["workflow_run::completed"]
    return flat_metadata


def get_flat_metadata():
    return flatten_metadata(get_basic_metadata())


def text_metadata(metadata, comment_char="#"):
    return "\n".join(
        [
            f"{comment_char} {k}: {v.replace('\n', '\n{comment_char} ')}"
            for k, v in flatten_metadata(metadata).items()
        ]
    )
