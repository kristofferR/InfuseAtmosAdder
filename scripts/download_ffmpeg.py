#!/usr/bin/env python3
"""Download a static FFmpeg build into the provided directory."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BASE_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
ARCHIVE_BY_ARCH = {
    "x86_64": "ffmpeg-master-latest-linux64-gpl.tar.xz",
    "amd64": "ffmpeg-master-latest-linux64-gpl.tar.xz",
    "aarch64": "ffmpeg-master-latest-linuxarm64-gpl.tar.xz",
    "arm64": "ffmpeg-master-latest-linuxarm64-gpl.tar.xz",
}
BINARIES = ("ffmpeg", "ffprobe")


def detect_archive() -> str:
    machine = platform.machine().lower()
    archive = ARCHIVE_BY_ARCH.get(machine)
    if not archive:
        supported = ", ".join(sorted(ARCHIVE_BY_ARCH))
        raise SystemExit(
            f"Unsupported architecture '{machine}'. Supported: {supported}."
        )
    return archive


def download_archive(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def extract_binaries(archive_path: Path, target_dir: Path) -> None:
    with tarfile.open(archive_path, mode="r:xz") as archive:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive.extractall(tmpdir)
            tmp_path = Path(tmpdir)
            # Assume archive contains a single top-level directory.
            top_level = next(p for p in tmp_path.iterdir() if p.is_dir())
            target_dir.mkdir(parents=True, exist_ok=True)
            for binary in BINARIES:
                source = top_level / "bin" / binary
                if not source.exists():
                    raise FileNotFoundError(f"Binary '{binary}' not found in archive")
                destination = target_dir / binary
                if destination.exists():
                    destination.unlink()
                shutil.copy2(source, destination)
                destination.chmod(0o755)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Directory where ffmpeg binaries should be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if binaries already exist.",
    )
    args = parser.parse_args(argv)

    if not args.force and all((args.target_dir / name).exists() for name in BINARIES):
        return 0

    archive_name = detect_archive()
    url = f"{BASE_URL}/{archive_name}"
    with tempfile.NamedTemporaryFile(suffix=".tar.xz", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        download_archive(url, tmp_path)
        extract_binaries(tmp_path, args.target_dir)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
