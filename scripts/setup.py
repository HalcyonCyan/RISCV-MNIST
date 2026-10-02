"""Create a local venv, install Pillow, and fetch a pinned, verified RARS."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import venv
import zipfile

ROOT = Path(__file__).resolve().parent.parent
RARS_URL = "https://github.com/TheThirdOne/rars/releases/download/v1.6/rars1_6.jar"
RARS_SHA256 = "780f730eb457b1ba609e968accc2c8b77d8f92c3d9dbf30cc7fdb3cfb14e8c24"


def download_rars() -> None:
    destination = ROOT / "tools" / "rars.jar"
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == RARS_SHA256:
        print("RARS 1.6 already verified.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jar.part")
    request = urllib.request.Request(RARS_URL, headers={"User-Agent": "riscv-mnist"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != RARS_SHA256 or not zipfile.is_zipfile(temporary):
            raise ValueError("RARS download did not match the pinned SHA-256")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print("RARS 1.6 downloaded and verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rars-only", action="store_true", help="Only prepare tools/rars.jar")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.exit(1, "Python 3.10 or newer is required.\n")
    if not args.rars_only:
        folder = ROOT / ".venv"
        executable = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not executable.is_file():
            venv.EnvBuilder(with_pip=True).create(folder)
        subprocess.run([str(executable), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    download_rars()
    print("Ready. Run .venv\\Scripts\\python.exe app.py (Windows) or .venv/bin/python app.py (macOS/Linux).")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Setup failed: {exc}")
