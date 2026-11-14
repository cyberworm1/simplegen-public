"""Build on the destination OS/architecture; no user assets enter the bundle."""

import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
probe = shutil.which("ffprobe")
if not probe:
    raise SystemExit("Install FFmpeg (including ffprobe) before building.")
args = [
    sys.executable,
    "-m",
    "PyInstaller",
    "--noconfirm",
    "--windowed",
    "--onedir",
    "--name",
    "simpleGen",
    "--paths",
    str(root / "src"),
    "--distpath",
    str(root / "dist"),
    "--workpath",
    str(root / "build"),
    "--specpath",
    str(root / "build"),
    "--add-data",
    f"{root / 'src/simplegen/skills'}:simplegen/skills",
    "--add-data",
    f"{root / 'src/simplegen/data'}:simplegen/data",
    "--collect-all",
    "boto3",
    "--collect-all",
    "botocore",
    "--add-binary",
    f"{probe}:.",
    str(root / "tools" / "entry.py"),
]
if sys.platform == "darwin":
    args += ["--osx-bundle-identifier", "local.simplegen.desktop"]
subprocess.run(args, cwd=root, check=True)
