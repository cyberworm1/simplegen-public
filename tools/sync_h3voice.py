"""Copy the h3voice modules simpleGen uses from their canonical home.

Canonical source: the external h3voice project; set H3VOICE_SRC to its h3voice/
folder. Writes src/simplegen/h3voice/ plus VENDORED.json
with each file's sha256; tests/test_singleshot.py checks both.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

FILES = ["__init__.py", "spec.py", "refs.py", "prompt.py", "lint.py", "graph.py", "schema.py", "ui.py"]
root = Path(__file__).resolve().parents[1]
if not os.environ.get("H3VOICE_SRC"):
    sys.exit("Set H3VOICE_SRC to the h3voice source folder.")
source = Path(os.environ["H3VOICE_SRC"])
target = root / "src" / "simplegen" / "h3voice"
if not source.is_dir():
    sys.exit(f"h3voice source not found: {source}")
target.mkdir(exist_ok=True)
manifest = {"source": "h3voice", "files": {}}
for name in FILES:
    shutil.copyfile(source / name, target / name)
    manifest["files"][name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
(target / "VENDORED.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"synced {len(FILES)} files from {source}")
