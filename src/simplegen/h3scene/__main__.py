"""Write the approved local template: python -m simplegen.h3scene template [--refmod] <folder>."""
import argparse
import json
import uuid
from pathlib import Path

from .. import h3scene_refmod as _refmod
from ..h3scene_family import FIXED, FIXED_REFMOD, FORMAT, KIND
from . import build


def template_bytes(refmod=False):
    """The approved example, shared by the app and the developer command."""
    fixed = FIXED_REFMOD if refmod else FIXED
    spec = {**fixed, "pictures": ["placeholder.png"], "steps": 30, "output_prefix": "simpleGen/example",
            "segments": [{"join": "first", "seconds": 5, "seed": 1, "prompt": "Example"}]}
    graph = _refmod.build({**spec, "refmods": []}) if refmod else build.build(spec)[0]
    native = graph.ui()
    # The builder creates a random editor document ID. Stabilize only that
    # metadata so refreshes and CLI exports produce identical template bytes.
    name = "simplegen:h3scene-refmod:template:1" if refmod else "simplegen:h3scene:template:1"
    native["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, name))
    native["extra"]["h3scene"] = {"kind": KIND, "format": FORMAT, "profile": fixed}
    return (json.dumps(native, indent=2) + "\n").encode("utf-8")


def template(folder, *, refmod=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = "H3_Ref2VA_Scene_Chain_RefMod_LOCAL_INT8.json" if refmod else "H3_Ref2VA_Scene_Chain_LOCAL_INT8.json"
    path = folder / name
    path.write_bytes(template_bytes(refmod=refmod))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["template"])
    parser.add_argument("--refmod", action="store_true", help="write the RefMod variant")
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(template(args.folder, refmod=args.refmod))


if __name__ == "__main__":
    main()
