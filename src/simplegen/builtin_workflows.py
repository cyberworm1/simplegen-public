"""App-owned workflow examples; user workflow folders remain independent."""

import logging
from pathlib import Path

from .h3scene_compat import LEGACY_FILENAME, legacy_template_bytes
from .settings import user_dir

SCENE_LABEL = "H3 Scene Chain — Local INT8 (built-in)"
SCENE_FILENAME = "H3_Ref2VA_Scene_Chain_LOCAL_INT8.json"
REFMOD_LABEL = "H3 Scene Chain + RefMod — Local INT8 (built-in)"
REFMOD_FILENAME = "H3_Ref2VA_Scene_Chain_RefMod_LOCAL_INT8.json"


# Workflow templates shipped in the package and installed beside the scene chain.
BUNDLED_FOLDER = Path(__file__).parent / "data" / "workflows"
BUNDLED = (
    ("H3_Persistent_References_Clonable_Chain.json",
     "H3 Persistent References Chain — Cloud BF16 (built-in)", "H3 Persistent References Chain"),
    ("H3_Persistent_References_Clonable_Chain_LOCAL_INT8_NVFP4.json",
     "H3 Persistent References Chain — Local INT8/NVFP4 (built-in)", "H3 Persistent References Chain"),
    ("H3_Ref2VA_Voice_SingleShot.json",
     "H3 Voice Single Shot — Cloud BF16 (built-in)", "H3 Voice Single Shot"),
    ("H3_Ref2VA_Voice_SingleShot_LOCAL_INT8.json",
     "H3 Voice Single Shot — Local INT8 (built-in)", "H3 Voice Single Shot"),
)


def bundled_template_path(filename: str) -> Path:
    return (user_dir() / "workflows/builtin" / filename).absolute()


def scene_template_path() -> Path:
    return (user_dir() / "workflows/builtin" / SCENE_FILENAME).absolute()


def refmod_template_path() -> Path:
    return (user_dir() / "workflows/builtin" / REFMOD_FILENAME).absolute()


def builtin_name(path: str | Path) -> str:
    name = Path(path).name
    bundled = {filename: title for filename, _, title in BUNDLED}
    if name in bundled:
        return bundled[name]
    return "H3 Scene Chain + RefMod" if name == REFMOD_FILENAME else "H3 Scene Chain"


def template_label(path: str | Path) -> str:
    name = Path(path).name
    return SCENE_FILENAME if name == LEGACY_FILENAME else name


def resolve_builtin_template(value: str | Path, folder: str = "") -> str:
    """Resolve earlier app selections while keeping user exports independent."""
    if not value:
        return str(value)
    path = Path(value).expanduser()
    current = scene_template_path().resolve()
    if path.is_absolute() and path.resolve() == current.with_name(LEGACY_FILENAME):
        return str(current)
    if str(value) == LEGACY_FILENAME and (
        not folder or Path(folder).expanduser().resolve() == current.parent
        or not (Path(folder).expanduser() / LEGACY_FILENAME).exists()
    ):
        return str(current)
    return str(value)


def is_builtin_template(value: str | Path, folder: str = "") -> bool:
    """Only the app-owned path is exempt from the linked-folder requirement."""
    if not value:
        return False
    path = Path(resolve_builtin_template(value, folder)).expanduser()
    owned = {scene_template_path().resolve(), refmod_template_path().resolve(),
             *(bundled_template_path(filename).resolve() for filename, _, _ in BUNDLED)}
    return path.is_absolute() and path.resolve() in owned


def ensure_scene_template() -> Path | None:
    """Refresh atomically when changed; an unavailable home never breaks the UI."""
    from .h3scene.__main__ import template_bytes
    from .scheduling import atomic_bytes

    path = scene_template_path()
    try:
        payload = template_bytes()
        if not path.is_file() or path.read_bytes() != payload:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_bytes(path, payload)
        stale = path.with_name(LEGACY_FILENAME)
        try:
            if not stale.is_symlink() and stale.is_file() and stale.read_bytes() == legacy_template_bytes(payload):
                stale.unlink()
        except OSError as error:
            logging.getLogger(__name__).warning("Earlier built-in workflow cleanup unavailable: %s", error)
        return path
    except (OSError, ValueError, RuntimeError) as error:
        logging.getLogger(__name__).warning("Built-in H3 Scene Chain workflow unavailable at %s: %s", path, error)
        return None


def ensure_refmod_template() -> Path | None:
    """The RefMod variant, refreshed the same way; the scene-chain template is independent."""
    from .h3scene.__main__ import template_bytes
    from .scheduling import atomic_bytes

    path = refmod_template_path()
    try:
        payload = template_bytes(refmod=True)
        if not path.is_file() or path.read_bytes() != payload:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_bytes(path, payload)
        return path
    except (OSError, ValueError, RuntimeError) as error:
        logging.getLogger(__name__).warning("Built-in H3 Scene Chain + RefMod workflow unavailable at %s: %s",
                                            path, error)
        return None


def ensure_bundled_templates() -> list[tuple[str, Path]]:
    """Install the packaged templates; each one refreshes atomically and fails alone."""
    from .scheduling import atomic_bytes

    installed = []
    for filename, label, _ in BUNDLED:
        path = bundled_template_path(filename)
        try:
            payload = (BUNDLED_FOLDER / filename).read_bytes()
            if not path.is_file() or path.read_bytes() != payload:
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_bytes(path, payload)
            installed.append((label, path))
        except OSError as error:
            logging.getLogger(__name__).warning("Built-in workflow %s unavailable at %s: %s", filename, path, error)
    return installed
