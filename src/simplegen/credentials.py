"""Private local RunPod configuration; never write credentials into a project."""

from __future__ import annotations

import json
import os
import stat
from importlib.resources import files
from pathlib import Path

from .settings import user_dir

REQUIRED = (
    "RUNPOD_API_KEY", "SG_ADMIN_TOKEN", "SG_S3_ACCESS_KEY", "SG_S3_SECRET_KEY",
    "SG_COMFY_USERNAME", "SG_COMFY_PASSWORD",
)


def _private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("RunPod setup directory must not be a symbolic link.")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def _exclusive_json(path: Path, value: dict) -> None:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.is_symlink() or not path.is_file():
            raise ValueError("RunPod setup path must be a regular file.") from None
        return
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def scaffold_runpod(base: Path | None = None) -> dict:
    root = Path(base).expanduser() if base is not None else user_dir() / "runpod"
    _private_directory(root)
    try:
        template = json.loads(files("simplegen").joinpath("data/global-runpod-profile.json").read_text())
    except (OSError, ValueError):
        raise ValueError("The bundled RunPod configuration template is missing.") from None
    paths = {"config_file": root / "global-profile.json", "credentials_file": root / "credentials.json",
             "schema_file": root / "object_info.json"}
    _exclusive_json(paths["config_file"], template)
    _exclusive_json(paths["credentials_file"], dict.fromkeys(
        ("RUNPOD_API_KEY", "SG_COMFY_USERNAME", "SG_COMFY_PASSWORD"), ""))
    _exclusive_json(paths["schema_file"], {})
    paths["global_profile_file"] = paths["config_file"]
    return {key: str(path.resolve()) for key, path in paths.items()}


def _read_private(path: Path) -> dict:
    try:
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError
        return data
    except (OSError, ValueError):
        raise ValueError("Cannot read private RunPod credentials; use a JSON file with mode 600.") from None


def load_credentials(settings) -> dict:
    """Environment overrides protected JSON values or {file,key} references."""
    from .global_session import _secret, is_global
    if is_global(settings):
        result = {"RUNPOD_API_KEY": _secret(settings, "RUNPOD_API_KEY")}
        if not settings.browser_login_file:
            result.update({key: _secret(settings, key) for key in
                           ("SG_COMFY_USERNAME", "SG_COMFY_PASSWORD")})
        return result
    filename = getattr(settings, "runpod_credentials_file", "")
    data = _read_private(Path(filename).expanduser()) if filename else {}
    result = {}
    for key in REQUIRED:
        value = os.environ.get(key) or data.get(key, "")
        if isinstance(value, dict) and set(value) == {"file", "key"}:
            linked = _read_private(Path(value["file"]).expanduser())
            value = linked.get(value["key"], "")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"RunPod setup is missing {key}.")
        result[key] = value.strip()
    if len(result["SG_ADMIN_TOKEN"]) < 32 or ":" in result["SG_COMFY_USERNAME"]:
        raise ValueError("RunPod control token or browser login format is invalid.")
    return result


def check_setup(settings) -> dict:
    """Local checks only; no provider requests, sessions or paid provisioning."""
    errors = []
    try:
        raw = json.loads(Path(settings.runpod_config_file).expanduser().read_text())
    except (OSError, ValueError, TypeError):
        raw = {}
    if raw.get("mode") == "global_desktop":
        from simplegen_runpod.global_gpu import load_global_profile
        try:
            load_global_profile(settings.runpod_config_file)
        except (OSError, ValueError, TypeError):
            errors.append("Complete the Global Volume desktop profile (your RunPod template ID and "
                          "Global Volume ID) before renting compute.")
        try:
            from .global_session import _secret
            _secret(settings, "RUNPOD_API_KEY")
            if settings.browser_login_file:
                login = _read_private(Path(settings.browser_login_file).expanduser())
                if not login.get("username") or not login.get("password"):
                    raise ValueError
            else:
                _secret(settings, "SG_COMFY_USERNAME")
                _secret(settings, "SG_COMFY_PASSWORD")
        except (OSError, ValueError, TypeError, KeyError):
            errors.append("Link protected RunPod API and browser credentials.")
        try:
            data = json.loads(Path(settings.runpod_schema_file).expanduser().read_text())
            schema = data.get("object_info", data)
            if not isinstance(schema, dict) or not schema:
                raise ValueError
        except (OSError, ValueError, TypeError, AttributeError):
            errors.append("Link the Global Volume ComfyUI object_info cache.")
        return {"ready": not errors, "issues": errors}
    if raw.get("schema_version") == 2:
        errors.append("Regional CPU-manager setup is deprecated for new runs. Select the Global Volume desktop profile.")
        return {"ready": False, "issues": errors}
    from simplegen_runpod.config import Config

    try:
        raw = json.loads(Path(settings.runpod_config_file).expanduser().read_text())
        policy = raw.get("policy", raw).copy()
        policy.pop("output_prefix", None)
        policy.pop("gpu_tiers", None)
        config = Config(**policy)
        config.validate()
        if not config.allow_provisioning:
            errors.append("Enable allow_provisioning in the RunPod configuration before creating compute.")
    except (OSError, ValueError, TypeError, AttributeError):
        errors.append("Complete the RunPod configuration with pinned images, volume, region and cost limits.")
    try:
        load_credentials(settings)
    except ValueError as exc:
        errors.append(str(exc))
    try:
        data = json.loads(Path(settings.runpod_schema_file).expanduser().read_text())
        schema = data.get("object_info", data)
        if not isinstance(schema, dict) or not schema or not all(isinstance(v, dict) for v in schema.values()):
            raise ValueError
    except (OSError, ValueError, TypeError, AttributeError):
        errors.append("Link a cached ComfyUI object_info JSON for the selected cloud image.")
    return {"ready": not errors, "issues": errors}
