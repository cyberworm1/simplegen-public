"""Small, portable user settings; credentials stay with the CLI or environment."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from urllib.parse import urlsplit

from .dimensions import ASPECT_RATIOS


def user_dir() -> Path:
    return Path(os.environ.get("SIMPLEGEN_HOME", Path.home() / ".simplegen")).expanduser()


def atomic_json(path: Path, data: object) -> None:
    if isinstance(data, dict) and "run_id" in data and "schema_version" in data:
        from .h3scene_compat import normalize_state
        data = normalize_state(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@dataclass
class Settings:
    project_folder: str = ""
    template_folder: str = ""
    template_file: str = ""
    output_folder: str = ""
    provider: str = "codex"
    model_tier: str = "Economy"
    model: str = ""
    reasoning: str = "low"
    steps: int | None = None  # Historical runs retain their accepted workflow settings.
    megapixels: float = 0.25
    duration: float = 15.0
    aspect_ratio: str = ""  # Older saved runs retain their template's dimensions.
    quantity: int = 1
    comfy_url: str = "http://127.0.0.1:8188"
    cloud: bool = False
    cloud_name: str = "RunPod"
    token_env: str = "SIMPLEGEN_COMFY_TOKEN"
    browser_login_file: str = ""
    connection_profiles: dict = field(default_factory=dict)
    managed_runpod: bool = False
    global_desktop: bool = False
    runpod_config_file: str = field(default_factory=lambda: str(user_dir() / "runpod/global-profile.json"))
    runpod_credentials_file: str = field(default_factory=lambda: str(user_dir() / "runpod/credentials.json"))
    runpod_schema_file: str = field(default_factory=lambda: str(user_dir() / "runpod/object_info.json"))
    runpod_gpu_tiers: list[str] = field(default_factory=list)
    runpod_default_workflow: str = "H3_Persistent_References_Clonable_Chain.json"
    runpod_default_model: str = "minimax_h3_ref2va_bf16.safetensors"
    runpod_default_clip: str = "qwen3vl_32b_h3_ultra_uncensored_heretic_bf16.safetensors"
    agent_timeout: int | None = None  # Legacy saved-run field; agent duration is unrestricted.
    generation_timeout: int = 21600
    preflight_min_free_gb: float = 5.0
    # Managed RunPod cost safety (Phase 5). 0 disables failure shutdown.
    failure_shutdown_minutes: int = 15
    startup_timeout_minutes: int = 45
    alert_webhook_url: str = ""
    # Phase 4: empty-queue window after verified downloads (keep-warm for additions/handoff).
    keep_warm_minutes: int = 5
    # Phase 8: 4K/60 upscale on a separate host. The token lives in a private file.
    upscale: bool = False
    upscale_url: str = ""
    upscale_framing: str = "long_side"
    upscale_framing_by_aspect: dict = field(default_factory=dict)
    upscale_preset: str = "remacri+rife"  # upscale.DEFAULT_METHOD; empty (older files): the host's default.

    def __post_init__(self) -> None:
        from .builtin_workflows import resolve_builtin_template
        self.template_file = resolve_builtin_template(self.template_file, self.template_folder)
        self.connection_profiles = {
            name: {**profile, "template_file": resolve_builtin_template(
                profile["template_file"], profile.get("template_folder", ""))}
            if isinstance(profile, dict) and "template_file" in profile else profile
            for name, profile in self.connection_profiles.items()
        }
        if self.managed_runpod is True:
            self.cloud = True

    @classmethod
    def defaults(cls) -> Settings:
        docs = Path.home() / "Documents"
        # No Workflow folder by default: the built-in workflows need none.
        return cls(
            steps=25,
            aspect_ratio="16:9",
            output_folder=str(docs / "simpleGen Projects"),
        )

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        path = path or user_dir() / "settings.json"
        if not path.exists():
            return cls.defaults()
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {field.name for field in fields(cls)}
        values = asdict(cls.defaults())
        values.update({k: v for k, v in data.items() if k in known})
        return cls(**values)

    def save(self, path: Path | None = None) -> None:
        self.validate(require_folders=False)
        atomic_json(path or user_dir() / "settings.json", asdict(self))

    def reference_roots(self) -> list[Path]:
        """Folders references may come from: the project, plus the Workflow folder when one is linked.

        An unlinked Workflow folder (a built-in workflow) adds nothing; an empty path would
        otherwise resolve to the app's working directory.
        """
        roots = [Path(self.project_folder).expanduser().resolve()]
        if self.template_folder.strip():
            roots.append(Path(self.template_folder).expanduser().resolve())
        return roots

    def validate(self, *, require_folders: bool = True) -> None:
        if self.steps is not None and (type(self.steps) is not int or not 1 <= self.steps <= 100):
            raise ValueError("Steps must be a whole number between 1 and 100.")
        if type(self.quantity) is not int or not 1 <= self.quantity <= 100:
            raise ValueError("Qty must be a whole number between 1 and 100.")
        if self.provider not in ("codex", "claude", "grok"):
            raise ValueError("Choose Codex, Claude Code, or Grok Build.")
        if self.model_tier not in ("Economy", "Balanced", "Strongest", "Custom"):
            raise ValueError("Choose a model tier.")
        if not math.isfinite(self.megapixels) or not 0.05 <= self.megapixels <= 16:
            raise ValueError("Megapixel must be between 0.05 and 16.")
        if not math.isfinite(self.duration) or not 1 <= self.duration <= 3600:
            raise ValueError("Duration must be between 1 and 3600 seconds.")
        if not isinstance(self.aspect_ratio, str) or (
            self.aspect_ratio != "" and self.aspect_ratio not in dict(ASPECT_RATIOS)
        ):
            raise ValueError("Choose a supported aspect ratio.")
        if type(self.managed_runpod) is not bool:
            raise ValueError("Managed RunPod must be enabled or disabled.")
        if self.managed_runpod and not self.cloud:
            raise ValueError("Managed RunPod requires cloud generation.")
        if not isinstance(self.runpod_gpu_tiers, list) or not all(
            isinstance(tier, str) and tier.strip() for tier in self.runpod_gpu_tiers
        ):
            raise ValueError("RunPod GPU tiers must be a list of GPU names.")
        for name in ("runpod_config_file", "runpod_credentials_file", "runpod_schema_file"):
            if not isinstance(getattr(self, name), str):
                raise ValueError("RunPod settings must contain file paths, not credential data.")  # noqa: TRY004 — validation boundary
        if not self.managed_runpod:
            endpoint = urlsplit(self.comfy_url)
            _ = endpoint.port
            if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
                raise ValueError("ComfyUI address must be an http:// or https:// URL.")
            if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
                raise ValueError("Use a plain ComfyUI URL; configure credentials separately.")
            if self.browser_login_file and endpoint.scheme != "https":
                raise ValueError("Browser login credentials require an HTTPS ComfyUI address.")
        if (isinstance(self.preflight_min_free_gb, bool)
                or not isinstance(self.preflight_min_free_gb, (int, float))
                or not math.isfinite(self.preflight_min_free_gb) or self.preflight_min_free_gb < 0):
            raise ValueError("Preflight minimum free space must be a non-negative number of GB.")
        if type(self.failure_shutdown_minutes) is not int or not 0 <= self.failure_shutdown_minutes <= 240:
            raise ValueError("Failure shutdown must be a whole number of minutes between 0 (off) and 240.")
        if type(self.startup_timeout_minutes) is not int or self.startup_timeout_minutes < 5:
            raise ValueError("Startup timeout must be a whole number of at least 5 minutes.")
        if type(self.keep_warm_minutes) is not int or not 5 <= self.keep_warm_minutes <= 60:
            raise ValueError("Keep-warm window must be a whole number of minutes between 5 and 60.")
        if not isinstance(self.alert_webhook_url, str) or (
                self.alert_webhook_url.strip()
                and not self.alert_webhook_url.strip().lower().startswith(("https://", "http://"))):
            raise ValueError("Alert webhook must be empty or an http(s) URL.")
        if type(self.upscale) is not bool:
            raise ValueError("4K60 upscale must be enabled or disabled.")
        if not isinstance(self.upscale_url, str) or (
                self.upscale_url.strip() and not _plain_http_url(self.upscale_url.strip())):
            raise ValueError("Upscale service must be empty or a plain http(s) URL.")
        from .upscale import FRAMINGS
        framings = dict(FRAMINGS)
        if self.upscale_framing not in framings:
            raise ValueError("Choose a supported 4K framing.")
        if not isinstance(self.upscale_framing_by_aspect, dict) or not all(
                key in dict(ASPECT_RATIOS) and value in framings
                for key, value in self.upscale_framing_by_aspect.items()):
            raise ValueError("4K framing overrides must map aspect ratios to supported framings.")
        if not isinstance(self.upscale_preset, str):
            raise ValueError("Upscale preset must be text.")  # noqa: TRY004 — validation boundary
        if self.generation_timeout < 10:
            raise ValueError("Generation timeout must be at least 10 seconds.")
        if require_folders:
            from .builtin_workflows import is_builtin_template
            folders = [("Project", self.project_folder)]
            if not is_builtin_template(self.template_file, self.template_folder):
                folders.append(("Template", self.template_folder))
            for label, value in folders:
                if not value or not Path(value).expanduser().is_dir():
                    raise ValueError(f"Choose an existing {label.lower()} folder.")
            if not self.output_folder.strip():
                raise ValueError("Choose an output folder.")

    def public_dict(self) -> dict:
        """Settings saved into project and batch state. The alert webhook is a
        shared secret for ntfy-style topics and lives only in global settings."""
        values = asdict(self)
        values.pop("alert_webhook_url", None)
        return values


def _plain_http_url(value: str) -> bool:
    try:
        parts = urlsplit(value)
        _ = parts.port
    except ValueError:
        return False
    return (parts.scheme in ("http", "https") and bool(parts.hostname) and not parts.username
            and not parts.password and not parts.query and not parts.fragment)


def with_generation_location(settings: Settings, cloud: bool) -> Settings:
    """Keep endpoint, authentication and workflow choices separate for local and cloud use.

    Profiles saved before workflows were per destination lack the workflow keys:
    the current workflow stays with the destination being left, and the other
    destination starts without a workflow (Cloud falls back to its default).
    """
    if cloud == settings.cloud:
        return replace(settings)
    keys = ("comfy_url", "token_env", "browser_login_file", "managed_runpod", "global_desktop", "cloud_name",
            "template_folder", "template_file")
    profiles = dict(settings.connection_profiles)
    profiles["cloud" if settings.cloud else "local"] = {
        key: getattr(settings, key) for key in keys
    }
    defaults = {
        "comfy_url": "" if cloud else "http://127.0.0.1:8188",
        "token_env": "" if cloud else "SIMPLEGEN_COMFY_TOKEN",
        "browser_login_file": "",
        "managed_runpod": cloud,
        "global_desktop": cloud,
        "cloud_name": "RunPod",
        # The folder is only a location; no workflow is chosen for the user.
        "template_folder": settings.template_folder,
        "template_file": settings.runpod_default_workflow if cloud else "",
    }
    target = profiles.get("cloud" if cloud else "local", defaults)
    values = {key: target.get(key, defaults[key]) for key in keys}
    if not cloud:
        values["managed_runpod"] = False
        values["global_desktop"] = False
    return replace(settings, cloud=cloud, connection_profiles=profiles, **values)


def with_runpod_config(settings: Settings, folder: Path) -> Settings:
    """Read connection metadata only; no lifecycle calls or credential copies."""
    folder = folder.expanduser().resolve()
    template = folder / "global-comfy-template.json"
    if template.is_file():
        from simplegen_runpod.global_gpu import load_global_profile

        from .credentials import scaffold_runpod
        try:
            metadata = json.loads(template.read_text(encoding="utf-8"))
            login_file = folder / "browser-login.json"
            schema_file = folder / "global-global-object-info.json"
            if not login_file.is_file() or not schema_file.is_file():
                raise ValueError
            profile_path = Path(scaffold_runpod()["global_profile_file"])
            # The folder's template must be the one the user's profile deploys.
            if metadata.get("id") != load_global_profile(profile_path).template_id:
                raise ValueError
        except (OSError, ValueError, TypeError, KeyError):
            raise ValueError("Global Volume setup needs its template, login and schema cache.") from None
        candidate = replace(settings, cloud=True, cloud_name="RunPod", managed_runpod=True,
                            global_desktop=True, browser_login_file=str(login_file),
                            runpod_config_file=str(profile_path), runpod_schema_file=str(schema_file))
        candidate.validate(require_folders=False)
        return candidate
    try:
        session = json.loads((folder / "gpu-session.json").read_text(encoding="utf-8"))
        url = session["url"]
        pod_id = session["pod_id"]
        if not isinstance(pod_id, str) or not pod_id.isalnum():
            raise ValueError
        if url != f"https://{pod_id}-8189.proxy.runpod.net":
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError("Choose a runpodConfig folder with a valid gpu-session.json.") from None
    login_file = folder / "browser-login.json"
    if not login_file.is_file():
        raise ValueError("The selected folder is missing browser-login.json.")
    candidate = replace(
        settings,
        comfy_url=url,
        cloud=True,
        cloud_name="RunPod",
        browser_login_file=str(login_file),
        managed_runpod=False,
    )
    candidate.validate(require_folders=False)
    return candidate
