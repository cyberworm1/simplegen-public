"""Build the ComfyUI API graph for one Ref2VA voice shot.

Node ids are readable strings ("r2v", "voice_narrator", ...) so a person or an
agent can find and patch nodes in the API JSON. Only literal inputs the recipe
cares about are set here; schema.complete() fills every other widget from the
target's /object_info, so node-version drift between local and RunPod does
not require code changes.

To extend the recipe (a LoRA, a sigma shift, an upscaler...), add nodes in
build() and rewire with G.link(). Nothing else needs to know about them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import refs as R
from .spec import Spec


@dataclass
class Graph:
    nodes: dict[str, dict] = field(default_factory=dict)
    lanes: dict[str, int] = field(default_factory=dict)  # layout column for the UI export

    def add(self, node_id: str, class_type: str, title: str, lane: int, **inputs: Any) -> str:
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id {node_id}")
        self.nodes[node_id] = {"class_type": class_type, "inputs": dict(inputs), "_meta": {"title": title}}
        self.lanes[node_id] = lane
        return node_id

    @staticmethod
    def link(node_id: str, slot: int = 0) -> list:
        return [node_id, slot]

    def api(self) -> dict:
        return {k: {**v, "inputs": dict(v["inputs"])} for k, v in self.nodes.items()}


def resolution(render: dict) -> tuple[int, int]:
    """Explicit width/height win; otherwise megapixels (x1024^2) at an aspect, snapped to 32.

    Matches the ResolutionSelector table in the official template
    (0.98 MP 16:9 -> 1344x768, 0.5 MP 16:9 -> 960x544).
    """
    if render.get("width") and render.get("height"):
        w, h = int(render["width"]), int(render["height"])
    else:
        a, b = (float(x) for x in str(render.get("aspect", "16:9")).split(":"))
        area = float(render.get("megapixels", 0.98)) * 1024 * 1024
        w = math.sqrt(area * a / b)
        h = w * b / a
        w, h = int(round(w / 32) * 32), int(round(h / 32) * 32)
    if w % 32 or h % 32:
        raise ValueError(f"width/height must be multiples of 32, got {w}x{h}")
    return w, h


def merged_render(profile: dict, spec: Spec) -> dict:
    render = dict(profile.get("render", {}))
    if {"width", "height"} & set(spec.render):
        render.pop("megapixels", None)
    if {"megapixels", "aspect"} & set(spec.render):
        render.pop("width", None)
        render.pop("height", None)
    render.update(spec.render)
    return render


def build(spec: Spec, profile: dict, prompt: str, files: dict[str, str] | None = None) -> Graph:
    """files maps picture/video/voice id -> filename as the target ComfyUI sees it."""
    files = files or {}
    refs = R.assign(spec)
    render = merged_render(profile, spec)
    width, height = resolution(render)
    G = Graph()
    link = G.link

    # Models (lane 0). Loader classes and filenames come from the profile.
    models = profile["models"]
    for key in ("unet", "clip", "video_vae", "audio_vae"):
        m = models[key]
        G.add(key, m["class"], m.get("title", key.upper().replace("_", " ")), 0, **m["inputs"])
    model = "unet"
    lora = render.get("lora") or profile.get("lora")
    if lora:
        model = G.add("lora", "LoraLoaderModelOnly", "TURBO LORA (optional)", 0, model=link("unet"),
                      lora_name=lora["lora_name"], strength_model=lora.get("strength_model", 1.0))

    # References (lane 1).
    r2v_refs: dict[str, list] = {}
    for p in spec.pictures:
        G.add(f"pic_{p.id}", "LoadImage", f"{refs[p.id].tag} {p.id}", 1, image=files.get(p.id, p.file.split("/")[-1]))
        r2v_refs[refs[p.id].socket] = link(f"pic_{p.id}")
    for v in spec.videos:
        G.add(f"vid_{v.id}", "LoadVideo", f"{refs[v.id].tag} {v.id}", 1, file=files.get(v.id, v.file.split("/")[-1]))
        G.add(f"vidparts_{v.id}", "GetVideoComponents", f"{v.id} frames + soundtrack", 1, video=link(f"vid_{v.id}"))
        r2v_refs[refs[v.id].socket] = link(f"vidparts_{v.id}", 0)
        if v.use_soundtrack:
            r2v_refs[refs[f"{v.id}#audio"].socket] = link(f"vidparts_{v.id}", 1)
    for v in spec.voices:
        G.add(f"voice_{v.id}", "LoadAudio", f"{refs[v.id].tag} {v.id} (raw)", 1, audio=files.get(v.id, v.file.split("/")[-1]))
        G.add(f"trim_{v.id}", "TrimAudioDuration", f"{refs[v.id].tag} {v.id} — trim near line length", 1,
              audio=link(f"voice_{v.id}"), start_index=float(v.start),
              duration=float(v.duration if v.duration is not None else R.MAX_AUDIO_SECONDS))
        r2v_refs[refs[v.id].socket] = link(f"trim_{v.id}")

    # Conditioning (lane 2).
    G.add("r2v", "MiniMaxH3ReferenceToVideo", "MiniMax H3 Reference to Video", 2,
          clip=link("clip"), vae=link("video_vae"), audio_vae=link("audio_vae"), prompt=prompt,
          width=width, height=height, length=R.frame_count(spec),
          ref_image_size=render.get("ref_image_size", "match"), **r2v_refs)
    positive = link("r2v", 0)
    for i, g in enumerate(sorted(spec.guides, key=lambda g: g.at_seconds)):
        extra: dict[str, Any] = {}
        if g.picture:
            extra["image"] = link(f"pic_{g.picture}")
        if g.voice:
            extra["audio"] = link(f"trim_{g.voice}")
        positive = link(G.add(f"guide_{i}", "MiniMaxH3AddGuide", f"GUIDE @ {g.at_seconds}s", 2,
                              positive=positive, vae=link("video_vae"), audio_vae=link("audio_vae"),
                              latent=link("r2v", 1), frame_idx=round(g.at_seconds * R.FPS), **extra), 0)

    # Sampling (lane 3).
    G.add("noise", "RandomNoise", "SEED", 3, noise_seed=int(spec.seed))
    G.add("sampler", "KSamplerSelect", "SAMPLER", 3, sampler_name=render.get("sampler", "res_multistep"))
    G.add("sigmas", "BasicScheduler", "STEPS", 3, model=link(model), scheduler=render.get("scheduler", "simple"),
          steps=int(render.get("steps", 20)), denoise=1.0)
    G.add("guider", "BasicGuider", "GUIDER", 3, model=link(model), conditioning=positive)
    G.add("sample", "SamplerCustomAdvanced", "SAMPLE (joint video + audio latent)", 3,
          noise=link("noise"), guider=link("guider"), sampler=link("sampler"), sigmas=link("sigmas"),
          latent_image=link("r2v", 1))

    sampled = link("sample")
    vram = render.get("vram_debug")
    if vram:  # KJNodes VRAM_Debug: unload DiT + encoder and empty the cache before decode
        sampled = link(G.add("vram", "VRAM_Debug", "FREE VRAM BEFORE DECODE", 3, any_input=sampled,
                             empty_cache=bool(vram.get("empty_cache", True)),
                             gc_collect=bool(vram.get("gc_collect", True)),
                             unload_all_models=bool(vram.get("unload_all_models", True))))

    # Decode + save (lane 4).
    prefix = f"{render.get('output_prefix', 'h3voice/')}{spec.name}"
    G.add("decode_video", "VAEDecode", "DECODE VIDEO", 4, samples=sampled, vae=link("video_vae"))
    G.add("decode_audio", "VAEDecodeAudio", "DECODE STEREO AUDIO", 4, samples=sampled, vae=link("audio_vae"))
    G.add("video", "CreateVideo", "MUX", 4, images=link("decode_video"), audio=link("decode_audio"),
          fps=float(render.get("fps", R.FPS)))
    G.add("save", "SaveVideo", "SAVE VIDEO", 4, video=link("video"), filename_prefix=prefix)
    if render.get("save_audio", True):
        G.add("save_audio", "SaveAudio", "SAVE AUDIO (voice check)", 4, audio=link("decode_audio"),
              filename_prefix=f"{prefix}_audio")
    return G
