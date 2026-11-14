"""H3 Scene Chain ComfyUI graph builder."""
import os
import uuid


MODELS = {
    "unet": ("minimax_h3_ref2va_pruned_int8_convrot.safetensors", "default", "cuda:0"),
    "clip": ("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "minimax", "cuda:1"),
    "vae": ("minimax_h3_video_vae_fp16.safetensors", "cuda:0"),
    "avae": ("minimax_h3_audio_vae_fp32.safetensors", "cuda:0"),
}


FPS = 24


MAX_PICTURES = 9


def snap_frames(frames):
    """Snap up to the 17k+5 lattice (min 5)."""
    f = max(5, int(round(frames)))
    return f + (5 - (f % 17)) % 17


def plan_segments(spec, warn=print):
    ctx = int(spec.get("context_frames", 22))
    extra = int(spec.get("extra_trim", 0))
    if not spec["segments"]:
        raise ValueError("need at least one segment")
    out = []
    for i, s in enumerate(spec["segments"], 1):
        join = s["join"]
        if join not in ("first", "continue", "cut"):
            raise ValueError(f"segment {i}: bad join {join!r}")
        if (i == 1) != (join == "first"):
            raise ValueError("segment 1 must be 'first' and only segment 1")
        if join == "continue" and not 1 <= ctx <= 39:
            raise ValueError("context_frames must be 1-39 for continuation")
        # Extender clamps the requested context to the previous decoded clip.
        trim = min(ctx, out[-1]["frames"]) if join == "continue" else 0
        want = s["seconds"] * FPS + trim
        frames = snap_frames(want)
        if not 124 <= frames <= 362:
            warn(f"warning: segment {i} clip is {frames} frames, outside trained range 124-362")
        xt = extra if join == "continue" else 0
        out.append({"index": i, "join": join, "frames": frames, "trim": trim,
                    "extra_trim": xt, "delivered": frames - trim - xt, "seed": int(s["seed"]),
                    "prompt": s["prompt"]})
    return out


class Graph:
    def __init__(self):
        self.nodes = []  # dicts
        self.links = []  # [id, src, slot, dst, dst_input_idx, type]
        self.groups = []

    def add(self, ctype, title, widgets, outs, pos, size=(330, 120), ins=None):
        n = {"id": len(self.nodes) + 1, "type": ctype, "title": title,
             "widgets": widgets, "outs": outs, "pos": list(pos), "size": list(size),
             "ins": []}
        self.nodes.append(n)
        return n

    def link(self, src, slot, dst, name, typ):
        lid = len(self.links) + 1
        inp = {"name": name, "type": typ, "link": lid}
        if any(k == name for k, _ in dst["widgets"]):
            inp["widget"] = {"name": name}
        dst["ins"].append(inp)
        self.links.append([lid, src["id"], slot, dst["id"], len(dst["ins"]) - 1, typ])

    def group(self, title, x, y, w, h, color):
        self.groups.append({"id": len(self.groups) + 1, "title": title,
                            "bounding": [x, y, w, h], "color": color,
                            "font_size": 24, "flags": {}})

    def api(self):
        p = {}
        for n in self.nodes:
            inp = {k: v for k, v in n["widgets"] if k is not None}
            for i in n["ins"]:
                src = next(l for l in self.links if l[0] == i["link"])
                inp[i["name"]] = [str(src[1]), src[2]]
            p[str(n["id"])] = {"class_type": n["type"], "inputs": inp,
                               "_meta": {"title": n["title"]}}
        return p

    def ui(self):
        nodes = []
        for k, n in enumerate(self.nodes):
            outs = []
            for oi, (oname, otype) in enumerate(n["outs"]):
                outs.append({"name": oname, "type": otype, "localized_name": oname,
                             "links": [l[0] for l in self.links
                                       if l[1] == n["id"] and l[2] == oi] or None})
            nodes.append({
                "id": n["id"], "type": n["type"], "pos": n["pos"], "size": n["size"],
                "flags": {}, "order": k, "mode": 0, "title": n["title"],
                "inputs": [dict(i, localized_name=i["name"]) for i in n["ins"]],
                "outputs": outs,
                "properties": {"Node name for S&R": n["type"]},
                "widgets_values": [v for _, v in n["widgets"]],
            })
        return {"id": str(uuid.uuid4()), "revision": 0,
                "last_node_id": len(self.nodes), "last_link_id": len(self.links),
                "nodes": nodes, "links": self.links, "groups": self.groups,
                "config": {}, "extra": {"ds": {"scale": 0.5, "offset": [100, 100]}},
                "version": 0.4}


def build(spec, picture_names=None, warn=print):
    pics = spec["pictures"]
    if not 1 <= len(pics) <= MAX_PICTURES:
        raise ValueError(f"need 1-{MAX_PICTURES} pictures, got {len(pics)}")
    names = picture_names or [os.path.basename(p) for p in pics]
    w, h = int(spec["width"]), int(spec["height"])
    if w % 32 or h % 32:
        raise ValueError("width/height must be multiples of 32")
    steps = int(spec.get("steps", 20))
    ctx = int(spec.get("context_frames", 22))
    prefix = spec.get("output_prefix", "simpleGen/run")
    plan = plan_segments(spec, warn)
    g = Graph()
    X = 0

    # ---- shared ----
    unet = g.add("UNETLoaderMultiGPU", "MODEL - Ref2VA INT8 (cuda:0)",
                 [("unet_name", MODELS["unet"][0]), ("weight_dtype", MODELS["unet"][1]),
                  ("device", MODELS["unet"][2])], [("MODEL", "MODEL")], (X, 60))
    clip = g.add("CLIPLoaderMultiGPU", "TEXT ENCODER - Qwen3-VL INT8 (cuda:1)",
                 [("clip_name", MODELS["clip"][0]), ("type", MODELS["clip"][1]),
                  ("device", MODELS["clip"][2])], [("CLIP", "CLIP")], (X, 220))
    vae = g.add("VAELoaderMultiGPU", "VIDEO VAE (cuda:0)",
                [("vae_name", MODELS["vae"][0]), ("device", MODELS["vae"][1])],
                [("VAE", "VAE")], (X, 380))
    avae = g.add("VAELoaderMultiGPU", "AUDIO VAE (cuda:0)",
                 [("vae_name", MODELS["avae"][0]), ("device", MODELS["avae"][1])],
                 [("VAE", "VAE")], (X, 540))
    sampler = g.add("KSamplerSelect", "SAMPLER", [("sampler_name", "res_multistep")],
                    [("SAMPLER", "SAMPLER")], (X, 700))
    sched = g.add("BasicScheduler", "STEPS / SCHEDULER",
                  [("scheduler", spec.get("scheduler", "beta")), ("steps", steps),
                   ("denoise", 1.0)], [("SIGMAS", "SIGMAS")], (X, 860), (330, 130))
    g.link(unet, 0, sched, "model", "MODEL")
    pic_nodes = []
    for i, nm in enumerate(names, 1):
        pn = g.add("LoadImage", f"Picture {i}", [("image", nm), (None, "image")],
                   [("IMAGE", "IMAGE"), ("MASK", "MASK")], (X, 1040 + (i - 1) * 330),
                   (330, 300))
        pic_nodes.append(pn)
    g.group("Shared - models, sampler, pictures", X - 20, 20, 380,
            1040 + len(names) * 330 - 20, "#3f789e")

    # ---- segments ----
    SX, COLW = 420, 600
    SEG_H = 1180
    fy = 20 + len(plan) * (SEG_H + 40)
    prev = None  # previous segment dict: latent node, decoded images node
    merged = None  # (images_node, images_slot, audio_node, audio_slot)
    for sg in plan:
        i = sg["index"]
        y = 20 + (i - 1) * (SEG_H + 40)
        tag = {"first": "first", "continue": "continue", "cut": "cut"}[sg["join"]]
        secs = sg["delivered"] / FPS
        title = f"Segment {i} - {tag}, {secs:.1f} s ({sg['delivered']} frames)"
        c = lambda col, row: (SX + col * COLW, y + 60 + row * 400)  # noqa: E731
        r2v = g.add("MiniMaxH3ReferenceToVideo", f"S{i} Prompt + references ({sg['frames']} frames)",
                    [("prompt", sg["prompt"]), ("width", w), ("height", h),
                     ("length", sg["frames"]),
                     ("ref_image_size", spec.get("ref_image_size", "match"))],
                    [("positive", "CONDITIONING"), ("LATENT", "LATENT")], c(0, 0), (560, 700))
        g.link(clip, 0, r2v, "clip", "CLIP")
        g.link(vae, 0, r2v, "vae", "VAE")
        g.link(avae, 0, r2v, "audio_vae", "VAE")
        for k, pn in enumerate(pic_nodes):
            g.link(pn, 0, r2v, f"ref_images.ref_image_{k}", "IMAGE")
        cond = (r2v, 0)
        trim_src = None
        if sg["join"] == "continue":
            ext = g.add("MiniMaxH3VideoExtender",
                        f"S{i} Extender - carry end of segment {i-1} (head mode)",
                        [("mode", "auto"), ("context_length", ctx),
                         ("encode_mode", spec.get("encode_mode", "video")),
                         ("anchor_mode", "head"), ("crop", spec.get("extender_crop", "disabled")),
                         ("audio_context_length", ctx), ("audio_mode", "timeline")],
                        [("conditioning", "CONDITIONING"), ("trim_frames", "INT")],
                        c(2, 0), (360, 360))
            g.link(r2v, 0, ext, "conditioning", "CONDITIONING")
            g.link(r2v, 1, ext, "latent", "LATENT")
            g.link(vae, 0, ext, "vae", "VAE")
            g.link(avae, 0, ext, "audio_vae", "VAE")
            g.link(prev["decode"], 0, ext, "prev_frames", "IMAGE")
            if spec.get("use_prev_latent", True):
                g.link(prev["sample"], 0, ext, "prev_latent", "LATENT")
            g.link(prev["adecode"], 0, ext, "prev_audio", "AUDIO")
            cond = (ext, 0)
            trim_src = ext
        noise = g.add("RandomNoise", f"S{i} Seed", [("noise_seed", sg["seed"]), (None, "fixed")],
                      [("NOISE", "NOISE")], c(1, 0), (300, 100))
        guider = g.add("BasicGuider", f"S{i} Guider", [], [("GUIDER", "GUIDER")],
                       c(1, 0.35), (300, 60))
        g.link(unet, 0, guider, "model", "MODEL")
        g.link(cond[0], cond[1], guider, "conditioning", "CONDITIONING")
        samp = g.add("SamplerCustomAdvanced", f"S{i} Sample (video + audio latent)", [],
                     [("output", "LATENT"), ("denoised_output", "LATENT")],
                     c(1, 0.7), (300, 200))
        g.link(noise, 0, samp, "noise", "NOISE")
        g.link(guider, 0, samp, "guider", "GUIDER")
        g.link(sampler, 0, samp, "sampler", "SAMPLER")
        g.link(sched, 0, samp, "sigmas", "SIGMAS")
        g.link(r2v, 1, samp, "latent_image", "LATENT")
        vram = g.add("VRAM_Debug", f"S{i} Free VRAM before decode",
                     [("empty_cache", True), ("gc_collect", True), ("unload_all_models", True)],
                     [("any_output", "*"), ("image_pass", "IMAGE"), ("model_pass", "MODEL"),
                      ("freemem_before", "INT"), ("freemem_after", "INT")],
                     c(3, 0), (300, 150))
        g.link(samp, 0, vram, "any_input", "LATENT")
        dec = g.add("VAEDecode", f"S{i} Decode video", [], [("IMAGE", "IMAGE")],
                    c(3, 0.5), (300, 60))
        g.link(vram, 0, dec, "samples", "LATENT")
        g.link(vae, 0, dec, "vae", "VAE")
        adec = g.add("VAEDecodeAudio", f"S{i} Decode audio", [], [("AUDIO", "AUDIO")],
                     c(3, 0.8), (300, 60))
        g.link(vram, 0, adec, "samples", "LATENT")
        g.link(avae, 0, adec, "vae", "VAE")
        lt = g.add("MiniMaxH3LoopTrim",
                   f"S{i} Trim overlap ({sg['trim'] + sg['extra_trim']} frames)" if sg["trim"]
                   else f"S{i} Trim (none, conforms audio length)",
                   [("trim_frames", sg["trim"] + sg["extra_trim"]), ("fps", float(FPS)), ("match_tail", True),
                    ("retain_overlap_frames", 0)],
                   [("images", "IMAGE"), ("audio", "AUDIO"), ("images_with_overlap", "IMAGE"),
                    ("overlap_frames", "INT")], c(4, 0), (320, 200))
        g.link(dec, 0, lt, "images", "IMAGE")
        g.link(adec, 0, lt, "audio", "AUDIO")
        if trim_src is not None and not sg["extra_trim"]:
            g.link(trim_src, 1, lt, "trim_frames", "INT")
        cv = g.add("CreateVideo", f"S{i} Mux clip", [("fps", float(FPS)), ("bit_depth", "auto"),
                                                       ("color_space", "sRGB"), ("codec", "none")],
                   [("VIDEO", "VIDEO")], c(4, 0.6), (300, 150))
        g.link(lt, 0, cv, "images", "IMAGE")
        g.link(lt, 1, cv, "audio", "AUDIO")
        sv = g.add("SaveVideo", f"S{i} Save seg{i:02d}",
                   [("filename_prefix", f"{prefix}/seg{i:02d}"), ("format", "auto"),
                    ("format.codec", "auto"), ("codec", "auto")],
                   [("video", "VIDEO")], c(4, 1.1), (300, 200))
        g.link(cv, 0, sv, "video", "VIDEO")
        g.group(title, SX - 20, y, 5 * COLW + 60, SEG_H - 100,
                {"first": "#3f8e4e", "continue": "#8e7a3f", "cut": "#8e3f6a"}[sg["join"]])
        prev = {"sample": samp, "decode": dec, "adecode": adec}
        # merge chain
        if merged is None:
            merged = (lt, 0, lt, 1)
        else:
            cont = sg["join"] == "continue"
            ms = int(spec.get("merge_seam_smooth", 0)) if cont else 0
            mb = int(spec.get("merge_blend_frames", 0)) if cont else 0
            mc = spec.get("merge_color_match", "disabled") if cont else "disabled"
            mg = g.add("MiniMaxH3VideoMerge", f"Merge segments 1-{i}",
                       [("seam_smooth", ms), ("blend_frames", mb), ("color_match", mc),
                        ("fps", float(FPS))],
                       [("images", "IMAGE"), ("audio", "AUDIO")],
                       (SX + ((i - 2) % 5) * COLW, fy + 60 + ((i - 2) // 5) * 240),
                       (320, 150))
            g.link(merged[0], merged[1], mg, "images_a", "IMAGE")
            g.link(merged[2], merged[3], mg, "audio_a", "AUDIO")
            g.link(lt, 0, mg, "images_b", "IMAGE")
            g.link(lt, 1, mg, "audio_b", "AUDIO")
            merged = (mg, 0, mg, 1)

    final_row = fy + 60 + ((len(plan) - 2) // 5 + 1) * 240
    cvf = g.add("CreateVideo", "Final - mux", [("fps", float(FPS)), ("bit_depth", "auto"),
                                               ("color_space", "sRGB"), ("codec", "none")],
                [("VIDEO", "VIDEO")], (SX + 2 * COLW, final_row), (300, 150))
    g.link(merged[0], merged[1], cvf, "images", "IMAGE")
    g.link(merged[2], merged[3], cvf, "audio", "AUDIO")
    svf = g.add("SaveVideo", "Final - Save final",
                [("filename_prefix", f"{prefix}/final"), ("format", "auto"),
                 ("format.codec", "auto"), ("codec", "auto")],
                [("video", "VIDEO")], (SX + 3 * COLW, final_row), (300, 200))
    g.link(cvf, 0, svf, "video", "VIDEO")
    g.group("Final - merge (hard cuts: no seam smoothing, no colour match)",
            SX - 20, fy, 5 * COLW + 60, final_row - fy + 260, "#5a5a8e")
    return g, plan
