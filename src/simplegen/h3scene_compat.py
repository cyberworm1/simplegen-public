"""Read earlier scene-chain records without changing their saved source files."""

import copy
import json


def normalize_state(state):
    state = copy.deepcopy(state)
    for old, new in (("h3lab_review", "h3scene_review"), ("h3lab_decision", "h3scene_decision")):
        if old in state:
            state.setdefault(new, state.pop(old))
    accepted = state.get("workflow_acceptance") or {}
    for old, new in (("lab_chain", "scene_chain"), ("h3lab_format", "h3scene_format")):
        if old in accepted:
            value = accepted.pop(old)
            accepted.setdefault(new, "scene_chain" if value == "lab_chain" else value)
    if isinstance(state.get("summary"), str):
        state["summary"] = state["summary"].replace("H3 lab chain", "H3 Scene Chain")
    if "lint_warnings" in accepted:
        accepted["lint_warnings"] = [warning.replace("H3 lab", "H3 Scene Chain")
                                     for warning in accepted["lint_warnings"]]
    return state


def read_state(path):
    return normalize_state(json.loads(path.read_text(encoding="utf-8")))


def normalize_creative(creative):
    if creative is None:
        return None
    creative = dict(creative)
    for old, new in (("lab_chain", "scene_chain"), ("lab_messages", "scene_messages")):
        if old in creative:
            creative.setdefault(new, creative.pop(old))
    if creative.get("skill") == "h3-lab-production":
        creative["skill"] = "h3-scene-production"
    return creative


# Profile labels written before 0.5.3. Only these descriptive strings changed; every
# value that affects the graph, the prompt hash or the checks is compared unchanged.
LEGACY_PROFILE_LABELS = {
    "prompt": {"production_v3.md": "h3-scene-production",
               "production_v3.md plus the RefMod section": "h3-scene-refmod-production"},
    "plan_contract": {
        "DECISIONS 13 plus optional warnings and overflow":
            "h3scene plan schema plus optional warnings and overflow",
        "DECISIONS 13 plus optional warnings and overflow plus refmods":
            "h3scene plan schema plus optional warnings and overflow plus refmods"},
}


def normalize_profile(profile):
    """Read a saved template profile with current labels; never changes the saved file."""
    if not isinstance(profile, dict):
        return profile
    profile = copy.deepcopy(profile)
    for key, labels in LEGACY_PROFILE_LABELS.items():
        if profile.get(key) in labels:
            profile[key] = labels[profile[key]]
    return profile


def template_metadata(template):
    if not isinstance(template, dict) or not isinstance(template.get("extra"), dict):
        return None
    extra = template["extra"]
    if "h3scene" in extra:
        data = extra["h3scene"]
        return data if isinstance(data, dict) and data.get("kind") == "scene_chain" else None
    data = extra.get("h3lab")
    return data if isinstance(data, dict) and data.get("kind") == "lab_chain" else None


def legacy_template_bytes(payload):
    """Reproduce the exact app-owned example for conservative stale-file cleanup."""
    import uuid

    native = json.loads(payload)
    native["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "simplegen:h3lab:template:1"))
    data = native["extra"].pop("h3scene")
    data["kind"] = "lab_chain"
    # That example carried the earlier profile labels.
    for key, labels in LEGACY_PROFILE_LABELS.items():
        earlier = {new: old for old, new in labels.items()}
        if data["profile"].get(key) in earlier:
            data["profile"][key] = earlier[data["profile"][key]]
    native["extra"]["h3lab"] = data
    return (json.dumps(native, indent=2) + "\n").encode("utf-8")


LEGACY_FILENAME = "H3_Lab_Chain_LOCAL_INT8.json"
