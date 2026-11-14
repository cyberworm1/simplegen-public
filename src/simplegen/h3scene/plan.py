"""H3 Scene Chain plan schema and shape checks."""


BLOCKS = ["subject_definitions", "summary", "retention_analysis",
          "detailed_description", "overall_soundscape", "non_diegetic_music"]


SCHEMA = {
    "type": "object", "required": ["plans"],
    "properties": {"plans": {"type": "array", "minItems": 1, "items": {
        "type": "object",
        "required": ["concept_id", "title", "script", "pictures", "segments"],
        "properties": {
            "concept_id": {"type": "string"}, "title": {"type": "string"},
            "script": {"type": "string"},
            "pictures": {"type": "array", "minItems": 1, "maxItems": 9, "items": {
                "type": "object", "required": ["path", "subject"],
                "properties": {"path": {"type": "string"}, "subject": {"type": "string"}}}},
            "segments": {"type": "array", "minItems": 1, "items": {
                "type": "object", "required": ["join", "seconds", "sections"],
                "properties": {
                    "join": {"type": "string", "enum": ["first", "continue", "cut"]},
                    "seconds": {"type": "number"},
                    "sections": {"type": "object", "required": BLOCKS,
                                 "properties": {b: {"type": "string"} for b in BLOCKS}}}}}}}}},
}


def check_plan(p):
    errs = []
    for k in ("concept_id", "title", "script", "pictures", "segments"):
        if k not in p:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if not 1 <= len(p["pictures"]) <= 9:
        errs.append("pictures not 1-9")
    segs = p["segments"]
    if not segs or segs[0].get("join") != "first":
        errs.append("segment 1 must be join=first")
    for i, s in enumerate(segs, 1):
        if i > 1 and s.get("join") not in ("continue", "cut"):
            errs.append(f"seg {i}: bad join {s.get('join')}")
        if not 5 <= float(s.get("seconds", 0)) <= 15:
            errs.append(f"seg {i}: seconds {s.get('seconds')} outside 5-15")
        sec = s.get("sections", {})
        if list(sec) != BLOCKS:
            errs.append(f"seg {i}: blocks not the six in order")
        elif any(not str(sec[b]).strip() for b in BLOCKS):
            errs.append(f"seg {i}: empty block")
        elif len(sec["detailed_description"].split()) < 40:
            errs.append(f"seg {i}: detailed_description under 40 words (placeholder?)")
    return errs
