"""Offline tests for the H3 Scene Chain + RefMod built-in workflow."""
import copy
import json
import struct
import zlib
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from simplegen import agents, builtin_workflows, editing, engine, preflight
from simplegen import h3scene_family as scene
from simplegen import h3scene_refmod as refmod
from simplegen.h3scene import run
from simplegen.h3scene.__main__ import template_bytes
from simplegen.settings import Settings, atomic_json
from simplegen.workflows import validate_graph

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures/real/refmod"
CHARACTER = "character/vabi_fisufo_gozodur_20261006_104720_aaa4dc"
TRUCK = "asset/fisufo_lete_tufal_20261006_110541_fd4358"
VERBATIM = "Mara never wears sunglasses."


def png(path):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 32) * 32)) + chunk(b"IEND", b""))


def schema():
    info = json.loads((ROOT / "tests/fixtures/real/schemas/local_object_info.json").read_text())
    info.update(json.loads((FIXTURES / "object_info.json").read_text()))
    return info


@pytest.fixture
def context(tmp_path):
    assets = tmp_path / "assets"
    (assets / "characters/mara").mkdir(parents=True)
    (assets / "INDEX.md").write_text("Index")
    (assets / "characters/mara/card.md").write_text(
        f"# Mara\n## Look\nStraw hat.\n## RefMod\n`{CHARACTER}`\n## Verbatim constraints\n{VERBATIM}\n")
    png(assets / "characters/mara/sheet.png")
    (assets / "props/truck").mkdir(parents=True)
    (assets / "props/truck/card.md").write_text(f"# Truck\n## RefMod\n- {TRUCK}.safetensors\n")
    (assets / "places/farm").mkdir(parents=True)
    (assets / "places/farm/card.md").write_text("# Farm\n## Images\n- `field.png`\n")
    png(assets / "places/farm/field.png")
    source = builtin_workflows.ensure_refmod_template()
    settings = Settings(project_folder=str(assets), template_folder="", template_file=str(source),
                        output_folder=str(tmp_path / "output"), duration=30, megapixels=.52,
                        aspect_ratio="16:9", steps=30)
    return settings, json.loads(source.read_text()), schema()


def candidate(seconds=(15, 15), pictures=("places/farm/field.png",), refmods=(CHARACTER,), verbatim=True):
    pictures = [{"path": p, "subject": "Picture"} for p in pictures]
    labels = " ".join(f"<Subject {i}> from <Picture {i}>." for i in range(1, len(pictures) + 1))
    defined = " ".join(f"<Subject {len(pictures) + i}> is shown in its references." for i in range(1, len(refmods) + 1))
    sections = {name: "Fixture text." for name in run.SECTIONS}
    sections["subject_definitions"] = f"{labels} {defined} {VERBATIM if verbatim else ''}".strip()
    sections["detailed_description"] = " ".join(["A person walks through the field."] * 12) + " <d>[English] Ready.</d>"
    segments = [{"join": "first" if i == 0 else "continue", "seconds": s, "sections": copy.deepcopy(sections)}
                for i, s in enumerate(seconds)]
    for segment in segments[1:]:
        segment["sections"]["detailed_description"] = " ".join(["Continuing the shot."] * 20)
    return {"concept_id": "fixture", "title": "Fixture", "script": "Owner", "input_mode": "script",
            "pictures": pictures, "refmods": [{"name": n, "subject": "RefMod"} for n in refmods],
            "segments": segments, "warnings": [], "overflow": None}


def prepare(settings, folder, state, info):
    def persist(status=None):
        if status:
            state["status"] = status
        atomic_json(folder / "state.json", state)
    return editing.prepare_compiled(settings, folder, state, info, Event(), lambda _: None, persist,
                                    Path(settings.template_file), {}, lambda p, v: ([], v["used_references"]))


def spec(pictures=("a.png",), refmods=(CHARACTER, TRUCK), joins=("first", "continue", "cut")):
    return {"pictures": list(pictures), "refmods": list(refmods), "width": 960, "height": 544, "steps": 20,
            "context_frames": 22, "extra_trim": 3, "output_prefix": "x",
            "segments": [{"join": j, "seconds": 5, "seed": i, "prompt": "p"} for i, j in enumerate(joins)]}


def test_builtin_is_offered_separately_and_scene_chain_is_unchanged():
    base = json.loads(template_bytes())
    variant = json.loads(template_bytes(refmod=True))
    assert not scene.is_refmod(base) and scene.is_refmod(variant)
    assert scene.profile(base) == json.loads(json.dumps(scene.FIXED))
    assert "refmod" not in scene.profile(base) and scene.profile(variant)["refmod"]["slots"] == 8
    path = builtin_workflows.ensure_refmod_template()
    assert path.read_bytes() == template_bytes(refmod=True)
    assert builtin_workflows.is_builtin_template(str(path))
    assert builtin_workflows.builtin_name(path) == "H3 Scene Chain + RefMod"
    # The production skill is the approved prompt verbatim plus the RefMod section.
    skills = ROOT / "src/simplegen/skills"
    base_skill = (skills / "h3-scene-production/SKILL.md").read_text()
    assert (skills / refmod.SKILL / "SKILL.md").read_text().startswith(base_skill + "\n## RefMods")


def test_loader_and_apply_match_owner_tested_workflows():
    graph = scene.compile_spec(spec(refmods=(CHARACTER, TRUCK)))
    tested = json.loads((FIXTURES / "Gozodur_Fisufo_Tufal_Rima_RefMods_5s_20steps.api.json").read_text())
    loader = next(n["inputs"] for n in graph.values() if n["class_type"] == refmod.LOADER)
    expected = tested["5"]["inputs"]
    assert set(loader) == set(expected)
    # Unused slots differ only in their inert component choice; the token limit is off for now.
    assert loader["max_total_tokens"] == 0
    assert {k: v for k, v in loader.items() if not k.startswith("components_") and k != "max_total_tokens"} == {
        k: v for k, v in expected.items() if not k.startswith("components_") and k != "max_total_tokens"}
    assert [loader[f"components_{i}"] for i in (1, 2)] == [expected[f"components_{i}"] for i in (1, 2)]
    applies = [n["inputs"] for n in graph.values() if n["class_type"] == refmod.APPLY_NODE]
    fixed = {k: v for k, v in tested["7"]["inputs"].items() if k not in ("conditioning", "mods")}
    fixed["max_total_tokens"] = 0
    assert len(applies) == 3 and all({k: v for k, v in a.items() if k not in ("conditioning", "mods")} == fixed
                                     for a in applies)
    shot = json.loads((FIXTURES / "Gozodur_Gugetonu_Sedig_RefMod_5s_20steps.api.json").read_text())
    assert shot["5"]["inputs"]["mod_1"] == CHARACTER and shot["5"]["inputs"]["components_1"] == "Visual"


@pytest.mark.parametrize("pictures", [("a.png", "b.png"), ()])
def test_every_segment_reads_conditioning_through_apply(pictures):
    api = scene.compile_spec(spec(pictures=pictures))
    validate_graph(api, schema())
    r2vs = {k for k, n in api.items() if n["class_type"] == "MiniMaxH3ReferenceToVideo"}
    applies = {k: n for k, n in api.items() if n["class_type"] == refmod.APPLY_NODE}
    assert sorted(a["inputs"]["conditioning"][0] for a in applies.values()) == sorted(r2vs)
    consumers = [n for n in api.values() if n["class_type"] in ("BasicGuider", "MiniMaxH3VideoExtender")]
    assert all(n["inputs"]["conditioning"][0] in applies or
               api[n["inputs"]["conditioning"][0]]["class_type"] == "MiniMaxH3VideoExtender" for n in consumers)
    extender = next(n for n in api.values() if n["class_type"] == "MiniMaxH3VideoExtender")
    assert extender["inputs"]["conditioning"][0] in applies
    loads = [n for n in api.values() if n["class_type"] == "LoadImage"]
    assert len(loads) == len(pictures)
    refs = [k for n in api.values() if n["class_type"] == "MiniMaxH3ReferenceToVideo" for k in n["inputs"]
            if k.startswith("ref_images.")]
    assert len(refs) == len(pictures) * 3


def test_without_refmods_the_approved_nodes_are_identical():
    base = scene.compile_spec({k: v for k, v in spec().items() if k != "refmods"})
    variant = scene.compile_spec(spec(refmods=()))
    applies = {k for k, n in variant.items() if n["class_type"] == refmod.APPLY_NODE}
    r2v_of = {k: variant[k]["inputs"]["conditioning"] for k in applies}
    for key, node in base.items():
        inputs = {name: (r2v_of[v[0]] if isinstance(v, list) and v[0] in applies else v)
                  for name, v in variant[key]["inputs"].items()}
        assert {**variant[key], "inputs": inputs} == node


def test_matchtype_output_type_is_checked():
    api = scene.compile_spec(spec())
    apply = next(k for k, n in api.items() if n["class_type"] == refmod.APPLY_NODE)
    decode = next(n for n in api.values() if n["class_type"] == "VAEDecode")
    decode["inputs"]["samples"] = [apply, 0]
    with pytest.raises(ValueError, match="incompatible connection"):
        validate_graph(api, schema())


@pytest.mark.parametrize("change,message", [
    (lambda g, a, r: g[a]["inputs"].update(retention=0.5), "changed approved settings"),
    (lambda g, a, r: g[next(k for k, n in g.items() if n["class_type"] == refmod.LOADER)]["inputs"].update(
        components_1="All"), "changed approved setting components_1"),
    (lambda g, a, r: next(n for n in g.values() if n["class_type"] == "BasicGuider")["inputs"].update(
        conditioning=[r, 0]), "bypasses"),
    (lambda g, a, r: g.pop(a), "exactly one RefMod Apply"),
])
def test_tampered_refmod_graph_rejected(change, message):
    template = json.loads(template_bytes(refmod=True))
    api = scene.compile_spec(spec(joins=("first",)))
    scene.verify(api, template)
    apply = next(k for k, n in api.items() if n["class_type"] == refmod.APPLY_NODE)
    r2v = next(k for k, n in api.items() if n["class_type"] == "MiniMaxH3ReferenceToVideo")
    change(api, apply, r2v)
    with pytest.raises(ValueError, match=message):
        scene.verify(api, template)
    with pytest.raises(ValueError, match="unexpected node types"):
        scene.verify(scene.compile_spec(spec()), json.loads(template_bytes()))


def test_card_names_and_name_problems(tmp_path):
    assert refmod.card_names("# X\n## RefMod\n- `a/b.safetensors`\n\nc_d\n## Look\nnot/this\n") == ["a/b", "c_d"]
    assert refmod.card_names("# X\n## Look\n") == []
    assert refmod.name_problem("refmodBuilder_work/abc/member_000")
    assert refmod.name_problem("../escape") and refmod.name_problem("a\\b")
    assert refmod.name_problem(CHARACTER) is None


@pytest.mark.parametrize("plan_change,error", [
    (lambda p: p["refmods"].append({"name": "character/unknown", "subject": "x"}), "not named in any card"),
    (lambda p: p["pictures"].append({"path": "characters/mara/sheet.png", "subject": "x"}), "RefMod replaces it"),
    (lambda p: p["refmods"].append(dict(p["refmods"][0])), "listed once"),
    (lambda p: p.update(refmods=[], pictures=[]), "at least one picture or RefMod"),
    (lambda p: p.update(refmods="x"), "refmods must be a list"),
    (lambda p: [s["sections"].update(subject_definitions="<Subject 1> from <Picture 1>.") for s in p["segments"]],
     "Verbatim constraints"),
])
def test_plan_checks(context, plan_change, error):
    settings, template, _ = context
    plan = candidate()
    plan_change(plan)
    root = Path(settings.project_folder)
    with pytest.raises(ValueError, match=error):
        scene.check(plan, settings, "fixture", concept="PERSON: Ready.", roots=[root], template=template)


def test_refmods_may_replace_every_picture(context):
    settings, template, info = context
    plan = candidate(pictures=(), refmods=(CHARACTER, TRUCK))
    root = Path(settings.project_folder)
    scene.check(plan, settings, "fixture", concept="PERSON: Ready.", roots=[root], template=template)
    graph = scene.compile_graph(template, plan, settings, "fixture")
    scene.verify(graph, template)
    validate_graph(graph, info)
    assert not [n for n in graph.values() if n["class_type"] == "LoadImage"]
    with pytest.raises(ValueError, match="<Picture 1> is outside"):
        bad = candidate(pictures=(), refmods=(CHARACTER,))
        bad["segments"][0]["sections"]["summary"] = "<Picture 1>"
        scene.check(bad, settings, "fixture", concept="PERSON: Ready.", roots=[root], template=template)


def test_base_scene_chain_still_requires_pictures(context, tmp_path):
    settings, _, _ = context
    base = json.loads(template_bytes())
    with pytest.raises(ValueError, match="1-9 pictures"):
        scene.check(candidate(pictures=()), settings, "fixture", concept="PERSON: Ready.",
                    roots=[Path(settings.project_folder)], template=base)


def test_schema_and_brief(context):
    settings, template, _ = context
    brief = scene.brief(template, settings, "fixture", Path("unused"), concept="PERSON: Ready.")
    assert brief["skill"] == refmod.SKILL and brief["refmod"]
    assert brief["scene_messages"][0]["content"].endswith(
        (ROOT / "src/simplegen/skills" / refmod.SKILL / "SKILL.md").read_text()[-200:])
    assert f"`{CHARACTER}`" in brief["scene_messages"][1]["content"]
    item = agents.plan_schema(creative=brief)["properties"]["plans"]["items"]
    assert "refmods" in item["required"] and item["properties"]["pictures"]["minItems"] == 0
    assert item["properties"]["refmods"]["items"]["required"] == ["name", "subject"]
    base_brief = scene.brief(json.loads(template_bytes()), settings, "fixture", Path("unused"), concept="x")
    assert "refmods" not in agents.plan_schema(creative=base_brief)["properties"]["plans"]["items"]["properties"]


def test_preflight_reports_card_refmods(context):
    settings, _, info = context
    checks = preflight.run_preflight(settings, object_info=info, resolve=lambda n: n)
    found = next(c for c in checks["checks"] if c["id"] == "h3scene_refmods")
    assert checks["status"] == "pass" and found["status"] == "pass" and CHARACTER in found["message"]
    envelope = next(c for c in checks["checks"] if c["id"] == "h3scene_envelope")
    assert "RefMods have been tested only" in envelope["message"]
    card = Path(settings.project_folder) / "props/truck/card.md"
    card.write_text("# Truck\n## RefMod\nasset/not_installed\n")
    checks = preflight.run_preflight(settings, object_info=info, resolve=lambda n: n)
    found = next(c for c in checks["checks"] if c["id"] == "h3scene_refmods")
    assert checks["status"] == "fail" and "not installed" in found["message"]
    card.write_text("# Truck\n## RefMod\nrefmodBuilder_work/abc/member_000\n")
    found = next(c for c in preflight.run_preflight(settings, object_info=info, resolve=lambda n: n)["checks"]
                 if c["id"] == "h3scene_refmods")
    assert found["status"] == "fail" and "work file" in found["message"]
    without = {k: v for k, v in info.items() if k != refmod.LOADER}
    checks = preflight.run_preflight(settings, object_info=without, resolve=lambda n: n)
    assert next(c for c in checks["checks"] if c["id"] == "template_structure")["status"] == "fail"


def test_prepare_accepts_refmod_plan_and_resumes(context, monkeypatch):
    settings, _, info = context
    folder, state = engine.create_run(settings, "PERSON: Ready.")
    monkeypatch.setattr(agents, "generate_plan",
                        lambda s, workspace, *a: atomic_json(workspace / "plan.json", candidate()))
    prepared, plan, _, used = prepare(settings, folder, state, info)
    assert state["workflow_acceptance"]["scene_chain"] == scene.KIND
    assert f"RefMods: {CHARACTER}" in prepared["summary"]
    assert [Path(p).name for p in used] == ["field.png"]
    graph = json.loads((folder / "workflow.json").read_text())
    loader = next(n["inputs"] for n in graph.values() if n["class_type"] == refmod.LOADER)
    assert loader["mod_1"] == CHARACTER and loader["mod_2"] == "(none)"
    monkeypatch.setattr(agents, "generate_plan", lambda *a: pytest.fail("agent on accepted resume"))
    resumed, resumed_plan, _, _ = prepare(settings, folder, state, info)
    assert resumed == prepared and resumed_plan == plan


def test_prepare_rejects_uninstalled_refmod(context, monkeypatch):
    settings, _, info = context
    info = copy.deepcopy(info)
    for i in range(1, 9):
        choices = info[refmod.LOADER]["input"]["required"][f"mod_{i}"][0]
        choices.remove(CHARACTER)
    folder, state = engine.create_run(settings, "PERSON: Ready.")
    monkeypatch.setattr(agents, "generate_plan",
                        lambda s, workspace, *a: atomic_json(workspace / "plan.json", candidate()))
    with pytest.raises(ValueError, match="unavailable choice"):
        prepare(settings, folder, state, info)
    assert not state.get("workflow_acceptance")


def test_engine_error_names_missing_refmod_builtin(context):
    settings, _, _ = context
    Path(settings.template_file).unlink()
    with pytest.raises(ValueError, match=r"built-in H3 Scene Chain \+ RefMod workflow is unavailable"):
        engine._template(replace(settings))
