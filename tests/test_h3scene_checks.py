"""Ported offline scene checker cases; no scene CLIs or providers are imported."""
import copy
import shutil
import struct
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from simplegen.h3scene import build, produce, run

OWNER = 'MARA: I am ready.\nTOMAS: So am I.\n'
SCRIPTS = Path(__file__).parent / "fixtures/real/h3scene/scripts"

def candidate(lines=("I am ready.", "So am I."), seconds=(15, 15)):
    sections = {block: "Fixture text." for block in produce.plan.BLOCKS}
    sections["detailed_description"] = ("Mara stands beside the crate and looks down into it. "
        "She leans forward carefully holding the edge of the wooden crate while the fruit stays "
        "still. She gestures toward the harvest row, then moves closer and waits to start counting. "
        + " ".join(f"<d>[English] {line}</d>" for line in lines))
    segments = [{"join": "first" if i == 0 else "continue", "seconds": value,
                 "sections": copy.deepcopy(sections)} for i, value in enumerate(seconds)]
    for segment in segments[1:]:
        segment["sections"]["detailed_description"] = " ".join(["Continuing the shot."] * 20)
    return {"concept_id": "agent-id", "title": "Ready", "script": "agent rewrite",
            "input_mode": "script", "requested_seconds": 999,
            "pictures": [{"path": "characters/mara/sheet.png", "subject": "Mara"}],
            "segments": segments}


def png(path):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 32) * 32))
                     + chunk(b"IEND", b""))


def clip(path, frames, color="black", audio="tone"):
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s=32x32:r=24"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000" if audio == "tone"
                 else "anullsrc=r=32000:cl=stereo"]
    args += ["-frames:v", str(frames), "-t", str(frames / 24), "-c:v", "libx264",
             "-threads", "1", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac"]
    subprocess.run(args + [str(path)], check=True, capture_output=True)


def plan(concept="test"):
    sections = {name: f"Text for {name}." for name in run.SECTIONS}
    sections["subject_definitions"] = "<Subject 1> from <Picture 1>."
    sections["summary"] = "[reference generation] Test action."
    sections["detailed_description"] = "[Shot 1] Says <d>[English] Test.</d>"
    return {"concept_id": concept, "title": "Test concept", "script": "Test script",
            "pictures": [{"path": "ref.png", "subject": "Subject 1"}],
            "segments": [{"join": join, "seconds": seconds, "sections": copy.deepcopy(sections)}
                         for join, seconds in (("first", 0.2), ("continue", 0.7), ("cut", 0.2))]}


class ProduceChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lib = self.root / "library"
        self.write("library/characters/mara/sheet.png", "fake image")

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_owner_extraction_and_normalization(self):
        self.assertEqual(produce.owner_lines('<d>[English] “I am ready.”</d> ignored NAME: text\n'
                                            '<d> So  am I. </d>'), ["I am ready.", "So am I."])
        self.assertEqual(produce.owner_lines('TITLE: A script\nMara: “I am ready.”\n'
                                            'Tomas: "So am I."\n'), ["I am ready.", "So am I."])
        for text in ('"youtube shorts" style video', "'I'm ready.'\n‘So am I.’",
                     "Mara says 'I'm ready.'", 'A note says “Byeee.”'):
            with self.subTest(text=text):
                self.assertEqual(produce.owner_lines(text), [])
        self.assertEqual(produce.owner_lines("A person crosses a field."), [])
        with self.assertRaisesRegex(ValueError, "malformed"):
            produce.owner_lines("<d>[English] Missing close")
        for text in ("MARA: ", "<d>[English] </d>"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "empty"):
                produce.owner_lines(text)


    def test_owner_script_fixtures(self):
        script = (SCRIPTS / "viv_rorevuvi_dun.txt").read_text(encoding="utf-8")
        self.assertEqual(produce.owner_lines(script), [
            ("Can we dunebe the kedis bemuza. It is tabutu vovizevun. It is tabutu I need butin. "
            "Mugave tegozefu can rugit need, and need is not a lalafob add."),
            ("I only fuki nipevi who are not gevigan. If you foro the job you have kavures lost it. "
            "The gazib ditozigus is kofiramo, sevolemik, and too segotu to say so. "
            "That is rokavo I can use."),
            "Post your sela. Never post your nagi.",
            "Comment BEMUZA if you have also zumigozi out devese who peganogo they needed a job.",
            "Nodef."])
        for filename, expected in (
                ("gozodur_intro_v2.txt", ["I'm Gozodur. I natu fizasosav for a bofuvo.",
                    "It's nonut out here, and I'm the only one that pizilezo gevef.",
                    "Sil what I muba?", "It's hard to miru midi gabi mekatet."]),
                ("Mobit_Intro.txt", [
                    ("I'm Mobit. Venazo here, if netepenot koliv bisivok, I letolos vomo my name. "
                    "Dimek, menoz, nolazibo, nebeta—tanuraze. I feb what I can, tizo do with what I can't."),
                    ("I dise degu into guna, puziv back kulel, keep the place ruposi. "
                    "If it can be nudane, papas, posufa, or kuzaturor—it's donuzapu zira.")])):
            with self.subTest(filename=filename):
                owner = (SCRIPTS / filename).read_text(encoding="utf-8")
                self.assertEqual(produce.owner_lines(owner), expected)
                self.assertEqual(produce.dialogue_checks(owner, candidate(expected)), ([], []))
                self.assertTrue(produce.verbatim_errors(owner, candidate(expected[1:])))


    def test_screenplay_paragraphs_cues_and_parentheticals(self):
        owner = ('"youtube shorts" style video\nINT. ROOM - DAY\nShe waves.\n'
                 'JORDAN (brightly)\n\n(quietly)\nI am\nready.\n\n'
                 '(a beat)\nSo am I.\nA second sentence.\n\n'
                 'ALEX\nYour turn.\n')
        self.assertEqual(produce.owner_lines(owner),
                         ["I am ready.", "So am I. A second sentence.", "Your turn."])


    def test_scene_headings_and_transitions_end_blocks(self):
        for boundary in ("INT. ROOM - DAY", "EXT. ROAD - NIGHT", "INT./EXT. CAR - DAY",
                         "CUT TO:", "SMASH CUT TO:", "DISSOLVE TO:", "FADE OUT.",
                         "FADE TO BLACK."):
            with self.subTest(boundary=boundary):
                owner = ('JORDAN\nI am ready.\n' + boundary
                         + '\nShe waves and says "ignore this".\nALEX\nSo am I.')
                self.assertEqual(produce.owner_lines(owner), ["I am ready.", "So am I."])


    def test_screenplay_misses_warn_while_explicit_lines_error(self):
        for owner in (OWNER, '<d>[English] I am ready.</d><d>So am I.</d>'):
            with self.subTest(owner=owner):
                value = candidate(["Changed line."])
                errors, _ = produce.check_candidate(value, owner, 30, self.lib)
                self.assertEqual(sum(e.startswith("verbatim:") for e in errors), 2)
                self.assertEqual(len(value["warnings"]), 1)
                self.assertTrue(value["warnings"][0].startswith("connective_lines:"))
        owner = 'JORDAN\nI am ready.\n\nSo am I.'
        for lines, misses in ((["Changed line."], 2), (["So am I.", "I am ready."], 1)):
            with self.subTest(lines=lines):
                value = candidate(lines)
                self.assertEqual(produce.check_candidate(value, owner, 30, self.lib)[0], [])
                self.assertEqual(sum(w.startswith("owner_dialogue_unverified:")
                                     for w in value["warnings"]), misses)
                self.assertEqual(produce.verbatim_errors(owner, value), [])
        mixed = 'JORDAN\nUncertain paragraph.\n\nALEX: Strict line.'
        errors, warnings = produce.dialogue_checks(mixed, candidate([]))
        self.assertEqual(len(errors), 1)
        self.assertIn("Strict line.", errors[0])
        self.assertEqual(len(warnings), 1)
        self.assertIn("Uncertain paragraph.", warnings[0])


    def test_connective_warning_for_every_format_and_unknown_input(self):
        for owner in (OWNER, '<d>[English] I am ready.</d><d>So am I.</d>',
                      'JORDAN\nI am ready.\n\nSo am I.',
                      'Direction with "I am ready." and "So am I."'):
            with self.subTest(owner=owner):
                value = candidate(["I am ready.", "Added line.", "So am I."])
                value["warnings"] = ["Provider warning."]
                self.assertEqual(produce.check_candidate(value, owner, 30, self.lib)[0], [])
                additions = [w for w in value["warnings"] if w.startswith("connective_lines:")]
                self.assertEqual(len(additions), 1)
                self.assertIn("Added line.", additions[0])
                self.assertIn("Provider warning.", value["warnings"])
                before = value["warnings"][:]
                produce.check_candidate(value, owner, 30, self.lib)
                self.assertEqual(value["warnings"], before)
        value = candidate([])
        self.assertEqual(produce.check_candidate(value, '"youtube shorts" style video', 30, self.lib)[0], [])
        self.assertTrue(value["warnings"][0].startswith("owner_dialogue_unverified:"))


    def test_reverse_check_accepts_verbatim_fragments_of_owner_paragraph(self):
        owner = 'JORDAN\nI am ready. So am I.'
        value = candidate()
        self.assertEqual(produce.check_candidate(value, owner, 30, self.lib)[0], [])
        self.assertTrue(value["warnings"][0].startswith("owner_dialogue_unverified:"))
        self.assertFalse(any(w.startswith("connective_lines:") for w in value["warnings"]))


    def test_verbatim_in_order_with_connective_lines(self):
        self.assertEqual(produce.verbatim_errors(OWNER, candidate(
            ["Hello.", "I am ready.", "One moment.", "So am I."])), [])
        for lines in (["So am I.", "I am ready."], ["I am ready."], ["I am ready!", "So am I."]):
            self.assertTrue(produce.verbatim_errors(OWNER, candidate(lines)))
        self.assertTrue(produce.verbatim_errors("A: Again.\nA: Again.", candidate(["Again."])))


    def test_verbatim_constraints_block(self):
        block = "A Robot has no face. A robot never has any spoken lines."
        self.write("library/characters/robot/card.md",
                   f"# Robot\n## Look\nx\n## Verbatim constraints\n{block}\n## Voice\nsilent\n")
        self.write("library/characters/robot/sheet.png", "fake image")

        def plan(definition):
            value = candidate()
            value["pictures"].append({"path": "characters/robot/sheet.png", "subject": "Robot"})
            for segment in value["segments"]:
                segment["sections"]["subject_definitions"] = definition
            return value
        base = "<Subject 1> is Mara in <Picture 1>. <Subject 2> is the Robot in <Picture 2>."
        good = plan(base + "  A Robot has no face.\n A robot never has any spoken lines.")
        self.assertEqual(produce.check_candidate(good, OWNER, 30, self.lib)[0], [])
        bad = plan(base + " A Robot has a smooth blank surface.")
        errors = produce.check_candidate(bad, OWNER, 30, self.lib)[0]
        self.assertEqual(sum("Verbatim constraints" in e for e in errors), 2)
        late = plan(base + " " + block)
        late["segments"][1]["sections"]["subject_definitions"] = base
        errors = produce.check_candidate(late, OWNER, 30, self.lib)[0]
        self.assertEqual([e[:6] for e in errors if "Verbatim" in e], ["seg 2:"])
        absent = plan("<Subject 1> is Mara in <Picture 1>.")  # Robot not defined: no demand
        self.assertEqual(produce.check_candidate(absent, OWNER, 30, self.lib)[0], [])




    def test_malformed_fields_preserve_independent_errors(self):
        value = candidate(["Changed."])
        value["pictures"][0]["path"] = "missing.png"
        value["segments"][0]["seconds"] = "bad"
        value["segments"][1]["sections"] = None
        errors, _ = produce.check_candidate(value, OWNER, 30, self.lib)
        for error in ("picture not found", "seconds must be a finite number", "blocks not the six",
                      "owner line 1", "owner line 2"):
            self.assertIn(error, "; ".join(errors))
        for mutate, expected in (
                (lambda p: p.update(warnings="Warning text"), "warnings must be a list of strings"),
                (lambda p: p.update(overflow=None), "overflow must be an object"),
                (lambda p: p.update(overflow={"needs_review": "yes"}), "needs_review must be a boolean"),
                (lambda p: p.update(input_mode="unknown"), "input_mode must be"),
                (lambda p: p.update(pictures=[None]), "pictures must be a list"),
                (lambda p: p.update(segments=[None]), "must be an object")):
            value = candidate()
            mutate(value)
            self.assertIn(expected, "; ".join(produce.check_candidate(value, OWNER, 30, self.lib)[0]))
        for seconds in (True, float("nan"), float("inf"), "bad"):
            value = candidate()
            value["segments"][0]["seconds"] = seconds
            self.assertIn("seconds must be a finite number", "; ".join(produce.check_candidate(value, OWNER, 30, self.lib)[0]))

    def test_warning_and_review_are_non_error_verdicts(self):
        value = candidate()
        value["warnings"] = ["Owner cast a character without a sheet."]
        errors, duration = produce.check_candidate(value, OWNER, 30, self.lib)
        self.assertEqual(errors, [])
        self.assertFalse(duration["needs_review"])
        self.assertEqual(value["warnings"], ["Owner cast a character without a sheet."])
        value["overflow"] = {"needs_review": True, "reason": "Owner must approve pacing."}
        errors, duration = produce.check_candidate(value, OWNER, 30, self.lib)
        self.assertEqual(errors, [])
        self.assertTrue(duration["needs_review"])
        self.assertEqual(value["overflow"]["reason"], "Owner must approve pacing.")

    def test_library_parameter_and_escape_verdicts(self):
        self.write("other/characters/zed/sheet.png", "fake image")
        other = self.root / "other"
        self.assertTrue(produce.check_candidate(candidate(), OWNER, 30, other)[0])
        good = candidate()
        good["pictures"] = [{"path": "characters/zed/sheet.png", "subject": "Zed"}]
        self.assertEqual(produce.check_candidate(good, OWNER, 30, other)[0], [])
        escape = candidate()
        escape["pictures"] = [{"path": "../library/characters/mara/sheet.png"}]
        self.assertTrue(any("picture not found" in e for e in produce.check_candidate(escape, OWNER, 30, other)[0]))


class RunChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix=".test-", dir=Path(__file__).resolve().parent)
        cls.fixture = Path(cls.temporary.name)
        cls.video = cls.fixture / "video.mp4"
        clip(cls.video, 5)


    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()


    def setUp(self):
        self.case = tempfile.TemporaryDirectory(prefix="case-", dir=self.fixture)
        self.root = Path(self.case.name)
        self.lib = self.root / "library"
        self.lib.mkdir(parents=True)
        png(self.lib / "ref.png")
        self.plan = plan()


    def tearDown(self):
        self.case.cleanup()


    def test_valid_plan_and_unchanged_prompt_content(self):
        self.assertEqual(run.validate_plan(self.plan, self.lib), [])
        self.plan["segments"][0]["sections"]["summary"] = "  Keep whitespace.\n"
        text = run.render_prompt(self.plan["segments"][0]["sections"])
        self.assertIn("summary:\n  Keep whitespace.\n", text)
        self.assertEqual([text.index(f"{name}:\n") for name in run.SECTIONS],
                         sorted(text.index(f"{name}:\n") for name in run.SECTIONS))


    def test_library_relative_and_absolute_paths(self):
        for path in ("ref.png", str(self.lib / "ref.png")):
            self.plan["pictures"][0]["path"] = path
            self.assertEqual(run.validate_plan(self.plan, self.lib), [])


    def test_all_validation_errors_at_once(self):
        bad = copy.deepcopy(self.plan)
        bad["pictures"][0] = {"path": "missing.png", "subject": ""}
        bad["segments"][0]["join"] = "cut"
        bad["segments"][1]["join"] = "bogus"
        bad["segments"][2]["join"] = "first"
        bad["segments"][0]["seconds"] = float("nan")
        sections = bad["segments"][0]["sections"]
        sections["summary"] = "<Picture 2> <Picture 0>"
        sections["detailed_description"] = "</d><d>"
        del sections["overall_soundscape"]
        sections["non_diegetic_music"] = " "
        errors = "\n".join(run.validate_plan(bad, self.lib))
        for text in ("file does not exist", "subject must", "only segment 1", "join must",
                     "positive finite", "<Picture 2>", "<Picture 0>", "balanced",
                     "overall_soundscape", "non_diegetic_music"):
            self.assertIn(text, errors)


    def test_image_contents_and_symlink_escape(self):
        shutil.copyfile(self.video, self.lib / "video.png")
        outside = self.root / "outside.png"
        png(outside)
        (self.lib / "escape.png").symlink_to(outside)
        for path, message in (("video.png", "not an image"), ("escape.png", "inside the asset folder")):
            self.plan["pictures"][0]["path"] = path
            self.assertIn(message, "\n".join(run.validate_plan(self.plan, self.lib)))


    def test_schema_count_and_dialogue_errors(self):
        self.assertTrue(run.validate_plan([], self.lib))
        for pictures in ([], self.plan["pictures"] * 10):
            bad = copy.deepcopy(self.plan)
            bad["pictures"] = pictures
            with mock.patch.object(run, "image_file", return_value=True):
                self.assertIn("need 1-9 pictures", run.validate_plan(bad, self.lib))
        for text in ("<d>missing close", "<d><d>nested</d></d>"):
            self.plan["segments"][0]["sections"]["detailed_description"] = text
            self.assertIn("balanced", "\n".join(run.validate_plan(self.plan, self.lib)))
        self.plan["segments"] = [None]
        self.assertIn("segment 1 must be an object", run.validate_plan(self.plan, self.lib))


    def test_smoke_timing_uses_builder(self):
        for segment, seconds in zip(self.plan["segments"], (8, 8, 5.5)):
            segment["seconds"] = seconds
        settings = {"width": 800, "height": 448, "steps": 20, "context_frames": 22, "seed_base": 2000}
        spec = run.make_spec(self.plan, settings, "test-hash", self.lib)
        _graph, timing = build.build(spec)
        self.assertEqual([s["frames"] for s in timing], [192, 226, 141])
        self.assertEqual([s["delivered"] for s in timing], [192, 204, 141])
        self.assertEqual([s["seed"] for s in timing], [2000, 2001, 2002])


    def test_picture_token_without_space_is_checked(self):
        self.plan["segments"][0]["sections"]["summary"] = "<Picture2> appears."
        self.assertIn("<Picture 2>", "\n".join(run.validate_plan(self.plan, self.lib)))


