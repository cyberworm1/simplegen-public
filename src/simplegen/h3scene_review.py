"""Append-only decisions on a saved H3 Scene Chain production plan."""
import copy
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from .editing import digest, read_json
from .engine import create_run, now
from .settings import Settings, atomic_json


class OverflowReview(RuntimeError):
    """The plan is saved but requires an owner decision before compilation."""


def decide(folder: Path, action: str, concept: str | None = None) -> tuple[Path, dict]:
    folder = Path(folder).resolve()
    state = read_json(folder / "state.json")
    review = state.get("h3scene_review")
    if not review or action not in ("approve", "edit", "deny") or state.get("submission_started"):
        raise ValueError("No pending H3 Scene Chain overflow review")
    if digest(folder / "plan.json") != review["plan_sha256"]:
        raise ValueError("Review plan changed; refusing the decision")
    settings = Settings(**state["settings"])
    if digest(folder / "template.original.json") != state["template_sha256"]:
        raise ValueError("Review template changed; refusing the decision")
    output = Path(settings.output_folder).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix=".h3scene-review-", dir=output) as staging:
        revision, child = create_run(replace(settings, output_folder=staging), concept or state["concept"])
        shutil.copyfile(folder / "template.original.json", revision / "template.original.json")
        child.update(settings=settings.public_dict(),
                     template_path=state["template_path"], template_sha256=state["template_sha256"],
                     forked_from={"run_id": state["run_id"], "folder": str(folder), "at": now()},
                     h3scene_decision={"action": action, "at": now(), "plan_sha256": review["plan_sha256"]},
                     h3scene_review=copy.deepcopy(review))
        if action != "edit":
            shutil.copyfile(folder / "plan.json", revision / "plan.json")
            shutil.copytree(folder / "agent-edits", revision / "agent-edits")
            child["workflow_edit_attempts"] = copy.deepcopy(state["workflow_edit_attempts"])
        else:
            child.pop("h3scene_review")
        child["status"] = "denied" if action == "deny" else "created"
        atomic_json(revision / "state.json", child)
        published = output / revision.name
        revision.rename(published)
    return published, child
