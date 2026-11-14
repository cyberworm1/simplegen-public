# simpleGen — agent guidance

Desktop app (PySide6) that turns a concept into a finished ComfyUI H3 video. An
official agent CLI prepares the script/prompts; deterministic code owns timing,
validation, submission, downloads and paid RunPod lifecycle. Read README.md and
DESIGN.md first; WORKFLOW_EDITING.md and RUNPOD.md cover the compiler and the
paid lifecycle.

## Rules

- The desktop app is the RunPod session owner. Do not add a background service.
- Never add a code path that creates, keeps or deletes paid compute without the
  existing durable-intent pattern (persist intent before the provider call,
  reconcile uncertain results, never repeat a create POST).
- Deterministic problems (timing, frames, IDs, paths, validation, model
  selection) are solved in code, never by the agent.
- Saved runs, schedules and batches are append-only: repairs and edits create
  linked revisions. Older state/settings files must keep loading.
- Unknown is never compatible: a check that cannot run reports "unverified".
- Keep Qt network/agent work off the UI thread.

## Development

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

Install ffmpeg (`apt-get install -y ffmpeg`) before running tests; without it
the real-media integration tests are skipped and gaps go unnoticed.

Cloud sessions: start them from claude.ai/code with this repository selected. A
session started from an uploaded local copy (`claude --cloud` when the
repository isn't attached) cannot push its branch.

Tests must never contact real services (ComfyUI, RunPod, agent CLIs) and must
set `SIMPLEGEN_HOME` to a temporary folder. `tests/fixtures/real/` holds
anonymized recordings of real runs and trimmed ComfyUI schemas (see its README);
treat them as read-only inputs and copy them to a temporary folder before
mutating. The schema files there are large: query them with Python rather than
reading them whole.
