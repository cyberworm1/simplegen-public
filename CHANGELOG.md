# Changelog

## Unreleased

- simpleGen is released under the MIT License (`LICENSE`).
- RunPod managed sessions use your own RunPod template and Global Volume: enter `template_id` and `volume_id` in the Global profile. The app ships no account IDs and refuses to rent compute until both are set; every other deployment check is unchanged, and saved sessions keep the IDs they were created with.
- A built-in workflow used without a linked Workflow folder no longer makes the app's working directory an allowed reference folder.
- The app now ships six built-in workflows, so no Workflow folder is needed: **H3 Scene Chain — Local INT8**, **H3 Scene Chain + RefMod — Local INT8**, **H3 Persistent References Chain — Cloud BF16** and **— Local INT8/NVFP4**, and **H3 Voice Single Shot — Cloud BF16** and **— Local INT8**. The last four are packaged as JSON in `src/simplegen/data/workflows/` and installed to the app's `workflows/builtin/` folder like the scene chain (`builtin_workflows.py`). Their example prompts are blank: the agent writes every prompt. A built-in workflow used without a linked Workflow folder no longer makes the app's working directory a reference root.
- Fresh installs have no default Workflow folder. Linking one is optional, for your own templates; existing settings keep their folder.
- The recorded test fixtures in `tests/fixtures/real/` are anonymized: creative content is replaced with consistent made-up words, machine paths and hosts are generic, and the ComfyUI schemas are trimmed to the nodes the tests use.
- workflowLab is a separate project and no longer lives in this repository. The H3 Scene Chain and H3 Scene Chain + RefMod workflows are unchanged (the golden graphs still match), and `h3scene/` is now simpleGen's own code: the sync tool and `VENDORED.json` are gone, and the lab-only leftovers were removed (the default `assets/library` folder and its path alias, the `wl/` output prefix, and the lab runner's clip checks). The template profile's descriptive `prompt` and `plan_contract` labels no longer name lab files; templates saved with the earlier labels still load. The golden tests now record uploaded picture names in `uploads.json` instead of reading lab images, so they run on any machine (previously 5 failed without the lab's picture files).
- RunPod REST v1 switches off on 2026-11-15. Every pod and network-volume call now uses REST v2 (`api.runpod.io/v2`): finding, checking and deleting pods, and volume lookups (Phase 7a). v2 data is converted at one boundary (`simplegen_runpod/provider.py`) to the v1 field names the app already reads, so saved sessions load unchanged. Pod lists read every page. A zero or missing cost is unknown, never free; a reconciled pod with unknown cost keeps its identity and the accepted quote. A create is still sent once and reconciled by exact name, although RunPod's v2 docs suggest retrying. Creating a Global Volume pod stays on GraphQL, because v2 can't attach a Global Volume yet (Phase 7b, waiting on RunPod; GraphQL ends January 2027). Tested with 37 new tests, a read-only live check and one short supervised live pod.
- New built-in workflow **H3 Scene Chain + RefMod — Local INT8**: the H3 Scene Chain with MiniMaxH3Mod RefMods used like reference pictures. Cards name RefMods under `## RefMod`; a RefMod replaces that card's pictures; one loader (Visual only) and a per-segment Apply use settings from tested RefMod runs; no combined token limit while the usable range is tested. Preflight fails on RefMods ComfyUI doesn't list. The graph validator now understands ComfyUI MatchType sockets. Card format: `docs/REFMOD_CARD_EXAMPLE.md`.
- A creative plan is no longer rejected for listing the agent's own read-only template copy (`agent-edits/attempt-N/workflow.readonly.json`) in `used_references`. Reported: a cloud batch failed with "Reference is missing or outside the linked folders". `editing.normalize_used_references` replaces that copy with the linked template it came from when the copy, snapshot and source still match the run's receipt (otherwise it is dropped), and drops any other path inside the job folder. Runs before validation in the creative-plan loop, edit mode and salvage; changes are recorded in the attempt and in `workflow_acceptance.used_references_normalized`. The agent's own `plan.json` is kept as returned; paths outside the linked folders are still rejected.
- An unfinished response no longer spends the one correction. The same batch's first attempt returned a progress note as the final plan (every section "pending", `/tmp/placeholder.png` references). `editing.unfinished_plan` recognises a compiled-chain plan with no prompt text and at least one placeholder word; the first such response per run is requested again (status `unfinished`), and a second is an ordinary rejection, so a run makes at most three agent calls. Empty or malformed plans are rejected as before.
- Tests: both real failures are reproduced through the end-to-end fake agent and as unit tests (14 new tests; all but the "outside the linked folders is still rejected" guard fail on 0.5.2).
- A batch whose GPU never started no longer soft-locks after the pod is gone. Reported: a B300 pod sat initializing past the 45-minute startup timeout, was terminated in RunPod, and the batch then refused Configure, Arm, Run now, Pause and Cancel ("owns a GPU") while Resume reconnected to the vanished pod and failed again. `batches.release_retired_session` archives a session that is retired with billing confirmed stopped, never held a job, and whose members never reached submission: its cost row is written, it moves to `retired_sessions` (append-only), and the next run selects a GPU from the approved list again. It runs on any batch action and when an armed batch is dispatched. Sessions with jobs, unresolved jobs, a pod that still exists, unconfirmed billing, a handoff, or a submitted or unreadable member keep the existing review path. The batch window's Cancel button uses the same rule (`batches.owns_gpu`).
  - Cost history: each later session of a batch gets its own row (`<run_id>-session2`, …) instead of overwriting the first, and the spend cap counts released sessions (`cost_safety.released_usd`).
  - Tests: 17 new tests built from the stuck batch's session record.
- A pod deleted or stopped in RunPod while it is starting now ends the startup wait within about 30 seconds instead of the full startup timeout (45 minutes). Reported: an H200 pod was deleted during startup and the batch kept waiting for ComfyUI. The readiness loop asks RunPod about the owned pod every 30 s (`STARTUP_POD_CHECK_SECONDS`). A 404 retires the session with billing confirmed stopped, so the batch can be released and run again. EXITED or TERMINATED ends the wait but keeps the pod owned and records a startup failure, so the failure shutdown deletes it. A failed lookup never ends the wait; a slow but running pod keeps the full timeout.

## 0.5.2 — 2026-10-01

- Agent settings can no longer reach a run in a state the agent CLI would refuse. Reported: Grok + "Balanced" + no model was saved and scheduled, then failed at dispatch. One rule (`agents.resolve_agent`) now drives preflight (a new "agent" check, skipped when accepted work is reused), the Settings, schedule edit and batch member dialogs, and `build_command`.
  - The Settings and schedule edit dialogs switch a tier with no model for the chosen agent (every Grok tier; a Codex tier with no match) to Custom, the CLI's signed-in default.
  - Also caught now, all of which used to fail late: an unsupported reasoning level, a Codex model missing from your Codex model list, any reasoning other than `auto` for Codex's default or an unlisted model (previously it fell through to Grok's list), free-text reasoning typos in batch member edits, and unknown agents.
  - Tests: every provider × tier × model × reasoning combination is checked against the launch path (17 new tests, which all fail on 0.5.1). The test suite now uses a realistic Codex model list and fails fast on unexpected warning dialogs instead of hanging.

## 0.5.1 — 2026-09-30

- Fixed a crash (segfault) when saving an edited scheduled job opened from the Scheduled list. The finished save thread was `deleteLater()`'d while its last Python reference was dropped, and the deferred delete then ran in the list's outer event loop. Dialog-owned workers (schedule edit, settings, repair, batches) now wait for the thread and stay referenced until their dialog is destroyed (`qtutil.retire_worker`). Your edit was always saved; the crash happened afterwards.
- Quoted concept lines are required speech only in a spoken context: the whole line is the quote (screenplay dialogue), a speech verb precedes it, or a speaker label colon precedes it. Quoted stage directions, descriptions and editorial notes are no longer forced into dialogue. Found while comparing local agents: a quoted stage direction made a silent character speak, and a script header, a hairstyle description and a trim note each cost a correction pass. Required lines match despite trailing punctuation. Applies to both the compiled chain and the single-shot voice checks.

## 0.5.0 — 2026-09-28

### Single-shot Ref2VA voice workflows (h3voice)
- New template family: one H3 Ref2VA shot (4–15 s) with character pictures and
  voice-timbre audio references. Recognized by `extra.h3voice` in the workflow;
  local (MultiGPU INT8) and cloud (core-loader BF16) variants.
- The agent returns a shot spec (new `h3-voice-shot` skill, strict schema); code
  renders the prompt, lints it, builds and structurally verifies the graph. Runs
  keep state schema 5; the family is recorded in `workflow_acceptance.single_shot`.
- Shared plan-attempt loop in `editing.py` (one correction pass, receipts) for both
  families; preflight structure checks for the new family, including
  cloud-targeted MultiGPU templates and missing node packs.
- Uploaded audio/video file names pass graph validation like images; the agent's
  project index lists audio files.
- Vendored h3voice modules with a hash manifest and `tools/sync_h3voice.py`.
- Not yet: Repair/model reselection for single-shot runs (refused with a message),
  Qty > 1.

## 0.4.1 — 2026-09-28

No code changes from 0.4.0.
- Version metadata is now 0.4.1 (`pyproject.toml`, `simplegen.__version__`).
- Documentation reconciled with current behavior: design, legacy recovery,
  batch, scheduling, variation and validation notes, and README.

## 0.4.0 — 2026-09-28

Six improvement phases.

### Phase 1 — Preflight gate and per-destination workflows
- A deterministic preflight checks workflow structure, target model availability
  (live local `/object_info` or the cached cloud schema, with its age), linked
  folders, the agent CLI and ffprobe, and free disk space. It gates Generate,
  Schedule, scheduled dispatch, adding to a batch, and batch preparation. Nothing
  starts agent work until it passes; an "unverified" result needs explicit
  confirmation.
- Local and Cloud each remember their own workflow folder and file.
- A batch whose reference folder is unavailable before any pod exists waits and
  retries instead of failing.

### Phase 2 — Repair without another agent pass
- **Repair…** creates a linked revision of a never-submitted run with new model
  selections, chosen only from the target's installed models. It is built in a
  staging folder, preflight-checked, then published atomically.
- Repaired runs can be run now or scheduled (`Scheduler.add_prepared`).
- Resume preflights never-submitted work and offers Repair when the check fails.
- Removing a never-started schedule releases its project for Resume or
  rescheduling; the history is kept.
- Manual cloud endpoints are checked against their live node schema.

### Phase 3 — Agent writes content, code builds the workflow
- New runs use state schema 5. The agent runs read-only and returns a creative
  plan: a timing option chosen from up to three legal layouts, and prompt
  sections per segment. Code compiles the native workflow deterministically and
  validates it with the unchanged validator.
- Content checks feed the single correction pass: sections, leftover sample text,
  dialogue tags, speech budget (warning above 1.2×, error above 1.5×), picture
  indices, reference files, and quoted concept lines.
- **Rebuild from saved agent work…** salvages rejected schema 2–4 attempts, and
  Repair can retarget a schema 5 plan to another compatible workflow file.
- Both native link formats are supported, and preflight checks compiler support
  before agent work.

### Phase 4 — Running-batch additions, warm-pod handoff, drafts while running
- Add concepts to a running batch through an atomic inbox. Each is prepared
  outside the batch lock and accepted by the batch owner after validation against
  the live pod schema.
- A completed batch can reopen during its keep-warm window (`keep_warm_minutes`,
  default 5).
- A due, compatible batch adopts a warm pod instead of creating a new one. The
  handoff is crash-recoverable, and the adopted pod inherits its earlier prompt
  IDs so automatic retirement still works.
- Optional per-batch spend cap, with projections from measured render times.
- The Generate draft is separate from running work. The local and cloud slots are
  independent, there is a Queue locally action, and each job has its own Cancel.

### Phase 5 — Cost safety
- Failure shutdown: after a managed run or batch fails, the GPU is deleted once
  every successful output is downloaded and verified and the queues have been
  empty for `failure_shutdown_minutes` (default 15; 0 disables it).
- Startup timeout (`startup_timeout_minutes`, default 45) replaces the 6-hour
  readiness wait.
- Alerts go to the desktop and an optional webhook (`alert_webhook_url`, stored
  only in global settings). Unresolved-job, uncollected-output and
  delete-uncertain states are reported and never deleted.
- Schema drift detection saves a live `/object_info` snapshot; **Use latest live
  schema** adopts it explicitly.
- Other owned sessions keep being polled while a batch runs.

### Phase 6 — Local reference cache
- Accepted work snapshots its references into a content-addressed store
  (`~/.simplegen/reference-cache`), with a `references.json` per run and a pin per
  run.
- Uploads, batch fingerprints and saved-run preflight use the verified cache, so
  runs no longer need the linked file server once accepted.
- Pins are released after success and unused blobs are collected. Failed and
  attention runs keep their cache.

### Tests
570 passed, 2 skipped, offline, including end-to-end tests with real ffmpeg.

## 0.3.0

Managed RunPod Global Volume sessions owned by the desktop, batches with
queue-ahead rendering, local scheduling, sequential variations, and local
out-of-memory transfer to RunPod.
