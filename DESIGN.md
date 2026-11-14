# simpleGen — Design (version 0.4.1)

Simple, Effective, Useful. A standalone desktop application (PySide6) with no
other editor or backend. Two tabs: **Generate** and
**Watch**. This document describes the current architecture and its invariants.
Detailed contracts live in the linked documents, and change history is in
[CHANGELOG.md](CHANGELOG.md).

## Principles

- **Deterministic code for deterministic problems.** Timing and frame lattice,
  dimensions, graph construction, validation, model availability, IDs, paths,
  job states and pod lifecycle are all code. The agent writes creative content
  only.
- **Durable state before every handoff.** Each run, schedule and batch has an
  atomic `state.json`. Intent is persisted before every agent call, upload,
  submission and paid provider mutation.
- **Never repeat an uncertain side effect.** Uncertain submissions are reconciled
  by run ID; uncertain pod creations by the exact owned pod name. Nothing is
  blindly resubmitted or recreated.
- **Append-only history.** Edits, repairs, salvage, retargets and removals create
  linked revisions or history entries; nothing overwrites accepted work.
- **Unknown is never compatible.** A check that can't run reports `unverified`,
  which requires explicit confirmation or retries later.
- **The desktop is the owner.** There is no background service. Closing the app
  detaches (work and pods continue, and Resume reconnects); it never cancels.

## Components (`src/simplegen/`)

| Area | Modules | Role |
|---|---|---|
| UI | `ui.py`, `batch_ui.py`, `schedule_ui.py`, `repair_ui.py` | Generate/Watch, Batches, Scheduled, Repair dialogs. All network, agent and media work runs off the UI thread. |
| Settings | `settings.py`, `credentials.py` | Global settings (`~/.simplegen/settings.json`), per-destination profiles, and private RunPod credentials outside projects. |
| Pipeline | `engine.py`, `variations.py` | Run creation, locking, preparation, upload, submission, monitoring, downloads, verification and reports; Qty sequences. |
| Agent | `agents.py`, `skills/h3-concept/SKILL.md` | Official CLIs (Codex, Claude Code, Grok) in subscription-only, read-only automation mode. |
| Workflow | `compiler.py`, `editing.py`, `workflow_edits.py`, `workflows.py`, `dimensions.py` | Timing options, creative plan lint, deterministic compiler, validator, and conversion to the API graph. |
| Built-in workflows | `builtin_workflows.py`, `data/workflows/`, `h3scene/` | The six built-in workflows, installed to `~/.simplegen/workflows/builtin/`; a Workflow folder is optional. |
| Preflight | `preflight.py` | Structure, compiler support, target models, folders, tools and disk checks. |
| Recovery | `repair.py`, `oom_transfer.py` | Repair, salvage, retarget; local out-of-memory transfer to a RunPod batch. |
| Scheduling | `scheduling.py` | Local scheduled requests, prepared (repaired) schedules, revisions. |
| Batches | `batches.py`, `additions.py` | Managed RunPod batches, the additions inbox, reopen, warm-pod handoff. |
| Cloud | `global_session.py`, `global_registry.py`, `cost_safety.py`, `resources.py`, `simplegen_runpod/` | Desktop-owned Global Volume GPU lifecycle, owned-session registry and polling, failure shutdown and alerts, cost history. |
| References | `refcache.py` | Content-addressed local reference cache with pins. |
| Transport | `comfy.py` | ComfyUI HTTP/WebSocket adapter: uploads, submit, monitor, download. |

## Run lifecycle

1. **Preflight** (Phase 1) runs against the run's target: local ComfyUI, a
   manual endpoint, or the cached managed-cloud schema. `fail` blocks;
   `unverified` needs confirmation. It runs again at scheduled dispatch and on
   Resume of never-submitted work.
2. **Create run.** A unique folder and `state.json` are created, and the template
   is snapshotted to `template.original.json`. There's no agent work yet.
3. **Prepare** (schema 5, Phase 3):
   - Code builds a workflow brief: up to three legal timing layouts with usable
     seconds, overlap and speech budget per segment; the prompt sections in order;
     reference slots; resolution and steps.
   - The agent (read-only) returns a creative plan: script, timing option,
     per-segment sections, references and variations.
   - Code lints the plan, compiles the native workflow and validates it with the
     unchanged validator. One correction pass is allowed.
   - Attempts are reserved before the CLI starts.
   - Acceptance records hashes in `workflow_acceptance` and snapshots references
     (Phase 6).
4. **Upload and submit.** References are uploaded from the verified cache. The
   submit intent is persisted, then the prompt ID. Scheduled local submissions
   wait for an empty local queue.
5. **Monitor and download.** Every returned media file is downloaded, checked by
   size and ffprobe, and receipted. The explicit final node gives the final cut.
6. **Complete.** The report is written, reference pins are released, and managed
   sessions acknowledge verified downloads.

Older runs keep their paths: schema 1 (legacy compiler), schemas 2–4 (agent
direct-edits the native JSON), and schema 4 adds Qty sequences. See
[WORKFLOW_EDITING.md](WORKFLOW_EDITING.md) and [VARIATIONS.md](VARIATIONS.md).

## Persistence and folders

Each run folder holds:
- `state.json`, `template.original.json`, `workflow.json`, `workflow.api.json`;
- `plan.json`, `script.txt`, `validation.json`, `used_references.txt`,
  `references.json`;
- `agent-edits/`, `events.jsonl`, `report.txt`, `final/` and `shots/`.

Originals (assets, show bible, templates) stay at their linked locations and are
only read. Named widget copies (`widgets_values_named`) are informational;
`widgets_values` is authoritative.

User configuration lives in `~/.simplegen/` (`SIMPLEGEN_HOME`):
- `settings.json`;
- `workflows/builtin/` (the built-in workflows, refreshed by the app);
- `schedules/`, the scheduler owner lock and history;
- `batches/roots.json`;
- `runpod/` (credentials, profile, cached schema, `global-sessions.json`);
- `reference-cache/`;
- `generation-history.csv`;
- `backups/`.

## Repair, salvage and retarget (Phases 2–3)

All three are deterministic and never call the agent. The source run is locked
and never written to. A revision is built in a `.repair-staging-*` folder,
preflight-checked (`fail` never publishes), and published with one atomic rename.
`repaired_from` records the kind, the changes, and old and new hashes, and chains
across repeated repairs.

- **Repair:** change model selections, limited to the target's choices.
- **Salvage:** rebuild a schema 5 plan from a rejected schema 2–4 attempt.
- **Retarget:** recompile a schema 5 plan against another workflow with the same
  prompt sections and reference slots.

## Local scheduling

Scheduling is local-only and runs in the desktop:
- one owner lock per desktop, and an append-only revision history for edits;
- a persisted retry instant (30 minutes) while blocked;
- dispatch waits for an empty local ComfyUI queue and a free local slot, plus a
  preflight re-check;
- prepared (repaired) projects can be scheduled directly;
- removing a never-started schedule releases its project.

See [SCHEDULING.md](SCHEDULING.md).

## Managed RunPod (desktop-owned Global Volume GPU)

- **Deployment:** private template, explicit Global Volume mount at
  `/workspace-global`, Secure Cloud, CUDA 13, global placement, a 200 GB
  disposable container disk, and one POST per saved create intent. See
  [RUNPOD.md](RUNPOD.md).
- **Batches:** a batch is the default. Two approved GPU types with ceilings,
  priority, a schedule, and queue-ahead rendering (stage, submit, then run) so the
  GPU never waits on downloads. See [RUNPOD_BATCHES.md](RUNPOD_BATCHES.md).
- **Additions (Phase 4):** new concepts are prepared outside the batch lock and
  requested through `<batch>/inbox/`. Only the batch owner changes membership,
  after validating against the live pod schema and the spend cap.
- **Keep-warm, reopen and handoff (Phase 4):** after verified downloads, a
  completed batch's pod stays up for `keep_warm_minutes`.
  - During that window, a new request **reopens** the batch.
  - Alternatively, a due, compatible batch **adopts** the pod: approved GPU type,
    rate within its ceiling, same profile and volume, and members valid against
    the live schema.
  - The handoff takes both batch locks in sorted order, persists intent first,
    and recovers forward or back.
  - The adopted session inherits the pod's existing prompt IDs.
- **Retirement:** normal retirement needs verified downloads, healthy
  observations and empty queues for the keep-warm window.
  - **Failure shutdown** (Phase 5) retires a failed session after
    `failure_shutdown_minutes`, only when every successful output is verified
    locally and every job is settled.
  - Startup failures with no registered job retire after the same period.
  - Nothing is deleted while outputs are uncollected, jobs are unresolved, a
    foreign job was seen, or a handoff is in progress. Uncertain deletes are
    retried and never reported as retired.
- **Cost controls:** two per-GPU ceilings and a batch ceiling, an optional total
  spend cap with projections, a startup timeout, alerts (desktop plus webhook),
  and schema drift snapshots with explicit adoption.
- **Registry:** every owned session is polled while the app is open, including
  while another batch runs.

Legacy regional CPU-manager sessions (schema v2) are recovery-only: the app
reconnects to them and retires them, but never starts new ones.

## Security and privacy

- Agents run read-only in their CLIs' sandboxes, with API-key variables removed.
- The selected AI provider receives the referenced project content, even when
  rendering is local.
- Credentials (RunPod key, gateway login, alert webhook) live only in private
  global configuration, never in project or batch state or logs.
- Tests never contact real services.

## Validation

Deterministic fixtures, fake CLIs, fake HTTP/WebSocket servers and pod monitors,
simulated clocks, Qt offscreen tests, and anonymized recordings of real runs and
ComfyUI schemas (`tests/fixtures/real/`). Live rendering, paid lifecycle and macOS
qualification are outside the automated tests.
