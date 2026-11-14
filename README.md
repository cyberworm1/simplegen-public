# simpleGen

**Simple, Effective, Useful.** Link production materials, enter a concept, Generate,
then watch the locally saved final cut. A standalone desktop app; no other editor
or backend is required.

Version 0.5.2. See [CHANGELOG.md](CHANGELOG.md) and [DESIGN.md](DESIGN.md) for the
architecture.

## Features at a glance

- **Concept to finished cut.** Write a concept, press Generate, then watch the final
  video. Every run gets its own folder with the script, prompts, workflow copies,
  final cut, every raw shot, receipts and a report.
- **Your agent, your subscription.** Codex, Claude Code or Grok Build prepare the
  script and prompts through their official CLIs. There are no API keys and no paid
  fallback. The agent reads your production folder (show bible, character references)
  and never edits your files.
- **The agent writes content, code builds the workflow** (new runs, schema 5). The
  agent sees the full timing layout, speech budget per segment, reference slots and
  prompt sections, and chooses from legal timing layouts. Code builds, lints and
  validates the ComfyUI graph deterministically, with one correction pass at most.
- **Preflight before anything costs money.** Workflow structure, model availability
  on the target machine, linked folders, tools and disk space are checked before any
  agent or GPU work. Unknown results are reported as "unverified", never "OK".
- **Repair instead of redo.** Change a blocked run's model selections, rebuild a run
  from saved agent work, or switch a saved run to another compatible workflow. Each
  repair creates a linked revision with no new agent pass.
- **Local, scheduled or cloud.** Render on local ComfyUI; schedule local runs; or use
  managed RunPod batches on a desktop-owned GPU (Global Volume, CUDA 13) with two
  approved GPU types, price ceilings and priority.
- **Batches that keep working.** Every job is queued up front so the GPU never waits
  for downloads. Concepts can be added to a running batch. A finished batch's
  keep-warm pod can be reopened, or handed to the next compatible batch without a
  new startup.
- **Cost safety.** The pod shuts down automatically after verified downloads and a
  keep-warm window, and also after failures once everything collectable is saved.
  There is a startup timeout, an optional spend cap with projections, and desktop and
  webhook alerts (for example ntfy). Nothing undownloaded is ever deleted without an
  explicit user action.
- **Crash-safe by design.** State is written atomically, intent is recorded before
  every paid or remote action, submissions are reconciled rather than repeated, and
  closing the app detaches instead of cancelling.
- **Works while the file server sleeps.** References are cached locally by content
  hash when work is accepted, and cleaned up after success.
- **Drafts while running.** Author, schedule or queue local work while a cloud batch
  runs, and the reverse.
- **Variations.** Qty renders distinct takes from one creative pass.

Planned, not implemented: direct uploads to YouTube, Instagram and X.

## Launch

On Linux, run `./launch.sh`. On macOS, open `launch.command` (or run it in Terminal).
First launch creates an isolated environment and installs the app dependencies.
Requires Python 3.11+, FFmpeg's `ffprobe`, and at least one signed-in official agent
CLI: Codex, Claude Code, or Grok Build. Dependencies pin Qt 6.8 for macOS Intel/ARM
support (macOS 12+); Linux needs a desktop environment with Qt's native libraries.

Sign in once through the chosen CLI using your existing subscription. simpleGen
uses the CLI's supported automation interface, not extracted subscription tokens.
It removes provider API-key environment variables and rejects Codex/Claude API-key
sessions. Subscription allowances still apply; it never upgrades models or falls
back to paid APIs. Grok uses its OAuth entry point and disables API-key auth for
the child process; sign in with `grok login --oauth`.
Turn off optional account-level extra usage if you want a strict subscription cap.

## Use

1. Link your existing **Project folder** (assets, character references, notes,
   show bible) and an **Output folder**. Linking a **Workflow folder** is
   optional, for your own templates; the built-in workflows need none, and fresh
   installs have no Workflow folder set.
2. Select a workflow. Six built-in workflows are always listed:
   - **H3 Scene Chain — Local INT8 (built-in)**
   - **H3 Scene Chain + RefMod — Local INT8 (built-in)**
   - **H3 Persistent References Chain — Cloud BF16 (built-in)**
   - **H3 Persistent References Chain — Local INT8/NVFP4 (built-in)**
   - **H3 Voice Single Shot — Cloud BF16 (built-in)**
   - **H3 Voice Single Shot — Local INT8 (built-in)**

   The app installs them under its home folder (`workflows/builtin/`). Their
   example prompts are blank: the agent writes every prompt. When a Workflow
   folder is linked, its top-level JSON files are listed alongside them.
   The list refreshes every three seconds while idle, retaining your selection
   when the file still exists. Polling pauses during generation and while the
   dropdown is open.
3. Choose **Codex**, **Claude Code**, or **Grok Build** using the radio buttons on
   Generate. Choose **Local** or **RunPod / Cloud** directly above your concept.
   Local defaults to `http://127.0.0.1:8188`; edit the visible **ComfyUI URL** as
   needed. Local and cloud connections are saved separately, and so is each
   destination's workflow folder and file: switching Local ↔ Cloud switches the
   workflow. A destination without a saved workflow starts unselected (Cloud
   falls back to its default workflow when set). New cloud setups
   use managed RunPod; existing manual cloud connections are retained. In Settings
   choose model tier/model and reasoning. Set **Steps**, **Megapixel**, total **Duration**,
   **Aspect ratio**, and **Qty** beside Duration. Duration applies to each video;
   Qty defaults to 1 and controls the number of variations.
   Steps defaults to 25 (range 1–100); every segment inherits the selected value.
   Saved and scheduled requests retain their own settings. Older saved runs keep
   their accepted workflow steps when resumed.
4. Enter a concept and press Generate. A deterministic **preflight** runs first
   (off the UI thread, no agent use): workflow structure, compatibility with the
   target's node schema (live `/object_info` for Local and manual cloud
   endpoints, or the cached schema with its age for managed RunPod), linked folders, agent CLI/`ffprobe`, and free space in the
   output folder (Settings key `preflight_min_free_gb`, default 5). A failure
   blocks with the specific reason; model mismatches name the loader, the missing
   model and the models that target offers (see Details). A check that cannot
   run — ComfyUI unreachable, schema cache missing, a linked folder missing or
   not responding (e.g. an unmounted SFTP share) — is **unverified** and needs
   explicit confirmation. Then the script and prompts are prepared
   automatically. The concept box remains editable for your next draft while
   this run uses its saved request. Watch opens on completion; the full saved
   path is selectable.
5. **Repair…** fixes a prepared run whose accepted workflow selects models the
   destination does not have (e.g. the cloud BF16 template on Local). It is
   offered on a failed run, in Resume saved run, on Scheduled rows that need
   attention, and when Resume is blocked by preflight. Pick each replacement
   from the destination's own models (nothing is chosen for you); Save revision
   writes a new linked project (`repaired_from`), keeps the accepted script,
   prompts and workflow edits, never changes the original, and publishes only
   after preflight passes (unverified needs confirmation). Then **Run now** or
   **Schedule…**; neither calls the agent again. Available for single runs
   (Qty 1) that were never submitted; batch members and Qty > 1 sequences are
   not supported yet. Changing to a different workflow file still needs Edit as
   new run.

Resume saved run preflights work that was never submitted before continuing:
`fail` blocks with the reason and offers Repair…, `unverified` needs
confirmation. Resuming a run that already has a ComfyUI job reconnects without
a check.

For **Local** generation, check **Schedule** beside Cancel, choose a local date
and 24-hour time (`MM/DD/YY HH:mm`), then click Generate. The request and workflow
snapshot are saved immediately, with no agent work, uploads or ComfyUI submission
before that time. Scheduling requires a passing preflight (or confirming an
unverified one); preflight runs again at dispatch. Schedule clears after saving so you can generate something else
immediately. **Scheduled (n)** lists requests with Run now, Edit, Reschedule and
Remove. Edit changes a never-started local request's concept, workflow, agent,
generation settings and time, while preserving earlier revisions.
When due requests are blocked, the button also shows how many are **waiting**.
A status line below the controls explains the next request's blocker, and the
schedule list shows each request's waiting reason and next retry time.

Keep simpleGen open and the computer awake. Due requests wait for the local run
slot and for the local ComfyUI queue to clear; a running cloud batch does not
hold them up. **Queue locally** schedules the current draft on Local, due now.

While any job runs, the Generate page stays a draft: destination, workflow,
agent, schedule and generation settings can change without affecting the running
job, which is shown in its own status area. One local run and one cloud batch can
run at the same time (a second local run is refused). A new concept can join a
running or still-warm RunPod batch with **Add to running batch…**, and a due
batch can take over another batch's warm GPU instead of starting a new one. See
[RunPod batches](RUNPOD_BATCHES.md). Missed times after closing
or sleeping remain eligible. Blocked requests retry every 30 minutes while the
app is open. Interrupted or failed runs need explicit review and Resume.
The selected instant stays fixed if the computer changes timezone. Ambiguous or
missing daylight-saving times are rejected. Scheduling applies to the entire Qty
set. This checkbox is local-only; managed RunPod has its own batch scheduler.
See [local scheduling behavior](SCHEDULING.md).

For multiple variations, describe the desired differences in the concept and set
Qty. The agent prepares the shared workflow and all prompts in one creative pass.
Each variation renders individually, with unchanged workflow batch sizes. Local
and one-off runs start the next variation after downloading the previous one.
RunPod batches queue their prepared workflows ahead so ComfyUI starts the next
render while simpleGen downloads completed videos and shots.
Numbered `variation-001`, `variation-002`, etc. folders retain scripts, workflows
and outputs. Watch lets you select completed final cuts, including from a paused
set. Resume skips verified completed work and reconnects existing ComfyUI job IDs.

Managed RunPod defaults to **Batch & schedule**. Generate adds your concept and Qty
to a saved batch. Open **Batches**, prepare the workflows, approve two suitable GPU
types and hourly ceilings, then choose **Schedule batch** or **Start now**. High
priority uses whichever approved GPU is available (cheaper if both); Low waits
for the cheaper GPU. Waiting batches poll every five minutes by default without
repeating agent preparation. Choose **One-off** for an immediate standalone run.
The batch window shows a live GPU catalog before preparation. Its batch hourly
ceiling can be raised above the profile default when approving GPU choices.
A local out-of-memory failure offers **Add to RunPod batch** with editable settings
and the configured cloud model/CLIP defaults; the original failed run is retained.
See [RunPod batch instructions](RUNPOD_BATCHES.md).

Managed RunPod uses one session for the whole set. GPU options show aggregate
runtime/cost estimates with one five-minute idle allowance. Normal shutdown waits
for every requested variation and all verified downloads, then five minutes of
empty queues. See [sequence recovery and compatibility](VARIATIONS.md).
After a failed run or batch, the GPU also shuts down once every finished output
is downloaded and the queues have been empty for **Failure shutdown** minutes
(Settings, default 15, 0 = off). GPU startup gives up after **GPU startup
timeout** (default 45 minutes). Alerts appear as desktop notifications and,
optionally, a plain-text **Alert webhook** (kept only in global settings). A changed live RunPod node schema
is reported; **Use latest live schema…** in Settings adopts it explicitly. See
[RunPod](RUNPOD.md).

Model and reasoning are independent. Codex choices come from its installed model
cache; Economy prefers the available small-model family. Claude offers the
haiku/sonnet/opus aliases. Grok accepts an explicit model ID (see `grok models`),
or **Custom** with an empty model to use its signed-in default. A tier is a
shortcut, not an assertion of token price. Unsupported reasoning/model choices
fail visibly; no automatic escalation occurs. An invalid workflow edit gets one
correction pass with a specific validation error, then fails visibly if still invalid.
Switching providers selects a compatible model at the chosen tier (Grok uses its
signed-in default until configured). Returning to a provider restores its choices
within the current app session. The selected provider and settings persist when
you generate or close the app. Provider buttons are disabled during a run; Resume
uses the original run's provider and connection settings.

Agent preparation and GPU resource assessments have no application-imposed time
limit, including resumed projects with an older saved timeout. Cancellation and
window-close handling remain available; elapsed time alone never stops the agent.
simpleGen does not impose an agent turn cap. Its workflow instructions identify the direct
editing tools available, require the supplied timing values, limit discovery to
essential references, and require a final structured response after the edits.

The agent uses a compact packaged [skill](src/simplegen/skills/h3-concept/SKILL.md)
and runs read-only. It reads a workflow brief (legal timing options with speech
budgets, the template's prompt sections and reference slots), chooses a timing
option and writes only content: the script, each segment's prompt sections and
reference paths. Code checks that content, builds the native workflow
deterministically (cloning segment cards, wiring links, timing, resolution,
steps and seeds), validates it, converts to execution format and handles
transfer/progress (state schema 5; see [workflow editing](WORKFLOW_EDITING.md)).
Runs saved before this release keep their previous behavior. A run whose agent
edits were rejected can be rebuilt from its saved agent work without another
agent pass, and a compiled run can switch to another compatible workflow file
from Repair. The agent can
inspect linked documents and images as needed; those providers receive that
content even with a local GPU. Model loaders, node implementations, subgraph
definitions and the linked original workflow are preserved.

## Workflow support

simpleGen targets the **H3RC native-subgraph reference chain**, shipped as the
built-in **H3 Persistent References Chain** workflows
(`H3_Persistent_References_Clonable_Chain.json` and its local variant, packaged in
`src/simplegen/data/workflows/`). Your own templates of the same family can be
used from a linked Workflow folder. The workflow's custom nodes and compatible
models must exist on the chosen ComfyUI instance; the app installs neither. Preflight checks structure, compiler support and model
availability against the target before any agent work. A template that selects
models the target doesn't have is blocked, with the missing and available models
named. If a model disappears after preparation, the accepted work is kept: use
**Repair…** to choose installed models, without another agent pass. Resume reuses
accepted agent work. Model selections are never silently substituted.

New runs (state schema 5) are compiled: the agent returns a creative plan and
code builds the native workflow (see [WORKFLOW_EDITING.md](WORKFLOW_EDITING.md)).
Runs saved by older versions (schemas 1–4) keep their original preparation path.

**H3 Persistent References Chain — Local INT8/NVFP4 (built-in)**
(`H3_Persistent_References_Clonable_Chain_LOCAL_INT8_NVFP4.json`) selects the INT8
reference model and NVFP4 AWQ text encoder; the Cloud BF16 workflow selects the
BF16 models. Choose the local one for a local run.
Named widget copies are ignored by structural validation; the normal widget
values determine execution.

Duration means total assembled film duration, accounting for context overlap.
Code offers up to three legal segment layouts for the requested duration. The agent
picks one to fit its story beats, and code builds the chain (older runs let the
agent edit the chain using the same legal-frame guidance). The validated duration
and dimensions appear in the report. Megapixels determine
the target pixel area; Aspect ratio selects 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9,
or 21:9. Width and height are rounded to the 32-pixel grid, so the final ratio can
differ slightly from the selected target. Older saved runs without a ratio retain
their template-based sizing when resumed. Very short requests
are bounded by the workflow's minimum legal window. Existing motion/audio assembly
is preserved. Each raw shot, including its context frames, also gets a video save.
The final cut is the explicitly identified final save node, not the newest file.

### H3 Scene Chain (h3scene)

The third family runs the reviewed H3 scene chain. Choose
**H3 Scene Chain — Local INT8 (built-in)** in the Workflow list for a **Local**,
**Qty 1** run. The app manages its template under the app home; no Workflow
folder or Python command is required. It remains listed on Cloud, where preflight
blocks it with the existing Local-only explanation. The linked
Project folder must contain `INDEX.md` and the asset cards under
`characters/`, `places/` and `props/` (`*/card.md`). Your script or concept is
used verbatim. The selected agent receives the approved production prompt,
requested seconds, your text, index and cards only; discovery tools are disabled.
Code checks the dialogue, picture paths and card constraints, then rebuilds the
approved graph with INT8 MultiGPU loaders (DiT/VAEs on cuda:0, Qwen3-VL on cuda:1),
22 context frames, extra trim 3 and plain concatenation. Steps and resolution
come from the Generate settings. The report shows both the plan's segment sum
and the delivered duration after frame snapping.

A plan exceeding Duration by more than 15 seconds, or flagged by the agent,
stops before compilation or upload. Review the saved reason and choose
**Approve**, **Edit as new run** or **Deny**; each decision saves a linked revision.
Approval uses the saved plan without another agent call. Local scheduling uses
the same preparation path. Cloud/RunPod, batches, variations and Repair are
unsupported; use Edit as new run to change a scene-chain workflow.

Optional developer tool: `python3 -m simplegen.h3scene template /path/to/workflow-folder`
writes `H3_Ref2VA_Scene_Chain_LOCAL_INT8.json` into your own Workflow folder. Existing
copies remain selectable and work as before.

Preflight reports the tested local envelope (a two-GPU workstation): up to 60
seconds at 960×544 and up to 30 seconds at 1120×640. Larger requests are reported
as unverified with the nearest tested point, without blocking. Requests at least
about 45 seconds and 0.7 MP show a host-RAM warning: 60 seconds at 1120×640 was
killed at final merge on a 64 GB host. The H3RC and h3voice families retain their existing
behavior.

#### H3 Scene Chain + RefMod

**H3 Scene Chain + RefMod — Local INT8 (built-in)** is the same chain with
MiniMaxH3Mod RefMods (packages built with refmodBuilder) used like reference
pictures. Give a card a `## RefMod` section holding the package's loader name,
one per line, for example:

```
## RefMod
character/mara_20261006_104720_aaa4dc
```

The agent receives the production prompt plus a RefMod section and lists its
choices in the plan's `refmods`. Code checks that each name comes from a card,
that a card with a RefMod contributes no picture (the RefMod replaces its sheet),
that `## Verbatim constraints` appear in every segment, and that the plan has at
least one picture or RefMod (pictures may be zero). The graph is the
approved builder's graph plus one `MiniMaxH3RefModsLoader` (8 slots, Visual only)
and, in every segment, one `MiniMaxH3RefModApply` between the Ref2VA conditioning
and the Extender or guider, with settings from tested RefMod runs and no
combined RefMod token limit (the usable range is still being tested). See
[Writing RefMod cards](docs/REFMOD_CARD_EXAMPLE.md). Preflight fails when a card names a RefMod that
ComfyUI does not list (or a refmodBuilder work file). RefMods were tested only in
single 5-second shots, so every request reports an unverified envelope.

### Single-shot voice workflows (h3voice)

A second template family renders **one** H3 Ref2VA shot (4–15 s, Qty 1) in which
reference pictures lock identity and short **voice clips** set how each character
sounds; the spoken words come from the dialogue the agent writes (the clip is a
voice-timbre reference, not played back). Pick it per destination:

| Destination | Workflow | Loaders |
|---|---|---|
| Local | **H3 Voice Single Shot — Local INT8 (built-in)** (`H3_Ref2VA_Voice_SingleShot_LOCAL_INT8.json`) | ComfyUI-MultiGPU: Ref2VA pruned INT8 DiT (21 GB) and both VAEs on cuda:0 (RTX 5070 Ti); Qwen3-VL-32B INT8 encoder (27 GB) alone on cuda:1 (RTX 4070); KJNodes `VRAM_Debug` frees VRAM before decode; `beta` scheduler |
| RunPod | **H3 Voice Single Shot — Cloud BF16 (built-in)** (`H3_Ref2VA_Voice_SingleShot.json`) | Core loaders only, default device: the four staged BF16 files |

The agent returns a structured shot spec (pictures, voice clips with trim
start/duration, subjects, dialogue) instead of prompt sections; code numbers the
`<Picture N>`/`<Audio N>` references, writes the six-block prompt, checks speech
budget, 15 s audio cap and quoted concept lines, and builds the graph. Voice files
(.wav/.flac/.mp3/.m4a/.ogg) must be inside the linked project folder; they are
cached, uploaded and receipted like images. Preflight fails a single-shot
workflow when Qty > 1, Duration > 15 s, a needed node pack (MultiGPU, KJNodes) is
missing, or a MultiGPU (local) workflow targets the cloud. Repair of single-shot
runs is not supported yet (use Edit as new run). Both workflows ship built in.
simpleGen vendors the pure modules of the separate h3voice project in
`src/simplegen/h3voice` (refresh with `H3VOICE_SRC=<h3voice source folder>
python tools/sync_h3voice.py`).

Other custom workflow families, API-only flattened graphs, ambiguous broadcast
connections and arbitrary topology redesign are not supported by the compiler. They fail clearly rather than being partially executed. This app does
not install nodes/models. New managed RunPod sessions deploy the selected Global
Volume GPU from the desktop after local setup.

## Cloud and outputs

Provide your running ComfyUI URL and turn on **Cloud GPU**. An amber billing notice
stays visible on both tabs. If the endpoint requires a bearer token, set an
environment variable (default `SIMPLEGEN_COMFY_TOKEN`) before launching and enter
its name in Settings. Tokens are not saved in project files. For an HTTPS gateway
using HTTP Basic authentication, set **Browser login file** to a local JSON file
with `username` and `password` fields. This takes precedence over the bearer token
and authenticates HTTP and WebSocket requests. Only the file path is saved.

For an existing RunPod setup folder, click **Settings → Use existing RunPod
setup…** and select it, then Save. This reads the Global Volume template metadata
(`global-comfy-template.json`, whose ID must match `template_id` in your Global
profile), links `browser-login.json` and the cached node schema
(`global-global-object-info.json`), and enables desktop-managed cloud generation. Import does not rent a pod. See
[RUNPOD.md](RUNPOD.md).

simpleGen talks to RunPod's REST v2 API. The one exception is creating a Global
Volume GPU pod, which still uses RunPod's GraphQL API because REST v2 can't attach
a Global Volume yet. GraphQL is retired in January 2027.

For managed sessions, enable **Manage Global Volume GPU from this desktop**,
scaffold the local profile in Settings, enter your RunPod template ID and Global
Volume ID in it (`template_id`, `volume_id`; see [RUNPOD.md](RUNPOD.md)), then link
the cached ComfyUI schema. The
Generate page shows fresh GPU availability and estimated cost options before any
paid GPU is requested. The selected GPU and session identity are saved in the
project, so reopening simpleGen reconnects to the same work. Closing the window
leaves the session running and pauses automatic shutdown. After every returned
shot and the final cut are downloaded, checked by byte count and probed locally,
the desktop acknowledges the complete manifest. Keep simpleGen running for five
minutes of empty queues so it can retire the GPU. Manual shutdown is an explicit
override; undownloaded GPU outputs are lost when its disposable disk is deleted.

Every concept gets a unique folder containing:

```text
state.json              Saved job ID, settings, edit attempts and download receipts
template.original.json  Original native workflow snapshot for comparison
workflow.json           Accepted native workflow edited directly by the agent
validation.json         Accepted attempt and artifact checksums
script.txt              Generated script
plan.json               Agent script and consulted reference paths
workflow.api.json       Validated workflow converted for execution
used_references.txt     Filenames/full source paths only
references.json         Cached copy of each reference: original path, sha256, size, blob
report.txt              Success/failure, final path and video information
agent-edits/            Separate attempt copies, CLI logs and validation errors
events.jsonl            Progress and diagnostics
final/                  Final cut
shots/                  All returned shot/media outputs
```

No existing videos, images or documents are copied into the project. Selected
references are uploaded to ComfyUI when needed.

**Reference cache.** When agent work is accepted (and for repair, salvage and
retarget revisions) every reference the run uses is copied, checked by sha256,
into `~/.simplegen/reference-cache/blobs/` and listed in the run's
`references.json`. Uploads, batch fingerprints and saved-run preflight use those
verified copies, so an accepted run, schedule or batch proceeds while the linked
project folder (e.g. an SFTP mount of a sleeping Mac) is unavailable. A copy
whose hash no longer matches is reported and never used; the original is used
only if it still matches the recorded hash. Copies are released when a run,
schedule or batch completes successfully; failed, cancelled, detached and
not-yet-run work keeps them for Repair and Resume. Settings shows the cache size
with **Clear unused**. Runs saved before the cache read their originals as before;
**Cache references now** in the saved-run dialog snapshots one. If a reference
cannot be read at acceptance, nothing is accepted and Resume retries without
another agent pass. Remote latent checkpoints remain
provider artifacts; the downloadable package contains the final video and every
saved shot video. Technical files (the editing contract, agent request/schema and
Comfy history) also remain in the run folder for diagnosis. New runs use state
schema 2; older schema-1 projects retain their original preparation/resume path.
Each edit attempt persists before the CLI starts, so Resume cannot reset the
two-attempt limit. Accepted edits are revalidated without further agent usage.
See [WORKFLOW_EDITING.md](WORKFLOW_EDITING.md) for the validation boundaries.

**Resume saved run…** opens the saved concept first. **Resume unchanged** reconnects
the original endpoint/job and finishes missing downloads. Editing the concept
disables that choice; **Edit as new run** places the revised concept and saved
settings on Generate. You can review them before clicking Generate. That click
creates a new project and starts new agent work, while retaining the source
project and its history. The new project uses the source's saved workflow
snapshot when the linked workflow file has changed or disappeared. The same
new-run action is available for schedules marked **Needs attention** in the
Scheduled list; pending schedules retain their in-place Edit action.

Resume does not submit an uncertain job a second time. Lost server history
or cancelled jobs are reported. A run interrupted before submission can continue
from its saved plan. Completed packages can be moved; video paths in state are
relative to the project folder. Linked original references may need relinking on
a different computer if making a new run.

**Generate anyway…** beside Resume is available for a saved agent response
rejected by the older extra-field format check. It shows the saved complaint
and extra fields, asks for confirmation, then rechecks the saved workflow,
model, inputs, settings and references without calling the agent again. The
override and artifact hashes are recorded in the run's validation receipt.

Cancel stops preparation/downloads and removes this job if queued. On ComfyUI
0.36+ it requests a targeted interruption for this job; older servers may keep a
running render going. **Cancellation/completion never shuts down an unmanaged
rented GPU; instance billing may continue.** For managed sessions, use **Shut
down compute** for an intentional override; ordinary completion waits for
verified downloads and the manager's empty-queue grace period. The progress bar
nodes and current sampling steps, not a predicted time-to-finish percentage.

## Development and packaging

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/python tools/build.py
```

Build on Linux for Linux, on an Intel Mac for macOS Intel, and on an ARM Mac for
macOS ARM. `dist/` contains the desktop bundle, including the skill and ffprobe;
the official agent CLI remains a separately installed, signed-in dependency.
ComfyUI can run elsewhere. Mac signing/notarization and hardware smoke checks are
separate release steps; a Linux test run cannot establish macOS compatibility.

See [DESIGN.md](DESIGN.md) for persistence boundaries.

Provider references: [Codex automation](https://learn.chatgpt.com/docs/non-interactive-mode),
[Claude Code automation](https://code.claude.com/docs/en/headless),
[Grok Build](https://docs.x.ai/build/overview),
[ComfyUI routes](https://docs.comfy.org/development/comfyui-server/comms_routes).

## License

MIT. See [LICENSE](LICENSE).
