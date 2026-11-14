# Direct workflow editing

## Schema 5: the agent writes content, code builds the workflow

New runs use state schema 5 (`simplegen/compiler.py`, `editing.prepare_compiled`).
The agent runs read-only (Codex `read-only` sandbox, Claude `Read,Glob,Grep`,
Grok read-only; no Edit or Write). It receives a workflow brief: up to three legal
timing options (the recommendation plus the nearest layouts with one fewer and
one more segment; per segment frames, `duration_override`, usable and overlap
seconds and a speech budget of `SPEECH_WORDS_PER_SECOND` = 2.5 words/s), the
template's prompt section names in order, the reference slots and which
`<Picture N>` each becomes, resolution, steps, duration and a read-only copy of
the workflow. It returns a creative plan: `script`, `used_references`,
`timing_option`, `references` (slot node id → absolute local image) and
`segments` (one `{"sections": {...}}` per segment), plus `variations` when
Qty > 1.

Code checks the plan (every section present and non-empty, no template sample
text, balanced `<d>` tags, consistent `<Subject N> (Sn)` pairs, dialogue within
each segment's budget (over budget ×1.2 + 3 words is a recorded warning in
`workflow_acceptance.lint_warnings`; over ×1.5 + 3 words is an error), `<Picture N>` within the slot
count, every slot filled with an existing image inside a linked folder, and
quoted concept lines spoken verbatim, in order, once). Problems feed the single
correction pass. The compiler then builds the native workflow deterministically:
it reuses the template's segment cards for the first segments (keeping their IDs,
report previews and wiring), clones the last card with fresh node and link IDs
for further segments, drops surplus cards with their report previews, wires the
`previous` chain, project fan-out and final `SaveVideo`, and sets
`duration_override`, `steps_override` 0, project steps/resolution, rendered
prompts, seeds and reference paths. Definitions, loaders and all other nodes are
unchanged. Every compiled workflow must pass `validate_edited_workflow`
unchanged. The compiler accepts every link form the validator accepts (6- and
5-element lists and link objects, keeping the template's own form), nodes
without output records, and templates whose segment prompt has no named
sections (one free-text `prompt` section). Phase 1 preflight's
`template_structure` check dry-runs the compiler (every timing option, with
placeholder content, validated) for new and schema 5 runs, so a template the
compiler cannot build fails before any agent call; older saved runs skip it. The accepted plan and compiled workflow are hashed in
`workflow_acceptance` (with `compiler_version` and `timing_option`).
Just before acceptance (schemas 2–5, and repair/salvage/retarget revisions) every
`LoadImage` slot value and `used_references` entry is copied into the local
reference cache and listed in `references.json`; an unreadable original fails
acceptance without spending the correction pass, and Resume retries it. The
agent still reads the originals while writing.

**Compatibility.** Schemas 1–4 keep their existing paths unchanged (schemas 2–4:
the agent edits a workflow copy under this document's rules below). Schema 5
runs need this release or later; an older release refuses them as an
unsupported project version. Repair (Phase 2), preflight (Phase 1), batches,
variations and cost safety (Phase 5) work with schema 5 runs.

**Rebuild from saved agent work** (`repair.salvage`) turns a schema 2–4 run that
failed workflow validation into a schema 5 revision without an agent call: the
last rejected attempt's card prompts (chain order from its links, else from
"SEGMENT NN" titles), `plan.json` script and `LoadImage` paths become a creative
plan, which is linted, compiled against the run's own snapshot, validated, and
published as a linked revision (`repaired_from.kind = "salvage"`). The timing
option must have the same segment count, or it is refused. Speech-budget
overruns in salvaged work are recorded as `lint_warnings`, not blockers.
Salvage publishes after lint and validation; the app then preflights the
revision and, if that fails, opens Repair/Retarget for it immediately.
Retarget records the old and new template path and sha256 in `repaired_from`.
**Use another workflow file…** in the Repair dialog (`repair.retarget`)
recompiles a schema 5 plan against another template when its prompt sections and
reference slot count match, then preflights before publishing.

## Single-shot voice templates (schema 5, h3voice family)

A workflow whose `extra.h3voice.kind` is `ref2va_voice_single_shot` skips the
chain compiler. `editing.prepare_compiled` hands it to `_prepare_single_shot`,
which shares the attempt/correction/acceptance loop (`_plan_attempts`). The agent
(skill `h3-voice-shot`, read-only) returns `{script, used_references, shot}`;
`singleshot.check` validates paths (inside linked folders, readable image/audio),
runs the h3voice lint (speech budget, 15 s audio, tags in range, balanced `<d>`)
and the quoted-concept-line rule; `singleshot.compile_graph` builds the API graph
from the template's embedded profile, and `singleshot.verify` enforces the node
allow-list, the template's exact model selections, links and one final SaveVideo.
`workflow.json` stores that API graph (hashed in the receipt with
`single_shot`, `h3voice_format` and `profile`). Seeds derive from the run ID;
Steps, Megapixel and Aspect come from settings; Duration is the clip length.

Both voice workflows ship built in (**H3 Voice Single Shot — Cloud BF16** and
**— Local INT8**), as do the two **H3 Persistent References Chain** workflows
(Cloud BF16 and Local INT8/NVFP4). They are packaged as JSON in
`src/simplegen/data/workflows/` and installed to
`<SIMPLEGEN_HOME>/workflows/builtin/` like the scene chain. Their example
prompts are blank: the agent writes every prompt.

## H3 Scene Chain templates (schema 5, h3scene family)

`extra.h3scene = {kind: "scene_chain", format: 1, profile: {...}}` identifies the
third family. Choose **H3 Scene Chain — Local INT8 (built-in)** in the app's Workflow
list for Local Qty 1; a Workflow folder is optional. Startup and refresh write
the one-segment native example atomically under
`<SIMPLEGEN_HOME>/workflows/builtin/H3_Ref2VA_Scene_Chain_LOCAL_INT8.json` only when its
content differs. A write failure is logged and omits the entry. The embedded
profile records the approved fixed settings and
model/device selections. Preflight checks its native graph against the target
schema and available models. Every run rebuilds its graph from the approved
profile and plan rather than editing that example.

The built-in remains listed on Cloud, with the existing Local-only preflight
block. Settings retain its absolute path independently of each destination's
Workflow folder. Schedules and runs snapshot it normally; Resume and Edit as
new run use the saved snapshot. The optional developer command
`python3 -m simplegen.h3scene template <folder>` writes the same template into a
user Workflow folder, and previously generated copies still work.

Compatibility readers preserve earlier saved review decisions and acceptance
receipts; new writes use the scene-chain names. Earlier built-in selections
resolve to the current template. An obsolete app-owned example is removed only
when its bytes exactly match the example the app previously wrote; modified
files and user exports remain intact.

`h3scene/` holds the scene-chain graph builder, plan schema and production checks.
They are simpleGen's own code (no external source or sync step). `h3scene_family.py` handles settings, prefixes, references and
structural verification. It uses the shared `_plan_attempts` loop with exactly
one correction and no unfinished-response allowance. `workflow.json` is the
compiled API graph; its acceptance receipt records `scene_chain`, `h3scene_format`,
plan/workflow hashes and checker warnings.

The agent's skill is `skills/h3-scene-production/SKILL.md`. Its input is the
`produce.build_messages` result, with the linked project as the asset folder: requested
seconds, the user's text verbatim, `INDEX.md`, and every `card.md` under characters,
places and props with path headers. It receives no bible, project inventory or
workflow brief as additional context. Each provider runs from a neutral temporary
working directory, with discovery tools disabled and the user's chosen model
and reasoning. The response is one plan in a `plans` envelope, with
`concept_id`, `title`, `script`, `input_mode`, `pictures`, `segments` and optional
`warnings`/`overflow`. Code restores the user's script and identifier, checks all
picture paths and image contents, strict dialogue in order (duplicates counted),
card verbatim constraints, joins and the six prompt blocks in order. `..` paths
and symlink escapes are rejected. Timing follows the builder's 17k+5 lattice,
including continue context and its trained-range warnings. Seeds come from the
run ID; render dimensions use the existing 32-pixel-grid mapping.

Computed overflow above +15 seconds or an agent review flag saves the plan and
`h3scene_review` hash/reason, leaving the attempt pending and the run in
`needs_review`. Compilation, reference capture/uploads and rendering have not
started. Saved-run **Approve**, **Edit as new run** and **Deny** each publish a
new linked revision (`forked_from`, `h3scene_decision`) while preserving the source.
Approve uses the exact hashed canonical review plan, without a new agent pass;
failed validation cannot replace it with new agent work. Edit starts fresh
preparation; Deny cannot render. Accepted scene-chain runs resume with the ordinary
hash verification and reference-cache path.

Only Local Qty 1 is supported. Cloud/RunPod MultiGPU, batches, variations and
Repair are refused with reasons. Local schedules snapshot and dispatch through
the same unchanged schema-5 preparation path. The tested envelope is an
informational preflight check: verified at ≤60 s / 960×544 or ≤30 s / 1120×640,
otherwise unverified with the nearest tested point. At ≥45 s / ≥0.7 MP it also
names the final-merge OOM on a 64 GB host. These envelope findings never block
or require a preflight override. The report includes the plan's real segment
sum and snapped delivered duration.

## Schemas 2–4

Runs from schemas 2–4 use this flow. The app stores an original native workflow snapshot,
then gives the selected agent a separate working copy. The agent directly edits
that JSON, including existing subgraph clones and exposed timing/quality fields.
The app provides deterministic timing/dimension guidance, validates the result,
converts it to ComfyUI execution format and stages references. It does not apply
the former automatic creative graph expansion to new runs.

The original snapshot and each attempt are separate. Validation occurs before
uploads/submission. One correction attempt is permitted for a malformed response
or invalid workflow, with the exact validation error and the prior attempt's copy.
Authentication, process failure, cancellation and timeout are not retried.
Attempt reservations persist before the CLI starts so restarting the app cannot
reset the usage limit. Partial attempts remain available for diagnosis.

The accepted native JSON, structured agent response and validation receipt are
saved before upload. Resume validates accepted artifacts again, then continues
without another agent call. Unknown submissions never cause another render.
Schema 1 runs retain the original preparation path; schemas 2–3 retain their direct
editing path. Existing workflow/script history is not migrated or overwritten.
Schema 4 adds sequential variations and remains readable only by current releases.
For Qty > 1 the agent returns additional complete segment prompt sets alongside
the base edit. Code substitutes only those prompts and seeds in numbered native
copies, validates each copy and assigns separate transport output paths. Model,
topology, resolution, timing, reference and batch settings remain shared. See
[VARIATIONS.md](VARIATIONS.md) for durable parent/child and RunPod semantics.

Validation is deterministic: unchanged subgraph definitions, existing node types,
permitted controls only, established continuation wiring, legal duration and
dimensions, prompt/reference completion and live ComfyUI input compatibility.
The agent gets no instruction to change node code or the original template.
Provider tool permissions restrict editing to the attempt workspace/file where
the installed CLI supports it. These permissions and JSON validation do not
constitute an OS sandbox for every third-party CLI.

GPU availability selection, cost history and session management are integrated
through the resource and managed-session adapters. Sequential variations share
one session; the final verified receipt covers the entire requested set.
