## Manual GPU selection — September 26, 2026

GPU selection no longer invokes a writing agent or a resource-assessment agent.
Refresh Catalog populates both GPU selectors directly with Secure Cloud Global
Volume catalog types, including temporarily unavailable types for later polling.
Choose two GPU types manually, with per-type price limits, a batch ceiling, and
priority. New selections start blank. Script/workflow preparation and validation
must finish before scheduling or starting compute, but do not restrict the GPU
catalog. Availability and price are checked again at dispatch. Catalog choices
are saved with approval; existing approved sessions retain their ownership and
shutdown rules. No agent runtime/cost estimates or suitability labels are shown.
One-off sessions likewise use catalog choices without an agent assessment.
Recorded usage/cost CSV history remains available. No GPU compatibility guarantee
is implied by catalog membership.

# RunPod batches

A batch is a persistent list of concepts rendered sequentially on one RunPod GPU.
It does not increase a workflow's batch size. Qty still creates individual
variations of its concept. Each concept keeps its own project folder, script,
workflow copies, final cut, shots, and report.

## Prepare, approve, schedule

Managed RunPod generation defaults to Batch & schedule. Add concepts to a draft
batch, then prepare it. Preparation uses the selected agent and cached cloud node
schema; it does not rent a pod. Edit or remove entries before execution. Editing
invalidates the prepared workflows and GPU approval and retains prior work.

Opening Batches or switching Generate to managed RunPod checks the live secure
GPU catalog without renting compute. The catalog lists current price, global
availability, available-location count, and VRAM even before preparation;
the two selectors use this catalog without agent approval. Hover over Locations to see the available
data centers. These are location counts, not exact numbers of rentable GPUs.
New sessions explicitly attach Global Volume and omit a data-center restriction.
CUDA 13, Secure Cloud, the approved GPU type, and spending limits still apply.
RunPod's Global Volume contract supports any region; no per-host compatibility
flag is exposed by its catalog, and actual allocation can still reject stale stock.

After preparation, approve your two manually selected GPU types and a maximum hourly rate for
each. The editable **Batch hourly ceiling** starts at your profile default and can
be raised explicitly above it (the packaged profile default is $3/hour). Each GPU's individual ceiling must fit within
it. Schedule batch / Start now saves these approvals. The profile and already
owned sessions keep their existing ceilings. GPU types can be approved even when currently unavailable. Model loading,
rendering, downloads, and the keep-warm idle period (`keep_warm_minutes`, default
five) all contribute to billing. Hourly limits are not a cap on the final bill;
set the optional batch **spend cap** for that (additions over its projection are
rejected, and exceeding it alerts but never cancels work). No agent-generated runtime
or cost estimates are used.

Choose a local start date/time or Start now. The desktop must remain open and
awake. A future scheduled batch does not provision before its scheduled time and
does not keep another batch's GPU running to bridge the gap. If it becomes due
while another batch's pod is still inside its keep-warm window and is
compatible, it adopts that warm pod instead of starting a new one. Raise
`keep_warm_minutes` to bridge longer gaps.

## Availability priority

Availability and prices are checked again when due, and at the configured retry
interval (five minutes by default). No reasoning-agent call is needed for each
poll; GPU selection never invokes an agent.

| At a fresh poll | High priority | Low priority |
| --- | --- | --- |
| Both approved GPUs available | Cheaper GPU | Cheaper GPU |
| Only cheaper GPU available | Cheaper GPU | Cheaper GPU |
| Only more expensive GPU available | More expensive GPU | Wait for cheaper GPU |
| Neither available | Wait | Wait |

An exact price tie uses the configured order. Rates above an approved ceiling
cannot authorize provisioning. Status explains the wait and next check. Once a
pod starts, the batch keeps that pod rather than switching GPUs mid-batch.
A catalog or pre-allocation availability/price failure returns to timed waiting.
An error from the actual deployment request requires review and reconciliation
of its saved creation intent; it never triggers a blind second paid request.

## Completion and recovery

Prepared batch jobs are submitted to ComfyUI's queue as individual workflows.
ComfyUI renders them one at a time and starts the next queued render without
waiting for simpleGen to download the preceding job. The desktop collects final
videos and all generated shots while later jobs can render. Downloading stays
serial to keep memory and disk usage bounded.

A render or download failure flags the batch for review; jobs already queued on
the pod can continue. simpleGen attempts to collect the other submitted jobs
without submitting new work. Saved prompt IDs are retained for recovery. Closing
the desktop also leaves those queued jobs running.
The session becomes eligible for automatic shutdown only after the entire batch
is complete and all required files have been downloaded and verified. Both
ComfyUI queues must then remain empty for the **keep-warm window** of fresh
observations (Settings → Keep-warm window, `keep_warm_minutes`, default 5,
5–60; frozen into each new session, existing sessions keep 5).

## Adding work to a running batch

**Add to running batch / warm pod…** (Batches) or **Add to running batch…**
(Generate, RunPod destination) prepares a new concept (Qty ≥ 1) with the batch's
settings, cached cloud schema and Phase 1 preflight, outside the batch lock, while
the pod keeps rendering. Before queueing it shows the projected extra cost. It
then writes one request, `<batch>/inbox/<run_id>.json`, and marks the run
`pending_batch` so it cannot join two batches. The running batch is the only
writer of membership: before each member's render and once more before sealing
it validates each request (fingerprint, same GPU profile and volume, the compiled
workflow against the **live pod's** node schema, and the spend cap), records the
member (`added_at`), stages references and submits through the normal
registration path, then files the request under `inbox/accepted/`. Rejections go
to `inbox/rejected/` with a reason; the run is kept. Resending the same run is
idempotent.

Before acknowledging downloads the batch drains its inbox and records
`sealed_at`. A request that arrives after that, while the completed batch's pod
is still in its keep-warm window, **reopens** the batch (`reopen`): verified
downloads are cleared, existing jobs and receipts are kept, and completion then
re-verifies every job, old and new. After shutdown has started
(`delete_intent`) or the pod is retired, requests are rejected with "start a new
batch".

**Warm-pod handoff.** When a batch is due and another batch's pod is in its
keep-warm window with an approved GPU type, a rate within the approved ceiling,
the same profile and volume, and a live schema that validates every prepared
member, the due batch adopts that pod instead of creating one
(`adopted_from`). Both batch locks are taken in sorted path order; the source
records `handoff_intent` first and becomes `handed_off` last, and recovery
finishes or rolls back an interrupted handoff. A pod being handed off, or
handed off, is never deleted by the source's idle countdown. At adoption the
pod's current queue and history (plus the source's recorded prompts) are saved
as `inherited_prompt_ids` and count as known, so the adopting batch can still
retire through keep-warm or failure shutdown; a prompt that appears after
adoption and belongs to neither batch still blocks automatic retirement. If the
pod's history cannot be read, there is no handoff.

**Spend cap.** An optional per-batch total (`max_total_usd`, set in the approval
form; a warning is shown when none is set). The projection is billed time × the
actual hourly rate, plus remaining renders × this session's measured average
render time (unknown until one render completes), plus the keep-warm window.
Additions that would exceed the cap are rejected with the numbers; crossing it
in actual spend alerts. With a cap set and no render finished yet, adding asks
for explicit confirmation showing the amount billed so far and "render time
unknown". Queued or running renders are never cancelled automatically. Cost
history records a reopened batch in portions (`<id>` up to the reopen,
`<id>-reopenN` after it), and an adopted pod's cost from adoption, so no time is
counted twice. Running jobs each have their own control: **Cancel local run**
and **Shut down batch…**; the main Cancel only ever affects the local run.

Closing the desktop detaches. Reopening can reconnect using the persisted batch,
owned pod, and prompt identities. Completed jobs are not submitted again. A
failed job is flagged for review and stops automatic batch progress; availability
polling applies only to work that has not been allocated. After a failure the
failure policy applies: once the recovery loop has tried every submitted job
(`recovery_completed_at`), every successful job's outputs are downloaded and
verified, and the queues have been empty for the chosen **Failure shutdown**
period (default 15 minutes, 0 = off), the pod is deleted with reason
`failure_idle` and a failure summary is recorded. If a successful job's output
is not downloaded the pod is kept and the batch shows *outputs not collected*;
if the app stopped mid-recovery (no marker) the pod is kept. Resume clears the
failure and stops the countdown. See [RunPod](RUNPOD.md). Explicit manual
shutdown remains an intentional override, and can lose undownloaded outputs on
disposable pod disk.

One-off remains available for an immediate individual concept/Qty run using the
existing GPU-selection flow. It pays its own pod startup cost. A manually started
RunPod pod is not automatically adopted by either managed mode.

## Local out-of-memory failures

A failed local OOM run offers **Add to RunPod batch** beside its failure, and through
**Resume saved run**. Review the copied concept and generation settings, then select
a draft batch. This creates a new project and cloud-template snapshot; it never
reuses the failed prompt ID or modifies the local project. Normal batch preparation
creates fresh agent work before GPU approval.

Settings contains **RunPod default workflow**, **RunPod default diffusion model**,
and **RunPod default CLIP**. The transfer snapshots the cloud workflow and applies
those model/CLIP selectors, preserving VAE choices and other template settings.
The bundled defaults target the H3 BF16 model and the qwen3vl H3 ultra uncensored
heretic BF16 encoder. A workflow filename is relative to the current workflow
folder, or it can be an absolute path. Source reference assets are not duplicated
into the project; accepted members snapshot them into the local reference cache,
and the approval fingerprint uses their recorded sha256 (not the original's mtime),
so a batch proceeds while the linked folder is unmounted. A completed batch
releases every member's cached copies.

Batch dialog exits, including Start now, wait asynchronously for the catalog
worker to finish before destroying the window. The batch remains durably armed
while closing; normal dispatch proceeds after the dialog has safely exited.
