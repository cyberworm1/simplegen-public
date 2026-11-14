# Local scheduling

Scheduling is local-only and runs inside the desktop, with no background service.
Generate with Schedule checked saves a project identity, settings, concept and
native template snapshot immediately. It does not contact the agent or ComfyUI,
upload assets, provision compute or submit a prompt. References remain linked
until the agent's work is accepted; from then on the run uses its verified local
reference cache (`references.json`), so a prepared schedule dispatches while the
linked project folder is unmounted. Removing a schedule keeps the cached copies.
An unchecked Schedule runs immediately; pending requests do not occupy the worker.

Each scheduled request has its own atomic record under the user configuration's
`schedules` directory. A process-held advisory lock permits only one desktop to
dispatch schedules. Projects retain their normal exclusive run lock and durable
submission reconciliation. Dispatch intent persists before starting the worker;
interrupted dispatch is offered for explicit Resume, never blindly restarted.
Schedule removal retains project files and never cancels ComfyUI execution.

The picker uses local system time, `MM/dd/yy HH:mm`. Store an unambiguous UTC
instant plus the selected local time/zone. Reject past times and nonexistent or
ambiguous daylight-saving times. A later timezone change does not move the saved
instant. Opening the app after a deadline, or a timer gap over 60 seconds spanning
a deadline, marks the request overdue. A never-started request remains eligible
for automatic availability checks. A blocked check saves its reason and a retry
instant 30 minutes later; this instant survives restart. Run now explicitly
bypasses the wait. Started, failed, cancelled, paused, removed, and uncertain
requests are never automatically retried; a started request needs explicit Resume.

## Preflight gate

`Scheduler.add` requires a preflight result (`simplegen/preflight.py`). `fail`
rejects the schedule and publishes no record. `unverified` is accepted only with
an explicit override, recorded as `preflight.override_unverified` in both the
schedule record and project state. The project state stores the receipt
(`preflight`: status, result, schema hash, template hash, timestamp).

At dispatch, after the queue is found idle, the availability probe re-runs
preflight on the saved `template.original.json` and saved settings against the
live local schema, before any agent work. `fail` moves the request to
**Needs attention** with the specific reason (shown in the Scheduled list) and
never starts it. `unverified` follows the 30-minute blocked-retry path above
(never started, retried later). A pass is recorded as `preflight_dispatch`.

## Repair and reschedule

A Needs-attention row whose run has accepted agent work but was never submitted
offers **Repair…** (`simplegen/repair.py`). The repair publishes a new linked
project; the original record becomes `repaired` with `repaired_to`
(`run_id`, `folder`, and `record_id` once the revision is scheduled). The
Scheduled list shows it as “Repaired → <revision folder>”; it is never
dispatched again, cannot be run now, rescheduled or repaired again, and can
only be removed. Its history is kept.

`Scheduler.add_prepared` schedules an already-prepared project without
`create_run`. Like `add`, it requires a passing (or explicitly overridden
unverified) preflight; the repair's own preflight is used. Its record carries
`prepared: true`, which lets the dispatch check accept a project that already
holds `workflow_acceptance` while still requiring that it is unreleased and
never submitted. Dispatch runs the same preflight re-check and then reuses the
accepted work; no agent is called.

**Remove** on a never-started request releases its project: the schedule data
moves to `schedule_history` in the project state (with the record id and
removal time) and the state leaves `scheduled`, so Resume works (a prepared
revision reuses its accepted work) and `add_prepared` can schedule it again.
A removed record never dispatches; when its id is reused, the removed record is
archived under `schedules/history/`. Removing a started request changes only
the record. Repair from a Needs-attention row first offers an existing,
not-yet-linked revision of the same run instead of creating a duplicate.

Due requests wait for the local run slot and an empty local ComfyUI queue; a
running cloud batch has its own slot and does not delay them (and cloud batch
dispatch does not wait for local work). **Queue locally** on Generate is a
one-click schedule that is due now (Phase 1 preflight applies). Availability
is visible below the Generate controls: the Scheduled button counts waiting
requests, and a separate status line explains the next request's blocker. The
list shows each request's reason (busy ComfyUI, connection failure, another
simpleGen operation, open dialog, or earlier scheduled requests) and next retry.
These messages
do not replace active generation progress. Queue probe results travel as a typed
dictionary payload so an idle boolean survives the worker signal unchanged.
Availability reads run outside the Qt thread. The engine also waits for empty queues immediately
before each scheduled submission, because other work may arrive during agent
preparation. There is no cross-client atomic reservation of an external ComfyUI
queue; another client can race a final empty-queue check. Existing prompt IDs are
reconnected without waiting for their own execution to finish first.

Cloud scheduling and unattended rental remain disabled. Qty schedules the whole
existing sequential variation set. No new shutdown policy or recurring schedules
are introduced. Existing unscheduled project schemas remain compatible; schedule
metadata is additive. Tests use simulated clocks/providers and temporary folders.

The Scheduled list can edit a request that has never started. The edit dialog
changes its concept, local workflow, agent/model choice, generation settings,
and local date/time. It does not change the Generate page's draft. Each Save
stores an immutable revision with the exact previous project state and workflow
snapshot. The schedule record carries an edit marker until the new state,
snapshot, and record are durable. Recovery finishes a verified interrupted edit
before dispatch or marks it for review if the revision is inconsistent. An edit
of concept/settings alone keeps the original workflow snapshot even if the
linked source file later changes.

Started schedules marked Needs attention offer **Edit as new run…**. This opens
the saved concept chooser. Resume unchanged reconnects the existing project;
Edit as new run stages a fresh project and keeps the original history intact.
No agent or compute work starts until Generate is clicked.
