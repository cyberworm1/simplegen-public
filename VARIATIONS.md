# Sequential variations

Qty defaults to 1. Larger quantities use one creative preparation (plus the existing
single correction opportunity): the agent edits the base native workflow for variation
1 and returns scripts and complete segment prompt replacements for the remaining
variations. References, topology, models, quality, duration and dimensions are shared.
Code changes only promoted prompts/seeds in native copies; existing transport code
assigns unique output paths. Workflow batch sizes never increase.

Schema 4 adds a parent sequence and numbered child projects. Schema 1–3 runs remain
readable without migration. The parent owns the lock, agent artifacts, upload cache,
RunPod session and ordered child identities. Every child has its own immutable accepted
workflow/plan, submission intent, prompt ID, download receipts and result. Children
cannot be resumed independently; Resume on a child resolves to its parent. A crash
during child creation can recreate only an uncommitted child, before submission.
Completed child files are checksum-verified on resume. An uncertain submission uses
the existing reconciliation path and never blindly resubmits. Failure pauses the set.

Cost CSV compatibility: the appended `quantity` column defaults to 1 for existing
rows. Writing history atomically upgrades the previous exact header while preserving
all old rows. New sequence rows record total rendering time and the video count;
resource estimates apply per-video runtime to the count deterministically. Unrelated
or unknown CSV layouts remain rejected.

Local and one-off sequences submit each child after the preceding child's outputs
are downloaded and verified. In a managed RunPod batch, prepared children are
queued ahead as individual workflows; ComfyUI renders sequentially while the
desktop downloads completed outputs. A download failure does not cancel jobs
already queued. References are cached by endpoint and content hash. The GPU quotation
describes the whole sequence, with unchanged peak VRAM. One managed session registers
each child job separately; download acknowledgement is withheld until every requested
child is complete and all receipts match the remote manifest. Existing five-minute
empty-queue retirement then applies. Detach, failure and manual shutdown retain their
existing semantics; no new automatic shutdown exceptions or services are introduced.

The Watch selector lists completed final cuts, including those from a paused set.
Each numbered folder retains its own script, native/API workflows, final and shots;
references remain filenames only. No paid resources are used by development checks.

## 0.4 changes (September 28, 2026)

- **Schema 5 (compiled) sequences:** the agent returns one creative plan with
  the shared timing option and references, plus `variations` (each with its own
  script and per-segment prompt sections). Code compiles each variation's native
  workflow from its own segments; seeds follow the existing rules. Content checks
  apply to every variation. Schema 4 sequences keep their original path.
- **Reference cache:** the sequence parent's `references.json` covers every
  child; children don't hold separate pins. Pins are released when the whole Qty
  set succeeds (or, in a batch, when the batch completes).
- **Batches:** a Qty > 1 concept can be added to a running batch through the
  additions inbox, and its children are queued like the other members.
- **Retirement:** "no new automatic shutdown exceptions" is replaced by the
  failure policy and keep-warm window (see
  [RUNPOD_BATCHES.md](RUNPOD_BATCHES.md) and [RUNPOD.md](RUNPOD.md)). A failed
  child still pauses the set, and the pod shuts down only after every collectable
  output is verified.
- **Not yet supported:** Repair and salvage of Qty > 1 sequences (they're
  refused with a clear message).
