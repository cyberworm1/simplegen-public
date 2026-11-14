# RunPod Global Volume GPU

Managed multi-concept sessions: [RunPod batches and scheduling](RUNPOD_BATCHES.md).
New managed requests default to Batch & schedule; One-off remains available.

New simpleGen cloud sessions use your own RunPod pod template (ComfyUI with the H3
nodes) and your own RunPod Global Volume holding the models. Enter both IDs as
`template_id` and `volume_id` in the Global profile (`global-profile.json`, created
by **Create RunPod configuration…** in RunPod setup); the app ships none and refuses
to rent compute until both are set. The GPU request explicitly
attaches the volume at
`/workspace-global`. The template supplies ComfyUI and its authenticated 8189
gateway. New GPU deployments request 200 GB of disposable container disk so
startup can stage the selected H3 models locally while retaining free space.
The deployment requires CUDA 13.
New Global Volume deployments leave the data center unspecified so RunPod can
allocate the approved GPU type across eligible Secure Cloud regions. The volume
is explicitly attached; no old regional volume is used. Catalog availability is
aggregated across all returned locations. Counts describe available data centers,
not exact GPU inventory. Live allocation remains authoritative.

**Settings → Use existing RunPod setup…** imports a setup folder
(`gpu-session.json`, `browser-login.json` and a cached node schema) to link its
private browser login, Global Volume template metadata and cached node schema.
Import never adopts an existing pod. Set a protected RunPod API key
in the local credentials file or environment; credentials never enter project
state or the deployment template. Setup and import make no paid requests.
simpleGen presents current GPU options for each new run; choosing one authorizes
one GPU deployment for that run. A unique pod name and create intent are saved
before the API request. If the response is uncertain, simpleGen reconciles that
exact owned name and does not repeat the create request.

The built-in **H3 Persistent References Chain — Cloud BF16** workflow
(`H3_Persistent_References_Clonable_Chain.json`) uses
`minimax_h3_ref2va_bf16.safetensors` and
`qwen3vl_32b_h3_ultra_uncensored_heretic_bf16.safetensors`. Its project widget list
matches the cloud H3RC node: width/height controls, without the local-only
`size_mode` and `megapixels` widgets. simpleGen still computes width/height from
the selected Megapixel and aspect ratio. The local INT8/NVFP4 workflow is
a separate built-in. Refresh the cached `/object_info` when the cloud node packages change.
Manually launched pods are not adopted by managed sessions; direct URL mode
requires an explicit unmanaged connection and leaves lifecycle control manual.

The desktop owns the new session. There is no CPU manager, S3 journal or extra
pod. Keep simpleGen running for automatic shutdown. It downloads every generated
video through authenticated `/view`, verifies positive complete byte counts and
probes each video locally. New outputs and inputs live on the disposable GPU
container; models live on the Global Volume. Normal termination requires all
downloads acknowledged and five minutes of fresh observations with both queues
empty. Closing simpleGen pauses that countdown; reopening resets it. Explicit
shutdown can discard outputs that have not been downloaded. Provider failures,
unknown ownership and missing readiness leave compute for inspection.
The app monitors every indexed owned session while it is open, even when another
project is selected. Its RunPod banner shows active sessions and attention needs.
First startup stages selected models before the authenticated gateway becomes
ready. The readiness wait uses **GPU startup timeout** (Settings, default 45
minutes, minimum 5), and can be cancelled or detached while startup continues on
the pod. If the wait expires, the session records a `startup` failure and alerts.
If no job was ever registered it is retired after the failure grace period below;
otherwise the normal failure rules apply.

**Failure shutdown.** Never pay for an idle
pod, including after failures. When a managed run or batch fails (not detach,
not cancel) the owning session records `owner_failure`; a batch also records
`recovery_completed_at` once its recovery loop has tried every submitted job.
Once every job the pod reports as successful has all its outputs downloaded
and verified locally, nothing runs, both queues are empty and observations are
fresh and healthy for **Failure shutdown** minutes (Settings, default 15, 0 =
off, max 240), the desktop deletes the pod (`delete_intent: failure_idle`) and
confirms absence with a follow-up read. A successful job with an undownloaded
output is never deleted: the session shows *outputs not collected* and alerts
until you resume. Unregistered jobs, a missing recovery marker, a poll gap, a
new desktop process, queue activity or an unhealthy observation block or
restart the countdown. Resume clears the failure. An uncertain delete is never
reported as retirement and is retried on a later poll.

A job reported successful whose outputs were never observed counts as not
collected. Jobs with no known outcome (`missing`, or `submission_pending`
without a prompt) keep the GPU and raise an *unresolved jobs* alert.
The app keeps polling every other owned session while a batch runs, so their
idle and failure countdowns continue; only the running batch's own session is
observed by its worker.

**Alerts.** Owner failure while billing, failure countdown started, retirement by
the failure policy, outputs not collected, unresolved jobs, startup timeout and
uncertain delete each alert once per session: a desktop notification (system
tray where available, otherwise the RunPod banner) and, optionally, a plain-text
POST to **Alert webhook** (ntfy-style). The webhook URL is kept only in global
settings (never in project or batch state) and its current value is used when
sending. The payload names the batch/run, event, GPU
type, hourly rate and elapsed billing time; it never contains credentials,
endpoints or file paths. Delivery failures never affect lifecycle decisions.

**Schema drift.** Once per session after readiness the live `/object_info` is
saved as `live-object-info-<sid>.json` beside the session state and compared with
the configured cache (`schema_drift`: hashes, added/removed node types, changed
loader model lists). The cache is never changed automatically; Settings →
**Use latest live schema…** keeps the old file under a timestamped name and
installs the snapshot. Armed and prepared batches then need preparing again.

Legacy regional schema-v2 CPU-manager sessions remain readable for recovery and
explicit shutdown. New CPU-manager sessions cannot be created. Older regional
volumes and pods are not changed by the Global Volume integration.

Global sessions freeze their deployment profile with the saved create intent.
Existing sessions frozen with the earlier 150 GB container request can still
reconcile or complete that intent; changing the packaged default does not resize
their pods. New profile files must specify 200 GB. A previously scaffolded local
profile is left intact until you update it, so inspect that file before
starting a new session.

