# Recorded fixtures

Recordings of real runs and ComfyUI schemas, used as regression inputs. All
creative content has been anonymized: every word of the original concepts,
scripts, plans, prompts and file paths was replaced with a consistent made-up
word, so the structure tests rely on (verbatim dialogue, word counts, ordering,
recorded hashes) holds while the content means nothing. Machine paths and host
names are generic. Treat all files as read-only inputs; tests must copy them to
a temporary folder before mutating.

The workflow templates these runs used ship with the app in
`src/simplegen/data/workflows/`.

## schemas/

Trimmed `/object_info` responses: only the node definitions the code and tests
use, and file lists cut to the files the repository references.

- `cloud_global_object_info.json` — a RunPod Global Volume pod (the schema
  batches prepare against). Has the H3 BF16 models.
- `local_object_info.json` — a local ComfyUI with INT8/NVFP4 H3 models and
  **no** H3 BF16 models.

## runs/

Project folders from real runs (logs, histories, media excluded).

- `accepted_after_correction_definitions_changed/` — batch member that rendered
  successfully. Attempt 1 was rejected ("Native definitions, metadata,
  configuration, and groups must remain unchanged"); attempt 2 accepted.
- `rejected_rewired_fixed_node_14/` — local schedule; both attempts rejected
  ("Unapproved rewiring of fixed node 14.source").
- `rejected_nonreciprocal_links/` — local schedule; both attempts rejected
  ("Native node 7 output has nonreciprocal links").
- `blocked_bf16_model_on_local/` — local schedule that used the cloud BF16
  template. Agent work was accepted, then rendering was blocked:
  "Node 1.unet_name: unavailable choice 'minimax_h3_ref2va_bf16…'". This is the
  primary regression case for preflight and repair.

Reference image paths inside these runs do not exist; tests must not depend on
them existing.

## h3scene/ and refmod/

H3 Scene Chain golden graphs (plan, compiled workflow, run record, uploaded
picture names), screenplay-format scripts for the dialogue extractor, a legacy
template and state for compatibility tests, and two RefMod workflows with the
RefMod loader schema.
