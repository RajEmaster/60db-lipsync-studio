# Lipsync Dubbing Service — Rust API

A small, fast HTTP gateway (Rust + [axum]) over the LatentSync dubbing pipeline. It
turns the pipeline into a clean REST API, serves a web UI and interactive API docs,
and **never stores any output** — rendered clips are delivered once and deleted.

- **Rust does only I/O**: HTTP, validation, job tracking, running the worker, streaming
  results. It never imports Python, never calls 60db, never touches the GPU.
- **Python does all the ML**: the API spawns the worker CLI
  (`../worker-py/servicectl.py`), which drives `LatentSync-trial/service_pipeline.py`.
- **Discrete per-chunk design**: analyze splits a video into chunks (sections);
  synthesize renders **one** chunk in one target language per call. Batching the whole
  video (looping over chunks and stitching) is the integrating app's job — there is no
  "do everything" endpoint by design.

---

## Table of contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Build & run](#build--run)
- [Configuration (env vars)](#configuration-env-vars)
- [The two-step flow](#the-two-step-flow)
- [Endpoints](#endpoints)
  - [GET /healthz](#get-healthz)
  - [GET /v1/defaults](#get-v1defaults)
  - [GET /v1/languages](#get-v1languages)
  - [GET /v1/voices](#get-v1voices)
  - [POST /v1/analyze](#post-v1analyze)
  - [POST /v1/synthesize](#post-v1synthesize)
  - [GET /v1/jobs/{job_id}](#get-v1jobsjob_id)
  - [DELETE /v1/jobs/{job_id}](#delete-v1jobsjob_id)
  - [GET /v1/files/{job_id}](#get-v1filesjob_id)
  - [DELETE /v1/selections/{selection_id}](#delete-v1selectionsselection_id)
  - [Docs & web UI](#docs--web-ui)
- [Settings reference](#settings-reference)
- [Data types](#data-types)
- [Errors](#errors)
- [Behavior & guarantees](#behavior--guarantees)
- [End-to-end example (curl)](#end-to-end-example-curl)
- [Integration notes](#integration-notes)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

---

## Architecture

```
                      HTTP
  client / web UI  ───────────▶  Rust API (axum, this crate)
                                  • validate, create job (in-memory registry)
                                  • GPU jobs serialized by a 1-permit semaphore
                                  • spawn the worker as a subprocess
                                        │
                                        ▼
                           servicectl.py  (../worker-py)
                                        │ imports
                                        ▼
                       service_pipeline.py  (LatentSync-trial)
                        clean-speech · STT · diarize · translate ·
                        voice clone · TTS · LatentSync render · mix
                                        │
                        60db API + GPU (LatentSync) + Demucs
```

- **Jobs are async.** `POST` returns `202 { job_id }` immediately; poll
  `GET /v1/jobs/{job_id}` until `status` is `done` or `failed`.
- **State is in-memory.** The job registry lives in the process; there is no database.
  Restarting the server forgets in-flight jobs (prepared selections on disk survive —
  see [Behavior](#behavior--guarantees)). To scale to multiple machines, replace the
  registry with a shared queue; the handlers don't change.
- **GPU is serialized.** A `tokio::sync::Semaphore` with **1 permit** means only one
  `synthesize` (GPU) render runs at a time; `analyze` jobs run concurrently.

---

## Prerequisites

1. **Rust** (stable; built with 1.96). `cargo` on `PATH`.
2. **The Python pipeline** set up in `../../LatentSync-trial`:
   - its virtualenv at `.venv` with all dependencies,
   - the LatentSync checkpoints downloaded,
   - a `.env` holding `SIXTYDB_API_KEY` (and optional `SIXTYDB_SDK_BASE_URL`,
     `SIXTYDB_WORKSPACE_ID`). **Never commit this file.**
3. A CUDA GPU for rendering (the API still starts without one, but `synthesize` will
   fail at render time).

The API process itself needs none of 60db/CUDA — it only shells out to the worker.

---

## Build & run

From this directory (`service/api-rust`):

```bash
cargo run                 # debug build, listens on 127.0.0.1:8080
# or
cargo build --release
./target/release/lipsync-service-api
```

On start it logs the resolved paths and the bind address:

```
python     = …/LatentSync-trial/.venv/Scripts/python.exe
servicectl = …/service/worker-py/servicectl.py
work_root  = …/LatentSync-trial/local_jobs/selections
web_dir    = …/service/web
lipsync-service-api on http://127.0.0.1:8080  (docs at /docs)
```

Then open **http://127.0.0.1:8080/** (web UI) or **/docs** (Swagger UI).

> Run from this folder so the default relative paths resolve, or set the env vars below.

---

## Configuration (env vars)

All optional; sensible defaults assume the repo layout is intact and the server runs
from `service/api-rust`.

| Env var | Default | Purpose |
|---|---|---|
| `BIND_ADDR` | `127.0.0.1:8080` | Address/port to listen on. Use `0.0.0.0:PORT` to expose. |
| `PIPELINE_DIR` | `../../LatentSync-trial` | The LatentSync pipeline directory. |
| `PYTHON` | `<PIPELINE_DIR>/.venv/Scripts/python.exe` | Python interpreter for the worker. |
| `SERVICECTL` | `../worker-py/servicectl.py` | The worker CLI the API runs. |
| `WORK_ROOT` | `<PIPELINE_DIR>/local_jobs/selections` | Where prepared selections are cached. **Must live under `PIPELINE_DIR`** (the render addresses sections relative to it). |
| `SERVICE_TMP` | `<PIPELINE_DIR>/local_jobs/service_tmp` | Scratch dir for uploads, result JSON and delivery clips. |
| `WEB_DIR` | `../web` | Static web UI served at `/` (served from disk, so it can be tuned without rebuilding). |

Paths are canonicalized at startup with the Windows `\\?\` extended-length prefix
stripped (that prefix otherwise breaks the render subprocess).

---

## The two-step flow

1. **Analyze** a video → get a `selection_id` and its **chunks** (sections), each with
   speaker turns and transcript. The prepared selection (clean-speech stem, turns,
   clones) is cached so repeated renders are cheap.
2. **Synthesize** one chunk → render it in a target language with whatever settings you
   choose → poll → download the clip (deleted on delivery).

Test a chunk across several languages/settings cheaply (the selection is reused), then
have your app call `synthesize` for each chunk with the chosen config and stitch them.

---

## Endpoints

Base URL below assumed to be `http://127.0.0.1:8080`. All bodies are JSON unless noted.

### GET /healthz
Liveness.
```json
{ "status": "ok" }
```

### GET /v1/defaults
The default settings the service uses, plus allowed values for the enumerated ones.
Use it to show the real defaults and send only the settings you want to change.
```json
{
  "defaults": {
    "steps": 20, "frames": 8, "guidance_scale": 1.5, "crop_smoothing": 0.4,
    "seed": 1247, "speed": 1.0, "clean_speech": true, "restore_background": true,
    "background_gain": 1.0, "face_restore": false, "restore_strength": 0.5,
    "engine_version": "1.5", "offload": "auto", "max_gpus": 0, "cpu_threads": 4,
    "refine_margin": 0.3, "section_seconds": 300, "boundary_search_seconds": 12,
    "vad_pause_seconds": 0.3, "clip_min_seconds": 10, "clip_target_seconds": 17,
    "clip_max_seconds": 25, "overlap_seconds": 0.4
  },
  "allowed": {
    "frames": [4, 8, 16],
    "engine_version": ["1.5", "1.6"],
    "offload": ["auto", "resident", "swap"]
  }
}
```

### GET /v1/languages
Supported target languages (names). ISO codes like `hi` also work.
```json
{ "languages": ["Hindi", "English", "Bengali", "…"], "note": "Names or ISO codes (e.g. \"hi\") both work." }
```

### GET /v1/voices
The account's 60db voices, for pinning a voice per speaker. (Calls 60db.)
```json
{ "voices": [{ "name": "Aarav (voice_e6feb2685df3)", "id": "voice_e6feb2685df3" }], "count": 98 }
```

### POST /v1/analyze
Prepare a selection. **`multipart/form-data`**:

| field | required | description |
|---|---|---|
| `file` | yes | the video file |
| `source_language` | no | ISO code, or `auto` (default) |
| `settings` | no | JSON object of **chunking** settings (see [Settings](#settings-reference)) |

```bash
curl -F "file=@clip.mp4" \
     -F 'settings={"section_seconds":300,"clean_speech":true}' \
     http://127.0.0.1:8080/v1/analyze
```
→ `202`
```json
{ "job_id": "7f3c1a9b…", "status": "queued" }
```
Poll the job; when `done`, its `manifest` holds the selection + chunks.

### POST /v1/synthesize
Render one chunk. **`application/json`**:

| field | required | description |
|---|---|---|
| `selection_id` | yes | from an analyze manifest |
| `section_id` | yes | which chunk (section `id` from the manifest) |
| `target_language` | yes | language name or ISO code |
| `speaker_voices` | no | `{ "Speaker 1": "<60db voice id>" }` — pin voices; omitted = auto-clone |
| `settings` | no | JSON object of **render** settings (see [Settings](#settings-reference)) |

```bash
curl -X POST http://127.0.0.1:8080/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"selection_id":"7f3c1a9b…","section_id":1,"target_language":"hi",
       "settings":{"steps":20,"frames":8,"engine_version":"1.5"}}'
```
→ `202 { "job_id": "…", "status": "queued" }`. Poll the job; when `done`, its
`video_url` serves the clip.

### GET /v1/jobs/{job_id}
Poll any job.
```json
{
  "status": "running",            // queued | running | done | failed
  "progress": 0.6,
  "manifest": { … } | null,       // set when a DONE analyze job
  "video_url": "/v1/files/…" | null, // set when a DONE synthesize job
  "error": "…" | null             // set when failed
}
```
`404` if the job id is unknown.

### DELETE /v1/jobs/{job_id}
Forget a job and delete its pending clip, if any.
```json
{ "job_id": "…", "removed": true }
```
`404` if unknown. Note: a render already running is **not** killed; its result is just
discarded.

### GET /v1/files/{job_id}
Stream the rendered clip for a **done** synthesize job. `Content-Type: video/mp4`.
**The clip is deleted on delivery** — a second request returns `404`. Fetch it once
(e.g. into a Blob) and keep your own copy.

### DELETE /v1/selections/{selection_id}
Release a prepared selection and everything cached for it.
```json
{ "selection_id": "…", "released": true }
```

### Docs & web UI
| Path | Served |
|---|---|
| `/` (and any unmatched path) | the web UI from `WEB_DIR` (static) |
| `/docs` | Swagger UI (self-hosted, no CDN) |
| `/rapidoc` | RapiDoc (self-hosted) |
| `/api-doc/openapi.json` | the OpenAPI 3.1 spec |

---

## Settings reference

All settings are optional; omit a setting to use its default (`GET /v1/defaults`).
Unknown keys are rejected. Pass them as the `settings` JSON object on the relevant call.

### Render settings (POST /v1/synthesize)

| key | type | range / values | default | meaning |
|---|---|---|---|---|
| `steps` | int | 2–50 | 20 | inference steps (12 = quick, 20–30 = quality) |
| `frames` | int | 4, 8, 16 | 8 | frames per batch (FPB) |
| `guidance_scale` | float | 1.0–2.5 | 1.5 | audio guidance (mouth follows voice) |
| `crop_smoothing` | float | 0.2–0.6 | 0.4 | mouth stability / anti-flutter |
| `seed` | int | ≥ 0 | 1247 | motion seed (repeatable) |
| `speed` | float | 0.5–2.0 | 1.0 | TTS speaking speed |
| `engine_version` | string | `1.5`, `1.6` | `1.5` | model: 1.5 = 256px fast, 1.6 = 512px sharper (needs big VRAM) |
| `offload` | string | `auto`, `resident`, `swap` | `auto` | GPU memory mode |
| `restore_background` | bool | — | true | keep original background music/noise under the dub |
| `background_gain` | float | 0.0–1.0 | 1.0 | background volume |
| `face_restore` | bool | — | false | GFPGAN face restoration on the generated mouth |
| `restore_strength` | float | 0.0–1.0 | 0.5 | face-restoration strength |
| `clip_target_seconds` | float | 4–25 | 17 | render clip length within the chunk |
| `overlap_seconds` | float | 0–1.5 | 0.4 | render clip overlap per side |

### Chunking settings (POST /v1/analyze)

| key | type | range | default | meaning |
|---|---|---|---|---|
| `section_seconds` | float | 10–600 | 300 | max chunk length (chunk size) |
| `vad_pause_seconds` | float | 0.1–2.0 | 0.3 | minimum silence counted as a pause |
| `clean_speech` | bool | — | true | separate speech from background before STT (best for noisy/music clips) |
| `boundary_search_seconds` | float | 0–30 | 12 | pause search per side when cutting chunks |
| `cpu_threads` | int | 1–16 | 4 | PyTorch CPU threads |

> `refine_margin`, `max_gpus`, `clip_min_seconds`, `clip_max_seconds` also exist in the
> defaults and may be passed, but are not surfaced in the web UI.

---

## Data types

**Manifest** (the `manifest` field of a done analyze job):
```json
{
  "selection_id": "b554135d1ebd4c2c924fa5d32703ea0c",
  "source_language": "auto",
  "fps": 25,
  "max_section_seconds": 300.0,
  "sections": [
    {
      "id": 1,
      "start": 0.0,
      "end": 4.0,
      "frames": 100,
      "speakers": ["Speaker 1", "Speaker 2"],
      "turns": [
        { "start": 0.57, "end": 2.44, "speaker": "Speaker 1", "text": "…" }
      ]
    }
  ],
  "created_at": 1795763729.68
}
```
Times are in seconds on the global timeline. `frames` are at `fps` (25).

---

## Errors

Errors return a JSON body:
```json
{ "error": "InvalidInput", "status": 400, "message": "target_language is required." }
```

| HTTP | `error` | When |
|---|---|---|
| 400 | `InvalidInput` | bad/missing fields, unknown or out-of-range setting, invalid upload |
| 404 | `NotFound` / `SelectionNotFound` | unknown job/selection, or a clip already delivered |
| 409 | `SelectionBusy` | a synthesize for that selection is already rendering (per-selection lock) |
| 500 | `ServerError` / `RenderError` | failed to save upload, render failed |
| 502 | `UpstreamError` | 60db returned a transient/server error |

Job-level failures surface in `GET /v1/jobs/{id}` as `status: "failed"` with `error`.

---

## Behavior & guarantees

- **No output is ever stored.** A synthesized clip is written to a temp file, streamed
  once from `/v1/files/{job_id}`, then deleted. After that the selection keeps only its
  prepared *input* (reference, clean-speech stem, turns, clones) — not the render.
- **Selections are cached, with a TTL.** The prepared input is kept so testing many
  languages on one chunk is cheap. Default TTL is **1800 s (30 min)**, enforced by the
  worker's `purge` (`servicectl purge`); the API does not auto-sweep, so schedule
  `purge` (cron) in production, or call `DELETE /v1/selections/{id}` when done.
- **One GPU render at a time.** `synthesize` is serialized by a 1-permit semaphore;
  `analyze` runs concurrently.
- **Per-selection lock.** Two synthesize calls on the *same* selection can't clobber
  each other — the second gets `409 SelectionBusy`.
- **Discrete per-chunk.** One chunk, one language, per call. No whole-video endpoint.
- **Privacy.** Uploads and results are transient; nothing is archived server-side.

---

## End-to-end example (curl)

```bash
BASE=http://127.0.0.1:8080

# 1) analyze
JOB=$(curl -s -F "file=@clip.mp4" $BASE/v1/analyze | jq -r .job_id)

# 2) poll until done, grab the selection id
until [ "$(curl -s $BASE/v1/jobs/$JOB | jq -r .status)" = "done" ]; do sleep 2; done
SEL=$(curl -s $BASE/v1/jobs/$JOB | jq -r .manifest.selection_id)

# 3) synthesize chunk 1 in Hindi at 12 steps
SJOB=$(curl -s -X POST $BASE/v1/synthesize -H "Content-Type: application/json" \
  -d "{\"selection_id\":\"$SEL\",\"section_id\":1,\"target_language\":\"hi\",\"settings\":{\"steps\":12}}" \
  | jq -r .job_id)

# 4) poll, then download the clip (deleted on delivery)
until [ "$(curl -s $BASE/v1/jobs/$SJOB | jq -r .status)" = "done" ]; do sleep 5; done
curl -s $BASE/v1/files/$SJOB -o out_hi.mp4

# 5) housekeeping
curl -s -X DELETE $BASE/v1/selections/$SEL
```

---

## Integration notes

- **Dub a whole video**: analyze once, then loop `synthesize` over every `section_id`
  from the manifest (serially — the GPU is one-at-a-time), download each clip, and
  stitch them in your app. The API intentionally does not stitch.
- **Testing a chunk** across languages/settings: reuse the same `selection_id`; only
  the render repeats, not the separation/transcription.
- **Pinning voices**: get ids from `GET /v1/voices`, pass `speaker_voices`.
- **Clients**: import `/api-doc/openapi.json` into Postman/Insomnia, or codegen a client.

---

## Troubleshooting

- **`/docs` is blank** — fixed: the docs UI is self-hosted (no CDN). Hard-refresh
  (Ctrl+F5). If still blank, check the browser console and that `/assets/swagger-ui-bundle.js`
  returns 200.
- **`address already in use` on start** — another instance holds the port. Stop it
  (find the PID on the port and kill it) or set a different `BIND_ADDR`.
- **synthesize fails immediately** — usually the pipeline: check the worker logs the API
  prints, that the venv/checkpoints exist and `.env` has `SIXTYDB_API_KEY`.
- **`SelectionNotFound` on synthesize** — the selection TTL expired or it was released;
  run analyze again.
- **Slow render** — see the pipeline tuning (resident vs swap, steps, FPB, GPU class).

---

## Project layout

```
service/
  api-rust/          this crate (the API)
    src/main.rs      all routes, handlers, worker spawning, OpenAPI
    assets/          bundled Swagger UI + RapiDoc (served locally)
    Cargo.toml
  worker-py/
    servicectl.py    worker CLI the API runs (analyze/synthesize/… )
  web/               the static web UI served at /
LatentSync-trial/
  service_pipeline.py  analyze()/synthesize() over the pipeline
```

[axum]: https://github.com/tokio-rs/axum
