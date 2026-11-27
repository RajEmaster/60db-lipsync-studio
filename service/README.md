# Lipsync dubbing service

A REST service wrapping the LatentSync dubbing pipeline, with a web UI and interactive
API docs. **Rust** is the API gateway; **Python** workers do all the ML. Outputs are
**never stored** — rendered, delivered once, then deleted. The API is **discrete per
chunk**: analyze a video into chunks, synthesize one chunk per call; batching the whole
video is the integrating app's job.

```
 client / web UI ──HTTP──▶ Rust API (api-rust) ──spawns──▶ servicectl.py (worker-py)
                                                              └─▶ service_pipeline.py
                                                                   (LatentSync, 60db, GPU)
```

## Layout

```
service/
  api-rust/    the Rust API (axum) — endpoints, web UI + docs hosting.  See api-rust/README.md
  worker-py/   servicectl.py — the worker CLI the API runs, + test.ps1
  web/         the static web UI served at /
LatentSync-trial/
  service_pipeline.py   analyze() / synthesize() over the existing pipeline
```

## Run it

```bash
cd service/api-rust
cargo run
```
Open **http://127.0.0.1:8080/** (web UI) or **/docs** (Swagger UI).

**Full API reference, settings, env vars, examples and troubleshooting:
[`api-rust/README.md`](api-rust/README.md).**

## Endpoints (summary)

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/analyze` | upload a video → prepare a selection (chunks + transcript) |
| POST | `/v1/synthesize` | render one chunk in a target language/settings |
| GET | `/v1/jobs/{id}` | poll a job |
| DELETE | `/v1/jobs/{id}` | forget a job |
| GET | `/v1/files/{id}` | download the clip (deleted on delivery) |
| GET | `/v1/languages` | supported languages |
| GET | `/v1/voices` | 60db voices (for per-speaker pinning) |
| GET | `/v1/defaults` | default settings + allowed values |
| DELETE | `/v1/selections/{id}` | release a prepared selection |
| GET | `/docs`, `/rapidoc`, `/api-doc/openapi.json` | interactive docs + spec |

## Worker CLI (no API) — quick local testing

```bash
cd service/worker-py
python servicectl.py analyze ../../LatentSync-trial/trial_inputs/sample_ref.mp4
python servicectl.py synthesize <selection_id> 1 hi --steps 12 --out ./out.mp4
# or the one-command demo:
./test.ps1 -Languages hi,es
```
Commands: `analyze`, `synthesize`, `release`, `purge`, `languages`, `voices`, `defaults`.
