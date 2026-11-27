# Lipsync dubbing service

A 2-endpoint service wrapping the LatentSync dubbing pipeline. **Rust** is the API
gateway; **Python** workers do all the ML. Outputs are **never stored** — rendered,
delivered, then deleted. Only the *prepared input* for a selection is cached briefly
(TTL) so a user can test many languages on one selection without re-separating.

```
            ┌─────────────┐   enqueue    ┌──────────────┐
 client ──▶ │  Rust API   │ ───────────▶ │  queue +     │
            │  (axum)     │ ◀─────────── │  job state   │   (Redis, in-memory; Phase 1)
            └─────┬───────┘   status     └──────┬───────┘
                  │ streams clip, deletes it     │ pull
                  ▼                              ▼
            temp delivery                 Python workers
                                          analyze / synthesize
                                          (service_pipeline.py →
                                           LatentSync-trial pipeline, 60db, GPU)
```

## Layout

```
service/
  api-rust/        Rust axum API (the two endpoints + files + auth)   ← Phase 2
  worker-py/       Python worker CLI/loop that calls the pipeline     ← Phase 0/1
LatentSync-trial/
  service_pipeline.py   analyze() / synthesize() over the existing pipeline  ← Phase 0
```

## The two operations

- **Analyze** — prepare a selection once: separate speech → STT → diarize. Returns a
  `selection_id` handle + per-section speaker turns + language. Cached under a TTL.
- **Synthesize** — render ONE selected section in a target language/settings. Returns
  the clip; the pipeline keeps nothing. Call it repeatedly on one selection to test
  languages, then per section with the chosen config.

## Phase 0 — try it now (no Rust, no queue)

From a shell with the LatentSync-trial venv:

```bash
cd service/worker-py

# 1) Analyze a video -> prints the manifest (note the selection_id + section ids)
python servicectl.py analyze ../../LatentSync-trial/trial_inputs/sample_ref.mp4

# 2) Synthesize section 1 in Hindi -> writes ./out.mp4, keeps nothing in the cache
python servicectl.py synthesize <selection_id> 1 hi --out ./out.mp4

# optional: pin a voice per speaker, change settings
python servicectl.py synthesize <selection_id> 1 es \
    --voices "Speaker 1=<60db_voice_id>" --steps 25 --out ./es.mp4

# housekeeping
python servicectl.py release <selection_id>
python servicectl.py purge --ttl 1800
```

Selections are cached under `LatentSync-trial/local_jobs/selections/` by default
(override with `--work-root` or `SERVICE_WORK_ROOT`). `purge` removes expired ones.

## Phase roadmap

- **Phase 0 (done):** `service_pipeline.py` + `servicectl.py` — analyze/synthesize,
  ephemeral outputs, TTL cache, typed errors.
- **Phase 1:** Redis (in-memory) queue + a Python worker loop (pull → run → report).
- **Phase 2:** the Rust axum API (`api-rust/`) — the four routes + API-key auth,
  enqueue to Redis, stream + delete result files.
- **Phase 3:** end-to-end on one box. **Phase 4:** quotas, retries, metrics, TTL sweep.

## Rust API (Phase 2 skeleton)

`api-rust/` runs today as a skeleton (routes return 501):

```bash
cd service/api-rust
cargo run         # listens on 127.0.0.1:8080 (override with BIND_ADDR)
```
