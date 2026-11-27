//! Rust API gateway for the LatentSync dubbing service.
//!
//! Responsibilities (and ONLY these): HTTP, validation, run jobs, report status,
//! stream result files and delete them on delivery. All ML is done by the Python
//! worker (`service/worker-py/servicectl.py`), which this server spawns as a
//! subprocess. The API never imports Python, never calls 60db, never touches the GPU.
//!
//! Single-box model: an in-memory job registry + a 1-permit GPU semaphore (synthesize
//! is serialized; analyze runs concurrently). Nothing rendered is retained — the clip
//! is streamed once from /v1/files/{job_id} and then deleted. To scale out later, swap
//! the in-memory registry for Redis; the handlers stay the same.
//!
//! Docs:  http://127.0.0.1:8080/docs        (ReDoc)
//! Spec:  http://127.0.0.1:8080/api-doc/openapi.json
//!
//! Endpoints:
//!   GET  /healthz
//!   POST /v1/analyze        multipart: file=<video> [, source_language]   -> 202 { job_id }
//!   POST /v1/synthesize     json: { selection_id, section_id, target_language, ... } -> 202 { job_id }
//!   GET  /v1/jobs/{job_id}   -> { status, progress, manifest | video_url, error }
//!   GET  /v1/files/{job_id}  -> streams the clip, then deletes it

use std::collections::HashMap;
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, Mutex};

use axum::{
    extract::{Multipart, Path, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Semaphore;
use tower_http::services::ServeDir;
use utoipa::{OpenApi, ToSchema};

/// Self-hosted API docs UI (RapiDoc). Bundled into the binary so /docs works with
/// NO internet / CDN — the earlier ReDoc page was blank when its CDN was unreachable.
const RAPIDOC_JS: &str = include_str!("../assets/rapidoc-min.js");
const DOCS_HTML: &str = r#"<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Lipsync Dubbing Service — API</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script type="module" src="/assets/rapidoc-min.js"></script>
</head>
<body>
  <rapi-doc
    spec-url="/api-doc/openapi.json"
    render-style="read"
    theme="light"
    show-header="false"
    allow-try="true"
    regular-font="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif">
  </rapi-doc>
</body>
</html>"#;

/// Self-hosted Swagger UI (the familiar one, with Try-it-out). Also bundled so it
/// needs no internet.
const SWAGGER_JS: &str = include_str!("../assets/swagger-ui-bundle.js");
const SWAGGER_CSS: &str = include_str!("../assets/swagger-ui.css");
const SWAGGER_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Lipsync Dubbing Service — API</title>
  <link rel="stylesheet" href="/assets/swagger-ui.css"/>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="/assets/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({
      url: "/api-doc/openapi.json",
      dom_id: "#swagger-ui",
      deepLinking: true,
      presets: [SwaggerUIBundle.presets.apis]
    });
  </script>
</body>
</html>"##;

// ============================================================ OpenAPI schemas

#[derive(Serialize, Deserialize, ToSchema)]
struct Turn {
    #[schema(example = 0.57)] start: f64,
    #[schema(example = 2.44)] end: f64,
    #[schema(example = "Speaker 1")] speaker: String,
    #[schema(example = "Are you a broke beauty like me?")] text: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
struct Section {
    #[schema(example = 1)] id: u32,
    #[schema(example = 0.0)] start: f64,
    #[schema(example = 300.0)] end: f64,
    #[schema(example = 7500)] frames: u32,
    #[schema(example = json!(["Speaker 1", "Speaker 2"]))] speakers: Vec<String>,
    turns: Vec<Turn>,
}

#[derive(Serialize, Deserialize, ToSchema)]
struct Manifest {
    #[schema(example = "b554135d1ebd4c2c924fa5d32703ea0c")] selection_id: String,
    #[schema(example = "en")] source_language: Option<String>,
    #[schema(example = 25)] fps: u32,
    #[schema(example = 300.0)] max_section_seconds: f64,
    sections: Vec<Section>,
}

/// All settings are optional; omitted ones use vetted defaults. Render settings apply
/// to synthesize; chunking settings apply to analyze.
#[derive(Serialize, Deserialize, ToSchema, Default)]
#[serde(default)]
struct Settings {
    // ---- render settings (synthesize) ----
    /// Inference steps. 12 = quick preview, 20-30 = quality.
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 20)]
    steps: Option<u32>,
    /// Frames per batch: 4, 8 or 16 (8 default).
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 8)]
    frames: Option<u32>,
    /// Audio-guidance strength 1.0–2.5 (how hard the mouth follows the voice).
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 1.5)]
    guidance_scale: Option<f64>,
    /// Mouth stability / anti-flutter 0.2–0.6.
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 0.4)]
    crop_smoothing: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 1247)]
    seed: Option<u64>,
    /// TTS speaking speed 0.5–2.0.
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 1.0)]
    speed: Option<f64>,
    /// Keep the original background music/noise under the dub.
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = true)]
    restore_background: Option<bool>,
    /// Background volume 0.0–1.0.
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 1.0)]
    background_gain: Option<f64>,
    /// GFPGAN face restoration on the generated mouth.
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = false)]
    face_restore: Option<bool>,
    /// Face-restoration strength 0.0–1.0.
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 0.5)]
    restore_strength: Option<f64>,
    /// LatentSync version: "1.5" (256px, fast) or "1.6" (512px, sharper, needs big VRAM).
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = "1.5")]
    engine_version: Option<String>,
    /// GPU memory mode: "auto" | "resident" | "swap".
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = "auto")]
    offload: Option<String>,
    /// Target render clip length (seconds) within the section.
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 17.0)]
    clip_target_seconds: Option<f64>,
    /// Render clip overlap per side (seconds).
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 0.4)]
    overlap_seconds: Option<f64>,
    // ---- chunking settings (analyze) ----
    /// Max section length in seconds (chunk size), 10–600.
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 300.0)]
    section_seconds: Option<f64>,
    /// Minimum silence (seconds) that counts as a pause, 0.1–2.0.
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 0.3)]
    vad_pause_seconds: Option<f64>,
    /// Separate speech from background before transcribing (best for noisy clips).
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = true)]
    clean_speech: Option<bool>,
    /// Pause search per side when cutting sections (seconds), 0–30.
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 12.0)]
    boundary_search_seconds: Option<f64>,
    /// PyTorch CPU threads 1–16.
    #[serde(skip_serializing_if = "Option::is_none")] #[schema(example = 4)]
    cpu_threads: Option<u32>,
}

#[derive(Serialize, Deserialize, ToSchema)]
struct SynthesizeRequest {
    #[schema(example = "b554135d1ebd4c2c924fa5d32703ea0c")] selection_id: String,
    #[schema(example = 1)] section_id: u32,
    #[schema(example = "hi")] target_language: String,
    #[schema(example = json!({"Speaker 1": "f05d13fe-d3b0-400d-91a8-5422c7dda0e5"}))]
    speaker_voices: Option<HashMap<String, String>>,
    settings: Option<Settings>,
}

#[derive(Serialize, Deserialize, ToSchema)]
struct JobAccepted {
    #[schema(example = "7f3c1a9b2d4e4f6a8b0c1d2e3f405162")] job_id: String,
    #[schema(example = "queued")] status: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
struct JobStatus {
    /// queued | running | done | failed
    #[schema(example = "running")] status: String,
    #[schema(example = 0.5)] progress: f64,
    manifest: Option<Manifest>,
    #[schema(example = "/v1/files/7f3c1a9b")] video_url: Option<String>,
    error: Option<String>,
}

#[derive(Serialize, Deserialize, ToSchema)]
struct ErrorResponse {
    #[schema(example = "InvalidInput")] error: String,
    #[schema(example = 400)] status: u16,
    #[schema(example = "target_language is required.")] message: String,
}

// ============================================================ app state

#[derive(Clone)]
struct AppConfig {
    python: PathBuf,
    servicectl: PathBuf,
    work_root: PathBuf,
    tmp_dir: PathBuf,
    web_dir: PathBuf,
}

struct JobRec {
    kind: &'static str, // "analyze" | "synthesize"
    status: String,     // queued | running | done | failed
    progress: f64,
    manifest: Option<Value>,
    video_path: Option<PathBuf>,
    error: Option<String>,
}

struct AppState {
    cfg: AppConfig,
    jobs: Mutex<HashMap<String, JobRec>>,
    gpu: Semaphore, // serialize synthesize (GPU) jobs
}

type Shared = Arc<AppState>;

impl AppState {
    fn set<F: FnOnce(&mut JobRec)>(&self, id: &str, f: F) {
        if let Some(j) = self.jobs.lock().unwrap().get_mut(id) {
            f(j);
        }
    }
}

// ============================================================ handlers

#[utoipa::path(get, path = "/healthz", tag = "service",
    responses((status = 200, description = "Service is alive")))]
async fn healthz() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// The self-hosted API docs page (RapiDoc).
async fn docs_page() -> Html<&'static str> {
    Html(DOCS_HTML)
}

/// The bundled RapiDoc script (served locally so the docs need no internet).
async fn rapidoc_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/javascript; charset=utf-8")], RAPIDOC_JS)
}

/// Swagger UI page (default at /docs).
async fn swagger_page() -> Html<&'static str> {
    Html(SWAGGER_HTML)
}

async fn swagger_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/javascript; charset=utf-8")], SWAGGER_JS)
}

async fn swagger_css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], SWAGGER_CSS)
}

/// Prepare a selection: separate speech, transcribe, diarize. Upload the video as
/// multipart/form-data under `file` (optional `source_language`). Returns a job id;
/// poll it, and the done job's `manifest` holds the sections + turns.
#[utoipa::path(post, path = "/v1/analyze", tag = "service",
    request_body(content_type = "multipart/form-data", description = "Fields: file=<video> (required), source_language=<optional>, settings=<optional JSON object of chunking settings: section_seconds, vad_pause_seconds, clean_speech, boundary_search_seconds, cpu_threads>"),
    responses(
        (status = 202, description = "Job enqueued", body = JobAccepted),
        (status = 400, description = "Invalid input", body = ErrorResponse),
    ))]
async fn analyze(State(st): State<Shared>, mut form: Multipart) -> Response {
    let mut upload: Option<PathBuf> = None;
    let mut source_language: Option<String> = None;
    let mut settings_json: Option<String> = None;

    loop {
        let field = match form.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return bad_request(&format!("malformed multipart: {e}")),
        };
        match field.name().unwrap_or("") {
            "settings" => {
                settings_json = field.text().await.ok().filter(|s| !s.trim().is_empty());
            }
            "file" => {
                let ext = field
                    .file_name()
                    .and_then(|n| FsPath::new(n).extension().and_then(|e| e.to_str()))
                    .unwrap_or("mp4")
                    .to_string();
                let bytes = match field.bytes().await {
                    Ok(b) => b,
                    Err(e) => return bad_request(&format!("could not read upload: {e}")),
                };
                let path = st.cfg.tmp_dir.join(format!("up_{}.{}", uuid::Uuid::new_v4().simple(), ext));
                if let Err(e) = tokio::fs::write(&path, &bytes).await {
                    return server_error(&format!("could not save upload: {e}"));
                }
                upload = Some(path);
            }
            "source_language" => {
                source_language = field.text().await.ok().filter(|s| !s.trim().is_empty());
            }
            _ => {}
        }
    }

    let upload = match upload {
        Some(p) => p,
        None => return bad_request("multipart field 'file' (the video) is required."),
    };

    let job_id = new_id();
    insert_job(&st, &job_id, "analyze");
    let st2 = st.clone();
    let id2 = job_id.clone();
    tokio::spawn(async move { run_analyze(st2, id2, upload, source_language, settings_json).await });

    accepted(&job_id)
}

/// Render ONE section of a prepared selection in a target language. Returns a job id;
/// the done job's `video_url` serves the clip once, then it is deleted.
#[utoipa::path(post, path = "/v1/synthesize", tag = "service",
    request_body = SynthesizeRequest,
    responses(
        (status = 202, description = "Job enqueued", body = JobAccepted),
        (status = 400, description = "Invalid input", body = ErrorResponse),
    ))]
async fn synthesize(State(st): State<Shared>, Json(req): Json<SynthesizeRequest>) -> Response {
    if req.selection_id.trim().is_empty() || req.target_language.trim().is_empty() {
        return bad_request("selection_id and target_language are required.");
    }
    let job_id = new_id();
    insert_job(&st, &job_id, "synthesize");
    let st2 = st.clone();
    let id2 = job_id.clone();
    tokio::spawn(async move { run_synthesize(st2, id2, req).await });
    accepted(&job_id)
}

/// Poll a job's status/result.
#[utoipa::path(get, path = "/v1/jobs/{job_id}", tag = "service",
    params(("job_id" = String, Path, description = "Job id from analyze/synthesize")),
    responses(
        (status = 200, description = "Job status", body = JobStatus),
        (status = 404, description = "Unknown job", body = ErrorResponse),
    ))]
async fn job_status(State(st): State<Shared>, Path(job_id): Path<String>) -> Response {
    let jobs = st.jobs.lock().unwrap();
    let Some(j) = jobs.get(&job_id) else {
        return error_json(StatusCode::NOT_FOUND, "NotFound", "Unknown job.");
    };
    let video_url = if j.kind == "synthesize" && j.status == "done" {
        Some(format!("/v1/files/{job_id}"))
    } else {
        None
    };
    Json(json!({
        "status": j.status,
        "progress": j.progress,
        "manifest": j.manifest,
        "video_url": video_url,
        "error": j.error,
    }))
    .into_response()
}

/// Stream the rendered clip for a done synthesize job, then delete it (never retained).
#[utoipa::path(get, path = "/v1/files/{job_id}", tag = "service",
    params(("job_id" = String, Path, description = "Synthesize job id")),
    responses(
        (status = 200, description = "The MP4 clip", content_type = "video/mp4"),
        (status = 404, description = "No clip (not ready or already delivered)", body = ErrorResponse),
    ))]
async fn file_download(State(st): State<Shared>, Path(job_id): Path<String>) -> Response {
    let path = {
        let jobs = st.jobs.lock().unwrap();
        match jobs.get(&job_id).and_then(|j| j.video_path.clone()) {
            Some(p) => p,
            None => return error_json(StatusCode::NOT_FOUND, "NotFound", "No clip for this job."),
        }
    };
    match tokio::fs::read(&path).await {
        Ok(bytes) => {
            let _ = tokio::fs::remove_file(&path).await; // delete on delivery
            st.set(&job_id, |j| j.video_path = None);
            (
                [(header::CONTENT_TYPE, "video/mp4")],
                bytes,
            )
                .into_response()
        }
        Err(_) => error_json(StatusCode::NOT_FOUND, "NotFound", "Clip already delivered or missing."),
    }
}

/// List the supported target languages (names; ISO codes like "hi" also work).
#[utoipa::path(get, path = "/v1/languages", tag = "service",
    responses((status = 200, description = "Supported languages")))]
async fn languages(State(st): State<Shared>) -> Response {
    match worker_json(&st.cfg, &["languages"]).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => server_error(&e),
    }
}

/// List the account's 60db voices (use an `id` to pin a voice per speaker).
#[utoipa::path(get, path = "/v1/voices", tag = "service",
    responses((status = 200, description = "60db voices")))]
async fn voices(State(st): State<Shared>) -> Response {
    match worker_json(&st.cfg, &["voices"]).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => server_error(&e),
    }
}

/// The default settings the service uses (and the allowed values), so a client can
/// show them and send only the ones it wants to change on synthesize/analyze.
#[utoipa::path(get, path = "/v1/defaults", tag = "service",
    responses((status = 200, description = "Default settings + allowed values")))]
async fn defaults(State(st): State<Shared>) -> Response {
    match worker_json(&st.cfg, &["defaults"]).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => server_error(&e),
    }
}

/// Release a prepared selection and everything cached for it.
#[utoipa::path(delete, path = "/v1/selections/{selection_id}", tag = "service",
    params(("selection_id" = String, Path, description = "Selection handle from analyze")),
    responses((status = 200, description = "Released")))]
async fn release_selection(State(st): State<Shared>, Path(sel): Path<String>) -> Response {
    let mut cmd = tokio::process::Command::new(&st.cfg.python);
    cmd.arg(&st.cfg.servicectl)
        .arg("--work-root").arg(&st.cfg.work_root)
        .arg("release").arg(&sel);
    match cmd.output().await {
        Ok(o) if o.status.success() => {
            Json(json!({ "selection_id": sel, "released": true })).into_response()
        }
        Ok(o) => server_error(&worker_error(&o.stderr)),
        Err(e) => server_error(&e.to_string()),
    }
}

/// Forget a job and delete its pending clip, if any. A running render is not killed;
/// its result is simply discarded.
#[utoipa::path(delete, path = "/v1/jobs/{job_id}", tag = "service",
    params(("job_id" = String, Path, description = "Job id")),
    responses(
        (status = 200, description = "Job forgotten"),
        (status = 404, description = "Unknown job", body = ErrorResponse),
    ))]
async fn cancel_job(State(st): State<Shared>, Path(job_id): Path<String>) -> Response {
    let removed = { st.jobs.lock().unwrap().remove(&job_id) };
    match removed {
        Some(j) => {
            if let Some(p) = j.video_path {
                let _ = tokio::fs::remove_file(p).await;
            }
            Json(json!({ "job_id": job_id, "removed": true })).into_response()
        }
        None => error_json(StatusCode::NOT_FOUND, "NotFound", "Unknown job."),
    }
}

// ============================================================ workers

async fn run_analyze(st: Shared, job_id: String, upload: PathBuf, lang: Option<String>,
                     settings_json: Option<String>) {
    st.set(&job_id, |j| { j.status = "running".into(); j.progress = 0.1; });
    let result = st.cfg.tmp_dir.join(format!("{job_id}.manifest.json"));

    let mut cmd = tokio::process::Command::new(&st.cfg.python);
    cmd.arg(&st.cfg.servicectl)
        .arg("--work-root").arg(&st.cfg.work_root)
        .arg("analyze").arg(&upload)
        .arg("--json-out").arg(&result);
    if let Some(l) = lang {
        cmd.arg("--source-language").arg(l);
    }
    if let Some(s) = settings_json {
        cmd.arg("--settings-json").arg(s);
    }

    let out = cmd.output().await;
    let _ = tokio::fs::remove_file(&upload).await;

    match out {
        Ok(o) if o.status.success() => match tokio::fs::read(&result).await {
            Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
                Ok(manifest) => st.set(&job_id, |j| {
                    j.status = "done".into();
                    j.progress = 1.0;
                    j.manifest = Some(manifest);
                }),
                Err(e) => fail(&st, &job_id, &format!("bad manifest json: {e}")),
            },
            Err(e) => fail(&st, &job_id, &format!("manifest not written: {e}")),
        },
        Ok(o) => fail(&st, &job_id, &worker_error(&o.stderr)),
        Err(e) => fail(&st, &job_id, &format!("failed to run worker: {e}")),
    }
    let _ = tokio::fs::remove_file(&result).await;
}

async fn run_synthesize(st: Shared, job_id: String, req: SynthesizeRequest) {
    st.set(&job_id, |j| { j.status = "queued".into(); j.progress = 0.05; });
    let _permit = st.gpu.acquire().await.expect("semaphore closed"); // serialize GPU work
    st.set(&job_id, |j| { j.status = "running".into(); j.progress = 0.2; });

    let video = st.cfg.tmp_dir.join(format!("{job_id}.mp4"));
    let result = st.cfg.tmp_dir.join(format!("{job_id}.result.json"));

    let mut cmd = tokio::process::Command::new(&st.cfg.python);
    cmd.arg(&st.cfg.servicectl)
        .arg("--work-root").arg(&st.cfg.work_root)
        .arg("synthesize")
        .arg(&req.selection_id)
        .arg(req.section_id.to_string())
        .arg(&req.target_language)
        .arg("--out").arg(&video)
        .arg("--json-out").arg(&result);

    if let Some(voices) = &req.speaker_voices {
        if !voices.is_empty() {
            let spec = voices.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(",");
            cmd.arg("--voices").arg(spec);
        }
    }
    if let Some(s) = &req.settings {
        if let Ok(spec) = serde_json::to_string(s) {
            if spec != "{}" {
                cmd.arg("--settings-json").arg(spec);
            }
        }
    }

    let out = cmd.output().await;
    let _ = tokio::fs::remove_file(&result).await;

    match out {
        Ok(o) if o.status.success() && video.is_file() => st.set(&job_id, |j| {
            j.status = "done".into();
            j.progress = 1.0;
            j.video_path = Some(video.clone());
        }),
        Ok(o) if o.status.success() => fail(&st, &job_id, "worker produced no video."),
        Ok(o) => fail(&st, &job_id, &worker_error(&o.stderr)),
        Err(e) => fail(&st, &job_id, &format!("failed to run worker: {e}")),
    }
}

/// servicectl prints a JSON error ({error,status,message}) to stderr on failure.
/// Return its message if parseable, else the last non-empty stderr line.
fn worker_error(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    for line in text.lines().rev() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            if let Some(msg) = v.get("message").and_then(|m| m.as_str()) {
                return msg.to_string();
            }
        }
        return line.to_string();
    }
    "worker failed with no output".into()
}

/// Run a read-only worker subcommand (languages/voices) and return its JSON.
async fn worker_json(cfg: &AppConfig, subcmd: &[&str]) -> Result<Value, String> {
    let tmp = cfg.tmp_dir.join(format!("q_{}.json", uuid::Uuid::new_v4().simple()));
    let mut cmd = tokio::process::Command::new(&cfg.python);
    cmd.arg(&cfg.servicectl).arg("--work-root").arg(&cfg.work_root);
    for a in subcmd {
        cmd.arg(a);
    }
    cmd.arg("--json-out").arg(&tmp);
    let out = cmd.output().await.map_err(|e| e.to_string())?;
    let result = if out.status.success() {
        match tokio::fs::read(&tmp).await {
            Ok(b) => serde_json::from_slice::<Value>(&b).map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        }
    } else {
        Err(worker_error(&out.stderr))
    };
    let _ = tokio::fs::remove_file(&tmp).await;
    result
}

// ============================================================ small helpers

fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn insert_job(st: &AppState, id: &str, kind: &'static str) {
    st.jobs.lock().unwrap().insert(
        id.to_string(),
        JobRec { kind, status: "queued".into(), progress: 0.0, manifest: None, video_path: None, error: None },
    );
}

fn fail(st: &AppState, id: &str, msg: &str) {
    tracing::warn!("job {id} failed: {msg}");
    st.set(id, |j| { j.status = "failed".into(); j.error = Some(msg.to_string()); });
}

fn accepted(job_id: &str) -> Response {
    (StatusCode::ACCEPTED, Json(json!({ "job_id": job_id, "status": "queued" }))).into_response()
}

fn error_json(code: StatusCode, error: &str, message: &str) -> Response {
    (code, Json(json!({ "error": error, "status": code.as_u16(), "message": message }))).into_response()
}

fn bad_request(message: &str) -> Response {
    error_json(StatusCode::BAD_REQUEST, "InvalidInput", message)
}

fn server_error(message: &str) -> Response {
    error_json(StatusCode::INTERNAL_SERVER_ERROR, "ServerError", message)
}

// ============================================================ openapi + main

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Lipsync Dubbing Service",
        version = "0.1.0",
        description = "Analyze a video into diarized sections, then synthesize one \
                       section in a target language. Outputs are never stored."
    ),
    paths(healthz, analyze, synthesize, job_status, file_download,
          languages, voices, defaults, release_selection, cancel_job),
    components(schemas(
        Turn, Section, Manifest, Settings, SynthesizeRequest,
        JobAccepted, JobStatus, ErrorResponse
    )),
    tags((name = "service", description = "Analyze & synthesize operations"))
)]
struct ApiDoc;

fn resolve_config() -> AppConfig {
    // Defaults assume the server runs from service/api-rust (cargo run) and the repo
    // layout is intact. Override any of these with env vars.
    let here = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let pipeline = env_path("PIPELINE_DIR", here.join("../../LatentSync-trial"));
    let python = env_path("PYTHON", pipeline.join(".venv/Scripts/python.exe"));
    let servicectl = env_path("SERVICECTL", here.join("../worker-py/servicectl.py"));
    let work_root = env_path("WORK_ROOT", pipeline.join("local_jobs/selections"));
    let tmp_dir = env_path("SERVICE_TMP", pipeline.join("local_jobs/service_tmp"));
    let web_dir = env_path("WEB_DIR", here.join("../web"));

    for (label, path, must_exist) in [
        ("PYTHON", &python, true),
        ("SERVICECTL", &servicectl, true),
    ] {
        if must_exist && !path.exists() {
            panic!("{label} not found at {} — set the {label} env var.", path.display());
        }
    }
    std::fs::create_dir_all(&work_root).ok();
    std::fs::create_dir_all(&tmp_dir).ok();

    AppConfig {
        python: clean_path(python),
        servicectl: clean_path(servicectl),
        work_root: clean_path(work_root),
        tmp_dir: clean_path(tmp_dir),
        web_dir: clean_path(web_dir),
    }
}

/// Absolute, `..`-resolved path WITHOUT the Windows `\\?\` extended-length prefix.
/// That prefix (added by `canonicalize`) propagates through the Python subprocess and
/// breaks ffmpeg / the render engine, so we strip it.
fn clean_path(path: PathBuf) -> PathBuf {
    let resolved = path.canonicalize().unwrap_or(path);
    let s = resolved.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => PathBuf::from(rest),
        _ => resolved,
    }
}

fn env_path(key: &str, default: PathBuf) -> PathBuf {
    std::env::var(key).map(PathBuf::from).unwrap_or(default)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cfg = resolve_config();
    tracing::info!("python     = {}", cfg.python.display());
    tracing::info!("servicectl = {}", cfg.servicectl.display());
    tracing::info!("work_root  = {}", cfg.work_root.display());
    tracing::info!("web_dir    = {}", cfg.web_dir.display());
    let web_dir = cfg.web_dir.clone();

    let state: Shared = Arc::new(AppState {
        cfg,
        jobs: Mutex::new(HashMap::new()),
        gpu: Semaphore::new(1), // one synthesize (GPU) job at a time
    });

    let app = Router::new()
        .route("/docs", get(swagger_page))            // Swagger UI (default)
        .route("/rapidoc", get(docs_page))            // RapiDoc alternative
        .route("/assets/swagger-ui-bundle.js", get(swagger_js))
        .route("/assets/swagger-ui.css", get(swagger_css))
        .route("/assets/rapidoc-min.js", get(rapidoc_js))
        .route("/api-doc/openapi.json", get(|| async { Json(ApiDoc::openapi()) }))
        .route("/healthz", get(healthz))
        .route("/v1/analyze", post(analyze))
        .route("/v1/synthesize", post(synthesize))
        .route("/v1/jobs/:job_id", get(job_status).delete(cancel_job))
        .route("/v1/files/:job_id", get(file_download))
        .route("/v1/languages", get(languages))
        .route("/v1/voices", get(voices))
        .route("/v1/defaults", get(defaults))
        .route("/v1/selections/:selection_id", axum::routing::delete(release_selection))
        .with_state(state)
        // The web UI (served from disk so it can be tuned without a rebuild). Any path
        // not matched above falls through to the static files, so "/" serves index.html.
        .fallback_service(ServeDir::new(&web_dir));

    let addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("failed to bind address");
    tracing::info!("lipsync-service-api on http://{addr}  (docs at /docs)");
    axum::serve(listener, app).await.expect("server error");
}
