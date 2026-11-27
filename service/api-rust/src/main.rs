//! Thin Rust API gateway for the LatentSync dubbing service.
//!
//! Responsibilities (and ONLY these): HTTP, auth, validation, enqueue jobs to the
//! queue, report status, stream result files, and delete them on delivery. It never
//! imports Python, never calls 60db, and never touches the GPU — the Python worker
//! pool does all compute. The queue is the only contract between them.
//!
//! This is the Phase-2 skeleton: routes and request/response schemas are in place and
//! the server runs with interactive OpenAPI docs; the handlers return 501 until the
//! Redis queue + worker are wired. The docs let the other team integrate against the
//! contract now.
//!
//! Interactive docs:  http://127.0.0.1:8080/docs   (Swagger UI)
//! Raw OpenAPI spec:  http://127.0.0.1:8080/api-doc/openapi.json
//!
//! Endpoints:
//!   GET  /healthz                     liveness
//!   POST /v1/analyze                  -> 202 { job_id }     (prepare a selection)
//!   POST /v1/synthesize               -> 202 { job_id }     (render one section)
//!   GET  /v1/jobs/{job_id}            -> { status, result } (poll)
//!   GET  /v1/files/{job_id}           -> streams the clip, then deletes it

use std::collections::HashMap;

use axum::{
    extract::Path,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use utoipa::{OpenApi, ToSchema};
use utoipa_redoc::{Redoc, Servable};

// --------------------------------------------------------------------- schemas

/// One diarized speaker turn with its recovered text (times are global seconds).
#[derive(Serialize, Deserialize, ToSchema)]
struct Turn {
    #[schema(example = 0.57)]
    start: f64,
    #[schema(example = 2.44)]
    end: f64,
    #[schema(example = "Speaker 1")]
    speaker: String,
    #[schema(example = "Are you a broke beauty like me?")]
    text: String,
}

/// A chunk of the video the model can transcribe/render as a unit.
#[derive(Serialize, Deserialize, ToSchema)]
struct Section {
    #[schema(example = 1)]
    id: u32,
    #[schema(example = 0.0)]
    start: f64,
    #[schema(example = 300.0)]
    end: f64,
    #[schema(example = 7500)]
    frames: u32,
    #[schema(example = json!(["Speaker 1", "Speaker 2"]))]
    speakers: Vec<String>,
    turns: Vec<Turn>,
}

/// The analyze result: a handle plus the diarized sections. Cached under a TTL so a
/// user can test many languages on one selection without re-separating.
#[derive(Serialize, Deserialize, ToSchema)]
struct Manifest {
    #[schema(example = "b554135d1ebd4c2c924fa5d32703ea0c")]
    selection_id: String,
    #[schema(example = "en")]
    source_language: Option<String>,
    #[schema(example = 25)]
    fps: u32,
    #[schema(example = 300.0)]
    max_section_seconds: f64,
    sections: Vec<Section>,
}

/// Body for POST /v1/analyze. Provide a public `source_url`, or upload the file as
/// multipart/form-data under the field `file`.
#[derive(Serialize, Deserialize, ToSchema)]
struct AnalyzeRequest {
    #[schema(example = "https://cdn.example.com/clip.mp4")]
    source_url: Option<String>,
    /// ISO code or "auto" (default).
    #[schema(example = "auto")]
    source_language: Option<String>,
}

/// Per-call render settings (all optional; vetted defaults are used otherwise).
#[derive(Serialize, Deserialize, ToSchema)]
struct Settings {
    /// Inference steps. 12 = quick preview, 20-30 = quality.
    #[schema(example = 20)]
    steps: Option<u32>,
    /// Frames per batch: 4, 8 or 16 (8 default).
    #[schema(example = 8)]
    frames: Option<u32>,
    #[schema(example = 1.5)]
    guidance_scale: Option<f64>,
    #[schema(example = 1247)]
    seed: Option<u64>,
    #[schema(example = 1.0)]
    speed: Option<f64>,
}

/// Body for POST /v1/synthesize. `speaker_voices` pins a 60db voice per speaker
/// (omit to auto-clone from the original).
#[derive(Serialize, Deserialize, ToSchema)]
struct SynthesizeRequest {
    #[schema(example = "b554135d1ebd4c2c924fa5d32703ea0c")]
    selection_id: String,
    #[schema(example = 1)]
    section_id: u32,
    #[schema(example = "hi")]
    target_language: String,
    #[schema(example = json!({"Speaker 1": "f05d13fe-d3b0-400d-91a8-5422c7dda0e5"}))]
    speaker_voices: Option<HashMap<String, String>>,
    settings: Option<Settings>,
}

/// 202 Accepted body: a job was enqueued; poll it at GET /v1/jobs/{job_id}.
#[derive(Serialize, Deserialize, ToSchema)]
struct JobAccepted {
    #[schema(example = "s_9c2a1f...")]
    job_id: String,
    #[schema(example = "queued")]
    status: String,
}

/// Job poll result. `manifest` is set when a done job was an analyze; `video_url`
/// when it was a synthesize. `error` is set on failure.
#[derive(Serialize, Deserialize, ToSchema)]
struct JobStatus {
    /// queued | running | done | failed
    #[schema(example = "running")]
    status: String,
    #[schema(example = 0.6)]
    progress: Option<f64>,
    manifest: Option<Manifest>,
    #[schema(example = "/v1/files/s_9c2a1f")]
    video_url: Option<String>,
    error: Option<String>,
}

/// Standard error body.
#[derive(Serialize, Deserialize, ToSchema)]
struct ErrorResponse {
    #[schema(example = "InvalidInput")]
    error: String,
    #[schema(example = 400)]
    status: u16,
    #[schema(example = "target_language is required.")]
    message: String,
}

// --------------------------------------------------------------------- handlers

#[utoipa::path(get, path = "/healthz", tag = "service",
    responses((status = 200, description = "Service is alive")))]
async fn healthz() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// Prepare a selection: separate speech, transcribe, diarize. Returns a job id;
/// poll it, and the done job's `manifest` holds the sections + turns.
#[utoipa::path(post, path = "/v1/analyze", tag = "service",
    request_body = AnalyzeRequest,
    responses(
        (status = 202, description = "Job enqueued", body = JobAccepted),
        (status = 400, description = "Invalid input", body = ErrorResponse),
    ))]
async fn analyze() -> impl IntoResponse {
    not_implemented("POST /v1/analyze")
}

/// Render ONE section of a prepared selection in a target language. Returns a job id;
/// the done job's `video_url` serves the clip once, then it is deleted.
#[utoipa::path(post, path = "/v1/synthesize", tag = "service",
    request_body = SynthesizeRequest,
    responses(
        (status = 202, description = "Job enqueued", body = JobAccepted),
        (status = 400, description = "Invalid input", body = ErrorResponse),
        (status = 404, description = "Unknown or expired selection", body = ErrorResponse),
        (status = 409, description = "Selection already rendering", body = ErrorResponse),
    ))]
async fn synthesize() -> impl IntoResponse {
    not_implemented("POST /v1/synthesize")
}

/// Poll a job's status/result.
#[utoipa::path(get, path = "/v1/jobs/{job_id}", tag = "service",
    params(("job_id" = String, Path, description = "Job id from analyze/synthesize")),
    responses(
        (status = 200, description = "Job status", body = JobStatus),
        (status = 404, description = "Unknown job", body = ErrorResponse),
    ))]
async fn job_status(Path(job_id): Path<String>) -> impl IntoResponse {
    not_implemented(&format!("GET /v1/jobs/{job_id}"))
}

/// Stream the rendered clip for a done synthesize job, then delete it (never retained).
#[utoipa::path(get, path = "/v1/files/{job_id}", tag = "service",
    params(("job_id" = String, Path, description = "Synthesize job id")),
    responses(
        (status = 200, description = "The MP4 clip", content_type = "video/mp4"),
        (status = 404, description = "No such clip (not ready or already delivered)", body = ErrorResponse),
    ))]
async fn file_download(Path(job_id): Path<String>) -> impl IntoResponse {
    not_implemented(&format!("GET /v1/files/{job_id}"))
}

fn not_implemented(endpoint: &str) -> (StatusCode, Json<Value>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": "not_implemented",
            "endpoint": endpoint,
            "message": "Phase 2: this route is wired to the Python worker queue next."
        })),
    )
}

// --------------------------------------------------------------------- openapi

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Lipsync Dubbing Service",
        version = "0.1.0",
        description = "Analyze a video into diarized sections, then synthesize one \
                       section in a target language. Outputs are never stored."
    ),
    paths(healthz, analyze, synthesize, job_status, file_download),
    components(schemas(
        Turn, Section, Manifest, AnalyzeRequest, Settings, SynthesizeRequest,
        JobAccepted, JobStatus, ErrorResponse
    )),
    tags((name = "service", description = "Analyze & synthesize operations"))
)]
struct ApiDoc;

// --------------------------------------------------------------------- main

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = Router::new()
        // Interactive docs at /docs, and the raw OpenAPI spec for Postman/clients.
        .merge(Redoc::with_url("/docs", ApiDoc::openapi()))
        .route("/api-doc/openapi.json", get(|| async { Json(ApiDoc::openapi()) }))
        .route("/healthz", get(healthz))
        .route("/v1/analyze", post(analyze))
        .route("/v1/synthesize", post(synthesize))
        .route("/v1/jobs/:job_id", get(job_status))
        .route("/v1/files/:job_id", get(file_download));

    let addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind address");
    tracing::info!("lipsync-service-api on http://{addr}  (docs at /docs)");
    axum::serve(listener, app).await.expect("server error");
}
