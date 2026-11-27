//! Thin Rust API gateway for the LatentSync dubbing service.
//!
//! Responsibilities (and ONLY these): HTTP, auth, validation, enqueue jobs to the
//! queue, report status, stream result files, and delete them on delivery. It never
//! imports Python, never calls 60db, and never touches the GPU — the Python worker
//! pool does all compute. The queue is the only contract between them.
//!
//! This file is the Phase-2 skeleton: the routes and shapes are in place and the
//! server runs; the handlers return 501 until the Redis queue + worker are wired.
//!
//! Endpoints:
//!   GET  /healthz                     liveness
//!   POST /v1/analyze                  -> 202 { job_id }        (prepare a selection)
//!   POST /v1/synthesize               -> 202 { job_id }        (render one section)
//!   GET  /v1/jobs/:job_id             -> { status, result }    (poll)
//!   GET  /v1/files/:job_id            -> streams the clip, then deletes it

use axum::{
    extract::Path,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/analyze", post(analyze))
        .route("/v1/synthesize", post(synthesize))
        .route("/v1/jobs/:job_id", get(job_status))
        .route("/v1/files/:job_id", get(file_download));

    let addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind address");
    tracing::info!("lipsync-service-api listening on {addr}");
    axum::serve(listener, app)
        .await
        .expect("server error");
}

async fn healthz() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// Phase 2: validate, store the upload to a temp path, enqueue an `analyze` job to
/// Redis, return 202 { job_id }. A Python worker consumes it and writes the manifest.
async fn analyze() -> impl IntoResponse {
    not_implemented("POST /v1/analyze")
}

/// Phase 2: validate { selection_id, section_id, target_language, settings }, enqueue
/// a `synthesize` job (serialized per GPU), return 202 { job_id }.
async fn synthesize() -> impl IntoResponse {
    not_implemented("POST /v1/synthesize")
}

/// Phase 2: read the job's status/result from the queue store.
async fn job_status(Path(job_id): Path<String>) -> impl IntoResponse {
    not_implemented(&format!("GET /v1/jobs/{job_id}"))
}

/// Phase 2: stream the rendered clip, then delete it (outputs are never retained).
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
