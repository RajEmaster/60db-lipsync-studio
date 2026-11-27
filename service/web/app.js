// Lipsync Dubbing Studio — drives the REST API (same origin).
const $ = (id) => document.getElementById(id);
let selectionId = null;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function setStatus(el, text, kind) {
  el.className = "status" + (kind ? " " + kind : "");
  el.innerHTML = (kind === "run" ? '<span class="spin"></span>' : "") + text;
}
async function jsonOrThrow(res) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.message || body.error || `HTTP ${res.status}`);
  return body;
}
async function pollJob(jobId, onTick) {
  for (;;) {
    const job = await jsonOrThrow(await fetch(`/v1/jobs/${jobId}`));
    if (onTick) onTick(job);
    if (job.status === "done") return job;
    if (job.status === "failed") throw new Error(job.error || "job failed");
    await sleep(2000);
  }
}
const num = (id) => { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : undefined; };
const bool = (id) => $(id).checked;
const str = (id) => $(id).value;

// ---- health + defaults + languages (on load) -------------------------------
(async function boot() {
  try {
    const b = await (await fetch("/healthz")).json();
    $("health").textContent = b.status === "ok" ? "API online" : "API ?";
    $("health").className = "pill ok";
  } catch { $("health").textContent = "API offline"; $("health").className = "pill bad"; }

  // Seed every form field from the server's real defaults, so the UI shows exactly
  // what the service uses — and the user can change any of them.
  try {
    const d = (await (await fetch("/v1/defaults")).json()).defaults || {};
    for (const [key, val] of Object.entries(d)) {
      const el = $(key);
      if (!el) continue;
      if (el.type === "checkbox") el.checked = Boolean(val);
      else el.value = val;
    }
    if ($("steps")) $("stepsVal").textContent = $("steps").value;
  } catch { /* keep the HTML defaults */ }

  loadLanguages();
})();

async function loadLanguages() {
  const sel = $("langSel");
  try {
    const b = await (await fetch("/v1/languages")).json();
    sel.innerHTML = "";
    (b.languages || []).forEach((name) => {
      const o = document.createElement("option");
      o.value = name; o.textContent = name; sel.appendChild(o);
    });
  } catch { sel.innerHTML = '<option value="Hindi">Hindi</option>'; }
}

// ---- settings builders -----------------------------------------------------
function chunkingSettings() {
  return {
    section_seconds: num("section_seconds"),
    vad_pause_seconds: num("vad_pause_seconds"),
    boundary_search_seconds: num("boundary_search_seconds"),
    cpu_threads: num("cpu_threads"),
    clean_speech: bool("clean_speech"),
  };
}
function renderSettings() {
  return {
    steps: num("steps"),
    frames: parseInt($("frames").value, 10),
    guidance_scale: num("guidance_scale"),
    crop_smoothing: num("crop_smoothing"),
    speed: num("speed"),
    seed: num("seed"),
    engine_version: str("engine_version"),
    offload: str("offload"),
    clip_target_seconds: num("clip_target_seconds"),
    overlap_seconds: num("overlap_seconds"),
    restore_background: bool("restore_background"),
    background_gain: num("background_gain"),
    face_restore: bool("face_restore"),
    restore_strength: num("restore_strength"),
  };
}

// ---- step 1: analyze -------------------------------------------------------
$("analyzeBtn").addEventListener("click", async () => {
  const file = $("videoInput").files[0];
  const st = $("analyzeStatus");
  if (!file) { setStatus(st, "Choose a video first.", "err"); return; }

  $("analyzeBtn").disabled = true;
  setStatus(st, "Uploading & chunking… (separation + transcription)", "run");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const lang = $("sourceLang").value.trim();
    if (lang) fd.append("source_language", lang);
    fd.append("settings", JSON.stringify(chunkingSettings()));

    const accepted = await jsonOrThrow(await fetch("/v1/analyze", { method: "POST", body: fd }));
    const job = await pollJob(accepted.job_id, (j) => setStatus(st, `Analyzing… (${j.status})`, "run"));
    renderChunks(job.manifest);
    setStatus(st, "Done — pick a chunk below to test.", "ok");
  } catch (e) {
    setStatus(st, "Analyze failed: " + e.message, "err");
  } finally { $("analyzeBtn").disabled = false; }
});

const fmt = (t) => Number(t).toFixed(2) + "s";
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));

function renderChunks(m) {
  selectionId = m.selection_id;
  $("selInfo").innerHTML = `selection <code>${m.selection_id}</code> · lang <b>${m.source_language || "auto"}</b> · ${m.sections.length} chunk(s)`;

  const grid = $("chunks"); grid.innerHTML = "";
  const secSel = $("sectionSel"); secSel.innerHTML = "";

  m.sections.forEach((s) => {
    const card = document.createElement("div");
    card.className = "chunk"; card.dataset.id = s.id;
    let turns = s.turns.map((t) => `<div class="turn"><span class="sp">${t.speaker}</span>${esc(t.text)}</div>`).join("");
    card.innerHTML =
      `<h3>Chunk ${s.id}</h3>` +
      `<div class="chmeta">${fmt(s.start)}–${fmt(s.end)} · speakers: ${s.speakers.join(", ") || "—"}</div>` +
      (turns || '<div class="turn muted">no speech</div>') +
      `<div class="pick">▶ test this chunk</div>`;
    card.addEventListener("click", () => selectChunk(s.id));
    grid.appendChild(card);

    const o = document.createElement("option");
    o.value = s.id; o.textContent = `Chunk ${s.id} (${fmt(s.start)}–${fmt(s.end)})`;
    secSel.appendChild(o);
  });

  $("chunksCard").classList.remove("hidden");
  $("synthCard").classList.remove("hidden");
  if (m.sections.length) selectChunk(m.sections[0].id);
}

function selectChunk(id) {
  $("sectionSel").value = id;
  document.querySelectorAll(".chunk").forEach((c) => c.classList.toggle("sel", Number(c.dataset.id) === Number(id)));
}
$("sectionSel").addEventListener("change", (e) => selectChunk(Number(e.target.value)));

// ---- voices ----------------------------------------------------------------
$("steps").addEventListener("input", (e) => ($("stepsVal").textContent = e.target.value));
$("loadVoices").addEventListener("click", async () => {
  const list = $("voiceList"); list.classList.remove("hidden"); list.innerHTML = "loading…";
  try {
    const b = await (await fetch("/v1/voices")).json();
    list.innerHTML = "";
    (b.voices || []).slice(0, 200).forEach((v) => {
      const d = document.createElement("div");
      d.textContent = `${v.name}  —  ${v.id}`; d.title = "click to copy id";
      d.addEventListener("click", () => navigator.clipboard.writeText(v.id));
      list.appendChild(d);
    });
  } catch (e) { list.innerHTML = "could not load voices: " + e.message; }
});
function parseVoices(text) {
  const map = {};
  (text || "").split("\n").forEach((line) => {
    const t = line.trim(); const i = t.indexOf("=");
    if (i > 0) map[t.slice(0, i).trim()] = t.slice(i + 1).trim();
  });
  return map;
}

// ---- step 2: synthesize ----------------------------------------------------
$("synthBtn").addEventListener("click", async () => {
  const st = $("synthStatus");
  if (!selectionId) { setStatus(st, "Analyze a video first.", "err"); return; }

  const body = {
    selection_id: selectionId,
    section_id: Number($("sectionSel").value),
    target_language: $("langSel").value,
    settings: renderSettings(),
  };
  const voices = parseVoices($("voices").value);
  if (Object.keys(voices).length) body.speaker_voices = voices;

  $("synthBtn").disabled = true;
  setStatus(st, "Rendering this chunk on the GPU (can take a couple of minutes)…", "run");
  try {
    const accepted = await jsonOrThrow(await fetch("/v1/synthesize", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }));
    const job = await pollJob(accepted.job_id, (j) => setStatus(st, `Rendering… (${j.status})`, "run"));

    setStatus(st, "Fetching the clip…", "run");
    const res = await fetch(job.video_url);
    if (!res.ok) throw new Error(`could not fetch clip (HTTP ${res.status})`);
    const url = URL.createObjectURL(await res.blob());
    $("player").src = url;
    $("downloadLink").href = url;
    $("downloadLink").download = `${selectionId}_chunk${body.section_id}_${body.target_language}.mp4`;
    $("videoCard").classList.remove("hidden");
    $("videoCard").scrollIntoView({ behavior: "smooth" });
    setStatus(st, "Done. Try another language or chunk — the chunk is reused.", "ok");
  } catch (e) {
    setStatus(st, "Synthesize failed: " + e.message, "err");
  } finally { $("synthBtn").disabled = false; }
});
