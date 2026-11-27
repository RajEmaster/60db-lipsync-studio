// Lipsync Dubbing Studio — drives the REST API (same origin).
const $ = (id) => document.getElementById(id);
const API = "";               // same origin as this page
let selectionId = null;

// ---- helpers ---------------------------------------------------------------
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

// Poll a job until it is done/failed. onTick(status) for progress text.
async function pollJob(jobId, onTick) {
  for (;;) {
    const res = await fetch(`${API}/v1/jobs/${jobId}`);
    const job = await jsonOrThrow(res);
    if (onTick) onTick(job);
    if (job.status === "done") return job;
    if (job.status === "failed") throw new Error(job.error || "job failed");
    await sleep(2000);
  }
}

// ---- health ----------------------------------------------------------------
(async function health() {
  try {
    const res = await fetch(`${API}/healthz`);
    const b = await res.json();
    $("health").textContent = b.status === "ok" ? "API online" : "API ?";
    $("health").className = "pill ok";
  } catch {
    $("health").textContent = "API offline";
    $("health").className = "pill bad";
  }
})();

// ---- languages -------------------------------------------------------------
async function loadLanguages() {
  const sel = $("langSel");
  try {
    const b = await (await fetch(`${API}/v1/languages`)).json();
    sel.innerHTML = "";
    (b.languages || []).forEach((name) => {
      const o = document.createElement("option");
      o.value = name;
      o.textContent = name;
      sel.appendChild(o);
    });
  } catch {
    sel.innerHTML = '<option value="Hindi">Hindi</option>';
  }
}

// ---- step 1: analyze -------------------------------------------------------
$("analyzeBtn").addEventListener("click", async () => {
  const file = $("videoInput").files[0];
  const st = $("analyzeStatus");
  if (!file) { setStatus(st, "Choose a video first.", "err"); return; }

  $("analyzeBtn").disabled = true;
  setStatus(st, "Uploading & analyzing… (separation + transcription)", "run");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const lang = $("sourceLang").value.trim();
    if (lang) fd.append("source_language", lang);

    const accepted = await jsonOrThrow(await fetch(`${API}/v1/analyze`, { method: "POST", body: fd }));
    const job = await pollJob(accepted.job_id, (j) =>
      setStatus(st, `Analyzing… (${j.status})`, "run"));

    renderManifest(job.manifest);
    setStatus(st, "Analysis complete.", "ok");
  } catch (e) {
    setStatus(st, "Analyze failed: " + e.message, "err");
  } finally {
    $("analyzeBtn").disabled = false;
  }
});

function fmt(t) { return Number(t).toFixed(2) + "s"; }

function renderManifest(m) {
  selectionId = m.selection_id;
  $("selInfo").innerHTML =
    `Selection <code>${m.selection_id}</code> · language <b>${m.source_language || "auto"}</b> · ` +
    `${m.sections.length} section(s)`;

  const wrap = $("sections");
  wrap.innerHTML = "";
  const secSel = $("sectionSel");
  secSel.innerHTML = "";

  m.sections.forEach((s) => {
    const div = document.createElement("div");
    div.className = "section";
    let html = `<h3>Section ${s.id} · ${fmt(s.start)}–${fmt(s.end)} · speakers: ${s.speakers.join(", ") || "—"}</h3>`;
    s.turns.forEach((t) => {
      html += `<div class="turn"><span class="t">${fmt(t.start)}–${fmt(t.end)}</span>` +
              `<span class="sp">${t.speaker}</span><span>${escapeHtml(t.text)}</span></div>`;
    });
    div.innerHTML = html;
    wrap.appendChild(div);

    const o = document.createElement("option");
    o.value = s.id;
    o.textContent = `Section ${s.id} (${fmt(s.start)}–${fmt(s.end)})`;
    secSel.appendChild(o);
  });

  $("resultCard").classList.remove("hidden");
  $("synthCard").classList.remove("hidden");
  loadLanguages();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---- voices ----------------------------------------------------------------
$("steps").addEventListener("input", (e) => ($("stepsVal").textContent = e.target.value));

$("loadVoices").addEventListener("click", async () => {
  const list = $("voiceList");
  list.classList.remove("hidden");
  list.innerHTML = "loading…";
  try {
    const b = await (await fetch(`${API}/v1/voices`)).json();
    list.innerHTML = "";
    (b.voices || []).slice(0, 200).forEach((v) => {
      const d = document.createElement("div");
      d.textContent = `${v.name}  —  ${v.id}`;
      d.title = "click to copy the id";
      d.addEventListener("click", () => navigator.clipboard.writeText(v.id));
      list.appendChild(d);
    });
  } catch (e) {
    list.innerHTML = "could not load voices: " + e.message;
  }
});

// ---- step 2: synthesize ----------------------------------------------------
$("synthBtn").addEventListener("click", async () => {
  const st = $("synthStatus");
  if (!selectionId) { setStatus(st, "Analyze a video first.", "err"); return; }

  const body = {
    selection_id: selectionId,
    section_id: Number($("sectionSel").value),
    target_language: $("langSel").value,
    settings: { steps: Number($("steps").value) },
  };
  const voices = parseVoices($("voices").value);
  if (Object.keys(voices).length) body.speaker_voices = voices;

  $("synthBtn").disabled = true;
  setStatus(st, "Submitting… the render runs on the GPU (this can take a couple of minutes).", "run");
  try {
    const accepted = await jsonOrThrow(
      await fetch(`${API}/v1/synthesize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }));

    const job = await pollJob(accepted.job_id, (j) =>
      setStatus(st, `Rendering… (${j.status})`, "run"));

    // Fetch the clip ONCE (the server deletes it on delivery), play from a blob.
    setStatus(st, "Fetching the rendered clip…", "run");
    const res = await fetch(`${API}${job.video_url}`);
    if (!res.ok) throw new Error(`could not fetch clip (HTTP ${res.status})`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    $("player").src = url;
    $("downloadLink").href = url;
    $("downloadLink").download = `${selectionId}_s${body.section_id}_${body.target_language}.mp4`;
    $("videoCard").classList.remove("hidden");
    $("videoCard").scrollIntoView({ behavior: "smooth" });
    setStatus(st, "Done.", "ok");
  } catch (e) {
    setStatus(st, "Synthesize failed: " + e.message, "err");
  } finally {
    $("synthBtn").disabled = false;
  }
});

function parseVoices(text) {
  const map = {};
  (text || "").split("\n").forEach((line) => {
    const t = line.trim();
    const i = t.indexOf("=");
    if (i > 0) map[t.slice(0, i).trim()] = t.slice(i + 1).trim();
  });
  return map;
}
