# Running Lipsyncstudio on Kaggle (LatentSync 1.6 / 1.5)

`kaggle_lipsyncstudio.ipynb` runs the full `local_ui.py` dubbing studio on Kaggle's
free GPUs and exposes it as a public `*.gradio.live` link. The notebook is
self-contained: it clones the public repo and applies all Linux patches in-cell,
so nothing in the repo needs to change. It downloads **both** LatentSync 1.5 (256px)
and 1.6 (512px) so you pick either in the UI's **Engine** section. On `GPU T4 ×2` the
app runs **one resident worker per GPU** and renders a video's clips in parallel across
both cards (~2× faster) — set **GPUs to use = 0** (all).

## One-time setup on Kaggle

1. Go to <https://www.kaggle.com/code> → **New Notebook** → **File → Import Notebook**
   and upload `kaggle_lipsyncstudio.ipynb` (or **File → Import from GitHub** once this
   file is pushed).
2. In the right-hand **Settings** panel:
   - **Accelerator:** `GPU T4 ×2` (2×16 GB, recommended — renders on both cards) or `GPU P100`.
   - **Internet:** `On` (required for clone, pip, and HuggingFace downloads).
3. **Add-ons → Secrets:** add a secret named **`SIXTYDB_API_KEY`** with your 60db key
   and tick it so it is attached to this notebook.
4. **Run All.** The last cell blocks and prints
   `Running on public URL: https://xxxx.gradio.live` — open that link.

## What the notebook does

| Step | Action |
| --- | --- |
| 1 | Clone `RajEmaster/60db-lipsync-studio`, install `libgl1`/`ffmpeg` |
| 2 | Build an isolated **Python 3.10** venv at `/kaggle/working/env310` with `uv` (auto-downloads CPython 3.10) |
| 3 | Into that env: `torch 2.5.1+cu121`, then the exact project lock (`trial-requirements-lock.txt`, read straight from the repo) + UI deps + Demucs |
| 4 | Repoint `settings.WORKER_PYTHON` (a Windows `.venv` path) at the 3.10 env so the spawned GPU worker and Demucs separator run there |
| 5 | Write `.env` from the `SIXTYDB_API_KEY` secret |
| 6 | Download LatentSync **1.5** unet + VAE + Whisper tiny + InsightFace buffalo_l (~5 GB) |
| 6b | Download LatentSync **1.6** unet (512px) into `checkpoints/v1.6/` (~5 GB) |
| 7 | Launch Gradio (`run_kaggle.py`) with `share=True` on `0.0.0.0:7860`, under the 3.10 env |

In the UI's **Engine** section: **version** = 1.6 (sharper 512px) or 1.5 (faster 256px);
**GPU memory mode** = auto; **GPUs to use** = 0 (all). The engine loads the model once per
section (not per clip), and on T4 ×2 clips fan out across both GPUs.

### Why a separate Python 3.10 env?

Kaggle's default kernel is a newer Python. The project's pinned versions
(scikit-image 0.22, decord 0.6, numba, ...) only ship wheels for Python 3.10, so on a
newer Python `pip` falls back to building them from source and fails
(`metadata-generation-failed: scikit-image`). Rebuilding 3.10 with `uv` and installing
the proven lock avoids this entirely and matches the environment the app was tested on.

## Notes & limits

- **Session lifetime:** the share link lives only while the Kaggle session runs
  (interactive GPU sessions run up to ~9 h; the weekly GPU quota is ~30 h). Re-run to
  get a fresh link. Checkpoints live in `/kaggle/working` for the session only.
- **1.6 vs 1.5:** 1.6 renders at 512px (sharper teeth/lips) and auto-selects `swap` memory
  mode on a 16 GB T4 — it fits a single T4 but is slower than 1.5's 256px. Keep **16
  frames/batch**; you can raise **Inference steps** (e.g. 20–30).
- **Multi-GPU:** on T4 ×2, leave **GPUs to use = 0**. The app starts one resident worker
  per card (`CUDA_VISIBLE_DEVICES` 0 and 1) draining a shared clip queue; a video's clips
  render in parallel. One GPU faulting degrades speed instead of failing the job.
- **Background restore** uses Demucs (cell with `demucs==4.0.1`). If that install fails,
  the app still works — just leave *"Restore original background"* unchecked in the UI.
- **Security:** a `gradio.live` link is public to anyone who has it and spends your 60db
  credits. Share it carefully; close the session when done.
- **Speed:** to avoid re-downloading the ~5 GB checkpoints every session, you can later
  publish them once as a Kaggle Dataset and attach it instead of running cell 5.
