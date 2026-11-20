# Running Lipsyncstudio on Kaggle (LatentSync 1.5)

`kaggle_lipsyncstudio.ipynb` runs the full `local_ui.py` dubbing studio on Kaggle's
free 16 GB GPU and exposes it as a public `*.gradio.live` link. The notebook is
self-contained: it clones the public repo and applies all Linux patches in-cell,
so nothing in the repo needs to change.

## One-time setup on Kaggle

1. Go to <https://www.kaggle.com/code> → **New Notebook** → **File → Import Notebook**
   and upload `kaggle_lipsyncstudio.ipynb` (or **File → Import from GitHub** once this
   file is pushed).
2. In the right-hand **Settings** panel:
   - **Accelerator:** `GPU P100` or `GPU T4 x2` (both are 16 GB).
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
| 7 | Launch Gradio (`run_kaggle.py`) with `share=True` on `0.0.0.0:7860`, under the 3.10 env |

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
- **16 GB headroom:** the renderer keeps its 4 GB-GPU memory-staging logic (harmless at
  16 GB). You can safely raise **Inference steps** (e.g. 30–50) and keep **16 frames/batch**.
- **Background restore** uses Demucs (cell with `demucs==4.0.1`). If that install fails,
  the app still works — just leave *"Restore original background"* unchecked in the UI.
- **Security:** a `gradio.live` link is public to anyone who has it and spends your 60db
  credits. Share it carefully; close the session when done.
- **Speed:** to avoid re-downloading the ~5 GB checkpoints every session, you can later
  publish them once as a Kaggle Dataset and attach it instead of running cell 5.
