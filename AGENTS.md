# Project boundaries

- Work on the active Lipsyncstudio app in `LatentSync-trial/`, using LatentSync 1.5 as the lip-sync engine.
- MuseTalk and OmniVoice are retired. Do not add imports or runtime dependencies on them.
- `vad-audio-lab/` and `archive/` are historical references. The active UI is `LatentSync-trial/local_ui.py`.
- Install VAD, audio libraries, and tools as packages. Do not clone dependencies unless source modifications are actually required.
- Keep custom RealtimeVAD policy and audio behavior in our application files; do not edit site-packages or third-party VAD source.
- Preserve existing checkpoints, environments, and trial results unless the user explicitly requests removal.
- Never commit API keys or `.env` files.
