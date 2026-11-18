# Lipsyncstudio

The active app and LatentSync 1.5 engine live in `LatentSync-trial/`.

## Set up a fresh Windows checkout

Install Python 3.10, an NVIDIA driver compatible with CUDA 12.1, and FFmpeg with
`ffmpeg` and `ffprobe` on PATH. Run these commands from the repository root in
PowerShell.

```powershell
py -3.10 -m venv .\LatentSync-trial\.ui-venv
py -3.10 -m venv .\LatentSync-trial\.venv
.\LatentSync-trial\.ui-venv\Scripts\python.exe -m pip install -r .\LatentSync-trial\requirements.txt
.\LatentSync-trial\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r .\LatentSync-trial\trial-requirements-lock.txt
.\LatentSync-trial\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r .\background-audio\requirements-separator-lock.txt
.\LatentSync-trial\.venv\Scripts\python.exe .\LatentSync-trial\download_trial.py
.\LatentSync-trial\.venv\Scripts\python.exe .\LatentSync-trial\download_faces.py
Copy-Item .\LatentSync-trial\.env.example .\LatentSync-trial\.env
```

Set `SIXTYDB_API_KEY` in the local `.env` before using 60db. Install SoX if you
want to use noise cleanup on uploaded speech. Demucs can download its weights
the first time background restoration runs. Review the terms of each upstream
checkpoint for your intended use. Model weights, credentials, environments,
and local test media and render outputs are excluded from Git.

The engine includes [ByteDance LatentSync](https://github.com/bytedance/LatentSync)
with local inference and face-processing changes. Its Apache license is in
[`LatentSync-trial/LICENSE`](LatentSync-trial/LICENSE). The vendored
InsightFace 0.7.3 source includes its [MIT license](LatentSync-trial/vendor/insightface-0.7.3/LICENSE).

## Run locally

Run `./LatentSync-trial/start_ui.ps1`, then open http://127.0.0.1:7861.
The main form uses the movie workflow for both short tests and full videos.
Set the transcription section length, render clip range, VAD settings, and
compute options directly on the form. Use **Start or resume job** to begin or
continue with a saved job ID. See [the pipeline guide](LatentSync-trial/PIPELINE.md).
The app uses 60db diarization and automatically links speakers across sections
with a [CPU WeSpeaker model](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM),
downloaded on the first job that needs cross-section matching.

The UI uses `.ui-venv`; GPU inference and Demucs use the existing `.venv`.
FFmpeg and ffprobe must be on PATH. VAD policy and audio processing live in
application files; installed third-party sources are not modified.

`vad-audio-lab/` contains historical reference notes. The app uses the
installed RealtimeVAD package.

Defaults are roughly five-minute transcription sections and 10–25 second
render clips. A resumed job uses its saved settings.

Configure SIXTYDB_API_KEY and optionally SIXTYDB_SDK_BASE_URL in LatentSync-trial/.env (see .env.example). The app uses SDK text_to_speech and get_voices methods. Generated WAV audio is validated before rendering. Preview creates a speech WAV; Generate creates speech then renders the video. Unverified legacy emotion controls are omitted.

Background separation runs from `background-audio/` when requested.

The separate supplied-speech render path can use SoX to learn steady noise
from a confirmed non-speech gap. Demucs separates the original background,
which is mixed with the complete speech track before the final video mux.
Set MUSICTALK_SOX to override the SoX binary location.

The SDK is pinned to sixtydb 2.2.0. The app includes a compatibility adapter for
the current top-level audio_base64 response; third-party package source is unchanged.
Development REST endpoint settings are no longer used.

The default speech source is now **Transcribe reference video**. Upload a video
with speech, select a 60db voice, and generate. RealtimeVAD locates original
utterances; the 60db SDK transcribes and regenerates each utterance. FFmpeg tempo
adjustment fits each replacement into its original time slot, preserving original
pauses and the overall video duration. Original background restoration remains
independent. Transcript text/JSON and replacement audio are saved under
LatentSync-trial/local_jobs/stt/. Custom text and uploaded audio remain available.
The SDK adapter fixes the current STT multipart field and Content-Type and wraps
headerless LINEAR16 TTS responses using their returned sample rate.
A 2.88-second end-to-end test with a different voice, 2 inference steps, and
background restoration completed in 108.84 seconds with 72 final video frames.
Timing fitting does not reproduce the original speaker's word-level prosody;
large duration differences can sound unnaturally fast or slow. STT errors carry
through to generated speech. This test does not establish long-video quality.

Translation dubbing: in Transcribe reference video mode, select Hindi, English,
another listed language, or type a language name. Select Original (no translation)
to keep the source language. Each original utterance is transcribed, translated
using SDK chat.completion (60db-tiny by default), synthesized in the chosen voice,
and fitted to its original timeline slot. Use a voice supporting the target
language. The UI and transcript artifacts retain both original and translated
text. SIXTYDB_TRANSLATION_MODEL overrides the chat model. Truncated or empty
translations stop generation instead of silently dubbing the original language.
Verified live English-to-Hindi transcription/translation/TTS with unchanged
reference duration and Hindi-to-English chat translation. The existing GPU
rendering and original-background mixing stages remain in use.
