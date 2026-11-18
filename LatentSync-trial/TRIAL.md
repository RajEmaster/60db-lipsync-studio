# Local RTX 3050 4 GB experiment

See [PIPELINE.md](PIPELINE.md) for the current RealtimeVAD/background-restoration
workflow. The test history below also describes earlier versions of the service.

This is a separate checkout of ByteDance LatentSync, using the official 1.5
checkpoint and the 256-pixel stage2 configuration. MuseTalk's environment is
not modified. A completed smoke test is not evidence of production quality.

The experimental runner uses FP16, CPU Whisper, CPU ONNX face detection,
VAE slicing/tiling, staged VAE/U-Net GPU residency, and small temporal batches.
It loads the checkpoint with memory mapping to limit system RAM use.

InsightFace 0.7.3 Python source is under `vendor/insightface-0.7.3`, downloaded
from PyPI. Its optional compiled mesh-rendering imports are disabled; only
ONNX detection and 106-point landmarks are used. The face detector's ONNX
provider is changed to CPU. Detection and landmark weights are from the
`public-data/insightface` Hugging Face mirror of the buffalo_l model bundle;
the original GitHub release download failed on this connection.

Model sources:
- https://huggingface.co/ByteDance/LatentSync-1.5
- https://huggingface.co/stabilityai/sd-vae-ft-mse
- https://huggingface.co/public-data/insightface

From this directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -u trial_inference.py
.\.venv\Scripts\python.exe -u trial_inference.py --steps 20 --output trial_outputs/test20.mp4
```

The default inputs are a 0.64-second excerpt of the repository's demo1 files.
Use relative paths without spaces for inputs/outputs, since upstream FFmpeg
commands use shell strings. The runner writes timing and CUDA memory figures
to `<output>.json` on success. It does not launch a server.

`--sequential-offload` is an experimental fallback for out-of-memory errors.
Reducing `--frames` can affect temporal consistency; the standard model uses 16.

## Results on this machine (2026-11-13)

Both tests completed and produced readable H.264 videos with AAC audio:

| Test | Runtime reported by runner | Peak CUDA allocated / reserved |
| --- | --- | --- |
| 4 frames, 2 steps | 28.14 s | 2534 / 2776 MiB |
| 16 frames, 20 steps | 39.42 s | 2712 / 2858 MiB |

The 0.64-second source excerpt produced 18 output frames at 25 fps (0.72 s)
due to audio feature padding. Runtime excludes Python imports. Guidance was
1.0; face inference resolution was 256, while the output retains the source
1080x1920 dimensions. These measurements establish short-clip functionality,
not long-video stability or a visual-quality comparison with MuseTalk.

Validated output: `trial_outputs/test20_standard16.mp4`.

## OmniVoice local interface

Run `./start_ui.ps1` in PowerShell, then open http://127.0.0.1:7861.
The page supports text-to-speech and uploaded/recorded audio. Voice selection,
preview, speed, emotion tags/style/intensity, stability, similarity, speech
synthesis steps, language, and advanced emotion controls use MuseTalk's existing
OmniVoice client. All GPU-related callbacks share one queue. The TTS process
and its children stop before the separate LatentSync worker loads.

The render length limit defaults to 0 (full speech); set it to 3–5 seconds for
a short test. The speech WAV and output video have separate players. Each
render's files and log are saved under `local_jobs/<job_id>/`.

Verified the Gradio text-to-speech endpoint through an actual OmniVoice request
(speed 1.1, happy style/tag, intensity 0.75) and a LatentSync render. The short
integration test used 16-frame batches and 2 diffusion steps, completed in
29.33 seconds for the render stage, and reserved at most 2886 MiB of CUDA
memory. Voice discovery returned 341 entries. Production UI defaults remain
20 diffusion steps and 16-frame batches.

## Standalone overlapping chunk preparation

From the MuseTalk directory:

```powershell
.\venv\Scripts\python.exe ..\LatentSync-trial\chunk_video.py --video "C:\path\video.mp4" --audio "C:\path\speech.wav"
```

`--audio` is optional: otherwise the input video's audio is used if present.
With no audio stream, only video chunks and the manifest are produced.
External speech defines the timeline duration and must not exceed the video.
The full speech is preserved as `speech_full.wav` at 16 kHz mono; only
sub-frame timing is rounded to the 25 fps timeline.

The standalone script creates a new `chunk_jobs/<id>/` folder containing
`chunk_001.mp4`, its corresponding WAV, subsequent pairs, a normalized
`timeline.mp4`, and `manifest.json`. Chunk MP4s are silent; speech is in the
matching WAV. Paths with spaces are supported.

Defaults: actual extracted clips are 5–7 seconds, target 6 seconds, with 0.4
seconds of extra context per side. The context counts toward the length limit,
so a 20-second source generally needs four clips. Quiet speech intervals are
preferred when compatible with the length bounds. Detection uses FFmpeg
`silencedetect`, not neural VAD; `--no-pauses` chooses balanced cuts instead.
Inputs shorter than 5 seconds remain a single shorter clip.

The manifest uses frame indices with exclusive end boundaries. After rendering
each chunk, retain `[keep_from_frame:keep_to_frame]`, join retained sections in
index order, and attach `speech_full.wav`. These boundaries cover the timeline
exactly once. This standalone script prepares files only. The server uses
`render_chunks.py` to run the full preparation/render/join workflow.

Tested on a 20-second demo excerpt: four clips within 5–7 seconds, exact frame
counts, matching audio sample counts, and 500 retained frames with no gaps.

## Chunked rendering from the server

The Generate button now runs `render_chunks.py` in the separate environment.
Speech is synthesized once before chunk preparation. Longer references are
processed only through the speech duration; shorter references repeat to cover
the speech. The optional length limit applies before chunking, with 0 meaning
full speech.

Each 5–7 second contextual clip runs in its own LatentSync process, still using
16-frame batches by default. Process exit releases that clip's RAM and GPU
allocations before the next starts. Completed results remain on disk.

The join stage trims each render to the manifest's exclusive frame boundaries,
verifies retained frame counts, concatenates the video sections, and muxes the
single complete speech track. Final frame count must equal the planned total.
No overlapping frames or audio segments are duplicated in the final video.
This uses context-and-trim, not crossfading or shared inference state, so seam
quality still depends on the input and needs visual review.

Files under `local_jobs/<job_id>/`:
- `chunks/manifest.json`: clip/context boundaries and matching media.
- `renders/chunk_*.mp4` and `.log`: independent renders and diagnostics.
- `retained/chunk_*.mp4`: trimmed sections used in the final timeline.
- `output.mp4`: final video returned to the browser, with complete speech.
- `output.mp4.json` and `render.log`: summary and overall progress.

A failed chunk stops the job and reports its index, timeline range, and log;
completed clips are retained. A face must be clearly detectable throughout
the reference. The server does not silently skip frames where detection fails.

End-to-end server verification: a 20-second, 720x1280 reference with uploaded
speech completed as four rendered clips. Final output:
`local_jobs/e2507a96fcb546b38fd3df20bed582cd/output.mp4`.
The integration test used 2 diffusion steps for pipeline verification (the UI
default remains 20). Peak CUDA reserved memory was 2938 MiB. Final output has
500 consecutive video frames at 25 fps and a continuous 20-second audio track.
