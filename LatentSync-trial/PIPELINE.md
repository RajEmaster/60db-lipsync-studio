# Lipsyncstudio pipeline

Run `start_ui.ps1` from this directory, then open http://127.0.0.1:7861. The app uses LatentSync 1.5 and presents the movie workflow directly on the main form.

## Long-video job

The main form works for both short tests and full projects. **Sections and
render clips** exposes section length, clip bounds, overlap, and VAD settings.
**Compute** exposes inference steps, frames per batch, and PyTorch CPU threads
for LatentSync and Demucs. The numbers below are defaults. A short video can
use a shorter transcription section to exercise multiple sections. New values
apply to a new job. Resume keeps the saved settings.

Maximum clip length is a hard cap including overlap. Lowering it clamps the
target and minimum if necessary. The planner can use shorter clips when the
requested minimum and maximum cannot cover the section exactly.

The job log shows local timestamps and elapsed time for section preparation,
transcription, speaker matching and clone lookup, turn translation and TTS,
background processing, clip rendering, and final assembly. After the first
turn, section synthesis shows an estimated total and time remaining based on
completed turns. That estimate covers dialogue translation, synthesis, timing
fit, and saving the dubbed track. The render and soundtrack mix have separate
timings. The actual synthesis duration is saved in each section's
`dub_ready.json`.

1. **Plan sections.** Copy the original upload into the job folder. Put section boundaries about five minutes apart on a 25 fps timeline. Search 12 seconds on either side of each nominal boundary for confirmed RealtimeVAD pauses in the original audio. Use the nominal boundary when there is no suitable pause. Keep a very short final tail with the previous section.
2. **Transcribe sections.** FFmpeg extracts each section as a frame-aligned video with lossless audio and a 16 kHz mono WAV for STT. 60db transcribes the WAV and returns speaker labels and timestamps. The app saves these results before moving on. It transcribes the whole project through separate section requests. Each section's timestamps are local to that section, and its offset in `sections.json` places them on the full timeline.
3. **Match speaker identities automatically.** The app uses the 60db labels directly within each section. Across sections, CPU WeSpeaker embeddings compare representative samples and link clear matches. The same label can refer to different people in different requests. Ambiguous matches get separate identities. A single section uses the returned labels directly. **Only one speaker in the entire video** assigns all turns to one identity. The map is saved with the job.
4. **Create and reuse one clone per person.** The app collects up to ten seconds from the longest detected turns for each mapped person across the sections. It pads the upload with silence to 10.1 seconds when needed for the service's duration check. The job saves the resulting 60db synthesis ID. A preexisting synthesis voice ID can be supplied in single-speaker mode. Clean solo speech helps both matching and cloning. The installed pipeline does not isolate simultaneous speakers and stops when 60db returns overlapping dialogue turns.

   When 60db rejects an automatic clone because the voice slots are full, the app
   removes the least recently used eligible Lipsyncstudio clone and retries once.
   Eligible clones have a local creation record and are private, idle, and absent
   from unfinished jobs. If none qualify, the job pauses with a quota error.
5. **Translate and dub a section.** Translate each dialogue turn with 60db chat when a target language is selected. Synthesize each translated turn with its mapped clone and fit it into the original time slot with FFmpeg tempo adjustment. The app saves the continuous dubbed track before starting GPU work. A section with no detected dialogue keeps its original video frames.
6. **Plan render clips.** Run RealtimeVAD on the section's **dubbed** speech. The planner targets 17-second video clips, prefers cuts in confirmed pauses, and uses frame-balanced cuts when no pause fits. The 10–25 second limits include 0.4 seconds of overlap on each available side. A whole section shorter than ten seconds can be one shorter clip.
7. **Render and assemble.** Run one LatentSync subprocess per render clip. Discard overlap frames and join the exact retained frame counts. If background restoration is enabled, CPU Demucs estimates the original section's ambience and FFmpeg mixes it with the continuous dubbed track. Join completed section video streams and PCM soundtracks in order, encode the final AAC track once, and verify the final video frame count.

### Resume and job files

Enter the original **Job ID** and select **Start or resume job**. The job retains its original settings. A failed or interrupted job reuses saved STT sections, its saved speaker map and voice IDs, completed section dubs, and validated GPU clip renders. A partially synthesized section starts its dubbing stage again. Keep enough disk space for the source copy, section media, render clips, and final MP4.

| Path under `local_jobs/long/<id>/` | Contents |
| --- | --- |
| `job.json`, `sections.json` | Original job settings and absolute section boundary frame indices |
| `speaker_samples.json`, `speaker_map.json`, `voices.json` | Reference samples, linked identities, and synthesis IDs |
| `section_NNNN/turns.json`, `speech/` | Speaker turns, source audio, translated transcript, and fitted speech |
| `section_NNNN/chunks*/manifest.json`, `renders/`, `retained/` | Context and trim boundaries, rendered chunks, and retained video |
| `section_NNNN/final_soundtrack.wav`, `output.mp4` | Section soundtrack and assembled section video |
| `output.mp4`, `transcript.json`, `run_*.log` | Final export, full transcript with absolute timestamps, and progress logs |

## Standard short-video render helpers

Single-speaker reference dubbing uses RealtimeVAD on the original audio, transcribes each detected utterance with 60db, and reuses one clone. Short multi-speaker mode uses a single 60db diarization request and a clone per returned speaker. Both translate each turn before speech synthesis and finish the dubbed soundtrack before planning the roughly five-second rendering clips. Text-to-speech and uploaded speech are also available. Optional SoX cleanup applies to uploaded speech.

## Workers and face detection

| Worker | Job |
| --- | --- |
| `local_ui.py` | Gradio inputs, job IDs, progress, subprocess supervision |
| `long_video.py`, `speaker_matching.py` | Section boundaries, batched STT, CPU cross-section speaker matching, resume and final assembly |
| `voice_replacement.py`, `tts_api.py`, `translation.py` | Speaker samples, 60db cloning/STT/TTS/chat, and utterance timing |
| `render_chunks.py`, `chunk_video.py`, `audio_pipeline.py` | RealtimeVAD render boundaries, per-clip workers, overlap trimming, soundtrack mix |
| `trial_inference.py` | LatentSync 1.5 with a 256-pixel face crop, CPU Whisper Tiny audio features, and staged GPU inference. The inference batch is 4, 8, or 16 frames independently of clip duration. |

LatentSync uses **InsightFace** on CPU through ONNX Runtime. Its detector is **SCRFD** using `det_10g.onnx` at a 512×512 detection size. The **106-point landmark model** supplies the face alignment and lip contour. The code filters detections by confidence, size, and aspect ratio and picks the largest eligible face in each frame. The generated mouth is composited inside a feathered region centered on the original lip landmarks. The blend peaks at 90% generated pixels, and pixels outside this region come from the source frame. Frames with no eligible face retain the original picture. An entirely faceless clip skips diffusion. The dubbed soundtrack and restored background continue across these frames on the original timeline.

Speaker diarization identifies voices on the audio track. It does not identify which visible face is talking. A shot with two visible faces can still render the larger face even when the smaller face is speaking. Active-speaker face tracking and rendering more than one face in the same frame are outside the current workflow.
