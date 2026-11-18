# Lipsyncstudio pipeline

Run `start_ui.ps1` from this directory, then open http://127.0.0.1:7861. The app uses LatentSync 1.5. The **Long-video dubbing** controls use the reference video, target language, and render settings from the main form.

## Long-video job

1. **Plan sections.** Copy the original upload into the job folder. Put section boundaries about five minutes apart on a 25 fps timeline. Search 12 seconds on either side of each nominal boundary for confirmed RealtimeVAD pauses in the original audio. Use the nominal boundary when there is no suitable pause. Keep a very short final tail with the previous section.
2. **Transcribe sections.** FFmpeg extracts each section as a frame-aligned video with lossless audio and a 16 kHz mono WAV for STT. 60db transcribes the WAV and returns speaker labels and timestamps. The app saves these results before moving on. It transcribes the whole project through separate section requests. Each section's timestamps are local to that section, and its offset in `sections.json` places them on the full timeline.
3. **Match speaker identities across sections.** The same label can refer to different people in different sections. For dialogue, the app supplies a ZIP with a sample for each section-local speaker and an editable JSON map. Give the same person the same name wherever they appear, upload the JSON, and resume with the same job ID. Use **Only one speaker in the entire video** when the whole video is known to contain one speaker to skip this review.
4. **Create and reuse one clone per person.** The app chooses the longest detected turn for each mapped person across the sections and uses up to its first ten seconds. It saves the resulting 60db synthesis ID for the job. A preexisting synthesis voice ID can be supplied in single-speaker mode. A clean solo sample is more useful than a longer sample with music, reverb, or another person talking. Listen to the review samples before approving the map. The installed pipeline does not isolate simultaneous speakers and stops when 60db returns overlapping dialogue turns.
5. **Translate and dub a section.** For each speaker turn, translate its text with 60db chat if a target language is selected, synthesize it with that person's clone, and fit it into the original time slot with FFmpeg tempo adjustment. The app saves a continuous dubbed track and transcript for that section before starting its GPU work. A section with no detected dialogue keeps its original video frames.
6. **Plan render clips.** Run RealtimeVAD on the section's **dubbed** speech. The planner targets 17-second video clips, prefers cuts in confirmed pauses, and uses frame-balanced cuts when no pause fits. The 10–25 second limits include 0.4 seconds of overlap on each available side. A whole section shorter than ten seconds can be one shorter clip.
7. **Render and assemble.** Run one LatentSync subprocess per render clip. Discard overlap frames and join the exact retained frame counts. If background restoration is enabled, CPU Demucs estimates the original section's ambience and FFmpeg mixes it with the continuous dubbed track. Join completed section video streams and PCM soundtracks in order, encode the final AAC track once, and verify the final video frame count.

### Resume and job files

Use the original **Long-video job ID** to resume. The job retains its original settings. A failed or interrupted job reuses saved STT sections, its approved speaker map and voice IDs, completed section dubs, and validated GPU clip renders. A partially synthesized section starts its dubbing stage again. Keep enough disk space for the source copy, section media, render clips, and final MP4.

| Path under `local_jobs/long/<id>/` | Contents |
| --- | --- |
| `job.json`, `sections.json` | Original job settings and absolute section boundary frame indices |
| `speaker_review.zip`, `speaker_map.json`, `voices.json` | Review samples, approved identities, and synthesis IDs |
| `section_NNNN/turns.json`, `speech/` | Speaker turns, source audio, translated transcript, and fitted speech |
| `section_NNNN/chunks*/manifest.json`, `renders/`, `retained/` | Context and trim boundaries, rendered chunks, and retained video |
| `section_NNNN/final_soundtrack.wav`, `output.mp4` | Section soundtrack and assembled section video |
| `output.mp4`, `transcript.json`, `run_*.log` | Final export, full transcript with absolute timestamps, and progress logs |

## Standard short-video job

Single-speaker reference dubbing uses RealtimeVAD on the original audio, transcribes each detected utterance with 60db, and reuses one clone. Short multi-speaker mode uses a single 60db diarization request and a clone per returned speaker. Both translate each turn before speech synthesis and finish the dubbed soundtrack before planning the roughly five-second rendering clips. Text-to-speech and uploaded speech are also available. Optional SoX cleanup applies to uploaded speech.

## Workers and face detection

| Worker | Job |
| --- | --- |
| `local_ui.py` | Gradio inputs, job IDs, progress, subprocess supervision |
| `long_video.py` | Section boundaries, batched STT, speaker mapping, resume and final assembly |
| `voice_replacement.py`, `tts_api.py`, `translation.py` | Speaker samples, 60db cloning/STT/TTS/chat, and utterance timing |
| `render_chunks.py`, `chunk_video.py`, `audio_pipeline.py` | RealtimeVAD render boundaries, per-clip workers, overlap trimming, soundtrack mix |
| `trial_inference.py` | LatentSync 1.5 with a 256-pixel face crop, CPU Whisper Tiny audio features, and staged GPU inference. The inference batch is 4, 8, or 16 frames independently of clip duration. |

LatentSync uses **InsightFace** on CPU through ONNX Runtime. Its detector is **SCRFD** using `det_10g.onnx` at a 512×512 detection size. The **106-point landmark model** supplies the face alignment used for rendering. The code filters detections by confidence, size, and aspect ratio and picks the largest eligible face in each frame. Frames with no eligible face retain the original picture. An entirely faceless clip skips diffusion. The dubbed soundtrack and restored background continue across these frames on the original timeline.

Speaker diarization identifies voices on the audio track. It does not identify which visible face is talking. A shot with two visible faces can still render the larger face even when the smaller face is speaking. Active-speaker face tracking and rendering more than one face in the same frame are outside the current workflow.
