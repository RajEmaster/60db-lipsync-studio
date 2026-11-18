# Background audio separation

separate_voice.py estimates vocals with Demucs on CPU and saves speaker_voice.wav
and background.wav. Input must be 44.1 kHz stereo WAV. Run with the existing
LatentSync-trial/.venv/Scripts/python.exe, using --input and --output arguments.

The Lipsyncstudio audio pipeline calls this worker when background restoration is enabled.
RealtimeVAD is independent and is not imported here. models/ retains the existing
Demucs weights. separator_packages/ contains the existing installed Demucs dependencies, including
torchaudio matched to the engine PyTorch version. The worker uses this installation
with the existing engine environment; no third-party source is modified.
