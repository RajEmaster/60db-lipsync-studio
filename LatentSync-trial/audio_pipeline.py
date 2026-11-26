"""Speech timing, original-background extraction, and final soundtrack mixing.

Keep model inference in subprocesses so its memory is released before rendering.
"""
import json
from pathlib import Path
import subprocess
import sys
from settings import WORKER_PYTHON

from chunk_video import probe, run

ROOT = Path(__file__).resolve().parent
LAB = ROOT
SAMPLE_RATE = 44100
SEPARATOR_PYTHON = WORKER_PYTHON


def speech_timestamps(speech_audio, job, minimum_pause=0.3):
    """Use confirmed RealtimeVAD endings on the speech driving lip-sync."""
    output = job / 'vad.json'
    vad_input = job / 'vad_input.wav'
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(speech_audio),
         '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(vad_input)])
    subprocess.run([sys.executable, '-u',
                    str(LAB / 'vad_timestamps.py'), '--audio', str(vad_input),
                    '--output', str(output), '--minimum-pause', str(minimum_pause)], check=True)
    return json.loads(output.read_text(encoding='utf-8'))


def separate_stems(original_video, folder, duration, cpu_threads=4, reuse=True):
    """Split the ORIGINAL soundtrack into (speech_voice, background) stems.

    Returns (voice_path, background_path), or (None, None) when the source has no
    audio. The stems are written side by side so the early speech-cleaning pass
    (transcribe/diarize/clone on clean vocals) and the later background mix never
    run Demucs on the same section twice: `reuse` returns existing stems as-is.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    voice, background = folder / 'speaker_voice.wav', folder / 'background.wav'
    if reuse and voice.is_file() and background.is_file():
        return voice, background
    info = probe(original_video)
    if not any(stream['codec_type'] == 'audio' for stream in info['streams']):
        print('Reference video has no audio; speech cleaning / background skipped.', flush=True)
        return None, None
    video_stream = next(stream for stream in info['streams'] if stream['codec_type'] == 'video')
    reference_duration = float(video_stream.get('duration') or info['format']['duration'])
    duration = min(duration, reference_duration)
    samples = round(duration * SAMPLE_RATE)
    mix = folder / 'original_mix.wav'
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(original_video),
         '-vn', '-af', f'aresample={SAMPLE_RATE},apad,atrim=end_sample={samples}',
         '-ar', str(SAMPLE_RATE), '-ac', '2', '-c:a', 'pcm_f32le', str(mix)])
    print('Separating speech from background...', flush=True)
    subprocess.run([str(SEPARATOR_PYTHON), '-u', str(ROOT.parent / 'background-audio' / 'separate_voice.py'),
                    '--input', str(mix), '--output', str(folder),
                    '--cpu-threads', str(cpu_threads)], check=True)
    return voice, background


def extract_background(original_video, job, speech_duration, cpu_threads=4):
    """Background stem of the ORIGINAL soundtrack (reuses the early speech-clean pass)."""
    _voice, background = separate_stems(original_video, job / 'original_audio', speech_duration, cpu_threads)
    return background


def build_soundtrack(speech, background, output, duration, background_gain=1.0):
    """Mix continuous speech with time-aligned ambience; never concatenate chunk audio."""
    if not 0 <= background_gain <= 1:
        raise ValueError('Background gain must be between 0 and 1.')
    samples = round(duration * SAMPLE_RATE)
    speech_stream = next(stream for stream in probe(speech)['streams'] if stream['codec_type'] == 'audio')
    # Explicit mono duplication avoids FFmpeg's default -3 dB center-channel downmix.
    channels = ('pan=stereo|c0=c0|c1=c0' if speech_stream['channels'] == 1
                else 'aformat=channel_layouts=stereo')
    speech_filter = (
        f'aresample={SAMPLE_RATE},{channels},'
        f'apad,atrim=end_sample={samples},asetpts=PTS-STARTPTS'
    )
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(speech)]
    if background is None:
        command += ['-af', speech_filter]
    else:
        # Repeat background only when the reference video also repeats to cover speech.
        command += ['-stream_loop', '-1', '-i', str(background), '-filter_complex',
                    f'[0:a]{speech_filter}[speech];'
                    f'[1:a]aresample={SAMPLE_RATE},atrim=end_sample={samples},'
                    f'asetpts=PTS-STARTPTS,volume={background_gain}[background];'
                    '[speech][background]amix=inputs=2:duration=first:normalize=0,'
                    'alimiter=limit=0.98:level=0:latency=1[mixed]', '-map', '[mixed]']
    command += ['-ar', str(SAMPLE_RATE), '-ac', '2', '-c:a', 'pcm_f32le', str(output)]
    run(command)
    return output
