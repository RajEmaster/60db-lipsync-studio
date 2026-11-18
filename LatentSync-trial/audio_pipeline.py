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


def speech_timestamps(speech_audio, job):
    """Use confirmed RealtimeVAD endings on the speech driving lip-sync."""
    output = job / 'vad.json'
    vad_input = job / 'vad_input.wav'
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(speech_audio),
         '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(vad_input)])
    subprocess.run([sys.executable, '-u',
                    str(LAB / 'vad_timestamps.py'), '--audio', str(vad_input),
                    '--output', str(output)], check=True)
    return json.loads(output.read_text(encoding='utf-8'))


def extract_background(original_video, job, speech_duration):
    """Estimate background from the ORIGINAL soundtrack, never from generated TTS."""
    info = probe(original_video)
    if not any(stream['codec_type'] == 'audio' for stream in info['streams']):
        print('Reference video has no audio; final soundtrack will contain speech only.', flush=True)
        return None
    video_stream = next(stream for stream in info['streams'] if stream['codec_type'] == 'video')
    reference_duration = float(video_stream.get('duration') or info['format']['duration'])
    duration = min(speech_duration, reference_duration)
    samples = round(duration * SAMPLE_RATE)
    folder = job / 'original_audio'
    folder.mkdir(exist_ok=True)
    mix = folder / 'original_mix.wav'
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(original_video),
         '-vn', '-af', f'aresample={SAMPLE_RATE},apad,atrim=end_sample={samples}',
         '-ar', str(SAMPLE_RATE), '-ac', '2', '-c:a', 'pcm_f32le', str(mix)])
    print('Separating original vocals from background on CPU...', flush=True)
    subprocess.run([str(SEPARATOR_PYTHON), '-u', str(ROOT.parent / 'background-audio' / 'separate_voice.py'),
                    '--input', str(mix), '--output', str(folder)], check=True)
    return folder / 'background.wav'


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
