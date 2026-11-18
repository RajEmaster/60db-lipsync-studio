"""Conservative SoX cleanup of replacement speech; never processes background effects."""
import os
from pathlib import Path
import shutil
import subprocess

import soundfile as sf
from vad import detect_speech


def sox_executable():
    configured = os.environ.get('MUSICTALK_SOX')
    if configured:
        return configured
    found = shutil.which('sox')
    if found:
        return found
    for variable in ('ProgramFiles(x86)', 'ProgramFiles', 'LOCALAPPDATA'):
        root = os.environ.get(variable)
        if root:
            for candidate in Path(root).glob('sox*/sox.exe'):
                return str(candidate)
    local = os.environ.get('LOCALAPPDATA')
    if local:
        for candidate in (Path(local) / 'Microsoft' / 'WinGet' / 'Packages').glob('ChrisBagwell.SoX*/sox*/sox.exe'):
            return str(candidate)
    raise RuntimeError('SoX was not found. Install SoX or set MUSICTALK_SOX to sox.exe.')


def clean_speech(audio, job):
    """Use a confirmed non-speech gap as the noise reference, preserving the timeline."""
    timing = detect_speech(audio)
    gaps = timing['pause_intervals']
    if not timing['speech_intervals'] or not gaps:
        print('SoX cleanup skipped: no speech or no reliable noise-only interval.', flush=True)
        return Path(audio), False
    start, end = max(gaps, key=lambda gap: gap[1] - gap[0])
    # Avoid edges close to speech. Use at most one second of the quiet interval.
    start += 0.05
    end = min(end - 0.05, start + 1.0)
    if end - start < 0.2:
        print('SoX cleanup skipped: noise reference is too short.', flush=True)
        return Path(audio), False
    data, rate = sf.read(audio, dtype='float32')
    noise = data[round(start * rate):round(end * rate)]
    if abs(noise).max() < 1e-6:
        print('SoX cleanup skipped: noise reference is effectively silent.', flush=True)
        return Path(audio), False
    folder = Path(job) / 'speech_cleanup'
    folder.mkdir(exist_ok=True)
    reference, profile = folder / 'noise_reference.wav', folder / 'noise.prof'
    output = folder / 'speech_cleaned.wav'
    sf.write(reference, noise, rate, subtype='PCM_16')
    sox = sox_executable()
    for command in ([sox, str(reference), '-n', 'noiseprof', str(profile)],
                    [sox, str(audio), str(output), 'pad', '0', '4096s',
                     'noisered', str(profile), '0.15', 'trim', '0', f'{len(data)}s']):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('SoX speech cleanup failed: ' + result.stderr[-1000:])
    original, cleaned = sf.info(audio), sf.info(output)
    if (original.frames, original.samplerate, original.channels) != (cleaned.frames, cleaned.samplerate, cleaned.channels):
        raise RuntimeError('SoX cleanup changed the speech timeline or format.')
    print('SoX cleaned replacement speech; original background effects are preserved separately.', flush=True)
    return output, True
