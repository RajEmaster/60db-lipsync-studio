"""Transcribe and replace utterances while preserving their original timeline slots."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import uuid

import numpy as np
import soundfile as sf

from chunk_video import run, probe
from tts_api import (
    AUTO_LANGUAGE, create_cloned_voice, detect_spoken_language, sdk_client, transcribe, synthesize,
    transcribe_speakers,
)
import requests
from vad import detect_speech
from translation import translate, ORIGINAL_LANGUAGE
from worker_lifecycle import exclusive_job_resource

ROOT = Path(__file__).resolve().parent
CLONE_SPEECH_SECONDS = 10
CLONE_UPLOAD_SECONDS = 10.1


def assemble_clone_sample(turns, output):
    """Collect up to ten seconds of speech from one speaker's longest turns."""
    remaining = CLONE_SPEECH_SECONDS * 16000
    with sf.SoundFile(output, mode='w', samplerate=16000, channels=1, subtype='PCM_16') as joined:
        for path, first, last in sorted(turns, key=lambda turn: turn[2] - turn[1], reverse=True):
            with sf.SoundFile(path) as source:
                source.seek(first)
                audio = source.read(min(remaining, last - first), dtype='int16')
            joined.write(audio)
            remaining -= len(audio)
            if not remaining:
                break


def reference_clone_record(video, language, sample_key=''):
    digest = hashlib.sha256()
    with Path(video).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    name = f'Lipsyncstudio {digest.hexdigest()[:12]} ({language})'
    if sample_key:
        name += f' {sample_key.split(":")[0]}'
    for setting in ('SIXTYDB_SDK_BASE_URL', 'SIXTYDB_API_KEY', 'SIXTYDB_WORKSPACE_ID'):
        digest.update(os.environ.get(setting, '').strip().encode())
        digest.update(b'\0')
    digest.update(f'instant:{language}'.encode())
    digest.update(sample_key.encode())
    folder = ROOT / 'local_jobs' / 'voice_clones'
    folder.mkdir(parents=True, exist_ok=True)
    return name, folder / f'{digest.hexdigest()}.json'


def clone_reference_voice(video, language, sample=None, sample_key=''):
    """Yield clone progress and the reusable voice ID once it can be selected."""
    language = (language or AUTO_LANGUAGE).strip().lower() or AUTO_LANGUAGE
    name, record = reference_clone_record(video, language, sample_key)
    result = reuse_clone_record(record)
    if result is None:
        if sample is None:
            yield None, 'Extracting original speaker audio for 60db voice cloning...'
            with TemporaryDirectory(dir=record.parent) as temp:
                original = Path(temp) / 'original.wav'
                run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video),
                     '-map', '0:a:0', '-t', '10', '-ac', '1', '-ar', '44100', str(original)])
                if not sf.info(original).frames:
                    raise ValueError('The reference video has no audio samples to clone.')
                yield None, 'Submitting the reference voice sample to 60db...'
                result = clone_sample(video, name, original, language, sample_key)
        else:
            result = clone_sample(video, name, sample, language, sample_key)
        temporary_record = record.with_suffix('.tmp')
        temporary_record.write_text(json.dumps(result), encoding='utf-8')
        temporary_record.replace(record)

    voice_id = str(result.get('id') or result['voice_id'])
    deadline = time.monotonic() + 20 * 60
    while True:
        if result.get('status') == 'failed':
            raise RuntimeError(f'60db could not create voice clone {voice_id}.')
        if time.monotonic() >= deadline:
            raise RuntimeError(f'60db is still processing voice {voice_id}. Generate again to resume checking it.')
        with sdk_client() as client:
            try:
                details = client.get_voice(voice_id)
            except requests.HTTPError as exc:
                if exc.response.status_code != 404:
                    raise
                details = None
        if isinstance(details, dict):
            details = details.get('data', details)
            if isinstance(details, dict):
                status = details.get('status')
                if status == 'failed':
                    raise RuntimeError(f'60db could not create voice clone {voice_id}.')
                # The lookup returns the synthesis ID, which can differ from the saved voice ID.
                synthesis_id = details.get('voice_provider_id') or details.get('voice_id') or details.get('id')
                if synthesis_id and status in (None, 'ready'):
                    yield str(synthesis_id), f'Using original speaker clone {voice_id}.'
                    return
        yield None, f'Waiting for 60db voice clone {voice_id}...'
        time.sleep(10)


def clone_sample(video, name, sample, language, sample_key):
    if language == AUTO_LANGUAGE:
        language = detect_spoken_language(sample)
        name, language_record = reference_clone_record(video, language, sample_key)
        cached = reuse_clone_record(language_record)
        if cached is not None:
            return cached
    audio, rate = sf.read(sample, dtype='int16')
    if not len(audio):
        raise ValueError('No speaker audio is available to clone.')
    minimum = round(CLONE_UPLOAD_SECONDS * rate)
    if len(audio) >= minimum:
        return create_cloned_voice(name, sample, language, recycle=True)
    with TemporaryDirectory() as temp:
        upload = Path(temp) / 'voice_sample.wav'
        padding = [(0, minimum - len(audio))] + [(0, 0)] * (audio.ndim - 1)
        sf.write(upload, np.pad(audio, padding), rate, subtype='PCM_16')
        print(f'Prepared {len(audio) / rate:.2f}s of speaker audio for 60db; '
              f'padded the upload to {CLONE_UPLOAD_SECONDS:g}s.', flush=True)
        return create_cloned_voice(name, upload, language, recycle=True)


def reuse_clone_record(record):
    with exclusive_job_resource('voice_slots.lock', 'Waiting for another clone request…'):
        if not record.is_file():
            return None
        result = json.loads(record.read_text(encoding='utf-8'))
        record.touch()
        return result


def tempo_filters(factor):
    filters = []
    while factor > 2:
        filters.append('atempo=2')
        factor /= 2
    while factor < 0.5:
        filters.append('atempo=0.5')
        factor /= 0.5
    return ','.join([*filters, f'atempo={factor:.10f}'])


def speaker_turns(original, rate, sample_count, language):
    if original.stat().st_size > 100_000_000:
        raise ValueError('The audio exceeds the 60db transcription upload limit. Use a shorter video or a render length limit.')
    turns = []
    for segment in transcribe_speakers(original, language):
        text = (segment.get('text') or '').strip()
        if not text:
            continue
        speaker = segment.get('speaker')
        if not isinstance(speaker, str) or not speaker.strip():
            raise RuntimeError('60db could not identify a speaker for every transcribed turn.')
        first = max(0, round(float(segment['start']) * rate))
        last = min(sample_count, round(float(segment['end']) * rate))
        if last > first:
            turns.append((first, last, speaker, text))
    turns.sort()
    for left, right in zip(turns, turns[1:]):
        if left[1] > right[0]:
            raise ValueError('60db returned overlapping speaker turns. Use a version with separate, non-overlapping dialogue.')
    return turns


def replace_voice(video, voice, speed=1.0, duration=0, target_language=ORIGINAL_LANGUAGE,
                  multiple_speakers=False, source_language=AUTO_LANGUAGE, prepared_turns=None,
                  speaker_voices=None, job_folder=None):
    """Yield progress, then a complete new speech track and timestamped transcript."""
    if (not voice and not multiple_speakers and prepared_turns is None) or not 0.5 <= float(speed) <= 2:
        raise ValueError('Select a replacement voice and a speed between 0.5 and 2.')
    info = probe(video)
    if not any(stream['codec_type'] == 'audio' for stream in info['streams']):
        raise ValueError('The reference video has no audio to transcribe.')
    video_stream = next(stream for stream in info['streams'] if stream['codec_type'] == 'video')
    length = float(video_stream.get('duration') or info['format']['duration'])
    if duration > 0:
        length = min(length, duration)
    folder = job_folder or ROOT / 'local_jobs' / 'stt' / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=True)
    original = folder / 'original_speech.wav'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video)]
    command += ['-t', str(length), '-vn', '-af', 'apad', '-ar', '16000', '-ac', '1', str(original)]
    if prepared_turns is None:
        try:
            run(command)
        except RuntimeError:
            raise ValueError('The reference video must contain an audio track for transcription.') from None
    source, rate = sf.read(original, dtype='float32')
    voices = speaker_voices or {}
    if prepared_turns is not None:
        turns = prepared_turns
    elif multiple_speakers:
        yield None, '', 'Transcribing and identifying speakers with 60db...'
        turns = speaker_turns(original, rate, len(source), source_language)
        longest = {}
        for first, last, speaker, _ in turns:
            if speaker not in longest or last - first > longest[speaker][1] - longest[speaker][0]:
                longest[speaker] = (first, last)
        for index, (speaker, (first, last)) in enumerate(longest.items(), 1):
            last = min(last, first + 10 * rate)
            sample = folder / f'speaker_{index:03d}.wav'
            if last - first < CLONE_SPEECH_SECONDS * rate:
                assemble_clone_sample([(original, start, stop) for start, stop, label, _ in turns
                                       if label == speaker], sample)
            else:
                sf.write(sample, source[first:last], rate, subtype='PCM_16')
            yield None, '', f'Preparing clone {index}/{len(longest)} for {speaker}...'
            for identity, message in clone_reference_voice(video, source_language, sample, f'{speaker}:{first}:{last}'):
                if identity:
                    voices[speaker] = identity
                yield None, '', message
    else:
        yield None, '', 'Finding speech in the original video...'
        turns = [(max(0, round(start * rate)), min(len(source), round(end * rate)), None, None)
                 for start, end in detect_speech(original)['speech_intervals']]
    if not turns and prepared_turns is None:
        raise ValueError('No speech was detected in the reference video.')
    timeline = np.zeros(len(source), dtype=np.float32)
    segments, lines = [], []
    synthesis_started = time.monotonic()
    for index, (first, last, speaker, text) in enumerate(turns, 1):
        if last <= first:
            continue
        start, end = first / rate, last / rate
        if text is None:
            clip = folder / f'segment_{index:03d}.wav'
            sf.write(clip, source[first:last], rate, subtype='PCM_16')
            yield None, '\n'.join(lines), f'Transcribing speech segment {index}/{len(turns)} with 60db...'
            text = transcribe(clip, source_language)
            if not text:
                lines.append(f'[{start:.2f}s - {end:.2f}s] No speech transcribed.')
                (folder / 'transcript.txt').write_text('\n'.join(lines), encoding='utf-8')
                yield None, '\n'.join(lines), f'Skipping segment {index}/{len(turns)}. No speech transcribed.'
                continue
        label = f'{speaker} · Original' if speaker else 'Original'
        lines.append(f'[{start:.2f}s - {end:.2f}s] {label}: {text}')
        if target_language != ORIGINAL_LANGUAGE:
            yield None, '\n'.join(lines), f'Translating segment {index}/{len(turns)} into {target_language} with 60db...'
        started = time.monotonic()
        spoken_text = translate(text, target_language)
        translation_seconds = time.monotonic() - started
        if target_language != ORIGINAL_LANGUAGE:
            lines.append(f'{target_language}: {spoken_text}\n')
        yield None, '\n'.join(lines), f'Generating replacement voice for segment {index}/{len(turns)}...'
        started = time.monotonic()
        generated = synthesize(spoken_text, voices[speaker] if speaker else voice, speed)
        tts_seconds = time.monotonic() - started
        started = time.monotonic()
        info = sf.info(generated)
        target_seconds = (last - first) / rate
        factor = info.duration / target_seconds
        fitted = folder / f'segment_{index:03d}_replacement.wav'
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', generated,
             '-af', f'{tempo_filters(factor)},aresample={rate},apad,atrim=end_sample={last-first}',
             '-ar', str(rate), '-ac', '1', str(fitted)])
        audio, _ = sf.read(fitted, dtype='float32')
        if len(audio) != last - first:
            raise RuntimeError('Replacement speech does not match its original timeline slot.')
        timeline[first:last] = audio
        fitting_seconds = time.monotonic() - started
        segments.append({'start': start, 'end': end, 'text': text,
                         'target_language': target_language, 'spoken_text': spoken_text,
                         'generated_audio': generated, 'tempo_factor': factor})
        if speaker:
            segments[-1].update(speaker=speaker, voice_id=voices[speaker])
        (folder / 'transcript.json').write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding='utf-8')
        (folder / 'transcript.txt').write_text('\n'.join(lines), encoding='utf-8')
        if prepared_turns is not None:
            elapsed = time.monotonic() - synthesis_started
            estimated_total = elapsed * len(turns) / index
            yield None, '\n'.join(lines), (
                f'Turn {index}/{len(turns)} ready. Translation {translation_seconds:.1f}s, '
                f'TTS {tts_seconds:.1f}s, timing fit {fitting_seconds:.1f}s. '
                f'Section synthesis elapsed {elapsed:.1f}s, estimated total {estimated_total:.1f}s, '
                f'remaining {max(0, estimated_total - elapsed):.1f}s.'
            )
    if not segments and prepared_turns is None:
        raise ValueError('60db could not find any transcribable speech in the reference video.')
    output = folder / 'replacement_speech.wav'
    sf.write(output, timeline, rate, subtype='PCM_16')
    yield str(output), '\n'.join(lines), 'Transcription and replacement speech ready. Starting lip-sync...'
