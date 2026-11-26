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

from chunk_video import RATE, run, probe
from tts_api import (
    AUTO_LANGUAGE, DEFAULT_VOICE, create_cloned_voice, detect_spoken_language, sdk_client,
    transcribe, synthesize, transcribe_speakers,
)
import requests
from vad import detect_speech
from translation import translate, ORIGINAL_LANGUAGE
from worker_lifecycle import exclusive_job_resource

ROOT = Path(__file__).resolve().parent
CLONE_SPEECH_SECONDS = 10
CLONE_UPLOAD_SECONDS = 10.1
MAX_DUB_TURN_SECONDS = 15

# Don't slow a dub below this tempo to fill its slot; pad natural silence instead of
# stretching speech into a sluggish, distorted drawl. The fast side is left uncapped so a
# longer translation is never truncated — length-aware translation keeps the factor near 1.
MIN_TEMPO = 0.85


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
        return create_cloned_voice(name, sample, language)
    with TemporaryDirectory() as temp:
        upload = Path(temp) / 'voice_sample.wav'
        padding = [(0, minimum - len(audio))] + [(0, 0)] * (audio.ndim - 1)
        sf.write(upload, np.pad(audio, padding), rate, subtype='PCM_16')
        print(f'Prepared {len(audio) / rate:.2f}s of speaker audio for 60db; '
              f'padded the upload to {CLONE_UPLOAD_SECONDS:g}s.', flush=True)
        return create_cloned_voice(name, upload, language)


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


def spoken_sample_bounds(path):
    """Find TTS edge silence while keeping quiet phonemes and internal pauses."""
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    window = max(1, round(0.01 * rate))
    remainder = len(audio) % window
    padded = np.pad(audio, ((0, (window - remainder) % window), (0, 0)))
    levels = np.sqrt(np.mean(padded.reshape(-1, window, audio.shape[1]) ** 2, axis=(1, 2)))
    active = np.flatnonzero(levels >= max(0.0003, float(levels.max()) * 0.015))
    if not len(active):
        return 0, len(audio), rate, len(audio)
    margin = round(0.02 * rate)
    return (max(0, int(active[0]) * window - margin),
            min(len(audio), (int(active[-1]) + 1) * window + margin), rate, len(audio))


DIAR_MERGE_GAP_SECONDS = 0.6   # merge consecutive same-speaker diarization turns across gaps this small
DIAR_MIN_TURN_SECONDS = 0.35   # drop diarization fragments shorter than this (backchannel/noise)


def speaker_turns(original, rate, sample_count, language):
    if original.stat().st_size > 100_000_000:
        raise ValueError('The audio exceeds the 60db transcription upload limit. Use a shorter video or a render length limit.')
    result = transcribe_speakers(original, language)
    diarization = result.get('speakers') if isinstance(result, dict) else None
    # Prefer 60db's real diarization (`speakers` array). On this backend the `segments`
    # text is NOT split per speaker, so reading it collapses every speaker into one turn.
    if isinstance(diarization, list) and any(
            isinstance(item, dict) and item.get('speaker') for item in diarization):
        turns = _turns_from_diarization(original, diarization, rate, sample_count, language)
        if turns:
            return turns
    # Fallback: backends that DO split text per speaker inside `segments`.
    return _turns_from_segments(result.get('segments') if isinstance(result, dict) else result,
                                rate, sample_count)


def _turns_from_diarization(original, diarization, rate, sample_count, language):
    """Build per-speaker turns from 60db's `speakers` array.

    60db diarizes into per-speaker time ranges but (on this backend) gives no
    per-speaker text, so each turn's words are recovered by re-transcribing its
    own audio slice. Adjacent same-speaker turns are merged across small gaps and
    micro-fragments are dropped, so a few clean turns replace many raw ones.
    """
    ranges = []
    for item in diarization:
        speaker = item.get('speaker')
        try:
            start, end = float(item['start']), float(item['end'])
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(speaker, str) and speaker.strip() and end > start:
            ranges.append([start, end, speaker.strip()])
    if not ranges:
        return []
    ranges.sort(key=lambda r: r[0])
    merged = [ranges[0]]
    for start, end, speaker in ranges[1:]:
        last = merged[-1]
        if speaker == last[2] and start - last[1] <= DIAR_MERGE_GAP_SECONDS:
            last[1] = max(last[1], end)   # same speaker, small gap -> one turn
        else:
            merged.append([start, end, speaker])
    kept = [r for r in merged if r[1] - r[0] >= DIAR_MIN_TURN_SECONDS]
    if not kept:   # never strand a whole section: keep the single longest turn
        kept = [max(merged, key=lambda r: r[1] - r[0])]
    turns = []
    with TemporaryDirectory() as temp, sf.SoundFile(original) as source:
        sample = Path(temp) / 'turn.wav'
        boundary = 0
        for start, end, speaker in kept:
            first = max(boundary, round(start * rate))   # keep turns non-overlapping and ordered
            last = min(sample_count, round(end * rate))
            if last <= first:
                continue
            source.seek(first)
            sf.write(sample, source.read(last - first, dtype='int16'), rate, subtype='PCM_16')
            try:
                text = transcribe(sample, language)
            except Exception as exc:
                # One turn's STT failing (even after retries) must not kill the whole job;
                # skip this turn so the rest of the dialogue still dubs.
                print(f'Skipped a {(last - first) / rate:.2f}s turn; 60db transcription failed: {exc}', flush=True)
                continue
            if text:
                turns.append((first, last, speaker, text))
                boundary = last
    return turns


def _turns_from_segments(segments, rate, sample_count):
    """Fallback for backends whose `segments` already carry per-speaker text."""
    if not isinstance(segments, list):
        raise RuntimeError('60db returned no speaker segments.')
    turns = []
    for segment in segments:
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


def split_long_speaker_turns(original, turns, language, minimum_pause=0.3):
    """Re-transcribe coarse diarization turns in bounded speech intervals."""
    limit = MAX_DUB_TURN_SECONDS * RATE
    if all(last - first <= limit for first, last, _, _ in turns):
        return turns
    speech = [(round(first * RATE), round(last * RATE))
              for first, last in detect_speech(original, minimum_pause=minimum_pause)['speech_intervals']]
    refined = []
    with TemporaryDirectory() as temp, sf.SoundFile(original) as source:
        sample = Path(temp) / 'turn.wav'
        for first, last, speaker, text in turns:
            if last - first <= limit:
                refined.append((first, last, speaker, text))
                continue
            ranges = [(max(first, start), min(last, end)) for start, end in speech
                      if start < last and end > first]
            start_count = len(refined)
            for start, end in ranges or [(first, last)]:
                for offset in range(start, end, limit):
                    stop = min(end, offset + limit)
                    source.seek(offset)
                    sf.write(sample, source.read(stop - offset, dtype='int16'), RATE, subtype='PCM_16')
                    rebuilt = transcribe(sample, language)
                    if rebuilt:
                        refined.append((offset, stop, speaker, rebuilt))
            if len(refined) == start_count:
                raise RuntimeError(
                    f'60db could not recover dialogue from the long turn at {first / RATE:.2f}–{last / RATE:.2f}s.'
                )
    return refined


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
        turns = split_long_speaker_turns(
            original, speaker_turns(original, rate, len(source), source_language), source_language
        )
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
        spoken_text = translate(text, target_language, target_seconds=(last - first) / rate)
        translation_seconds = time.monotonic() - started
        if target_language != ORIGINAL_LANGUAGE:
            lines.append(f'{target_language}: {spoken_text}\n')
        yield None, '\n'.join(lines), f'Generating replacement voice for segment {index}/{len(turns)}...'
        active_voice = (voices(speaker) if callable(voices) else voices[speaker]) if speaker else voice
        started = time.monotonic()
        try:
            generated = synthesize(spoken_text, active_voice, speed)
        except RuntimeError as exc:
            # A clone that keeps failing 60db synthesis (even after retries) must not
            # hard-stop the whole job; fall back to the default voice for this turn.
            if not active_voice or active_voice == DEFAULT_VOICE:
                raise
            print(f'Voice {active_voice} failed 60db synthesis ({exc}); '
                  f'using the default voice for segment {index}.', flush=True)
            generated = synthesize(spoken_text, DEFAULT_VOICE, speed)
        tts_seconds = time.monotonic() - started
        started = time.monotonic()
        target_seconds = (last - first) / rate
        speech_start, speech_end, generated_rate, total_samples = spoken_sample_bounds(generated)
        factor = (speech_end - speech_start) / generated_rate / target_seconds
        # Clamp the slow side so a short dub isn't stretched into an unnatural drawl; the
        # leftover slot becomes natural silence. Fast side uncapped -> no words truncated.
        factor = max(factor, MIN_TEMPO)
        fitted = folder / f'segment_{index:03d}_replacement.wav'
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', generated,
             '-af', f'atrim=start_sample={speech_start}:end_sample={speech_end},asetpts=PTS-STARTPTS,'
                    f'{tempo_filters(factor)},aresample={rate},apad,atrim=end_sample={last-first}',
             '-ar', str(rate), '-ac', '1', str(fitted)])
        audio, _ = sf.read(fitted, dtype='float32')
        if len(audio) != last - first:
            raise RuntimeError('Replacement speech does not match its original timeline slot.')
        timeline[first:last] = audio
        fitting_seconds = time.monotonic() - started
        segments.append({'start': start, 'end': end, 'text': text,
                         'target_language': target_language, 'spoken_text': spoken_text,
                         'generated_audio': generated, 'tempo_factor': factor,
                         'tts_trim_start_seconds': speech_start / generated_rate,
                         'tts_trim_end_seconds': (total_samples - speech_end) / generated_rate})
        if speaker:
            segments[-1].update(speaker=speaker, voice_id=active_voice)
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
