"""Prepare and render a resumable long-video dubbing job."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import soundfile as sf

from chunk_video import FPS, RATE, probe, run
from job_logging import log_event
from render_chunks import frame_count, save_json, valid_render
from speaker_matching import match_speakers
from tts_api import sdk_client
from voice_replacement import (
    CLONE_SPEECH_SECONDS, MAX_DUB_TURN_SECONDS, assemble_clone_sample, clone_reference_voice, replace_voice,
    speaker_turns, split_long_speaker_turns,
)
from vad import detect_speech
from locking_compat import try_lock, unlock
from worker_lifecycle import watch_parent

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / 'local_jobs' / 'long'
LONG_RENDER_DEFAULTS = {
    'section_seconds': 300, 'boundary_search_seconds': 12, 'vad_pause_seconds': 0.3,
    'clip_min_seconds': 10, 'clip_target_seconds': 17, 'clip_max_seconds': 25,
    'overlap_seconds': 0.4, 'cpu_threads': 4,
}


def clip_lengths_with_cap(minimum, target, maximum):
    minimum = min(minimum, maximum)
    return minimum, max(minimum, min(target, maximum))


def long_render_settings(values):
    try:
        settings = {key: float(values.get(key, default)) for key, default in LONG_RENDER_DEFAULTS.items()}
    except (TypeError, ValueError):
        raise ValueError('Enter a number for each section and render setting.') from None
    if not all(math.isfinite(value) for value in settings.values()):
        raise ValueError('Enter finite values for the section and render settings.')
    if not 10 <= settings['section_seconds'] <= 600:
        raise ValueError('Set transcription sections between 10 and 600 seconds.')
    if not 0 <= settings['boundary_search_seconds'] <= 30:
        raise ValueError('Set the section pause search between 0 and 30 seconds.')
    if not 0.1 <= settings['vad_pause_seconds'] <= 2:
        raise ValueError('Set the minimum VAD pause between 0.1 and 2 seconds.')
    minimum, target, maximum = (settings[f'clip_{key}_seconds'] for key in ('min', 'target', 'max'))
    if not all(4 <= value <= 25 for value in (minimum, target, maximum)):
        raise ValueError('Enter clip lengths between 4 and 25 seconds.')
    minimum, target = clip_lengths_with_cap(minimum, target, maximum)
    settings.update(clip_min_seconds=minimum, clip_target_seconds=target)
    overlap = settings['overlap_seconds']
    if not 0 <= overlap <= 1.5 or 2 * round(overlap * FPS) >= math.ceil(minimum * FPS):
        raise ValueError('Use 0 to 1.5 seconds of overlap per side and less than half the minimum clip length.')
    threads = settings['cpu_threads']
    if not 1 <= threads <= 16 or not threads.is_integer():
        raise ValueError('Choose a whole number of CPU worker threads from 1 to 16.')
    settings['cpu_threads'] = int(threads)
    return settings


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


@contextmanager
def job_lock(job):
    with (job / 'job.lock').open('a+b') as handle:
        if not handle.tell():
            handle.write(b'0')
            handle.flush()
        if not try_lock(handle):
            raise RuntimeError('This long-video job is already running.')
        try:
            yield
        finally:
            unlock(handle)


def start_job(job, video, options):
    options = {**options, **long_render_settings(options)}
    info = probe(video)
    stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
    if stream is None or not any(s['codec_type'] == 'audio' for s in info['streams']):
        raise ValueError('Upload a video containing an audio track.')
    duration = float(stream.get('duration') or info['format']['duration'])
    if options['duration'] > 0:
        duration = min(duration, options['duration'])
    frames = round(duration * FPS)
    if frames < 1:
        raise ValueError('The upload contains no frames to render.')
    source = job / ('source' + Path(video).suffix.lower())
    temporary = source.with_name('source-copy.tmp')
    shutil.copyfile(video, temporary)
    temporary.replace(source)
    save_json(job / 'job.json', {**options, 'source': source.name, 'frames': frames})


def take_section(source, section, start, frames):
    reference = section / 'reference.mp4'
    original = section / 'speech' / 'original_speech.wav'
    original.parent.mkdir(parents=True, exist_ok=True)
    if not reference.exists():
        temporary = reference.with_name('reference-partial.mp4')
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(start / FPS),
             '-i', str(source), '-map', '0:v:0', '-map', '0:a:0',
             '-vf', f'fps={FPS}:start_time=0,tpad=stop_mode=clone:stop_duration=0.08',
             '-frames:v', str(frames), '-t', str(frames / FPS), '-c:v', 'libx264',
             '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'alac', str(temporary)])
        if not valid_render(temporary, frames, exact=True):
            raise RuntimeError(f'{section.name} did not produce {frames} video frames.')
        temporary.replace(reference)
    if not original.exists():
        temporary = original.with_name('original-partial.wav')
        normalize_audio(reference, temporary, frames, RATE, 1, 'pcm_s16le')
        temporary.replace(original)
    return reference, original


def normalize_audio(source, output, frames, rate, channels, codec, gain=1.0):
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source), '-vn',
         '-af', f'aresample={rate},apad,atrim=end_sample={frames * (rate // FPS)},volume={gain}',
         '-ar', str(rate), '-ac', str(channels), '-c:a', codec, str(output)])


def section_boundaries(job, config):
    path = job / 'sections.json'
    if path.exists():
        return read_json(path)
    bounds = [0]
    total = config['frames']
    section_frames = round(config['section_seconds'] * FPS)
    radius = min(round(config['boundary_search_seconds'] * FPS), section_frames // 4)
    for nominal in range(section_frames, total, section_frames):
        # Keep a short final tail with the preceding section.
        if total - nominal < radius:
            break
        if not radius:
            bounds.append(nominal)
            continue
        first = nominal - radius
        window = job / 'boundary_audio.wav'
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(first / FPS),
             '-i', str(job / config['source']), '-map', '0:a:0', '-t', str(2 * radius / FPS),
             '-ac', '1', '-ar', str(RATE), '-c:a', 'pcm_s16le', str(window)])
        candidates = []
        if sf.info(window).frames:
            for start, end in detect_speech(window, minimum_pause=config['vad_pause_seconds'])['pause_intervals']:
                candidates.append(first + round((start + end) * 0.5 * FPS))
        bounds.append(min(candidates, key=lambda frame: abs(frame - nominal)) if candidates else nominal)
    bounds.append(total)
    save_json(path, bounds)
    return bounds


def prepare_sections(job, config):
    sections, observations = [], []
    started = time.monotonic()
    bounds = section_boundaries(job, config)
    log_event(f'Section plan ready in {time.monotonic() - started:.1f}s. {len(bounds) - 1} sections.')
    for index, (start, end) in enumerate(zip(bounds, bounds[1:]), 1):
        frames = end - start
        section = job / f'section_{index:04d}'
        log_event(f'Preparing section {index}, starting at video time {start / FPS:.2f}s…')
        started = time.monotonic()
        reference, original = take_section(job / config['source'], section, start, frames)
        log_event(f'{section.name} media ready in {time.monotonic() - started:.1f}s.')
        transcript = section / 'turns.json'
        if transcript.exists():
            turns = read_json(transcript)
            log_event(f'{section.name} reused its saved transcription ({len(turns)} turns).')
        else:
            log_event(f'{section.name} transcription and speaker diarization started.')
            started = time.monotonic()
            turns = speaker_turns(original, RATE, frames * (RATE // FPS), config['source_language'])
            save_json(transcript, turns)
            log_event(f'{section.name} transcription finished in {time.monotonic() - started:.1f}s ({len(turns)} turns).')
        if (any(last - first > MAX_DUB_TURN_SECONDS * RATE for first, last, _, _ in turns)
                and not ((section / 'dub_ready.json').is_file() and (section / 'speech/replacement_speech.wav').is_file())):
            log_event(f'{section.name} refining coarse speaker turns at detected speech pauses.')
            started = time.monotonic()
            refined = split_long_speaker_turns(original, turns, config['source_language'])
            save_json(transcript, refined)
            turns = refined
            log_event(f'{section.name} recovered {len(turns)} bounded turns in {time.monotonic() - started:.1f}s.')
        sections.append((section, reference, frames, turns))
        longest = {}
        for first, last, label, _ in turns:
            candidate = longest.get(label)
            if candidate is None or last - first > candidate[1] - candidate[0]:
                longest[label] = (first, last)
        for ordinal, (label, (first, last)) in enumerate(longest.items(), 1):
            last = min(last, first + 10 * RATE)
            sample = section / f'speaker_{ordinal:03d}.wav'
            if not sample.exists():
                temporary = sample.with_suffix('.tmp.wav')
                with sf.SoundFile(original) as audio:
                    audio.seek(first)
                    sf.write(temporary, audio.read(last - first, dtype='int16'), RATE, subtype='PCM_16')
                temporary.replace(sample)
            observations.append({
                'turn_label': f'{section.name}/{label}',
                'sample': sample.relative_to(job).as_posix(),
                'start_seconds': start / FPS + first / RATE,
                'sample_frames': last - first,
            })
    save_json(job / 'speaker_samples.json', observations)
    return sections, observations


def speaker_map(job, config, observations):
    started = time.monotonic()
    keys = {item['turn_label'] for item in observations}
    path = job / 'speaker_map.json'
    if path.exists():
        mapping = read_json(path)
    elif config['single_speaker']:
        mapping = dict.fromkeys(sorted(keys), 'Speaker 1')
    else:
        mapping = match_speakers(job, observations)
    if (not isinstance(mapping, dict) or set(mapping) != keys
            or any(not isinstance(name, str) or not name.strip() for name in mapping.values())):
        raise ValueError('The saved speaker map does not match the transcribed turns.')
    if not path.exists():
        save_json(path, mapping)
    log_event(f'Speaker mapping ready in {time.monotonic() - started:.1f}s. '
              f'Using {len(set(mapping.values()))} identities.')
    return mapping


def section_voice_resolver(job, config, mapping, observations, sections):
    path = job / 'voices.json'
    voices = read_json(path) if path.exists() else {}
    best = {}
    for item in observations:
        identity = mapping[item['turn_label']]
        if identity not in best or item['sample_frames'] > best[identity]['sample_frames']:
            best[identity] = item
    if config.get('voice_id'):
        return lambda section, identity: config['voice_id']
    turns_by_speaker = {}
    for section, _, _, turns in sections:
        original = section / 'speech' / 'original_speech.wav'
        for first, last, label, _ in turns:
            identity = mapping[f'{section.name}/{label}']
            turns_by_speaker.setdefault(identity, []).append((original, first, last))
    sample_indices = {identity: index for index, identity in enumerate(best, 1)}
    last_scope = None

    def resolve(section, identity):
        nonlocal last_scope
        scope = (section.name, identity)
        if last_scope == scope:
            return voices[identity]
        save_json(job / 'active_voice.json', {'section': section.name, 'speaker': identity})
        if identity in voices:
            with sdk_client() as client:
                live_ids = {voice.get('voice_provider_id') or voice.get('voice_id')
                            for voice in client.get_my_voices()['data']}
            if voices[identity] in live_ids:
                last_scope = scope
                return voices[identity]
            log_event(f'Recreating the recycled clone for {identity}.')
        item = best[identity]
        sample = job / item['sample']
        if item['sample_frames'] < CLONE_SPEECH_SECONDS * RATE:
            directory = job / 'clone_samples'
            directory.mkdir(exist_ok=True)
            sample = directory / f'speaker_{sample_indices[identity]:03d}.wav'
            temporary = sample.with_suffix('.tmp.wav')
            assemble_clone_sample(turns_by_speaker[identity], temporary)
            temporary.replace(sample)
        log_event(f'Resolving 60db voice for {identity}…')
        started = time.monotonic()
        for voice_id, message in clone_reference_voice(sample, config['source_language'], sample):
            log_event(message)
            if voice_id:
                voices[identity] = voice_id
                save_json(path, voices)
        log_event(f'Voice for {identity} ready in {time.monotonic() - started:.1f}s.')
        last_scope = scope
        return voices[identity]

    return resolve


def render_section(section, reference, frames, turns, config, mapping, voices):
    section_started = time.monotonic()
    output = section / 'output.mp4'
    soundtrack = section / 'final_soundtrack.wav'
    summary = section / 'output.mp4.json'
    if summary.exists() and soundtrack.is_file() and valid_render(output, frames, exact=True):
        log_event(f'Reusing completed {section.name}.')
        return output, soundtrack
    summary.unlink(missing_ok=True)
    if not turns:
        # 60db found no dialogue. Preserve the frames and keep ambience only if requested.
        temporary = section / 'silent-section.mp4'
        run(['ffmpeg', '-v', 'error', '-y', '-i', str(reference), '-an', '-c:v', 'copy', str(temporary)])
        temporary.replace(output)
        if config['restore_background']:
            source = reference
        else:
            source = section / 'silence.wav'
            sf.write(source, [0.0], 44100, subtype='PCM_16')
        normalize_audio(source, soundtrack, frames, 44100, 2, 'pcm_f32le',
                        config['background_gain'] if config['restore_background'] else 1.0)
        save_json(summary, {'total_frames': frames, 'chunks': 0})
        log_event(f'{section.name} completed in {time.monotonic() - section_started:.1f}s. No dialogue to synthesize.')
        return output, soundtrack
    dubbing = section / 'dub_ready.json'
    audio = section / 'speech' / 'replacement_speech.wav'
    if not dubbing.is_file() or not audio.is_file():
        log_event(f'{section.name} synthesis started. {len(turns)} turns across {frames / FPS:.2f}s of video.')
        synthesis_started = time.monotonic()
        routed = [(first, last, mapping[f'{section.name}/{label}'], text) for first, last, label, text in turns]
        for replacement, _, message in replace_voice(
                reference, None, config['speed'], target_language=config['target_language'],
                prepared_turns=routed, speaker_voices=voices, job_folder=audio.parent):
            log_event(f'{section.name} {message}')
            if replacement:
                elapsed = time.monotonic() - synthesis_started
                save_json(dubbing, {'samples': sf.info(audio).frames, 'synthesis_seconds': round(elapsed, 2)})
                log_event(f'{section.name} synthesis finished in {elapsed:.1f}s.')
    else:
        saved = read_json(dubbing)
        elapsed = saved.get('synthesis_seconds')
        timing = f' Original synthesis took {elapsed:.1f}s.' if elapsed is not None else ''
        log_event(f'{section.name} reused its saved dubbed audio.{timing}')
    command = [sys.executable, '-u', str(ROOT / 'render_chunks.py'),
               '--job', str(section.relative_to(ROOT)), '--video', str(reference),
               '--audio', str(audio), '--parent-pid', str(os.getpid()),
               '--steps', str(config['steps']), '--frames', str(config['batch_frames']),
               '--resume', '--min-seconds', str(config['clip_min_seconds']),
               '--target-seconds', str(config['clip_target_seconds']),
               '--max-seconds', str(config['clip_max_seconds']),
               '--overlap', str(config['overlap_seconds']),
               '--vad-pause-seconds', str(config['vad_pause_seconds']),
               '--cpu-threads', str(config['cpu_threads']),
               '--engine-mode', str(config.get('engine_mode', 'resident')),
               '--version', str(config.get('engine_version', '1.5')),
               '--offload', str(config.get('offload', 'auto')),
               '--gpus', str(int(config.get('max_gpus', 0)))]
    if config.get('guidance_scale') is not None:
        command += ['--guidance-scale', str(config['guidance_scale'])]
    command += ['--seed', str(int(config.get('seed', 1247))),
                '--crop-smoothing', str(float(config.get('crop_smoothing', 0.2))),
                '--restore-strength', str(float(config.get('restore_strength', 0.5)))]
    if config.get('face_restore'):
        command.append('--face-restore')
    if config['restore_background']:
        command += ['--background-video', str(reference), '--background-gain', str(config['background_gain'])]
    log_event(f'{section.name} render and mix started.')
    started = time.monotonic()
    subprocess.run(command, cwd=ROOT, check=True)
    if not valid_render(output, frames, exact=True):
        raise RuntimeError(f'{section.name} has the wrong number of output frames.')
    log_event(f'{section.name} render and mix finished in {time.monotonic() - started:.1f}s. '
              f'Section total {time.monotonic() - section_started:.1f}s after transcription and speaker mapping.')
    return output, soundtrack


def assemble(job, rendered, frames):
    video_lines, audio_lines = [], []
    for source, audio in rendered:
        video = source.with_name('join_video.mp4')
        count = frame_count(source)
        audio_info = sf.info(audio)
        if audio_info.samplerate != 44100 or audio_info.frames != count * (44100 // FPS):
            raise RuntimeError(f'{source.parent.name} soundtrack does not match its frame count.')
        if not valid_render(video, count, exact=True) or video.stat().st_mtime_ns < source.stat().st_mtime_ns:
            temporary_video = video.with_name('join_video-partial.mp4')
            run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-an', '-c:v', 'copy', str(temporary_video)])
            temporary_video.replace(video)
        video_lines.append(f"file '{video.relative_to(job).as_posix()}'\nduration {count / FPS:.2f}\n")
        audio_lines.append(f"file '{audio.relative_to(job).as_posix()}'\nduration {count / FPS:.2f}\n")
    (job / 'video_sections.txt').write_text(''.join(video_lines), encoding='ascii')
    (job / 'audio_sections.txt').write_text(''.join(audio_lines), encoding='ascii')
    output = job / 'output.mp4'
    temporary = job / 'output-partial.mp4'
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
         '-f', 'concat', '-safe', '0', '-i', str(job / 'video_sections.txt'),
         '-f', 'concat', '-safe', '0', '-i', str(job / 'audio_sections.txt'),
         '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-t', str(frames / FPS),
         '-movflags', '+faststart', str(temporary)])
    if frame_count(temporary) != frames:
        raise RuntimeError('The final video does not match the planned frame count.')
    temporary.replace(output)
    return output


def save_project_transcript(job, sections):
    timeline = []
    offset = 0
    for section, _, frames, turns in sections:
        if turns:
            entries = read_json(section / 'speech' / 'transcript.json')
            for entry in entries:
                timeline.append({
                    **entry, 'section': section.name,
                    'start': offset / FPS + entry['start'],
                    'end': offset / FPS + entry['end'],
                })
        offset += frames
    save_json(job / 'transcript.json', timeline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--video')
    parser.add_argument('--options')
    parser.add_argument('--parent-pid', type=int)
    args = parser.parse_args()
    if args.parent_pid:
        watch_parent(args.parent_pid)
    if len(args.job) != 32 or any(char not in '0123456789abcdef' for char in args.job):
        parser.error('Use the 32-character long-video job ID.')
    job = JOBS / args.job
    if args.video:
        job.mkdir(parents=True, exist_ok=True)
    if not job.is_dir():
        parser.error('Long-video job not found.')
    with job_lock(job):
        started = time.monotonic()
        log_event(f'Job {args.job} started or resumed.')
        if not (job / 'job.json').exists():
            if not args.video or not args.options:
                parser.error('This job needs its original upload and options.')
            start_job(job, args.video, read_json(Path(args.options)))
        config = read_json(job / 'job.json')
        config.update(long_render_settings(config))
        log_event(f'Render clips: {config["clip_min_seconds"]:g}s minimum, '
                  f'{config["clip_target_seconds"]:g}s target, {config["clip_max_seconds"]:g}s maximum.')
        if ((job / 'completed.json').exists() and (job / 'transcript.json').exists()
                and valid_render(job / 'output.mp4', config['frames'], exact=True)):
            log_event(f'Reused completed output: {job / "output.mp4"}')
            return
        sections, observations = prepare_sections(job, config)
        mapping = speaker_map(job, config, observations)
        resolve = section_voice_resolver(job, config, mapping, observations, sections)
        rendered = [render_section(section, reference, frames, turns, config, mapping,
                                   lambda identity, section=section: resolve(section, identity))
                    for section, reference, frames, turns in sections]
        save_project_transcript(job, sections)
        log_event('Final assembly started.')
        assembly_started = time.monotonic()
        output = assemble(job, rendered, config['frames'])
        save_json(job / 'completed.json', {'frames': config['frames'], 'sections': len(sections)})
        log_event(f'Final assembly finished in {time.monotonic() - assembly_started:.1f}s. '
                  f'This run took {time.monotonic() - started:.1f}s. Completed output: {output}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        log_event(f'Job stopped: {exc}')
        raise
