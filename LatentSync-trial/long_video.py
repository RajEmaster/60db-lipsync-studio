"""Prepare and render a resumable long-video dubbing job."""
import argparse
from contextlib import contextmanager
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

from chunk_video import FPS, RATE, probe, run
from render_chunks import frame_count, save_json, valid_render
from voice_replacement import clone_reference_voice, replace_voice, speaker_turns
from vad import detect_speech
from worker_lifecycle import watch_parent

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / 'local_jobs' / 'long'
SECTION_FRAMES = 300 * FPS


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


@contextmanager
def job_lock(job):
    with (job / 'job.lock').open('a+b') as handle:
        if not handle.tell():
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError('This long-video job is already running.') from None
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def start_job(job, video, options):
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
             '-vf', 'fps=25,tpad=stop_mode=clone:stop_duration=0.08',
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
    radius = min(12 * FPS, SECTION_FRAMES // 4)
    for nominal in range(SECTION_FRAMES, total, SECTION_FRAMES):
        # Keep a short final tail with the preceding section.
        if total - nominal < radius:
            break
        first = nominal - radius
        window = job / 'boundary_audio.wav'
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(first / FPS),
             '-i', str(job / config['source']), '-map', '0:a:0', '-t', str(2 * radius / FPS),
             '-ac', '1', '-ar', str(RATE), '-c:a', 'pcm_s16le', str(window)])
        candidates = []
        if sf.info(window).frames:
            for start, end in detect_speech(window)['pause_intervals']:
                candidates.append(first + round(max(start, min(end, radius / FPS)) * FPS))
        bounds.append(min(candidates, key=lambda frame: abs(frame - nominal)) if candidates else nominal)
    bounds.append(total)
    save_json(path, bounds)
    return bounds


def prepare_sections(job, config):
    sections, observations = [], []
    bounds = section_boundaries(job, config)
    for index, (start, end) in enumerate(zip(bounds, bounds[1:]), 1):
        frames = end - start
        section = job / f'section_{index:04d}'
        print(f'Preparing section {index}, starting at {start / FPS:.2f}s...', flush=True)
        reference, original = take_section(job / config['source'], section, start, frames)
        transcript = section / 'turns.json'
        if transcript.exists():
            turns = read_json(transcript)
        else:
            print('Transcribing this section and identifying speaker turns with 60db...', flush=True)
            turns = speaker_turns(original, RATE, frames * (RATE // FPS), config['source_language'])
            save_json(transcript, turns)
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


def speaker_map(job, config, observations, mapping_file):
    keys = {item['turn_label'] for item in observations}
    approved = job / 'speaker_map.json'
    template = job / 'speaker_map_template.json'
    if approved.exists():
        mapping = read_json(approved)
        if mapping_file and read_json(mapping_file) != mapping:
            raise ValueError('This job already has an approved speaker map. Start a new job to change it.')
    elif mapping_file:
        mapping = read_json(mapping_file)
    elif config['single_speaker']:
        mapping = dict.fromkeys(sorted(keys), 'Speaker 1')
    elif not keys:
        mapping = {}
    else:
        save_json(template, dict.fromkeys(sorted(keys), ''))
        archive = job / 'speaker_review.tmp'
        with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
            bundle.write(template, template.name)
            bundle.write(job / 'speaker_samples.json', 'speaker_samples.json')
            for item in observations:
                bundle.write(job / item['sample'], item['sample'])
        archive.replace(job / 'speaker_review.zip')
        print('Speaker review required. Listen to the samples and give the same person the same name in the map.', flush=True)
        return None
    if (not isinstance(mapping, dict) or set(mapping) != keys
            or any(not isinstance(name, str) or not name.strip() for name in mapping.values())):
        raise ValueError('Fill in every label in the speaker map template, keeping its keys unchanged.')
    if not approved.exists():
        save_json(approved, mapping)
    return mapping


def resolve_voices(job, config, mapping, observations):
    path = job / 'voices.json'
    voices = read_json(path) if path.exists() else {}
    best = {}
    for item in observations:
        identity = mapping[item['turn_label']]
        if identity not in best or item['sample_frames'] > best[identity]['sample_frames']:
            best[identity] = item
    if config.get('voice_id'):
        return dict.fromkeys(best, config['voice_id'])
    for identity, item in best.items():
        if identity in voices:
            continue
        sample = job / item['sample']
        print(f'Resolving 60db voice for {identity}...', flush=True)
        for voice_id, message in clone_reference_voice(sample, config['source_language'], sample):
            print(message, flush=True)
            if voice_id:
                voices[identity] = voice_id
                save_json(path, voices)
    return voices


def render_section(section, reference, frames, turns, config, mapping, voices):
    output = section / 'output.mp4'
    soundtrack = section / 'final_soundtrack.wav'
    summary = section / 'output.mp4.json'
    if summary.exists() and soundtrack.is_file() and valid_render(output, frames, exact=True):
        print(f'Reusing completed {section.name}.', flush=True)
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
        return output, soundtrack
    dubbing = section / 'dub_ready.json'
    audio = section / 'speech' / 'replacement_speech.wav'
    if not dubbing.is_file() or not audio.is_file():
        routed = [(first, last, mapping[f'{section.name}/{label}'], text) for first, last, label, text in turns]
        for replacement, _, message in replace_voice(
                reference, None, config['speed'], target_language=config['target_language'],
                prepared_turns=routed, speaker_voices=voices, job_folder=audio.parent):
            print(message, flush=True)
            if replacement:
                save_json(dubbing, {'samples': sf.info(audio).frames})
    command = [sys.executable, '-u', str(ROOT / 'render_chunks.py'),
               '--job', str(section.relative_to(ROOT)), '--video', str(reference),
               '--audio', str(audio), '--parent-pid', str(os.getpid()),
               '--steps', str(config['steps']), '--frames', str(config['batch_frames']),
               '--resume', '--min-seconds', '10', '--target-seconds', '17', '--max-seconds', '25']
    if config['restore_background']:
        command += ['--background-video', str(reference), '--background-gain', str(config['background_gain'])]
    subprocess.run(command, cwd=ROOT, check=True)
    if not valid_render(output, frames, exact=True):
        raise RuntimeError(f'{section.name} has the wrong number of output frames.')
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
    parser.add_argument('--speaker-map', type=Path)
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
        if not (job / 'job.json').exists():
            if not args.video or not args.options:
                parser.error('This job needs its original upload and options.')
            start_job(job, args.video, read_json(Path(args.options)))
        config = read_json(job / 'job.json')
        if ((job / 'completed.json').exists() and (job / 'transcript.json').exists()
                and valid_render(job / 'output.mp4', config['frames'], exact=True)):
            print(f'Completed output: {job / "output.mp4"}', flush=True)
            return
        sections, observations = prepare_sections(job, config)
        mapping = speaker_map(job, config, observations, args.speaker_map)
        if mapping is None:
            return
        voices = resolve_voices(job, config, mapping, observations)
        rendered = [render_section(section, reference, frames, turns, config, mapping, voices)
                    for section, reference, frames, turns in sections]
        save_project_transcript(job, sections)
        output = assemble(job, rendered, config['frames'])
        save_json(job / 'completed.json', {'frames': config['frames'], 'sections': len(sections)})
        print(f'Completed output: {output}', flush=True)


if __name__ == '__main__':
    main()
