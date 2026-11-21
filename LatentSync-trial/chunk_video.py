"""Split media into bounded, overlapping clips and record exact join boundaries.

This prepares files only; it does not load LatentSync or launch a server.
"""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import uuid

FPS = 25
RATE = 16000


def run(command):
    if Path(command[0]).stem.lower() == 'ffmpeg':
        command = [command[0], '-nostdin', *command[1:]]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    return result


def probe(path):
    return json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]).stdout)


def count_frames(path):
    """Exact video frame count; raises if the file is unreadable (e.g. missing moov atom)."""
    stream = next(s for s in probe(path)['streams'] if s['codec_type'] == 'video')
    count = stream.get('nb_frames')
    if count and count != 'N/A':
        return int(count)
    result = run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                  '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(path)])
    return int(result.stdout.strip())


def silence_intervals(audio, threshold=-35, minimum=0.2):
    """Detect quiet intervals, not linguistic sentence boundaries or model VAD."""
    result = run(['ffmpeg', '-hide_banner', '-nostats', '-i', str(audio), '-af',
                  f'silencedetect=noise={threshold}dB:d={minimum}', '-f', 'null', '-'])
    intervals, start = [], None
    for line in result.stderr.splitlines():
        match = re.search(r'silence_(start|end):\s*([\d.]+)', line)
        if match:
            if match[1] == 'start':
                start = float(match[2])
            elif start is not None:
                intervals.append((start, float(match[2])))
                start = None
    return intervals


def plan_chunks(total_frames, minimum=5, maximum=7, target=6, overlap=0.2, silences=()):
    """Bound actual extracted clip lengths, including context, in frame units."""
    high = math.floor(maximum * FPS + 1e-8)
    low = min(math.ceil(minimum * FPS - 1e-8), high)
    padding = round(overlap * FPS)
    if not 0 < low <= high or not minimum <= target <= maximum:
        raise ValueError('Use 0 < min <= target <= max.')
    if padding < 0 or 2 * padding >= low:
        raise ValueError('Overlap must be nonnegative and less than half the minimum clip length.')
    if total_frames < 1:
        raise ValueError('The input contains no video frames.')
    desired = max(1, round(total_frames / (target * FPS)))
    feasible = []
    for count in range(1, math.ceil(total_frames / max(1, low - 2 * padding)) + 2):
        context = [padding * ((i > 0) + (i < count - 1)) for i in range(count)]
        minimums = [max(1, low - value) for value in context]
        maximums = [high - value for value in context]
        if sum(minimums) <= total_frames <= sum(maximums):
            feasible.append((count, minimums, maximums))
    if not feasible:
        if total_frames <= high:
            feasible = [(1, [total_frames], [total_frames])]
        else:
            # The minimum is a preference when the timeline cannot fit under the hard cap.
            count = math.ceil((total_frames - 2 * padding) / (high - 2 * padding))
            maximums = [high - padding * ((i > 0) + (i < count - 1)) for i in range(count)]
            minimums = [min(capacity, total_frames // count) for capacity in maximums]
            feasible = [(count, minimums, maximums)]
    count, minimums, maximums = min(feasible, key=lambda item: (abs(item[0] - desired), item[0]))
    boundaries, reasons = [0], []
    for index in range(count - 1):
        previous = boundaries[-1]
        earliest = max(previous + minimums[index], total_frames - sum(maximums[index + 1:]))
        latest = min(previous + maximums[index], total_frames - sum(minimums[index + 1:]))
        ideal = min(latest, max(earliest, round(total_frames * (index + 1) / count)))
        candidates = []
        for start, end in silences:
            left = max(earliest, math.ceil(start * FPS))
            right = min(latest, math.floor(end * FPS))
            if left <= right:
                candidates.append(min(right, max(left, ideal)))
        boundary = min(candidates, key=lambda value: abs(value - ideal)) if candidates else ideal
        boundaries.append(boundary)
        reasons.append('silence' if candidates else 'balanced_length')
    boundaries.append(total_frames)
    chunks = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        clip_start = max(0, start - padding)
        clip_end = min(total_frames, end + padding)
        chunks.append({
            'index': index + 1,
            'core_start_frame': start, 'core_end_frame': end,
            'clip_start_frame': clip_start, 'clip_end_frame': clip_end,
            'clip_duration_seconds': (clip_end - clip_start) / FPS,
            'core_start_seconds': start / FPS, 'core_end_seconds': end / FPS,
            'keep_from_frame': start - clip_start, 'keep_to_frame': end - clip_start,
            'keep_frame_count': end - start,
            'boundary_after': reasons[index] if index < len(reasons) else 'end',
        })
    return chunks


def chunk_video(video, audio=None, output_dir=None, minimum=5, maximum=7, target=6,
                overlap=0.4, prefer_pauses=True, pause_intervals=None):
    video = Path(video).resolve()
    metadata = probe(video)
    video_stream = next((s for s in metadata['streams'] if s['codec_type'] == 'video'), None)
    if video_stream is None:
        raise ValueError('Input must contain a video stream.')
    video_duration = float(video_stream.get('duration') or metadata['format']['duration'])
    speech_source = Path(audio).resolve() if audio else (
        video if any(s['codec_type'] == 'audio' for s in metadata['streams']) else None)
    duration = video_duration
    if audio:
        speech_metadata = probe(speech_source)
        if not any(s['codec_type'] == 'audio' for s in speech_metadata['streams']):
            raise ValueError('The supplied speech file has no audio stream.')
        duration = float(speech_metadata['format']['duration'])
        if duration > video_duration + 1 / FPS:
            raise ValueError('Speech is longer than the video. Extend the reference video before chunking; speech will not be truncated.')
    total_frames = max(1, round(duration * FPS))
    duration = total_frames / FPS
    # Validate before making output files.
    plan_chunks(total_frames, minimum, maximum, target, overlap)
    output = Path(output_dir or Path(__file__).resolve().parent / 'chunk_jobs' / uuid.uuid4().hex).resolve()
    output.mkdir(parents=True, exist_ok=False)
    normalized = output / 'timeline.mp4'
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video),
         '-an', '-vf', 'fps=25,tpad=stop_mode=clone:stop_duration=0.08',
         '-frames:v', str(total_frames), '-c:v', 'libx264', '-preset', 'fast',
         '-crf', '18', '-pix_fmt', 'yuv420p', str(normalized)])
    full_audio = None
    silences = []
    if speech_source:
        full_audio = output / 'speech_full.wav'
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(speech_source),
             '-vn', '-af', f'aresample={RATE},apad,atrim=end_sample={total_frames * (RATE // FPS)}', '-ar', str(RATE), '-ac', '1',
             '-c:a', 'pcm_s16le', str(full_audio)])
        if pause_intervals is not None:
            silences = pause_intervals
        elif prefer_pauses:
            silences = silence_intervals(full_audio)
    chunks = plan_chunks(total_frames, minimum, maximum, target, overlap, silences)
    for chunk in chunks:
        name = f"chunk_{chunk['index']:03d}"
        start = chunk['clip_start_frame'] / FPS
        frames = chunk['clip_end_frame'] - chunk['clip_start_frame']
        clip_video = output / (name + '.mp4')
        # Write to a temp name, verify it is complete, then rename. An interrupted or
        # disk-starved write never leaves a truncated final clip (which would fail later
        # with "moov atom not found" and be reused forever on resume).
        clip_partial = output / (name + '.partial.mp4')
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(start),
             '-i', str(normalized), '-an', '-frames:v', str(frames), '-c:v', 'libx264',
             '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', str(clip_partial)])
        if count_frames(clip_partial) != frames:
            raise RuntimeError(f'{name} produced {count_frames(clip_partial)} frames, expected {frames}.')
        clip_partial.replace(clip_video)
        chunk['video'] = clip_video.name
        chunk['audio'] = None
        if full_audio:
            clip_audio = output / (name + '.wav')
            clip_audio_partial = output / (name + '.partial.wav')
            first_sample = chunk['clip_start_frame'] * (RATE // FPS)
            last_sample = chunk['clip_end_frame'] * (RATE // FPS)
            run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                 '-i', str(full_audio), '-af',
                 f'atrim=start_sample={first_sample}:end_sample={last_sample},asetpts=PTS-STARTPTS',
                 '-c:a', 'pcm_s16le', str(clip_audio_partial)])
            probe(clip_audio_partial)  # raises if the wav did not finish writing
            clip_audio_partial.replace(clip_audio)
            chunk['audio'] = clip_audio.name
        print(f"{name}: {chunk['clip_duration_seconds']:.2f}s with context; keep timeline "
              f"{chunk['core_start_seconds']:.2f}-{chunk['core_end_seconds']:.2f}s", flush=True)
    manifest = {'version': 1, 'fps': FPS, 'sample_rate': RATE, 'total_frames': total_frames,
                'duration_seconds': duration, 'source_video': str(video),
                'source_audio': str(speech_source) if speech_source else None,
                'full_audio': full_audio.name if full_audio else None,
                'overlap_per_side_seconds': round(overlap * FPS) / FPS,
                'length_limits_include_overlap': True,
                'pause_detection': 'ffmpeg_silencedetect' if prefer_pauses and full_audio else 'off',
                'chunks': chunks}
    path = output / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Manifest: {path}', flush=True)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, help='Input reference video')
    parser.add_argument('--audio', help='Speech WAV (otherwise use video audio if present)')
    parser.add_argument('--output-dir', help='New folder for this job; must not exist yet')
    parser.add_argument('--min-seconds', type=float, default=5)
    parser.add_argument('--max-seconds', type=float, default=7)
    parser.add_argument('--target-seconds', type=float, default=6)
    parser.add_argument('--overlap', type=float, default=0.4, help='Context on each side, included in the 5–7 second limit')
    parser.add_argument('--no-pauses', action='store_true', help='Use balanced cuts without silence detection')
    args = parser.parse_args()
    for executable in ('ffmpeg', 'ffprobe'):
        if not shutil.which(executable):
            parser.error(f'{executable} is not available on PATH')
    try:
        chunk_video(args.video, args.audio, args.output_dir, args.min_seconds, args.max_seconds,
                    args.target_seconds, args.overlap, not args.no_pauses)
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f'Chunking failed: {exc}\n')
