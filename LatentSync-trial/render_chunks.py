"""Prepare overlapping clips, render them sequentially, and join one final video."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from settings import WORKER_PYTHON
from speech_cleanup import clean_speech

from chunk_video import chunk_video, probe, run
from job_logging import log_event
from audio_pipeline import speech_timestamps, extract_background, build_soundtrack
from worker_client import RenderPool, detect_gpu_count

ROOT = Path(__file__).resolve().parent


def frame_count(path):
    stream = next(s for s in probe(path)['streams'] if s['codec_type'] == 'video')
    count = stream.get('nb_frames')
    if count and count != 'N/A':
        return int(count)
    result = run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                  '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(path)])
    return int(result.stdout.strip())


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def valid_render(path, required_frames, exact=False):
    if not path.is_file():
        return False
    try:
        count = frame_count(path)
    except (RuntimeError, ValueError, StopIteration):
        return False
    return count == required_frames if exact else count >= required_frames


def join_chunks(manifest_path, rendered, output, soundtrack=None):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if len(rendered) != len(manifest['chunks']):
        raise ValueError('Every chunk must finish rendering before joining.')
    retained = output.parent / 'retained'
    retained.mkdir(exist_ok=True)
    clips = []
    for chunk, source in zip(manifest['chunks'], rendered):
        if frame_count(source) < chunk['keep_to_frame']:
            raise RuntimeError(f'Rendered chunk {chunk["index"]} is missing required frames.')
        target = retained / f"chunk_{chunk['index']:03d}.mp4"
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
             '-an', '-vf', f"trim=start_frame={chunk['keep_from_frame']}:end_frame={chunk['keep_to_frame']},setpts=PTS-STARTPTS",
             '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', str(target)])
        if frame_count(target) != chunk['keep_frame_count']:
            raise RuntimeError(f'Trimmed chunk {chunk["index"]} has the wrong frame count.')
        clips.append(target)
    # All files are inside this job; relative filenames also handle Windows paths with spaces.
    listing = retained / 'concat.txt'
    listing.write_text(''.join(f"file '{clip.name}'\n" for clip in clips), encoding='utf-8')
    audio = soundtrack or manifest_path.parent / manifest['full_audio']
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
         '-i', str(listing), '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
         '-c:v', 'copy', '-c:a', 'aac', '-aac_coder', 'fast', '-t', str(manifest['duration_seconds']),
         '-movflags', '+faststart', str(output)])
    if frame_count(output) != manifest['total_frames']:
        raise RuntimeError('Final video frame count does not match the planned timeline.')
    return manifest


def resolve_gpus(requested):
    """Which GPU indices to render across. requested<=0 means every visible GPU."""
    available = detect_gpu_count()
    if available <= 0:
        return [0]   # no CUDA visible from here; the worker will error clearly if truly absent
    count = min(requested, available) if requested and requested > 0 else available
    return list(range(max(1, count)))


def render_pending_cold(pending, renders, manifest_path, args, total):
    """Fallback path: one cold subprocess per clip (previous behavior), sequential."""
    stats_by_index = {}
    for chunk in pending:
        index = chunk['index']
        output = renders / f'chunk_{index:03d}.mp4'
        log_path = renders / f'chunk_{index:03d}.log'
        stats_path = Path(str(output) + '.json')
        log_event(f"Rendering chunk {index}/{total}: {chunk['clip_duration_seconds']:.2f}s "
                  f"(section timeline {chunk['core_start_seconds']:.2f}–{chunk['core_end_seconds']:.2f}s)")
        chunk_started = time.monotonic()
        command = [str(WORKER_PYTHON), '-u', str(ROOT / 'trial_inference.py'),
                   '--video', (manifest_path.parent / chunk['video']).resolve().relative_to(ROOT).as_posix(),
                   '--audio', (manifest_path.parent / chunk['audio']).resolve().relative_to(ROOT).as_posix(),
                   '--output', output.resolve().relative_to(ROOT).as_posix(),
                   '--frames', str(args.frames), '--steps', str(args.steps), '--cpu-threads', str(args.cpu_threads)]
        with log_path.open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not output.is_file():
            raise RuntimeError(f"Chunk {index} failed (timeline {chunk['core_start_seconds']:.2f}–{chunk['core_end_seconds']:.2f}s). "
                               f'Log: {log_path}\n' +
                               log_path.read_text(encoding='utf-8', errors='replace')[-3500:])
        stats_by_index[index] = json.loads(stats_path.read_text(encoding='utf-8'))
        log_event(f'Chunk {index} saved in {time.monotonic() - chunk_started:.1f}s. '
                  'Worker exited and released its memory.')
    return stats_by_index


def render_pending_pool(pending, renders, manifest_path, args, total):
    """Render pending clips in parallel across one resident worker per GPU."""
    gpus = resolve_gpus(args.gpus)
    gpus = gpus[:max(1, min(len(gpus), len(pending)))]   # no idle workers past clip count
    index_by_id, clips = {}, []
    for chunk in pending:
        index = chunk['index']
        clip_id = f'{index:03d}'
        index_by_id[clip_id] = index
        clips.append({'id': clip_id,
                      'video': manifest_path.parent / chunk['video'],
                      'audio': manifest_path.parent / chunk['audio'],
                      'output': renders / f'chunk_{index:03d}.mp4',
                      'frames': args.frames, 'steps': args.steps,
                      'guidance_scale': args.guidance_scale})
    done = {'n': 0}

    def on_done(clip_id, clip_stats):
        index = index_by_id[clip_id]
        save_json(Path(str(renders / f'chunk_{index:03d}.mp4') + '.json'), clip_stats)
        done['n'] += 1
        log_event(f"Chunk {index} done ({done['n']}/{len(pending)} this run) in "
                  f"{clip_stats['seconds']:.1f}s on GPU, peak {clip_stats['peak_cuda_reserved_mib']} MiB.")

    started = time.monotonic()
    with RenderPool(renders / '.engine_queue', gpus, version=args.version, offload=args.offload,
                    cpu_threads=args.cpu_threads, parent_pid=args.parent_pid, log_dir=renders) as pool:
        log_event(f'Render pool ready: {len(pool.ready_workers)} GPU worker(s) for LatentSync '
                  f'{args.version}. Rendering {len(pending)} clips across {len(gpus)} GPU(s).')
        results = pool.render_all(clips, on_done=on_done)
    log_event(f'Rendered {len(pending)} clips in {time.monotonic() - started:.1f}s '
              f'across {len(gpus)} GPU(s).')
    return {index_by_id[clip_id]: clip_stats for clip_id, clip_stats in results.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--background-video', help='Original upload with its soundtrack still intact')
    parser.add_argument('--background-gain', type=float, default=1.0)
    parser.add_argument('--clean-speech', action='store_true', help='Reduce steady noise in replacement speech using SoX')
    parser.add_argument('--parent-pid', type=int, help='Owning UI process; workers stop if it exits')
    parser.add_argument('--resume', action='store_true', help='Reuse validated artifacts for an unchanged job')
    parser.add_argument('--preview-clips', action='store_true', help='Save playable clip previews in resumable jobs')
    parser.add_argument('--min-seconds', type=float, default=4)
    parser.add_argument('--max-seconds', type=float, default=6)
    parser.add_argument('--target-seconds', type=float, default=5)
    parser.add_argument('--overlap', type=float, default=0.4)
    parser.add_argument('--vad-pause-seconds', type=float, default=0.3)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--engine-mode', choices=['resident', 'cold'], default='resident',
                        help="'resident' loads the model once per section; 'cold' reloads per clip")
    parser.add_argument('--version', default='1.6', help='LatentSync engine version (1.5 or 1.6)')
    parser.add_argument('--offload', default='auto', choices=['auto', 'resident', 'swap'])
    parser.add_argument('--guidance-scale', type=float, default=None)
    parser.add_argument('--gpus', type=int, default=0,
                        help='Number of GPUs to render across (0 = all available)')
    args = parser.parse_args()
    if args.parent_pid:
        from worker_lifecycle import watch_parent
        watch_parent(args.parent_pid)
    if not 0 <= args.background_gain <= 1:
        parser.error('--background-gain must be between 0 and 1')
    if not 0.1 <= args.vad_pause_seconds <= 2 or not 1 <= args.cpu_threads <= 16:
        parser.error('Use 0.1 to 2 seconds for the VAD pause and 1 to 16 CPU threads.')
    started = time.monotonic()
    job = Path(args.job)
    job.mkdir(parents=True, exist_ok=True)
    video, audio = Path(args.video), Path(args.audio)
    cleanup_applied = False
    if args.clean_speech:
        audio, cleanup_applied = clean_speech(audio, job)
    audio_duration = float(probe(audio)['format']['duration'])
    video_duration = float(probe(video)['format']['duration'])
    if video_duration + 1 / 25 < audio_duration:
        log_event('Extending the reference video by looping it to cover the full speech.')
        extended = job / 'extended_reference.mp4'
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-stream_loop', '-1',
             '-i', str(video), '-an', '-t', str(audio_duration), '-vf', 'fps=25',
             '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', str(extended)])
        video = extended
    background = None
    marker = job / 'background_ready.json'
    background_saved = args.resume and marker.exists()
    if background_saved:
        saved = json.loads(marker.read_text(encoding='utf-8'))
        background = job / saved if saved else None
    if background and not background.is_file() and not args.background_video:
        raise RuntimeError('The saved background track is missing and the original is unavailable.')
    if args.background_video and (not background_saved or (background and not background.is_file())):
        background_started = time.monotonic()
        log_event('Background separation started.')
        background = extract_background(args.background_video, job, audio_duration, args.cpu_threads)
        log_event(f'Background preparation finished in {time.monotonic() - background_started:.1f}s.')
        if args.resume:
            save_json(marker, background.relative_to(job).as_posix() if background else None)
    elif background_saved:
        log_event('Reused saved background preparation.')
    plan = job / 'chunk_plan.json'
    if args.resume and plan.exists():
        manifest_path = job / json.loads(plan.read_text(encoding='utf-8'))
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        log_event('Reused saved render clip plan.')
    else:
        vad_started = time.monotonic()
        log_event('Finding confirmed speech pauses with RealtimeVAD.')
        timing = speech_timestamps(audio, job, args.vad_pause_seconds)
        log_event(f'RealtimeVAD finished in {time.monotonic() - vad_started:.1f}s.')
        clips_started = time.monotonic()
        log_event(f'Preparing clips near {args.target_seconds:g}s with {args.overlap:g}s context per side.')
        chunk_dir = job / 'chunks'
        if args.resume and chunk_dir.exists():
            chunk_dir = job / f'chunks_{uuid.uuid4().hex}'
        manifest_path = chunk_video(video, audio, chunk_dir, minimum=args.min_seconds,
                                    maximum=args.max_seconds, target=args.target_seconds,
                                    overlap=args.overlap, prefer_pauses=False, pause_intervals=timing['pause_intervals'])
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['pause_detection'] = 'RealtimeVAD_confirmed_speech_ends'
        manifest['speech_intervals'] = timing['speech_intervals']
        for chunk in manifest['chunks']:
            if chunk['boundary_after'] == 'silence':
                chunk['boundary_after'] = 'vad_pause'
            if args.resume and not args.preview_clips:
                continue
            preview = manifest_path.parent / f"chunk_{chunk['index']:03d}_with_audio.mp4"
            run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                 '-i', str(manifest_path.parent / chunk['video']),
                 '-i', str(manifest_path.parent / chunk['audio']), '-map', '0:v:0', '-map', '1:a:0',
                 '-c:v', 'copy', '-c:a', 'aac', '-aac_coder', 'fast',
                 '-t', str(chunk['clip_duration_seconds']), str(preview)])
            chunk['preview_video'] = preview.name
        save_json(manifest_path, manifest)
        if args.resume:
            save_json(plan, manifest_path.relative_to(job.resolve()).as_posix())
        log_event(f'Prepared {len(manifest["chunks"])} clips in {time.monotonic() - clips_started:.1f}s.')
    renders = job / 'renders'
    renders.mkdir(exist_ok=True)
    total = len(manifest['chunks'])

    # Reuse validated chunks; collect the rest to render (in parallel across GPUs).
    reused, pending = {}, []
    for chunk in manifest['chunks']:
        index = chunk['index']
        output = renders / f'chunk_{index:03d}.mp4'
        stats_path = Path(str(output) + '.json')
        if args.resume and stats_path.is_file() and valid_render(output, chunk['keep_to_frame']):
            try:
                saved = json.loads(stats_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                saved = {}
            # Old cold renders have no 'version' and were LatentSync 1.5.
            if (saved.get('frames_per_batch') == args.frames and saved.get('steps') == args.steps
                    and saved.get('version', '1.5') == args.version):
                reused[index] = saved
                log_event(f'Reusing completed chunk {index}/{total}.')
                continue
        stats_path.unlink(missing_ok=True)
        pending.append(chunk)

    fresh = {}
    if pending:
        if args.engine_mode == 'cold':
            fresh = render_pending_cold(pending, renders, manifest_path, args, total)
        else:
            fresh = render_pending_pool(pending, renders, manifest_path, args, total)

    rendered, stats = [], []
    for chunk in manifest['chunks']:
        index = chunk['index']
        output = renders / f'chunk_{index:03d}.mp4'
        clip_stats = reused.get(index) or fresh.get(index)
        if clip_stats is None or not valid_render(output, chunk['keep_to_frame']):
            raise RuntimeError(f'Chunk {index} is missing or incomplete after rendering.')
        rendered.append(output)
        stats.append(clip_stats)

    mux_started = time.monotonic()
    log_event('Trimming overlap and joining the video with the continuous speech track.')
    output = job / 'output.mp4'
    soundtrack = build_soundtrack(manifest_path.parent / manifest['full_audio'], background,
                                  job / 'final_soundtrack.wav', manifest['duration_seconds'], args.background_gain)
    join_chunks(manifest_path, rendered, output, soundtrack)
    log_event(f'Soundtrack mix, clip join, and mux finished in {time.monotonic() - mux_started:.1f}s.')
    summary = {'output': str(output), 'seconds': round(time.monotonic() - started, 2),
               'peak_cuda_reserved_mib': max(s['peak_cuda_reserved_mib'] for s in stats),
               'chunks': len(rendered), 'duration_seconds': manifest['duration_seconds'],
               'total_frames': manifest['total_frames'], 'chunk_stats': stats,
               'vad_engine': 'RealtimeVAD', 'background_restored': background is not None,
               'sox_cleanup_requested': args.clean_speech, 'sox_cleanup_applied': cleanup_applied,
               'background_gain': args.background_gain, 'soundtrack': str(soundtrack)}
    save_json(Path(str(output) + '.json'), summary)
    print('CHUNKED_RENDER_SUCCESS ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    main()
