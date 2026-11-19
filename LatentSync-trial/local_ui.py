"""Lipsyncstudio UI with uploaded speech and a 60db TTS API connection."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import gradio as gr
from chunk_video import probe
from tts_api import AUTO_LANGUAGE, create_cloned_voice, sdk_client, synthesize, DEFAULT_VOICE, list_voices
from voice_replacement import clone_reference_voice, replace_voice
from translation import LANGUAGES, ORIGINAL_LANGUAGE
from long_video import JOBS, LONG_RENDER_DEFAULTS, clip_lengths_with_cap, long_render_settings
from render_chunks import save_json, valid_render

ROOT = Path(__file__).resolve().parent
WORKER_PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
UI_PYTHON = Path(sys.executable)
SOURCE_LANGUAGES = [
    ('Auto-detect', AUTO_LANGUAGE), ('English', 'en'), ('Hindi', 'hi'), ('Bengali', 'bn'),
    ('Tamil', 'ta'), ('Telugu', 'te'), ('Marathi', 'mr'), ('Gujarati', 'gu'),
    ('Kannada', 'kn'), ('Malayalam', 'ml'), ('Punjabi', 'pa'), ('Odia', 'or'),
    ('Spanish', 'es'), ('French', 'fr'), ('German', 'de'), ('Arabic', 'ar'),
]

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')


def refresh_voices(current):
    try:
        choices = list_voices()
        ids = [identity for _, identity in choices]
        selected = current or (DEFAULT_VOICE if DEFAULT_VOICE in ids else (ids[0] if ids else DEFAULT_VOICE))
        if selected not in ids:
            choices.append((f'Custom voice ({selected})', selected))
        return gr.update(choices=choices, value=selected), f'{len(ids)} voices loaded.'
    except (ValueError, RuntimeError) as exc:
        return gr.skip(), f'{exc} You can still enter a voice ID manually.'


def preview_voice(voice, text, speed):
    try:
        return synthesize(text or 'Hello, this is a preview of your selected voice.', voice, speed)
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from None


def clone_voice(name, sample, language):
    try:
        result = create_cloned_voice(name, sample, language)
    except (ValueError, RuntimeError, OSError) as exc:
        raise gr.Error(str(exc)) from None
    voice_id = str(result.get('id') or result['voice_id'])
    try:
        with sdk_client() as client:
            details = client.get_voice(voice_id)
        details = details.get('data', details)
        synthesis_id = details.get('voice_provider_id') or details.get('voice_id') or details.get('id')
        if synthesis_id and details.get('status') in (None, 'ready'):
            return gr.update(value=str(synthesis_id)), f'60db voice ID: {voice_id}\nStatus: ready'
    except RuntimeError:
        pass
    status = result.get('status') or 'submitted'
    return gr.skip(), f'60db voice ID: {voice_id}\nStatus: {status}. Refresh voices to select it when ready.'


def require_reference_audio(video):
    deadline = time.monotonic() + 10
    while True:
        try:
            stream_types = {stream['codec_type'] for stream in probe(video)['streams']}
            break
        except RuntimeError as exc:
            detail = str(exc).lower()
            # On Windows Gradio may finish copying a re-upload into its cache
            # after returning the upload response.
            if not any(message in detail for message in ('moov atom not found', 'no such file', 'permission denied')):
                raise
            if time.monotonic() >= deadline:
                raise ValueError('The uploaded file is incomplete or could not be read. Re-upload the original video.') from exc
            time.sleep(0.25)
    if 'video' not in stream_types:
        raise ValueError('Upload a file with a video track.')
    if 'audio' not in stream_types:
        raise ValueError(
            'The uploaded video has no audio track. Upload a copy with its original sound '
            'to transcribe, clone voices, or restore background audio.'
        )


def generate_with_speech(source, text, voice, video, audio, duration, steps, frames,
                         restore_background, background_gain, speed, cleanup, target_language=ORIGINAL_LANGUAGE,
                         use_reference_voice=False, source_language=AUTO_LANGUAGE, multiple_speakers=False):
    if not video or not Path(video).is_file():
        raise gr.Error('Upload a reference video.')
    transcript = ''
    if source == 'Transcribe reference video':
        try:
            require_reference_audio(video)
            if use_reference_voice and not multiple_speakers:
                yield None, None, 'Preparing to clone the uploaded speaker...', transcript
                for voice_id, message in clone_reference_voice(video, source_language):
                    if voice_id:
                        voice = voice_id
                    yield gr.skip(), gr.skip(), message, transcript
            for replacement, transcript, message in replace_voice(
                video, voice, speed, duration, target_language,
                multiple_speakers=multiple_speakers, source_language=source_language,
            ):
                if replacement:
                    audio = replacement
                yield gr.skip(), audio if replacement else gr.skip(), message, transcript
        except (ValueError, RuntimeError, OSError) as exc:
            raise gr.Error(str(exc)) from None
    elif source == 'API text to speech':
        yield None, None, 'Generating speech with 60db...', transcript
        audio = preview_voice(voice, text, speed) if (text or '').strip() else None
        if audio is None:
            raise gr.Error('Enter speech text.')
        yield gr.skip(), audio, 'Speech ready. Starting LatentSync...', transcript
    for output, status in generate(video, audio, duration, steps, frames, restore_background, background_gain,
                                   cleanup and source == 'Upload audio'):
        yield output, audio, status, transcript


def convert_media(source, target, duration, audio=False):
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source)]
    if duration > 0:
        command += ['-t', str(duration)]
    if audio:
        command += ['-vn', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le']
    else:
        command += ['-an', '-vf', 'fps=25', '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p']
    command += [str(target)]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError(f'Could not prepare uploaded media: {result.stderr[-2000:]}')


def stop_worker(process):
    if process is None or process.poll() is not None:
        return
    import psutil
    try:
        children = psutil.Process(process.pid).children(recursive=True)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=10)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_long_video(video, job_id, single_speaker, voice_id, source_language,
                   target_language, speed, duration, steps, frames, restore_background, background_gain,
                   *render_values):
    job_id = (job_id or '').strip().lower()
    if job_id and (len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id)):
        raise gr.Error('Enter the 32-character long-video job ID.')
    command = [str(UI_PYTHON), '-u', str(ROOT / 'long_video.py'), '--parent-pid', str(os.getpid())]
    if job_id:
        job = JOBS / job_id
        if not job.is_dir():
            raise gr.Error('Long-video job not found.')
        if not (job / 'job.json').is_file():
            if not video or not Path(video).is_file():
                raise gr.Error('Upload the original video again to finish preparing this job.')
            command += ['--video', video, '--options', str(job / 'options.json')]
    else:
        if not video:
            raise gr.Error('Upload a reference video.')
        try:
            require_reference_audio(video)
        except (ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from None
        try:
            render_settings = long_render_settings(dict(zip(LONG_RENDER_DEFAULTS, render_values)))
        except (TypeError, ValueError) as exc:
            raise gr.Error(str(exc)) from None
        job_id = uuid.uuid4().hex
        job = JOBS / job_id
        job.mkdir(parents=True)
        options = job / 'options.json'
        save_json(options, {
            'single_speaker': single_speaker, 'voice_id': (voice_id or '').strip() if single_speaker else '',
            'source_language': (source_language or '').strip().lower() or AUTO_LANGUAGE,
            'target_language': target_language, 'speed': speed,
            'duration': float(duration or 0), 'steps': int(steps), 'batch_frames': int(frames),
            'restore_background': restore_background, 'background_gain': background_gain,
            **render_settings,
        })
        command += ['--video', video, '--options', str(options)]
    command += ['--job', job_id]
    log_path = job / f'run_{uuid.uuid4().hex}.log'
    process = None
    yield job_id, None, f'Long-video job {job_id}. Keep this ID to resume the job.'
    try:
        with log_path.open('wb') as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            while process.poll() is None:
                with log_path.open('rb') as read:
                    read.seek(max(0, log_path.stat().st_size - 3500))
                    tail = read.read().decode('utf-8', errors='replace')
                yield job_id, gr.skip(), tail or 'Preparing long-video job...'
                time.sleep(2)
        if process.returncode:
            with log_path.open('rb') as read:
                read.seek(max(0, log_path.stat().st_size - 3500))
                tail = read.read().decode('utf-8', errors='replace')
            yield job_id, None, f'Job stopped. Use the same job ID to resume.\n{tail}'
        elif (job / 'completed.json').is_file():
            yield job_id, str(job / 'output.mp4'), f'Long-video output is ready.\nLog: {log_path}'
        else:
            yield job_id, None, f'Job stopped before export. Use job ID {job_id} to resume. Log: {log_path}'
    finally:
        stop_worker(process)


def generate(video, audio, duration, steps, frames, restore_background=True, background_gain=1.0, cleanup=False):
    if not video or not Path(video).is_file():
        raise gr.Error('Upload a reference video.')
    if not audio or not Path(audio).is_file():
        raise gr.Error('Upload speech audio, or record it with the microphone.')
    job = ROOT / 'local_jobs' / uuid.uuid4().hex
    job.mkdir(parents=True)
    yield None, f'Preparing video and audio for render job {job.name}…'
    try:
        convert_media(video, job / 'video.mp4', duration)
        convert_media(audio, job / 'audio.wav', duration, audio=True)
        background = None
        if restore_background:
            original = Path(video)
            background = job / f'original_reference{original.suffix}'
            shutil.copyfile(original, background)
        save_json(job / 'render_options.json', {
            'frames': int(frames), 'steps': int(steps), 'cleanup': bool(cleanup),
            'background': background.name if background else None, 'background_gain': background_gain,
        })
    except (OSError, RuntimeError, ValueError) as exc:
        raise gr.Error(str(exc)) from None
    yield from run_standard_render(job)


def resume_standard_render(job_id):
    job_id = (job_id or '').strip().lower()
    if len(job_id) != 32 or any(ch not in '0123456789abcdef' for ch in job_id):
        raise gr.Error('Enter the 32-character render job ID shown in the render status.')
    job = ROOT / 'local_jobs' / job_id
    if not job.is_dir():
        raise gr.Error('Render job not found.')
    yield from run_standard_render(job)


def run_standard_render(job):
    log_path = job / 'render.log'
    if log_path.exists():
        log_path = job / f'render_retry_{uuid.uuid4().hex}.log'
    output = job / 'output.mp4'
    summary = job / 'output.mp4.json'
    process = None
    try:
        if summary.is_file():
            stats = json.loads(summary.read_text(encoding='utf-8'))
            if valid_render(output, stats['total_frames'], exact=True):
                yield str(output), f'Render job {job.name} is already complete. Opened the saved video.'
                return
        options = job / 'render_options.json'
        if not options.is_file():
            raise RuntimeError('This older job has no saved render settings. Browser resume is available for new jobs.')
        settings = json.loads(options.read_text(encoding='utf-8'))
        relative = job.relative_to(ROOT)
        command = [str(UI_PYTHON), '-u', str(ROOT / 'render_chunks.py'),
                   '--video', (relative / 'video.mp4').as_posix(),
                   '--audio', (relative / 'audio.wav').as_posix(),
                   '--job', relative.as_posix(),
                   '--frames', str(settings['frames']), '--steps', str(settings['steps']),
                   '--parent-pid', str(os.getpid()), '--resume', '--preview-clips']
        if settings['cleanup']:
            command += ['--clean-speech']
        command += ['--background-gain', str(settings['background_gain'])]
        if settings['background']:
            command += ['--background-video', str(job / settings['background'])]
        with log_path.open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'},
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            while process.poll() is None:
                tail = log_path.read_text(encoding='utf-8', errors='replace')[-2200:]
                yield gr.skip(), f'Render job {job.name}\n\n{tail}'
                time.sleep(2)
        details = log_path.read_text(encoding='utf-8', errors='replace')
        if process.returncode or not output.is_file():
            raise RuntimeError(f'Render stopped. Use Resume standard render with job ID {job.name}.\n'
                               f'Log: {log_path}\n\n{details[-3500:]}')
        stats = json.loads(summary.read_text(encoding='utf-8'))
        yield str(output), (
            f"Render job {job.name} complete. {stats['chunks']} chunks joined, {stats['duration_seconds']:.2f}s total. "
            f"Peak GPU memory reserved: {stats['peak_cuda_reserved_mib']} MiB.\n"
            f"RealtimeVAD cuts. Background restored: {stats['background_restored']}.\n"
            f"Saved to: {output}\nLog: {log_path}"
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from None
    finally:
        stop_worker(process)


def adjust_clip_controls(minimum, target, maximum):
    if minimum is None or target is None:
        return gr.skip(), gr.skip()
    return clip_lengths_with_cap(minimum, target, maximum)


def build_ui():
    with gr.Blocks(title='Lipsyncstudio') as demo:
        gr.Markdown('# Lipsyncstudio')
        with gr.Row():
            with gr.Column():
                video = gr.File(label='Reference video', file_types=['video'], type='filepath')
                with gr.Row():
                    source_language = gr.Dropdown(
                        SOURCE_LANGUAGES, value=AUTO_LANGUAGE, allow_custom_value=True,
                        label='Language spoken in the video',
                    )
                    target_language = gr.Dropdown(
                        LANGUAGES, value='Hindi', allow_custom_value=True, label='Translate video speech into',
                    )
                long_single = gr.Checkbox(label='Only one speaker in the entire video', value=False)
                long_voice = gr.Textbox(label='Existing 60db synthesis voice ID (optional, single speaker only)')
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label='Speaking speed')
                restore_background = gr.Checkbox(value=True, label='Restore original background music / noise')
                background_gain = gr.Slider(0, 1, value=1, step=0.05, label='Background volume')

            with gr.Column():
                gr.Markdown('### Sections and render clips')
                with gr.Row():
                    section_seconds = gr.Number(
                        value=LONG_RENDER_DEFAULTS['section_seconds'], minimum=10, maximum=600, precision=0,
                        label='Transcription section (seconds)',
                    )
                    long_duration = gr.Number(
                        value=0, minimum=0, label='Process only the first N seconds',
                        info='Use 0 for the full video.',
                    )
                with gr.Row():
                    clip_min = gr.Number(
                        value=LONG_RENDER_DEFAULTS['clip_min_seconds'], minimum=4, maximum=25,
                        label='Minimum clip (seconds)',
                    )
                    clip_target = gr.Number(
                        value=LONG_RENDER_DEFAULTS['clip_target_seconds'], minimum=4, maximum=25,
                        label='Target clip (seconds)',
                    )
                    clip_max = gr.Slider(
                        4, 25, step=0.04, value=LONG_RENDER_DEFAULTS['clip_max_seconds'],
                        label='Maximum clip (seconds)', info='Hard limit including overlap.',
                    )
                overlap = gr.Slider(
                    0, 1.5, step=0.04, value=LONG_RENDER_DEFAULTS['overlap_seconds'],
                    label='Clip overlap (seconds per side)',
                )
                with gr.Row():
                    boundary_search = gr.Slider(
                        0, 30, step=1, value=LONG_RENDER_DEFAULTS['boundary_search_seconds'],
                        label='Section pause search (seconds per side)', info='Use 0 for fixed section cuts.',
                    )
                    vad_pause = gr.Slider(
                        0.1, 2, step=0.05, value=LONG_RENDER_DEFAULTS['vad_pause_seconds'],
                        label='Minimum VAD pause (seconds)',
                    )
                gr.Markdown('### Compute')
                with gr.Row():
                    steps = gr.Slider(2, 50, value=20, step=1, label='Inference steps')
                    frames = gr.Dropdown([4, 8, 16], value=16, label='Frames per batch',
                                         info='Choose 4 or 8 to reduce VRAM use.')
                    cpu_threads = gr.Slider(
                        1, 16, step=1, value=LONG_RENDER_DEFAULTS['cpu_threads'], label='PyTorch CPU threads',
                    )

        long_job = gr.Textbox(label='Job ID', info='Leave blank for a new job. Resume uses the saved settings.')
        long_button = gr.Button('Start or resume job', variant='primary')
        with gr.Row():
            long_status = gr.Textbox(label='Job progress', interactive=False, lines=8)
            output = gr.Video(label='Generated video', interactive=False, show_download_button=True)

        clip_max.input(adjust_clip_controls, inputs=[clip_min, clip_target, clip_max],
                       outputs=[clip_min, clip_target], queue=False, api_name=False)
        long_button.click(
            run_long_video, inputs=[video, long_job, long_single, long_voice, source_language,
                                    target_language, speed, long_duration, steps, frames, restore_background, background_gain,
                                    section_seconds, boundary_search, vad_pause, clip_min, clip_target, clip_max, overlap,
                                    cpu_threads],
            outputs=[long_job, output, long_status], concurrency_id='pipeline', api_name='long_video',
            show_progress='minimal',
        )
    return demo


if __name__ == '__main__':
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name='127.0.0.1', server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7861')), inbrowser=False,
        allowed_paths=[str(ROOT / 'trial_inputs'), str(ROOT / 'local_jobs')],
    )
