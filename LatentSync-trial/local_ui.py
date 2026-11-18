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
from tts_api import create_cloned_voice, sdk_client, synthesize, DEFAULT_VOICE, list_voices
from voice_replacement import clone_reference_voice, replace_voice
from translation import LANGUAGES, ORIGINAL_LANGUAGE
from long_video import JOBS
from render_chunks import save_json, valid_render

ROOT = Path(__file__).resolve().parent
WORKER_PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
UI_PYTHON = Path(sys.executable)

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
    if not any(stream['codec_type'] == 'audio' for stream in probe(video)['streams']):
        raise ValueError(
            'The uploaded video has no audio track. Upload a copy with its original sound '
            'to transcribe, clone voices, or restore background audio.'
        )


def generate_with_speech(source, text, voice, video, audio, duration, steps, frames,
                         restore_background, background_gain, speed, cleanup, target_language=ORIGINAL_LANGUAGE,
                         use_reference_voice=False, source_language='en', multiple_speakers=False):
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


def run_long_video(video, job_id, mapping, single_speaker, voice_id, source_language,
                   target_language, speed, duration, steps, frames, restore_background, background_gain):
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
        if not video or not Path(video).is_file():
            raise gr.Error('Upload a reference video.')
        try:
            require_reference_audio(video)
        except (ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from None
        if not (source_language or '').strip():
            raise gr.Error('Enter the language spoken in the video.')
        job_id = uuid.uuid4().hex
        job = JOBS / job_id
        job.mkdir(parents=True)
        options = job / 'options.json'
        save_json(options, {
            'single_speaker': single_speaker, 'voice_id': (voice_id or '').strip() if single_speaker else '',
            'source_language': source_language.strip(), 'target_language': target_language, 'speed': speed,
            'duration': float(duration or 0), 'steps': int(steps), 'batch_frames': int(frames),
            'restore_background': restore_background, 'background_gain': background_gain,
        })
        command += ['--video', video, '--options', str(options)]
    command += ['--job', job_id]
    if mapping:
        command += ['--speaker-map', mapping]
    log_path = job / f'run_{uuid.uuid4().hex}.log'
    process = None
    yield job_id, None, None, f'Long-video job {job_id}. Keep this ID to resume the job.'
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
                yield job_id, gr.skip(), gr.skip(), tail or 'Preparing long-video job...'
                time.sleep(2)
        if process.returncode:
            with log_path.open('rb') as read:
                read.seek(max(0, log_path.stat().st_size - 3500))
                tail = read.read().decode('utf-8', errors='replace')
            yield job_id, None, None, f'Job stopped. Use the same job ID to resume.\n{tail}'
        elif (job / 'completed.json').is_file():
            yield job_id, None, str(job / 'output.mp4'), 'Long-video output is ready.'
        else:
            yield job_id, str(job / 'speaker_review.zip'), None, (
                'Download the review bundle. Match the samples to people, fill in the speaker map JSON, '
                'then upload that JSON and run again with the same job ID.'
            )
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


def build_ui():
    with gr.Blocks(title='Lipsyncstudio') as demo:
        gr.Markdown('# Lipsyncstudio - LatentSync 1.5')
        gr.Markdown('Transcribe the original video with 60db, translate it into your target language, select a replacement voice, and lip-sync in roughly five-second chunks. Replacement utterances are fitted to their original time slots, preserving pauses. The final video combines the complete speech with background sound separated from your original video. Short references repeat to cover the speech.')
        with gr.Row():
            with gr.Column():
                source = gr.Radio(['Transcribe reference video', 'API text to speech', 'Upload audio'], value='Transcribe reference video', label='Speech source')
                target_language = gr.Dropdown(LANGUAGES, value='Hindi', allow_custom_value=True, label='Translate video speech into', info='Applies to transcription mode. Enter another language or choose Original to change only the voice. Choose a replacement voice that supports your target language.')
                with gr.Group() as reference_voice_group:
                    use_reference_voice = gr.Checkbox(
                        value=True, label='Dub in the uploaded speaker’s voice',
                        info='Sends up to the first 10 seconds of audio to 60db. Use clear speech from one speaker. To reuse a dashboard clone, turn this off and select it below.',
                    )
                    source_language = gr.Textbox(value='en', label='Language spoken in the video', info='Enter the source language code, such as en or hi.')
                    multiple_speakers = gr.Checkbox(
                        value=False, label='Clone each speaker separately',
                        info='Use this for dialogue with multiple speakers. 60db identifies the speakers and the app creates one clone per speaker.',
                    )
                with gr.Group() as text_group:
                    text = gr.Textbox(visible=False, label='Text to speak', lines=6, placeholder='Enter the speech for your video...')
                    voice = gr.Dropdown(label='60db voice', choices=[('Current voice', DEFAULT_VOICE)], value=DEFAULT_VOICE, allow_custom_value=True, info='Select a discovered voice or enter a voice ID.')
                    refresh = gr.Button('Refresh voices')
                    voice_status = gr.Textbox(label='Voice service', interactive=False)
                    speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label='Speaking speed')
                    preview = gr.Button('Preview selected voice')
                    voice_sample = gr.Audio(label='Selected voice preview', type='filepath', interactive=False)
                    with gr.Accordion('Clone a voice with 60db', open=False):
                        clone_name = gr.Textbox(label='Voice name')
                        clone_language = gr.Textbox(label='Source language code', placeholder='en')
                        clone_samples = gr.File(
                            label='Original recording of the speaker',
                            file_types=['.wav', '.mp3', '.flac'],
                        )
                        gr.Markdown('Upload a clear recording of one speaker. A short sample can be submitted on its own.')
                        clone = gr.Button('Create voice clone')
                        clone_status = gr.Textbox(label='Clone status and reusable voice ID', interactive=False)
                with gr.Group(visible=False) as audio_group:
                    audio = gr.Audio(label='Speech audio', sources=['upload', 'microphone'], type='filepath')
                video = gr.Video(label='Reference video', sources=['upload'])
                cleanup = gr.Checkbox(value=True, label='Clean steady noise from uploaded speech with SoX', info='Uses a detected quiet section. Skipped for API speech or when no suitable noise reference exists.')
                restore_background = gr.Checkbox(value=True, label='Restore original background music / noise')
                background_gain = gr.Slider(0, 1, value=1, step=0.05, label='Background volume', info='1 keeps the separated background at its original level. Original vocals are excluded; some separation leakage can remain.')
                duration = gr.Slider(0, 120, value=0, step=1, label='Render length limit (seconds)', info='0 processes the full video in transcription mode, or full speech in other modes. A limit also shortens the transcription input.')
                steps = gr.Slider(2, 50, value=20, step=1, label='Inference steps', info='20 is a useful starting point. More steps take longer.')
                frames = gr.Dropdown([4, 8, 16], value=16, label='Frames per batch', info='16 passed our local test. Try 4 or 8 if GPU memory runs out.')
                run = gr.Button('Generate lip-synced video', variant='primary')
                with gr.Accordion('Resume standard render', open=False):
                    standard_job = gr.Textbox(label='Standard render job ID')
                    standard_resume = gr.Button('Resume standard render')
                with gr.Accordion('Long-video dubbing', open=False):
                    gr.Markdown('Uses the reference video, target language, and render settings above. Processes sections of about five minutes with 10–25 second render clips.')
                    long_job = gr.Textbox(label='Long-video job ID', info='Leave blank for a new job. Keep the generated ID to resume.')
                    long_single = gr.Checkbox(label='Only one speaker in the entire video', value=False)
                    long_voice = gr.Textbox(label='Existing 60db synthesis voice ID (optional, single speaker only)')
                    long_duration = gr.Number(value=0, minimum=0, label='Long-video test limit in seconds',
                                              info='Use 0 for the full video.')
                    long_map = gr.File(label='Reviewed speaker map', file_types=['.json'])
                    long_button = gr.Button('Start or resume long-video job')
                    long_review = gr.File(label='Speaker review bundle', interactive=False)
                    long_status = gr.Textbox(label='Long-video progress', interactive=False, lines=8)
            with gr.Column():
                transcript_output = gr.Textbox(label='Original and translated transcript', lines=10, interactive=False)
                speech_output = gr.Audio(label='Speech used for this video', type='filepath', interactive=False)
                output = gr.Video(label='Generated video')
                status = gr.Textbox(label='Render status', lines=12, interactive=False)
        source.change(lambda selected: (gr.update(visible=selected != 'Upload audio'),
                                        gr.update(visible=selected == 'Upload audio'),
                                        gr.update(visible=selected == 'API text to speech'),
                                        gr.update(visible=selected == 'Transcribe reference video'),
                                        gr.update(visible=selected == 'Transcribe reference video')),
                      inputs=[source], outputs=[text_group, audio_group, text, target_language, reference_voice_group], queue=False)
        refresh.click(refresh_voices, inputs=[voice], outputs=[voice, voice_status], concurrency_id='pipeline', api_name='refresh_voices')
        demo.load(refresh_voices, inputs=[voice], outputs=[voice, voice_status], concurrency_id='pipeline', api_name=False)
        preview.click(preview_voice, inputs=[voice, text, speed], outputs=[voice_sample], concurrency_id='pipeline', api_name='preview_voice')
        clone.click(clone_voice, inputs=[clone_name, clone_samples, clone_language],
                    outputs=[voice, clone_status], concurrency_id='pipeline', api_name='clone_voice')
        voice.change(lambda: None, outputs=[voice_sample], queue=False)
        run.click(generate_with_speech, inputs=[source, text, voice, video, audio, duration, steps, frames,
                                               restore_background, background_gain, speed, cleanup, target_language,
                                               use_reference_voice, source_language, multiple_speakers],
                  outputs=[output, speech_output, status, transcript_output], concurrency_id='pipeline', api_name='generate')
        standard_resume.click(resume_standard_render, inputs=[standard_job], outputs=[output, status],
                              concurrency_id='pipeline', api_name='resume_standard_render')
        long_button.click(
            run_long_video, inputs=[video, long_job, long_map, long_single, long_voice, source_language,
                                    target_language, speed, long_duration, steps, frames, restore_background, background_gain],
            outputs=[long_job, long_review, output, long_status], concurrency_id='pipeline', api_name='long_video',
            show_progress='minimal',
        )
    return demo


if __name__ == '__main__':
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name='127.0.0.1', server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7861')), inbrowser=False,
        allowed_paths=[str(ROOT / 'trial_inputs'), str(ROOT / 'local_jobs')],
    )
