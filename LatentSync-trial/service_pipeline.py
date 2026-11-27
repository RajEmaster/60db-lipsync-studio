"""Service-layer API over the LatentSync dubbing pipeline.

Two operations power the public service:

  analyze(media)                 -> prepare a *selection* ONCE (separate speech,
                                    transcribe, diarize) and return a manifest:
                                    a handle + per-section speaker turns + language.
  synthesize(selection, section) -> render ONE selected section in a target
                                    language and return the clip path. The caller
                                    streams that file and deletes it.

Design goals (why this module is thin and defensive):

  * OUTPUTS ARE NEVER RETAINED. Each synthesize renders to a delivery file; the
    section's render artifacts are scrubbed immediately afterwards. Nothing
    rendered is kept by the pipeline.
  * The only thing cached between calls is the prepared INPUT for a selection
    (clean-speech stem, turns, clones), under a work dir governed by a TTL, so a
    user can test many target languages on one selection without re-separating.
  * It composes the existing building blocks (start_job, prepare_sections,
    speaker_map, section_voice_resolver, render_section). No pipeline logic is
    duplicated here — bug fixes in the pipeline flow straight through.
  * It is defensive: inputs are validated, failures raise typed ServiceError
    subclasses the API layer maps to HTTP codes, and a failed analyze leaves no
    partial selection behind.

This module is import-safe and has no global side effects; it never binds a port
or starts a worker. The worker loop and the Rust API live under ../service/.
"""
from contextlib import contextmanager
from pathlib import Path
import shutil
import time
import uuid

from chunk_video import FPS, RATE
from job_logging import log_event
from locking_compat import try_lock, unlock
from long_video import (
    long_render_settings, prepare_sections, read_json, render_section,
    save_json, section_voice_resolver, speaker_map, start_job,
)
from tts_api import AUTO_LANGUAGE
from translation import ORIGINAL_LANGUAGE

# The render step addresses each section relative to the pipeline dir (it runs
# render_chunks with cwd here and passes --job section.relative_to(ROOT)), so every
# selection work dir MUST live under this root.
PIPELINE_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- errors
class ServiceError(Exception):
    """Base class; the API layer maps subclasses to HTTP status codes."""
    http_status = 500


class InvalidInput(ServiceError):
    http_status = 400


class SelectionNotFound(ServiceError):
    http_status = 404

    def __init__(self, selection_id):
        super().__init__(f'Unknown or expired selection: {selection_id}')


class SelectionBusy(ServiceError):
    http_status = 409


class UpstreamError(ServiceError):
    """A dependency (60db) failed transiently or returned an error."""
    http_status = 502


class RenderError(ServiceError):
    http_status = 500


# --------------------------------------------------------------------------- settings
# Service defaults mirror the vetted UI defaults. Callers override a subset per call.
_DEFAULTS = {
    'steps': 20, 'frames': 8, 'guidance_scale': 1.5, 'crop_smoothing': 0.4,
    'seed': 1247, 'speed': 1.0, 'clean_speech': True, 'restore_background': True,
    'background_gain': 1.0, 'face_restore': False, 'restore_strength': 0.5,
    'engine_version': '1.5', 'offload': 'auto', 'max_gpus': 0, 'cpu_threads': 4,
    'refine_margin': 0.30, 'section_seconds': 300, 'boundary_search_seconds': 12,
    'vad_pause_seconds': 0.3, 'clip_min_seconds': 10, 'clip_target_seconds': 17,
    'clip_max_seconds': 25, 'overlap_seconds': 0.4,
}
_ALLOWED_FRAMES = (4, 8, 16)
_ENGINE_VERSIONS = ('1.5', '1.6')
_OFFLOAD_MODES = ('auto', 'resident', 'swap')

DEFAULT_TTL_SECONDS = 1800   # prepared selections expire after 30 min of no use


def _validated_settings(overrides):
    """Merge caller overrides over the defaults, rejecting unknown/invalid keys."""
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise InvalidInput('settings must be an object.')
    unknown = set(overrides) - set(_DEFAULTS)
    if unknown:
        raise InvalidInput(f'Unknown settings: {sorted(unknown)}.')
    settings = {**_DEFAULTS, **overrides}
    try:
        if int(settings['frames']) not in _ALLOWED_FRAMES:
            raise InvalidInput(f'frames must be one of {_ALLOWED_FRAMES}.')
        if not 2 <= int(settings['steps']) <= 50:
            raise InvalidInput('steps must be between 2 and 50.')
        if not 1.0 <= float(settings['guidance_scale']) <= 2.5:
            raise InvalidInput('guidance_scale must be between 1.0 and 2.5.')
        if not 0.2 <= float(settings['crop_smoothing']) <= 0.6:
            raise InvalidInput('crop_smoothing must be between 0.2 and 0.6.')
        if not 0.5 <= float(settings['speed']) <= 2.0:
            raise InvalidInput('speed must be between 0.5 and 2.0.')
        if not 0.0 <= float(settings['restore_strength']) <= 1.0:
            raise InvalidInput('restore_strength must be between 0.0 and 1.0.')
        if not 0.0 <= float(settings['background_gain']) <= 1.0:
            raise InvalidInput('background_gain must be between 0.0 and 1.0.')
        if int(settings['seed']) < 0:
            raise InvalidInput('seed must be >= 0.')
    except (TypeError, ValueError) as exc:
        raise InvalidInput(f'Invalid setting value: {exc}') from None
    if settings['engine_version'] not in _ENGINE_VERSIONS:
        raise InvalidInput(f'engine_version must be one of {_ENGINE_VERSIONS}.')
    if settings['offload'] not in _OFFLOAD_MODES:
        raise InvalidInput(f'offload must be one of {_OFFLOAD_MODES}.')
    # The render/section numeric ranges are validated authoritatively by the pipeline.
    long_render_settings(settings)
    return settings


def _build_options(source_language, target_language, settings, speaker_voices):
    """Full options dict in the exact shape the pipeline's start_job expects."""
    return {
        'single_speaker': False, 'voice_id': '',
        'source_language': (source_language or AUTO_LANGUAGE),
        'target_language': target_language,
        'speed': float(settings['speed']), 'duration': 0.0,
        'steps': int(settings['steps']), 'batch_frames': int(settings['frames']),
        'restore_background': bool(settings['restore_background']),
        'background_gain': float(settings['background_gain']),
        'clean_speech': bool(settings['clean_speech']),
        'engine_version': settings['engine_version'],
        'offload': settings['offload'], 'engine_mode': 'resident',
        'max_gpus': int(settings['max_gpus']),
        'guidance_scale': float(settings['guidance_scale']),
        'crop_smoothing': float(settings['crop_smoothing']),
        'seed': int(settings['seed']),
        'face_restore': bool(settings['face_restore']),
        'restore_strength': float(settings['restore_strength']),
        'refine_margin': float(settings['refine_margin']),
        'speaker_voices': speaker_voices or {},
        'section_seconds': settings['section_seconds'],
        'boundary_search_seconds': settings['boundary_search_seconds'],
        'vad_pause_seconds': settings['vad_pause_seconds'],
        'clip_min_seconds': settings['clip_min_seconds'],
        'clip_target_seconds': settings['clip_target_seconds'],
        'clip_max_seconds': settings['clip_max_seconds'],
        'overlap_seconds': settings['overlap_seconds'],
        'cpu_threads': int(settings['cpu_threads']),
    }


# --------------------------------------------------------------------------- helpers
def _resolve_work_root(work_root):
    resolved = Path(work_root).resolve()
    if not resolved.is_relative_to(PIPELINE_ROOT):
        raise InvalidInput(
            f'work_root must live under the pipeline dir ({PIPELINE_ROOT}); the render '
            f'step addresses sections relative to it. Got: {resolved}')
    return resolved


def _selection_dir(work_root, selection_id):
    if not selection_id or len(selection_id) != 32 or any(
            c not in '0123456789abcdef' for c in selection_id):
        raise InvalidInput('selection_id must be a 32-char hex handle.')
    return _resolve_work_root(work_root) / selection_id


@contextmanager
def _selection_lock(workdir):
    """Serialize synthesize() per selection so concurrent language tests on the
    same selection never clobber each other's render state."""
    with (workdir / 'synth.lock').open('a+b') as handle:
        if not handle.tell():
            handle.write(b'0')
            handle.flush()
        if not try_lock(handle):
            raise SelectionBusy('This selection is already rendering; retry shortly.')
        try:
            yield
        finally:
            unlock(handle)


def _translate_error(exc):
    """Map an arbitrary pipeline exception to a typed ServiceError."""
    if isinstance(exc, ServiceError):
        return exc
    message = str(exc)
    if 'HTTP 5' in message or 'timed out' in message or '502' in message:
        return UpstreamError(f'60db upstream error: {message}')
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return InvalidInput(message)
    return RenderError(message)


def _build_manifest(selection_id, config, sections, mapping):
    out_sections = []
    cursor = 0   # running frame offset -> global timeline
    for section, _reference, frames, turns in sections:
        start_s = cursor / FPS
        cursor += frames
        speakers, out_turns = [], []
        for first, last, label, text in turns:
            identity = mapping.get(f'{section.name}/{label}', label)
            if identity not in speakers:
                speakers.append(identity)
            out_turns.append({
                'start': round(start_s + first / RATE, 3),
                'end': round(start_s + last / RATE, 3),
                'speaker': identity, 'text': text,
            })
        out_sections.append({
            'id': int(section.name.rsplit('_', 1)[-1]),
            'start': round(start_s, 3), 'end': round((cursor) / FPS, 3),
            'frames': frames, 'speakers': speakers, 'turns': out_turns,
        })
    return {
        'selection_id': selection_id,
        'source_language': config.get('source_language'),
        'fps': FPS,
        'max_section_seconds': config.get('section_seconds'),
        'sections': out_sections,
    }


# Everything a synthesize re-render must discard so a new language/settings render
# starts fresh. The prepared INPUT (reference.mp4, clean vocals, stems, turns, clones)
# is deliberately NOT in this list, so language tests stay cheap.
_RENDER_ARTIFACTS = (
    'output.mp4', 'output.mp4.json', 'final_soundtrack.wav', 'dub_ready.json',
    'chunk_plan.json', 'vad.json', 'vad_input.wav', 'silent-section.mp4', 'silence.wav',
)


def _reset_render_state(section):
    """Scrub language/settings-specific render + dub artifacts; keep prepared input."""
    for name in _RENDER_ARTIFACTS:
        (section / name).unlink(missing_ok=True)
    speech = section / 'speech'
    if speech.is_dir():
        (speech / 'replacement_speech.wav').unlink(missing_ok=True)
        (speech / 'transcript.json').unlink(missing_ok=True)
        (speech / 'transcript.txt').unlink(missing_ok=True)
        for clip in speech.glob('segment_*.wav'):
            clip.unlink(missing_ok=True)
    for pattern in ('renders', 'chunks', 'chunks_*'):
        for path in section.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


def _find_section(sections, section_id):
    for entry in sections:
        if int(entry[0].name.rsplit('_', 1)[-1]) == int(section_id):
            return entry
    return None


# --------------------------------------------------------------------------- API
def analyze(media_path, work_root, *, source_language=None, settings=None, selection_id=None):
    """Prepare a selection: separate speech, transcribe and diarize. Returns the
    manifest (handle + per-section speaker turns). Leaves no partial state on failure."""
    media_path = Path(media_path)
    if not media_path.is_file():
        raise InvalidInput(f'Media file not found: {media_path}')
    settings = _validated_settings(settings)
    selection_id = selection_id or uuid.uuid4().hex
    workdir = _selection_dir(work_root, selection_id)
    if workdir.exists():
        raise InvalidInput(f'Selection {selection_id} already exists.')
    workdir.mkdir(parents=True)
    try:
        options = _build_options(source_language, ORIGINAL_LANGUAGE, settings, {})
        start_job(workdir, str(media_path), options)
        config = read_json(workdir / 'job.json')
        config.update(long_render_settings(config))
        sections, observations = prepare_sections(workdir, config)
        mapping = speaker_map(workdir, config, observations)
        manifest = _build_manifest(selection_id, config, sections, mapping)
        manifest['created_at'] = time.time()
        save_json(workdir / 'manifest.json', manifest)
        log_event(f'Selection {selection_id} analyzed: {len(manifest["sections"])} section(s).')
        return manifest
    except BaseException as exc:   # noqa: BLE001 - clean up ANY failure, then re-raise typed
        shutil.rmtree(workdir, ignore_errors=True)
        raise _translate_error(exc) if isinstance(exc, Exception) else exc


def synthesize(selection_id, section_id, target_language, work_root,
               *, speaker_voices=None, settings=None, delivery_dir=None):
    """Render ONE section of a prepared selection in `target_language`. Returns the
    delivery video path. The caller must stream the file and then delete it; the
    section's render artifacts are scrubbed here so nothing is retained."""
    workdir = _selection_dir(work_root, selection_id)
    if not (workdir / 'manifest.json').is_file():
        raise SelectionNotFound(selection_id)
    if not target_language or not str(target_language).strip():
        raise InvalidInput('target_language is required.')
    settings = _validated_settings(settings)
    with _selection_lock(workdir):
        try:
            config = read_json(workdir / 'job.json')
            config.update(long_render_settings(config))
            config['target_language'] = str(target_language).strip()
            config['speaker_voices'] = speaker_voices or {}
            # per-call setting overrides (never touch config['frames'] = video length)
            config.update(
                steps=int(settings['steps']), batch_frames=int(settings['frames']),
                guidance_scale=float(settings['guidance_scale']),
                crop_smoothing=float(settings['crop_smoothing']),
                seed=int(settings['seed']), speed=float(settings['speed']),
                restore_background=bool(settings['restore_background']),
                background_gain=float(settings['background_gain']),
                face_restore=bool(settings['face_restore']),
                restore_strength=float(settings['restore_strength']),
                offload=settings['offload'], max_gpus=int(settings['max_gpus']),
                engine_version=settings['engine_version'], cpu_threads=int(settings['cpu_threads']),
            )
            sections, observations = prepare_sections(workdir, config)   # cached -> cheap
            mapping = speaker_map(workdir, config, observations)
            resolve = section_voice_resolver(workdir, config, mapping, observations, sections)
            target = _find_section(sections, section_id)
            if target is None:
                raise InvalidInput(f'section {section_id} is not in selection {selection_id}.')
            section, reference, frames, turns = target
            _reset_render_state(section)   # force a fresh render for THIS language/settings
            log_event(f'Synthesizing selection {selection_id} section {section_id} -> '
                      f'{config["target_language"]}.')
            output, _soundtrack = render_section(
                section, reference, frames, turns, config, mapping,
                lambda identity, section=section: resolve(section, identity))
            delivery_root = Path(delivery_dir) if delivery_dir else (workdir / 'deliveries')
            delivery_root.mkdir(parents=True, exist_ok=True)
            delivered = delivery_root / (
                f'{selection_id}_s{int(section_id)}_{config["target_language"]}_'
                f'{uuid.uuid4().hex[:8]}.mp4')
            shutil.move(str(output), str(delivered))
            _reset_render_state(section)   # retain nothing rendered in the selection
            return {
                'selection_id': selection_id, 'section_id': int(section_id),
                'target_language': config['target_language'],
                'video_path': str(delivered), 'frames': frames,
            }
        except BaseException as exc:   # noqa: BLE001
            raise _translate_error(exc) if isinstance(exc, Exception) else exc


def release(selection_id, work_root):
    """Drop a prepared selection and everything cached for it."""
    workdir = _selection_dir(work_root, selection_id)
    existed = workdir.exists()
    shutil.rmtree(workdir, ignore_errors=True)
    return existed


def purge_expired(work_root, ttl_seconds=DEFAULT_TTL_SECONDS):
    """Remove selections whose manifest is older than the TTL. Returns ids purged."""
    root = Path(work_root)
    if not root.is_dir():
        return []
    cutoff = time.time() - ttl_seconds
    purged = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        manifest = child / 'manifest.json'
        try:
            created = read_json(manifest).get('created_at', 0) if manifest.is_file() else 0
        except Exception:
            created = 0
        if created < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            purged.append(child.name)
    return purged
