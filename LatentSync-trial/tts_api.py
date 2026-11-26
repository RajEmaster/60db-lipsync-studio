"""60db Python SDK speech synthesis; credentials stay in server-side environment variables."""
import base64
import io
import os
from pathlib import Path
import time
import uuid
import wave

from contextlib import contextmanager

import requests
import soundfile as sf
from sixtydb import SixtyDBClient
from dotenv import load_dotenv
from voice_slots import recycle_voice_slots
from worker_lifecycle import exclusive_job_resource

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
DEFAULT_VOICE = '130b0b2d-feb7-4ae0-985c-5c401cdf1115'
AUTO_LANGUAGE = 'auto'


class CompatibleSixtyDBClient(SixtyDBClient):
    """Accept the current audio response while retaining the official SDK interface."""

    RETRY_STATUS = {500, 502, 503, 504}

    def _request(self, method, endpoint, **kwargs):
        kwargs.setdefault('timeout', (15, 300))
        files = kwargs.get('files')
        attempts = 4
        for attempt in range(1, attempts + 1):
            # Rewind any upload streams so a retry resends the full file.
            if files:
                for value in files.values():
                    stream = value[1] if isinstance(value, (tuple, list)) else value
                    if hasattr(stream, 'seek'):
                        try:
                            stream.seek(0)
                        except (OSError, ValueError):
                            pass
            try:
                response = super()._request(method, endpoint, **kwargs)
                break
            except requests.HTTPError as exc:
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                if status in self.RETRY_STATUS and attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise
            except (requests.ConnectionError, requests.Timeout):
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise
        if endpoint == '/tts-synthesize' and isinstance(response, dict):
            if response.get('success') and response.get('audio_base64'):
                raw = base64.b64decode(response['audio_base64'], validate=True)
                # Some voices return headerless LINEAR16 despite requesting WAV.
                if not raw.startswith(b'RIFF') and response.get('encoding') == 'LINEAR16':
                    if len(raw) % 2:
                        raise ValueError('Invalid LINEAR16 audio length.')
                    output = io.BytesIO()
                    with wave.open(output, 'wb') as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(int(response['sample_rate']))
                        wav.writeframes(raw)
                    return output.getvalue()
                return raw
        return response

    def speech_to_text(self, audio_file, language=None, diarize=False):
        # SDK 2.2.0 uses the old 'audio' field and bypasses its session/timeout.
        # The current service expects 'file' and a multipart Content-Type.
        language = (language or '').strip().lower()
        data = {'language': language} if language and language != AUTO_LANGUAGE else {}
        if diarize:
            data['diarize'] = 'true'
        return self._request('POST', '/stt', files={
            'file': (Path(audio_file.name).name, audio_file, 'audio/wav'),
        }, data=data, headers={'Content-Type': None})

    def create_voice(self, name, audio_file, **params):
        try:
            return self._request(
                'POST', '/voices', files={'audio_file': audio_file},
                data={'name': name, **params}, headers={'Content-Type': None},
            )
        except requests.HTTPError as exc:
            if exc.response.status_code != 400 or is_voice_limit(exc):
                raise
            try:
                payload = exc.response.json()
            except ValueError:
                payload = {}
            detail = next((payload[key] for key in ('message', 'error', 'detail')
                           if isinstance(payload, dict) and isinstance(payload.get(key), str)), None)
            details = payload.get('details') if isinstance(payload, dict) else None
            if isinstance(details, list):
                detail = '; '.join(item for item in details if isinstance(item, str)) or detail
            raise RuntimeError(f'60db rejected the voice sample: {detail[:300]}' if detail
                               else '60db rejected the voice sample (HTTP 400).') from None


@contextmanager
def sdk_client():
    key = os.environ.get('SIXTYDB_API_KEY', '').strip()
    if not key:
        raise ValueError('Set SIXTYDB_API_KEY in .env and restart the server.')
    client = CompatibleSixtyDBClient(
        api_key=key,
        base_url=os.environ.get('SIXTYDB_SDK_BASE_URL', 'https://api.60db.ai').rstrip('/'),
    )
    client.session.trust_env = False
    try:
        yield client
    except requests.Timeout:
        raise RuntimeError('60db SDK request timed out. Please retry.') from None
    except requests.HTTPError as exc:
        if is_voice_limit(exc):
            raise RuntimeError('The 60db workspace has reached its custom voice limit.') from None
        raise RuntimeError(f'60db SDK returned HTTP {exc.response.status_code}. Check your key, voice ID, and account limits.') from None
    except Exception as exc:
        message = str(exc).replace(key, '[redacted]')
        raise RuntimeError(f'60db SDK error: {message}') from None
    finally:
        client.session.close()


def is_voice_limit(error):
    if error.response.status_code not in (400, 403, 409, 429):
        return False
    try:
        body = error.response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get('code') == 'VOICE_LIMIT_REACHED'


def create_cloned_voice(name, sample_path, language):
    name, language = name.strip(), language.strip()
    if not name or not language:
        raise ValueError('Enter a voice name and the source language code.')
    if not sample_path:
        raise ValueError('Provide a recording of the speaker.')
    path = Path(sample_path)
    content_types = {'.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.flac': 'audio/flac'}
    if not path.is_file() or path.suffix.lower() not in content_types:
        raise ValueError(f'Provide an existing WAV, MP3, or FLAC file: {path.name}')

    lock = exclusive_job_resource('voice_slots.lock', 'Waiting for another clone request...')
    with lock, path.open('rb') as audio, sdk_client() as client:
        workspace = os.environ.get('SIXTYDB_WORKSPACE_ID', '').strip()
        if not workspace:
            workspaces = client.get_workspaces()['data']
            if len(workspaces) != 1:
                raise ValueError('Set SIXTYDB_WORKSPACE_ID to the numeric workspace ID where the clone should be saved.')
            workspace = workspaces[0]['id']
        upload = (path.name, audio, content_types[path.suffix.lower()])
        metadata = {'language': language, 'gender': 'neutral', 'workspace_id': workspace}
        try:
            result = client.create_voice(name, upload, **metadata)
        except requests.HTTPError as exc:
            if not is_voice_limit(exc):
                raise
            if not recycle_voice_slots(client, workspace, limit=1):
                raise RuntimeError('The voice limit is full and no inactive Lipsyncstudio clone can be recycled.') from None
            audio.seek(0)
            result = client.create_voice(name, upload, **metadata)
    if isinstance(result, dict):
        result = result.get('data', result)
    if not isinstance(result, dict) or not (result.get('id') or result.get('voice_id')):
        raise RuntimeError('60db returned no voice ID. Check the workspace voice library before submitting again.')
    return result


def list_voices():
    with sdk_client() as client:
        responses = [client.get_voices()]
        try:
            responses.append(client.get_my_voices())
        except requests.HTTPError as exc:
            if exc.response.status_code != 403:
                raise
    try:
        voices = []
        for data in responses:
            for _ in range(3):
                if isinstance(data, list):
                    break
                if not isinstance(data, dict):
                    raise ValueError()
                data = next((data[k] for k in ('voices', 'data', 'built_in_voices', 'items', 'results') if k in data), None)
            if not isinstance(data, list):
                raise ValueError()
            voices.extend(data)
        choices, seen = [], set()
        for item in voices:
            if not isinstance(item, dict):
                continue
            identity = item.get('voice_provider_id') or item.get('voice_id') or item.get('id') or item.get('voiceId')
            if not identity or str(identity) in seen:
                continue
            identity = str(identity)
            name = item.get('name') or item.get('display_name') or item.get('voice_name') or identity
            choices.append((f'{name} ({identity})', identity))
            seen.add(identity)
        if voices and not choices:
            raise ValueError()
        return choices
    except (ValueError, TypeError):
        raise RuntimeError('Voice discovery returned an unsupported response format.') from None


def synthesize(text, voice=DEFAULT_VOICE, speed=0.7):
    text = (text or '').strip()
    voice = (voice or '').strip()
    if not text or not voice:
        raise ValueError('Enter speech text and a voice ID.')
    if not 0.5 <= float(speed) <= 2:
        raise ValueError('Speaking speed must be between 0.5 and 2.')
    with sdk_client() as client:
        try:
            raw = client.text_to_speech(text=text, voice_id=voice, speed=float(speed), output_format='wav')
        except requests.HTTPError as exc:
            if not 500 <= exc.response.status_code <= 599:
                raise
            raise RuntimeError(f'60db TTS returned HTTP {exc.response.status_code} while synthesizing '
                               f'{len(text)} characters. Resume the job to retry synthesis.') from None
    try:
        if not isinstance(raw, bytes) or not raw:
            raise ValueError()
        samples, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
        if len(samples) == 0:
            raise ValueError()
    except (ValueError, TypeError, sf.LibsndfileError):
        raise RuntimeError('60db SDK returned invalid audio.') from None
    folder = ROOT / 'local_jobs' / 'tts'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f'{uuid.uuid4().hex}.wav'
    sf.write(output, samples, rate, subtype='PCM_16')
    return str(output)


TRANSIENT_HTTP = (500, 502, 503, 504)


def _speech_to_text(audio_path, **kwargs):
    """Call 60db STT, retrying transient server errors (502/503/504/timeouts).

    Per-turn re-transcription makes many small STT calls, so one transient blip
    must not abort a whole job. Non-transient errors (bad key, limits) raise at once.
    """
    for attempt in range(4):
        try:
            with sdk_client() as client, Path(audio_path).open('rb') as audio:
                return client.speech_to_text(audio, **kwargs)
        except RuntimeError as exc:
            message = str(exc)
            transient = 'timed out' in message or any(f'HTTP {code}' in message for code in TRANSIENT_HTTP)
            if not transient or attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))


def transcribe(audio_path, language=None):
    result = _speech_to_text(audio_path, language=language)
    for _ in range(3):
        if isinstance(result, str):
            return result.strip()
        if not isinstance(result, dict):
            break
        if result.get('success') is False:
            raise RuntimeError('60db could not transcribe this audio.')
        if 'no_speech_detected' in (result.get('warning_codes') or []):
            return ''
        result = next((result[k] for k in ('text', 'transcript', 'transcription', 'data') if k in result), None)
    if isinstance(result, str):
        return result.strip()
    raise RuntimeError('60db returned an unsupported transcription response.')


def transcribe_speakers(audio_path, language=None):
    """Return 60db's full diarized STT result (both `segments` and `speakers`).

    The response carries a coarse `segments` transcription AND a separate
    `speakers` array of per-speaker time ranges — the actual diarization. On
    backends without word alignment the segment text is NOT split per speaker
    (60db warns with `speaker_alignment_unavailable`), so callers must prefer the
    `speakers` array and recover each turn's text by re-transcribing its slice.
    """
    result = _speech_to_text(audio_path, language=language, diarize=True)
    if isinstance(result, dict):
        result = result.get('data', result)
    if not isinstance(result, dict) or not (
        isinstance(result.get('speakers'), list) or isinstance(result.get('segments'), list)
    ):
        raise RuntimeError('60db returned no speaker segments.')
    return result


def detect_spoken_language(audio_path):
    with sdk_client() as client, Path(audio_path).open('rb') as audio:
        result = client.speech_to_text(audio)
    if isinstance(result, dict):
        result = result.get('data', result)
    language = result.get('language') if isinstance(result, dict) else None
    if isinstance(language, str) and len(language) in (2, 3) and language.isalpha() and language not in ('und', 'mul'):
        return language.lower()
    raise ValueError('60db could not detect the clone sample language. Choose the spoken language from the dropdown.')
